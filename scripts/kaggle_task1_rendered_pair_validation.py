"""Fit bounded Poisson corrections with actual public-render candidate preferences."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from glyph_atlas_metric_ranker import _deduplicate
from glyph_contour_retrieval import contour_retrieval_scores
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_gallery_validation import image_descriptors, standardized
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_v8_metric_validation import ATLAS_WEIGHT, CONTOUR_WEIGHT, SESSIONS, fit_v8, rendered_bank, summarize


PAIRS = 6
MARGIN_OCF = 5.0
MIN_POSITIVE_OCF = 55.0
RATE_LIMIT = 0.18
HEAD_SEED = 211


@dataclass(frozen=True)
class CalibrationPairs:
    features: torch.Tensor
    rates: torch.Tensor
    mean_rates: torch.Tensor
    observed: torch.Tensor
    valid: torch.Tensor


class RenderCorrection(nn.Module):
    def __init__(self, descriptor_size, neural_size):
        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(descriptor_size, 24), nn.SiLU(), nn.Linear(24, neural_size))
        nn.init.zeros_(self.mapping[-1].weight)
        nn.init.zeros_(self.mapping[-1].bias)

    def forward(self, descriptor):
        return RATE_LIMIT * self.mapping(descriptor).tanh()


@torch.no_grad()
def descriptor_basis(descriptors):
    torch.manual_seed(HEAD_SEED)
    center = descriptors.mean(dim=0)
    _, _, basis = torch.pca_lowrank(descriptors - center, q=64, center=False, niter=3)
    coordinate_scale = ((descriptors - center) @ basis).std(dim=0, unbiased=False).clamp_min(0.02)
    return center, basis, coordinate_scale


def coordinates(descriptors, projection):
    center, basis, scale = projection
    return ((descriptors - center) @ basis / scale).clamp(-5, 5)


@torch.no_grad()
def backbone_views(model, query, session):
    normalized, _ = standardize(feature_view(query, "evoked-total"), model.normalization[session])
    direct = model.decoder.predict_variants(session, normalized.mean(dim=1))
    contours = 0.75 * direct["refined"].float() + 0.25 * direct["refined_gray"].float()
    shape = standardized(image_descriptors(contours) @ model.atlas.contours.T)
    ridge = standardized(contour_retrieval_scores(model.contour, query, session)["contour_ridge"])
    return shape, ridge


def open_score(model, rates, query, shape, ridge):
    rate = standardized(count_template_logits(query, rates, "rate"))
    timed = standardized(count_template_logits(query, rates, "timed"))
    neural = standardized(0.5 * (rate + timed))
    v4 = standardized(0.5 * (shape + neural))
    return model.prior.variants(ATLAS_WEIGHT * v4 + CONTOUR_WEIGHT * ridge)["usage_unseen"]


@torch.no_grad()
def full_rates(model):
    result = {}
    for session in SESSIONS:
        with torch.autocast(device_type="cuda"):
            result[session] = model.encoder.expected_counts(model.atlas.images, session, batch_size=192).float()
    return result


@torch.no_grad()
def calibration_pairs(model, images, queries, rendered, atlas_features, public_rates, scorer, seed):
    visual = image_descriptors((images >= 128).to(torch.uint8) * 255) @ model.atlas.contours.T
    proposals = [visual.topk(12, dim=1).indices]
    for session, query in queries.items():
        shape, ridge = backbone_views(model, query, session)
        proposals.append(open_score(model, public_rates[session], query, shape, ridge).topk(2, dim=1).indices)
    indices, valid = _deduplicate(torch.cat(proposals, dim=1), len(rendered))
    row, col = valid.nonzero(as_tuple=True)
    guess = (rendered[indices[row, col]] >= 128).cpu().numpy()
    target = (images[row] >= 128).cpu().numpy()
    ocs = scorer(guess, target, device=str(images.device), batch_size=12)
    exact = indices.new_full(indices.shape, -torch.inf, dtype=torch.float32)
    exact[row, col] = torch.as_tensor(ocs, device=images.device, dtype=exact.dtype)
    best, winning = exact.max(dim=1, keepdim=True)
    lower = exact.masked_fill(exact > best - MARGIN_OCF, -torch.inf)
    negative_quality, negative = lower.topk(PAIRS, dim=1)
    kept = (best[:, 0] >= MIN_POSITIVE_OCF) & torch.isfinite(negative_quality[:, 0])
    options = indices.gather(1, torch.cat((winning, negative), dim=1))
    balanced = torch.Generator(device=images.device).manual_seed(seed)
    observed = []
    for values in queries.values():
        repeat_order = torch.rand(values.shape[:2], generator=balanced, device=images.device).argsort(dim=1)
        arranged = values.gather(1, repeat_order[:, :, None].expand_as(values))
        observed.append(arranged.unflatten(1, (2, 2)).mean(dim=2))
    pair_data = CalibrationPairs(atlas_features[options], torch.stack([rates[options] for rates in public_rates.values()]),
                                 torch.stack([rates.mean(dim=0) for rates in public_rates.values()]),
                                 torch.stack(observed), torch.isfinite(negative_quality))
    metadata = {"public_calibration_ocf_pairs": len(ocs), "calibration_glyphs_with_pairs": int(kept.sum()),
                "mean_best_quality_kept": float(best[kept].mean()),
                "mean_positive_negative_gap": float((best - negative_quality)[pair_data.valid & kept[:, None]].mean())}
    return pair_data, kept, metadata


def objective_parts(candidates, selection, correction):
    counts = candidates.observed[:, selection].unflatten(-1, (-1, 10))[..., 2:]
    delta = correction(candidates.features[selection])
    expected = 0.98 * candidates.rates[:, selection] * delta.exp()[None] + 0.02 * candidates.mean_rates[:, None, None]
    expected = expected.unflatten(-1, (-1, 10))[..., 2:].clamp_min(1e-6)
    flat_counts = counts.flatten(-2)
    flat_expected = expected.flatten(-2)
    timed = torch.einsum("sbhd,sbkd->sbhk", flat_counts, flat_expected.log()) - flat_expected.sum(-1)[:, :, None]
    pooled = expected.sum(-1)
    totals = torch.einsum("sbhu,sbku->sbhk", counts.sum(-1), pooled.log()) - pooled.sum(-1)[:, :, None]
    view = 0.50 * (timed + totals)
    nll = (expected[:, :, :1] - counts * expected[:, :, :1].log()).mean()
    return view[..., :1] - view[..., 1:], nll, delta


@torch.no_grad()
def calibration_metrics(candidates, selection, correction):
    margins, nll, delta = objective_parts(candidates, selection, correction)
    valid = candidates.valid[selection]
    centered = margins.permute(1, 3, 0, 2)[valid]
    return {"glyphs": len(selection), "pair_margins": centered.shape[0], "positive_count_nll": float(nll),
            "mean_correct_halves": float((centered > 0).float().mean()),
            "agreement_both_halves_correct": float((centered > 0).all(dim=-1).float().mean()),
            "mean_absolute_log_rate_change": float(delta.abs().mean()), "max_log_rate_change": float(delta.abs().max())}


def fit_corrections(candidates, train, check, args):
    device = candidates.features.device
    zero = lambda values: values.new_zeros((*values.shape[:-1], candidates.rates.shape[-1]))
    with torch.no_grad():
        base_margins, _, _ = objective_parts(candidates, train, zero)
        valid = candidates.valid[train]
        scale = base_margins.masked_fill(~valid[None, :, None], 0).abs().mean(dim=(1, 2, 3)).clamp_min(1)
    models = {}
    reports = {"base": calibration_metrics(candidates, check, zero)}
    for objective in ("generative", "pairwise"):
        torch.manual_seed(HEAD_SEED)
        generator = torch.Generator(device=device).manual_seed(HEAD_SEED + 1)
        model = RenderCorrection(candidates.features.shape[-1], candidates.rates.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.04)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0002)
        for _ in range(args.epochs):
            order = train[torch.randperm(len(train), generator=generator, device=device)]
            for sampled in order.split(32):
                optimizer.zero_grad(set_to_none=True)
                margin, nll, delta = objective_parts(candidates, sampled, model)
                if objective == "generative":
                    loss = nll
                else:
                    valid = candidates.valid[sampled]
                    pairwise = F.softplus(-margin / scale[:, None, None, None]).permute(1, 3, 0, 2)[valid]
                    loss = pairwise.mean() + 0.10 * nll
                (loss + 0.05 * delta.square().mean()).backward()
                optimizer.step()
            schedule.step()
        models[objective] = model.eval()
        reports[objective] = calibration_metrics(candidates, check, model)
    return models, reports


def deeper_proposal(base, variant):
    core = base[:, :16]
    neighbors = variant[:, :24]
    novel = (neighbors[:, :, None] != core[:, None]).all(dim=2)
    order = torch.arange(neighbors.shape[1], device=base.device).expand_as(neighbors)
    selected = order.masked_fill(~novel, neighbors.shape[1]).topk(8, largest=False, dim=1).values
    return torch.cat((core, neighbors.gather(1, selected)), dim=1)


@torch.no_grad()
def quality(counts, rates):
    matrix = count_template_logits(counts, rates, "timed")
    correct = matrix.diagonal()[:, None]
    ranks = 1 + (matrix > correct).sum(dim=1)
    observations = counts.mean(dim=1).unflatten(-1, (-1, 10))[..., 2:]
    expected = rates.unflatten(-1, (-1, 10))[..., 2:].clamp_min(1e-6)
    return {"count_nll": float((expected - observations * expected.log()).mean()),
            "top1": float((ranks == 1).float().mean()), "top32": float((ranks <= 32).float().mean())}


@torch.no_grad()
def evaluate(model, queries, public_rates, rendered, images, geometry, corrections, selected, scorer, args,
             *, comparison=("pairwise", "generative")):
    public, clean = geometry
    public_adjustment = {name: torch.cat([head(features) for features in public.split(512)])
                         for name, head in corrections.items()}
    clean_adjustment = {name: head(clean) for name, head in corrections.items()}
    reference = (images[selected] >= 128).cpu().numpy()
    paired, diagnostic, depths, expanded = {}, {}, {}, {}
    exact = PairedOCF(scorer, rendered, reference, images.device, 12)
    for session, held_query in queries.items():
        subset = held_query[selected]
        shape, ridge = backbone_views(model, subset, session)
        rate = public_rates[session]
        variants = {"base": rate, **{name: rate * delta.exp() for name, delta in public_adjustment.items()}}
        options = {name: open_score(model, rates, subset, shape, ridge).topk(32, dim=1).indices
                   for name, rates in variants.items()}
        first = {name: exact.evaluate(rows[:, :1])[:, 0] for name, rows in options.items()}
        for name, values in first.items():
            paired.setdefault(name, []).append(values)
        closed_rates = model.encoder.expected_counts(images, session)
        closed = {"base": quality(held_query, closed_rates), **{
            name: quality(held_query, closed_rates * delta.exp()) for name, delta in clean_adjustment.items()}}
        oracle = {}
        for name, shape_order in options.items():
            budget = shape_order[:, :1].expand(-1, 32).clone()
            budget[:args.oracle_images] = shape_order[:args.oracle_images]
            values = exact.evaluate(budget)[:args.oracle_images]
            oracle[name] = {str(depth): values[:, :depth].max(1) for depth in (16, 32)}
        expansions = {"pure_depth": options["base"][:, :24],
                      **{name: deeper_proposal(options["base"], options[name]) for name in corrections}}
        expansion_score = {}
        for name, rows in expansions.items():
            budget = rows[:, :1].expand_as(rows).clone()
            budget[:args.oracle_images] = rows[:args.oracle_images]
            expansion_score[name] = exact.evaluate(budget)[:args.oracle_images].max(1)
        for name, outcomes in oracle.items():
            for depth, values in outcomes.items():
                depths.setdefault((name, depth), []).append(values)
        for name, values in expansion_score.items():
            expanded.setdefault(name, []).append(values)
        report = {"first_choice": {name: summarize([values], [first["base"]]) for name, values in first.items()},
                  "equal_depth_oracle": {name: {depth: summarize([values], [oracle["base"][depth]])
                                                for depth, values in rows.items()} for name, rows in oracle.items()},
                  "equal_24_candidate_expansion_oracle": {name: summarize([values], [expansion_score["pure_depth"]])
                                                          for name, values in expansion_score.items()},
                  "closed_image_dictionary": closed,
                  "different_first_choice": {name: int((rows[:, 0] != options["base"][:, 0]).sum())
                                             for name, rows in options.items()},
                  "unique_exact_pairs_to_date": len(exact.values)}
        diagnostic[session] = report
        print(json.dumps({"session": session, **report}), flush=True)
    primary, control = comparison
    return {"sessions": diagnostic,
            "paired_open_ocf": {name: summarize(values, paired["base"]) for name, values in paired.items()},
            f"{primary}_vs_{control}": summarize(paired[primary], paired[control]),
            "aggregate_depth_oracle": {name: {depth: summarize(depths[name, depth], depths["base", depth])
                                               for depth in ("16", "32")} for name in paired},
            "aggregate_expansion_oracle": {name: summarize(values, expanded["pure_depth"]) for name, values in expanded.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=5573)
    parser.add_argument("--calibration-glyphs", type=int, default=192)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--oracle-images", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_rendered_pair_validation.json"))
    args = parser.parse_args()
    if not 2 <= args.oracle_images <= args.ocf_images <= 320 or args.calibration_glyphs < 96 or args.epochs <= 0:
        parser.error("Use sufficient calibration glyphs and a valid paired OCF sample.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this image-to-spike comparison in an authorized Kaggle GPU Notebook.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, torch.device("cuda:0"))
    avoid = (961, 2209, 613, 3307, 8723)
    fit, held = outer_split(len(images), args.seed, avoid, images.device)
    generator = torch.Generator(device=images.device).manual_seed(args.seed)
    fit_order = fit[torch.randperm(len(fit), device=images.device, generator=generator)]
    calibration, training = fit_order[:args.calibration_glyphs], fit_order[args.calibration_glyphs:]
    scorer = official_scorer(root)
    groups = grouped_blocks(root, load_catalog(root), "mice1", images.device, len(images))
    model = fit_v8({session: counts[training] for session, counts in groups.items()}, images[training], HEAD_SEED)
    rendered = rendered_bank(model)
    projection = descriptor_basis(model.atlas.contours)
    public = coordinates(model.atlas.contours, projection)
    public_rates = full_rates(model)
    cal = {session: counts[calibration] for session, counts in groups.items()}
    candidates, eligible, coverage = calibration_pairs(model, images[calibration], cal, rendered, public, public_rates, scorer, args.seed)
    cal_positions = torch.arange(len(calibration), device=images.device)
    learning = cal_positions[eligible & (cal_positions < 3 * len(calibration) // 4)]
    comparison = cal_positions[eligible & (cal_positions >= 3 * len(calibration) // 4)]
    if len(learning) < 32 or len(comparison) < 12:
        raise ValueError("Too few fitted-render calibration comparisons met the predeclared OCF criteria.")
    corrections, checks = fit_corrections(candidates, learning, comparison, args)
    print(json.dumps({"calibration_coverage": coverage, "independent_calibration_check": checks}), flush=True)
    selection = torch.randperm(len(held), device=images.device, generator=generator)[:args.ocf_images]
    held_images = images[held]
    clean = coordinates(image_descriptors((held_images >= 128).to(torch.uint8) * 255), projection)
    held_queries = {session: counts[held] for session, counts in groups.items()}
    results = evaluate(model, held_queries, public_rates, rendered, held_images, (public, clean), corrections, selection, scorer, args)
    output = {"mouse": "mice1", "seed": args.seed, "avoided_320_glyph_splits": avoid,
              "train_glyphs": len(training), "calibration_glyphs": len(calibration), "outer_glyphs": len(held),
              "public_atlas_candidates": len(rendered), "ocf_per_session": len(selection), "oracle_per_session": args.oracle_images,
              "log_rate_clip": RATE_LIMIT, "epochs": args.epochs, "pair_margin_ocf": MARGIN_OCF,
              "calibration_coverage": coverage, "independent_calibration_check": checks,
              "protocol": "Frozen v8 ranking policy, paired full open-atlas comparison. Exact reference preference pairs are "
                          "formed only on independent labeled calibration glyphs. Held references are used after public "
                          "proposals, and in a separate explicit closed-gallery neural diagnostic. Actual test data unused.",
              "seconds": time.monotonic() - started, **results}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
