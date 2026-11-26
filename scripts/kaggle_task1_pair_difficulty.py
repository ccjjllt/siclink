"""Compare rendered preferences with matched narrow and wide OCF gaps."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from glyph_atlas_metric_ranker import _deduplicate
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_diagnostics import official_scorer
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_task1_rendered_pair_validation import (
    HEAD_SEED, RATE_LIMIT, RenderCorrection, backbone_views, coordinates, descriptor_basis,
    evaluate, full_rates, open_score,
)
from kaggle_v8_metric_validation import fit_v8, rendered_bank
from kaggle_gallery_validation import image_descriptors


PREVIOUS_OUTER = (961, 2209, 613, 3307, 8723, 5573)
NEAR_GAP = (4.0, 18.0)
FAR_GAP = 25.0
MIN_RENDER_OCF = 24.0
MAX_PAIRS = 6


@dataclass(frozen=True)
class PairBatch:
    features: torch.Tensor
    rates: torch.Tensor
    valid: torch.Tensor
    gap: torch.Tensor


@torch.no_grad()
def construct_pairs(model, images, queries, rendered, public, public_rates, scorer, seed):
    visual = image_descriptors((images >= 128).to(torch.uint8) * 255) @ model.atlas.contours.T
    visual_candidates = visual.topk(20, dim=1).indices
    base_scores = []
    for session, query in queries.items():
        shape, ridge = backbone_views(model, query, session)
        base_scores.append(open_score(model, public_rates[session], query, shape, ridge))
    neural_candidates = [values.topk(6, dim=1).indices for values in base_scores]
    anchors = torch.cat([visual_candidates[:, :4], *neural_candidates], dim=1)
    proposals = torch.cat([visual_candidates, *neural_candidates], dim=1)
    candidates, valid = _deduplicate(proposals, len(rendered))
    row, slot = valid.nonzero(as_tuple=True)
    result = scorer((rendered[candidates[row, slot]] >= 128).cpu().numpy(),
                    (images[row] >= 128).cpu().numpy(), device=str(images.device), batch_size=12)
    exact = images.new_zeros(candidates.shape, dtype=torch.float32)
    exact[row, slot] = torch.as_tensor(result, device=images.device, dtype=exact.dtype)

    edges = torch.triu_indices(candidates.shape[1], candidates.shape[1], offset=1, device=images.device)
    left, right = edges
    left_ocf, right_ocf = exact[:, left], exact[:, right]
    gap = (left_ocf - right_ocf).abs()
    anchored = (candidates[:, :, None] == anchors[:, None, :]).any(dim=-1)
    eligible = valid[:, left] & valid[:, right] & (anchored[:, left] | anchored[:, right])
    eligible &= torch.minimum(left_ocf, right_ocf) >= MIN_RENDER_OCF
    near = eligible & (gap >= NEAR_GAP[0]) & (gap <= NEAR_GAP[1])
    far = eligible & (gap >= FAR_GAP)

    proposed_scores = torch.stack([values.gather(1, candidates) for values in base_scores])
    ambiguity = (proposed_scores[:, :, left] - proposed_scores[:, :, right]).abs().mean(dim=0)
    priority = (left_ocf + right_ocf) / 200 - 0.08 * ambiguity.clamp_max(5)
    budget = torch.minimum(near.sum(dim=1), far.sum(dim=1)).clamp_max(MAX_PAIRS)
    selected = torch.arange(MAX_PAIRS, device=images.device)[None, :] < budget[:, None]
    all_rates = torch.stack(tuple(public_rates.values()))
    bands = {}
    for name, available in (("near", near), ("far", far)):
        picked = priority.masked_fill(~available, -torch.inf).topk(MAX_PAIRS, dim=1).indices
        first = candidates.gather(1, left[picked])
        second = candidates.gather(1, right[picked])
        order = left_ocf.gather(1, picked) >= right_ocf.gather(1, picked)
        indices = torch.stack([torch.where(order, first, second), torch.where(order, second, first)], dim=-1)
        bands[name] = PairBatch(public[indices], all_rates[:, indices], selected, gap.gather(1, picked))

    generator = torch.Generator(device=images.device).manual_seed(seed)
    halves = []
    for counts in queries.values():
        order = torch.rand(counts.shape[:2], generator=generator, device=images.device).argsort(dim=1)
        arranged = counts.gather(1, order[:, :, None].expand_as(counts))
        halves.append(arranged.unflatten(1, (2, 2)).mean(dim=2))
    diagnostics = {"exact_candidate_images": len(result), "calibration_glyphs_with_both_bands": int(selected.any(1).sum()),
                   "matched_pairs_per_band": int(selected.sum()),
                   "pair_mean_exact_ocf_gap": {name: float(values.gap[selected].mean()) for name, values in bands.items()},
                   "fraction_of_glyphs_with_near": float(near.any(1).float().mean()),
                   "fraction_of_glyphs_with_far": float(far.any(1).float().mean())}
    return bands, torch.stack(halves), all_rates.mean(dim=1), diagnostics


def margins(batch, observed, mean_rates, selection, head):
    delta = head(batch.features[selection])
    expected = 0.98 * batch.rates[:, selection] * delta.exp()[None] + 0.02 * mean_rates[:, None, None, None]
    expected = expected.unflatten(-1, (-1, 10))[..., 2:].clamp_min(1e-6)
    count = observed[:, selection].unflatten(-1, (-1, 10))[..., 2:]
    log_expected = expected.log()
    timed = torch.einsum("sghut,sgpkut->sghpk", count, log_expected) - expected.sum(dim=(-1, -2))[:, :, None]
    rates = expected.sum(dim=-1)
    pooled = torch.einsum("sghu,sgpku->sghpk", count.sum(dim=-1), rates.log()) - rates.sum(dim=-1)[:, :, None]
    log_likelihood = 0.5 * (timed + pooled)
    return (log_likelihood[..., 0] - log_likelihood[..., 1]).permute(1, 3, 0, 2), delta


@torch.no_grad()
def pair_metrics(bands, observed, mean_rates, selection, head):
    result = {}
    for name, batch in bands.items():
        values, delta = margins(batch, observed, mean_rates, selection, head)
        values = values[batch.valid[selection]]
        correct = values > 0
        result[name] = {"pairs": len(values), "fraction_half_preferences_correct": float(correct.float().mean()),
                        "fraction_both_half_preferences_correct": float(correct.all(dim=-1).float().mean()),
                        "mean_abs_log_rate_delta": float(delta.abs().mean())}
    return result


def fit_models(bands, observed, mean_rates, train, check, epochs):
    device = observed.device
    n_counts = observed.shape[-1]
    zero = lambda values: values.new_zeros((*values.shape[:-1], n_counts))
    results = {"base": pair_metrics(bands, observed, mean_rates, check, zero)}
    models = {}
    for name, band in (("near", "near"), ("far", "far"), ("shuffled", "near")):
        batch = bands[band]
        with torch.no_grad():
            raw, _ = margins(batch, observed, mean_rates, train, zero)
            scale = raw[batch.valid[train]].abs().mean(dim=(0, 2)).clamp_min(1)
        torch.manual_seed(HEAD_SEED)
        generator = torch.Generator(device=device).manual_seed(HEAD_SEED + 1)
        label_rng = torch.Generator(device=device).manual_seed(HEAD_SEED + 2)
        signs = 2 * torch.randint(0, 2, batch.valid.shape, generator=label_rng, device=device) - 1
        head = RenderCorrection(batch.features.shape[-1], n_counts).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.002, weight_decay=0.04)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0002)
        for _ in range(epochs):
            order = train[torch.randperm(len(train), generator=generator, device=device)]
            for selection in order.split(24):
                optimizer.zero_grad(set_to_none=True)
                value, delta = margins(batch, observed, mean_rates, selection, head)
                if name == "shuffled":
                    value = value * signs[selection, :, None, None]
                available = batch.valid[selection]
                loss = F.softplus(-value[available] / scale[:, None]).mean()
                (loss + 0.05 * delta[available].square().mean()).backward()
                optimizer.step()
            schedule.step()
        models[name] = head.eval()
        results[name] = pair_metrics(bands, observed, mean_rates, check, head)
    return models, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=6379)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--calibration-glyphs", type=int, default=320)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--oracle-images", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_pair_difficulty.json"))
    args = parser.parse_args()
    if not 2 <= args.oracle_images <= args.ocf_images <= 320 or args.epochs < 1:
        parser.error("Use positive epochs and a valid paired OCF sample.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the rendered pair comparison in a Kaggle GPU Notebook.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held = outer_split(len(images), args.seed, PREVIOUS_OUTER, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    order = fit[torch.randperm(len(fit), generator=generator, device=device)]
    calibration, fit = order[:args.calibration_glyphs], order[args.calibration_glyphs:]
    groups = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    model = fit_v8({session: values[fit] for session, values in groups.items()}, images[fit], HEAD_SEED)
    rendered = rendered_bank(model)
    scorer = official_scorer(root)
    projection = descriptor_basis(model.atlas.contours)
    public = coordinates(model.atlas.contours, projection)
    rates = full_rates(model)
    queries = {session: values[calibration] for session, values in groups.items()}
    bands, observed, mean_rates, coverage = construct_pairs(model, images[calibration], queries, rendered, public,
                                                            rates, scorer, args.seed)
    eligible = bands["near"].valid.any(dim=1).nonzero(as_tuple=True)[0]
    boundary = 3 * len(calibration) // 4
    train, check = eligible[eligible < boundary], eligible[eligible >= boundary]
    print(json.dumps({"calibration_coverage": coverage, "train_glyphs": len(train), "check_glyphs": len(check)}),
          flush=True)
    if not len(train) or not len(check):
        raise ValueError("Both calibration partitions need glyphs with same-budget near and far pairs.")
    corrections, checks = fit_models(bands, observed, mean_rates, train, check, args.epochs)
    print(json.dumps({"independent_calibration_check": checks}), flush=True)

    hold_queries = {session: values[held] for session, values in groups.items()}
    selection = torch.randperm(len(held), generator=generator, device=device)[:args.ocf_images]
    clean = coordinates(image_descriptors((images[held] >= 128).to(torch.uint8) * 255), projection)
    results = evaluate(model, hold_queries, rates, rendered, images[held], (public, clean), corrections, selection,
                       scorer, args, comparison=("near", "far"))
    report = {"mouse": "mice1", "seed": args.seed, "previous_outer_seeds": PREVIOUS_OUTER, "fit_glyphs": len(fit),
              "calibration_glyphs": len(calibration), "held_glyphs": len(held), "baseline": "v8",
              "calibration_train_glyphs": len(train), "calibration_check_glyphs": len(check), "seconds": time.monotonic() - started,
              "ocf_per_session": len(selection), "oracle_per_session": args.oracle_images,
              "near_gap": NEAR_GAP, "far_minimum_gap": FAR_GAP, "minimum_per_render_ocf": MIN_RENDER_OCF,
              "max_abs_log_rate_change": RATE_LIMIT, "calibration_coverage": coverage, "calibration_check": checks,
              "protocol": "Matched near/far pair counts per fit glyph and session, same architectures and fit schedule, "
                          "random label signs within near pairs as control. Rank the full fixed atlas from real spikes "
                          "for held glyphs. Actual test data unused.", **results}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {report['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
