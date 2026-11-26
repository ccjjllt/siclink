"""Compare matched within- and cross-recording empirical spike metrics on Task 3."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_episodic_assignment import _projection
from glyph_mbr import contour_signatures
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_task3_density_top7_recheck import top7_predictions
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_part_discovery import standardize_rows
from kaggle_task3_repeat_posterior import fit_models, score_sources


VIEW = "evoked-multiscale"
WEIGHTS = (0.0, 0.15, 0.25, 0.40)
METHODS = ("frozen", "same_day", "cross_day")


class PairMetric(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.query = self.adjustment(dimensions)
        self.support = self.adjustment(dimensions)

    @staticmethod
    def adjustment(dimensions):
        head = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, 64), nn.SiLU(), nn.Linear(64, dimensions))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def encode_queries(self, query):
        return F.normalize(query + 0.30 * self.query(query), dim=-1)

    def encode_support(self, template):
        return F.normalize(template + 0.30 * self.support(template), dim=-1)


@torch.no_grad()
def source_representation(sources, fit_glyphs, args):
    selected = {name: counts[fit_glyphs] for name, counts in sources.items()}
    projection, training = _projection(selected, VIEW, args.metric_dimensions)
    support = []
    for name, all_counts in sources.items():
        _, moments = standardize(feature_view(selected[name], VIEW))
        normalized, _ = standardize(feature_view(all_counts, VIEW), moments)
        support.append(normalized.mean(dim=1) @ projection)
    return projection, training, tuple(support)


@torch.no_grad()
def negative_banks(trials, seed):
    support = F.normalize(trials.mean(dim=(0, 2)), dim=-1)
    score = (support @ support.T).fill_diagonal_(-torch.inf)
    hard = score.topk(24, dim=1).indices
    sampler = torch.Generator(device=trials.device).manual_seed(seed)
    uniform = torch.rand(score.shape, generator=sampler, device=trials.device).fill_diagonal_(-1)
    uniform.scatter_(1, hard, -1)
    return hard, uniform.topk(40, dim=1).indices


def fit_metric(trials, banks, args, *, across_recordings):
    sessions, classes, repeats, dimensions = trials.shape
    device = trials.device
    if repeats != 3 or sessions < 2:
        raise ValueError("Train on three disjoint repeats in multiple source recordings.")
    hard, broad = banks
    torch.manual_seed(args.seed + 107)
    draw = torch.Generator(device=device).manual_seed(args.seed + 108)
    model = PairMetric(dimensions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.04)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.metric_epochs, eta_min=0.0002)
    template_sums = trials.sum(dim=2)
    for epoch in range(args.metric_epochs):
        total_loss = trials.new_zeros(())
        for day in range(sessions):
            shuffled = torch.randperm(classes, generator=draw, device=device)
            for target in shuffled.split(192):
                source_repeat = torch.randint(repeats, (len(target),), generator=draw, device=device)
                peer = (day + torch.randint(1, sessions, (len(target),), generator=draw, device=device)) % sessions
                reference_day = peer if across_recordings else torch.full_like(peer, day)
                local_order = torch.rand((len(target), hard.shape[1]), generator=draw, device=device).argsort(dim=1)[:, :12]
                broad_order = torch.rand((len(target), broad.shape[1]), generator=draw, device=device).argsort(dim=1)[:, :20]
                negative = torch.cat((hard[target].gather(1, local_order), broad[target].gather(1, broad_order)), dim=1)
                candidates = torch.cat((target[:, None], negative), dim=1)
                response = trials[day, target, source_repeat]
                references = (template_sums[reference_day[:, None], candidates]
                              - trials[reference_day[:, None], candidates, source_repeat[:, None]]) / (repeats - 1)
                observed = model.encode_queries(response)
                support = model.encode_support(references)
                scores = 12 * (observed[:, None] * support).sum(dim=-1)
                loss = F.cross_entropy(scores, torch.zeros(len(target), dtype=torch.long, device=device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach() * len(target)
        schedule.step()
        if epoch == 0 or epoch + 1 == args.metric_epochs:
            print(json.dumps({"metric": "cross_day" if across_recordings else "same_day", "epoch": epoch + 1,
                              "source_loss": float(total_loss / (sessions * classes))}), flush=True)
    return model.eval()


@torch.no_grad()
def neural_scores(query, local_repeats, projection, peers, models):
    local, normalization = standardize(feature_view(local_repeats, VIEW))
    query, _ = standardize(feature_view(query, VIEW), normalization)
    query = query.mean(dim=1) @ projection
    local = local.mean(dim=1) @ projection
    result = {}
    for name, model in models.items():
        encoded = F.normalize(query, dim=-1) if model is None else model.encode_queries(query)
        templates = tuple(F.normalize(template, dim=-1) if model is None else model.encode_support(template)
                          for template in (local, *peers))
        local_logits = standardize_rows(encoded @ templates[0].T)
        peer_mean = torch.stack(templates[1:]).mean(dim=0)
        peer_logits = standardize_rows(encoded @ peer_mean.T)
        result[name] = standardize_rows(0.50 * local_logits + 0.50 * peer_logits)
    return result


def ranks(scores, labels):
    correct = scores.gather(1, labels[:, None])
    rank = 1 + (scores > correct).sum(dim=1)
    return {f"top{depth}": float((rank <= depth).float().mean()) for depth in (1, 7, 32)}


def variant_key(method, weight):
    return f"{method}_{weight:.2f}" if weight else "v12_top7"


@torch.no_grad()
def evaluate(baseline_model, session, query, local, projection, peers, heads, images, masks, targets, truth, scorer, args):
    v11, _, density = score_sources(baseline_model, query, images, session, "mice1", args.dimensions)
    anchor, chosen = top7_predictions(v11, density)
    next_six = v11.clone().scatter_(1, anchor[:, None], -torch.inf).topk(15, dim=1).indices[:, :6]
    original_pool = torch.cat((anchor[:, None], next_six), dim=1)
    choices = {"v12_top7": chosen, "v12_full": density.argmax(dim=1)}
    logits = neural_scores(query, local, projection, peers, heads)
    full = {"v12_full": ranks(density, truth)}
    for method, score in logits.items():
        full[f"{method}_standalone"] = ranks(score, truth)
        for weight in WEIGHTS[1:]:
            key = variant_key(method, weight)
            adjusted = density.lerp(score, weight)
            if weight == 0.25:
                full[key] = ranks(adjusted, truth)
                choices[f"{method}_full_0.25"] = adjusted.argmax(dim=1)
            position = adjusted.gather(1, original_pool).argmax(dim=1, keepdim=True)
            choices[key] = original_pool.gather(1, position).squeeze(1)
    rows = truth.argsort()[targets]
    predictions = torch.stack([guess[rows] for guess in choices.values()])
    values, count = unique_pair_ocf(predictions, targets.expand_as(predictions), masks, scorer, images.device, 384)
    return dict(zip(choices, values.cpu().numpy())), {
        "full_gallery_rank": full, "v12_top7_contains_true": float((original_pool == truth[:, None]).any(dim=1).float().mean()),
        "identity_top1": {name: float((choice == truth).float().mean()) for name, choice in choices.items()},
        "changes_vs_original": {name: int((choice != chosen).sum()) for name, choice in choices.items()}, "unique_exact_pairs": count}


def paired(values, baseline):
    difference = (values - baseline).reshape(-1, values.shape[-1]).mean(axis=0)
    return {"ocf": float(values.mean()), "paired_delta": float(difference.mean()),
            "paired_glyph_se": float(difference.std(ddof=1) / len(difference)**0.5), "rows": int(values.size)}


def summarize(data, baseline="v12_top7"):
    return {name: paired(scores, data[baseline]) for name, scores in data.items()}


def append_rows(records, rows):
    for name, scores in rows.items():
        records.setdefault(name, []).append(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-seed", type=int, default=7481)
    parser.add_argument("--repeat-seed", type=int, default=7469)
    parser.add_argument("--ocf-images", type=int, default=192)
    parser.add_argument("--calibration-images", type=int, default=128)
    parser.add_argument("--metric-dimensions", type=int, default=96)
    parser.add_argument("--metric-epochs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_measured_metric_validation.json"))
    parser.set_defaults(seed=211, encoder_epochs=18, epochs=8, dimensions=128)
    args = parser.parse_args()
    if min(args.ocf_images, args.calibration_images) < 2 or min(args.metric_dimensions, args.metric_epochs) < 1:
        parser.error("Provide positive split and fit sizes.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the measured metric comparison with Kaggle GPU and labeled competition data.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    masks = images >= 128
    signature, energy = contour_signatures(images)
    scorer = official_scorer(root)
    catalog = load_catalog(root)
    recordings = grouped_blocks(root, catalog, "mice1", device, len(images))
    training, held, inner_fit, inner_held = {}, {}, {}, {}
    for i, (name, counts) in enumerate(recordings.items()):
        training[name], held[name] = split_repeats(counts, args.repeat_seed + i)
        inner_fit[name], inner_held[name] = split_repeats(training[name], args.repeat_seed + 200 + i)
    del recordings
    rng = np.random.default_rng(args.sample_seed)
    partition = torch.as_tensor(rng.permutation(len(images)), device=device)
    outer_rows, inner_rows, metric_rows = partition.split((args.ocf_images, args.calibration_images,
                                                          len(images) - args.ocf_images - args.calibration_images))
    if len(metric_rows) < 512:
        parser.error("Keep at least 512 independent source metric glyphs.")
    inner_model, inner_report = fit_models(inner_fit, images, signature, energy, "mice1", args)
    outer_model, outer_report = fit_models(training, images, signature, energy, "mice1", args)
    print("Fitted exact v12 validation topology on two and three disjoint repeat banks", flush=True)
    output = {"mouse": "mice1", "complete_gallery": len(images), "seed": args.seed, "repeat_seed": args.repeat_seed,
              "sample_seed": args.sample_seed, "metric_fit_glyphs": len(metric_rows), "inner_scoring_glyphs": len(inner_rows),
              "outer_scoring_glyphs": len(outer_rows), "frozen_primary_weight": 0.25, "calibration_weight_grid": WEIGHTS,
              "v12_fits": {"inner": inner_report, "outer": outer_report}, "sessions": {},
              "protocol": "Same-day and cross-day metric heads use exactly matched source-day query repeats and negatives. "
                          "The outer recording and both OCF-scored glyph partitions are excluded from metric fitting. "
                          "Both heads use identical measured templates at inference. The inner and outer held trials are removed "
                          "before their respective target-session template fits. No actual competition test data are accessed."}
    inner_scores, outer_scores = {}, {}
    for session in training:
        sources = {name: values for name, values in training.items() if name != session}
        projection, trials, peers = source_representation(sources, metric_rows, args)
        bank = negative_banks(trials, args.seed + 777)
        heads = {"frozen": None,
                 "same_day": fit_metric(trials, bank, args, across_recordings=False),
                 "cross_day": fit_metric(trials, bank, args, across_recordings=True)}
        inner_order = torch.as_tensor(rng.permutation(len(images)), device=device)
        outer_order = torch.as_tensor(rng.permutation(len(images)), device=device)
        calibration, inner_diag = evaluate(inner_model, session, inner_held[session][inner_order], inner_fit[session],
                                          projection, peers, heads, images, masks, inner_rows, inner_order, scorer, args)
        final, outer_diag = evaluate(outer_model, session, held[session][outer_order], training[session], projection,
                                     peers, heads, images, masks, outer_rows, outer_order, scorer, args)
        append_rows(inner_scores, calibration)
        append_rows(outer_scores, final)
        output["sessions"][session] = {"metric_training_sessions": list(sources), "inner": inner_diag,
                                       "outer": outer_diag, "paired_ocf": summarize(final)}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, "paired_ocf": summarize(final)}), flush=True)

    inner = {name: np.stack(samples) for name, samples in inner_scores.items()}
    outer = {name: np.stack(samples) for name, samples in outer_scores.items()}
    chosen = {method: max(WEIGHTS, key=lambda weight: (inner[variant_key(method, weight)].mean(), -weight))
              for method in METHODS}
    output["paired_outer_ocf"] = summarize(outer)
    output["paired_cross_vs_same_fixed_weight"] = paired(outer["cross_day_0.25"], outer["same_day_0.25"])
    output["paired_cross_vs_frozen_fixed_weight"] = paired(outer["cross_day_0.25"], outer["frozen_0.25"])
    output["inner_calibration_ocf"] = summarize(inner)
    output["calibrated_weights"] = chosen
    output["calibrated_outer_ocf"] = {method: paired(outer[variant_key(method, weight)], outer["v12_top7"])
                                      for method, weight in chosen.items()}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
