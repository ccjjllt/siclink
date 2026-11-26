"""Compare glyph-conditional predictive precision with the exact v9 Task 2 assignment."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import _repeat_covariance, blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_mbr import contour_signatures
from glyph_retrieval import _reliable_axes, _whiten, assign_glyphs
from kaggle_diagnostics import official_scorer, standardize
from kaggle_hierarchical_gallery_validation import validated_logits
from kaggle_task3_metric_validation import unique_pair_ocf


WINDOWS = ((50, 100), (100, 150), (150, 200), (200, 250))
BANDS = ("global", "conditional", "conditional_repeat")
FUSION_WEIGHT = 0.25


def counts_in_window(counts, start, stop):
    return counts.unflatten(-1, (-1, 10))[..., start // 25:stop // 25].sum(dim=-1).sqrt()


@dataclass
class BandTemplates:
    projection: torch.Tensor
    templates: torch.Tensor
    day_variance: torch.Tensor
    conditional_variance: torch.Tensor
    source_repeat_variance: torch.Tensor


@torch.no_grad()
def fit_band(training, window, neighbors, dimensions):
    normalized = [standardize(counts_in_window(counts, *window))[0] for counts in training.values()]
    means = torch.stack([rows.mean(dim=1) for rows in normalized])
    repeat_covariance = torch.stack([_repeat_covariance(rows) for rows in normalized]).mean(dim=0)
    ridge = repeat_covariance.diagonal().mean().clamp_min(0.05) * 0.25
    regularized = repeat_covariance.clone()
    regularized.diagonal().add_(ridge)
    lower = torch.linalg.cholesky(regularized)
    whitened = _whiten(means.flatten(0, 1), lower).reshape_as(means)
    glyph_mask = torch.ones(means.shape[1], dtype=torch.bool, device=means.device)
    axes = _reliable_axes(whitened, glyph_mask)[:, -dimensions:]
    projection = torch.linalg.solve_triangular(lower.T, axes, upper=True)
    embedded = means @ projection
    templates = embedded.mean(dim=0)
    centered = embedded - templates
    source_days = len(normalized)
    trial_variance = (projection * (repeat_covariance @ projection)).sum(dim=0).clamp_min(1e-5)
    # The source-day sample variance separates repeat-mean noise from variation
    # that a different recording day contributes even when repeat count grows.
    empirical = centered.square().sum(dim=0) / (source_days - 1)
    trial_mean_variance = trial_variance / normalized[0].shape[1]
    global_day = (empirical.mean(dim=0) - trial_mean_variance).clamp_min(0)
    prediction_ensemble = 1 + 1 / source_days
    if neighbors is None:
        corrected_day = prediction_ensemble * global_day
        return BandTemplates(projection, templates, corrected_day, corrected_day, trial_variance)
    trial_count = normalized[0].shape[1]
    local_repeat_mean_noise = torch.stack([
        ((rows - center[:, None]) @ projection).square().sum(dim=1) / (trial_count * (trial_count - 1))
        for rows, center in zip(normalized, means)
    ]).mean(dim=0)
    local_day = (empirical - local_repeat_mean_noise)[neighbors].mean(dim=1).clamp_min(0)
    # Regularize in the units of repeat variance, also when day variance is small.
    local_scale = global_day + trial_mean_variance
    deviation = (local_day - global_day).clamp(-0.75 * local_scale, 1.5 * local_scale)
    conditional = (global_day + 0.45 * deviation).clamp_min(0)
    return BandTemplates(projection, templates, prediction_ensemble * global_day,
                         prediction_ensemble * conditional, trial_variance)


def normalize_assignment_scores(logits):
    zero = logits - logits.mean(dim=1, keepdim=True)
    zero = zero - zero.mean(dim=0, keepdim=True)
    return zero / zero.square().mean().sqrt().clamp_min(1e-6)


def gaussian_scores(query, means, variance):
    precision = variance.reciprocal()
    cross = query @ (means * precision).T
    query_quadratic = query.square() @ precision.T
    template_term = (means.square() * precision + variance.log()).sum(dim=-1)
    return cross - 0.5 * (query_quadratic + template_term)


@torch.no_grad()
def score_band(model, query, window, *, global_only=False):
    observed = standardize(counts_in_window(query, *window))[0] @ model.projection
    center = observed.mean(dim=1)
    repeat_residual = observed - center[:, None]
    repeats = observed.shape[1]
    trial_var = repeat_residual.square().sum(dim=1) / (repeats - 1)
    # All target estimates use only anonymous repeated responses. The normal
    # observation model is anchored by labeled source noise and its existing axes.
    pooled = model.source_repeat_variance.lerp(trial_var.mean(dim=0), 0.35)
    mean_variance = (1.33 * pooled / repeats + 0.03).expand_as(model.templates)
    global_score = normalize_assignment_scores(gaussian_scores(center, model.templates, model.day_variance + mean_variance))
    if global_only:
        return global_score
    conditional_score = gaussian_scores(center, model.templates, model.conditional_variance + mean_variance)
    reliability = (1 / (0.80 + 0.20 * (trial_var / pooled.clamp_min(1e-5)).mean(dim=1).clamp(0.4, 2.5)))[:, None]
    conditional_score = normalize_assignment_scores(conditional_score)
    weighted = normalize_assignment_scores(reliability * conditional_score)
    return (global_score, conditional_score, weighted), {
        "day_variance_mean": float(model.day_variance.mean()),
        "mean_variance_rel_difference": float(((model.conditional_variance - model.day_variance).abs() / mean_variance).mean()),
        "query_reliability_std": float(reliability.std(unbiased=False)),
    }


@torch.no_grad()
def score_recording(sources, query, neighbors, dimensions):
    bands, diagnostics = [], []
    for window in WINDOWS:
        model = fit_band(sources, window, neighbors, dimensions)
        scores, report = score_band(model, query, window)
        bands.append(scores)
        diagnostics.append({"window_ms": window, **report})
    return {name: torch.stack(scores) for name, scores in zip(BANDS, zip(*bands))}, diagnostics


@torch.no_grad()
def calibration_rows(scores, truth, count, generator):
    rows = torch.randperm(len(truth), device=truth.device, generator=generator)[:count]
    correct = truth[rows]
    shared = (scores["global"][:, rows].mean(dim=0) + scores["conditional"][:, rows].mean(dim=0)) * 0.5
    shared.scatter_(1, correct[:, None], -torch.inf)
    candidates = torch.cat((correct[:, None], shared.topk(32, dim=1).indices), dim=1)
    return {key: stack[:, rows].gather(2, candidates[None].expand(len(WINDOWS), -1, -1)).permute(1, 2, 0)
            for key, stack in scores.items()}


def fit_window_weights(features):
    values = torch.cat(features)
    raw = values.new_zeros((len(WINDOWS),), requires_grad=True)
    optimizer = torch.optim.Adam((raw,), lr=0.035)
    labels = torch.zeros(len(values), device=values.device, dtype=torch.long)
    for _ in range(75):
        optimizer.zero_grad(set_to_none=True)
        weights = 0.50 * raw.softmax(dim=0) + 0.50 / len(WINDOWS)
        loss = F.cross_entropy(1.5 * (values @ weights), labels) + 0.02 * raw.square().mean()
        loss.backward()
        optimizer.step()
    return 0.50 * raw.detach().softmax(dim=0) + 0.50 / len(WINDOWS)


def fit_episode_weights(sources, neighbors, args, generator):
    calibration = {name: [] for name in BANDS}
    episodes = []
    for held, counts in sources.items():
        fit = {name: data for name, data in sources.items() if name != held}
        order = torch.randperm(len(counts), device=counts.device, generator=generator)
        scores, _ = score_recording(fit, counts[order], neighbors, args.dimensions)
        sampled = calibration_rows(scores, order, args.calibration_per_episode, generator)
        for name in BANDS:
            calibration[name].append(sampled[name])
        episodes.append({"query_session": held, "template_sessions": list(fit), "labeled_rows": len(sampled["global"])})
    return {name: fit_window_weights(features) for name, features in calibration.items()}, episodes


def row_metrics(matrix, truth, exact_assignment):
    target = matrix.gather(1, truth[:, None])
    rank = 1 + (matrix > target).sum(dim=1)
    return {"assignment_top1": float((exact_assignment == truth).float().mean()),
            "row_top1": float((rank <= 1).float().mean()), "row_top8": float((rank <= 8).float().mean()),
            "row_top32": float((rank <= 32).float().mean()), "true_labels_outside_top32": int((rank > 32).sum()),
            "median_true_label_rank": float(rank.float().median()), "unique_assignment_labels": len(exact_assignment.unique())}


def paired_results(values, anchor):
    by_glyph = (values - anchor).reshape(-1, values.shape[-1]).mean(axis=0)
    return {"ocf": float(values.mean()), "ocf_n": values.size, "paired_delta": float(by_glyph.mean()),
            "paired_glyph_se": float(by_glyph.std(ddof=1) / np.sqrt(len(by_glyph)))}


def evaluate_fold(mouse, sources, query, order, selected, images, masks, neighbors, scorer, args, generator):
    started = time.monotonic()
    weights, episodes = fit_episode_weights(sources, neighbors, args, generator)
    band_scores, diagnostics = score_recording(sources, query, neighbors, args.dimensions)
    device = images.device
    torch.manual_seed(args.seed)
    validated = validated_logits(mouse, sources, query, images, device, args)
    hierarchical = fit_hierarchical_gallery(sources)
    eb = hierarchical.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    v9 = blend_hierarchical_logits(validated, eb)
    print(f"{mouse}: full-gallery exact assignment", flush=True)
    logits = {"v9": v9}
    for kind, scores in band_scores.items():
        uniform = scores.mean(dim=0)
        learned = (scores * weights[kind][:, None, None]).sum(dim=0)
        logits[f"v9_{kind}_uniform"] = v9.lerp(normalize_assignment_scores(uniform), FUSION_WEIGHT)
        logits[f"v9_{kind}_learned"] = v9.lerp(normalize_assignment_scores(learned), FUSION_WEIGHT)
        if kind == "conditional":
            logits["conditional_only"] = learned
    del band_scores
    assigned = {name: assign_glyphs(matrix) for name, matrix in logits.items()}
    decoded = torch.stack([choice[order.argsort()[selected]] for choice in assigned.values()])
    exact, pairs = unique_pair_ocf(decoded, selected.expand_as(decoded), masks, scorer, device, 384)
    metrics = dict(zip(logits, exact.cpu().numpy()))
    reports = {name: {**paired_results(values, metrics["v9"]), **row_metrics(logits[name], order, assigned[name])}
               for name, values in metrics.items()}
    comparison = paired_results(metrics["v9_conditional_learned"], metrics["v9_global_learned"])
    return metrics, {"sources": list(sources), "episode_calibration": episodes, "band_diagnostics": diagnostics,
                     "learned_temporal_weights": {name: values.tolist() for name, values in weights.items()},
                     "metrics": reports, "conditional_vs_global_same_learning": comparison,
                     "exact_unique_ocf_pairs": pairs, "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--held-sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session2", "session4"))
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--calibration-per-episode", type=int, default=640)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--ocf-seed", type=int, default=1601)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_conditional_precision_validation.json"))
    parser.set_defaults(encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if args.ocf_images < 128 or min(args.dimensions, args.calibration_per_episode) < 2:
        parser.error("Provide at least 128 scored rows and valid source episodes.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run full-gallery Task 2 validation on Kaggle GPU.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    masks = images >= 128
    signatures, _ = contour_signatures(images)
    neighbors = (signatures @ signatures.T).fill_diagonal_(-torch.inf).topk(48, dim=1).indices
    del signatures
    scorer = official_scorer(root)
    selected = torch.as_tensor(np.random.default_rng(args.ocf_seed).choice(len(images), args.ocf_images, replace=False), device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 6203)
    output = {"baseline": "v9 complete 65/35 neural and repeat-adapted hierarchical assignment in both mice; v11 mice1 adds separate exact-OCF reranking",
              "gallery_count": len(images), "gallery_complete": True, "ocf_rows_per_block": len(selected),
              "windows_ms_after_onset": WINDOWS, "feature_dimensions_per_band": args.dimensions, "fusion_weight": FUSION_WEIGHT,
              "source_episode_rows": args.calibration_per_episode, "seed": args.seed, "ocf_seed": args.ocf_seed, "mice": {}}
    for mouse in args.mice:
        data = grouped_blocks(root, catalog, mouse, device, len(images))
        record = {"folds": {}}
        output["mice"][mouse] = record
        kept = {}
        for held in args.held_sessions:
            print(f"{mouse}/{held}: source-only conditional covariance and time-band fitting", flush=True)
            sources = {name: trial for name, trial in data.items() if name != held}
            order = torch.randperm(len(images), device=device, generator=generator)
            metrics, report = evaluate_fold(mouse, sources, data[held][order], order, selected,
                                            images, masks, neighbors, scorer, args, generator)
            record["folds"][held] = report
            for method, values in metrics.items():
                kept.setdefault(method, []).append(values)
            stacks = {method: np.stack(values) for method, values in kept.items()}
            record["summary"] = {name: paired_results(values, stacks["v9"]) for name, values in stacks.items()}
            record["conditional_vs_global_same_learning"] = paired_results(stacks["v9_conditional_learned"], stacks["v9_global_learned"])
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"mouse": mouse, "held_session": held, **report}), flush=True)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
