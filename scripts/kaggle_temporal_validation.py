"""Validate timing filters learned from repeatable, cross-session glyph responses."""

import argparse
import json
from pathlib import Path
import time

import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs, split_repeats
from kaggle_diagnostics import official_scorer, seen_task, task1


def temporal_filters(groups, time_bins, components=3, regularization=0.25):
    """Find within-unit time axes that agree across sessions after whitening repeat noise."""
    centers = []
    residual_covariance = None
    for counts in groups.values():
        stabilized = counts.unflatten(-1, (-1, time_bins)).sqrt()
        average = stabilized.mean(dim=1)
        centers.append(average - average.mean(dim=0))
        residual = (stabilized - average[:, None]).flatten(0, 1)
        covariance = torch.einsum("gut,guv->utv", residual, residual) / (len(counts) * (counts.shape[1] - 1))
        residual_covariance = covariance if residual_covariance is None else residual_covariance + covariance
    repeated = torch.stack(centers)
    n_sessions, n_glyphs = repeated.shape[:2]
    total = repeated.sum(dim=0)
    signal = (torch.einsum("gut,guv->utv", total, total)
              - torch.einsum("sgut,sguv->utv", repeated, repeated)) / (n_sessions * (n_sessions - 1) * n_glyphs)
    noise = residual_covariance / n_sessions
    diagonal = noise.diagonal(dim1=-1, dim2=-2)
    diagonal.add_(regularization * diagonal.mean(dim=1, keepdim=True).clamp_min(0.02))
    factor = torch.linalg.cholesky(noise)
    whitened = torch.linalg.solve_triangular(factor, signal, upper=False)
    whitened = torch.linalg.solve_triangular(factor, whitened.transpose(-1, -2), upper=False)
    values, basis = torch.linalg.eigh((whitened + whitened.transpose(-1, -2)) * 0.5)
    values = values[:, -components:].clamp_min(0)
    filters = torch.linalg.solve_triangular(factor.transpose(-1, -2), basis[:, :, -components:], upper=True)
    weights = (values / (values + 0.05)).sqrt()
    return filters, weights, values


def temporal_features(counts, time_bins, filters, weights):
    counts = counts.unflatten(-1, (-1, time_bins))
    standardized = torch.einsum("grut,utd->grud", counts.sqrt(), filters) * weights
    rate = counts[..., time_bins // 5:].sum(-1, keepdim=True).sqrt()
    features = torch.cat((standardized, rate), dim=-1)
    return features.flatten(-2)


def transform(groups, time_bins, filters, weights):
    return {session: temporal_features(counts, time_bins, filters, weights) for session, counts in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-ms", type=int, default=10)
    parser.add_argument("--ocf-images", type=int, default=32)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/temporal.json"))
    args = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    score = official_scorer(root)
    fit, held = split_glyphs(len(images), 320, args.seed, device)
    time_bins = catalog["window_ms"] // args.bin_ms
    results = []
    for mouse in ("mice1", "mice2"):
        raw = grouped_blocks(root, catalog, mouse, device, len(images), bin_ms=args.bin_ms)
        task1_fit = {s: counts[fit] for s, counts in raw.items()}
        filters, weights, eigenvalues = temporal_filters(task1_fit, time_bins)
        del task1_fit
        opened = task1(transform(raw, time_bins, filters, weights), images, fit, held,
                       score, device, args.ocf_images)

        task2_fit = {s: counts for s, counts in raw.items() if s != "session4"}
        filters, weights, _ = temporal_filters(task2_fit, time_bins)
        cross = seen_task({}, images, transform(task2_fit, time_bins, filters, weights),
                          transform({"session4": raw["session4"]}, time_bins, filters, weights),
                          True, score, device, args.ocf_images, args.seed)

        task3_folds = {s: split_repeats(counts, args.seed + i) for i, (s, counts) in enumerate(raw.items())}
        task3_fit = {s: pair[0] for s, pair in task3_folds.items()}
        held_repeats = {s: task3_folds[s][1] for s in ("session1", "session4")}
        filters, weights, _ = temporal_filters(task3_fit, time_bins)
        single = seen_task({}, images, transform(task3_fit, time_bins, filters, weights),
                           transform(held_repeats, time_bins, filters, weights), False,
                           score, device, args.ocf_images, args.seed)
        report = {"mouse": mouse, "view": f"learned-{args.bin_ms}ms", "task1": opened,
                  "task2": cross, "task3": single,
                  "median_repeatable_eigenvalues": eigenvalues.median(0).values.tolist()}
        results.append(report)
        print(json.dumps(report), flush=True)
        args.output.write_text(json.dumps({"seed": args.seed, "device": str(device),
                                           "seconds": time.monotonic() - start,
                                           "results": results}, indent=2) + "\n")
        del raw, task2_fit, task3_folds, task3_fit, held_repeats


if __name__ == "__main__":
    main()
