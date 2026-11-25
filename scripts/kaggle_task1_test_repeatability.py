"""Measure neural re-identification from disjoint official Task 1 repeats."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import load_catalog, locate_data, read_trials, training_groups
from kaggle_diagnostics import feature_view, standardize


VIEWS = ("evoked-total", "early-late", "evoked-multiscale")


def fit_signal_space(training):
    values, normalization = standardize(training)
    mean = values.mean(1)
    residuals = (values - mean[:, None]).flatten(0, 1)
    covariance = residuals.T @ residuals / (len(values) * (values.shape[1] - 1))
    covariance.diagonal().add_(0.20)
    factor = torch.linalg.cholesky(covariance)
    first, second = values[:, :2].mean(1), values[:, 2:].mean(1)
    first = torch.linalg.solve_triangular(factor, (first - first.mean(0)).T.contiguous(), upper=False).T
    second = torch.linalg.solve_triangular(factor, (second - second.mean(0)).T.contiguous(), upper=False).T
    cross = first.T @ second / (len(mean) - 1)
    variance, basis = torch.linalg.eigh((cross + cross.T) * 0.5)
    projection = torch.linalg.solve_triangular(factor.T, basis.contiguous(), upper=True)
    return normalization, projection, variance.clamp_min(0)


def inspect_split(left, right, variances, repeats):
    left, right = left - left.mean(0), right - right.mean(0)
    labels = torch.arange(len(left))
    correlation = (left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-10)
    shrink = variances / (variances + 1 / repeats)
    uncertainty = (1 + shrink) / repeats
    right_mean = right * shrink
    bayes = lambda samples, centers: (samples / uncertainty) @ centers.T - 0.5 * (centers.square() / uncertainty).sum(1)
    scores = {"whitened_cosine": F.normalize(left) @ F.normalize(right).T,
              "latent_gaussian_forward": bayes(left, right_mean),
              "latent_gaussian_reverse": bayes(right, left * shrink)}
    output = {"split_half_correlation": float(correlation)}
    for name, matrix in scores.items():
        winners = matrix.topk(5, dim=1).indices
        output[name] = {"top1": float((winners[:, 0] == labels).float().mean()),
                        "top5": float((winners == labels[:, None]).any(1).float().mean())}
    return output


def inspect_block(counts, space, rng, max_repeats):
    normalization, projection, variance = space
    counts, _ = standardize(counts, normalization)
    embedding = counts @ projection
    orders = torch.as_tensor(rng.random(embedding.shape[:2]).argsort(axis=1))
    embedding = embedding.gather(1, orders[:, :, None].expand_as(embedding))
    return {n: inspect_split(embedding[:, :n].mean(1), embedding[:, -n:].mean(1), variance, n)
            for n in (1, 2, 4, 5, 10) if n <= max_repeats}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/test_repeatability.json"))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run against the mounted competition input in a Kaggle Notebook.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)
    training_entries = {(item["mouse"], item["session"]): item for item in catalog["blocks"] if item["task"] == "train"}
    test_entries = [item for item in catalog["blocks"] if item["task"] == "task1" and item["split"] == "test"]
    output = {"protocol": "Spike-only identity of disjoint repeat sets within each officially grouped example. The training comparison uses different labeled glyphs from the signal-space fit. No images, test labels, sample IDs, or cross-recording row order enter scoring.",
              "seed": args.seed, "blocks": []}
    for item in test_entries:
        key = (item["mouse"], item["session"])
        block = read_trials(root, catalog, training_entries[key], device)
        train = training_groups(block, int(block.labels.max()) + 1)
        query = read_trials(root, catalog, item, device).counts
        order = rng.permutation(len(train))
        reference, fitting = train[order[:len(query)]], train[order[len(query):]]
        row = {"mouse": key[0], "session": key[1], "groups": len(query), "test_repeats": query.shape[1], "views": {}}
        for view in VIEWS:
            space = fit_signal_space(feature_view(fitting, view))
            row["views"][view] = {"held_training": inspect_block(feature_view(reference, view), space, rng, 2),
                                   "official_grouped_repeats": inspect_block(feature_view(query, view), space, rng,
                                                                             query.shape[1] // 2)}
        output["blocks"].append(row)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"{key[0]}/{key[1]}: compared disjoint neural repeat sets for {len(query)} official groups", flush=True)


if __name__ == "__main__":
    main()
