"""Measure how stimulus windows affect the original baseline on labeled Kaggle folds."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs, split_repeats
from kaggle_baseline import fit_ridge, retrieve, summarize_training


def feature_view(counts, name):
    counts = counts.unflatten(-1, (-1, 10))
    if name == "original":
        values = torch.cat((counts, counts.sum(-1, keepdim=True)), dim=-1)
    elif name == "evoked-multiscale":
        delayed = counts[..., 2:]
        totals = delayed.reshape(*delayed.shape[:-1], 4, 2).sum(-1)
        values = torch.cat((totals, totals.sum(-1, keepdim=True)), dim=-1)
    elif name == "evoked-total":
        values = counts[..., 2:].sum(-1, keepdim=True)
    elif name == "early-late":
        values = torch.stack((counts[..., 2:5].sum(-1), counts[..., 5:].sum(-1)), dim=-1)
    else:
        raise ValueError(name)
    return values.sqrt_().flatten(-2)


def standardize(counts, stats=None):
    if stats is None:
        trials = counts.flatten(0, 1)
        stats = (trials.mean(0), (trials.var(0, unbiased=False) + 0.05).sqrt())
    return (counts - stats[0]) / stats[1], stats


def prototypes_and_noise(counts):
    labels = torch.arange(len(counts), device=counts.device).repeat_interleave(counts.shape[1])
    return summarize_training(counts.flatten(0, 1), labels, len(counts))


def summarize_metric(scorer, predictions, reference, device):
    values = scorer(predictions.cpu().numpy().astype(np.float32) / 255,
                    reference.cpu().numpy() >= 128, device=str(device), batch_size=4)
    return {"ocf": float(values.mean()), "ocf_n": len(values)}


def task1(groups, images, fit, held, score, device, size):
    train, stats = {}, {}
    for session, counts in groups.items():
        normalized, stats[session] = standardize(counts[fit])
        train[session] = normalized.mean(1)
    labels = (images[fit] >= 128).float().flatten(1)
    pooled = torch.stack(list(train.values())).mean(0)
    global_weights, global_bias = fit_ridge(pooled, labels)
    n_per_block = max(1, size // 2)
    estimate, targets = [], []
    for session in ("session1", "session4"):
        query, _ = standardize(groups[session][held[:n_per_block]], stats[session])
        features = query.mean(1)
        local_weights, local_bias = fit_ridge(train[session], labels)
        pixels = 0.70 * (features @ local_weights + local_bias)
        pixels += 0.30 * (features @ global_weights + global_bias)
        threshold = 0.25 * 0.45 + 0.75 * torch.quantile(pixels, 1 - labels.mean().item(), dim=1, keepdim=True)
        estimate.append((pixels >= threshold).reshape(-1, 64, 64).to(torch.uint8) * 255)
        targets.append(images[held[:n_per_block]])
    return summarize_metric(score, torch.cat(estimate), torch.cat(targets), device)


def seen_task(groups, images, fit_counts, query_counts, cross_day, score, device, size, seed):
    learned = {}
    for session, counts in fit_counts.items():
        features, stats = standardize(counts)
        learned[session] = (*prototypes_and_noise(features), stats)
    pooled = torch.stack([item[0] for item in learned.values()]).mean(0)
    pool_noise = torch.stack([item[1] for item in learned.values()]).mean(0)
    correct, n = 0, 0
    estimate, targets = [], []
    random = np.random.default_rng(seed)
    for session, counts in query_counts.items():
        local = learned.get(session)
        features, _ = standardize(counts, None if cross_day else local[2])
        if cross_day:
            templates, noise = pooled, pool_noise
        else:
            templates = 0.75 * local[0] + 0.25 * pooled
            noise = 0.75 * local[1] + 0.25 * pool_noise
        prediction = retrieve(features.mean(1), templates, noise)
        correct += int((prediction == torch.arange(len(counts), device=device)).sum())
        n += len(counts)
        selection = random.choice(len(images), min(len(images), max(1, size // len(query_counts))), replace=False)
        selection = torch.as_tensor(selection, device=device)
        estimate.append((images[prediction[selection]] >= 128).to(torch.uint8) * 255)
        targets.append(images[selection])
    summary = summarize_metric(score, torch.cat(estimate), torch.cat(targets), device)
    summary.update(top1=correct / n, top1_n=n)
    return summary


def official_scorer(root):
    bundled = root.parent / "participant_package_en" / "participant_package_en"
    if not (bundled / "src" / "ocf" / "score.py").exists():
        raise FileNotFoundError(f"Official OCF module missing at {bundled}")
    sys.path.insert(0, str(bundled))
    from src.ocf.score import score
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocf-images", type=int, default=32, help="paired OCF images per mouse/task/view")
    parser.add_argument("--views", nargs="+", default=["original", "evoked-multiscale", "evoked-total", "early-late"])
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/diagnostics.json"))
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    score = official_scorer(root)
    fit, held = split_glyphs(len(images), 320, args.seed, device)
    results = []

    for mouse in ("mice1", "mice2"):
        raw = grouped_blocks(root, catalog, mouse, device, len(images))
        for view in args.views:
            groups = {session: feature_view(counts, view) for session, counts in raw.items()}
            task2_query = {"session4": groups["session4"]}
            task2_train = {session: counts for session, counts in groups.items() if session != "session4"}
            repeat_folds = {session: split_repeats(counts, args.seed + index)
                            for index, (session, counts) in enumerate(groups.items())}
            task3_train = {session: fold[0] for session, fold in repeat_folds.items()}
            task3_query = {session: repeat_folds[session][1] for session in ("session1", "session4")}
            result = {"mouse": mouse, "view": view,
                      "task1": task1(groups, images, fit, held, score, device, args.ocf_images),
                      "task2": seen_task(groups, images, task2_train, task2_query, True,
                                         score, device, args.ocf_images, args.seed),
                      "task3": seen_task(groups, images, task3_train, task3_query, False,
                                         score, device, args.ocf_images, args.seed)}
            results.append(result)
            print(json.dumps(result), flush=True)
            args.output.write_text(json.dumps({"seed": args.seed, "device": str(device),
                                               "seconds": time.monotonic() - started,
                                               "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
