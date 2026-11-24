"""Train a spike-count baseline and write the SiClink submission on Kaggle."""

import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
import torch


DATA_DIR = "siclink_chinese_character_decoding_dataset"
COMPETITION = "siclink-mouse-visual-reconstruction-bci-challenge"
BIN_MS = 25
TASK_SIZES = {"task1": 2720, "task2": 5496, "task3": 21984}


def read_block(root, entry, sample_rate, window_ms, device, stats=None):
    """Bin individual trials, normalize within session, and average repeats."""
    with np.load(root / entry["path"]) as raw:
        sample_ids = raw["sample_ids"]
        group_ptr = raw["group_ptr"]
        trial_ptr = raw["trial_ptr"]
        targets = raw["target"] if "target" in raw else None
        samples_per_bin = sample_rate * BIN_MS // 1000
        num_bins = (window_ms + BIN_MS - 1) // BIN_MS
        units = entry["n_units"]
        trial_index = np.repeat(np.arange(len(trial_ptr) - 1), np.diff(trial_ptr))
        times = raw["spike_samples"] // samples_per_bin
        indices = (trial_index * units + raw["unit_index"]) * num_bins + times
        counts = np.bincount(indices, minlength=(len(trial_ptr) - 1) * units * num_bins)

    counts = torch.from_numpy(counts).to(device=device, dtype=torch.float32).reshape(-1, units, num_bins)
    totals = counts.sum(dim=2, keepdim=True)
    features = torch.cat((counts, totals), dim=2).sqrt_().flatten(1)
    if stats is None:
        scale = (features.var(dim=0, unbiased=False) + 0.05).sqrt()
        stats = (features.mean(dim=0), scale)
    mean, scale = stats
    features = (features - mean) / scale

    repeats = np.diff(group_ptr)
    # Every block in this release has one repeat count shared by its samples.
    if np.any(repeats != repeats[0]):
        raise ValueError(f"Expected a fixed repeat count in {entry['path']}")
    features = features.reshape(len(sample_ids), int(repeats[0]), -1).mean(dim=1)
    labels = None if targets is None else torch.as_tensor(targets, dtype=torch.long, device=device)
    return sample_ids, features, labels, stats


def summarize_training(features, labels, num_glyphs):
    totals = features.new_zeros((num_glyphs, features.shape[1]))
    totals.index_add_(0, labels, features)
    counts = torch.bincount(labels, minlength=num_glyphs)
    prototypes = totals / counts[:, None]
    residual = features - prototypes[labels]
    covariance = residual.T @ residual / len(residual)
    return prototypes, covariance


def fit_ridge(features, images):
    feature_mean = features.mean(dim=0)
    image_mean = images.mean(dim=0)
    centered = features - feature_mean
    gram = centered.T @ centered / len(features)
    gram.diagonal().add_(0.16)
    weights = torch.cholesky_solve(centered.T @ (images - image_mean) / len(features),
                                   torch.linalg.cholesky(gram))
    return weights, image_mean - feature_mean @ weights


def retrieve(features, prototypes, covariance):
    regularized = covariance.clone()
    regularized.diagonal().add_(0.35)
    weights = torch.cholesky_solve(prototypes.T, torch.linalg.cholesky(regularized)).T
    bias = -0.5 * (prototypes * weights).sum(dim=1)
    return (features @ weights.T + bias).argmax(dim=1)


def fit_mouse(root, catalog, mouse, device):
    with np.load(root / "train" / "pictures.npz") as data:
        masks = torch.as_tensor(data["images"] >= 128, device=device)
    images = masks.flatten(1).float()
    glyph_shape = tuple(catalog["image_size"])
    image_bank = masks.to(dtype=torch.uint8).mul_(255)
    foreground = images.mean().item()
    blocks = sorted((block for block in catalog["blocks"] if block["mouse"] == mouse),
                    key=lambda block: (block["task"], block["session"]))
    rate, window = catalog["sample_rate_hz"], catalog["window_ms"]

    session_training = {}
    for entry in blocks:
        if entry["task"] == "train":
            _, features, labels, stats = read_block(root, entry, rate, window, device)
            prototypes, covariance = summarize_training(features, labels, len(images))
            session_training[entry["session"]] = (prototypes, covariance, stats)
    pooled = torch.stack([item[0] for item in session_training.values()]).mean(dim=0)
    pooled_cov = torch.stack([item[1] for item in session_training.values()]).mean(dim=0)
    pooled_weights, pooled_bias = fit_ridge(pooled, images)
    print(f"{mouse} on {device}: fitted {len(session_training)} labeled sessions", flush=True)

    predictions = []
    for entry in blocks:
        task, session = entry["task"], entry["session"]
        if task == "train":
            continue
        # A cross-day block gets only label-free calibration from its own spikes.
        local = session_training.get(session)
        ids, features, _, _ = read_block(root, entry, rate, window, device, local[2] if local else None)
        if task == "task1":
            local_weights, local_bias = fit_ridge(local[0], images)
            estimate = 0.70 * (features @ local_weights + local_bias)
            estimate += 0.30 * (features @ pooled_weights + pooled_bias)
            # Blend a fixed threshold with the labeled training bank's foreground fraction.
            area_threshold = torch.quantile(estimate, 1.0 - foreground, dim=1, keepdim=True)
            predicted_images = (estimate >= 0.25 * 0.45 + 0.75 * area_threshold)
            predicted_images = predicted_images.reshape(-1, *glyph_shape).to(torch.uint8).mul_(255)
        else:
            if local is None:
                prototypes, covariance = pooled, pooled_cov
            else:
                prototypes = 0.75 * local[0] + 0.25 * pooled
                covariance = 0.75 * local[1] + 0.25 * pooled_cov
            predicted_images = image_bank[retrieve(features, prototypes, covariance)]
        predictions.append((task, ids, predicted_images.cpu().numpy()))
        print(f"{mouse} {task}/{session}: {len(ids):,} predictions", flush=True)
    return predictions


def write_submission(path, predictions):
    rows = {task: {} for task in TASK_SIZES}
    for task, ids, images in predictions:
        for sample_id, pixels in zip(ids, images):
            if sample_id in rows[task]:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            rows[task][sample_id] = pixels
    ordered_ids = {task: [f"test_{task}_pic{i}" for i in range(count)] for task, count in TASK_SIZES.items()}
    for task, ids in ordered_ids.items():
        if rows[task].keys() != set(ids):
            raise ValueError(f"{task} ID coverage differs from the official sample submission")

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        with archive.open("submission.csv", "w") as entry, io.TextIOWrapper(entry, newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("sample_id", "prediction"))
            for task, ids in ordered_ids.items():
                for sample_id in ids:
                    png = io.BytesIO()
                    Image.fromarray(rows[task][sample_id]).save(png, format="PNG")
                    writer.writerow((sample_id, base64.b64encode(png.getvalue()).decode("ascii")))
    print(f"{path}: {sum(TASK_SIZES.values()):,} predictions", flush=True)


def main():
    inputs = Path("/kaggle/input")
    locations = (inputs / "competitions" / COMPETITION / DATA_DIR, inputs / COMPETITION / DATA_DIR)
    root = next((path for path in locations if (path / "catalog.json").is_file()), None)
    if root is None:
        root = next((path.parent for path in inputs.rglob("catalog.json") if path.parent.name == DATA_DIR), None)
    if root is None:
        raise FileNotFoundError(f"Competition Data is not mounted. Kaggle inputs: {list(inputs.iterdir())}")
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    gpu_count = torch.cuda.device_count()
    print(f"Data: {root} | CUDA devices: {gpu_count}", flush=True)
    torch.set_num_threads(4)
    mice = sorted({block["mouse"] for block in catalog["blocks"]})
    workers = min(len(mice), max(gpu_count, 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = []
        for index, mouse in enumerate(mice):
            device = torch.device(f"cuda:{index % gpu_count}" if gpu_count else "cpu")
            jobs.append(executor.submit(fit_mouse, root, catalog, mouse, device))
        predictions = [group for job in jobs for group in job.result()]
    write_submission(Path("/kaggle/working/kaggle_submission.zip"), predictions)


if __name__ == "__main__":
    main()
