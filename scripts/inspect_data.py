"""Summarize the Kaggle catalog and inspect one mouse/session on CPU."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "siclink_chinese_character_decoding_dataset"
)
TASKS = ("train", "task1", "task2", "task3")
BIN_MS = 25


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_ROOT, help="directory containing catalog.json")
    parser.add_argument("--split", choices=TASKS, default="train")
    parser.add_argument("--mouse", choices=("mice1", "mice2"), default="mice1")
    parser.add_argument("--session", help="defaults to the first available session for the mouse and split")
    args = parser.parse_args()

    catalog_path = args.data / "catalog.json"
    if not catalog_path.is_file():
        parser.error(f"{catalog_path} is missing. See the download instructions in README.md or pass --data.")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    blocks = catalog["blocks"]
    by_session = {
        block["session"]: block
        for block in blocks
        if block["task"] == args.split and block["mouse"] == args.mouse
    }
    session = args.session or min(by_session, default="")
    if session not in by_session:
        available = ", ".join(sorted(by_session)) or "none"
        parser.error(f"No matching block for {args.split}/{args.mouse}. Available sessions: {available}.")
    block = by_session[session]

    sample_counts = Counter()
    block_counts = Counter()
    for item in blocks:
        sample_counts[item["task"]] += item["n_samples"]
        block_counts[item["task"]] += 1
    height, width = catalog["image_size"]
    window_ms = catalog["window_ms"]
    sample_rate = catalog["sample_rate_hz"]
    print(f"Data: {args.data}")
    print(f"Images: {width}×{height} | window: {window_ms} ms | sampling: {sample_rate:,} Hz")
    for task in TASKS:
        print(f"  {task:5} {sample_counts[task]:>7,} samples across {block_counts[task]} blocks")

    with np.load(args.data / block["path"]) as raw:
        trials_per_sample = np.diff(raw["group_ptr"])
        spikes_per_trial = np.diff(raw["trial_ptr"])
        unit_index = raw["unit_index"]
        spike_samples = raw["spike_samples"]
        n_trials = len(spikes_per_trial)

        print(f"\nBlock: {block['path']}")
        print(f"Samples: {len(trials_per_sample):,} | trials: {n_trials:,} | units: {block['n_units']}")
        print(f"Trials per sample (min–max): {trials_per_sample.min()}–{trials_per_sample.max()}")
        if "target" in raw:
            glyphs, repeats = np.unique(raw["target"], return_counts=True)
            print(f"Labeled glyphs: {len(glyphs):,} | samples per glyph: {repeats.min()}–{repeats.max()}")
        print(f"Total spikes: {len(spike_samples):,} | mean per trial: {spikes_per_trial.mean():.2f}")
        print(f"Active units: {len(np.unique(unit_index))}/{block['n_units']}")

        edges_ms = np.append(np.arange(0, window_ms, BIN_MS), window_ms)
        spike_counts = np.histogram(spike_samples, bins=edges_ms * sample_rate / 1000)[0]
        print(f"Population PSTH (spikes per trial, {BIN_MS} ms bins):")
        for start, end, count in zip(edges_ms[:-1], edges_ms[1:], spike_counts):
            print(f"  {start:3}–{end:3} ms  {count / n_trials:7.2f}")


if __name__ == "__main__":
    main()
