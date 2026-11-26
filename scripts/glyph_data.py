"""Load SiClink spike counts and construct labeled research splits."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


COMPETITION = "siclink-mouse-visual-reconstruction-bci-challenge"
DATA_DIR = "siclink_chinese_character_decoding_dataset"
BIN_MS = 25


@dataclass
class Block:
    session: str
    ids: np.ndarray
    counts: torch.Tensor
    labels: torch.Tensor | None


def locate_data(input_root=Path("/kaggle/input")):
    candidates = (input_root / "competitions" / COMPETITION / DATA_DIR,
                  input_root / COMPETITION / DATA_DIR, input_root / DATA_DIR)
    for path in candidates:
        if (path / "catalog.json").is_file():
            return path
    for path in input_root.rglob("catalog.json"):
        if path.parent.name == DATA_DIR:
            return path.parent
    raise FileNotFoundError(f"SiClink competition data is missing under {input_root}")


def load_catalog(root):
    return json.loads((root / "catalog.json").read_text(encoding="utf-8"))


def load_images(root, device):
    with np.load(root / "train" / "pictures.npz") as stored:
        return torch.as_tensor(stored["images"], device=device)


def read_trials(root, catalog, entry, device, *, bin_ms=BIN_MS, load_labels=True):
    """Return raw count features of shape (samples, repeats, units × bins)."""
    bins, excess = divmod(catalog["window_ms"], bin_ms)
    samples_per_bin, fractional = divmod(catalog["sample_rate_hz"] * bin_ms, 1000)
    if excess or fractional:
        raise ValueError("Bin width must divide the window and align with the sample rate")
    units = entry["n_units"]
    with np.load(root / entry["path"]) as raw:
        ids = raw["sample_ids"]
        group_ptr = raw["group_ptr"]
        trial_ptr = raw["trial_ptr"]
        labels = raw["target"] if load_labels and "target" in raw else None
        trial_sizes = np.diff(trial_ptr)
        trial_ids = np.repeat(np.arange(len(trial_sizes)), trial_sizes)
        time_bins = raw["spike_samples"] // samples_per_bin
        indices = (trial_ids * units + raw["unit_index"]) * bins + time_bins
        counts = np.bincount(indices, minlength=len(trial_sizes) * units * bins)

    repeats = np.diff(group_ptr)
    if not np.all(repeats == repeats[0]):
        raise ValueError(f"Uneven repeat counts in {entry['path']}")
    counts = torch.from_numpy(counts).to(device=device, dtype=torch.float32)
    counts = counts.reshape(len(ids), int(repeats[0]), units * bins)
    if labels is not None:
        labels = torch.as_tensor(labels, device=device, dtype=torch.long)
    return Block(entry["session"], ids, counts, labels)


def training_groups(block, n_glyphs):
    """Collect labeled training trials by glyph without using the sample IDs."""
    labels = block.labels
    repeats = torch.bincount(labels, minlength=n_glyphs)
    if not bool((repeats == repeats[0]).all()) or int(repeats[0]) == 0:
        raise ValueError(f"Expected equal labeled repeats per glyph in {block.session}")
    order = torch.argsort(labels, stable=True)
    return block.counts[order].reshape(n_glyphs, -1, block.counts.shape[-1])


def grouped_blocks(root, catalog, mouse, device, n_glyphs, *, bin_ms=BIN_MS):
    entries = sorted((e for e in catalog["blocks"] if e["mouse"] == mouse and e["task"] == "train"),
                     key=lambda e: e["session"])
    groups = {}
    for entry in entries:
        block = read_trials(root, catalog, entry, device, bin_ms=bin_ms)
        groups[block.session] = training_groups(block, n_glyphs)
    return groups


def split_glyphs(n_glyphs, count, seed, device):
    random = np.random.default_rng(seed)
    selected = random.permutation(n_glyphs)
    training = torch.as_tensor(np.sort(selected[count:]), device=device)
    heldout = torch.as_tensor(np.sort(selected[:count]), device=device)
    return training, heldout


def split_repeats(counts, seed):
    """Choose a different reproducible held repeat for each labeled glyph."""
    n_glyphs, repeats, features = counts.shape
    random = np.random.default_rng(seed)
    held = random.integers(0, repeats, size=n_glyphs)
    order = np.argsort(held[:, None] == np.arange(repeats)[None], axis=1, kind="stable")
    order = torch.as_tensor(order, device=counts.device)
    arranged = counts.gather(1, order[:, :, None].expand(-1, -1, features))
    return arranged[:, :-1], arranged[:, -1:]


def as_trial_blocks(groups):
    """Flatten grouped data for supervised models while retaining session labels."""
    blocks = []
    for session, counts in groups.items():
        labels = torch.arange(len(counts), device=counts.device).repeat_interleave(counts.shape[1])
        blocks.append((session, counts.flatten(0, 1), labels))
    return blocks
