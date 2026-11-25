"""Compact fast-timing features from the official sparse SiClink spike events."""

from dataclasses import dataclass

import numpy as np
import torch

from glyph_contour_retrieval import _agreement, _fit_readout
from glyph_reconstruction import RidgeReadout
from glyph_retrieval import _moments, _prototypes, _reliable_axes, _standardize, _whiten


@dataclass
class EventBlock:
    coarse: torch.Tensor
    phase: torch.Tensor | None
    event: torch.Tensor | None


def _rapid_features(times, flat_units, count, first_edge, middle_edge, last_edge, fine_width):
    sustained = times >= middle_edge
    windows = 2 * flat_units + sustained
    first = np.full(2 * count, last_edge, dtype=np.int32)
    np.minimum.at(first, windows, times)
    first = first.reshape(count, 2)
    latency = np.exp(-(first - np.array((first_edge, middle_edge))) / (5 * fine_width))
    latency[first == last_edge] = 0

    order = np.lexsort((times, flat_units))
    sorted_windows = windows[order]
    gaps = np.diff(times[order])
    rapid = (sorted_windows[1:] == sorted_windows[:-1]) & (gaps > 0) & (gaps <= fine_width)
    bursts = np.bincount(sorted_windows[1:][rapid], minlength=2 * count).reshape(count, 2)
    return np.stack((latency, np.sqrt(bursts)), axis=-1).reshape(count, 4)


def read_event_block(root, catalog, entry, n_glyphs, views):
    """Read a training session once, retaining no dense 5 ms event raster."""
    fine_width, remainder = divmod(catalog["sample_rate_hz"] * 5, 1000)
    if remainder or catalog["window_ms"] != 250:
        raise ValueError("Fast timing expects a 250 ms window with integral 5 ms sample bins")
    n_units = entry["n_units"]
    with np.load(root / entry["path"]) as raw:
        labels = raw["target"]
        groups = np.diff(raw["group_ptr"])
        trial_sizes = np.diff(raw["trial_ptr"])
        times = raw["spike_samples"].astype(np.int32, copy=False)
        units = raw["unit_index"]
    if len(labels) != len(trial_sizes) or len(groups) != len(labels) or not np.all(groups == 1):
        raise ValueError("Expected one labeled trial per training sample")
    repetitions = np.bincount(labels, minlength=n_glyphs)
    if len(repetitions) != n_glyphs or repetitions.min() == 0 or not np.all(repetitions == repetitions[0]):
        raise ValueError("Expected equal repeat counts for all training glyphs")

    first_edge, middle_edge, last_edge = 10 * fine_width, 25 * fine_width, 50 * fine_width
    valid = (times >= first_edge) & (times < last_edge)
    flat_units = np.repeat(np.arange(len(trial_sizes)), trial_sizes)[valid] * n_units + units[valid]
    times = times[valid]
    bins = (times - first_edge) // fine_width
    coarse_indices = flat_units * 8 + bins // 5
    count = len(trial_sizes) * n_units
    coarse = np.bincount(coarse_indices, minlength=8 * count).reshape(count, 8).astype(np.int16)
    stable_order = np.argsort(labels, kind="stable")
    shape = (n_glyphs, int(repetitions[0]), n_units)

    def grouped(values):
        width = values.shape[-1]
        per_trial = values.reshape(len(trial_sizes), n_units, width)
        return torch.from_numpy(per_trial[stable_order].reshape(*shape, width))

    phase = event = None
    if "phase5" in views:
        # Signed weights sum to zero within every coarse bin, so a flat 5 ms
        # distribution contributes no timing feature regardless of its count.
        moments = np.bincount(coarse_indices, weights=(bins % 5 - 2) * 0.5, minlength=8 * count)
        phase = (moments.reshape(count, 8) / np.sqrt(1 + coarse.astype(np.float32))).astype(np.float16)
        phase = grouped(phase)
    if "event" in views:
        event = _rapid_features(times, flat_units, count, first_edge, middle_edge, last_edge, fine_width)
        event = grouped(event.astype(np.float16))
    return EventBlock(grouped(coarse), phase, event)


def event_features(block, view, device):
    if view not in ("count25", "phase5", "event"):
        raise ValueError(f"Unknown sparse timing view {view}")
    counts = block.coarse.to(device=device, dtype=torch.float32)
    grouped = counts.unflatten(-1, (4, 2)).sum(dim=-1)
    features = torch.cat((counts, grouped, grouped.sum(dim=-1, keepdim=True)), dim=-1).sqrt_()
    if view != "count25":
        added = block.phase if view == "phase5" else block.event
        if added is None:
            raise ValueError(f"Missing sparse timing view {view}")
        features = torch.cat((features, added.to(device=device, dtype=torch.float32)), dim=-1)
    return features.flatten(-2)


@dataclass
class EventContourReadout:
    moments: dict
    projection: torch.Tensor
    signature: torch.Tensor
    energy: torch.Tensor
    pooled: RidgeReadout
    local: dict[str, RidgeReadout]


@torch.no_grad()
def fit_event_contours(training, signature, energy):
    """Apply the same contour ridge to already constructed single-trial views."""
    moments, normalized, centers, covariances = {}, {}, [], []
    for session, values in training.items():
        moments[session] = _moments(values)
        values = _standardize(values, *moments[session])
        normalized[session] = values
        classes, repeats = values.shape[:2]
        labels = torch.arange(classes, device=values.device).repeat_interleave(repeats)
        center, _, covariance = _prototypes(values.flatten(0, 1), labels, classes)
        covariances.append(covariance)
        centers.append(center)

    noise = torch.stack(covariances).mean(dim=0)
    noise.diagonal().add_(0.25)
    factor = torch.linalg.cholesky(noise)
    centers = torch.stack(centers)
    white = _whiten(centers.flatten(0, 1), factor).reshape(centers.shape)
    available = torch.ones(centers.shape[1], dtype=torch.bool, device=centers.device)
    stable = _reliable_axes(white, available)[:, -192:]
    projection = torch.linalg.solve_triangular(factor.T, stable, upper=True)
    projected = {name: values @ projection for name, values in normalized.items()}
    targets = torch.cat((signature, energy.log()[:, None]), dim=1)
    pooled = _fit_readout(tuple(projected.values()), targets, ridge=0.09)
    local = {name: _fit_readout((values,), targets, ridge=0.14) for name, values in projected.items()}
    return EventContourReadout(moments, projection, signature, energy, pooled, local)


@torch.inference_mode()
def event_contour_scores(model, query, session):
    features = _standardize(query, *model.moments[session]).mean(dim=1) @ model.projection
    estimate = 0.80 * model.pooled.predict(features) + 0.20 * model.local[session].predict(features)
    return _agreement(estimate, model)
