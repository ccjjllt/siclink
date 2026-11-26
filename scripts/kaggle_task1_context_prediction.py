"""Probe whether surround-predicted orientation mismatch improves held neural likelihood."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import _deviance, count_template_logits
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standard_error
from kaggle_task1_microns_spatial_readout import new_whole_glyphs


MOUSE = "mice1"
SEED = 96821
HELD_COUNT = 320
EXACT_COUNT = 64
GRID = 8
PATCH = 8
RIDGE = 0.12
OUTPUT = Path("/kaggle/working/task1_context_prediction.json")
FREQUENCIES = ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1),
               (2, 2), (0, 3), (3, 0), (1, 3), (3, 1))
SURROUND = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
            (-2, 0), (0, -2), (0, 2), (2, 0))
TERMS = ("additive", "separate_energies", "shuffled_mismatch", "predictive_mismatch")


def image_features(images):
    """Signed DCT features from nonoverlapping cells, with surrounding pixels disjoint from the center."""
    glyphs = images.float() / 255
    positions = (torch.arange(PATCH, dtype=glyphs.dtype) + 0.5) / PATCH
    frequencies = torch.as_tensor(FREQUENCIES, dtype=glyphs.dtype)
    vertical = (torch.pi * frequencies[:, 0, None] * positions).cos()
    horizontal = (torch.pi * frequencies[:, 1, None] * positions).cos()
    basis = (vertical[:, :, None] * horizontal[:, None, :]).flatten(1)
    basis = F.normalize(basis, dim=1)
    patches = F.unfold(glyphs[:, None], kernel_size=PATCH, stride=PATCH).transpose(1, 2)
    actual = (patches @ basis.T).transpose(0, 1).contiguous()
    map_grid = actual[:, :, :6].reshape(GRID, GRID, len(images), 6)
    map_grid = F.pad(map_grid.permute(2, 3, 0, 1), (2, 2, 2, 2))
    ring = torch.cat([map_grid[:, :, 2 + row:2 + row + GRID, 2 + col:2 + col + GRID]
                      for row, col in SURROUND], dim=1).flatten(2).permute(2, 0, 1).contiguous()
    return actual, ring, glyphs.mean((1, 2))


def fit_indices(size):
    _, previous, _ = new_whole_glyphs(size, torch.device("cpu"))
    available = np.setdiff1d(np.arange(size), previous.numpy())
    rng = np.random.default_rng(SEED)
    held = np.sort(rng.choice(available, HELD_COUNT, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    return torch.from_numpy(fit), torch.from_numpy(held)


def fit_centers(actual, ink, observed, order):
    """Fix cellular locations by split-glyph agreement, partialing out total foreground ink."""
    evoked = (observed[..., 2:].sum(-1) + 0.1).sqrt()
    center = evoked.mean(dim=1, keepdim=True)
    target = ((evoked - center) / evoked.std(dim=1, keepdim=True).clamp_min(0.1)).mean(0)
    agreements = []
    for part in order.tensor_split(2):
        visual = actual[:, part].transpose(0, 1).flatten(1)
        neural = target[part]
        ink_axis = F.normalize(ink[part] - ink[part].mean(), dim=0)
        normalized = []
        for values in (visual, neural):
            values = values - values.mean(0)
            values = values - ink_axis[:, None] * (ink_axis @ values)[None]
            normalized.append(F.normalize(values, dim=0))
        agreements.append(normalized[1].T @ normalized[0])
    reliable = (agreements[0] * agreements[1]).reshape(-1, GRID * GRID, actual.shape[-1]).mean(-1)
    smooth = F.avg_pool2d(reliable.reshape(-1, 1, GRID, GRID), 3, stride=1, padding=1, count_include_pad=False)
    return smooth.flatten(1).argmax(1)


def ridge_rates(features, target, evaluate):
    """Batched or per-field ridge with fit-only normalization and an unpenalized intercept."""
    location = features.mean(dim=-2, keepdim=True)
    scale = features.std(dim=-2, keepdim=True).clamp_min(1e-4)
    design = (features - location) / scale
    intercept = target.mean(dim=-2, keepdim=True)
    gram = design.transpose(-2, -1) @ design
    gram.diagonal(dim1=-2, dim2=-1).add_(RIDGE * design.shape[-2])
    weights = torch.cholesky_solve(design.transpose(-2, -1) @ (target - intercept), torch.linalg.cholesky(gram))
    return ((evaluate - location) / scale) @ weights + intercept


def center_expectation(actual, ring, fit, held, order):
    first, second = order.tensor_split(2)
    predicted = torch.empty_like(actual)
    held_predictions = []
    for train, calibration in ((first, second), (second, first)):
        use, unseen = fit[train], torch.cat((fit[calibration], held))
        estimates = ridge_rates(ring[:, use], actual[:, use], ring[:, unseen])
        predicted[:, fit[calibration]] = estimates[:, :len(calibration)]
        held_predictions.append(estimates[:, len(calibration):])
    predicted[:, held] = torch.stack(held_predictions).mean(0)
    return predicted


def reshuffle_expectation(actual, ring, predicted, fit, held, occupied, rng):
    """Derange only within matching center-ink and mean surround-ink strata."""
    shuffled = predicted.clone()
    changed = 0
    for site in occupied.tolist():
        covariates = torch.stack((actual[site, :, 0], ring[site, :, ::6].mean(1)), dim=1)
        cuts = torch.quantile(covariates[fit], torch.tensor((0.25, 0.50, 0.75)), dim=0)
        bins = [torch.bucketize(covariates[:, index].contiguous(), cuts[:, index].contiguous()) for index in range(2)]
        strata = 4 * bins[0] + bins[1]
        for pool in (fit, held):
            for value in strata[pool].unique():
                close = pool[strata[pool] == value]
                if len(close) > 1:
                    rows = close[torch.from_numpy(rng.permutation(len(close)))]
                    shuffled[site, rows] = predicted[site, rows.roll(1)]
                    changed += len(rows)
    return shuffled, changed / (len(occupied) * (len(fit) + len(held)))


def predict_spikes(actual, ring, predicted, shuffled, ink, centers, train, fit, held):
    sessions, _, units, bins = train.shape
    outputs = {term: torch.empty(sessions, len(held), units, bins) for term in TERMS}
    target = (train + 0.1).sqrt()
    for site in centers.unique().tolist():
        selected = centers == site
        location = actual[site, fit].mean(0)
        scale = actual[site, fit].std(0).clamp_min(1e-3)
        actual_z = (actual[site] - location) / scale
        expected_z = (predicted[site] - location) / scale
        shuffled_z = (shuffled[site] - location) / scale
        additive = torch.cat((actual[site], ring[site], predicted[site], ink[:, None]), dim=1)
        features = {
            "additive": additive,
            "separate_energies": torch.cat((additive, actual_z.square() + expected_z.square()), dim=1),
            "shuffled_mismatch": torch.cat((additive, (actual_z - shuffled_z).square()), dim=1),
            "predictive_mismatch": torch.cat((additive, (actual_z - expected_z).square()), dim=1),
        }
        y = target[:, :, selected].transpose(0, 1).flatten(1)
        for term, design in features.items():
            estimated = ridge_rates(design[fit], y, design[held])
            estimated = (estimated.clamp_min(0).square() - 0.1).clamp_min(1e-5)
            outputs[term][:, :, selected] = estimated.reshape(len(held), sessions, int(selected.sum()), bins).transpose(0, 1)
    return outputs


def paired(left, right):
    delta = (left - right).reshape(-1, left.shape[-1]).mean(0)
    return {"mean": float(delta.mean()), "se": float(standard_error(delta)), "glyphs": len(delta)}


def evaluate(outputs, queries, images, exact, sample):
    observed = torch.stack([counts.mean(1) for counts in queries.values()]).unflatten(-1, (-1, 10))
    outcome, internals = {}, {}
    for term, rates in outputs.items():
        deviance = _deviance(observed, rates)
        panels = {phase: deviance[..., start:end].sum((-1, -2)).numpy()
                  for phase, start, end in (("early", 2, 4), ("late", 4, 10), ("evoked", 2, 10))}
        logits = torch.stack([count_template_logits(counts, one.flatten(1), "timed")
                              for counts, one in zip(queries.values(), rates)])
        views = dict(zip(queries, logits))
        views["oracle_known_identity_16"] = logits.sum(0)
        retrieval, ocf = {}, []
        identity = torch.arange(len(images))[:, None]
        for name, scores in views.items():
            candidates = scores.topk(16, dim=1).indices
            match = candidates == identity
            result = exact.evaluate(candidates[sample, :1])[:, 0]
            ocf.append(result)
            retrieval[name] = {"exact_ocf": float(result.mean()),
                               **{f"top_{width}": float(match[:, :width].any(1).float().mean()) for width in (1, 5, 16)}}
        outcome[term] = {"deviance_per_glyph_repeat": {phase: float(values.mean()) for phase, values in panels.items()},
                         "deviance_by_day": {phase: values.mean(1).tolist() for phase, values in panels.items()},
                         "known_gallery": retrieval}
        internals[term] = panels, np.stack(ocf)
    treatment, treatment_ocf = internals["predictive_mismatch"]
    contrasts = {}
    for control in TERMS[:-1]:
        panels, ocf = internals[control]
        contrasts[control] = {"deviance_reduction": {name: paired(values, treatment[name]) for name, values in panels.items()},
                              "exact_ocf_4": paired(treatment_ocf[:-1], ocf[:-1]),
                              "exact_ocf_known_identity_16": paired(treatment_ocf[-1], ocf[-1])}
    return outcome, contrasts


@torch.inference_mode()
def main():
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this labeled diagnostic only in a Kaggle Notebook.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    if len(images) != 2748:
        raise ValueError("Expected the full labeled SiClink glyph collection.")
    fit, held = fit_indices(len(images))
    rng = np.random.default_rng(SEED + 1)
    order = torch.from_numpy(rng.permutation(len(fit)))
    catalog = load_catalog(root)
    catalog["blocks"] = [entry for entry in catalog["blocks"] if entry["task"] == "train"]
    days = grouped_blocks(root, catalog, MOUSE, device, len(images))
    if len(days) != 4 or any(day.shape[1] != 4 for day in days.values()):
        raise ValueError("Expected four labeled recordings with four trials per character.")
    queries = {name: values[held] for name, values in days.items()}
    train = torch.stack([values[fit].mean(1) for values in days.values()]).unflatten(-1, (-1, 10))
    del days
    print(f"Fit anatomy and surround predictor on {len(fit)} true training glyphs", flush=True)
    actual, surround, ink = image_features(images)
    centers = fit_centers(actual[:, fit], ink[fit], train, order)
    predicted = center_expectation(actual, surround, fit, held, order)
    shuffled, changed = reshuffle_expectation(actual, surround, predicted, fit, held, centers.unique(), rng)
    observed, expected = actual[centers, :][:, held], predicted[centers, :][:, held]
    null = actual[centers, :][:, fit].mean(dim=1, keepdim=True)
    prediction_r2 = 1 - (observed - expected).square().sum() / (observed - null).square().sum().clamp_min(1e-5)
    outputs = predict_spikes(actual, surround, predicted, shuffled, ink, centers, train, fit, held)
    if any(not torch.isfinite(rates).all() for rates in outputs.values()):
        raise RuntimeError("A readout returned nonfinite firing rates.")
    sample = torch.from_numpy(np.sort(rng.choice(len(held), EXACT_COUNT, replace=False)))
    print("Score held firing rates and the known whole-glyph gallery", flush=True)
    exact = PairedOCF(official_scorer(root), images[held], (images[held[sample]] >= 128).numpy(), device, 12)
    outcome, contrasts = evaluate(outputs, queries, images[held], exact, sample)
    report = {
        "mouse": MOUSE, "seed": SEED, "fit_glyphs": len(fit), "held_glyphs": len(held),
        "held_disjoint_from_previous_microns_spatial": True, "sessions": list(queries),
        "field_sites_used": int(centers.unique().numel()), "cell_locations": centers.tolist(),
        "held_center_prediction_r2": float(prediction_r2), "shuffled_context_fraction": changed,
        "ocf_glyphs": len(sample), "unique_ocf_pairs": len(exact.values), "relative_ridge": RIDGE,
        "separate_energy_algebra": "c²+e² compared with c²+e²−2ce in predictive_mismatch",
        "arms": outcome, "predictive_mismatch_vs": contrasts, "formal_inputs_used": False,
        "seconds": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"readout_contrasts": contrasts, "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
