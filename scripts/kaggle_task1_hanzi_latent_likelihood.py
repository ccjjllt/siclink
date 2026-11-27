"""Test calibrated likelihoods of frozen glyph features on new labeled training holdouts."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_bipartite_encoding import timing_view
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import build_font_atlas, standardized
from kaggle_task1_hanzi_peer_links import fresh_cohort as link_holdout
from kaggle_task1_mice2_hanzi_recheck import fresh_cohort as mice2_hanzi_holdout
from kaggle_task1_microns_hanzi_recheck import fresh_cohort as mice1_hanzi_holdout
from kaggle_task1_native_hanzi import (OCR_CHECKSUM, OCR_REVISION, OCR_SOURCE, SEED as OCR_SEED,
                                     feature_extractors, fit_response_ridge, split_glyphs as pilot_holdout)
from kaggle_task1_native_hanzi_open_atlas import (exact_summary, image_coordinates, independent_split,
                                                recalls, visual_proxy)


MOUSE = "mice1"
SEED = 223019
HELD = 320
CALIBRATION = 384
OCF_SAMPLE = 96
DEPTHS = 64
ARMS = ("cosine", "calibrated_isotropic", "diagonal_likelihood", "shrinkage_likelihood")
OUTPUT = Path("/kaggle/working/task1_hanzi_latent_likelihood.json")


def previous_cohorts(size):
    return {"ocr_pilot": pilot_holdout(size, np.random.default_rng(OCR_SEED))[1],
            "ocr_open_atlas": independent_split(size)[1], "mice1_v16_ocr": mice1_hanzi_holdout(size)[1],
            "mice2_v18_ocr": mice2_hanzi_holdout(size)[1], "ocr_cross_day_links": link_holdout(size)[1]}


def split(size, rng, earlier):
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(tuple(earlier.values()))))
    held = np.sort(rng.choice(eligible, HELD, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    calibration = torch.tensor(np.sort(rng.choice(len(fit), CALIBRATION, replace=False)))
    training = torch.tensor(np.setdiff1d(np.arange(len(fit)), calibration.numpy()))
    return (torch.tensor(fit), torch.tensor(held), training, calibration,
            {name: int(np.intersect1d(held, cohort).size) for name, cohort in earlier.items()})


def calibrated_templates(coordinates, atlas, repeats):
    """Estimate feature-space attenuation separately from the stochastic readout error."""
    average = repeats.mean(1)
    centered = coordinates - coordinates.mean(0)
    slope = ((centered * (average - average.mean(0))).sum(0) /
             centered.square().sum(0).clamp_min(1e-6)).clamp_min(0)
    templates = (atlas.T - coordinates.mean(0)) * slope + average.mean(0)
    residual = average - (centered * slope + average.mean(0))
    dispersed = (repeats - average[:, None]).flatten(0, 1)
    covariance = residual.T @ residual / (len(repeats) - 1)
    within = dispersed.T @ dispersed / (len(repeats) * (repeats.shape[1] - 1)) / repeats.shape[1]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance - within)
    model_error = (eigenvectors * eigenvalues.clamp_min(0)) @ eigenvectors.T
    total = model_error + within
    variance = total.diagonal()
    floor = 0.15 * variance.mean()
    shrinkage = 0.5 * total + 0.5 * torch.diag(variance)
    shrinkage.diagonal().add_(floor)
    return templates, variance + floor, shrinkage, {"median_attenuation": float(slope.median()),
        "estimated_repeat_fraction_of_error": float(within.trace() / total.trace()), "variance_floor": float(floor)}


def dot_likelihood(query, templates):
    return query @ templates.T - 0.5 * templates.square().sum(1)


def score_day(counts, fit, held, training, calibration, image_codes):
    response = timing_view(counts.unflatten(-1, (-1, 10)))
    normalizer, center, scale = fit_response_ridge(response[fit[training]].mean(1))
    target = (response[held].mean(1) - center) / scale
    contour, contour_atlas = image_codes["contour"]
    shape = standardized(F.normalize(target @ (normalizer @ contour[training])) @ F.normalize(contour_atlas, dim=0))
    coordinates, atlas = image_codes["frozen_hanzi"]
    projection = normalizer @ coordinates[training]
    query = target @ projection
    repeats = ((response[fit[calibration]] - center) / scale) @ projection
    templates, diagonal, covariance, diagnostics = calibrated_templates(coordinates[calibration], atlas, repeats)
    root = torch.linalg.cholesky(covariance)
    whitened_query = torch.linalg.solve_triangular(root, query.T, upper=False).T
    whitened_templates = torch.linalg.solve_triangular(root, templates.T, upper=False).T
    variance = diagonal.sqrt()
    latent = (F.normalize(query) @ F.normalize(atlas, dim=0), dot_likelihood(query, templates),
              dot_likelihood(query / variance, templates / variance), dot_likelihood(whitened_query, whitened_templates))
    scores = {name: standardized(0.5 * shape + 0.5 * standardized(values)) for name, values in zip(ARMS, latent, strict=True)}
    return scores, diagnostics


def report_stage(report, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication", action="store_true", help="Repeat on mice2 with a disjoint glyph cohort")
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir() or not OUTPUT.parent.is_dir():
        raise RuntimeError("Run the held comparison in a Kaggle CPU Notebook.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cpu")
    mouse, seed, sample_count = ("mice2", 223071, 160) if args.replication else (MOUSE, SEED, OCF_SAMPLE)
    rng = np.random.default_rng(seed)
    root = locate_data()
    images, catalog = load_images(root, device), load_catalog(root)
    earlier = previous_cohorts(len(images))
    if args.replication:
        earlier["mice1_latent_likelihood"] = split(len(images), np.random.default_rng(SEED), earlier)[1]
    fit, held, training, calibration, overlap = split(len(images), rng, earlier)
    atlas, provenance = build_font_atlas(images[fit[training[rng.choice(len(training), 128, replace=False)]]], device)
    if len(atlas.images) != 27_550:
        raise ValueError("Expected the shared public Noto atlas.")
    image_codes = image_coordinates(images[fit], atlas.images, {"frozen_hanzi": feature_extractors()["frozen_hanzi"]},
                                    normalize_atlas=False)
    report = {"mouse": mouse, "seed": seed, "fitting_glyphs": len(training), "calibration_glyphs": len(calibration),
              "held_glyphs": len(held), "overlap_by_previous_cohort": overlap, "atlas": provenance,
              "ocr": {"source": OCR_SOURCE, "revision": OCR_REVISION, "sha256": OCR_CHECKSUM}, "recordings": {},
              "protocol": "All arms share the renderer, atlas, and 50% standardized contour branch. The other 50% is OCR "
                          "cosine or a Gaussian likelihood in the identical 128-axis frozen OCR coordinates. Calibrate one "
                          "attenuation per axis and uncertainty on different labeled glyphs from the neural ridge fit. "
                          "Subtract repeat noise from four-repeat residual covariance, project the result to positive semidefinite, "
                          "then restore four-repeat noise. Full covariance shrinks 50% toward its diagonal. Exclude held images "
                          "from all fitting. Pooling calibrated scores over independently observed days uses known identity and "
                          "is only a 16-trial proxy. No formal inputs and no v19/v20 reranker."}
    report_stage(report, started, "features_ready")
    groups = grouped_blocks(root, catalog, mouse, device, len(images), bin_ms=25)
    top, summed = {}, {}
    for day, counts in groups.items():
        scores, report["recordings"][day] = score_day(counts, fit, held, training, calibration, image_codes)
        top[day] = {arm: value.topk(DEPTHS, dim=1).indices for arm, value in scores.items()}
        for arm, value in scores.items():
            if arm in summed:
                summed[arm].add_(value)
            else:
                summed[arm] = value
        report_stage(report, started, day + "_ranked")
    top["pooled_16"] = {arm: value.topk(DEPTHS, dim=1).indices for arm, value in summed.items()}
    references = images[held]
    proxy, confident, similarity = visual_proxy(atlas, references)
    report.update(top_k=recalls(top, proxy, confident), trusted_glyphs=int(confident.sum()), proxy_similarity=similarity)
    report_stage(report, started, "rank_metrics")

    sampled = torch.tensor(rng.choice(len(held), sample_count, replace=False))
    positions = torch.arange(sample_count)
    metric = PairedOCF(official_scorer(root), atlas.images, (references[sampled] >= 128).numpy(), device, 12)
    report["exact_ocf"] = {}
    for condition in ("single_day_4", "pooled_16"):
        measured = {}
        for arm in ARMS:
            if condition == "pooled_16":
                picks = top["pooled_16"][arm][sampled, 0]
            else:
                by_day = torch.stack([top[day][arm][sampled, 0] for day in groups])
                picks = by_day[positions // (sample_count // len(groups)), positions]
            measured[arm] = metric.evaluate(picks[:, None])[:, 0]
        report["exact_ocf"][condition] = {arm: exact_summary(value, measured["cosine"]) for arm, value in measured.items()}
        report["unique_exact_pairs"] = len(metric.values)
        report_stage(report, started, condition + "_scored")
    report_stage(report, started, "complete")


if __name__ == "__main__":
    main()
