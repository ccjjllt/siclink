"""Test whether visible glyph properties predict repeatable image-to-spike model errors."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.functional import normalize

from glyph_contour_retrieval import contour_retrieval_scores
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_gallery_validation import align_training_glyphs, font_scores, image_descriptors, standardized
from kaggle_task1_contour_update import ATLAS_WEIGHT, CONTOUR_WEIGHT
from kaggle_v8_metric_validation import SESSIONS, fit_v8, rendered_bank, summarize


RIDGE = 2.0
CORRECTION_SCALE = 0.50
ROOT = Path("/kaggle/input")


def portions(length, seed, calibration_glyphs):
    order = torch.as_tensor(np.random.default_rng(seed).permutation(length))
    return order[320 + calibration_glyphs:], order[320:320 + calibration_glyphs], order[:320]


def visual_kernel(left, right):
    cosine = left @ right.T
    return 0.5 * (cosine + cosine.pow(3))


def remove_means(kernel, column_mean):
    return kernel - column_mean - kernel.mean(dim=1, keepdim=True) + column_mean.mean()


@torch.no_grad()
def fit_residuals(model, calibration, aligned, seed):
    """Always calibrate on labeled glyphs separate from the base image-to-spike map."""
    descriptors = normalize(image_descriptors((aligned >= 128).to(torch.uint8) * 255), dim=1)
    kernel = visual_kernel(descriptors, descriptors)
    center = kernel.mean(dim=0)
    centered = remove_means(kernel, center)
    centered.diagonal().add_(RIDGE)
    bias, residual = {}, []
    for session in SESSIONS:
        counts = calibration[session].mean(dim=1)
        rates = model.encoder.expected_counts(aligned, session)
        error = (counts + 0.1).sqrt() - (rates + 0.1).sqrt()
        bias[session] = error.mean(dim=0)
        residual.append(error - bias[session])
    shared = torch.stack(residual).mean(dim=0)
    order = torch.as_tensor(np.random.default_rng(seed + 5).permutation(len(shared)))
    factor = torch.linalg.cholesky(centered)
    fitted = torch.cholesky_solve(shared, factor)
    shuffled = torch.cholesky_solve(shared[order], factor)
    return descriptors, center, fitted, shuffled, bias


def correction_maps(predictors, target):
    descriptors, center, fitted, shuffled, _ = predictors
    conditional, permuted = [], []
    for rows in normalize(target, dim=1).split(512):
        kernel = remove_means(visual_kernel(rows, descriptors), center)
        conditional.append(kernel @ fitted)
        permuted.append(kernel @ shuffled)
    return torch.cat(conditional), torch.cat(permuted)


def adjust_rates(rates, offset):
    original = (rates + 0.1).sqrt()
    bounded = offset.clamp(-0.50 * original, 0.50 * original)
    return ((original + bounded).square() - 0.1).clamp_min(1e-5)


def count_mean_quality(counts, rates):
    """Poisson loss on paired held true images, omitting a common factorial term."""
    y = counts.mean(dim=1).unflatten(-1, (-1, 10))[..., 2:]
    mean = rates.unflatten(-1, (-1, 10))[..., 2:].clamp_min(1e-5)
    return float((mean - y * mean.log()).flatten(1).sum(dim=1).mean())


def closed_quality(counts, rates):
    n = len(counts)
    target = torch.arange(n)
    scores = count_template_logits(counts, rates, "timed")
    ranked = scores.topk(16, dim=1).indices
    matches = ranked == target[:, None]
    return {"mean_count_nll": count_mean_quality(counts, rates),
            "top1": float(matches[:, 0].float().mean()),
            "top5": float(matches[:, :5].any(dim=1).float().mean()),
            "top16": float(matches.any(1).float().mean())}


def variants(base, bias, inferred, shuffled):
    return {"base": base, "bias_only": adjust_rates(base, bias),
            "conditional": adjust_rates(base, bias + CORRECTION_SCALE * inferred),
            "shuffled": adjust_rates(base, bias + CORRECTION_SCALE * shuffled)}


@torch.no_grad()
def score_session(model, query, aligned, bias, maps, closed_maps, session, rank_rows, exact, oracle_rows):
    closed_rates = model.encoder.expected_counts(aligned, session)
    closed = {name: closed_quality(query, rates) for name, rates in variants(closed_rates, bias, *closed_maps).items()}

    query = query[rank_rows]
    features, _ = standardize(feature_view(query, "evoked-total"), model.normalization[session])
    direct = model.decoder.predict_variants(session, features.mean(dim=1))
    directional = standardized(contour_retrieval_scores(model.contour, query, session)["contour_ridge"])
    atlas_rates = model.encoder.expected_counts(model.atlas.images, session, batch_size=192)
    if not atlas_rates.shape == maps[0].shape == maps[1].shape:
        raise ValueError("The visual residual maps must use the same atlas and spike features")

    proposals, measurements = {}, {}
    for name, rates in variants(atlas_rates, bias, *maps).items():
        anchor = standardized(font_scores(model.atlas, rates, query, direct)["font_fused"])
        blended = ATLAS_WEIGHT * anchor + CONTOUR_WEIGHT * directional
        score = model.prior.variants(blended)["usage_unseen"]
        proposals[name] = score.topk(12, dim=1).indices
        measurements[name] = exact.evaluate(proposals[name][:, :1])[:, 0]
    entries = {name: summarize([values], [measurements["base"]]) for name, values in measurements.items()}
    for name in entries:
        entries[name]["new_selected_glyphs"] = int((proposals[name][:, 0] != proposals["base"][:, 0]).sum())
    upper = {}
    for name in ("base", "bias_only", "conditional"):
        subset = proposals[name][:oracle_rows]
        # PairedOCF expects one reference per row; reuse the full first column for unused rows.
        window = proposals[name][:, :1].expand(-1, 12).clone()
        window[:oracle_rows] = subset
        upper[name] = exact.evaluate(window)[:oracle_rows].max(axis=1)
    oracle = {name: summarize([values], [upper["base"]]) for name, values in upper.items()}
    return measurements, {"closed_neural": closed, "paired_open_ocf": entries, "oracle_open_top12": oracle}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=4723)
    parser.add_argument("--model-seed", type=int, default=211)
    parser.add_argument("--calibration-glyphs", type=int, default=384)
    parser.add_argument("--ocf-images", type=int, default=64)
    parser.add_argument("--oracle-images", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_forward_residual.json"))
    args = parser.parse_args()
    if not 2 <= args.oracle_images <= args.ocf_images <= 320 or args.calibration_glyphs < 2:
        parser.error("Choose valid nonoverlapping calibration and outer sample sizes.")
    if not ROOT.is_dir():
        raise RuntimeError("Run this font and encoder comparison on Kaggle, not locally.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    training, calibration, held = portions(len(images), args.seed, args.calibration_glyphs)
    if len(training) < 512:
        parser.error("Reserve at least 512 glyphs to fit the main image-to-spike model.")
    groups = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    fitted = {session: counts[training] for session, counts in groups.items()}
    model = fit_v8(fitted, images[training], args.model_seed)
    del fitted
    calibration_counts = {session: counts[calibration] for session, counts in groups.items()}
    aligned, alignment = align_training_glyphs(model.atlas, images[calibration])
    predictor = fit_residuals(model, calibration_counts, aligned, args.seed)
    maps = correction_maps(predictor, model.atlas.contours)
    aligned_held, _ = align_training_glyphs(model.atlas, images[held])
    held_shapes = image_descriptors((aligned_held >= 128).to(torch.uint8) * 255)
    closed_maps = correction_maps(predictor, held_shapes)
    rank_rows = torch.as_tensor(np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_images, replace=False))
    references = (images[held[rank_rows]] >= 128).numpy()
    scorer = PairedOCF(official_scorer(root), rendered_bank(model), references, device, 12)
    print(f"Fit on {len(training)} glyphs, calibrate model residuals on {len(calibration)}, evaluate {len(held)}", flush=True)

    output = {"device": str(device), "mouse": "mice1", "seed": args.seed, "model_seed": args.model_seed,
              "fit_glyphs": len(training), "calibration_glyphs": len(calibration), "held_glyphs": len(held),
              "atlas_images": len(model.atlas.images), "paired_ocf_images_per_session": args.ocf_images,
              "oracle_images_per_session": args.oracle_images, "calibration_alignment": alignment,
              "conditional_kernel_ridge": RIDGE, "conditional_weight": CORRECTION_SCALE,
              "protocol": "Freeze the same CPU-refit v8 ranking and rendering for every open-atlas arm. "
                          "Fit an image-conditional residual against measured spikes on separate labeled calibration glyphs. "
                          "Use held reference pictures only for closed bank diagnostics and exact OCF after open-atlas selection. "
                          "The shuffled control preserves per-session mean bias and mispairs only the conditional residuals. "
                          "All evaluation is on labeled training data with four repeats; the v11 exact OCF ranker is unused.",
              "sessions": {}}
    scores = {}
    for session in SESSIONS:
        measured, block = score_session(model, groups[session][held], aligned_held, predictor[-1][session], maps,
                                        closed_maps, session, rank_rows, scorer, args.oracle_images)
        output["sessions"][session] = block
        for name, values in measured.items():
            scores.setdefault(name, []).append(values)
        output["paired_open_ocf"] = {name: summarize(values, scores["base"]) for name, values in scores.items()}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, **block}), flush=True)


if __name__ == "__main__":
    main()
