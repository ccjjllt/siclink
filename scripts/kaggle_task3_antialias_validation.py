"""Compare shape-preserving glyph antialiasing on held labeled image pairs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from glyph_data import load_images, locate_data
from glyph_mbr import contour_signatures, contour_utility
from kaggle_diagnostics import official_scorer


def trials_for(rows, bank, signature, energy, seed):
    """Construct fixed exact, near, moderately similar, and unrelated pairs."""
    similarity = contour_utility(signature[rows], energy[rows], signature[bank], energy[bank])
    ranked = similarity.argsort(dim=1, descending=True, stable=True).numpy()
    middle = round(0.10 * len(bank))
    unrelated = np.random.default_rng(seed).integers(len(bank) // 2, len(bank), size=len(rows))
    return {"exact": rows, "nearest": bank[ranked[:, 0]], "moderate": bank[ranked[:, middle]],
            "unrelated": bank[ranked[np.arange(len(rows)), unrelated]]}


def rendered(images):
    reference_mask = images >= 128
    hard = reference_mask.astype(np.float32)
    soft = images.astype(np.float32) / 255
    gauss035 = gaussian_filter(hard, sigma=(0, 0.35, 0.35), mode="constant")
    gauss065 = gaussian_filter(hard, sigma=(0, 0.65, 0.65), mode="constant")
    candidates = {"binary": hard, "native_edge25": 0.75 * hard + 0.25 * soft,
                  "gaussian035": 0.25 * hard + 0.75 * gauss035,
                  "gaussian065_edge35": 0.65 * hard + 0.35 * gauss065}
    for name, pixels in candidates.items():
        # Match the PNG quantization and check preservation of every stroke mask.
        pixels = np.rint(255 * pixels).astype(np.uint8)
        if not np.array_equal(pixels >= 128, reference_mask):
            raise ValueError(f"{name} changed the thresholded glyph's stroke support")
        yield name, pixels.astype(np.float32) / 255


def paired(values, baseline):
    delta = values - baseline
    return {"ocf": float(values.mean()), "delta_vs_binary": float(delta.mean()),
            "paired_glyph_se": float(delta.std(ddof=1) / np.sqrt(len(delta))), "n": len(delta)}


def score_fold(images, targets, cases, scorer, report, save):
    masks = images[targets] >= 128
    measurements = {}
    for scenario, choices in cases.items():
        scores = {name: scorer(pixels, masks, device="cpu", batch_size=8)
                  for name, pixels in rendered(images[choices])}
        measurements[scenario] = scores
        report[scenario] = {name: paired(values, scores["binary"]) for name, values in scores.items()}
        save()
        print(json.dumps({"scenario": scenario, "metrics": report[scenario]}), flush=True)
    return measurements


def choose_renderer(calibration, max_exact_loss):
    """Select one fixed render using only paired calibration glyphs."""
    candidate_scores = {}
    for name in calibration["exact"]:
        exact_loss = float((calibration["exact"]["binary"] - calibration["exact"][name]).mean())
        near_gain = calibration["nearest"][name] - calibration["nearest"]["binary"]
        moderate_gain = calibration["moderate"][name] - calibration["moderate"]["binary"]
        candidate_scores[name] = {"exact_loss": exact_loss,
                                  "mean_near_and_moderate_gain": float((0.5 * (near_gain + moderate_gain)).mean())}
    eligible = (name for name, item in candidate_scores.items() if item["exact_loss"] <= max_exact_loss)
    selected = max(eligible, key=lambda name: candidate_scores[name]["mean_near_and_moderate_gain"])
    return selected, candidate_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2179)
    parser.add_argument("--calibration-glyphs", type=int, default=32)
    parser.add_argument("--held-glyphs", type=int, default=48)
    parser.add_argument("--calibration-max-exact-loss", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_antialias_validation.json"))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this image-only comparison on Kaggle CPU.")
    if min(args.calibration_glyphs, args.held_glyphs) < 2 or args.calibration_max_exact_loss < 0:
        parser.error("Choose at least two glyphs per fold and a nonnegative calibration loss limit.")

    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, "cpu")
    score = official_scorer(root)
    order = np.random.default_rng(args.seed).permutation(len(images))
    split = args.calibration_glyphs + args.held_glyphs
    calibration, held, bank = order[:args.calibration_glyphs], order[args.calibration_glyphs:split], order[split:]
    if len(bank) < 20:
        parser.error("Keep at least twenty independently sampled morphology-neighbor gallery glyphs.")
    signature, energy = contour_signatures(images)
    images = images.numpy()
    output = {"seed": args.seed, "calibration_glyphs": len(calibration), "held_glyphs": len(held),
              "matched_gallery_glyphs": len(bank), "calibration_max_exact_loss": args.calibration_max_exact_loss,
              "protocol": "The target sets and shared alternate-glyph gallery are disjoint. Alternate predictions are "
                          "chosen by reference-image similarity to simulate mismatch. Uses no neural prediction, "
                          "competition test data, or held OCF for renderer selection.",
              "calibration": {}, "held": {}}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    calibration_cases = trials_for(calibration, bank, signature, energy, args.seed + 1)
    fit_scores = score_fold(images, calibration, calibration_cases, score, output["calibration"], save)
    selected, selection = choose_renderer(fit_scores, args.calibration_max_exact_loss)
    output.update(selected_renderer=selected, calibration_selection=selection)
    save()
    print(f"Calibration selected {selected}", flush=True)
    held_cases = trials_for(held, bank, signature, energy, args.seed + 2)
    score_fold(images, held, held_cases, score, output["held"], save)
    output["selected_renderer_held"] = {scenario: summary[selected] for scenario, summary in output["held"].items()}
    save()
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
