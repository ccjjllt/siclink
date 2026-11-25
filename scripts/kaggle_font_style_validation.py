"""Evaluate a train-only shared rendering correction with paired held-image OCF."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import load_images, locate_data, split_glyphs
from glyph_font_oracle import (FONT_SOURCES, aggregate, descriptor, fit_font_transform,
                               render_font, shortlist, thumbnails, warp)
from glyph_font_style import fit_font_style
from kaggle_diagnostics import official_scorer


def coarse_reference(images):
    return thumbnails((images >= 128).to(torch.uint8) * 255)


def raster_candidates(font, transform, candidate_indices, device):
    raster = []
    for indices in candidate_indices.flatten().split(256):
        pixels = font[indices, None].to(device=device, dtype=torch.float32) / 255
        raster.append(warp(pixels, transform))
    return torch.cat(raster).reshape(*candidate_indices.shape, 1, 64, 64)


@torch.no_grad()
def pair_training_images(font, bank, transform, training, device):
    coarse = descriptor(coarse_reference(training))
    aliases, indices = shortlist(bank, transform, coarse, top_k=4)
    candidate = raster_candidates(font, transform, indices, device)
    reference = (training[:, None].to(device) >= 128).float()
    binary = candidate >= 0.5
    intersection = (binary & reference[:, None].bool()).flatten(2).sum(dim=2)
    union = binary.flatten(2).sum(dim=2) + reference.flatten(1).sum(dim=1, keepdim=True) - intersection
    overlap = intersection / union.clamp_min(1)
    confidence, best = (0.75 * aliases.to(device) + 0.25 * overlap).max(dim=1)
    pairs = candidate[torch.arange(len(candidate), device=device), best]
    confidence = ((confidence - 0.60) / 0.06).sigmoid().square()
    return pairs, reference, confidence


def exact_scores(scorer, masks, targets, device):
    batch, count, _, height, width = masks.shape
    predictions = (masks >= 0.5).reshape(-1, height, width).float().cpu().numpy()
    references = np.repeat(targets, count, axis=0)
    results = scorer(predictions, references, device=str(device), batch_size=4)
    return np.asarray(results).reshape(batch, count)


def summarize(scores, baseline):
    best = scores.max(axis=1)
    delta = best - baseline.max(axis=1)
    return {"top_approximate": aggregate(scores[:, 0]), "top_k_oracle": aggregate(best),
            "paired_top_k_delta": float(delta.mean()),
            "paired_top_k_se": float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None,
            "paired_top_1_delta": float((scores[:, 0] - baseline[:, 0]).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--held-glyphs", type=int, default=320)
    parser.add_argument("--evaluation-glyphs", type=int, default=32)
    parser.add_argument("--fit-glyphs", type=int, default=128)
    parser.add_argument("--style-fit-glyphs", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ablate", action="store_true", help="score the geometry-only and stroke-only variants")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/font_style_validation.json"))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this font calibration experiment in a Kaggle Notebook.")
    started = time.monotonic()
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.set_num_threads(4)
    root = locate_data()
    scorer = official_scorer(root)
    images = load_images(root, torch.device("cpu"))
    if not 0 < args.evaluation_glyphs <= args.held_glyphs < len(images):
        parser.error("Choose a valid set of outer held glyphs and a smaller evaluation sample.")
    if (not 0 < max(args.fit_glyphs, args.style_fit_glyphs) <= len(images) - args.held_glyphs
            or min(args.fit_glyphs, args.style_fit_glyphs, args.top_k, args.steps, args.batch_size) < 1):
        parser.error("Choose positive calibration settings and keep calibration within the fit split.")
    fit, held = split_glyphs(len(images), args.held_glyphs, args.seed, torch.device("cpu"))
    # Match the original font oracle's fit and held draws for paired comparison.
    rng = np.random.default_rng(args.seed + 1)
    base_fit = fit[torch.as_tensor(rng.choice(len(fit), args.fit_glyphs, replace=False))]
    queries = held[torch.as_tensor(rng.choice(len(held), args.evaluation_glyphs, replace=False))]
    refine_rows = np.random.default_rng(args.seed + 2).choice(len(fit), args.style_fit_glyphs, replace=False)
    style_fit = fit[torch.as_tensor(refine_rows)]

    font, provenance = render_font(*FONT_SOURCES[0])
    bank = thumbnails(font)
    transform, proxy = fit_font_transform(bank, coarse_reference(images[base_fit]))
    print(f"Fit base font transform over {len(font):,} candidates", flush=True)
    source, target, confidence = pair_training_images(font, bank, transform, images[style_fit], device)
    style, fit_summary = fit_font_style(source, target, confidence, steps=args.steps,
                                       batch_size=args.batch_size, seed=args.seed)
    output = {"device": str(device), "seed": args.seed, "held_glyphs": len(held),
              "evaluation_glyphs": len(queries), "top_k": args.top_k, "asset": provenance,
              "global_transform": asdict(transform), "global_similarity": proxy,
              "style_parameters": asdict(style), "style_fit": fit_summary}
    print(json.dumps({"style": asdict(style), "fit": fit_summary}), flush=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    del source, target, confidence

    # Both renderers get identical candidates. The held reference can define
    # an oracle shortlist here because these scores measure shape coverage.
    held_images = images[queries]
    _, indices = shortlist(bank, transform, descriptor(coarse_reference(held_images)), args.top_k)
    base = raster_candidates(font, transform, indices, device)
    references = (held_images >= 128).numpy()
    baseline = exact_scores(scorer, base, references, device)
    variants = {"shared_style": style}
    if args.ablate:
        variants["geometry_only"] = replace(style, stroke_adjustment=0)
        variants["stroke_only"] = replace(style, scale_x=1, scale_y=1, shift_x_pixels=0,
                                          shift_y_pixels=0, rotation_degrees=0)
    scores = {"original": summarize(baseline, baseline)}
    for name, variant in variants.items():
        corrected = variant.apply(base.flatten(0, 1)).reshape_as(base)
        values = exact_scores(scorer, corrected, references, device)
        scores[name] = summarize(values, baseline)
    output["held"] = scores
    output["seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
