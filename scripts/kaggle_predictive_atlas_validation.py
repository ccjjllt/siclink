"""Test out-of-fold noise-aware Task 1 glyph scores against candidate v4."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_encoding import fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_predictive_likelihood import fit_predictive_likelihood
from glyph_reconstruction import fit_glyph_reconstructor
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_gallery_validation import (align_training_glyphs, build_font_atlas, font_scores,
                                       image_descriptors, standard_error, standardized)


BASELINE = "candidate_v4"
DEPTHS = (4, 16, 32)


def evaluate(choices, atlas, style, references, score, device, depth_images):
    binary = torch.cat([(style.apply(batch[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                        for batch in atlas.images.split(1024)])
    evaluator = PairedOCF(score, binary, references, device, 12)
    top = {name: evaluator.evaluate(ranked[:, :1])[:, 0] for name, ranked in choices.items()}
    depth = {}
    if depth_images:
        for name, ranked in choices.items():
            exact = evaluator.evaluate(ranked[:depth_images])
            depth[name] = {k: exact[:, :k].max(axis=1) for k in DEPTHS}
    return top, depth


def summarize(vectors, reference):
    scores = np.stack(vectors)
    paired = (scores - np.stack(reference)).mean(axis=0)
    return {"ocf": float(scores.mean()), "n": scores.size, "unique_glyphs": len(paired),
            "delta_vs_candidate_v4": float(paired.mean()), "paired_se": standard_error(paired)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session1", "session4"))
    parser.add_argument("--ocf-images", type=int, default=64)
    parser.add_argument("--depth-images", type=int, default=8)
    parser.add_argument("--calibration-glyphs", type=int, default=512)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/predictive_atlas_validation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the experiment with Kaggle competition input and a GPU.")
    if not 0 <= args.depth_images <= args.ocf_images <= 320 or not args.ocf_images or args.calibration_glyphs < 2:
        parser.error("Use 1–320 evaluation glyphs and at least two fit-only calibration glyphs.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    fit, held = split_glyphs(len(images), 320, args.seed, device)
    fit_images = images[fit]
    metric_rows = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_images, replace=False)
    style_rows = np.random.default_rng(args.seed + 2).choice(len(fit), 128, replace=False)
    calibration = torch.as_tensor(np.random.default_rng(args.seed + 17).choice(
        len(fit), args.calibration_glyphs, replace=False), device=device)
    remaining = torch.ones(len(fit), dtype=torch.bool, device=device)
    remaining[calibration] = False
    query_rows = held[torch.as_tensor(metric_rows, device=device)]
    atlas, provenance, codepoints = build_font_atlas(fit_images[style_rows], device, return_codepoints=True)
    aligned, agreement = align_training_glyphs(atlas, fit_images)
    style, style_fit = fit_aligned_font_style(aligned, fit_images, seed=args.seed)
    prior = fit_atlas_prior(atlas, codepoints, fit_images)
    score = official_scorer(root)
    output = {"seed": args.seed, "ocf_images_per_block": args.ocf_images, "depth_images_per_block": args.depth_images,
              "fit_glyphs": len(fit), "calibration_glyphs": len(calibration), "held_glyphs": len(held),
              "encoder_epochs": args.encoder_epochs, "style_fit": style_fit, "atlas": provenance,
              "training_alignment_agreement": agreement, "prior": prior.diagnostics, "blocks": []}
    scores_by_method, depths_by_method = {}, {}
    for mouse in ("mice1", "mice2"):
        data = grouped_blocks(root, catalog, mouse, device, len(images))
        training = {name: counts[fit] for name, counts in data.items()}
        smaller_training = {name: counts[remaining] for name, counts in training.items()}
        nested = fit_glyph_encoder(smaller_training, aligned[remaining], epochs=args.encoder_epochs, seed=args.seed)
        likelihood = {name: fit_predictive_likelihood(values[calibration],
                                                       nested.expected_counts(aligned[calibration], name))
                      for name, values in training.items() if name in args.sessions}
        del nested, smaller_training
        encoder = fit_glyph_encoder(training, aligned, epochs=args.encoder_epochs, seed=args.seed)
        prototypes, stats = {}, {}
        for name, counts in training.items():
            features, stats[name] = standardize(feature_view(counts, "evoked-total"))
            prototypes[name] = features.mean(dim=1)
        decoder = fit_glyph_reconstructor(prototypes, fit_images, seed=args.seed,
                                          refine_epochs=12, contour_epochs=3)
        for name in args.sessions:
            observed = data[name][query_rows]
            features, _ = standardize(feature_view(observed, "evoked-total"), stats[name])
            direct = decoder.predict_variants(name, features.mean(1))
            with torch.autocast(device_type="cuda"):
                expected = encoder.expected_counts(atlas.images, name, batch_size=256).float()
            base = font_scores(atlas, expected, observed, direct)
            shape = standardized(base["font_contour"])
            whitened = standardized(likelihood[name].logits(observed, expected, include_model_error=False))
            modeled = standardized(likelihood[name].logits(observed, expected, include_model_error=True))
            contenders = {BASELINE: base["font_fused"], "repeat_noise": 0.5 * (shape + whitened),
                          "predictive_error": 0.5 * (shape + modeled),
                          "predictive_blend": 0.62 * base["font_fused"] + 0.38 * (0.5 * (shape + modeled))}
            choices = {label: prior.variants(logits)["usage_unseen"].topk(max(DEPTHS), dim=1).indices
                       for label, logits in contenders.items()}
            # References are read only after every candidate has been ranked.
            reference = images[query_rows] >= 128
            visual_match = (image_descriptors(reference.to(torch.uint8) * 255) @ atlas.contours.T).argmax(dim=1)
            paired, depth = evaluate(choices, atlas, style, reference.cpu().numpy(), score, device, args.depth_images)
            block = {"mouse": mouse, "session": name, "calibration": likelihood[name].diagnostics, "results": {}}
            for label, values in paired.items():
                summary = summarize([values], [paired[BASELINE]])
                summary["visual_neighbor_in_top32"] = float((choices[label] == visual_match[:, None]).any(1).float().mean())
                scores_by_method.setdefault(label, []).append(values)
                if depth:
                    summary["oracle_ocf_by_depth"] = {k: float(v.mean()) for k, v in depth[label].items()}
                    for k, values_depth in depth[label].items():
                        depths_by_method.setdefault((label, k), []).append(values_depth)
                block["results"][label] = summary
            output["blocks"].append(block)
            output["paired_summary"] = {label: summarize(values, scores_by_method[BASELINE])
                                         for label, values in scores_by_method.items()}
            for (label, k), values in depths_by_method.items():
                output["paired_summary"][label].setdefault("oracle_by_depth", {})[k] = summarize(
                    values, depths_by_method[(BASELINE, k)])
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(block), flush=True)
        del data, training, encoder, decoder


if __name__ == "__main__":
    main()
