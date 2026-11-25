"""Validate uncertainty-weighted decoded contour features for the open atlas."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_contour_retrieval import _response_features, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_encoding import fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_mbr import contour_signatures, contour_utility
from glyph_retrieval import _standardize
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_exact_rerank_validation import fit_direct
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, font_scores, standard_error, standardized


BASELINE = "v4"


@torch.no_grad()
def contour_estimates(model, counts, session):
    response = _response_features(counts, model.view, model.n_bins)
    response = _standardize(response, *model.moments[session]).mean(dim=1) @ model.projection
    return model.readout.predict(response)


@torch.no_grad()
def contour_error(estimate, reference):
    residual = reference - estimate
    offset = residual.mean(dim=0)
    variance = (residual - offset).square().mean(dim=0)
    x, y = estimate - estimate.mean(dim=0), reference - reference.mean(dim=0)
    correlation = ((x * y).mean(dim=0) / (x.square().mean(dim=0) * y.square().mean(dim=0) + 1e-14).sqrt())
    # Fit-only validation controls which shapes are distinguishable. Retain a
    # small floor so a coordinate with an uncertain correlation is still used.
    floor = 0.2 * variance.mean().clamp_min(1e-8)
    weight = (0.025 + correlation.clamp(0, 1).square()) / (variance + floor)
    weight = weight / weight.mean()
    return offset, weight, {"mean_correlation": float(correlation.mean()),
                            "p90_weight": float(torch.quantile(weight, 0.90)),
                            "coordinates_correlation_over_0.15": int((correlation > 0.15).sum())}


def score_contours(estimate, signature, energy, error):
    offset, weights, _ = error
    log_energy = estimate[:, -1].clamp(energy.log().amin(), energy.log().amax())
    contours = contour_utility(estimate[:, :-1], log_energy.exp(), signature, energy)
    # Candidate-specific square terms let the shortlisting compare in the
    # original contour coordinates; no held image supplies a shape distance.
    reference = torch.cat((signature, energy.log()[:, None]), dim=1)
    center = reference.mean(0)
    reference -= center
    normalized = (estimate + offset - center) * weights
    uncertainty = normalized @ reference.T - 0.5 * (reference.square() @ weights)
    return contours, uncertainty


def summarize(values, baseline):
    stacked = np.asarray(values)
    delta = (stacked - np.asarray(baseline)).mean(axis=0)
    return {"ocf": float(stacked.mean()), "delta": float(delta.mean()),
            "paired_se": standard_error(delta), "n": stacked.size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=887)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", default=("session1", "session4"),
                        choices=("session1", "session2", "session3", "session4"))
    parser.add_argument("--ocf-images", type=int, default=96)
    parser.add_argument("--depth-images", type=int, default=8)
    parser.add_argument("--calibration-glyphs", type=int, default=320)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/atlas_contour_likelihood.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the open font and OCF experiment on Kaggle GPU.")
    if not 0 <= args.depth_images <= args.ocf_images <= 320 or not args.ocf_images:
        parser.error("Score 1–320 outer glyphs and choose a valid subset for the depth diagnostic.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    fit, held = split_glyphs(len(images), 320, args.seed, device)
    if not 1 < args.calibration_glyphs < len(fit) - 128:
        parser.error("Keep at least 129 fitted glyphs besides the inner calibration group.")
    metric = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_images, replace=False)
    chosen = held[torch.as_tensor(metric, device=device)]
    permutation = torch.as_tensor(np.random.default_rng(args.seed + 17).permutation(len(fit)), device=device)
    calibration, inner = permutation[:args.calibration_glyphs], permutation[args.calibration_glyphs:]
    fit_images = images[fit]
    style_rows = torch.as_tensor(np.random.default_rng(args.seed + 2).choice(len(fit), 128, replace=False),
                                 device=device)
    atlas, provenance, codepoints = build_font_atlas(fit_images[style_rows], device, return_codepoints=True)
    aligned, agreement = align_training_glyphs(atlas, fit_images)
    prior = fit_atlas_prior(atlas, codepoints, fit_images)
    style, style_fit = fit_aligned_font_style(aligned, fit_images, seed=args.seed)
    signature, energy = contour_signatures(aligned)
    atlas_signature, atlas_energy = contour_signatures(atlas.images)
    rendered = torch.cat([(style.apply(chunk[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                          for chunk in atlas.images.split(1024)])
    reference = (images[chosen] >= 128).cpu().numpy()
    scorer = PairedOCF(official_scorer(root), rendered, reference, device, 12)
    output = {"atlas": provenance, "style_fit": style_fit, "training_alignment_agreement": agreement,
              "seed": args.seed, "fit_glyphs": len(fit), "fit_only_calibration_glyphs": len(calibration),
              "mice": list(args.mice), "sessions": list(args.sessions),
              "ocf_glyphs_per_block": len(chosen), "depth_glyphs_per_block": args.depth_images,
              "weighting": "Per-session predictive error and correlation from fit-only out-of-fold contours.",
              "blocks": []}
    readings, recall = {}, {}
    for mouse in dict.fromkeys(args.mice):
        grouped = grouped_blocks(root, catalog, mouse, device, len(images))
        fit_responses = {name: counts[fit] for name, counts in grouped.items()}
        nested = fit_contour_retrieval({name: counts[inner] for name, counts in fit_responses.items()},
                                      signature[inner], energy[inner], task="task2", seed=args.seed)
        ref = torch.cat((signature[calibration], energy[calibration].log()[:, None]), dim=1)
        errors = {name: contour_error(contour_estimates(nested, fit_responses[name][calibration], name), ref)
                  for name in args.sessions}
        del nested
        contour = fit_contour_retrieval(fit_responses, signature, energy, task="task2", seed=args.seed)
        encoder = fit_glyph_encoder(fit_responses, aligned, epochs=18, seed=args.seed)
        decoder, stats = fit_direct(fit_responses, fit_images, args.seed)
        for session in args.sessions:
            counts = grouped[session][chosen]
            features, _ = standardize(feature_view(counts, "evoked-total"), stats[session])
            direct = decoder.predict_variants(session, features.mean(dim=1))
            with torch.autocast(device_type="cuda"):
                rates = encoder.expected_counts(atlas.images, session, batch_size=192).float()
            base = standardized(font_scores(atlas, rates, counts, direct)["font_fused"])
            estimate = contour_estimates(contour, counts, session)
            unweighted, likelihood = score_contours(estimate, atlas_signature, atlas_energy, errors[session])
            methods = {BASELINE: base, "contour_blend": 0.6 * base + 0.4 * standardized(unweighted),
                       "uncertainty_only": standardized(likelihood),
                       "uncertainty_blend": 0.6 * base + 0.4 * standardized(likelihood)}
            rankings = {name: prior.variants(scores)["usage_unseen"].topk(32, dim=1).indices
                        for name, scores in methods.items()}
            paired = {name: scorer.evaluate(index[:, :1])[:, 0] for name, index in rankings.items()}
            block = {"mouse": mouse, "session": session, "error_fit": errors[session][2], "metrics": {}}
            for name, values in paired.items():
                block["metrics"][name] = summarize([values], [paired[BASELINE]])
                readings.setdefault(name, []).append(values)
            if args.depth_images:
                depth = {name: scorer.evaluate(index[:args.depth_images]) for name, index in rankings.items()}
                for name, candidates in depth.items():
                    for width in (8, 16, 32):
                        recall.setdefault((name, width), []).append(candidates[:, :width].max(axis=1))
                merged = np.maximum(depth[BASELINE][:, :16].max(1), depth["contour_blend"][:, :16].max(1))
                recall.setdefault(("union_baseline_and_contour", 32), []).append(merged)
            output["blocks"].append(block)
            output["paired_summary"] = {name: summarize(values, readings[BASELINE]) for name, values in readings.items()}
            output["best_candidate_diagnostics"] = {f"{name}_top{width}": summarize(values, recall[BASELINE, width])
                                                     for (name, width), values in recall.items()}
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(block), flush=True)
        del grouped, fit_responses, contour, encoder, decoder


if __name__ == "__main__":
    main()
