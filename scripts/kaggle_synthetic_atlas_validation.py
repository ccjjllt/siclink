"""Test whether simulated public-font spike responses improve unseen-glyph reconstruction."""

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
from glyph_mbr import contour_signatures, contour_utility
from glyph_synthetic_contour import estimate_contours, fit_synthetic_contours
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_exact_rerank_validation import fit_direct
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, standard_error, standardized, font_scores


def summary(measurements, reference):
    scores = np.asarray(measurements)
    difference = (scores - np.asarray(reference)).mean(axis=0)
    return {"ocf": float(scores.mean()), "paired_delta": float(difference.mean()),
            "paired_glyph_se": standard_error(difference), "n": scores.size}


def contour_targets(signature, energy):
    return torch.cat((signature, energy.log()[:, None]), dim=1)


@torch.no_grad()
def evaluate(models, counts, session, atlas, atlas_signature, atlas_energy, prior):
    encoder, decoder, normalization, contour = models
    binned, _ = standardize(feature_view(counts, "evoked-total"), normalization[session])
    direct = decoder.predict_variants(session, binned.mean(dim=1))
    with torch.autocast(device_type="cuda"):
        rates = encoder.expected_counts(atlas.images, session, batch_size=192).float()
    anchor = standardized(font_scores(atlas, rates, counts, direct)["font_fused"])
    log_energy = atlas_energy.log()
    lower, upper = log_energy.aminmax()
    options = {"v4": anchor}
    for name, predicted in estimate_contours(contour, counts, session).items():
        shape = contour_utility(predicted[:, :-1], predicted[:, -1].clamp(lower, upper).exp(),
                                atlas_signature, atlas_energy)
        options[name] = 0.60 * anchor + 0.40 * standardized(shape)
    return {name: prior.variants(scores)["usage_unseen"].topk(16, dim=1).indices
            for name, scores in options.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=961)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--depth-images", type=int, default=8)
    parser.add_argument("--synthetic-glyphs", type=int, default=8192)
    parser.add_argument("--synthetic-weight", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/synthetic_atlas_validation.json"))
    args = parser.parse_args()
    if not 0 <= args.depth_images <= args.ocf_images <= 320 or args.ocf_images < 2:
        parser.error("Score 2 to 320 holdout glyphs; the depth sample must be a subset.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle GPU with mounted competition data.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    fit, held = split_glyphs(len(images), 320, args.seed, device)
    fit_images = images[fit]
    picked = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_images, replace=False)
    chosen = held[torch.as_tensor(picked, device=device)]
    style_rows = torch.as_tensor(np.random.default_rng(args.seed + 2).choice(len(fit), 128, replace=False),
                                 device=device)
    atlas, provenance, codepoints = build_font_atlas(fit_images[style_rows], device, return_codepoints=True)
    if not 2 <= args.synthetic_glyphs <= len(atlas.images) or not 0 < args.synthetic_weight < 1:
        parser.error("Use a sampled public atlas and a proper synthetic fit weight.")
    aligned, agreement = align_training_glyphs(atlas, fit_images)
    prior = fit_atlas_prior(atlas, codepoints, fit_images)
    style, style_fit = fit_aligned_font_style(aligned, fit_images, seed=args.seed)
    atlas_signature, atlas_energy = contour_signatures(atlas.images)
    simulated_rows = torch.as_tensor(np.random.default_rng(args.seed + 3).choice(
        len(atlas.images), args.synthetic_glyphs, replace=False), device=device)
    rendered = torch.cat([(style.apply(batch[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                          for batch in atlas.images.split(1024)])
    references = (images[chosen] >= 128).cpu().numpy()
    scorer = PairedOCF(official_scorer(root), rendered, references, device, 12)
    grouped = grouped_blocks(root, catalog, "mice1", device, len(images))
    training = {name: values[fit] for name, values in grouped.items()}
    print(f"Fit the encoder and the decoder on {len(fit):,} real stimuli", flush=True)
    encoder = fit_glyph_encoder(training, aligned, epochs=18, seed=args.seed)
    decoder, stats = fit_direct(training, fit_images, args.seed)
    contour, calibration = fit_synthetic_contours(training, aligned, contour_targets(*contour_signatures(aligned)),
                                                    atlas.images[simulated_rows], contour_targets(
                                                        atlas_signature[simulated_rows], atlas_energy[simulated_rows]),
                                                    encoder, seed=args.seed + 4, fraction=args.synthetic_weight)
    output = {"mouse": "mice1", "seed": args.seed, "training_real_glyphs": len(fit), "public_atlas": provenance,
              "style_fit": style_fit, "training_alignment_agreement": agreement, "simulated_glyphs": len(simulated_rows),
              "simulated_pairs_per_glyph_and_session": 2, "repeats_per_simulated_pair": 4,
              "synthetic_weight": args.synthetic_weight, "calibration": calibration, "blocks": {},
              "protocol": "Outer held glyphs supply recorded spikes for ranking, and their image labels are used "
              "only in exact OCF. Synthetic spikes and target contours use a fixed public font atlas after fit-only "
              "geometry. v8, real Fourier and both augmented branches share the same 60/40 atlas-contour blend."}
    observed, depth = {}, {}
    for session, counts in grouped.items():
        ranks = evaluate((encoder, decoder, stats, contour), counts[chosen], session,
                         atlas, atlas_signature, atlas_energy, prior)
        scores = {name: scorer.evaluate(index[:, :1])[:, 0] for name, index in ranks.items()}
        block = {name: summary([values], [scores["v8"]]) for name, values in scores.items()}
        for name, values in scores.items():
            observed.setdefault(name, []).append(values)
        if args.depth_images:
            for name, rank in ranks.items():
                best = scorer.evaluate(rank[:args.depth_images]).max(axis=1)
                depth.setdefault(name, []).append(best)
        output["blocks"][session] = block
        output["paired_summary"] = {name: summary(values, observed["v8"]) for name, values in observed.items()}
        output["best_of_16_diagnostic"] = {name: summary(values, depth["v8"]) for name, values in depth.items()}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, "methods": block}), flush=True)


if __name__ == "__main__":
    main()
