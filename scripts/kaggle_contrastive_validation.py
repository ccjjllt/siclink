"""Compare cross-modal discriminative ranking against aligned Poisson glyph retrieval."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_contrastive import fit_glyph_contrastive
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_encoding import fit_glyph_encoder
from glyph_reconstruction import fit_glyph_reconstructor
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_gallery_validation import (align_training_glyphs, build_font_atlas, font_scores, score_block,
                                       standard_error, standardized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session1", "session4"))
    parser.add_argument("--ocf-per-session", type=int, default=64)
    parser.add_argument("--style-fit-glyphs", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--refine-epochs", type=int, default=12)
    parser.add_argument("--contour-epochs", type=int, default=3)
    parser.add_argument("--contrastive-epochs", type=int, default=30)
    parser.add_argument("--contrastive-batch-size", type=int, default=192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/contrastive_validation.json"))
    args = parser.parse_args()
    if not 1 <= args.ocf_per_session <= 320 or min(args.style_fit_glyphs, args.encoder_epochs,
                                                    args.refine_epochs, args.contrastive_batch_size) < 1:
        parser.error("Use positive experiment settings and at most 320 OCF glyphs per block.")
    if not 0 <= args.contour_epochs <= args.refine_epochs or args.contrastive_epochs < 0:
        parser.error("Choose valid contour and contrastive epoch counts.")
    device = torch.device(args.device)
    if not Path("/kaggle/input").is_dir() or device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle GPU Notebook with the competition input and Internet enabled.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    score = official_scorer(root)
    train, held = split_glyphs(len(images), 320, args.seed, device)
    query_rows = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_per_session, replace=False)
    queries = held[torch.as_tensor(query_rows, device=device)]
    if args.style_fit_glyphs > len(train):
        parser.error("Style calibration needs no more glyphs than the fit split contains.")
    style_rows = np.random.default_rng(args.seed + 2).choice(len(train), args.style_fit_glyphs, replace=False)
    atlas, provenance = build_font_atlas(images[train[torch.as_tensor(style_rows, device=device)]], device)
    fit_images = images[train]
    aligned_images, agreement = align_training_glyphs(atlas, fit_images)
    print(json.dumps({"atlas": provenance, "training_alignment_agreement": agreement}), flush=True)

    output = {"device": str(device), "seed": args.seed, "held_glyphs": len(held),
              "ocf_per_session": len(queries), "encoder_epochs": args.encoder_epochs,
              "refine_epochs": args.refine_epochs, "contour_epochs": args.contour_epochs,
              "contrastive_epochs": args.contrastive_epochs, "contrastive_batch_size": args.contrastive_batch_size,
              "training_alignment_agreement": agreement, "atlas": provenance, "rows": []}
    results = {}
    baseline_name = "aligned_gallery_fused"
    for mouse in args.mice:
        grouped = grouped_blocks(root, catalog, mouse, device, len(images))
        training = {session: trials[train] for session, trials in grouped.items()}
        discriminative = fit_glyph_contrastive(training, fit_images, aligned_images,
                                              epochs=args.contrastive_epochs,
                                              batch_size=args.contrastive_batch_size, seed=args.seed, verbose=True)
        gallery = discriminative.image_embeddings(atlas.contours, descriptors=True)
        print(f"{mouse}: cross-modal models fit", flush=True)

        encoder = fit_glyph_encoder(training, aligned_images, epochs=args.encoder_epochs, seed=args.seed)
        prototypes, calibration = {}, {}
        for session, counts in training.items():
            values, calibration[session] = standardize(feature_view(counts, "evoked-total"))
            prototypes[session] = values.mean(dim=1)
        decoder = fit_glyph_reconstructor(prototypes, fit_images, seed=args.seed,
                                          refine_epochs=args.refine_epochs, contour_epochs=args.contour_epochs)
        print(f"{mouse}: aligned Poisson and direct reconstruction comparators fit", flush=True)
        for session in args.sessions:
            trials = grouped[session][queries]
            scores = discriminative.rank(trials, session, gallery)
            inputs, _ = standardize(feature_view(trials, "evoked-total"), calibration[session])
            direct = decoder.predict_variants(session, inputs.mean(1))
            with torch.autocast(device_type="cuda"):
                rates = encoder.expected_counts(atlas.images, session, batch_size=256)
            baseline = font_scores(atlas, rates.float(), trials, direct)["font_fused"]
            scores[baseline_name] = baseline
            for name in ("cca", "contrastive"):
                scores[f"{name}_fused"] = 0.55 * standardized(baseline) + 0.45 * standardized(scores[name])
            estimated = {name: (atlas.images[logits.argmax(1)] >= 128).to(torch.uint8) * 255
                         for name, logits in scores.items()}
            estimated["evoked_total_refined"] = direct["refined"]
            # Held images are accessed after ranking and used only for OCF.
            references = (images[queries] >= 128).cpu().numpy()
            paired = score_block(score, estimated, references, device)
            for variant, values in paired.items():
                delta = values - paired[baseline_name]
                row = {"mouse": mouse, "session": session, "variant": variant,
                       "ocf": float(values.mean()), "ocf_n": len(values),
                       "paired_delta_vs_aligned_gallery": float(delta.mean()), "paired_delta_se": standard_error(delta)}
                output["rows"].append(row)
                results.setdefault(variant, []).append(values)
                print(json.dumps(row), flush=True)

            block_baseline = np.stack(results[baseline_name])
            summary = {}
            for variant, result in results.items():
                blocks = np.stack(result)
                delta = (blocks - block_baseline).mean(axis=0)
                summary[variant] = {"ocf": float(blocks.mean()), "ocf_n": blocks.size,
                                    "paired_delta_vs_aligned_gallery": float(delta.mean()),
                                    "paired_delta_se": standard_error(delta), "unique_held_glyphs": len(delta)}
            output["paired_summary"] = summary
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            del rates
        del discriminative, gallery, encoder, decoder, grouped, training


if __name__ == "__main__":
    main()
