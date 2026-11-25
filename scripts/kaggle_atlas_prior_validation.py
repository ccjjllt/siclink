"""Compare train-only Unicode and unseen-shape priors on the fixed Noto atlas."""

import argparse
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_encoding import fit_glyph_encoder
from glyph_reconstruction import fit_glyph_reconstructor
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_gallery_validation import (align_training_glyphs, build_font_atlas, font_scores,
                                       score_block, standard_error, standardized)


def neural_blocks(atlas, root, catalog, image_count, fit_images, font_images, fit, queries, args, device):
    """Yield dense atlas logits. Query images and glyph identities are absent."""
    for mouse in args.mice:
        grouped = grouped_blocks(root, catalog, mouse, device, image_count)
        training = {session: counts[fit] for session, counts in grouped.items()}
        encoder = fit_glyph_encoder(training, font_images, epochs=args.encoder_epochs, seed=args.seed)
        prototypes, calibration = {}, {}
        for session, counts in training.items():
            values, calibration[session] = standardize(feature_view(counts, "evoked-total"))
            prototypes[session] = values.mean(dim=1)
        decoder = fit_glyph_reconstructor(prototypes, fit_images, seed=args.seed,
                                          refine_epochs=args.refine_epochs, contour_epochs=args.contour_epochs)
        for session in args.sessions:
            trials = grouped[session][queries]
            features, _ = standardize(feature_view(trials, "evoked-total"), calibration[session])
            direct = decoder.predict_variants(session, features.mean(1))
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                rates = encoder.expected_counts(atlas.images, session, batch_size=256)
            fused = font_scores(atlas, rates.float(), trials, direct)["font_fused"]
            print(f"{mouse}/{session}: formed all public-candidate scores for {len(trials)} spike samples", flush=True)
            yield (mouse, session), fused
            del rates
        del encoder, decoder, grouped, training


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
    parser.add_argument("--support-consensus", action="store_true",
                        help="also infer a shared repertoire from independently shuffled spike-only cohort scores")
    parser.add_argument("--joint-assignment", action="store_true",
                        help="compare distinct within-block assignment and soft reward for cross-block overlap")
    parser.add_argument("--style-output", action="store_true",
                        help="also render selected glyphs with a shared fit-only geometric correction")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/atlas_prior_validation.json"))
    args = parser.parse_args()
    if not 1 <= args.ocf_per_session <= 320 or min(args.style_fit_glyphs, args.encoder_epochs, args.refine_epochs) < 1:
        parser.error("Use positive experiment settings and at most 320 OCF glyphs per block.")
    if not 0 <= args.contour_epochs <= args.refine_epochs:
        parser.error("Choose a valid contour epoch count.")
    if (args.support_consensus or args.joint_assignment) and len(set(args.mice)) * len(set(args.sessions)) < 2:
        parser.error("Repertoire experiments need at least two recording blocks.")
    device = torch.device(args.device)
    if not Path("/kaggle/input").is_dir() or device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Run this experiment in a Kaggle GPU Notebook with competition input and Internet enabled.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    score = official_scorer(root)
    fit, held = split_glyphs(len(images), 320, args.seed, device)
    query_rows = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_per_session, replace=False)
    query_rows = torch.as_tensor(query_rows, device=device)
    if args.style_fit_glyphs > len(fit):
        parser.error("Style calibration cannot exceed the fit split size.")
    style_rows = np.random.default_rng(args.seed + 2).choice(len(fit), args.style_fit_glyphs, replace=False)
    style_rows = torch.as_tensor(style_rows, device=device)
    fit_images = images[fit]
    atlas, provenance, codepoints = build_font_atlas(fit_images[style_rows], device, return_codepoints=True)
    font_images, contour_agreement = align_training_glyphs(atlas, fit_images)
    prior = fit_atlas_prior(atlas, codepoints, fit_images)
    print(json.dumps({"atlas": provenance, "prior": prior.diagnostics}), flush=True)

    output = {"device": str(device), "seed": args.seed, "fit_glyphs": len(fit), "held_glyphs": len(held),
              "ocf_per_session": len(query_rows), "encoder_epochs": args.encoder_epochs,
              "refine_epochs": args.refine_epochs, "contour_epochs": args.contour_epochs,
              "training_alignment_agreement": contour_agreement, "atlas": provenance,
              "prior": prior.diagnostics, "rows": []}
    style = None
    if args.style_output:
        from glyph_font_style import fit_aligned_font_style

        style, summary = fit_aligned_font_style(font_images, fit_images, seed=args.seed)
        output["output_style"] = {"parameters": asdict(style), "fit": summary}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    full_blocks = args.support_consensus or args.joint_assignment
    ranking_rows = held if full_blocks else held[query_rows]
    scores_by_block = dict(neural_blocks(atlas, root, catalog, len(images), fit_images, font_images, fit,
                                        ranking_rows, args, device))

    consensus = None
    if args.support_consensus:
        from glyph_atlas_consensus import fit_support_consensus

        rng = torch.Generator(device=device).manual_seed(args.seed + 3)
        shuffled = {block: matrix[torch.randperm(len(matrix), generator=rng, device=device)]
                    for block, matrix in scores_by_block.items()}
        consensus = fit_support_consensus(shuffled, temperature=0.55)
        output["support_consensus"] = {str(block): diagnostics for block, diagnostics in consensus.diagnostics.items()}
        del shuffled

    assignments = {}
    if args.joint_assignment:
        from glyph_atlas_assignment import distinct_atlas_glyphs, shared_atlas_assignment

        rng = torch.Generator(device=device).manual_seed(args.seed + 13)
        shuffle = {key: torch.randperm(len(values), generator=rng, device=device)
                   for key, values in scores_by_block.items()}
        output["assignment_mean_overlap"] = {}
        for name, transform in (("gallery", standardized), ("usage", lambda values: prior.variants(values)["usage_unseen"])):
            shuffled = {key: transform(values[shuffle[key]]) for key, values in scores_by_block.items()}
            distinct = {key: distinct_atlas_glyphs(values) for key, values in shuffled.items()}
            joint = shared_atlas_assignment(shuffled)
            for rule, labels in (("distinct", distinct), ("joint", joint)):
                method = f"{name}_{rule}"
                assignments[method] = {key: indices[shuffle[key].argsort()][query_rows]
                                       for key, indices in labels.items()}
                overlap = [float(torch.isin(left, right).sum()) / min(len(left), len(right))
                           for left, right in combinations(labels.values(), 2)]
                output["assignment_mean_overlap"][method] = float(np.mean(overlap))

    baseline_name = "aligned_gallery_fused"
    accumulated = {}
    for (mouse, session), matrix in scores_by_block.items():
        block = (mouse, session)
        base = matrix[query_rows] if full_blocks else matrix
        candidates = prior.variants(base)
        candidates[baseline_name] = base
        if consensus is not None:
            for strength in (0.45, 0.90):
                candidates[f"consensus_{strength:.2f}"] = consensus.rerank((mouse, session), base, strength=strength)
                candidates[f"usage_unseen_consensus_{strength:.2f}"] = consensus.rerank(
                    (mouse, session), candidates["usage_unseen"], strength=strength)
        selections = {name: logits.argmax(1) for name, logits in candidates.items()}
        selections.update({name: blocks[block] for name, blocks in assignments.items()})
        predictions = {name: (atlas.images[labels] >= 128).to(torch.uint8) * 255
                       for name, labels in selections.items()}
        if style is not None:
            for name in (baseline_name, "usage_unseen", *assignments):
                alpha = atlas.images[selections[name], None].float() / 255
                predictions[f"{name}_style"] = (style.apply(alpha)[:, 0] >= 0.5).to(torch.uint8) * 255
        # Held images enter only here for paired official OCF, after ranking.
        references = (images[held[query_rows]] >= 128).cpu().numpy()
        paired = score_block(score, predictions, references, device)
        for variant, values in paired.items():
            delta = values - paired[baseline_name]
            row = {"mouse": mouse, "session": session, "variant": variant, "ocf": float(values.mean()),
                   "ocf_n": len(values), "paired_delta_vs_aligned_gallery": float(delta.mean()),
                   "paired_delta_se": standard_error(delta)}
            output["rows"].append(row)
            accumulated.setdefault(variant, []).append(values)
            print(json.dumps(row), flush=True)

        baselines = np.stack(accumulated[baseline_name])
        output["paired_summary"] = {}
        for variant, values in accumulated.items():
            blocks = np.stack(values)
            glyph_delta = (blocks - baselines).mean(axis=0)
            output["paired_summary"][variant] = {"ocf": float(blocks.mean()), "ocf_n": blocks.size,
                                                  "paired_delta_vs_aligned_gallery": float(glyph_delta.mean()),
                                                  "paired_delta_se": standard_error(glyph_delta),
                                                  "unique_held_glyphs": len(glyph_delta)}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
