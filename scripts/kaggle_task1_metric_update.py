"""Replace only mice2 Task 1 predictions with the validated OCF reranker."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_encoding import fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import (SHORTLIST, fit_calibrated_ranker, fit_direct, shortlist,
                                           split_calibration)
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas
from kaggle_merge_candidate_outputs import encode_grayscale_png, mounted_submission, replace_submission_rows, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle GPU for the fit-only metric calibration and inference.")
    torch.set_num_threads(4)
    started = time.monotonic()
    base = mounted_submission(args.base_sha)

    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cuda:0")
    training_images = load_images(root, device)
    training = grouped_blocks(root, catalog, "mice2", device, len(training_images))
    entries = sorted((row for row in catalog["blocks"]
                      if row["mouse"] == "mice2" and row["split"] == "test" and row["task"] == "task1"),
                     key=lambda row: row["session"])
    sessions = tuple(row["session"] for row in entries)
    style_rows = torch.as_tensor(np.random.default_rng(args.seed + 2).choice(len(training_images), 128, replace=False),
                                 device=device)
    atlas, provenance, codepoints = build_font_atlas(training_images[style_rows], device, return_codepoints=True)
    aligned, agreement = align_training_glyphs(atlas, training_images)
    style, style_fit = fit_aligned_font_style(aligned, training_images, seed=args.seed)
    rendered = torch.cat([(style.apply(batch[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                          for batch in atlas.images.split(1024)])
    split = split_calibration(len(training_images), 96, 384, device, args.seed)
    ranker, likelihoods, metrics = fit_calibrated_ranker(atlas, codepoints, training_images, aligned, training,
                                                         split, rendered, official_scorer(root), sessions, args.seed)
    prior = fit_atlas_prior(atlas, codepoints, training_images)
    encoder = fit_glyph_encoder(training, aligned, epochs=18, seed=args.seed)
    decoder, stats = fit_direct(training, training_images, args.seed)

    predictions = {}
    for entry in entries:
        block = read_trials(root, catalog, entry, device)
        candidates = shortlist(atlas, prior, encoder, decoder, stats, likelihoods[block.session], block.counts, block.session)
        selected = ranker.select(candidates, anchor_ocf=2.0)
        if not bool((selected >= 0).all()):
            raise ValueError("Every Task 1 sample needs a valid public candidate.")
        images = rendered[selected].cpu().numpy()
        for sample_id, pixels in zip(block.ids, images):
            if sample_id in predictions or not sample_id.startswith("test_task1_pic"):
                raise ValueError("Expected unique Task 1 IDs in each mice2 recording.")
            predictions[sample_id] = encode_grayscale_png(pixels)
        print(f"mice2/{block.session}: {len(block.ids)} Task 1 predictions", flush=True)
    if len(predictions) != TASK_SIZES["task1"] // 2:
        raise ValueError("Expected a full set of mice2 Task 1 replacements.")

    working = Path("/kaggle/working")
    path = working / "kaggle_submission.zip"
    rows, changed = replace_submission_rows(base, path, predictions)
    manifest = {"base_sha256": args.base_sha, "submission_sha256": sha256(path), "rows": rows,
                "target": "mice2 task1", "selected_rows": len(predictions), "changed_predictions": changed,
                "sessions": sessions, "seed": args.seed, "shortlist": SHORTLIST, "anchor_ocf": 2.0,
                "style": asdict(style), "style_fit": style_fit, "training_alignment_agreement": agreement,
                "atlas": provenance, "prior": prior.diagnostics, "metric_fit": metrics,
                "seconds": time.monotonic() - started}
    (working / "source_candidates.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
