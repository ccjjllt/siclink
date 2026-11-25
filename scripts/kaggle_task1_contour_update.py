"""Update mice1 Task 1 with the fixed aligned atlas and directional contour blend."""

import argparse
from dataclasses import asdict, replace
from itertools import zip_longest
import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_encoding import fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_mbr import contour_signatures
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import feature_view, standardize
from kaggle_exact_rerank_validation import fit_direct
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, font_scores, standardized
from kaggle_merge_candidate_outputs import encode_grayscale_png, mounted_submission, replace_submission_rows, sha256


MOUSE = "mice1"
ATLAS_WEIGHT = 0.60
CONTOUR_WEIGHT = 0.40
ENCODER_EPOCHS = 18


def verify_unmodified_rows(base, output, predictions):
    """Verify source CSV bytes for every row outside the replacement set."""
    targets = {sample_id.encode("ascii") for sample_id in predictions}
    with ZipFile(base) as original, ZipFile(output) as update:
        with original.open("submission.csv") as raw, update.open("submission.csv") as changed:
            for before, after in zip_longest(raw, changed):
                if before == after:
                    continue
                if before is None or after is None:
                    raise ValueError("The base and replacement must have the same number of CSV lines.")
                before_id = before.split(b",", 1)[0]
                if before_id not in targets or before_id != after.split(b",", 1)[0]:
                    raise ValueError("A CSV row outside mice1 Task 1 changed.")


@torch.no_grad()
def predict(block, atlas, contour, encoder, decoder, stats, prior, style):
    features, _ = standardize(feature_view(block.counts, "evoked-total"), stats[block.session])
    direct = decoder.predict_variants(block.session, features.mean(dim=1))
    with torch.autocast(device_type="cuda"):
        rates = encoder.expected_counts(atlas.images, block.session, batch_size=192).float()
    anchor = standardized(font_scores(atlas, rates, block.counts, direct)["font_fused"])
    # Task 2 pooled readout has no local or nonlinear branch. The public atlas
    # energy range in contour_retrieval_scores supplies the validated clamp.
    directional = contour_retrieval_scores(contour, block.counts, block.session)["contour_ridge"]
    blended = ATLAS_WEIGHT * anchor + CONTOUR_WEIGHT * standardized(directional)
    selected = prior.variants(blended)["usage_unseen"].argmax(dim=1)
    mask = style.apply(atlas.images[selected, None].float() / 255)[:, 0] >= 0.5
    return (mask.to(torch.uint8) * 255).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use Kaggle GPU for the all-label atlas and contour fit.")
    torch.set_num_threads(4)
    started = time.monotonic()
    base = mounted_submission(args.base_sha)
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cuda:0")
    images = load_images(root, device)
    training = grouped_blocks(root, catalog, MOUSE, device, len(images))
    entries = sorted((entry for entry in catalog["blocks"]
                      if entry["mouse"] == MOUSE and entry["split"] == "test" and entry["task"] == "task1"),
                     key=lambda entry: entry["session"])
    if not entries or {entry["session"] for entry in entries} != set(training):
        raise ValueError("Expected one Task 1 recording for each fitted mice1 session.")

    style_rows = np.random.default_rng(args.seed + 2).choice(len(images), 128, replace=False)
    atlas, provenance, codepoints = build_font_atlas(images[torch.as_tensor(style_rows, device=device)],
                                                    device, return_codepoints=True)
    aligned, agreement = align_training_glyphs(atlas, images)
    prior = fit_atlas_prior(atlas, codepoints, images)
    style, style_fit = fit_aligned_font_style(aligned, images, seed=args.seed)
    fitted = fit_contour_retrieval(training, *contour_signatures(aligned), task="task2", seed=args.seed)
    atlas_signature, atlas_energy = contour_signatures(atlas.images)
    contour = replace(fitted, signature=atlas_signature, energy=atlas_energy)
    del fitted
    encoder = fit_glyph_encoder(training, aligned, epochs=ENCODER_EPOCHS, seed=args.seed)
    decoder, stats = fit_direct(training, images, args.seed)
    del training

    predictions, sessions = {}, {}
    for entry in entries:
        block = read_trials(root, catalog, entry, device)
        if block.counts.shape[1] != 20:
            raise ValueError("Expected official twenty-repeat Task 1 query groups.")
        rendered = predict(block, atlas, contour, encoder, decoder, stats, prior, style)
        for sample_id, pixels in zip(block.ids, rendered):
            if sample_id in predictions or not sample_id.startswith("test_task1_pic"):
                raise ValueError("Expected unique Task 1 IDs across the mice1 recordings.")
            predictions[sample_id] = encode_grayscale_png(pixels)
        sessions[block.session] = len(block.ids)
        print(f"{MOUSE}/{block.session}: {len(block.ids):,} Task 1 predictions", flush=True)
    if len(predictions) != TASK_SIZES["task1"] // 2:
        raise ValueError("Expected exactly the official mice1 Task 1 rows.")

    output = Path("/kaggle/working/kaggle_submission.zip")
    staged = output.with_suffix(".tmp")
    try:
        rows, changed = replace_submission_rows(base, staged, predictions)
        verify_unmodified_rows(base, staged, predictions)
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)
    manifest = {"base_sha256": args.base_sha, "submission_sha256": sha256(output), "rows": rows,
                "target": "mice1 task1", "selected_rows": len(predictions), "changed_predictions": changed,
                "verified_unchanged_rows": rows - len(predictions), "sessions": sessions, "seed": args.seed,
                "fit_glyphs": len(images), "task1_repeats": 20, "encoder_epochs": ENCODER_EPOCHS,
                "anchor_weight": ATLAS_WEIGHT, "directional_contour_weight": CONTOUR_WEIGHT,
                "contour_training": "evoked-multiscale pooled repeat readout",
                "style": asdict(style), "style_fit": style_fit, "training_alignment_agreement": agreement,
                "atlas": provenance, "prior_variant": "usage_unseen", "prior": prior.diagnostics,
                "seconds": time.monotonic() - started}
    output.with_name("source_candidates.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
