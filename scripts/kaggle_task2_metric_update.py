"""Fit the Task 2 OCF assignment model on labeled days and update mice1 session5."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, fit_atlas_metric_ranker
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_mbr import contour_signatures
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import official_scorer
from kaggle_merge_candidate_outputs import encode_grayscale_png, mounted_submission, replace_submission_rows, sha256
from kaggle_task1_contour_update import verify_unmodified_rows
from kaggle_task2_metric_validation import (NOMINATIONS, SESSIONS, metric_labels, score_block, shortlist,
                                           shortlisted_v9_score, sparse_assignment)


MOUSE, TARGET_SESSION = "mice1", "session5"
ANCHOR, CALIBRATION_GLYPHS = 8.0, 128


def fit_cross_recording_ranker(root, training, images, masks, signature, energy, ink, args):
    device = images.device
    score = official_scorer(root)
    rng = np.random.default_rng(args.seed)
    selected = torch.as_tensor(rng.choice(len(images), CALIBRATION_GLYPHS, replace=False), device=device)
    candidate_parts, label_parts, calibration = [], [], {}
    for session in SESSIONS:
        sources = {name: counts for name, counts in training.items() if name != session}
        order = torch.as_tensor(rng.permutation(len(images)), device=device)
        matrices, prior = score_block(MOUSE, sources, training[session][order], images, signature, energy,
                                      device, args)
        rows = order.argsort()[selected]
        candidates = shortlist({name: scores[rows] for name, scores in matrices.items()}, prior[rows], energy, ink)
        target, pairs = metric_labels(candidates, selected, masks, score, device, 384)
        candidate_parts.append(candidates)
        label_parts.append(target)
        calibration[session] = {"neural_fit_sources": list(sources), "ocf_pairs": pairs}
        print(f"{session}: {pairs:,} exact OCF calibration pairs", flush=True)
        del matrices, prior
    ranker = fit_atlas_metric_ranker(concat_atlas_candidates(candidate_parts), torch.cat(label_parts), seed=args.seed)
    return ranker, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--seed", type=int, default=211)
    parser.set_defaults(encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle GPU for the exact OCF calibration.")
    torch.set_num_threads(4)
    started = time.monotonic()
    base = mounted_submission(args.base_sha)
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cuda:0")
    images = load_images(root, device)
    masks = images >= 128
    signatures, energy = contour_signatures(images)
    ink = masks.float().mean(dim=(1, 2))
    training = grouped_blocks(root, catalog, MOUSE, device, len(images))
    entries = [entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE and entry["split"] == "test"
               and entry["task"] == "task2" and entry["session"] == TARGET_SESSION]
    if len(entries) != 1 or set(training) != set(SESSIONS):
        raise ValueError("Expected four labeled recordings and one new mice1 recording.")

    ranker, calibration = fit_cross_recording_ranker(root, training, images, masks, signatures, energy, ink, args)
    block = read_trials(root, catalog, entries[0], device)
    if block.counts.shape[:2] != (len(masks), 4):
        raise ValueError("Expected the full glyph bank and four repeats per query.")
    matrices, baseline = score_block(MOUSE, training, block.counts, images, signatures, energy, device, args)
    candidates = shortlist(matrices, baseline, energy, ink)
    objective = ranker.predict_ocf(candidates) + ANCHOR * shortlisted_v9_score(matrices["v9"], candidates)
    selected, fallback = sparse_assignment(candidates, objective, baseline)
    rendered = (masks[selected].to(torch.uint8) * 255).cpu().numpy()
    predictions = {sample_id: encode_grayscale_png(pixels) for sample_id, pixels in zip(block.ids, rendered)}
    if len(predictions) != TASK_SIZES["task2"] // 2 or any(not item.startswith("test_task2_pic") for item in predictions):
        raise ValueError("Expected the complete set of mice1 Task 2 IDs.")

    output = Path("/kaggle/working/kaggle_submission.zip")
    staged = output.with_suffix(".tmp")
    try:
        rows, changed = replace_submission_rows(base, staged, predictions)
        verify_unmodified_rows(base, staged, predictions)
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)
    manifest = {"base_sha256": args.base_sha, "submission_sha256": sha256(output), "rows": rows,
                "target": "mice1 task2", "target_session": TARGET_SESSION, "selected_rows": len(predictions),
                "changed_predictions": changed, "verified_unchanged_non_target_rows": rows - len(predictions),
                "anchor_ocf": ANCHOR, "sparse_fallback": fallback, "distinct_glyphs": int(selected.unique().numel()),
                "differences_vs_recomputed_v9": int((selected != baseline).sum()), "nominations": NOMINATIONS,
                "calibration": calibration, "ranker": ranker.diagnostics, "seed": args.seed,
                "seconds": time.monotonic() - started}
    output.with_name("source_candidates.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
