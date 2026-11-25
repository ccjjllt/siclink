"""Update the replicated mice2 session1 Task 1 block in the v9 archive."""

import argparse
import json
from pathlib import Path
import time

import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_merge_candidate_outputs import encode_grayscale_png, mounted_submission, replace_submission_rows, sha256
from kaggle_task1_contour_update import verify_unmodified_rows
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank, shortlist


MOUSE, SESSION = "mice2", "session1"
ANCHOR = 4.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run the exact OCF calibration and inference on Kaggle GPU.")
    torch.set_num_threads(4)
    started = time.monotonic()
    base = mounted_submission(args.base_sha)
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cuda:0")
    images = load_images(root, device)
    training = grouped_blocks(root, catalog, MOUSE, device, len(images))
    entries = [entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE
               and entry["split"] == "test" and entry["task"] == "task1" and entry["session"] == SESSION]
    if len(entries) != 1:
        raise ValueError("Expected one official mice2 session1 Task 1 block.")

    split = split_calibration(len(images), 96, 384, device, args.seed)
    ranker, likelihoods, nested, codepoints = nested_ranker(training, images, split, official_scorer(root), args.seed)
    model = fit_v8(training, images, args.seed)
    if not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("The fitted ranker and full encoder need identical public codepoint order.")
    del training
    trial = read_trials(root, catalog, entries[0], device)
    if trial.counts.shape[1] != 20:
        raise ValueError("Expected 20 measured repeats per official Task 1 group.")
    candidates, _ = shortlist(model, likelihoods[SESSION], trial.counts, SESSION)
    selected = ranker.select(candidates, anchor_ocf=ANCHOR)
    if not bool((selected >= 0).all()):
        raise ValueError("Each Task 1 sample needs a valid public glyph.")
    predictions = {sample_id: encode_grayscale_png(pixels)
                   for sample_id, pixels in zip(trial.ids, rendered_bank(model)[selected].cpu().numpy())}
    if (len(predictions) != TASK_SIZES["task1"] // (2 * len(SESSIONS))
            or any(not sample_id.startswith("test_task1_pic") for sample_id in predictions)):
        raise ValueError("Expected 340 distinct official Task 1 rows.")

    output = Path("/kaggle/working/kaggle_submission.zip")
    staged = output.with_suffix(".tmp")
    try:
        rows, changed = replace_submission_rows(base, staged, predictions)
        verify_unmodified_rows(base, staged, predictions)
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)
    manifest = {"base_sha256": args.base_sha, "submission_sha256": sha256(output), "rows": rows,
                "target": "mice2 session1 task1", "selected_rows": len(predictions), "changed_predictions": changed,
                "verified_unchanged_non_target_rows": rows - len(predictions), "repeats": trial.counts.shape[1],
                "seed": args.seed, "anchor_ocf": ANCHOR, "nominations": NOMINATIONS,
                "fit_glyphs": len(images), "nested_fit": nested, "full_fit": model.report,
                "ranker": ranker.diagnostics, "seconds": time.monotonic() - started}
    output.with_name("source_candidates.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
