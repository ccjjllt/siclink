"""Update only mice1 Task 1 with the exact OCF ranker anchored to v8."""

import argparse
import csv
import io
import json
from pathlib import Path
import time
from zipfile import ZipFile

import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_merge_candidate_outputs import encode_grayscale_png, mounted_submission, replace_submission_rows, sha256
from kaggle_task1_contour_update import verify_unmodified_rows
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank, shortlist


V8_SHA = "eea6a9f43c24fd7e3ba37cb8d16c7d273df5ef00e24befbcd4bc677802638eaa"


def base_target_predictions(path, target_ids):
    """Read stored v8 prediction strings for the already processed target IDs."""
    with ZipFile(path) as archive:
        with io.TextIOWrapper(archive.open("submission.csv"), encoding="utf-8", newline="") as stream:
            rows = csv.reader(stream)
            if next(rows) != ["sample_id", "prediction"]:
                raise ValueError("Expected the official v8 CSV header.")
            predictions = {sample_id: prediction for sample_id, prediction in rows if sample_id in target_ids}
    if set(predictions) != set(target_ids):
        raise ValueError("The verified v8 archive must contain every mice1 Task 1 ID.")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", default=V8_SHA)
    parser.add_argument("--anchor-ocf", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--metric-glyphs", type=int, default=96)
    parser.add_argument("--noise-glyphs", type=int, default=384)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle GPU for fit-only exact OCF calibration and inference.")
    if args.anchor_ocf < 0 or min(args.metric_glyphs, args.noise_glyphs) < 2:
        parser.error("Choose nonnegative v8 anchoring and positive nested calibration sizes.")
    torch.set_num_threads(4)
    started = time.monotonic()
    base = mounted_submission(args.base_sha)
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cuda:0")
    images = load_images(root, device)
    if args.metric_glyphs + args.noise_glyphs > len(images) - 512:
        parser.error("Reserve at least 512 glyphs for the inner v8 encoder fit.")
    training = grouped_blocks(root, catalog, "mice1", device, len(images))
    entries = sorted((entry for entry in catalog["blocks"]
                      if entry["mouse"] == "mice1" and entry["split"] == "test" and entry["task"] == "task1"),
                     key=lambda entry: entry["session"])
    if tuple(entry["session"] for entry in entries) != SESSIONS:
        raise ValueError("Expected all four mice1 Task 1 recording sessions.")
    split = split_calibration(len(images), args.metric_glyphs, args.noise_glyphs, device, args.seed)
    ranker, likelihoods, nested, codepoints = nested_ranker(training, images, split, official_scorer(root), args.seed)
    model = fit_v8(training, images, args.seed)
    if not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("Nested calibration and full v8 fit must use the same atlas codepoint order.")
    del training
    rendered = rendered_bank(model)

    predictions, recreated, changed_by_ranker, sessions = {}, {}, set(), {}
    for entry in entries:
        block = read_trials(root, catalog, entry, device)
        if block.counts.shape[1] != 20:
            raise ValueError("Expected official twenty-repeat Task 1 query groups.")
        candidates, baseline = shortlist(model, likelihoods[block.session], block.counts, block.session)
        chosen = ranker.select(candidates, anchor_ocf=args.anchor_ocf)
        if not bool((chosen >= 0).all()):
            raise ValueError("Every query needs a valid public atlas candidate.")
        baseline_png = rendered[baseline].cpu().numpy()
        ranked_png = rendered[chosen].cpu().numpy()
        changed = (chosen != baseline).cpu().tolist()
        for sample_id, control, prediction, differs in zip(block.ids, baseline_png, ranked_png, changed):
            if sample_id in predictions or not sample_id.startswith("test_task1_pic"):
                raise ValueError("Expected unique Task 1 IDs across mice1 recordings.")
            recreated[sample_id] = encode_grayscale_png(control)
            predictions[sample_id] = encode_grayscale_png(prediction) if differs else recreated[sample_id]
            if differs:
                changed_by_ranker.add(sample_id)
        sessions[block.session] = {"queries": len(block.ids), "ranker_changed_atlas_candidate": sum(changed)}
        print(f"mice1/{block.session}: {sum(changed):,} of {len(block.ids):,} decisions differ from the v8 anchor", flush=True)
    if len(predictions) != TASK_SIZES["task1"] // 2:
        raise ValueError("Expected exactly the official mice1 Task 1 rows.")

    original = base_target_predictions(base, predictions)
    recreated_agreement = sum(recreated[sample_id] == payload for sample_id, payload in original.items())
    for sample_id in predictions.keys() - changed_by_ranker:
        predictions[sample_id] = original[sample_id]
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
                "verified_unchanged_non_target_rows": rows - len(predictions), "sessions": sessions,
                "v8_recreated_matching_rows": recreated_agreement, "ranker_changed_choices": len(changed_by_ranker),
                "original_v8_rows_retained_for_unchanged_ranker_choices": len(predictions) - len(changed_by_ranker),
                "seed": args.seed, "anchor_ocf": args.anchor_ocf, "nominations": NOMINATIONS,
                "fit_glyphs": len(images), "metric_calibration_glyphs": len(split[0]),
                "noise_calibration_glyphs": len(split[1]), "nested_fit": nested, "full_v8_fit": model.report,
                "ranker": ranker.diagnostics, "seconds": time.monotonic() - started}
    output.with_name("source_candidates.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
