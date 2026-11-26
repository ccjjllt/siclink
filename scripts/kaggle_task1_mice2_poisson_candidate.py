"""Package the verified mice2 Task 1 1.4× Poisson decision over immutable v17."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import official_scorer
from kaggle_merge_candidate_outputs import encode_grayscale_png, mounted_submission, replace_submission_rows, sha256
from kaggle_task1_crossfit_candidate import fit_crossfit_policy
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED
from kaggle_task1_mice2_crossfit_candidate import MOUSE, QUERY_REPEATS, TASK
from kaggle_task1_v11_repeat_weight_transfer import (ANCHOR_OCF, POISSON_AMPLIFICATION, decisions, normalized_sources,
                                                      weighted_fusion)
from kaggle_task3_density_candidate import open_png, stored_predictions, verify_archive
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, rendered_bank


BASE_SHA = "61dc0a81e8ff1f3f5c323f516818947c24886f7a6708ad71b84b9ee38aa52ec5"
BASE_SOURCE = "https://www.kaggle.com/code/ccjjllt/siclink-candidate-v17-task3-utility/versions/1"
VALIDATIONS = ("https://www.kaggle.com/code/ccjjllt/siclink-task1-mice2-microns-spatial-atlas/versions/1",
               "https://www.kaggle.com/code/ccjjllt/siclink-task1-mice2-poisson-repeat-recheck/versions/1")


@torch.no_grad()
def predict(root, catalog, entries, model, ranker, noise):
    per_day = TASK_SIZES[TASK] // (2 * len(SESSIONS))
    blocks, scope = {}, set()
    for entry in entries:
        trial = read_trials(root, catalog, entry, model.atlas.images.device, load_labels=False)
        day, ids = trial.session, tuple(trial.ids)
        if (trial.counts.shape[:2] != (per_day, QUERY_REPEATS) or len(ids) != per_day or len(set(ids)) != per_day
                or scope.intersection(ids) or any(not sample.startswith("test_task1_pic") for sample in ids)):
            raise ValueError("Expected distinct official mice2 Task 1 groups with exactly twenty measured repeats.")
        scope.update(ids)
        original = atlas_scores(model, noise[day], trial.counts, day)
        amplified = {**original, **weighted_fusion(model, normalized_sources(original), POISSON_AMPLIFICATION)}
        choices, diagnostics = decisions(original, amplified, ranker)
        before, after = choices["v11"], choices["poisson_same_budget"]
        if bool(((before < 0) | (after < 0)).any()):
            raise ValueError("Both nominal policies must select an atlas glyph for every query.")
        blocks[day] = (ids, before.tolist(), after.tolist(), diagnostics)
    if len(scope) != TASK_SIZES[TASK] // 2:
        raise ValueError("The patch scope must comprise all four complete mice2 Task 1 recordings.")
    return blocks, scope


def protected_replacements(blocks, stored, rendered):
    updates, changed_ids, summaries = {}, {}, {}
    for day, (ids, before, after, diagnostics) in blocks.items():
        changed_ids[day] = []
        counts = {"nominal_v14_recreated_base_pixels": 0, "changed_atlas_choices": 0,
                  "suggested_changes_retained_due_to_reproduction_mismatch": 0,
                  "different_atlas_choice_with_identical_pixels": 0}
        for sample, incumbent, proposed in zip(ids, before, after, strict=True):
            with open_png(stored[sample]) as original:
                reference = np.asarray(original)
                matched = np.array_equal(reference, rendered[incumbent])
                counts["nominal_v14_recreated_base_pixels"] += int(matched)
                if incumbent == proposed:
                    continue
                counts["changed_atlas_choices"] += 1
                if not matched:
                    counts["suggested_changes_retained_due_to_reproduction_mismatch"] += 1
                    continue
                if np.array_equal(reference, rendered[proposed]):
                    counts["different_atlas_choice_with_identical_pixels"] += 1
                    continue
            updates[sample] = encode_grayscale_png(rendered[proposed])
            changed_ids[day].append(sample)
        summaries[day] = {**counts, "queries": len(ids), "changed_predictions": len(changed_ids[day]),
                          "unrecreated_base_pixels_retained": len(ids) - counts["nominal_v14_recreated_base_pixels"],
                          "proposal_diagnostics": diagnostics}
        print(json.dumps({"session": day, **summaries[day]}), flush=True)
    return updates, changed_ids, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Extracted competition directory containing catalog.json")
    parser.add_argument("--base", type=Path, help="Exact verified v17 ZIP to patch")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use a remote Kaggle or Colab GPU with access to the official training data and scorer.")
    if ANCHOR != ANCHOR_OCF:
        raise RuntimeError("The protected proposal policy must use the nominal v14 anchor.")
    base = args.base.resolve(strict=True) if args.base else mounted_submission(BASE_SHA)
    if sha256(base) != BASE_SHA:
        raise ValueError("The base ZIP must match the verified immutable v17 SHA-256.")
    root = args.data_root.resolve(strict=True) if args.data_root else locate_data()
    output = args.output_dir.resolve()
    archive = output / "kaggle_submission.zip"
    if archive.resolve() == base.resolve():
        raise ValueError("Give the patched ZIP an output directory distinct from the verified v17 base archive.")
    # The official scorer package must be extracted alongside the supplied data root.
    scorer = official_scorer(root)

    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    catalog = load_catalog(root)
    images = load_images(root, device)
    training = grouped_blocks(root, catalog, MOUSE, device, len(images))
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE
                      and entry["split"] == "test" and entry["task"] == TASK), key=lambda entry: entry["session"])
    if (len(images) != 2748 or tuple(training) != SESSIONS or tuple(entry["session"] for entry in entries) != SESSIONS
            or any(block.shape[:2] != (len(images), 4) for block in training.values())):
        raise ValueError("Expected the complete labeled gallery and four official mice2 recordings.")

    ranker, folds, noise, codes = fit_crossfit_policy(training, images, scorer)
    model = fit_v8(training, images, MODEL_SEED)
    del training
    if not torch.equal(model.codepoints.cpu(), codes):
        raise ValueError("The outer model and all calibration folds must use the exact same public atlas order.")
    rendered = rendered_bank(model).cpu().numpy()
    blocks, scope = predict(root, catalog, entries, model, ranker, noise)
    updates, changed_ids, sessions = protected_replacements(blocks, stored_predictions(base, scope), rendered)

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".mice2-poisson-", dir=output) as temporary:
        stage = Path(temporary)
        staged_archive = stage / archive.name
        rows, changed = replace_submission_rows(base, staged_archive, updates)
        if (rows != 30_200 or changed != len(updates) or updates.keys() - scope
                or verify_archive(base, staged_archive, updates) != changed):
            raise ValueError("The full candidate must preserve every out-of-scope and unreproduced row.")
        changes = stage / "changed_task1_mice2_ids.json"
        changes.write_text(json.dumps(changed_ids, indent=2) + "\n", encoding="utf-8")
        manifest = {"stage": "complete", "candidate": "v18-mice2-poisson", "base_sha256": BASE_SHA,
                    "base_source": BASE_SOURCE, "validation_notebooks": VALIDATIONS, "submission_sha256": sha256(staged_archive),
                    "rows": rows, "target": "mice2 task1", "target_rows": len(scope), "changed_predictions": changed,
                    "unmodified_rows_verified_byte_exact": rows - changed, "outside_scope_verified_byte_exact": rows - len(scope),
                    "changed_ids_file": changes.name, "changed_ids_sha256": sha256(changes), "sessions": sessions,
                    "nominal_v14_recreated_base_pixels": sum(day["nominal_v14_recreated_base_pixels"] for day in sessions.values()),
                    "source_fit_repeats": 4, "formal_query_repeats": QUERY_REPEATS, "formal_labels_loaded": False,
                    "poisson_multiplier": POISSON_AMPLIFICATION, "same_distinct_candidate_budget": True,
                    "v14_anchor_ocf": ANCHOR, "v14_nominations": NOMINATIONS, "seed": MODEL_SEED,
                    "inference_noise_source": "literal v14 calibration fold zero, with identical split and fit seed",
                    "v14_four_fold_calibration": folds, "v14_ranker": ranker.diagnostics,
                    "v14_outer_fit": model.report, "every_changed_row_recreated_nominal_v14_base_pixels": True,
                    "archive_verification": "official ID ordering, exact untouched CSV rows, ZIP CRC, 64x64 grayscale PNG",
                    "seconds": time.monotonic() - started}
        report = stage / "source_candidates.json"
        report.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged_archive.replace(archive)
        changes.replace(output / changes.name)
        report.replace(output / report.name)
    print(json.dumps({"stage": manifest["stage"], "submission_sha256": manifest["submission_sha256"],
                      "changed_predictions": changed, "seconds": manifest["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
