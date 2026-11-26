"""Prepare the measured equal-budget Poisson variant over the scored v14."""

import json
from pathlib import Path
import time

import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_merge_candidate_outputs import mounted_submission, replace_submission_rows, sha256
from kaggle_task1_crossfit_candidate import fit_crossfit_head, formatted_choice, same_pixels
from kaggle_task1_metric_scale import MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE
from kaggle_task1_v8_metric_update import V8_SHA, base_target_predictions
from kaggle_task1_v11_repeat_weight_transfer import (POISSON_AMPLIFICATION, decisions, normalized_sources,
                                                     weighted_fusion)
from kaggle_task3_density_candidate import verify_archive
from kaggle_v8_metric_validation import SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank


V14_SHA = "1da6c1c7f6ba4c2dbf388921569baf9fb061a55ff78a822e4823079e5570c1c6"
MOUSE, TASK = "mice1", "task1"
REPEATS = 20


@torch.no_grad()
def predict(root, catalog, entries, model, ranker, noise):
    per_recording = TASK_SIZES[TASK] // (2 * len(SESSIONS))
    blocks, scope = {}, set()
    for entry in entries:
        block = read_trials(root, catalog, entry, model.atlas.images.device)
        ids = tuple(block.ids)
        if (block.counts.shape[:2] != (per_recording, REPEATS) or len(ids) != per_recording
                or len(set(ids)) != per_recording or scope.intersection(ids)
                or any(not sample.startswith("test_task1_pic") for sample in ids)):
            raise ValueError("Expected distinct official mice1 groups with twenty measured repeats each.")
        scope.update(ids)
        original = atlas_scores(model, noise[block.session], block.counts, block.session)
        higher = {**original, **weighted_fusion(model, normalized_sources(original), POISSON_AMPLIFICATION)}
        choices, diagnostics = decisions(original, higher, ranker)
        incumbent, amended = choices["v11"], choices["poisson_same_budget"]
        if bool(((incumbent < 0) | (amended < 0)).any()):
            raise ValueError("Every query should select a valid public glyph.")
        blocks[block.session] = (ids, original["v8"].argmax(1).tolist(), incumbent.tolist(), amended.tolist(),
                                 diagnostics)
        print(json.dumps({"session": block.session, "queries": len(ids), **diagnostics}), flush=True)
    if len(scope) != TASK_SIZES[TASK] // 2:
        raise ValueError("The patch must be scoped to the four complete mice1 Task 1 recordings.")
    return blocks, scope


def select_replacements(blocks, stored, v8, bank):
    changes, changed_ids, report = {}, {}, {}
    for session, (ids, v8_indices, incumbent, amended, diagnostics) in blocks.items():
        changed_ids[session] = []
        reproduced = preserved = same_output = 0
        for sample, original_v8, before, after in zip(ids, v8_indices, incumbent, amended, strict=True):
            current = formatted_choice(before, original_v8, v8[sample], bank)
            matches = same_pixels(current, stored[sample])
            reproduced += int(matches)
            if before == after:
                continue
            if not matches:
                preserved += 1
                continue
            prediction = formatted_choice(after, original_v8, v8[sample], bank)
            if same_pixels(prediction, stored[sample]):
                same_output += 1
                continue
            changes[sample] = prediction
            changed_ids[session].append(sample)
        report[session] = {"queries": len(ids), "reproduced_v14_pixels": reproduced,
                           "unreproduced_v14_rows_preserved": len(ids) - reproduced,
                           "proposed_changes_preserved_due_to_reproduction_mismatch": preserved,
                           "proposed_distinct_glyphs_with_equal_images": same_output,
                           "changed_predictions": len(changed_ids[session]), "proposal_diagnostics": diagnostics}
        print(json.dumps({"session": session, **report[session]}), flush=True)
    return changes, changed_ids, report


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a private Kaggle T4 with the competition and both source notebooks mounted.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    base, inherited = mounted_submission(V14_SHA), mounted_submission(V8_SHA)
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    training = grouped_blocks(root, catalog, MOUSE, device, len(images))
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE
                      and entry["split"] == "test" and entry["task"] == TASK), key=lambda entry: entry["session"])
    if (len(images) != 2748 or tuple(training) != SESSIONS or tuple(entry["session"] for entry in entries) != SESSIONS
            or any(values.shape[:2] != (len(images), 4) for values in training.values())):
        raise ValueError("Expected all labeled glyphs and all four official mice1 recordings.")

    scorer = official_scorer(root)
    split = split_calibration(len(images), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    _, noise, fold_zero, codes = nested_ranker(training, images, split, scorer, MODEL_SEED)
    ranker, folds = fit_crossfit_head(training, images, scorer, codes)
    model = fit_v8(training, images, MODEL_SEED)
    del training
    if not torch.equal(model.codepoints.cpu(), codes):
        raise ValueError("The public atlas order must match the exact calibration.")
    bank = rendered_bank(model).cpu().numpy()
    blocks, scope = predict(root, catalog, entries, model, ranker, noise)
    changes, changed_ids, sessions = select_replacements(
        blocks, base_target_predictions(base, scope), base_target_predictions(inherited, scope), bank)

    working = Path("/kaggle/working")
    output = working / "kaggle_submission.zip"
    temp = output.with_suffix(".tmp")
    try:
        rows, changed = replace_submission_rows(base, temp, changes)
        if changed != len(changes) or verify_archive(base, temp, changes) != changed:
            raise ValueError("Expected the complete archive with only the listed updates.")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)

    ids_path = working / "changed_task1_mice1_ids.json"
    ids_path.write_text(json.dumps(changed_ids, indent=2) + "\n", encoding="utf-8")
    report = {"candidate": "v15-poisson", "base_sha256": V14_SHA, "inherited_v8_sha256": V8_SHA,
              "submission_sha256": sha256(output), "rows": rows, "target": "mice1 task1",
              "target_rows": len(scope), "changed_predictions": changed, "all_unchanged_verified": rows - changed,
              "other_rows_verified_unchanged": rows - len(scope), "changed_ids_file": ids_path.name,
              "changed_ids_sha256": sha256(ids_path), "sessions": sessions, "query_repeats": REPEATS,
              "poisson_multiplier": POISSON_AMPLIFICATION, "every_replaced_row_recreated_v14_pixels": True,
              "same_distinct_candidate_budget": True, "four_fold_calibration": folds, "fold_zero": fold_zero,
              "seed": MODEL_SEED, "ranker": ranker.diagnostics, "outer": model.report,
              "seconds": time.monotonic() - started}
    (working / "source_candidates.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
