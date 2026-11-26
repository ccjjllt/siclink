"""Package the validated mice1 Task 1 four-fold ranker as private v13 over v12."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, fit_atlas_metric_ranker
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_merge_candidate_outputs import encode_grayscale_png, mounted_submission, replace_submission_rows, sha256
from kaggle_task1_crossfit_metric_scale import calibrated_fold, fold_indices
from kaggle_task1_metric_scale import (ANCHOR, MATCHED_ROWS_PER_EPOCH, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE, RANKER_BATCH,
                                       RANKER_EPOCHS)
from kaggle_task1_v8_metric_update import V8_SHA, base_target_predictions
from kaggle_task3_density_candidate import open_png, verify_archive
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank, shortlist


V12_SHA = "c4b3b9e89719cb9e7db8687be343d8653b54c533f979a1cc00e755315c756039"
MOUSE = "mice1"
TASK = "task1"
QUERY_REPEATS = 20


def fit_crossfit_head(training, images, scorer, atlas_order):
    noise, splits = fold_indices(len(images), images.device)
    candidates, targets, reports = [], [], {}
    for fold, (metric, inner) in enumerate(splits):
        print(f"Calibration fold {fold}: fitting {len(inner):,} labeled upstream glyphs", flush=True)
        graph, exact, _, codepoints, report = calibrated_fold(training, images, metric, noise, inner, scorer)
        if not torch.equal(atlas_order, codepoints):
            raise ValueError("Every independently calibrated fold must share exact public font column order.")
        candidates.append(graph)
        targets.append(exact)
        reports[str(fold)] = report
    expanded = fit_atlas_metric_ranker(concat_atlas_candidates(candidates), torch.cat(targets), seed=MODEL_SEED,
                                      epochs=RANKER_EPOCHS, batch_rows=RANKER_BATCH,
                                      rows_per_epoch=MATCHED_ROWS_PER_EPOCH)
    return expanded, reports


def formatted_choice(index, anchor, inherited, bank):
    return inherited if index == anchor else encode_grayscale_png(bank[index])


def same_pixels(original, replacement):
    if original == replacement:
        return True
    with open_png(original) as first, open_png(replacement) as second:
        return np.array_equal(np.asarray(first), np.asarray(second))


def make_replacements(blocks, base, inherited, bank):
    changes, reports, changed_ids = {}, {}, {}
    for name, (ids, anchors, originals, proposed) in blocks.items():
        record = {"queries": len(ids), "inherited_v8_for_stock_choice": 0, "inherited_v8_for_pooled_choice": 0,
                  "v12_recreated_png_bytes_match": 0, "v12_recreated_pixels_match": 0,
                  "changed_atlas_choices": 0, "changed_png_rows": 0,
                  "changed_choices_retained_due_to_unrecreated_v12_pixels": 0}
        changed_ids[name] = []
        for sample_id, anchor, old, new in zip(ids, anchors, originals, proposed, strict=True):
            actual = base[sample_id]
            prior = inherited[sample_id]
            original = formatted_choice(old, anchor, prior, bank)
            reproduced = same_pixels(actual, original)
            record["inherited_v8_for_stock_choice"] += int(old == anchor)
            record["inherited_v8_for_pooled_choice"] += int(new == anchor)
            record["v12_recreated_png_bytes_match"] += int(actual == original)
            record["v12_recreated_pixels_match"] += int(reproduced)
            if new == old:
                continue
            record["changed_atlas_choices"] += 1
            if not reproduced:
                record["changed_choices_retained_due_to_unrecreated_v12_pixels"] += 1
                continue
            replacement = formatted_choice(new, anchor, prior, bank)
            if same_pixels(actual, replacement):
                continue
            changes[sample_id] = replacement
            record["changed_png_rows"] += 1
            changed_ids[name].append(sample_id)
        reports[name] = record
        print(json.dumps({"session": name, **record}), flush=True)
    return changes, reports, changed_ids


def predict(root, catalog, entries, model, stock, expanded, uncertainty):
    ids_in_scope, blocks = set(), {}
    rows = TASK_SIZES[TASK] // (2 * len(SESSIONS))
    for entry in entries:
        block = read_trials(root, catalog, entry, model.atlas.images.device)
        if block.counts.shape[:2] != (rows, QUERY_REPEATS):
            raise ValueError("Every official mice1 Task 1 recording needs 340 groups of 20 spikes trials.")
        current, anchor = shortlist(model, uncertainty[block.session], block.counts, block.session)
        first = stock.select(current, anchor_ocf=ANCHOR)
        second = expanded.select(current, anchor_ocf=ANCHOR)
        if (first < 0).any() or (second < 0).any():
            raise ValueError("Both heads must select a valid font candidate for every row.")
        ids = tuple(block.ids)
        if len(set(ids)) != rows or ids_in_scope.intersection(ids) or any(not x.startswith("test_task1_pic") for x in ids):
            raise ValueError("The target must contain uniquely named official Task 1 sample rows.")
        ids_in_scope.update(ids)
        blocks[block.session] = (ids, anchor.cpu().tolist(), first.cpu().tolist(), second.cpu().tolist())
    if len(ids_in_scope) != TASK_SIZES[TASK] // 2:
        raise ValueError("The proposed patch must cover exactly the samples from mice1 Task 1.")
    return blocks, ids_in_scope


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a private Kaggle T4 with the competition and both verified source candidates mounted.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    source = mounted_submission(V12_SHA)
    v8 = mounted_submission(V8_SHA)
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    training = grouped_blocks(root, catalog, MOUSE, device, len(images))
    entries = sorted((row for row in catalog["blocks"] if row["mouse"] == MOUSE and row["split"] == "test"
                      and row["task"] == TASK), key=lambda row: row["session"])
    if (len(images) != 2748 or tuple(training) != SESSIONS or tuple(e["session"] for e in entries) != SESSIONS
            or any(values.shape[:2] != (len(images), 4) for values in training.values())):
        raise ValueError("Expect 2,748 labeled glyphs and four mice1 Task 1 recordings.")
    scorer = official_scorer(root)
    original_split = split_calibration(len(images), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    print("Rebuilding the archived v12 Task 1 selection with the original nested_ranker", flush=True)
    stock, uncertainty, stock_fit, codes = nested_ranker(training, images, original_split, scorer, MODEL_SEED)
    expanded, folds = fit_crossfit_head(training, images, scorer, codes)
    model = fit_v8(training, images, MODEL_SEED)
    del training
    if not torch.equal(model.codepoints.cpu(), codes):
        raise ValueError("The official proposal graph must share the calibrated public atlas order.")
    bank = rendered_bank(model).cpu().numpy()
    blocks, target_ids = predict(root, catalog, entries, model, stock, expanded, uncertainty)
    old = base_target_predictions(source, target_ids)
    inherited = base_target_predictions(v8, target_ids)
    changes, sessions, changed_ids = make_replacements(blocks, old, inherited, bank)

    working = Path("/kaggle/working")
    released = working / "kaggle_submission.zip"
    staged = released.with_suffix(".tmp")
    try:
        rows, updated = replace_submission_rows(source, staged, changes)
        if updated != len(changes) or verify_archive(source, staged, changes) != updated:
            raise ValueError("The validated archive must contain all and only the intentional updates.")
        staged.replace(released)
    finally:
        staged.unlink(missing_ok=True)

    def total(field):
        return sum(row[field] for row in sessions.values())

    id_path = working / "changed_task1_mice1_ids.json"
    id_path.write_text(json.dumps(changed_ids, indent=2) + "\n", encoding="utf-8")
    manifest = {"candidate": "v13", "base_sha256": V12_SHA, "inherited_anchor_v8_sha256": V8_SHA,
                "submission_sha256": sha256(released), "rows": rows, "target": "mice1 task1",
                "target_rows": len(target_ids), "changed_predictions": updated,
                "changed_atlas_choices": total("changed_atlas_choices"),
                "changed_choices_retained_due_to_unrecreated_v12_pixels": total("changed_choices_retained_due_to_unrecreated_v12_pixels"),
                "v12_unrecreated_pixel_rows_retained": len(target_ids) - total("v12_recreated_pixels_match"),
                "every_replaced_row_recreated_v12_pixels": True,
                "changed_ids_file": id_path.name, "changed_ids_sha256": sha256(id_path),
                "other_rows_verified_unchanged": rows - len(target_ids), "all_unchanged_verified": rows - updated,
                "v12_recreated_matching_pixels": total("v12_recreated_pixels_match"),
                "v12_recreated_matching_png_bytes": total("v12_recreated_png_bytes_match"),
                "inherited_v8_rows_in_reconstruction": total("inherited_v8_for_stock_choice"),
                "sessions": sessions, "fit_glyphs": len(images), "training_repeats": 4, "query_repeats": QUERY_REPEATS,
                "seed": MODEL_SEED, "anchor_ocf": ANCHOR, "nominations": NOMINATIONS,
                "noise_calibration_glyphs": len(original_split[1]),
                "pooled_calibration_glyphs": sum(r["calibration_glyphs"] for r in folds.values()),
                "optimizer_steps_per_head": RANKER_EPOCHS * MATCHED_ROWS_PER_EPOCH // RANKER_BATCH,
                "unique_calibration_pairs": sum(r["unique_exact_calibration_pairs"] for r in folds.values()),
                "calibration_folds": folds, "literal_v11_neural_fit": stock_fit,
                "v11_ranker": stock.diagnostics, "crossfit_ranker": expanded.diagnostics,
                "full_v8_fit": model.report, "atlas_order_verified": True, "source_hashes_verified": True,
                "row_order_verified": True, "png_verified": "all 30,200 base64 payloads decode to 64x64 grayscale PNG with CRC check",
                "seconds": time.monotonic() - started}
    (working / "source_candidates.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
