"""Package the validated mice2 Task 1 four-fold ranker as private v14 over v13."""

import json
from pathlib import Path
import time

import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import (SHORTLIST as LEGACY_NOMINATIONS, fit_calibrated_ranker,
                                           shortlist as legacy_shortlist, split_calibration)
from kaggle_gallery_validation import align_training_glyphs
from kaggle_merge_candidate_outputs import encode_grayscale_png, mounted_submission, replace_submission_rows, sha256
from kaggle_task1_crossfit_candidate import fit_crossfit_head, same_pixels
from kaggle_task1_metric_scale import (ANCHOR, MATCHED_ROWS_PER_EPOCH, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE, RANKER_BATCH,
                                       RANKER_EPOCHS)
from kaggle_task3_density_candidate import stored_predictions, verify_archive
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank, shortlist


V13_SHA = "f478a5c0e1a68b1c0dfd5e016432861ada3b0f2c1433d0a6fb3b0e761e8ce540"
MOUSE, TASK = "mice2", "task1"
MODERN_SESSION = "session1"
LEGACY_ANCHOR = 2.0
QUERY_REPEATS = 20


@torch.no_grad()
def predict(entries, root, catalog, model, modern, expanded, legacy, modern_noise, legacy_noise):
    per_session = TASK_SIZES[TASK] // (2 * len(SESSIONS))
    scope, blocks = set(), {}
    for entry in entries:
        block = read_trials(root, catalog, entry, model.atlas.images.device)
        if block.counts.shape[:2] != (per_session, QUERY_REPEATS):
            raise ValueError("Require 340 official groups of 20 measured trials per mice2 recording.")
        candidates, _ = shortlist(model, modern_noise[block.session], block.counts, block.session)
        selected = expanded.select(candidates, anchor_ocf=ANCHOR)
        if block.session == MODERN_SESSION:
            original = modern.select(candidates, anchor_ocf=ANCHOR)
        else:
            old = legacy_shortlist(model.atlas, model.prior, model.encoder, model.decoder, model.normalization,
                                   legacy_noise[block.session], block.counts, block.session)
            original = legacy.select(old, anchor_ocf=LEGACY_ANCHOR)
        if bool(((original < 0) | (selected < 0)).any()):
            raise ValueError("Both decision policies must choose a public glyph for all queries.")
        ids = tuple(block.ids)
        if (len(ids) != per_session or len(set(ids)) != per_session or scope.intersection(ids)
                or any(not key.startswith("test_task1_pic") for key in ids)):
            raise ValueError("Only unique official mice2 Task 1 prediction IDs may enter packaging.")
        scope.update(ids)
        blocks[block.session] = (ids, original.cpu().tolist(), selected.cpu().tolist())
    if len(scope) != TASK_SIZES[TASK] // 2:
        raise ValueError("The scoped patch must cover the four complete mice2 Task 1 recordings.")
    return blocks, scope


def replacements(blocks, stored, images):
    updates, summaries, changed = {}, {}, {}
    for session, (ids, incumbent, selected) in blocks.items():
        record = {"queries": len(ids), "incumbent": "v10_modern_96" if session == MODERN_SESSION else "v6_legacy_96",
                  "v13_recreated_pixels": 0, "v13_recreated_png_bytes": 0, "changed_atlas_choices": 0,
                  "changed_choices_retained_due_to_unrecreated_v13_pixels": 0,
                  "changed_choices_with_identical_pixels": 0, "changed_predictions": 0}
        changed[session] = []
        for sample_id, old, new in zip(ids, incumbent, selected, strict=True):
            original = stored[sample_id]
            rerendered = encode_grayscale_png(images[old])
            reproduced = same_pixels(original, rerendered)
            record["v13_recreated_pixels"] += int(reproduced)
            record["v13_recreated_png_bytes"] += int(original == rerendered)
            if old == new:
                continue
            record["changed_atlas_choices"] += 1
            if not reproduced:
                record["changed_choices_retained_due_to_unrecreated_v13_pixels"] += 1
                continue
            amended = encode_grayscale_png(images[new])
            if same_pixels(original, amended):
                record["changed_choices_with_identical_pixels"] += 1
                continue
            updates[sample_id] = amended
            changed[session].append(sample_id)
        record["changed_predictions"] = len(changed[session])
        record["v13_unrecreated_pixel_rows_retained"] = record["queries"] - record["v13_recreated_pixels"]
        summaries[session] = record
        print(json.dumps({"session": session, **record}), flush=True)
    return updates, changed, summaries


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on one private Kaggle T4 with the competition and verified v13 source mounted.")
    begin = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    source = mounted_submission(V13_SHA)
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    training = grouped_blocks(root, catalog, MOUSE, device, len(images))
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE and entry["split"] == "test"
                      and entry["task"] == TASK), key=lambda entry: entry["session"])
    if (len(images) != 2748 or tuple(training) != SESSIONS or tuple(e["session"] for e in entries) != SESSIONS
            or any(block.shape[:2] != (len(images), 4) for block in training.values())):
        raise ValueError("Expected four complete official mice2 Task 1 recordings and the labeled gallery.")
    scorer = official_scorer(root)
    split = split_calibration(len(images), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    print("Recreating the original v10 head for mice2 session1", flush=True)
    modern, modern_noise, modern_fit, codes = nested_ranker(training, images, split, scorer, MODEL_SEED)
    print("Calibrating the validated mice2 four-fold head", flush=True)
    expanded, folds = fit_crossfit_head(training, images, scorer, codes)
    print("Fitting the common outer model and the archived v6 head for sessions2–4", flush=True)
    model = fit_v8(training, images, MODEL_SEED)
    if not torch.equal(model.codepoints.cpu(), codes):
        raise ValueError("The outer model and all held calibration folds must share public font column order.")
    bank = rendered_bank(model)
    aligned, _ = align_training_glyphs(model.atlas, images)
    legacy, legacy_noise, legacy_fit = fit_calibrated_ranker(model.atlas, model.codepoints, images, aligned, training,
                                                             split, bank, scorer, SESSIONS, MODEL_SEED)
    del training, aligned
    groups, scope = predict(entries, root, catalog, model, modern, expanded, legacy, modern_noise, legacy_noise)
    base_rows = stored_predictions(source, scope)
    updates, changed, sessions = replacements(groups, base_rows, bank.cpu().numpy())

    work = Path("/kaggle/working")
    released = work / "kaggle_submission.zip"
    temporary = released.with_suffix(".tmp")
    try:
        rows, patched = replace_submission_rows(source, temporary, updates)
        if patched != len(updates) or verify_archive(source, temporary, updates) != patched:
            raise ValueError("The verified archive must contain all and only the intended mice2 Task 1 changes.")
        temporary.replace(released)
    finally:
        temporary.unlink(missing_ok=True)

    changed_path = work / "changed_task1_mice2_ids.json"
    changed_path.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")

    def total(key):
        return sum(item[key] for item in sessions.values())

    manifest = {"candidate": "v14", "base_sha256": V13_SHA, "submission_sha256": sha256(released),
                "target": "mice2 task1", "rows": rows, "target_rows": len(scope), "changed_predictions": patched,
                "other_rows_verified_unchanged": rows - len(scope), "all_unchanged_verified": rows - patched,
                "changed_ids_file": changed_path.name, "changed_ids_sha256": sha256(changed_path),
                "v13_recreated_matching_pixels": total("v13_recreated_pixels"),
                "v13_recreated_matching_png_bytes": total("v13_recreated_png_bytes"),
                "v13_unrecreated_pixel_rows_retained": total("v13_unrecreated_pixel_rows_retained"),
                "changed_choices_retained_due_to_unrecreated_v13_pixels": total("changed_choices_retained_due_to_unrecreated_v13_pixels"),
                "changed_atlas_choices": total("changed_atlas_choices"), "every_replaced_row_recreated_v13_pixels": True,
                "sessions": sessions, "fit_glyphs": len(images), "training_repeats": 4, "query_repeats": QUERY_REPEATS,
                "seed": MODEL_SEED, "expanded_anchor_ocf": ANCHOR, "legacy_anchor_ocf": LEGACY_ANCHOR,
                "expanded_nominations": NOMINATIONS, "legacy_nominations": LEGACY_NOMINATIONS,
                "noise_calibration_glyphs": len(split[1]), "calibration_glyphs": sum(r["calibration_glyphs"] for r in folds.values()),
                "optimizer_steps_per_head": RANKER_EPOCHS * MATCHED_ROWS_PER_EPOCH // RANKER_BATCH,
                "unique_exact_calibration_pairs": sum(r["unique_exact_calibration_pairs"] for r in folds.values()),
                "crossfit_ranker": expanded.diagnostics, "crossfit_folds": folds, "modern_incumbent_fit": modern_fit,
                "modern_incumbent_ranker": modern.diagnostics, "legacy_incumbent_fit": legacy_fit,
                "shared_outer_fit": model.report, "only_v13_source_archive": True, "atlas_order_verified": True,
                "row_order_verified": True, "archive_and_png_crc_verified": True,
                "png_verified": "all 30,200 base64 payloads decode to 64x64 grayscale PNG",
                "seconds": time.monotonic() - begin}
    (work / "source_candidates.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
