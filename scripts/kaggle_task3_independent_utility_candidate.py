"""Build the validated independent Task 3 utility candidate over v16 MICrONS 1.4."""

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_exact_utility import gallery_geometry, gallery_utility
from glyph_mbr import contour_signatures, fit_temperature, glyph_posterior, similarity_from_signatures
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import official_scorer
from kaggle_merge_candidate_outputs import encode_grayscale_png, replace_submission_rows, sha256
from kaggle_task3_density_candidate import MOUSE, TASK, open_png, stored_predictions, verify_archive
from kaggle_task3_half_balance_recheck import REPEAT_SEED, SEED, SHORTLIST, TEMP, cohorts, disjoint_trial_split
from kaggle_task3_part_discovery import SESSIONS
from kaggle_task3_rankband_ocf_validation import fit_visual_metric, neural_rows, stage
from kaggle_task3_relational_consensus import ACTUAL, INDEPENDENT, UTILITY_WEIGHT, choose_images
from kaggle_task3_repeat_posterior import CLASSIFIER_WEIGHT, fit_models


BASE_NAME = "kaggle_submission_microns_poisson140.zip"
BASE_SHA = "a342f28c350b51b803cf53dc5369210d522706aa0fc96de57cbdb805ec12a20e"
BASE_SOURCE = "https://www.kaggle.com/code/ccjjllt/siclink-candidate-v16-microns/versions/1"
SUPPORT = "https://www.kaggle.com/code/ccjjllt/siclink-task3-half-balance-recheck/versions/1"
SETTINGS = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128, visual_pairs=14336, query_batch=64)
WORKING = Path("/kaggle/working")


def verified_base():
    matches = (path for path in Path("/kaggle/input").rglob(BASE_NAME) if path.is_file())
    archive = next((path for path in matches if sha256(path) == BASE_SHA), None)
    if archive is None:
        raise FileNotFoundError("Mount the exact v16 MICrONS 1.4 ZIP verified by SHA-256.")
    return archive


def fit_visual(masks, signature, energy, visual, scorer):
    geometry = gallery_geometry(masks, signature, energy)
    proxy = similarity_from_signatures(signature, energy)
    draw = torch.Generator(device=masks.device).manual_seed(SETTINGS.seed + 641)
    metric, strata, image_pairs = fit_visual_metric(masks, geometry, proxy, visual, scorer, draw, SETTINGS)
    return gallery_utility(metric, geometry), {"fit_glyphs": len(visual), "sampled_pairs_by_stratum": strata,
                                               "unique_exact_pairs": image_pairs}


def calibrate(recordings, gallery, signature, energy, labels):
    # Match the validated nested repeat split. Calibration rows are the recorded third repeat.
    inner, calibration, _, _ = disjoint_trial_split(recordings, MOUSE)
    if len(labels) != TEMP or any(part.shape[1] != 1 for part in calibration.values()):
        raise ValueError("Expected the original independent source-day calibration cohort.")
    model, report = fit_models(inner, gallery, signature, energy, MOUSE, SETTINGS)
    panels = {day: neural_rows(model, counts[labels], gallery, day, SETTINGS).mixed
              for day, counts in calibration.items()}
    temperatures = {}
    for target in SESSIONS:
        days = [day for day in SESSIONS if day != target]
        fit = fit_temperature([(panels[day], labels) for day in days])
        temperatures[target] = {**fit, "source_days": days}
    return temperatures, report


@torch.no_grad()
def predict(model, entries, root, catalog, gallery, utility, temperatures, started):
    # Pass only spike counts to the policy. The sample IDs are used to write the corresponding predictions.
    assignments, seen = {}, set()
    for entry in entries:
        block = read_trials(root, catalog, entry, gallery.device, load_labels=False)
        day = block.session
        ids = tuple(block.ids)
        if (block.counts.shape[:2] != (len(gallery), 1) or len(ids) != len(gallery) or len(set(ids)) != len(ids)
                or seen.intersection(ids) or any(not key.startswith("test_task3_pic") for key in ids)):
            raise ValueError("Expected distinct mice1 Task 3 formal groups of one measured trial each.")
        seen.update(ids)
        candidates = neural_rows(model, block.counts, gallery, day, SETTINGS)
        if candidates.action.shape != (len(ids), SHORTLIST):
            raise ValueError("Every query needs the exact production seven-action shortlist.")
        mass = glyph_posterior(candidates.mixed, temperatures[day]["temperature"], balance=False)
        decision = choose_images(candidates, {INDEPENDENT: mass}, utility)
        assignments[day] = (ids, decision[ACTUAL].tolist(), decision[INDEPENDENT].tolist())
        stage("formal_spikes_scored", started, session=day, queries=len(ids),
              temperature=temperatures[day]["temperature"],
              changed_atlas_choices=int((decision[ACTUAL] != decision[INDEPENDENT]).sum()))
    if len(seen) != TASK_SIZES[TASK] // 2:
        raise ValueError("Only the four complete mice1 Task 3 recordings may enter the archive patch.")
    return assignments, seen


def changed_images(assignments, stored, rendered):
    substitutions, changed, summary = {}, {}, {}
    for day, (ids, incumbent, chosen) in assignments.items():
        count = {"queries": len(ids), "v14_recreated_pixel_exact": 0, "changed_atlas_choices": 0,
                 "suggested_changes_skipped_recreation_mismatch": 0, "same_rendering_as_stored": 0}
        changed[day] = []
        for sample, old, proposed in zip(ids, incumbent, chosen, strict=True):
            with open_png(stored[sample]) as original:
                pixels = np.asarray(original)
                matched = np.array_equal(pixels, rendered[old])
                count["v14_recreated_pixel_exact"] += int(matched)
                if old == proposed:
                    continue
                count["changed_atlas_choices"] += 1
                if not matched:
                    count["suggested_changes_skipped_recreation_mismatch"] += 1
                    continue
                if np.array_equal(pixels, rendered[proposed]):
                    count["same_rendering_as_stored"] += 1
                    continue
            substitutions[sample] = encode_grayscale_png(rendered[proposed])
            changed[day].append(sample)
        summary[day] = {**count, "changed_predictions": len(changed[day]),
                        "unrecreated_v14_rows_preserved": len(ids) - count["v14_recreated_pixel_exact"]}
    return substitutions, changed, summary


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a private Kaggle T4 and the verified v16 allowlisted input archive.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    base = verified_base()
    root = locate_data()
    catalog = load_catalog(root)
    gallery = load_images(root, device)
    masks = gallery >= 128
    signature, energy = contour_signatures(gallery)
    scorer = official_scorer(root)
    training = grouped_blocks(root, catalog, MOUSE, device, len(gallery))
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE
                      and entry["split"] == "test" and entry["task"] == TASK), key=lambda entry: entry["session"])
    if (len(gallery) != 2748 or tuple(training) != SESSIONS or tuple(entry["session"] for entry in entries) != SESSIONS
            or any(values.shape[:2] != (len(gallery), 4) for values in training.values())):
        raise ValueError("Expected four labeled mice1 recordings and their corresponding Task 3 recordings.")
    visual, partitions, _ = cohorts(len(gallery), device)
    _, temperature_glyphs = partitions[MOUSE]
    if bool(torch.isin(visual, temperature_glyphs).any()):
        raise ValueError("Image utility fitting and spike temperature calibration must retain separate reference glyphs.")
    utility, image_fit = fit_visual(masks, signature, energy, visual, scorer)
    stage("visual_utility_fitted", started, **image_fit)
    temperatures, temp_fit = calibrate(training, gallery, signature, energy, temperature_glyphs)
    stage("source_temperatures_fitted", started, temperatures=temperatures)
    model, training_fit = fit_models(training, gallery, signature, energy, MOUSE, SETTINGS)
    del training
    assignments, target_ids = predict(model, entries, root, catalog, gallery, utility, temperatures, started)
    patches, changed, sessions = changed_images(assignments, stored_predictions(base, target_ids),
                                               (masks.to(torch.uint8) * 255).cpu().numpy())
    archive = WORKING / "kaggle_submission.zip"
    temporary = archive.with_suffix(".tmp")
    try:
        rows, count = replace_submission_rows(base, temporary, patches)
        if count != len(patches) or not count or patches.keys() - target_ids or verify_archive(base, temporary, patches) != count:
            raise ValueError("Only pixel-exact reproducible mice1 Task 3 rows may change.")
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    list_path = WORKING / "changed_task3_mice1_ids.json"
    list_path.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
    report = {"stage": "complete", "candidate": "v17", "base_archive": BASE_NAME, "base_sha256": BASE_SHA,
              "base_source": BASE_SOURCE, "validation_notebook": SUPPORT, "submission_sha256": sha256(archive),
              "rows": rows, "target": "mice1 task3", "target_rows": len(target_ids), "changed_predictions": count,
              "untargeted_rows_byte_exact": rows - len(target_ids), "unchanged_rows_byte_exact": rows - count,
              "changed_ids_file": list_path.name, "changed_ids_sha256": sha256(list_path),
              "recreated_v14_pixel_exact": sum(row["v14_recreated_pixel_exact"] for row in sessions.values()),
              "block_size": len(gallery), "source_fit_repeats": 4, "formal_query_repeats": 1,
              "same_v14_candidate_budget": SHORTLIST, "classifier_weight": CLASSIFIER_WEIGHT,
              "utility_weight": UTILITY_WEIGHT, "posterior": "rowwise_temperature_softmax",
              "temperature_glyph_seed": SEED, "repeat_seed": REPEAT_SEED, "temperature_fit": temp_fit,
              "temperature_by_session": temperatures, "settings": vars(SETTINGS), "visual_fit": image_fit,
              "full_neural_fit": training_fit, "sessions": sessions, "formal_labels_loaded": False,
              "candidate_archive_verified": "exact CSV ordering, unchanged row bytes, CRC and 64x64 grayscale PNG",
              "only_pixel_exact_recreated_v14_rows_changed": True, "seconds": time.monotonic() - started}
    (WORKING / "source_candidates.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    stage("complete", started, archive_sha256=report["submission_sha256"], changed_predictions=count)


if __name__ == "__main__":
    main()
