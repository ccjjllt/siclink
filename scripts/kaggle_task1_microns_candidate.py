"""Build two mice1 Task 1 MICrONS rate candidates over the verified v14 ZIP."""

import json
from pathlib import Path
import time

import torch

from glyph_atlas_metric_ranker import describe_atlas_candidates, pack_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_merge_candidate_outputs import mounted_submission, replace_submission_rows, sha256
from kaggle_task1_bipartite_validation import replace_poisson
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE
from kaggle_task1_microns_open_atlas import ATLAS_CHUNK, CORE_BATCH, RIDGE, fit_kernel_readout, stream_atlas
from kaggle_task1_microns_transfer import (BLACK_LEVEL, FNN_SOURCE, FNN_VERSION, FRAMES, OFFICIAL_MD5,
                                          PUBLIC_ARCHIVE, STREAM, load_released_core)
from kaggle_task1_poisson_candidate import MOUSE, REPEATS, TASK, V14_SHA, select_replacements
from kaggle_task1_v11_repeat_weight_transfer import (POISSON_AMPLIFICATION, matched_pool, normalized_sources,
                                                      weighted_fusion)
from kaggle_task1_v8_metric_update import V8_SHA, base_target_predictions
from kaggle_task3_density_candidate import verify_archive
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank


MULTIPLIERS = {"microns_poisson100": 1.0, "microns_poisson140": POISSON_AMPLIFICATION}
CORE_SHA256 = "c378e8fbf1e7a9e44f07d5d85fba972b946d73adc096fbf58c69cc524052e1e1"
VALIDATION = "https://www.kaggle.com/code/ccjjllt/siclink-task1-microns-open-atlas/versions/1"
WORKING = Path("/kaggle/working")


def publish(report, stage, started):
    report.update(stage=stage, seconds=time.monotonic() - started)
    (WORKING / "source_candidates.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def predict(root, catalog, entries, model, ranker, likelihood, atlas_rates):
    per_recording = TASK_SIZES[TASK] // (2 * len(SESSIONS))
    device = model.atlas.images.device
    groups = {name: {} for name in MULTIPLIERS}
    scope = set()
    for day_index, entry in enumerate(entries):
        trial = read_trials(root, catalog, entry, device)
        ids = tuple(trial.ids)
        if (trial.counts.shape[:2] != (per_recording, REPEATS) or len(ids) != per_recording
                or len(set(ids)) != per_recording or scope.intersection(ids)
                or any(not sample.startswith("test_task1_pic") for sample in ids)):
            raise ValueError("Expected distinct formal mice1 Task 1 groups with exactly twenty measured repeats.")
        scope.update(ids)
        original = atlas_scores(model, likelihood[trial.session], trial.counts, trial.session)
        baseline = pack_atlas_candidates(original, NOMINATIONS, baseline="v8")
        incumbent = ranker.select(baseline, anchor_ocf=ANCHOR)
        inherited_choice = original["v8"].argmax(1).tolist()
        rates = atlas_rates[day_index].to(device)
        if rates.shape != (len(model.atlas.images), trial.counts.shape[-1]):
            raise ValueError("Template columns must align with this day's neural response bins.")
        replacement = replace_poisson(model, original, trial.counts, rates)
        replacement_sources = normalized_sources(replacement)
        for name, weight in MULTIPLIERS.items():
            scored = replacement if weight == 1.0 else {**replacement, **weighted_fusion(model, replacement_sources, weight)}
            proposal = pack_atlas_candidates(scored, NOMINATIONS, baseline="v8")
            indices, valid, new = matched_pool(baseline, proposal)
            reordered = describe_atlas_candidates(scored, NOMINATIONS, baseline.baseline, baseline.rank_depth, indices, valid)
            selected = ranker.select(reordered, anchor_ocf=ANCHOR)
            if bool(((incumbent < 0) | (selected < 0)).any()):
                raise ValueError("Every query needs an eligible candidate before and after replacement.")
            diagnostics = {"same_unique_count_every_row": True,
                           "mean_candidates_per_glyph": float(valid.sum(dim=1).float().mean()),
                           "mean_replaced_candidates": float(new.float().mean()),
                           "rows_with_replacements": int((new > 0).sum()),
                           "selected_label_changes_vs_v14": int((selected != incumbent).sum())}
            groups[name][trial.session] = (ids, inherited_choice, incumbent.tolist(), selected.tolist(), diagnostics)
            print(json.dumps({"candidate": name, "session": trial.session, "queries": len(ids), **diagnostics}), flush=True)
    if len(scope) != TASK_SIZES[TASK] // 2:
        raise ValueError("The replacement scope must comprise exactly four recordings of mice1 Task 1.")
    return groups, scope


def package(name, replacements, changed_ids, sessions, base, scope):
    output = WORKING / f"kaggle_submission_{name}.zip"
    temporary = output.with_suffix(".tmp")
    try:
        row_count, changed = replace_submission_rows(base, temporary, replacements)
        if not 0 < changed == len(replacements) or set(replacements) - scope:
            raise ValueError("Expected changed predictions inside mice1 Task 1 only.")
        if verify_archive(base, temporary, replacements) != changed:
            raise ValueError("The complete candidate ZIP must contain exactly the listed replacements.")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    difference = WORKING / f"changed_task1_mice1_ids_{name}.json"
    difference.write_text(json.dumps(changed_ids, indent=2) + "\n", encoding="utf-8")
    return {"archive": output.name, "submission_sha256": sha256(output), "changed_ids_file": difference.name,
            "changed_ids_sha256": sha256(difference), "rows": row_count, "changed_predictions": changed,
            "unchanged_rows_verified_byte_for_byte": row_count - changed,
            "out_of_scope_rows_verified_byte_for_byte": row_count - len(scope),
            "poisson_multiplier": MULTIPLIERS[name], "sessions": sessions}


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a private Kaggle T4 and the verified v14 and v8 notebook outputs.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    base = mounted_submission(V14_SHA)
    inherited = mounted_submission(V8_SHA)
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    training = grouped_blocks(root, catalog, MOUSE, device, len(images))
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE
                      and entry["split"] == "test" and entry["task"] == TASK), key=lambda entry: entry["session"])
    if (len(images) != 2748 or tuple(training) != SESSIONS or tuple(entry["session"] for entry in entries) != SESSIONS
            or any(group.shape[:2] != (len(images), 4) for group in training.values())):
        raise ValueError("Expected all 2,748 labeled glyphs with four repeats in all four mice1 recordings.")
    rates = torch.stack([group.mean(1) for group in training.values()])
    n_features = rates.shape[-1]
    stabilized = (rates + 0.1).sqrt().transpose(0, 1).flatten(1)
    del rates

    cores, archive_sha256 = load_released_core()
    if archive_sha256 != CORE_SHA256:
        raise ValueError("The public MICrONS checkpoint must match the validated transfer run.")
    core = cores["published_core"].float().requires_grad_(False)
    del cores
    report = {"purpose": "Candidate preparation only", "base_sha256": V14_SHA, "inherited_v8_sha256": V8_SHA,
              "target": "mice1 task1", "target_rows": TASK_SIZES[TASK] // 2, "query_repeats": REPEATS,
              "fit_glyphs": len(images), "seed": MODEL_SEED,
              "original_ranker_anchor_ocf": ANCHOR, "original_nominations": NOMINATIONS,
              "same_distinct_candidate_budget": True, "every_replaced_row_recreated_v14_pixels": True,
              "validation_notebook": VALIDATION, "sources": {
                  "microns": {"code": FNN_SOURCE, "code_version": FNN_VERSION, "code_license": "MIT",
                              "code_license_url": f"{FNN_SOURCE}/blob/{FNN_VERSION}/LICENSE",
                              "checkpoint_url": PUBLIC_ARCHIVE, "checkpoint_member": "params_core.pt",
                              "archive_sha256": archive_sha256, "archive_expected_md5": OFFICIAL_MD5,
                              "checkpoint_license": "CC BY 4.0 per the BossDB MICrONS Minnie project listing",
                              "checkpoint_license_url": "https://bossdb.org/project/microns-minnie",
                              "checkpoint_documentation": "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
                                                          "functional_data/foundation_model/readme_v3.md"}},
              "core_inference": {"dtype": "float32", "frames_at_30hz": FRAMES, "dark_prefix_frames": 3,
                                 "common_black_level": BLACK_LEVEL, "stream": STREAM, "training_batch": CORE_BATCH,
                                 "atlas_chunk": ATLAS_CHUNK, "kernel_ridge": RIDGE, "single_psd_solve": True},
              "candidates": {}}
    publish(report, "source_verified", started)

    scorer = official_scorer(root)
    split = split_calibration(len(images), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    _, likelihood, folded_noise, codes = nested_ranker(training, images, split, scorer, MODEL_SEED)
    ranker, folds = fit_crossfit_head(training, images, scorer, codes)
    model = fit_v8(training, images, MODEL_SEED)
    del training
    if not torch.equal(model.codepoints.cpu(), codes):
        raise ValueError("The v14 calibration and both candidates must use the identical ordered public atlas.")
    report["sources"]["atlas"] = model.report["atlas"]
    report["sources"]["unihan"] = model.report["prior"]["unicode"]
    report["v14"] = {"fold_zero": folded_noise, "four_fold_calibration": folds, "ranker": ranker.diagnostics,
                     "outer": model.report}
    publish(report, "v14_ready", started)

    kernel_started = time.monotonic()
    readout = fit_kernel_readout(core, images, stabilized)
    core_time = time.monotonic() - kernel_started
    atlas_rates, milestones = stream_atlas(core, readout, model.atlas.images, n_features, ATLAS_CHUNK)
    report["core_inference"].update(fit_seconds=core_time, kernel_dimension=readout.anchors.shape[1],
                                    atlas_rate_shape=list(atlas_rates.shape), streaming_progress=milestones)
    del core, readout, images, stabilized
    publish(report, "microns_atlas_ready", started)

    bank = rendered_bank(model).cpu().numpy()
    groups, scope = predict(root, catalog, entries, model, ranker, likelihood, atlas_rates)
    stored = base_target_predictions(base, scope)
    legacy = base_target_predictions(inherited, scope)
    for name, blocks in groups.items():
        changes, changed_ids, sessions = select_replacements(blocks, stored, legacy, bank)
        report["candidates"][name] = package(name, changes, changed_ids, sessions, base, scope)
        publish(report, name + "_verified", started)
    publish(report, "complete", started)


if __name__ == "__main__":
    main()
