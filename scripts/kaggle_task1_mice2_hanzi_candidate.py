"""Package the held-validated v18 + frozen Hanzi mice2 policy over immutable v19."""

import argparse
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch
from torch.nn import functional as F

from glyph_atlas_metric_ranker import describe_atlas_candidates, pack_atlas_candidates
from glyph_bipartite_encoding import timing_view
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from kaggle_baseline import TASK_SIZES
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standardized
from kaggle_merge_candidate_outputs import replace_submission_rows, sha256
from kaggle_task1_crossfit_candidate import fit_crossfit_policy
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED
from kaggle_task1_mice2_crossfit_candidate import MOUSE, QUERY_REPEATS, TASK
from kaggle_task1_mice2_poisson_candidate import protected_replacements
from kaggle_task1_microns_hanzi_candidate import GALLERY_SIZE, REPORT, WORKING, publish, verified_archive
from kaggle_task1_microns_hanzi_recheck import ATLAS_SIZE, CONTOUR_OCR_WEIGHT, calibration_key, fit_ocr_atlas, with_hanzi
from kaggle_task1_native_hanzi import (OCR_CHECKSUM, OCR_REVISION, OCR_SOURCE, PC_COUNT, RIDGE as OCR_RIDGE,
                                       STAGE, fit_response_ridge)
from kaggle_task1_v11_repeat_weight_transfer import (ANCHOR_OCF, POISSON_AMPLIFICATION, matched_pool, normalized_sources,
                                                      weighted_fusion)
from kaggle_task3_density_candidate import stored_predictions, verify_archive
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, rendered_bank


BASE_SHA = "7f3f483be15f28d6be707783a2fcedaa65b49473eea8020a040edf6e5c768fed"
HELD_REPORT_SHA = "07adc19e2f9e7a025890cb4f91866807643d881b5abf8e351906dc3773b50ea0"


@torch.no_grad()
def predict(root, catalog, entries, model, ranker, noise, ocr_decoder, prototypes, report, output, started):
    per_day = TASK_SIZES[TASK] // (2 * len(SESSIONS))
    device = model.atlas.images.device
    blocks, scope = {}, set()
    for entry in entries:
        day = entry["session"]
        publish(report, output, f"{day}_started", started)
        query = read_trials(root, catalog, entry, device, load_labels=False)
        ids = tuple(query.ids)
        if (query.counts.shape[:2] != (per_day, QUERY_REPEATS) or len(ids) != per_day or len(set(ids)) != per_day
                or scope.intersection(ids) or any(not sample.startswith("test_task1_pic") for sample in ids)):
            raise ValueError("Expected distinct official mice2 Task 1 groups of twenty measured repeats.")
        scope.update(ids)

        stock = atlas_scores(model, noise[day], query.counts, day)
        original = pack_atlas_candidates(stock, NOMINATIONS, baseline="v8")
        sources = normalized_sources(stock)
        v18 = {**stock, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}
        projection, center, scale = ocr_decoder[day]
        timed = timing_view(query.counts.cpu().unflatten(-1, (-1, 10))).mean(1)
        decoded = F.normalize(((timed - center) / scale) @ projection)
        ocr = standardized(decoded.to(device) @ prototypes)
        fusion = with_hanzi(model, v18, sources, ocr)

        selected, pools = {}, {}
        for name, scores in (("v18", v18), ("hanzi", fusion)):
            indices, valid, _ = matched_pool(original, pack_atlas_candidates(scores, NOMINATIONS, baseline="v8"))
            graph = describe_atlas_candidates(scores, NOMINATIONS, original.baseline, original.rank_depth, indices, valid)
            selected[name] = ranker.select(graph, anchor_ocf=ANCHOR_OCF)
            pools[name] = graph
        budget = original.valid.sum(1)
        if (any(not torch.equal(pool.valid.sum(1), budget) for pool in pools.values())
                or any(bool((choice < 0).any()) for choice in selected.values())):
            raise ValueError("Both policies must retain v14's exact per-query distinct candidate count.")
        before, after = selected["v18"], selected["hanzi"]
        old, proposed = pools["v18"], pools["hanzi"]
        common = ((proposed.indices[:, :, None] == old.indices[:, None, :]) & old.valid[:, None]).any(dim=2)
        novel = (~common & proposed.valid).sum(1)
        audit = {"same_distinct_candidate_budget": True, "mean_candidate_count": float(budget.float().mean()),
                 "mean_new_candidates_vs_v18": float(novel.float().mean()), "rows_with_new_candidates": int((novel > 0).sum()),
                 "proposed_choice_changes": int((before != after).sum())}
        blocks[day] = (ids, before.tolist(), after.tolist(), audit)
        report["inference_progress"][day] = {"queries": len(ids), **audit}
        publish(report, output, f"{day}_scored", started)
    if len(scope) != TASK_SIZES[TASK] // 2:
        raise ValueError("The patch scope must contain all four complete official mice2 Task 1 recordings.")
    return blocks, scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Extracted competition directory containing catalog.json")
    parser.add_argument("--base", type=Path, help="Exact verified v19 submission ZIP")
    parser.add_argument("--output-dir", type=Path, default=WORKING)
    parser.add_argument("--calibration-cache", type=Path, help="Save and resume fit-only calibration folds in this directory")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this candidate on a remote Kaggle or Colab GPU.")
    if ANCHOR != ANCHOR_OCF:
        raise ValueError("The replay must use the exact v18/v14 ranker anchor.")
    started = time.monotonic()
    torch.set_num_threads(4)
    cpu, device = torch.device("cpu"), torch.device("cuda:0")
    output = args.output_dir.resolve()
    archive = output / "kaggle_submission.zip"
    base = verified_archive(args.base, BASE_SHA)
    if archive.resolve() == base.resolve():
        raise ValueError("The output ZIP must have a directory distinct from immutable v19.")
    output.mkdir(parents=True, exist_ok=True)
    root = args.data_root.resolve(strict=True) if args.data_root else locate_data()
    catalog, scorer = load_catalog(root), official_scorer(root)
    labeled_cpu = load_images(root, cpu)
    groups = grouped_blocks(root, catalog, MOUSE, cpu, len(labeled_cpu))
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE and entry["split"] == "test"
                      and entry["task"] == TASK), key=lambda entry: entry["session"])
    if (len(labeled_cpu) != GALLERY_SIZE or tuple(groups) != SESSIONS or tuple(entry["session"] for entry in entries) != SESSIONS
            or any(counts.shape[:2] != (GALLERY_SIZE, 4) for counts in groups.values())):
        raise ValueError("Expected the full labeled gallery and official mice2 Task 1 recordings.")
    cache_key, cache_dir = None, None
    if args.calibration_cache is not None:
        cache_key = calibration_key(torch.arange(GALLERY_SIZE), torch.empty(0, dtype=torch.long), labeled_cpu, groups,
                                    scorer, device, mouse=MOUSE)
        cache_dir = args.calibration_cache.resolve() / cache_key
        print(f"Calibration snapshots: {cache_dir}/fold-*.pt", flush=True)
    training = {day: counts.to(device) for day, counts in groups.items()}
    labeled = labeled_cpu.to(device)
    report = {"candidate": "v20-mice2-hanzi", "base_sha256": BASE_SHA, "held_validation_report_sha256": HELD_REPORT_SHA,
              "nominal_policy": "v18 original SiClink Poisson 1.4x", "target": "mice2 task1", "target_rows": TASK_SIZES[TASK] // 2,
              "fit_glyphs": GALLERY_SIZE, "source_fit_repeats": 4, "formal_query_repeats": QUERY_REPEATS,
              "formal_labels_loaded": False, "microns_features_used": False, "seed": MODEL_SEED, "anchor_ocf": ANCHOR_OCF,
              "nominations": NOMINATIONS, "poisson_multiplier": POISSON_AMPLIFICATION,
              "fixed_ocr_share_in_decoded_contour": CONTOUR_OCR_WEIGHT, "same_v14_distinct_candidate_budget": True,
              "inference_progress": {}, "ocr": {"source": OCR_SOURCE, "revision": OCR_REVISION, "sha256": OCR_CHECKSUM,
                                                "stage": STAGE, "pca_components": PC_COUNT, "ridge_per_fit_glyph": OCR_RIDGE}}
    if cache_dir is not None:
        report["calibration_cache"] = {"key": cache_key, "directory": str(cache_dir)}
    publish(report, output, "inputs_verified", started)
    publish(report, output, "fourfold_started", started)
    ranker, folds, noise, codepoints = fit_crossfit_policy(training, labeled, scorer, cache_dir=cache_dir, cache_key=cache_key)
    report["v14_fourfold"] = {"folds": folds, "ranker": ranker.diagnostics, "inference_noise": "literal fold zero"}
    publish(report, output, "fourfold_ready", started)

    publish(report, output, "outer_model_started", started)
    model = fit_v8(training, labeled, MODEL_SEED)
    del training, labeled
    if len(codepoints) != ATLAS_SIZE or not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("The outer model and calibration folds must share the exact public atlas order.")
    font, unihan = model.report["atlas"]["sha256"], model.report["prior"]["unicode"]["sha256"]
    for fold in folds.values():
        source = fold["atlas_and_font_style"]
        if (source["atlas"]["sha256"], source["prior"]["unicode"]["sha256"]) != (font, unihan):
            raise ValueError("Every fold and the outer model must share the same public font and Unihan sources.")
    rendered = rendered_bank(model).cpu().contiguous().numpy()
    report["v14_outer"] = {**model.report, "codepoint_order_sha256": hash_bytes(codepoints.numpy().tobytes()).hexdigest(),
                           "styled_rendering_sha256": hash_bytes(rendered.tobytes()).hexdigest()}
    publish(report, output, "outer_model_ready", started)

    publish(report, output, "hanzi_started", started)
    coordinates, prototypes = fit_ocr_atlas(labeled_cpu, model.atlas.images)
    ocr_decoder = {}
    for day, counts in groups.items():
        observation = timing_view(counts.unflatten(-1, (-1, 10))).mean(1)
        ridge, center, scale = fit_response_ridge(observation)
        ocr_decoder[day] = (ridge @ coordinates, center, scale)
    del labeled_cpu, groups, coordinates
    publish(report, output, "hanzi_ready", started)

    blocks, scope = predict(root, catalog, entries, model, ranker, noise, ocr_decoder, prototypes, report, output, started)
    updates, changed_ids, sessions = protected_replacements(blocks, stored_predictions(base, scope), rendered, base_policy="v18")
    report["sessions"] = sessions
    publish(report, output, "archive_verification_started", started)
    with TemporaryDirectory(prefix=".mice2-hanzi-", dir=output) as folder:
        temporary = Path(folder)
        staged_archive = temporary / archive.name
        rows, changed = replace_submission_rows(base, staged_archive, updates)
        if (rows != 30_200 or changed != len(updates) or updates.keys() - scope
                or verify_archive(base, staged_archive, updates) != changed):
            raise ValueError("The complete candidate must preserve all other v19 rows and only change replay-protected mice2 rows.")
        ids_path = temporary / "changed_task1_mice2_ids.json"
        ids_path.write_text(json.dumps(changed_ids, indent=2) + "\n", encoding="utf-8")
        report.update(stage="complete", seconds=time.monotonic() - started, submission_sha256=sha256(staged_archive),
                      rows=rows, changed_predictions=changed, changed_ids_file=ids_path.name, changed_ids_sha256=sha256(ids_path),
                      unchanged_rows_verified_byte_exact=rows - changed, outside_scope_verified_byte_exact=rows - len(scope),
                      nominal_v18_recreated_base_pixels=sum(day["nominal_v18_recreated_base_pixels"] for day in sessions.values()),
                      every_changed_row_recreated_nominal_v18_base_pixels=True, renderer="v18 styled public atlas",
                      archive_verification="official IDs and ordering, exact untouched CSV rows, ZIP CRC, 64x64 grayscale PNG")
        manifest = temporary / REPORT
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for path in (staged_archive, ids_path, manifest):
            path.replace(output / path.name)
    print(json.dumps({"stage": "complete", "seconds": report["seconds"], "submission_sha256": report["submission_sha256"],
                      "changed_predictions": changed, "replayed_pixels": report["nominal_v18_recreated_base_pixels"]}), flush=True)


if __name__ == "__main__":
    main()
