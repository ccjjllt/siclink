"""Package the held-validated v16 + frozen Hanzi policy over immutable v18."""

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
from kaggle_merge_candidate_outputs import mounted_submission, replace_submission_rows, sha256
from kaggle_task1_bipartite_validation import replace_poisson
from kaggle_task1_crossfit_candidate import fit_crossfit_policy
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED
from kaggle_task1_microns_candidate import CORE_SHA256
from kaggle_task1_microns_hanzi_recheck import ATLAS_SIZE, CONTOUR_OCR_WEIGHT, calibration_key, fit_ocr_atlas, with_hanzi
from kaggle_task1_microns_open_atlas import ATLAS_CHUNK, RIDGE, fit_kernel_readout, stream_atlas
from kaggle_task1_microns_transfer import BLACK_LEVEL, FNN_VERSION, FRAMES, PUBLIC_ARCHIVE, STREAM, load_released_core
from kaggle_task1_native_hanzi import (OCR_CHECKSUM, OCR_REVISION, OCR_SOURCE, PC_COUNT, RIDGE as OCR_RIDGE,
                                       STAGE, fit_response_ridge)
from kaggle_task1_poisson_candidate import MOUSE, REPEATS, TASK, select_replacements
from kaggle_task1_v11_repeat_weight_transfer import POISSON_AMPLIFICATION, matched_pool, normalized_sources, weighted_fusion
from kaggle_task1_v8_metric_update import V8_SHA, base_target_predictions
from kaggle_task3_density_candidate import verify_archive
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, rendered_bank


BASE_SHA = "c75f2802ca6c1b5a69cb6f1d60e701bb96a9017c90352473e89a915f048c26b6"
HELD_REPORT_SHA = "cb86f749504c0e4b47620c968bbeb20ec25d7a987d6e4583fb0af8751be90ffd"
WORKING = Path("/kaggle/working")
GALLERY_SIZE = 2748
REPORT = "source_candidates.json"


def verified_archive(path, expected):
    source = path.resolve(strict=True) if path is not None else mounted_submission(expected)
    if sha256(source) != expected:
        raise ValueError(f"The mounted archive {source} must match immutable SHA-256 {expected}.")
    return source


def publish(report, output, stage, started):
    report.update(stage=stage, seconds=time.monotonic() - started)
    temporary = output / ".source_candidates.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / REPORT)
    print(json.dumps({"stage": stage, "seconds": round(report["seconds"], 3)}), flush=True)


@torch.no_grad()
def predict(root, catalog, entries, model, ranker, noise, rates, ocr_decoder, ocr_prototypes, report, output, started):
    per_day = TASK_SIZES[TASK] // (2 * len(SESSIONS))
    device = model.atlas.images.device
    blocks, scope = {}, set()
    for position, entry in enumerate(entries):
        day = entry["session"]
        publish(report, output, f"{day}_started", started)
        query = read_trials(root, catalog, entry, device, load_labels=False)
        ids = tuple(query.ids)
        if (query.counts.shape[:2] != (per_day, REPEATS) or len(ids) != per_day or len(set(ids)) != per_day
                or scope.intersection(ids) or any(not sample.startswith("test_task1_pic") for sample in ids)):
            raise ValueError("Expected distinct official mice1 Task 1 groups of twenty measured repeats.")
        scope.update(ids)

        stock = atlas_scores(model, noise[day], query.counts, day)
        original = pack_atlas_candidates(stock, NOMINATIONS, baseline="v8")
        inherited = stock["v8"].argmax(1)
        templates = rates[position].to(device)
        if templates.shape != (len(model.atlas.images), query.counts.shape[-1]):
            raise ValueError("MICrONS rates must match the atlas columns and this recording's neural response bins.")
        direct = replace_poisson(model, stock, query.counts, templates)
        sources = normalized_sources(direct)
        v16 = {**direct, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}

        projection, center, scale = ocr_decoder[day]
        timed = timing_view(query.counts.cpu().unflatten(-1, (-1, 10))).mean(1)
        predicted = F.normalize(((timed - center) / scale) @ projection)
        ocr = standardized(predicted.to(device) @ ocr_prototypes)
        fusion = with_hanzi(model, v16, sources, ocr)

        selected, pools = {}, {}
        for name, scores in (("v16", v16), ("hanzi", fusion)):
            indices, valid, _ = matched_pool(original, pack_atlas_candidates(scores, NOMINATIONS, baseline="v8"))
            graph = describe_atlas_candidates(scores, NOMINATIONS, original.baseline, original.rank_depth, indices, valid)
            selected[name] = ranker.select(graph, anchor_ocf=ANCHOR)
            pools[name] = graph
        budget = original.valid.sum(1)
        if (any(not torch.equal(pool.valid.sum(1), budget) for pool in pools.values())
                or any(bool((choice < 0).any()) for choice in selected.values())):
            raise ValueError("The original v14 per-row distinct candidate budget must match both policies.")
        before, after = selected["v16"], selected["hanzi"]
        old, proposed = pools["v16"], pools["hanzi"]
        overlaps = ((proposed.indices[:, :, None] == old.indices[:, None, :]) & old.valid[:, None]).any(dim=2)
        novel = (~overlaps & proposed.valid).sum(1)
        audit = {"same_distinct_candidate_budget": True, "mean_candidate_count": float(budget.float().mean()),
                 "mean_new_candidates_vs_v16": float(novel.float().mean()), "rows_with_new_candidates": int((novel > 0).sum()),
                 "v16_inherited_v8_selections": int((before == inherited).sum()),
                 "hanzi_inherited_v8_selections": int((after == inherited).sum()),
                 "proposed_choice_changes": int((before != after).sum())}
        blocks[day] = (ids, inherited.tolist(), before.tolist(), after.tolist(), audit)
        report["inference_progress"][day] = {"queries": len(ids), **audit}
        publish(report, output, f"{day}_scored", started)
    if len(scope) != TASK_SIZES[TASK] // 2:
        raise ValueError("The patch scope must be the complete official mice1 Task 1 recordings.")
    return blocks, scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Extracted competition directory containing catalog.json")
    parser.add_argument("--base", type=Path, help="Exact verified v18 submission ZIP")
    parser.add_argument("--inherited-v8", type=Path, help="Exact v8 ZIP used by v16's anchor rendering rule")
    parser.add_argument("--output-dir", type=Path, default=WORKING)
    parser.add_argument("--calibration-cache", type=Path, help="Save and resume fit-only calibration folds in this directory")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this candidate on a remote Kaggle or Colab GPU.")
    started = time.monotonic()
    torch.set_num_threads(4)
    cpu, device = torch.device("cpu"), torch.device("cuda:0")
    output = args.output_dir.resolve()
    archive = output / "kaggle_submission.zip"
    base, inherited = verified_archive(args.base, BASE_SHA), verified_archive(args.inherited_v8, V8_SHA)
    if archive.resolve() in (base.resolve(), inherited.resolve()):
        raise ValueError("The output ZIP must have a directory distinct from both immutable sources.")
    output.mkdir(parents=True, exist_ok=True)
    root = args.data_root.resolve(strict=True) if args.data_root else locate_data()
    catalog = load_catalog(root)
    scorer = official_scorer(root)
    labeled_cpu = load_images(root, cpu)
    groups = grouped_blocks(root, catalog, MOUSE, cpu, len(labeled_cpu))
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE and entry["split"] == "test"
                      and entry["task"] == TASK), key=lambda entry: entry["session"])
    if (len(labeled_cpu) != GALLERY_SIZE or tuple(groups) != SESSIONS or tuple(entry["session"] for entry in entries) != SESSIONS
            or any(counts.shape[:2] != (GALLERY_SIZE, 4) for counts in groups.values())):
        raise ValueError("Expected the complete labeled gallery and all four official mice1 recordings.")

    cache_key, cache_dir = None, None
    if args.calibration_cache is not None:
        cache_key = calibration_key(torch.arange(GALLERY_SIZE), torch.empty(0, dtype=torch.long), labeled_cpu, groups,
                                    scorer, device)
        cache_dir = args.calibration_cache.resolve() / cache_key
        print(f"Calibration snapshots: {cache_dir}/fold-*.pt", flush=True)
    training = {day: counts.to(device) for day, counts in groups.items()}
    labeled = labeled_cpu.to(device)
    stabilized = (torch.stack([counts.mean(1) for counts in training.values()]) + 0.1).sqrt().transpose(0, 1).flatten(1)
    feature_count = next(iter(training.values())).shape[-1]
    report = {"candidate": "v19-mice1-hanzi", "base_sha256": BASE_SHA, "inherited_v8_sha256": V8_SHA,
              "held_validation_report_sha256": HELD_REPORT_SHA, "target": "mice1 task1", "target_rows": TASK_SIZES[TASK] // 2,
              "fit_glyphs": GALLERY_SIZE, "source_fit_repeats": 4, "formal_query_repeats": REPEATS,
              "formal_labels_loaded": False, "seed": MODEL_SEED, "anchor_ocf": ANCHOR, "nominations": NOMINATIONS,
              "poisson_multiplier": POISSON_AMPLIFICATION, "fixed_ocr_share_in_decoded_contour": CONTOUR_OCR_WEIGHT,
              "same_v14_distinct_candidate_budget": True, "inference_progress": {},
              "microns": {"code_version": FNN_VERSION, "archive": PUBLIC_ARCHIVE, "sha256": CORE_SHA256,
                           "frames": FRAMES, "stream": STREAM, "black_level": BLACK_LEVEL, "kernel_ridge": RIDGE,
                           "atlas_chunk": ATLAS_CHUNK},
              "ocr": {"source": OCR_SOURCE, "revision": OCR_REVISION, "sha256": OCR_CHECKSUM, "stage": STAGE,
                      "pca_components": PC_COUNT, "ridge_per_fit_glyph": OCR_RIDGE}}
    if cache_dir is not None:
        report["calibration_cache"] = {"key": cache_key, "directory": str(cache_dir)}
    publish(report, output, "inputs_verified", started)

    publish(report, output, "microns_checkpoint_loading", started)
    cores, checkpoint = load_released_core(work_dir=None)
    if checkpoint != CORE_SHA256:
        raise ValueError("The pretrained core must exactly match the checkpoint used by deployed v16.")
    core = cores["published_core"].float().requires_grad_(False)
    del cores
    publish(report, output, "fourfold_started", started)
    ranker, folds, noise, codepoints = fit_crossfit_policy(training, labeled, scorer, cache_dir=cache_dir, cache_key=cache_key)
    report["v14_fourfold"] = {"folds": folds, "ranker": ranker.diagnostics, "inference_noise": "literal fold zero"}
    publish(report, output, "fourfold_ready", started)

    publish(report, output, "outer_model_started", started)
    model = fit_v8(training, labeled, MODEL_SEED)
    del training
    if len(codepoints) != ATLAS_SIZE or not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("The outer model and calibration folds must use the exact same public atlas order.")
    font, unihan = model.report["atlas"]["sha256"], model.report["prior"]["unicode"]["sha256"]
    for fold in folds.values():
        source = fold["atlas_and_font_style"]
        if (source["atlas"]["sha256"], source["prior"]["unicode"]["sha256"]) != (font, unihan):
            raise ValueError("Every fold and the outer model must use identical public atlas sources.")
    rendered = rendered_bank(model).cpu().contiguous().numpy()
    report["v14_outer"] = {**model.report, "codepoint_order_sha256": hash_bytes(codepoints.numpy().tobytes()).hexdigest(),
                           "styled_rendering_sha256": hash_bytes(rendered.tobytes()).hexdigest()}
    publish(report, output, "outer_model_ready", started)

    publish(report, output, "microns_rates_started", started)
    readout = fit_kernel_readout(core, labeled, stabilized)
    rates, milestones = stream_atlas(core, readout, model.atlas.images, feature_count, ATLAS_CHUNK)
    del core, readout, stabilized
    report["microns"].update(atlas_rate_shape=list(rates.shape), streaming_progress=milestones)
    publish(report, output, "microns_rates_ready", started)

    publish(report, output, "hanzi_started", started)
    coordinates, prototypes = fit_ocr_atlas(labeled_cpu, model.atlas.images)
    ocr_decoder = {}
    for day, counts in groups.items():
        observation = timing_view(counts.unflatten(-1, (-1, 10))).mean(1)
        ridge, center, scale = fit_response_ridge(observation)
        ocr_decoder[day] = (ridge @ coordinates, center, scale)
    del labeled, labeled_cpu, groups, coordinates
    publish(report, output, "hanzi_ready", started)

    blocks, scope = predict(root, catalog, entries, model, ranker, noise, rates, ocr_decoder, prototypes, report, output, started)
    del rates
    patch, changed_ids, sessions = select_replacements(blocks, base_target_predictions(base, scope),
                                                      base_target_predictions(inherited, scope), rendered)
    for session in sessions.values():
        session["nominal_v16_recreated_v18_pixels"] = session.pop("reproduced_v14_pixels")
        session["unrecreated_v18_rows_preserved"] = session.pop("unreproduced_v14_rows_preserved")
    report["sessions"] = sessions
    publish(report, output, "archive_verification_started", started)

    with TemporaryDirectory(prefix=".mice1-hanzi-", dir=output) as folder:
        temporary = Path(folder)
        staged_archive = temporary / archive.name
        rows, changed = replace_submission_rows(base, staged_archive, patch)
        if (rows != 30_200 or not 0 < changed == len(patch) or patch.keys() - scope
                or verify_archive(base, staged_archive, patch) != changed):
            raise ValueError("The complete candidate must contain only pixel-protected mice1 Task 1 changes.")
        ids_path = temporary / "changed_task1_mice1_ids.json"
        ids_path.write_text(json.dumps(changed_ids, indent=2) + "\n", encoding="utf-8")
        report.update(stage="complete", seconds=time.monotonic() - started, submission_sha256=sha256(staged_archive),
                      changed_ids_file=ids_path.name, changed_ids_sha256=sha256(ids_path), rows=rows,
                      changed_predictions=changed, unchanged_rows_verified_byte_exact=rows - changed,
                      outside_scope_verified_byte_exact=rows - len(scope),
                      nominal_v16_recreated_v18_pixels=sum(day["nominal_v16_recreated_v18_pixels"] for day in sessions.values()),
                      every_changed_row_replayed_v16_pixels=True, renderer="v16 inherited v8 anchor or styled public atlas",
                      archive_verification="official IDs and ordering, exact untouched CSV bytes, ZIP CRC, 64x64 grayscale PNG")
        manifest = temporary / REPORT
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for path in (staged_archive, ids_path, manifest):
            path.replace(output / path.name)
    print(json.dumps({"stage": "complete", "seconds": report["seconds"], "submission_sha256": report["submission_sha256"],
                      "changed_predictions": changed, "replayed_pixels": report["nominal_v16_recreated_v18_pixels"]}), flush=True)


if __name__ == "__main__":
    main()
