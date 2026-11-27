"""Package the PP-OCRv6 mice2 Task 1 branch over the immutable v20 archive."""

import argparse
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import time

import torch
from torch.nn import functional as F

from glyph_atlas_metric_ranker import describe_atlas_candidates, pack_atlas_candidates
from glyph_bipartite_encoding import timing_view
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from kaggle_baseline import TASK_SIZES
from kaggle_gallery_validation import standardized
from kaggle_merge_candidate_outputs import replace_submission_rows, sha256
from kaggle_task1_crossfit_metric_scale import fold_indices
from kaggle_task1_metric_scale import MODEL_SEED
from kaggle_task1_mice2_crossfit_candidate import MOUSE, QUERY_REPEATS, TASK
from kaggle_task1_mice2_poisson_candidate import protected_replacements
from kaggle_task1_mice2_repeat_matched_ranker import model_of, stock_folds
from kaggle_task1_microns_hanzi_candidate import GALLERY_SIZE, REPORT, WORKING, publish, verified_archive
from kaggle_task1_microns_hanzi_recheck import ATLAS_SIZE, CONTOUR_OCR_WEIGHT, fit_ocr_atlas, with_hanzi
from kaggle_task1_native_hanzi import OCR_CHECKSUM, OCR_REVISION, OCR_SOURCE, fit_response_ridge
from kaggle_task1_ppocrv6_readout import fit_coordinates, read_glyphs, recognizer
from kaggle_task1_v11_repeat_weight_transfer import (ANCHOR_OCF, POISSON_AMPLIFICATION, matched_pool, normalized_sources,
                                                      weighted_fusion)
from kaggle_task3_density_candidate import stored_predictions, verify_archive
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, rendered_bank


BASE_SHA = "b098c3bc62aa96b289b72c512e977fa246fb37f3ddfdab32ffe786d3e7ba8281"
CACHE_KEY = "0689d58d2efed18028eb6a4db4fbc727b4e8fced9827efd80dc14e83a28e7c04"
FOLD_SHA = (
    "8549fa26a7867aeaad409c9b1f4136a573c7633e8c31869b8620a2d8a71720c4",
    "05c8b012c5ce1d532aa8a214e62af318572e8ed8a82626713fb6224bfa831816",
    "0d8f11b028014a17c0a710575abfa32a5ad777a83fe924ab8898773ab3940628",
    "e6bbd7ac68b009bc3c0c8b2add80e19ae707071d139b464409b15665218bff11",
)
CODEPOINT_SHA = "6959180f8533d4a2342b8de6d0e585c2e0c0f31b6aee605e72e639b9bf26ae28"
RENDERED_SHA = "3367df84d3740b8d53d904f42613119ed7ee0679ea01be02512fdb9a20e6ce91"
FONT_SHA = "2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b"
UNIHAN_SHA = "f7a48b2b545acfaa77b2d607ae28747404ce02baefee16396c5d2d7a8ef34b5e"
PPOCR_SHARE = 0.5


def verified_folds(path):
    if path is None:
        candidates = sorted(Path("/kaggle/input").rglob(CACHE_KEY))
    else:
        path = path.resolve(strict=True)
        candidates = (path, path / CACHE_KEY)
    for candidate in candidates:
        files = [candidate / f"fold-{fold}.pt" for fold in range(len(FOLD_SHA))]
        if all(file.is_file() for file in files) and tuple(sha256(file) for file in files) == FOLD_SHA:
            return candidate
    raise FileNotFoundError("Supply the four unchanged full-fit v20 calibration folds using --calibration-cache.")


def restore_policy(cache, device):
    # The original cache key includes source files and CUDA runtime. Authenticate the snapshots
    # directly so the archived fold scope is preserved on either backend.
    noise_rows, splits = fold_indices(GALLERY_SIZE, torch.device("cpu"))
    folds, hashes = stock_folds(cache, splits, noise_rows, CACHE_KEY, device)
    codes = folds[0][3]
    if (len(codes) != ATLAS_SIZE or any(not torch.equal(codes, fold[3]) for fold in folds)
            or hash_bytes(codes.numpy().tobytes()).hexdigest() != CODEPOINT_SHA):
        raise ValueError("The archived folds must share the verified v20 public atlas order.")
    ranker = model_of([fold[0] for fold in folds], [fold[1] for fold in folds])
    return ranker, folds[0][2], codes, tuple(hashes)


@torch.no_grad()
def fit_decoders(groups, labeled, atlas, report, output, started):
    stock_labels, stock_bank = fit_ocr_atlas(labeled, atlas)
    publish(report, output, "stock_ocr_ready", started)
    session, metadata = recognizer()
    raster = torch.cat((labeled, atlas.cpu()))
    encoded = read_glyphs(session, raster, metadata)
    pp_labels, pp_bank = fit_coordinates(encoded["ppocrv6_global"], len(labeled))
    metadata["pca_dimensions"] = pp_labels.shape[1]
    report["ppocrv6"] = metadata
    decoders = {}
    for day, counts in groups.items():
        observation = timing_view(counts.unflatten(-1, (-1, 10))).mean(1)
        ridge, center, scale = fit_response_ridge(observation)
        decoders[day] = (ridge @ stock_labels, ridge @ pp_labels, center, scale)
    return decoders, stock_bank, pp_bank


@torch.no_grad()
def score_day(model, ranker, noise, day, counts, decoder, stock_bank, pp_bank):
    stock = atlas_scores(model, noise, counts, day)
    ordinary = pack_atlas_candidates(stock, NOMINATIONS, baseline="v8")
    sources = normalized_sources(stock)
    amplified = {**stock, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}
    native_projector, pp_projector, center, scale = decoder
    timed = timing_view(counts.cpu().unflatten(-1, (-1, 10))).mean(1)
    observation = (timed - center) / scale
    native = standardized(F.normalize(observation @ native_projector).to(stock_bank.device) @ stock_bank)
    pp = standardized(F.normalize(observation @ pp_projector) @ pp_bank).to(native.device)
    combined = (1 - PPOCR_SHARE) * standardized(native) + PPOCR_SHARE * standardized(pp)
    choices, graphs = [], []
    for evidence in (native, combined):
        scores = with_hanzi(model, amplified, sources, evidence)
        indices, valid, _ = matched_pool(ordinary, pack_atlas_candidates(scores, NOMINATIONS, baseline="v8"))
        graph = describe_atlas_candidates(scores, NOMINATIONS, ordinary.baseline, ordinary.rank_depth, indices, valid)
        graphs.append(graph)
        choices.append(ranker.select(graph, anchor_ocf=ANCHOR_OCF))
    budget = ordinary.valid.sum(1)
    if any(not torch.equal(graph.valid.sum(1), budget) for graph in graphs) or any(
            bool((choice < 0).any()) for choice in choices):
        raise ValueError("Keep the original per-query distinct candidate count for both v20 and PP-OCRv6.")
    original, proposal = graphs
    common = ((proposal.indices[:, :, None] == original.indices[:, None, :]) & original.valid[:, None]).any(2)
    novel = (~common & proposal.valid).sum(1)
    return choices[0].tolist(), choices[1].tolist(), {
        "same_distinct_candidate_budget": True, "mean_candidate_count": float(budget.float().mean()),
        "mean_new_candidates_vs_v20": float(novel.float().mean()),
        "rows_with_new_candidates": int((novel > 0).sum()),
        "proposed_choice_changes": int((choices[0] != choices[1]).sum()),
    }


def predict(root, catalog, model, ranker, noise, decoders, stock_bank, pp_bank, report, output, started):
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE and entry["split"] == "test"
                      and entry["task"] == TASK), key=lambda entry: entry["session"])
    if tuple(entry["session"] for entry in entries) != SESSIONS:
        raise ValueError("Expected the four official mice2 Task 1 recording blocks.")
    per_day = TASK_SIZES[TASK] // (2 * len(SESSIONS))
    device = model.atlas.images.device
    scope, blocks = set(), {}
    for entry in entries:
        day = entry["session"]
        query = read_trials(root, catalog, entry, device, load_labels=False)
        ids = tuple(map(str, query.ids))
        if (query.counts.shape[:2] != (per_day, QUERY_REPEATS) or len(ids) != per_day or len(set(ids)) != per_day
                or scope.intersection(ids) or any(not sample.startswith("test_task1_pic") for sample in ids)):
            raise ValueError("Expected distinct official mice2 Task 1 rows with twenty measured repeats.")
        scope.update(ids)
        previous, proposed, diagnostics = score_day(model, ranker, noise[day], day, query.counts, decoders[day],
                                                    stock_bank, pp_bank)
        blocks[day] = (ids, previous, proposed, diagnostics)
        report["inference_progress"][day] = {"queries": len(ids), **diagnostics}
        publish(report, output, f"{day}_scored", started)
    return blocks, scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Extracted competition dataset containing catalog.json")
    parser.add_argument("--base", type=Path, help="Exact v20 ZIP to patch")
    parser.add_argument("--calibration-cache", type=Path,
                        help="Immutable full-fit v20 four-fold directory or its parent")
    parser.add_argument("--output-dir", type=Path, default=WORKING)
    parser.add_argument("--device", default="cuda:0", help="CUDA (default) or CPU with pixel-protected replay")
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    cpu, device = torch.device("cpu"), torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Choose CPU or an available remote CUDA device.")
    base = verified_archive(args.base, BASE_SHA)
    cache = verified_folds(args.calibration_cache)
    output = args.output_dir.resolve()
    archive = output / "kaggle_submission.zip"
    if archive.resolve() == base.resolve():
        raise ValueError("Choose an output directory separate from the immutable v20 ZIP.")
    output.mkdir(parents=True, exist_ok=True)
    root = args.data_root.resolve(strict=True) if args.data_root else locate_data()
    catalog = load_catalog(root)
    labeled = load_images(root, cpu)
    groups = grouped_blocks(root, catalog, MOUSE, cpu, len(labeled))
    if (labeled.shape != (GALLERY_SIZE, 64, 64) or labeled.dtype != torch.uint8 or tuple(groups) != SESSIONS
            or any(counts.shape[:2] != (GALLERY_SIZE, 4) for counts in groups.values())):
        raise ValueError("Expected all labeled v20 gallery glyphs and four complete training recordings.")
    report = {
        "candidate": "mice2-task1-v20-ppocrv6", "base_sha256": BASE_SHA, "target": "mice2 task1",
        "device": str(device), "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "seed": MODEL_SEED, "anchor_ocf": ANCHOR_OCF, "nominations": NOMINATIONS,
        "fit_glyphs": len(labeled), "source_fit_repeats": 4, "formal_query_repeats": QUERY_REPEATS,
        "formal_labels_loaded": False, "poisson_multiplier": POISSON_AMPLIFICATION,
        "stock_ocr": {"source": OCR_SOURCE, "revision": OCR_REVISION, "sha256": OCR_CHECKSUM},
        "fixed_ocr_share_in_decoded_contour": CONTOUR_OCR_WEIGHT, "ppocrv6_share_in_ocr_branch": PPOCR_SHARE,
        "calibration_cache": {"key": CACHE_KEY, "directory": str(cache)}, "inference_progress": {},
    }
    publish(report, output, "inputs_verified", started)
    ranker, noise, codes, hashes = restore_policy(cache, device)
    report["calibration_cache"].update(sha256=hashes, ranker=ranker.diagnostics,
                                       inference_noise="archived full-fit fold zero")
    publish(report, output, "v20_ranker_ready", started)
    training = {day: counts.to(device) for day, counts in groups.items()}
    model = fit_v8(training, labeled.to(device), MODEL_SEED)
    del training
    if not torch.equal(model.codepoints.cpu(), codes):
        raise ValueError("The outer v20 scorer must use the exact calibrated public atlas order.")
    if (model.report["atlas"]["sha256"], model.report["prior"]["unicode"]["sha256"]) != (FONT_SHA, UNIHAN_SHA):
        raise ValueError("Use the original v20 font and Unihan assets.")
    rendered = rendered_bank(model).cpu().contiguous().numpy()
    rendering_hash = hash_bytes(rendered.tobytes()).hexdigest()
    report["v20_outer"] = {**model.report, "codepoint_order_sha256": CODEPOINT_SHA,
                           "styled_rendering_sha256": rendering_hash, "archived_styled_rendering_sha256": RENDERED_SHA,
                           "recreated_full_atlas_pixels": rendering_hash == RENDERED_SHA}
    publish(report, output, "v20_outer_ready", started)
    decoders, stock_bank, pp_bank = fit_decoders(groups, labeled, model.atlas.images, report, output, started)
    del groups, labeled
    publish(report, output, "ppocrv6_ready", started)
    blocks, scope = predict(root, catalog, model, ranker, noise, decoders, stock_bank, pp_bank, report, output, started)
    updates, changed_ids, sessions = protected_replacements(blocks, stored_predictions(base, scope), rendered,
                                                           base_policy="v20")
    report["sessions"] = sessions
    publish(report, output, "archive_verification_started", started)
    with TemporaryDirectory(prefix=".mice2-ppocrv6-", dir=output) as folder:
        staging = Path(folder)
        staged = staging / archive.name
        if updates:
            rows, changed = replace_submission_rows(base, staged, updates)
        else:
            copyfile(base, staged)
            rows, changed = sum(TASK_SIZES.values()), 0
        if (len(scope) != TASK_SIZES[TASK] // 2 or rows != sum(TASK_SIZES.values()) or changed != len(updates)
                or updates.keys() - scope or verify_archive(base, staged, updates) != changed):
            raise ValueError("Only replay-protected mice2 Task 1 predictions may differ from v20.")
        ids_path = staging / "changed_task1_mice2_ids.json"
        ids_path.write_text(json.dumps(changed_ids, indent=2) + "\n", encoding="utf-8")
        report.update(stage="complete", seconds=time.monotonic() - started, submission_sha256=sha256(staged),
                      rows=rows, target_rows=len(scope), changed_predictions=changed, changed_ids_file=ids_path.name,
                      changed_ids_sha256=sha256(ids_path), unchanged_rows_verified_byte_exact=rows - changed,
                      outside_scope_verified_byte_exact=rows - len(scope),
                      nominal_v20_recreated_base_pixels=sum(day["nominal_v20_recreated_base_pixels"]
                                                           for day in sessions.values()),
                      every_changed_row_recreated_v20_base_pixels=True)
        manifest = staging / REPORT
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for path in (staged, ids_path, manifest):
            path.replace(output / path.name)
    print(json.dumps({"stage": "complete", "submission_sha256": report["submission_sha256"],
                      "changed_predictions": changed, "replayed_pixels": report["nominal_v20_recreated_base_pixels"]}), flush=True)


if __name__ == "__main__":
    main()
