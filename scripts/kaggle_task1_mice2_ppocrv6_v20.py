"""Frozen v20 held comparison of the published PP-OCRv6 global glyph representation."""

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_predictive_likelihood import fit_predictive_likelihood
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs, standardized
from kaggle_task1_bipartite_validation import image_proxy
from kaggle_task1_crossfit_metric_scale import fold_indices
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED
from kaggle_task1_mice2_repeat_matched_cpu import HELD_GLYPHS as REPEAT_COUNT
from kaggle_task1_mice2_repeat_matched_ranker import MOUSE, SEED as REPEAT_SEED, model_of, stock_folds
from kaggle_task1_mice2_textverifier_recheck import fresh_cohort, tensor_hash
from kaggle_task1_microns_hanzi_recheck import FIXED, FULL, NOMINATE, V14, V16, evaluate, fit_ocr_logits, with_hanzi
from kaggle_task1_native_hanzi import OCR_CHECKSUM, OCR_REVISION, response_views
from kaggle_task1_native_witness_utility import QUERY_GLYPHS as WITNESS_COUNT, SEED as WITNESS_SEED
from kaggle_task1_ppocrv6_readout import SEED as PPOCR_PILOT_SEED, distinct_cohort, fit_coordinates, read_glyphs, recognizer
from kaggle_task1_shared_nominations import SEED as SHARE_SEED, evaluation_cohort
from kaggle_task1_textverifier import publish
from kaggle_task1_v11_repeat_weight_transfer import POISSON_AMPLIFICATION, normalized_sources, pooled_scores, weighted_fusion
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, rendered_bank, summarize


SEED = 224679
TOTAL, FRESH = 128, 32
OUTPUT = Path("/kaggle/working/task1_mice2_ppocrv6_v20.json")
ARMS = {V14: "v14_reference", V16: "v20", FULL: "v20_ppocrv6", FIXED: "fixed_pool", NOMINATE: "nomination_only"}


def balanced_panel(outer, size, rng):
    positions = np.arange(len(outer))
    repeat = np.random.default_rng(REPEAT_SEED + 99).choice(positions, size=REPEAT_COUNT, replace=False)
    never_repeat = np.setdiff1d(positions, repeat)
    witness = np.random.default_rng(WITNESS_SEED).choice(never_repeat, size=WITNESS_COUNT, replace=False)
    shared, _ = evaluation_cohort(outer, np.random.default_rng(SHARE_SEED))
    previous = {"repeat_matched_cpu": outer[repeat], "native_witness_cpu": outer[witness], "shared_nominations_cpu": shared}
    used = np.unique(np.concatenate([rows.numpy() for rows in previous.values()]))
    fresh = np.setdiff1d(outer.numpy(), used)
    if len(fresh) != FRESH:
        raise ValueError("Expected the full 32 unused glyphs from the eligible v20 CPU cohort.")
    _, pilot, _ = distinct_cohort(size, np.random.default_rng(PPOCR_PILOT_SEED))
    if np.intersect1d(outer, pilot).size:
        raise ValueError("Validation must be disjoint from the PP-OCR exploratory cohort.")
    repeated = rng.choice(used, TOTAL - FRESH, replace=False)
    new_blocks = rng.permutation(fresh).reshape(len(SESSIONS), -1)
    old_blocks = repeated.reshape(len(SESSIONS), -1)
    panel = np.concatenate((new_blocks, old_blocks), axis=1).ravel()
    assignments = np.repeat(np.arange(len(SESSIONS)), TOTAL // len(SESSIONS))
    order = panel.argsort()
    held = torch.from_numpy(panel[order])
    day = torch.from_numpy(assignments[order])
    overlaps = {name: int(np.intersect1d(held, earlier).size) for name, earlier in previous.items()}
    return held, day, np.sort(fresh), overlaps


@torch.inference_mode()
def ppocr_logits(labeled, atlas, views):
    session, metadata = recognizer()
    encoded = read_glyphs(session, torch.cat((labeled, atlas)), metadata)
    features, bank = fit_coordinates(encoded["ppocrv6_global"], len(labeled))
    logits = {day: standardized(F.normalize(query @ (ridge @ features)) @ bank)
              for day, (ridge, _, query) in views.items()}
    metadata["pca_dimensions"] = features.shape[1]
    return logits, metadata


def compare_conditions(base, amplified, model, sources, stock, new, ranker, proxy, trusted, exact):
    incumbent = with_hanzi(model, amplified, sources, stock)
    alternate = 0.5 * (standardized(stock) + standardized(new))
    proposed = with_hanzi(model, amplified, sources, alternate)
    scores, details = evaluate(exact, base, incumbent, proposed, ranker, proxy, trusted, torch.arange(len(stock)))
    details["recall"] = {ARMS[arm]: value for arm, value in details["recall"].items()}
    details["changed_choices_vs_v20"] = {ARMS[arm]: value for arm, value in details.pop("selected_changes_vs_v16").items()}
    details["mean_new_candidates_vs_v20"] = details.pop("hanzi_mean_novel_candidates_vs_v16")
    details["rows_with_new_candidates_vs_v20"] = details.pop("hanzi_rows_with_new_candidates_vs_v16")
    return scores, details


def paired(scores):
    result = {ARMS[arm]: summarize([values], [scores[V16]]) for arm, values in scores.items()}
    result["full_vs_fixed_pool"] = summarize([scores[FULL]], [scores[FIXED]])
    result["full_vs_nomination_only"] = summarize([scores[FULL]], [scores[NOMINATE]])
    return result


def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU notebook with the official training data.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    root = locate_data()
    scorer = official_scorer(root)
    images = load_images(root, device)
    if images.shape != (2748, 64, 64) or images.dtype != torch.uint8:
        raise ValueError("Use the complete labeled grayscale glyph gallery.")
    fit, outer, independence = fresh_cohort(len(images))
    held, days, fresh, overlaps = balanced_panel(outer, len(images), rng)
    fresh_mask = np.isin(held.numpy(), fresh)
    references = images[held]
    report_file = next(Path("/kaggle/input").rglob("stock_report.json"), None)
    if report_file is None:
        raise FileNotFoundError("Mount the private training-only archive of all four v20 calibration folds.")
    archived = json.loads(report_file.read_text(encoding="utf-8"))
    if (archived["stage"] != "complete" or archived["mouse"] != MOUSE or archived["held_gallery_indices"] != outer.tolist()
            or archived["fit_gallery_indices_sha256"] != tensor_hash(fit)
            or archived["training_raster_sha256"] != tensor_hash(images)):
        raise ValueError("The mounted v20 archive must match the same labeled gallery and outer split.")
    noise, splits = fold_indices(len(fit), device)
    folds, hashes = stock_folds(report_file.parent, splits, noise, archived["calibration_cache"]["key"], device)
    codes = folds[0][3]
    if any(not torch.equal(codes, fold[3]) for fold in folds):
        raise ValueError("Every v20 calibration fold must use the same atlas order.")
    ranker = model_of([fold[0] for fold in folds], [fold[1] for fold in folds])
    del folds
    inner, noise = fit[splits[0][1]], fit[noise]
    report = {"mouse": MOUSE, "seed": SEED, "formal_queries_loaded": False,
              "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "training_raster_sha256": tensor_hash(images),
              "stock_report_sha256": sha256(report_file.read_bytes()).hexdigest(), "stock_fold_sha256": hashes,
              "fit_gallery_indices_sha256": tensor_hash(inner), "fit_glyph_count": len(inner), "noise_glyph_count": len(noise),
              "codepoint_order_sha256": tensor_hash(codes), "held_gallery_indices": held.tolist(), "prior_outer_overlap": independence,
              "held_overlap_with_exploratory_ppocr": 0, "earlier_v20_cpu_overlap": overlaps, "fresh_cpu_gallery_indices": fresh.tolist(),
              "same_day_assignments": {day: held[days == i].tolist() for i, day in enumerate(SESSIONS)},
              "hanzi": {"revision": OCR_REVISION, "sha256": OCR_CHECKSUM}, "ppocrv6_fraction_of_ocr_branch": 0.5,
              "candidate_budget": NOMINATIONS, "anchor_ocf": ANCHOR, "poisson_weight": POISSON_AMPLIFICATION,
              "protocol": "Refit the same v20 inner CPU stem and archived fourfold ranker. For every query use the same "
                          "renderer and v14-matched unique candidate count. Replace half the standardized incumbent OCR "
                          "branch with the new recognizer's global features; evaluate the unchanged v20 strategy, full "
                          "blend, fixed stock lists, and changed nominations with stock scoring. Four repeats use one "
                          "assigned day per glyph, including eight unused and 24 prior CPU scored glyphs per day. "
                          "Pooled 16 uses four known-linked train days. Held images provide only proxy recall and official OCF.",
              "recordings": {}}
    publish(report, OUTPUT, started, "stem_started")
    counts = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(counts) != SESSIONS or any(block.shape[:2] != (len(images), 4) for block in counts.values()):
        raise ValueError("Expected all four labeled days and four measured repeats per image.")
    labeled = images[inner]
    model = fit_v8({day: block[inner] for day, block in counts.items()}, labeled, MODEL_SEED)
    if not torch.equal(model.codepoints, codes):
        raise ValueError("The refit stem's codepoints must match the calibration archive.")
    rendered = rendered_bank(model)
    report["rendered_sha256"] = tensor_hash(rendered)
    report["atlas"] = model.report
    publish(report, OUTPUT, started, "ocr_started")
    views = response_views(counts, inner, held, rng)
    stock = fit_ocr_logits(labeled, model.atlas.images, views)
    publish(report, OUTPUT, started, "ppocr_started")
    ppocr, report["published_ocr"] = ppocr_logits(labeled, model.atlas.images, views)
    publish(report, OUTPUT, started, "ppocr_ready")
    proxy, trusted = image_proxy(model, references)
    report["trusted_proxy_count"] = int(trusted.sum())
    aligned, _ = align_training_glyphs(model.atlas, images[noise])

    sources_by_day, same_day = {}, {}
    same_day_pairs = 0
    for index, (day, block) in enumerate(counts.items()):
        predictive = fit_predictive_likelihood(block[noise], model.encoder.expected_counts(aligned, day))
        base = atlas_scores(model, predictive, block[held], day)
        sources_by_day[day] = normalized_sources(base)
        restricted = torch.where(days == index)[0]
        base = {name: values[restricted] for name, values in base.items()}
        sources = {name: values[restricted] for name, values in sources_by_day[day].items()}
        amplified = {**base, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}
        exact = PairedOCF(scorer, rendered, (references[restricted] >= 128).numpy(), device, 12)
        values, details = compare_conditions(base, amplified, model, sources, stock[day][restricted], ppocr[day][restricted],
                                              ranker, proxy[restricted], trusted[restricted], exact)
        for arm, score in values.items():
            if arm not in same_day:
                same_day[arm] = np.empty(len(held))
            same_day[arm][restricted.numpy()] = score
        same_day_pairs += len(exact.values)
        report["recordings"][day] = {"paired_ocf": paired(values), **details}
        publish(report, OUTPUT, started, f"{day}_scored")
    report["same_day_four_repeats"] = paired(same_day)
    report["same_day_unused_32"] = paired({name: scores[fresh_mask] for name, scores in same_day.items()})
    sources = {name: torch.stack([sources_by_day[day][name] for day in SESSIONS]).mean(0) for name in sources_by_day[SESSIONS[0]]}
    base, amplified = pooled_scores(model, sources)
    exact = PairedOCF(scorer, rendered, (references >= 128).numpy(), device, 12)
    values, details = compare_conditions(base, amplified, model, sources, torch.stack(tuple(stock.values())).mean(0),
                                          torch.stack(tuple(ppocr.values())).mean(0), ranker, proxy, trusted, exact)
    report["pooled_16_known_correspondence"] = {"paired_ocf": paired(values), **details}
    report["pooled_16_unused_32"] = paired({name: scores[fresh_mask] for name, scores in values.items()})
    report["unique_exact_pairs_within_conditions"] = {"same_day": same_day_pairs, "pooled_16": len(exact.values)}
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
