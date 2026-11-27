"""Compare the v18 mice2 Poisson policy with a fixed frozen-Hanzi blend on held glyphs."""

import argparse
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task1_bipartite_validation import image_proxy
from kaggle_task1_crossfit_candidate import fit_crossfit_policy
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED
from kaggle_task1_mice2_microns_spatial_atlas import fresh_glyphs as prior_spatial_cohort
from kaggle_task1_mice2_poisson_repeat_recheck import fresh_holdout as prior_poisson_cohort
from kaggle_task1_microns_hanzi_recheck import (ATLAS_SIZE, CONTOUR_OCR_WEIGHT, FIXED, FULL, NOMINATE, V14, V16,
                                             calibration_key, evaluate as compare_policies, fit_ocr_logits,
                                             fresh_cohort as mice1_direct_cohort, with_hanzi)
from kaggle_task1_microns_open_atlas import publish
from kaggle_task1_microns_repeat_recheck import accumulate
from kaggle_task1_native_hanzi import (OCR_CHECKSUM, OCR_REVISION, OCR_SOURCE, PC_COUNT, RIDGE as OCR_RIDGE, SEED as OCR_SEED,
                                       STAGE, response_views, split_glyphs as ocr_pilot_cohort)
from kaggle_task1_native_hanzi_open_atlas import independent_split as ocr_full_cohort
from kaggle_task1_v11_repeat_weight_transfer import POISSON_AMPLIFICATION, normalized_sources, pooled_scores, weighted_fusion
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, rendered_bank, summarize


MOUSE = "mice2"
SEED = 222311
HOLDOUT = 320
V18, HANZI = "v18_poisson_1p4", "v18_hanzi"
HANZI_FIXED, HANZI_NOMINATE = "v18_hanzi_fixed_pool", "v18_hanzi_nominations_only"
ARMS = {V14: V14, V16: V18, FULL: HANZI, FIXED: HANZI_FIXED, NOMINATE: HANZI_NOMINATE}
OUTPUT = Path("/kaggle/working/task1_mice2_hanzi_recheck.json")


def fresh_cohort(size):
    cpu = torch.device("cpu")
    previous = {"mice2_spatial_atlas": prior_spatial_cohort(size, cpu)[1],
                "mice2_poisson_repeat_recheck": prior_poisson_cohort(size, cpu)[1],
                "native_hanzi_pilot": ocr_pilot_cohort(size, np.random.default_rng(OCR_SEED))[1],
                "native_hanzi_full_atlas": ocr_full_cohort(size)[1],
                "mice1_v16_hanzi_direct": mice1_direct_cohort(size)[1]}
    excluded = np.unique(np.concatenate(tuple(previous.values())))
    eligible = np.setdiff1d(np.arange(size), excluded)
    if len(eligible) < HOLDOUT:
        raise ValueError(f"Only {len(eligible)} glyphs remain outside the prior OCR and repeat cohorts.")
    held = np.sort(np.random.default_rng(SEED).choice(eligible, HOLDOUT, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    overlap = {name: int(np.intersect1d(held, rows).size) for name, rows in previous.items()}
    return torch.from_numpy(fit), torch.from_numpy(held), {"eligible": len(eligible), "overlap_by_cohort": overlap}


def evaluate(exact, base, v18, fusion, ranker, proxy, trusted, inverse):
    outcomes, details = compare_policies(exact, base, v18, fusion, ranker, proxy, trusted, inverse)
    details["recall"] = {ARMS[arm]: recall for arm, recall in details["recall"].items()}
    details["selected_changes_vs_v18"] = {ARMS[arm]: count for arm, count in details.pop("selected_changes_vs_v16").items()}
    details["hanzi_mean_novel_candidates_vs_v18"] = details.pop("hanzi_mean_novel_candidates_vs_v16")
    return {ARMS[arm]: outcome for arm, outcome in outcomes.items()}, details


def compare(values):
    return {"vs_v14": {arm: summarize(score, values[V14]) for arm, score in values.items()},
            "vs_v18": {arm: summarize(values[arm], values[V18]) for arm in (HANZI, HANZI_FIXED, HANZI_NOMINATE)},
            "full_vs_fixed_pool": summarize(values[HANZI], values[HANZI_FIXED]),
            "full_vs_nominations_only": summarize(values[HANZI], values[HANZI_NOMINATE])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Extracted competition directory containing catalog.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--calibration-cache", type=Path, help="Save and resume fit-only calibration folds in this directory")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use a remote Kaggle or Colab GPU for held open-atlas evaluation.")
    started = time.monotonic()
    torch.set_num_threads(4)
    cpu, device = torch.device("cpu"), torch.device("cuda:0")
    root = args.data_root.resolve(strict=True) if args.data_root else locate_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scorer = official_scorer(root)
    images = load_images(root, cpu)
    if len(images) != 2748:
        raise ValueError("Expected all 2,748 labeled glyphs.")
    fit, held, independence = fresh_cohort(len(images))
    groups = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images))
    if tuple(groups) != SESSIONS or any(counts.shape[:2] != (len(images), 4) for counts in groups.values()):
        raise ValueError("Expected all four mice2 labeled recordings with four repeats for every glyph.")
    labeled_cpu = images[fit]
    cache_key, cache_dir = None, None
    if args.calibration_cache is not None:
        cache_key = calibration_key(fit, held, labeled_cpu, groups, scorer, device, mouse=MOUSE)
        cache_dir = args.calibration_cache.resolve() / cache_key
        print(f"Calibration snapshots: {cache_dir}/fold-*.pt", flush=True)
    views = response_views(groups, fit, held, np.random.default_rng(SEED + 1))
    training = {day: counts[fit].to(device) for day, counts in groups.items()}
    queries = {day: counts[held].to(device) for day, counts in groups.items()}
    labeled, references = labeled_cpu.to(device), images[held].to(device)
    del groups, images, labeled_cpu

    report = {"mouse": MOUSE, "outer_seed": SEED, "v14_seed": MODEL_SEED, "fit_glyphs": len(fit), "held_glyphs": len(held),
              "held_gallery_indices": held.tolist(), "cohort_independence": independence, "formal_queries_loaded": False,
              "repeat_conditions": {"single_day": 4, "known_correspondence_pooled": 16}, "microns_features_used": False,
              "nominal_policy": "v18 original SiClink Poisson 1.4x", "anchor_ocf": ANCHOR, "nominations": NOMINATIONS,
              "poisson_multiplier": POISSON_AMPLIFICATION, "fixed_ocr_share_in_decoded_contour": CONTOUR_OCR_WEIGHT,
              "ocr": {"source": OCR_SOURCE, "revision": OCR_REVISION, "sha256": OCR_CHECKSUM, "stage": STAGE,
                      "pca_components": PC_COUNT, "ridge_per_fit_glyph": OCR_RIDGE}, "sessions": {},
              "protocol": "Fit the v14 ranker once with four ordered calibration folds. Retain the exact v18 original Poisson "
                          "1.4x policy, with original atlas column order and equal v14-matched distinct candidates. The frozen OCR "
                          "blend changes only v4/v8 and uses the same fitted renderer in every arm. Compare exact paired OCF for "
                          "four measured repeats in each day and known-correspondence pooled 16, with the latter the primary contrast. "
                          "Held references provide the retrospective visual proxy and exact OCF. Neither formal test queries nor "
                          "MICrONS readout data are used."}
    if cache_dir is not None:
        report["calibration_cache"] = {"key": cache_key, "directory": str(cache_dir)}
    publish(report, args.output, started, "fourfold_started")
    ranker, folds, noise, codepoints = fit_crossfit_policy(training, labeled, scorer, cache_dir=cache_dir, cache_key=cache_key)
    report["v14_fit"] = {"noise": "literal calibration fold zero", "folds": folds, "ranker": ranker.diagnostics}
    publish(report, args.output, started, "fourfold_ready")
    publish(report, args.output, started, "outer_model_started")
    model = fit_v8(training, labeled, MODEL_SEED)
    del training
    if len(codepoints) != ATLAS_SIZE or not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("The full atlas column order must match all four folds.")
    font, unihan = model.report["atlas"]["sha256"], model.report["prior"]["unicode"]["sha256"]
    for fold in folds.values():
        source = fold["atlas_and_font_style"]
        if (source["atlas"]["sha256"], source["prior"]["unicode"]["sha256"]) != (font, unihan):
            raise ValueError("Every fold and the outer model must have identical public atlas sources.")
    rendered = rendered_bank(model)
    report["v14_fit"].update(outer=model.report, atlas_size=ATLAS_SIZE,
                            codepoint_order_sha256=sha256(codepoints.contiguous().numpy().tobytes()).hexdigest(),
                            rendered_pixel_sha256=sha256(rendered.cpu().contiguous().numpy().tobytes()).hexdigest())
    publish(report, args.output, started, "outer_model_ready")

    publish(report, args.output, started, "hanzi_started")
    ocr = fit_ocr_logits(labeled, model.atlas.images, views)
    del labeled, views
    publish(report, args.output, started, "hanzi_ready")
    proxy, trusted = image_proxy(model, references)
    report["trusted_proxy_glyphs"] = int(trusted.sum())
    exact = PairedOCF(scorer, rendered, (references >= 128).cpu().numpy(), device, 12)
    rng = np.random.default_rng(SEED + 2)
    total, measured = {}, {}
    for day, counts in queries.items():
        publish(report, args.output, started, f"{day}_started")
        order = torch.as_tensor(rng.permutation(len(held)), device=device)
        inverse = order.argsort()
        base = atlas_scores(model, noise[day], counts[order], day)
        sources = normalized_sources(base)
        v18 = {**base, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}
        fusion = with_hanzi(model, v18, sources, ocr[day][order])
        values, details = evaluate(exact, base, v18, fusion, ranker, proxy[order], trusted[order], inverse)
        for arm, score in values.items():
            measured.setdefault(arm, []).append(score)
        accumulate(total, sources, inverse)
        report["sessions"][day] = {**details, "paired_exact_ocf": compare({arm: [score] for arm, score in values.items()})}
        report["same_day_four_repeats"] = compare(measured)
        del base, sources, v18, fusion
        publish(report, args.output, started, f"{day}_scored")

    publish(report, args.output, started, "pooled_16_started")
    sources = {name: matrix.div_(len(SESSIONS)) for name, matrix in total.items()}
    base, v18 = pooled_scores(model, sources)
    fusion = with_hanzi(model, v18, sources, torch.stack(tuple(ocr.values())).mean(0))
    values, details = evaluate(exact, base, v18, fusion, ranker, proxy, trusted, torch.arange(len(held), device=device))
    report["pooled_16_known_correspondence"] = {**details, "paired_exact_ocf": compare({arm: [score] for arm, score in values.items()})}
    day_delta = (np.stack(measured[HANZI]) - np.stack(measured[V18])).mean(0)
    report["hanzi_pooled_16_vs_mean_day_4_gain"] = summarize([values[HANZI] - values[V18]], [day_delta])
    report["unique_exact_pairs_scored"] = len(exact.values)
    publish(report, args.output, started, "complete")


if __name__ == "__main__":
    main()
