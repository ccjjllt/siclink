"""Replicate frozen single-Hanzi feature retrieval across the full public Noto atlas."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import build_font_atlas, image_descriptors, standard_error, standardized
from kaggle_task1_native_hanzi import (HELD_COUNT, OCR_CHECKSUM, OCR_REVISION, OCR_SOURCE, PC_COUNT, RIDGE, SEED as PILOT_SEED,
                                      component_scope, context_scope, descriptors, feature_extractors, fitted_pca,
                                      microns_scope, response_views, split_glyphs as pilot_split)


SEED = 221879
MICE = ("mice1", "mice2")
DEPTHS = (1, 5, 16, 32, 64)
PAIRS = 64
OCF_GLYPHS_PER_DAY = 16
ATLAS_SIZE = 27550
FUSION = "equal_contour_hanzi"
OUTPUT = Path("/kaggle/working/task1_native_hanzi_open_atlas.json")


def independent_split(size):
    cpu = torch.device("cpu")
    excluded = {"native_hanzi_pilot": pilot_split(size, np.random.default_rng(PILOT_SEED))[1],
                "component_raster": component_scope(size)[1],
                "context_prediction": context_scope(size)[1],
                "microns_repeat_recheck": microns_scope(size, cpu)[1]}
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(tuple(excluded.values()))))
    held = np.sort(np.random.default_rng(SEED).choice(eligible, HELD_COUNT, replace=False))
    fitting = np.setdiff1d(np.arange(size), held)
    overlap = {name: int(np.intersect1d(held, rows).size) for name, rows in excluded.items()}
    return torch.from_numpy(fitting), torch.from_numpy(held), overlap


def image_coordinates(fit_images, candidates, extractors, *, normalize_atlas=True):
    """Stream identical release preprocessing through each feature extractor."""
    all_images = torch.cat((fit_images, candidates))
    features = {}
    for start in range(0, len(all_images), 512):
        for arm, batch in descriptors(all_images[start:start + 512], extractors).items():
            if arm not in features:
                features[arm] = batch.new_empty((len(all_images), batch.shape[1]))
            features[arm][start:start + len(batch)].copy_(batch)
    coordinates = {}
    for arm in tuple(features):
        fit, atlas = fitted_pca(features.pop(arm), len(fit_images)).split((len(fit_images), len(candidates)))
        if normalize_atlas:
            atlas = F.normalize(atlas)
        coordinates[arm] = (fit.contiguous(), atlas.T.contiguous())
    return coordinates


def nominate(scores):
    fusion = 0.5 * (standardized(scores["contour"]) + standardized(scores["frozen_hanzi"]))
    return {arm: values.topk(DEPTHS[-1], dim=1).indices for arm, values in {**scores, FUSION: fusion}.items()}


def fixed_rankings(views, image_codes):
    pooled = {}
    result = {}
    for day, (ridge_map, _, query) in views.items():
        scores = {}
        for arm, (fitting, atlas) in image_codes.items():
            predicted = F.normalize(query @ (ridge_map @ fitting))
            day_score = standardized(predicted @ atlas)
            scores[arm] = day_score
            if arm in pooled:
                pooled[arm].add_(day_score)
            else:
                pooled[arm] = day_score.clone()
        result[day] = nominate(scores)
    result["pooled_16"] = nominate(pooled)
    return result


def visual_proxy(atlas, held_images):
    """Assign evaluation-only independent nearest Noto references after ranking."""
    signatures = image_descriptors((held_images >= 128).to(torch.uint8) * 255)
    best, nearest = (signatures @ atlas.contours.T).topk(2, dim=1)
    confident = (best[:, 0] - best[:, 1] >= 0.02) & (best[:, 0] >= 0.80)
    return nearest[:, 0], confident, float(best[:, 0].mean())


def recalls(rankings, proxy, confident):
    result = {}
    for day, arms in rankings.items():
        result[day] = {}
        for arm, columns in arms.items():
            targets = columns == proxy[:, None]
            result[day][arm] = {str(k): float(targets[:, :k].any(1).float().mean()) for k in DEPTHS}
            result[day][arm]["confident"] = {str(k): float(targets[confident, :k].any(1).float().mean())
                                              for k in DEPTHS}
    return result


def exact_comparison(metric, rank, sample, days, condition):
    values = {}
    for arm, shortlist in rank["pooled_16"].items():
        if condition == "pooled_16":
            selections = shortlist[sample, 0]
        else:
            selections = torch.stack([rank[day][arm][sample, 0] for day in days])
            positions = torch.arange(len(sample))
            selections = selections[positions // OCF_GLYPHS_PER_DAY, positions]
        values[arm] = metric.evaluate(selections[:, None])[:, 0]
    measured = {arm: exact_summary(scores, values["contour"]) for arm, scores in values.items()}
    measured["frozen_vs_random"] = exact_summary(values["frozen_hanzi"], values["random_mobilenet"])
    measured["equal_fusion_vs_frozen"] = exact_summary(values[FUSION], values["frozen_hanzi"])
    return measured


def exact_summary(values, reference):
    delta = values - reference
    return {"ocf": float(values.mean()), "paired_delta": float(delta.mean()), "paired_se": standard_error(delta),
            "paired_glyphs": len(values)}


def publish(result, started, stage):
    result.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{stage} after {result['seconds']:.1f}s", flush=True)


@torch.inference_mode()
def main():
    if not Path("/kaggle/input").is_dir() or not OUTPUT.parent.is_dir():
        raise RuntimeError("Run the open atlas replication in a Kaggle CPU Notebook.")
    device = torch.device("cpu")
    torch.set_num_threads(4)
    started = time.monotonic()
    rng = np.random.default_rng(SEED + 1)
    root = locate_data()
    images, catalog = load_images(root, device), load_catalog(root)
    fit, held, overlap = independent_split(len(images))
    random_style_fit = rng.choice(len(fit), 128, replace=False)
    extractors = feature_extractors()
    atlas, provenance = build_font_atlas(images[fit[random_style_fit]], device)
    if len(atlas.images) != ATLAS_SIZE:
        raise RuntimeError("The shared full Noto atlas must contain 27,550 candidates.")
    print(f"Fixed all {ATLAS_SIZE:,} public glyphs before scoring", flush=True)
    features_started = time.monotonic()
    image_codes = image_coordinates(images[fit], atlas.images, extractors)
    feature_seconds = time.monotonic() - features_started
    print(f"Extracted and fit the three image features in {feature_seconds:.1f}s", flush=True)
    result = {"seed": SEED, "device": str(device), "fit_glyphs": len(fit), "held_glyphs": len(held),
              "overlap_with_previous_held_glyphs": overlap, "atlas_size": len(atlas.images),
              "paired_glyphs_per_mouse_condition": PAIRS, "same_day_exact_glyphs_per_day": OCF_GLYPHS_PER_DAY,
              "font": provenance, "pca_components": PC_COUNT, "ridge_per_glyph": RIDGE,
              "feature_seconds": feature_seconds, "ocr": {"checkpoint": OCR_SOURCE, "sha256": OCR_CHECKSUM,
                                                          "source_revision": OCR_REVISION, "license": "Apache-2.0"},
              "protocol": "Fix every Noto codepoint and common calibrated output raster using fit images only. "
                          "The fit-only readouts predict 128-dimensional image features from labeled neural counts. "
                          "Mean same-day four-repeat scores are compared by row z score, which are summed across days "
                          "for a co-indexed proxy of 16 repeats. Fusion uses equally standardized contour and OCR scores. "
                          "Held images enter only after rankings, for nearest-font morphology proxies and exact OCF. "
                          "The same-day exact scores allocate 16 distinct paired glyphs to each of four days, "
                          "using the same 64 glyphs as the co-indexed 16-repeat comparison. "
                          "Official SiClink training glyphs and labeled trials are the only neural data.",
              "mice": {}}
    rankings = {}
    days = {}
    for mouse in MICE:
        groups = grouped_blocks(root, catalog, mouse, device, len(images), bin_ms=25)
        views = response_views(groups, fit, held, rng)
        days[mouse] = tuple(views)
        del groups
        rankings[mouse] = fixed_rankings(views, image_codes)
        print(f"Ranked all candidates for {mouse}", flush=True)

    held_images = images[held]
    proxy, confident, match_similarity = visual_proxy(atlas, held_images)
    result.update(confident_proxy_glyphs=int(confident.sum()), proxy_mean_similarity=match_similarity)
    for mouse in MICE:
        result["mice"][mouse] = {"top_k": recalls(rankings[mouse], proxy, confident), "exact_ocf": {}}
    publish(result, started, "rank_metrics")
    for mouse, report in result["mice"].items():
        for day, measurements in report["top_k"].items():
            print(json.dumps({"mouse": mouse, "recording": day, "top_k": measurements}), flush=True)
    scorer = official_scorer(root)
    sample = torch.from_numpy(rng.choice(len(held), PAIRS, replace=False))
    metric = PairedOCF(scorer, atlas.images, (held_images[sample] >= 128).numpy(), device, 12)
    for condition in ("pooled_16", "single_day_4"):
        for mouse in MICE:
            measured = exact_comparison(metric, rankings[mouse], sample, days[mouse], condition)
            result["mice"][mouse]["exact_ocf"][condition] = measured
            result["exact_ocf_pairs_computed"] = len(metric.values)
            publish(result, started, f"{mouse}_{condition}")
            print(json.dumps(measured), flush=True)
    publish(result, started, "complete")


if __name__ == "__main__":
    main()
