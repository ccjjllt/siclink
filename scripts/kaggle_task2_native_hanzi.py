"""Test frozen single-Hanzi shape features for mice2 across held recording days."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_bipartite_encoding import timing_view
from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, combine_logits, fit_retrieval, retrieval_score_components
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standard_error
from kaggle_task1_native_hanzi import OCR_CHECKSUM, OCR_REVISION, OCR_SOURCE, PC_COUNT, RIDGE
from kaggle_task1_native_hanzi import descriptors, feature_extractors, fitted_pca
from kaggle_task2_local_cycles import held_glyphs as cycles_holdout
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics
from kaggle_task3_metric_validation import unique_pair_ocf


MOUSE = "mice2"
SOURCE_DAYS, TARGET_DAYS = ("session1", "session3"), ("session2", "session4")
MODEL_SEED, SEED = 211, 285527
OCF_COUNT = 96
SHAPE_WEIGHT = 0.10
TEMPERATURE = 1 / 3
EPOCHS = 12
ARMS = ("v9", "contour", "random_mobilenet", "frozen_hanzi")
OUTPUT = Path("/kaggle/working/task2_native_hanzi.json")


def scoring_glyphs(size, rng):
    # The local-cycle experiment already excludes its earlier mice2 scoring cohorts.
    from kaggle_task2_mice2_ranker_scale import previous_mice2_cohorts

    older = previous_mice2_cohorts(size)
    scale = np.random.default_rng(41721).choice(np.setdiff1d(np.arange(size), older), size=256, replace=False)
    cycles, _ = cycles_holdout(size, 52873, 768, torch.device("cpu"))
    excluded = np.unique(np.concatenate((older, scale, cycles.numpy())))
    eligible = np.setdiff1d(np.arange(size), excluded)
    scored = rng.choice(eligible, len(TARGET_DAYS) * OCF_COUNT, replace=False)
    return dict(zip(TARGET_DAYS, torch.from_numpy(scored).split(OCF_COUNT), strict=True)), len(excluded)


def neural_features(counts):
    return timing_view(counts.unflatten(-1, (-1, 10))).mean(1)


def fit_shape_models(sources, visual):
    values = torch.stack([neural_features(counts) for counts in sources.values()])
    day_center = values.mean(1)
    day_scale = values.std(1).clamp_min(0.04)
    normalized = ((values - day_center[:, None]) / day_scale[:, None]).flatten(0, 1)
    gram = normalized.T @ normalized
    gram.diagonal().add_(RIDGE * len(normalized))
    ridge_map = torch.cholesky_solve(normalized.T, torch.linalg.cholesky(gram))
    models = {name: (ridge_map @ target.repeat(len(sources), 1), F.normalize(target).T.contiguous())
              for name, target in visual.items()}
    return models, day_center.mean(0), day_scale.square().mean(0).sqrt()


def score_day(query, order, v9, hierarchy, shapes, center, scale):
    validated = combine_logits(retrieval_score_components(v9, query, None, task="task2"), "task2")
    empirical = hierarchy.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    stock = logit_scale(blend_hierarchical_logits(validated, empirical))
    counts = (neural_features(query) - center) / scale
    scores = {name: logit_scale(F.normalize(counts @ weights) @ template)
              for name, (weights, template) in shapes.items()}
    comparisons = {"v9": stock, **{name: logit_scale(stock + SHAPE_WEIGHT * value) for name, value in scores.items()}}
    choices = {name: assign_glyphs(matrix) for name, matrix in comparisons.items()}
    rows = torch.arange(len(query))
    details = {}
    for name, matrix in comparisons.items():
        log_likelihood = (matrix[rows, order] / TEMPERATURE - (matrix / TEMPERATURE).logsumexp(1))
        selected = choices[name]
        if len(selected.unique()) != len(query):
            raise RuntimeError("Each complete assignment must use all 2,748 glyphs exactly once.")
        details[name] = {**row_diagnostics(matrix, order), "identity_nll": float(-log_likelihood.mean()),
                         "assigned_top1": float((selected == order).float().mean()),
                         "changed_from_v9": int((selected != choices["v9"]).sum()),
                         "unique_assigned_glyphs": len(selected)}
    return choices, details, {name: row_diagnostics(matrix, order) for name, matrix in scores.items()}


def paired(values, baseline):
    delta = (values - baseline).numpy()
    return {"ocf": float(values.mean()), "paired_delta": float(delta.mean()), "paired_se": standard_error(delta),
            "scored_glyphs": len(delta)}


def summarize(measured):
    results = {name: paired(values, measured["v9"]) for name, values in measured.items()}
    results["frozen_vs_contour"] = paired(measured["frozen_hanzi"], measured["contour"])
    results["frozen_vs_random"] = paired(measured["frozen_hanzi"], measured["random_mobilenet"])
    return results


def publish(report, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


@torch.no_grad()
def main():
    if not Path("/kaggle/input").is_dir() or not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU Notebook with official training inputs.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    root = locate_data()
    images = load_images(root, device)
    masks = images >= 128
    samples, excluded = scoring_glyphs(len(images), rng)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images), bin_ms=25)
    sources = {day: recordings.pop(day) for day in SOURCE_DAYS}
    visual = {name: fitted_pca(features, len(images)) for name, features in
              descriptors(images, feature_extractors()).items()}
    shapes, center, scale = fit_shape_models(sources, visual)
    print(f"Shape readouts fitted after {time.monotonic() - started:.1f}s", flush=True)
    with torch.enable_grad():
        v9 = fit_retrieval(as_trial_blocks(sources), len(images), images=images, device=device, seed=MODEL_SEED,
                           epochs=EPOCHS)
    hierarchy = fit_hierarchical_gallery(sources, view="total")
    report = {"mouse": MOUSE, "seed": SEED, "source_days": list(sources), "held_days": list(TARGET_DAYS),
              "gallery_size": len(images), "ocf_glyphs_per_day": OCF_COUNT, "previous_scoring_exclusions": excluded,
              "shape_weight_vs_v9": SHAPE_WEIGHT, "nll_temperature": TEMPERATURE, "v9_classifier_epochs": EPOCHS,
              "pca_components": PC_COUNT, "ridge_per_source_observation": RIDGE,
              "ocr": {"checkpoint": OCR_SOURCE, "revision": OCR_REVISION, "sha256": OCR_CHECKSUM, "license": "Apache-2.0"},
              "protocol": "Both evaluation days are withheld from every labeled neural fit. Their query rows are "
                          "permuted independently. The v9 recipe is refitted on the same two source days with its "
                          "existing unlabeled query normalization and repeat-noise estimate. Shape readouts use "
                          "source-only normalization and fit across the same two source days; the known 2,748 glyph "
                          "images and source-day label associations are available. All arms use a dense Hungarian "
                          "assignment and output the identical official gallery. v9 has no exact-OCF reranker here. "
                          "The scoring cohorts are disjoint by glyph across the two target days and from selected "
                          "earlier Task 2 scoring cohorts. All spikes are drawn from official labeled training.",
              "recordings": {}}
    selections = {}
    for held in TARGET_DAYS:
        order = torch.from_numpy(rng.permutation(len(images)))
        query = recordings[held][order]
        choices, metrics, standalone = score_day(query, order, v9, hierarchy, shapes, center, scale)
        sample_rows = order.argsort()[samples[held]]
        selections[held] = torch.stack([choices[name][sample_rows] for name in ARMS])
        report["recordings"][held] = {"rank_and_assignment": metrics, "standalone_shape_rank": standalone,
                                      "exact_ocf_glyph_indices": samples[held].tolist()}
        publish(report, started, f"{held}_ranks")
        print(json.dumps({"day": held, "rank_and_assignment": metrics, "standalone_shape_rank": standalone}), flush=True)

    scorer = official_scorer(root)
    all_measurements = []
    for held in TARGET_DAYS:
        selected = selections[held]
        results, pairs = unique_pair_ocf(selected, samples[held].expand_as(selected), masks, scorer, device, 96)
        measured = dict(zip(ARMS, results, strict=True))
        report["recordings"][held].update(exact_ocf=summarize(measured), exact_ocf_unique_pairs=pairs)
        all_measurements.append(measured)
        publish(report, started, f"{held}_ocf")
        print(json.dumps({"day": held, "exact_ocf": report["recordings"][held]["exact_ocf"]}), flush=True)
    report["pooled_ocf"] = summarize({name: torch.cat([values[name] for values in all_measurements]) for name in ARMS})
    publish(report, started, "complete")


if __name__ == "__main__":
    main()
