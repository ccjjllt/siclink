"""Recheck fixed Task 2 marginal calibration with three source days per target."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, combine_logits, fit_retrieval, retrieval_score_components
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task2_constrained_ceiling import fresh_samples as ceiling_cohorts
from kaggle_task2_native_hanzi import SEED as HANZI_SEED, scoring_glyphs as hanzi_cohorts
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics
from kaggle_task2_rank_marginals import (ADAPTED, BLEND, CEILING_CROWDED, EPOCHS, LINEAR, MODEL_SEED, MOUSE,
                                         TEMPERATURE, V9, evaluation_glyphs, linear, marginal_logits, normal_scores,
                                         responses, source_cdf, summarize, whiten)
from kaggle_v8_metric_validation import SESSIONS


SEED = 286201
OCF_GLYPHS = 256
EXACT_ARMS = (V9, LINEAR, ADAPTED)
OUTPUT = Path("/kaggle/working/task2_rank_marginals_recheck.json")
PROVENANCE = {"v5_task2_sha256": "a8c85da2a7f2733c92f7cb8e9b5fc24d341ec1abee1a3d8da13f1075c75dc760",
              "v9_sha256": "b5817a9cffb29246422452f524cdff941a4990722681c5c3046da514eef5f8ee",
              "v11_sha256": "20e4dd1f80f63f5635f0bc5cf16369500bb4d15b2d2d3ea9971a8925b07f21e8",
              "v20_sha256": "b098c3bc62aa96b289b72c512e977fa246fb37f3ddfdab32ffe786d3e7ba8281",
              "mice1_task2_update_sha256": "6a98a2614879d66a35dbd6d0cb6aa8bba266fbd528defde54ace7930c19e9663",
              "verified_task2_rows": 5496, "mice2_task2_rows_unchanged_since_v9": 2748,
              "mice1_task2_only_changes_v9_to_v20": 378,
              "policy": "validated-hierarchical\u2014balanced retrieval and total empirical Bayes, full Hungarian gallery",
              "validated_weight": 0.65, "empirical_bayes_weight": 0.35, "query_repeat_noise_mix": 0.5}


def evaluation_sample(size):
    prior, _ = evaluation_glyphs(size)
    ceiling, _, _ = ceiling_cohorts(size)
    hanzi, _ = hanzi_cohorts(size, np.random.default_rng(HANZI_SEED))
    older = {"rank_marginals_primary": prior.numpy(), "exact_ceiling_crowded": np.asarray(CEILING_CROWDED),
             "exact_ceiling_uniform": np.concatenate(tuple(ceiling.values())),
             "native_hanzi_scoring": np.concatenate(tuple(hanzi.values()))}
    excluded = np.unique(np.concatenate(tuple(older.values())))
    eligible = np.setdiff1d(np.arange(size), excluded)
    sample = np.sort(np.random.default_rng(SEED).choice(eligible, OCF_GLYPHS, replace=False))
    overlap = {name: int(np.intersect1d(sample, rows).size) for name, rows in older.items()}
    return torch.from_numpy(sample), {"eligible": len(eligible), "overlap_by_cohort": overlap}


def fit_marginal_templates(sources):
    raw = tuple(responses(counts) for counts in sources.values())
    linear_days = torch.stack([linear(day) for day in raw])
    rank_days = torch.stack([normal_scores(day, source_cdf(day)) for day in raw])
    # The denominator preserves the original two-source difference covariance when only two days are supplied.
    average = linear_days.mean(0)
    drift = (linear_days - average).flatten(0, 1)
    covariance = drift.T @ drift / ((len(raw) - 1) * (len(average) - 1))
    shrunk = 0.80 * covariance + 0.20 * torch.diag(covariance.diagonal())
    shrunk.diagonal().add_(0.10)
    factor = torch.linalg.cholesky(shrunk)
    return whiten(average, factor), whiten(rank_days.mean(0), factor), factor, source_cdf(torch.cat(raw))


def publish(report, phase, started):
    report.update(stage=phase, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": phase, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Run on a private Kaggle CPU notebook with official labeled training data.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    if len(images) != 2748:
        raise ValueError("Expected the complete known 2,748-glyph gallery.")
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(recordings) != SESSIONS or any(counts.shape[:2] != (len(images), 4) for counts in recordings.values()):
        raise ValueError("Expected four complete labeled days with four real repeats per glyph.")
    sample, independence = evaluation_sample(len(images))
    rng = np.random.default_rng(SEED + 1)
    metric = PairedOCF(official_scorer(root), images, (images[sample] >= 128).numpy(), device, 12)
    report = {"mouse": MOUSE, "seed": SEED, "model_seed": MODEL_SEED, "v9_fit_epochs": EPOCHS, "gallery_size": len(images),
              "archive_provenance": PROVENANCE, "scoring_cohort": independence, "scored_glyph_indices": sample.tolist(),
              "labeled_source_days_per_fold": 3, "held_target_days": SESSIONS, "measured_repeats": 4,
              "blend_weight": BLEND, "primary_arm": LINEAR, "exact_ocf_arms": EXACT_ARMS,
              "identity_nll_temperature": TEMPERATURE, "formal_test_accessed": False, "recordings": {},
              "protocol": "Recheck the exact v20 mice2 Task 2 policy family on three labeled days per independently permuted held "
                          "day. The original experiment trained on two source days per target. Retain the original seed, epochs, "
                          "25% augmentation, per-unit transforms, covariance shrinkage and full one-to-one assignment. With "
                          "three sources, pool the same cross-day residual covariance around each glyph's source-day mean. "
                          "Use a new matched 256-glyph OCF panel in every held recording. The complete target block supplies only "
                          "unlabeled marginal responses. The frozen-source CDF remains a full-gallery identity control and the "
                          "three designated arms receive exact OCF. No formal query or archived prediction is used in fitting."}
    publish(report, "inputs_ready", started)
    scores = {name: [] for name in EXACT_ARMS}
    for day in SESSIONS:
        sources = {name: counts for name, counts in recordings.items() if name != day}
        publish(report, f"{day}_fitting", started)
        torch.manual_seed(MODEL_SEED)
        with torch.enable_grad():
            retrieval = fit_retrieval(as_trial_blocks(sources), len(images), images=images, device=device,
                                      seed=MODEL_SEED, epochs=EPOCHS)
        hierarchy = fit_hierarchical_gallery(sources, view="total")
        templates = fit_marginal_templates(sources)
        order = torch.from_numpy(rng.permutation(len(images)))
        query = recordings[day][order]
        raw = combine_logits(retrieval_score_components(retrieval, query, None, task="task2"), "task2")
        empirical = hierarchy.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
        hybrid = blend_hierarchical_logits(raw, empirical)
        baseline = logit_scale(hybrid)
        alternatives = marginal_logits(query, templates)
        logits = {V9: hybrid, **{name: baseline.lerp(extra, BLEND) for name, extra in alternatives.items()}}
        chosen = {name: assign_glyphs(scores) for name, scores in logits.items()}
        if any(choice.unique().numel() != len(images) for choice in chosen.values()):
            raise ValueError("Every arm must use each gallery glyph exactly once.")
        diagnostics = {}
        rows = torch.arange(len(order))
        for name, matrix in logits.items():
            density = logit_scale(matrix) / TEMPERATURE
            diagnostics[name] = {**row_diagnostics(matrix, order),
                                 "identity_nll_at_fixed_temperature": float((density.logsumexp(1) - density[rows, order]).mean()),
                                 "assigned_top1": float((chosen[name] == order).float().mean()),
                                 "changed_vs_v9": int((chosen[name] != chosen[V9]).sum()),
                                 "distinct_outputs": int(chosen[name].unique().numel())}
        report["recordings"][day] = {"source_days": tuple(sources), "diagnostics": diagnostics}
        publish(report, f"{day}_identity_ready", started)
        sample_rows = order.argsort()[sample]
        exact = {name: metric.evaluate(chosen[name][sample_rows, None])[:, 0] for name in EXACT_ARMS}
        for name, values in exact.items():
            scores[name].append(values)
        report["recordings"][day].update(ocf_vs_v9={name: summarize(values, exact[V9]) for name, values in exact.items()},
                                         rank_vs_linear=summarize(exact[ADAPTED], exact[LINEAR]),
                                         unique_exact_pairs_so_far=len(metric.values))
        publish(report, f"{day}_scored", started)

    glyph_scores = {name: np.stack(values).mean(0) for name, values in scores.items()}
    report["ocf_vs_v9"] = {name: summarize(values, glyph_scores[V9]) for name, values in glyph_scores.items()}
    report["rank_vs_linear"] = summarize(glyph_scores[ADAPTED], glyph_scores[LINEAR])
    report["ocf_rows_per_arm"] = len(sample) * len(SESSIONS)
    report["unique_exact_pairs"] = len(metric.values)
    publish(report, "complete", started)


if __name__ == "__main__":
    main()
