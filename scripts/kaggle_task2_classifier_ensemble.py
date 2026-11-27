"""Compare a three-seed v9 classifier ensemble against stock with fixed Task 2 CDF."""

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import (_standardize, _train_classifier, _variance_stabilize, assign_glyphs, combine_logits,
                             fit_retrieval, retrieval_score_components)
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task2_anchored_cdf import evaluation_sample as anchored_cohort
from kaggle_task2_constrained_ceiling import fresh_samples as ceiling_cohorts
from kaggle_task2_day_shift import sample_glyphs as shift_cohorts
from kaggle_task2_native_hanzi import SEED as HANZI_SEED, scoring_glyphs as hanzi_cohorts
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics
from kaggle_task2_rank_marginals import (ADAPTED, BLEND, CEILING_CROWDED, EPOCHS, MODEL_SEED, MOUSE,
                                         TEMPERATURE, V9, evaluation_glyphs, marginal_logits, summarize)
from kaggle_task2_rank_marginals_recheck import PROVENANCE, evaluation_sample as recheck_cohort, fit_marginal_templates
from kaggle_v8_metric_validation import SESSIONS


SEED = 286351
HELD_DAYS = ("session2", "session4")
CLASSIFIER_SEEDS = (MODEL_SEED, 587, 1499)
SCORE_GLYPHS = 128
STOCK_CDF, ENSEMBLE_CDF, OMIT_CDF = "stock_cdf", "ensemble_cdf", "omit_classifier_cdf"
ARMS = (V9, STOCK_CDF, ENSEMBLE_CDF, OMIT_CDF)
OUTPUT = Path("/kaggle/working/task2_classifier_ensemble.json")


def evaluation_sample(size):
    first, _ = evaluation_glyphs(size)
    recheck, _ = recheck_cohort(size)
    anchor, _ = anchored_cohort(size)
    anchor_followup, _ = anchored_cohort(size, followup=True)
    shift, oracle = shift_cohorts(size)
    ceiling, _, _ = ceiling_cohorts(size)
    hanzi, _ = hanzi_cohorts(size, np.random.default_rng(HANZI_SEED))
    older = {"rank_marginals": first.numpy(), "rank_marginals_recheck": recheck.numpy(),
             "anchored": anchor.numpy(), "anchored_followup": anchor_followup.numpy(),
             "day_shift": shift.numpy(), "day_shift_oracle": oracle.numpy(),
             "exact_ceiling_uniform": np.concatenate(tuple(ceiling.values())), "exact_ceiling_crowded": np.asarray(CEILING_CROWDED),
             "native_hanzi": np.concatenate(tuple(hanzi.values()))}
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(tuple(older.values()))))
    selected = np.sort(np.random.default_rng(SEED).choice(eligible, SCORE_GLYPHS, replace=False))
    return torch.from_numpy(selected), {"eligible": len(eligible),
                                        "overlap": {name: int(np.intersect1d(values, selected).size)
                                                    for name, values in older.items()}}


def ensemble_classifier(model, blocks, query, stock, stage):
    """Keep the stock classifier's within-row contrast while averaging independently fitted evidence."""
    features, labels = [], []
    for session, counts, target in blocks:
        fitted = model.sessions[session]
        features.append(_standardize(_variance_stabilize(counts, n_bins=model.n_bins), fitted.mean, fitted.scale))
        labels.append(target)
    features, labels = torch.cat(features), torch.cat(labels)
    stock_mean = stock.mean(dim=1, keepdim=True)
    stock_rms = (stock - stock_mean).square().mean(dim=1, keepdim=True).sqrt()
    stock_standard = (stock - stock_mean) / stock_rms.clamp_min(1e-6)
    total = stock_standard.clone()
    correlations = {}
    for seed in CLASSIFIER_SEEDS[1:]:
        stage(seed)
        with torch.enable_grad():
            classifier = _train_classifier(features, labels, len(model.available), model.available, EPOCHS, seed, True)
        candidate = retrieval_score_components(replace(model, network=classifier), query, None, task="task2")["classifier"]
        centered = candidate - candidate.mean(dim=1, keepdim=True)
        z = centered / centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        correlations[str(seed)] = float((z * stock_standard).mean())
        total.add_(z)
    mean_z = total / len(CLASSIFIER_SEEDS)
    return stock_mean + mean_z * stock_rms, {"mean_row_correlation_with_seed_211": correlations,
                                             "mean_ensemble_standardized_row_rms": float(mean_z.square().mean(1).sqrt().mean())}


def publish(report, phase, started):
    report.update(stage=phase, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": phase, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU notebook with official labeled training recordings.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if len(images) != 2748 or tuple(recordings) != SESSIONS or any(part.shape[:2] != (len(images), 4)
                                                                  for part in recordings.values()):
        raise ValueError("Require four labeled recordings with four real repeats per glyph.")
    sample, independence = evaluation_sample(len(images))
    metric = PairedOCF(official_scorer(root), images, (images[sample] >= 128).numpy(), device, 12)
    rng = np.random.default_rng(SEED + 1)
    report = {"mouse": MOUSE, "seed": SEED, "device": str(device), "held_days": HELD_DAYS, "gallery_size": len(images),
              "labeled_sources_per_fold": 3, "measured_repeats": 4, "archive_provenance": PROVENANCE,
              "scored_glyph_indices": sample.tolist(), "scored_cohort": independence, "formal_test_accessed": False,
              "classifier_seeds": CLASSIFIER_SEEDS, "classifier_epochs": EPOCHS, "cdf_weight": BLEND,
              "primary_arm": ENSEMBLE_CDF, "paired_reference": STOCK_CDF, "recordings": {},
              "protocol": "Fit the exact seed-211 v9 retrieval once per recording split. Reuse its fitted input scaling, "
                          "all nonclassifier evidence, and hierarchical evidence unchanged. Fit two additional classifiers "
                          "with the same architecture, labeled sources, augmentation and epoch budget. Take the equal mean "
                          "of three query-row standardized classifier matrices, then restore the seed-211 per-row mean and "
                          "RMS units. Average all model seeds before adding v9 evidence. The stock and ensemble use the "
                          "same target empirical CDF and 25% score weight. The omit-classifier arm re-normalizes the fixed "
                          "remaining contribution. All arms assign the full gallery once and receive paired official OCF "
                          "on the same disjoint sample in independently permuted training days."}
    publish(report, "inputs_ready", started)
    exact = {name: [] for name in ARMS}
    for day in HELD_DAYS:
        sources = {name: values for name, values in recordings.items() if name != day}
        blocks = as_trial_blocks(sources)
        order = torch.from_numpy(rng.permutation(len(images)))
        query = recordings[day][order]
        publish(report, f"{day}_seed_{MODEL_SEED}_fitting", started)
        torch.manual_seed(MODEL_SEED)
        with torch.enable_grad():
            model = fit_retrieval(blocks, len(images), device=device, seed=MODEL_SEED, epochs=EPOCHS, images=images)
        stock = retrieval_score_components(model, query, None, task="task2")
        ensemble, ensemble_diagnostics = ensemble_classifier(
            model, blocks, query, stock["classifier"], lambda seed: publish(report, f"{day}_seed_{seed}_fitting", started))
        hierarchy = fit_hierarchical_gallery(sources, view="total")
        eb = hierarchy.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
        target_cdf = marginal_logits(query, fit_marginal_templates(sources))[ADAPTED]
        hybrid = blend_hierarchical_logits(combine_logits(stock, "task2"), eb)
        ensemble_hybrid = blend_hierarchical_logits(combine_logits({**stock, "classifier": ensemble}, "task2"), eb)
        omit = blend_hierarchical_logits(combine_logits({key: value for key, value in stock.items() if key != "classifier"},
                                                        "task2"), eb)
        scores = {V9: hybrid, **{arm: logit_scale(values).lerp(target_cdf, BLEND)
                                for arm, values in ((STOCK_CDF, hybrid), (ENSEMBLE_CDF, ensemble_hybrid), (OMIT_CDF, omit))}}
        choices = {name: assign_glyphs(values) for name, values in scores.items()}
        if any(value.unique().numel() != len(images) for value in choices.values()):
            raise ValueError("Every assignment must preserve the complete known gallery.")
        truth_rows = torch.arange(len(order))
        diagnostics = {}
        for arm, values in scores.items():
            probability = logit_scale(values) / TEMPERATURE
            diagnostics[arm] = {**row_diagnostics(values, order),
                                "identity_nll_at_fixed_temperature": float((probability.logsumexp(1) - probability[truth_rows, order]).mean()),
                                "assigned_top1": float((choices[arm] == order).float().mean()),
                                "changed_vs_stock_cdf": int((choices[arm] != choices[STOCK_CDF]).sum()),
                                "distinct_outputs": int(choices[arm].unique().numel())}
        report["recordings"][day] = {"source_days": tuple(sources), "identity": diagnostics, "ensemble": ensemble_diagnostics}
        publish(report, f"{day}_assignments_ready", started)
        sample_rows = order.argsort()[sample]
        measured = {name: metric.evaluate(values[sample_rows, None])[:, 0] for name, values in choices.items()}
        for arm, values in measured.items():
            exact[arm].append(values)
        report["recordings"][day]["ocf_vs_stock_cdf"] = {
            arm: summarize(values, measured[STOCK_CDF]) for arm, values in measured.items()}
        publish(report, f"{day}_scored", started)
    means = {name: np.stack(values).mean(axis=0) for name, values in exact.items()}
    report.update(ocf_vs_stock_cdf={name: summarize(values, means[STOCK_CDF]) for name, values in means.items()},
                  ocf_vs_v9={name: summarize(values, means[V9]) for name, values in means.items()},
                  ocf_rows_per_arm=len(sample) * len(HELD_DAYS), unique_exact_pairs=len(metric.values))
    publish(report, "complete", started)


if __name__ == "__main__":
    main()
