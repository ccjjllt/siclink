"""Replicate the fixed 25 ms population trajectory score on complementary Task 2 days."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import as_trial_blocks, load_catalog, load_images, locate_data, read_trials, training_groups
from glyph_hierarchical_gallery import fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, fit_retrieval
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task2_anchored_cdf import evaluation_sample as anchored_cohort
from kaggle_task2_causal_motifs import (AXES, CDF, COARSE_MS, MOTIF_WEIGHT, ONSET_MS, RATE, TAU_MS, WINDOWS_MS,
                                        directed_features, evaluation_sample as motif_cohort, motif_logits, source_axes)
from kaggle_task2_cdf_proxy_cycles import evaluation_sample as proxy_cohort
from kaggle_task2_classifier_ensemble import evaluation_sample as ensemble_cohort
from kaggle_task2_day_shift import cdf_scores, hybrid_scores, sample_glyphs as shift_cohorts
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics
from kaggle_task2_rank_marginals import EPOCHS, MODEL_SEED, MOUSE, TEMPERATURE, V9, evaluation_glyphs, summarize
from kaggle_task2_rank_marginals_recheck import PROVENANCE, evaluation_sample as rank_recheck, fit_marginal_templates
from kaggle_task2_repeat_precision import evaluation_sample as precision_cohort
from kaggle_v8_metric_validation import SESSIONS


SEED = 286567
HELD_DAYS = ("session1", "session3")
OCF_GLYPHS, CHECK_TRIALS = 160, 32
SHUFFLED = "within_phase_permuted_counts"
ARMS = (V9, CDF, RATE, SHUFFLED)
OUTPUT = Path("/kaggle/working/task2_count_trajectory_recheck.json")
PRIOR_REPORT_SHA = "c671bc78b9c38143653f03370aca10b09ae0186beff8ca72012cd123c0bcee82"
PHASES = tuple(((start - ONSET_MS) // COARSE_MS, (stop - ONSET_MS) // COARSE_MS) for start, stop in WINDOWS_MS)


def evaluation_sample(size):
    first, _ = evaluation_glyphs(size)
    recheck, _ = rank_recheck(size)
    anchor, _ = anchored_cohort(size)
    followup, _ = anchored_cohort(size, followup=True)
    shift, oracle = shift_cohorts(size)
    ensemble, _ = ensemble_cohort(size)
    precision, _ = precision_cohort(size)
    proxy, _ = proxy_cohort(size)
    motif, _ = motif_cohort(size)
    older = {"cdf_first": first.numpy(), "cdf_recheck": recheck.numpy(), "anchored": anchor.numpy(),
             "anchored_followup": followup.numpy(), "day_shift": shift.numpy(), "day_shift_oracle": oracle.numpy(),
             "classifier_ensemble": ensemble.numpy(), "repeat_precision": precision.numpy(), "proxy_cycles": proxy.numpy(),
             "causal_motifs_primary": motif.numpy()}
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(tuple(older.values()))))
    selected = np.sort(np.random.default_rng(SEED).choice(eligible, OCF_GLYPHS, replace=False))
    return torch.from_numpy(selected), {name: int(np.intersect1d(rows, selected).size) for name, rows in older.items()}


def coarse_days(root, catalog, device, n_glyphs):
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE and entry["task"] == "train"),
                     key=lambda entry: entry["session"])
    original, grouped = {}, {}
    for entry in entries:
        day = entry["session"]
        record = read_trials(root, catalog, entry, device)
        if record.counts.shape[:2] != (n_glyphs * 4, 1):
            raise ValueError("Expected four real labeled single-trial samples per glyph.")
        trial_order = torch.argsort(record.labels, stable=True).reshape(n_glyphs, 4)
        original[day] = (record.counts[:, 0].unflatten(-1, (-1, 10)).numpy(), trial_order.numpy())
        grouped[day] = training_groups(record, n_glyphs)
    if tuple(grouped) != SESSIONS:
        raise ValueError("Require all four labeled days.")
    return original, grouped


def coarse_motifs(activity):
    """Integrate exclusive exponential traces analytically over each uniform 25 ms bin."""
    upper = np.triu_indices(AXES, k=1)
    features = []
    for tau in TAU_MS:
        decay = float(np.float32(np.exp(-1 / tau)))
        block_decay = np.float32(decay ** COARSE_MS)
        geometric_sum = np.float32((1 - decay ** COARSE_MS) / (1 - decay))
        previous = np.zeros_like(activity[:, 0])
        for left, right in PHASES:
            moment = np.zeros((len(activity), AXES, AXES), dtype=activity.dtype)
            for index in range(left, right):
                current = activity[:, index]
                moment += previous[:, :, None] * current[:, None, :] * geometric_sum
                previous = block_decay * previous + (1 - block_decay) * current
            antisymmetric = moment - moment.swapaxes(1, 2)
            features.append(antisymmetric[:, upper[0], upper[1]] / ((right - left) * COARSE_MS))
    return np.concatenate(features, axis=1)


def counts_features(original, directions, rng):
    counts, trials = original
    evoked = counts[..., ONSET_MS // COARSE_MS:]
    scale = (evoked.sum(-1).var(0) + 1) ** 0.5
    projection = np.ascontiguousarray(directions / scale[:, None])
    coarse = (evoked.swapaxes(1, 2) @ projection) / COARSE_MS
    activity = coarse - coarse.mean(0, keepdims=True)
    raw = coarse_motifs(activity)

    # Use only a small reference batch. All fitted and scored features use eight coarse steps.
    dense = directed_features(np.repeat(activity[:CHECK_TRIALS], COARSE_MS, axis=1))
    residual = raw[:CHECK_TRIALS] - dense
    if not np.allclose(raw[:CHECK_TRIALS], dense, rtol=2e-5, atol=2e-7):
        raise ValueError("Analytic rate trajectories do not reproduce the dense reference calculation.")

    n_bins = coarse.shape[1]
    permuted_order = np.empty(coarse.shape[:2], dtype=np.int64)
    for left, right in PHASES:
        permuted_order[:, left:right] = left + rng.random((len(coarse), right - left)).argsort(axis=1)
        if not np.all(np.sort(permuted_order[:, left:right], axis=1) == np.arange(left, right)):
            raise ValueError("The matched shuffle must preserve every vector in its original phase.")
    shuffled = coarse[np.arange(len(coarse))[:, None], permuted_order]
    shuffled_features = coarse_motifs(shuffled - shuffled.mean(0, keepdims=True))
    return {RATE: torch.from_numpy(raw[trials]), SHUFFLED: torch.from_numpy(shuffled_features[trials])}, {
        "checked_reference_trials": CHECK_TRIALS, "max_reference_abs_error": float(np.abs(residual).max()),
        "relative_reference_rms": float(np.sqrt(np.square(residual).mean()) / max(np.sqrt(np.square(dense).mean()), 1e-12)),
        "permuted_25ms_bins_fraction": float(np.mean(permuted_order != np.arange(n_bins))),
        "exact_multiset_preserved_in_each_phase": True}


def publish(report, phase, started):
    report.update(stage=phase, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": phase, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Run in a private Kaggle CPU notebook with the official labeled recordings.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    if len(images) != 2748:
        raise ValueError("Expected the full 2,748-glyph gallery.")
    original, recordings = coarse_days(root, load_catalog(root), device, len(images))
    sample, overlap = evaluation_sample(len(images))
    metric = PairedOCF(official_scorer(root), images, (images[sample] >= 128).numpy(), device, 12)
    rng = np.random.default_rng(SEED + 1)
    report = {"mouse": MOUSE, "seed": SEED, "held_days": HELD_DAYS, "gallery_size": len(images),
              "labeled_days_per_fold": 3, "target_repeats": 4, "axes": AXES, "tau_ms": TAU_MS, "windows_ms": WINDOWS_MS,
              "motif_score_weight": MOTIF_WEIGHT, "prior_report_sha256": PRIOR_REPORT_SHA, "archive_provenance": PROVENANCE,
              "scored_glyph_indices": sample.tolist(), "overlap_by_earlier_cohort": overlap, "formal_test_accessed": False,
              "primary_arm": RATE, "paired_reference": CDF, "recordings": {},
              "protocol": "Replicate the previous fixed 25 ms count trajectory on the other two held recordings "
                          "and new scored glyphs. Compute the same causal products directly over the eight coarse "
                          "vectors via the finite geometric sum of each exponential. Check equivalence against "
                          "the old millisecond expansion on a small bounded sample. Use the same six fitted source "
                          "axes, 90 coordinates, Gaussian gallery, 0.10 score weight, and frozen v9 plus 25% CDF. "
                          "The control permutes the ordering of coarse vectors independently per trial within each "
                          "of the two original phases. Every phase's original multiunit count vectors and total "
                          "counts are kept. Refit the matched scorer in both arms. Use full one-to-one assignment."}
    publish(report, "inputs_ready", started)
    exact = {name: [] for name in ARMS}
    for day in HELD_DAYS:
        sources = {name: counts for name, counts in recordings.items() if name != day}
        order = torch.from_numpy(rng.permutation(len(images)))
        query = recordings[day][order]
        publish(report, f"{day}_fitting", started)
        torch.manual_seed(MODEL_SEED)
        with torch.enable_grad():
            retrieval = fit_retrieval(as_trial_blocks(sources), len(images), images=images, device=device,
                                      epochs=EPOCHS, seed=MODEL_SEED)
        hierarchy = fit_hierarchical_gallery(sources, view="total")
        hybrid = hybrid_scores(retrieval, hierarchy, query)
        fixed = cdf_scores(hybrid, query, fit_marginal_templates(sources))
        directions, spectrum = source_axes(sources)
        features, equivalence = {}, {}
        for name, arrays in original.items():
            control_rng = np.random.default_rng(SEED + 1009 * SESSIONS.index(name))
            features[name], equivalence[name] = counts_features(arrays, directions, control_rng)
        scores = {V9: hybrid, CDF: fixed}
        for arm in (RATE, SHUFFLED):
            training = tuple(features[name][arm] for name in sources)
            scores[arm] = fixed + MOTIF_WEIGHT * motif_logits(training, features[day][arm][order])
        selected = {name: assign_glyphs(matrix) for name, matrix in scores.items()}
        if any(value.unique().numel() != len(images) for value in selected.values()):
            raise ValueError("The full assignment must retain each official glyph exactly once.")
        rows = torch.arange(len(order))
        diagnostics = {}
        for name, values in scores.items():
            density = logit_scale(values) / TEMPERATURE
            diagnostics[name] = {**row_diagnostics(values, order),
                                 "identity_nll_at_fixed_temperature": float((density.logsumexp(1) - density[rows, order]).mean()),
                                 "assigned_top1": float((selected[name] == order).float().mean()),
                                 "changed_vs_cdf": int((selected[name] != selected[CDF]).sum()),
                                 "distinct_outputs": int(selected[name].unique().numel())}
        report["recordings"][day] = {"source_days": tuple(sources), "eigenvalues": spectrum,
                                     "full_gallery": diagnostics, "analytic_equivalence": equivalence}
        publish(report, f"{day}_assignments_ready", started)
        scored_rows = order.argsort()[sample]
        measured = {name: metric.evaluate(chosen[scored_rows, None])[:, 0] for name, chosen in selected.items()}
        for name, values in measured.items():
            exact[name].append(values)
        report["recordings"][day]["ocf_vs_cdf"] = {name: summarize(values, measured[CDF]) for name, values in measured.items()}
        report["recordings"][day]["trajectory_vs_phase_shuffle"] = summarize(measured[RATE], measured[SHUFFLED])
        publish(report, f"{day}_scored", started)
    average = {name: np.stack(values).mean(0) for name, values in exact.items()}
    report.update(ocf_vs_cdf={name: summarize(values, average[CDF]) for name, values in average.items()},
                  trajectory_vs_phase_shuffle=summarize(average[RATE], average[SHUFFLED]),
                  ocf_rows_per_arm=len(sample) * len(HELD_DAYS), unique_exact_pairs=len(metric.values))
    publish(report, "complete", started)


if __name__ == "__main__":
    main()
