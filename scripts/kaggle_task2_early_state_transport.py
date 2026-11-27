"""Adapt evoked rank responses to an anonymous target day's early population state."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, fit_retrieval
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task2_anchored_cdf import evaluation_sample as anchored_cohort
from kaggle_task2_causal_motifs import evaluation_sample as motif_cohort
from kaggle_task2_cdf_proxy_cycles import evaluation_sample as proxy_cohort
from kaggle_task2_classifier_ensemble import evaluation_sample as ensemble_cohort
from kaggle_task2_count_trajectory_recheck import evaluation_sample as trajectory_cohort
from kaggle_task2_day_shift import cdf_scores, hybrid_scores, sample_glyphs as shift_cohorts
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics
from kaggle_task2_rank_marginals import (BLEND, EPOCHS, MODEL_SEED, MOUSE, TEMPERATURE, V9,
                                         evaluation_glyphs, normal_scores, responses, source_cdf, summarize, whiten)
from kaggle_task2_rank_marginals_recheck import PROVENANCE, evaluation_sample as rank_recheck, fit_marginal_templates
from kaggle_task2_repeat_precision import evaluation_sample as precision_cohort
from kaggle_v8_metric_validation import SESSIONS


SEED = 286611
HELD_DAYS = ("session2", "session4")
OCF_GLYPHS, EARLY_AXES = 128, 12
RIDGE, TARGET_MOMENT_WEIGHT = 0.35, 0.65
CDF, REAL, SOURCE, SHUFFLED = "stock_cdf", "target_early_state", "source_early_state", "shuffled_target_early_state"
EXACT_ARMS = (V9, CDF, REAL, SOURCE, SHUFFLED)
OUTPUT = Path("/kaggle/working/task2_early_state_transport.json")


def evaluation_sample(size):
    first, _ = evaluation_glyphs(size)
    recheck, _ = rank_recheck(size)
    anchor, _ = anchored_cohort(size)
    followup, _ = anchored_cohort(size, followup=True)
    shift, oracle = shift_cohorts(size)
    ensemble, _ = ensemble_cohort(size)
    precision, _ = precision_cohort(size)
    proxy, _ = proxy_cohort(size)
    motifs, _ = motif_cohort(size)
    trajectory, _ = trajectory_cohort(size)
    older = {"cdf_first": first.numpy(), "cdf_recheck": recheck.numpy(), "anchored": anchor.numpy(),
             "anchored_followup": followup.numpy(), "day_shift": shift.numpy(), "day_shift_oracle": oracle.numpy(),
             "classifier_ensemble": ensemble.numpy(), "repeat_precision": precision.numpy(), "proxy_cycles": proxy.numpy(),
             "causal_motifs": motifs.numpy(), "count_trajectory_recheck": trajectory.numpy()}
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(tuple(older.values()))))
    selected = np.sort(np.random.default_rng(SEED).choice(eligible, OCF_GLYPHS, replace=False))
    return torch.from_numpy(selected), {name: int(np.intersect1d(rows, selected).size) for name, rows in older.items()}


def early_late(counts):
    bins = counts.unflatten(-1, (-1, 10))
    early_units = bins[..., 0]
    early = torch.cat((early_units.sqrt(), early_units.sum(-1, keepdim=True).sqrt()), dim=-1)
    return early, bins[..., 2:].sum(-1)


def calibrated(fit, query):
    mean = fit.mean(dim=(0, 1))
    scale = (fit.var(dim=(0, 1), unbiased=False) + 0.05).sqrt()
    return (query - mean) / scale


def repeat_residuals(values):
    residual = values - values.mean(1, keepdim=True)
    return residual.flatten(0, 1) / (len(values) * (values.shape[1] - 1)) ** 0.5


def source_innovation(sources, factor):
    raw = tuple(early_late(values) for values in sources.values())
    scaled = tuple((calibrated(early, early), calibrated(late, late)) for early, late in raw)
    residual = tuple(repeat_residuals(early[..., :-1]) for early, _ in scaled)
    covariance = torch.stack([value.T @ value for value in residual]).mean(0)
    _, axes = torch.linalg.eigh(covariance)
    axes = axes[:, -EARLY_AXES:].contiguous()
    x = tuple(state_coordinates(early, axes) for early, _ in scaled)
    moments = tuple(regression_moments(early, late) for early, (_, late) in zip(x, scaled))
    gram, cross = (torch.stack(value).mean(0) for value in zip(*moments))
    coefficient = solve_map(gram, cross)
    raw_ranks = tuple(pooled_ranks((late - early @ coefficient).mean(1)) for early, (_, late) in zip(x, scaled))
    gallery = whiten(torch.stack(raw_ranks).mean(0), factor)
    return axes, gallery, coefficient, gram, cross


def state_coordinates(early, axes):
    return torch.cat((early[..., :-1] @ axes, early[..., -1:]), dim=-1)


def regression_moments(early, late):
    x, y = repeat_residuals(early), repeat_residuals(late)
    return x.T @ x, x.T @ y


def solve_map(gram, cross):
    regularized = gram.clone()
    regularized.diagonal().add_(RIDGE * gram.diagonal().mean().clamp_min(0.02))
    return torch.linalg.solve(regularized, cross)


def pooled_ranks(values):
    return normal_scores(values, source_cdf(values))


def rank_scores(values, gallery, factor):
    observed = whiten(pooled_ranks(values), factor)
    return logit_scale(observed @ gallery.T - 0.5 * gallery.square().sum(1))


def transport_deltas(query, split, templates, state, rng):
    """Learn target-state coupling on one half and score the other; swap and average."""
    _, null_gallery, factor, _ = templates
    axes, gallery, source_coefficient, source_gram, source_cross = state
    halves = query.gather(1, split[..., None].expand_as(query)).split(2, dim=1)
    observables = tuple(early_late(counts) for counts in halves)
    deltas, validation = {}, []
    for fit in range(2):
        score = 1 - fit
        early_fit, late_fit = observables[fit]
        early_score, late_score = observables[score]
        fit_states = state_coordinates(calibrated(early_fit, early_fit), axes)
        score_states = state_coordinates(calibrated(early_fit, early_score), axes)
        fit_rates, score_rates = calibrated(late_fit, late_fit), calibrated(late_fit, late_score)
        target_gram, target_cross = regression_moments(fit_states, fit_rates)
        # A swap flips the two-repeat early difference while preserving both raw vector histograms.
        sign = torch.from_numpy((1 - 2 * (rng.permutation(len(query)) % 2)).astype(np.float32))
        shuffled_states = fit_states - fit_states.mean(1, keepdim=True)
        _, shuffled_cross = regression_moments(shuffled_states * sign[:, None, None], fit_rates)
        gram = source_gram.lerp(target_gram, TARGET_MOMENT_WEIGHT)
        coefficients = {SOURCE: source_coefficient,
                        REAL: solve_map(gram, source_cross.lerp(target_cross, TARGET_MOMENT_WEIGHT)),
                        SHUFFLED: solve_map(gram, source_cross.lerp(shuffled_cross, TARGET_MOMENT_WEIGHT))}
        baseline = rank_scores(responses(halves[score]), null_gallery, factor)
        held_rate_variance = repeat_residuals(score_rates).square().sum().clamp_min(1e-8)
        detail = {}
        for name, coefficient in coefficients.items():
            innovation = score_rates - score_states @ coefficient
            score_matrix = rank_scores(innovation.mean(1), gallery, factor)
            difference = 0.5 * (score_matrix - baseline)
            if name in deltas:
                deltas[name].add_(difference)
            else:
                deltas[name] = difference
            detail[name] = {"held_repeat_variance_explained": float(1 - repeat_residuals(innovation).square().sum() / held_rate_variance),
                            "coefficient_rms": float(coefficient.square().mean().sqrt()),
                            "half_rank_score_correction_rms": float((difference * 2).square().mean().sqrt())}
        validation.append(detail)
    return deltas, validation


def publish(report, stage, started):
    report.update(stage=stage, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": stage, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU notebook with authorized labeled recordings.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if len(images) != 2748 or tuple(recordings) != SESSIONS or any(values.shape[:2] != (len(images), 4)
                                                                  for values in recordings.values()):
        raise ValueError("Require four whole labeled recording days and four measured repeats per glyph.")
    sample, overlap = evaluation_sample(len(images))
    metric = PairedOCF(official_scorer(root), images, (images[sample] >= 128).numpy(), device, 12)
    rng = np.random.default_rng(SEED + 1)
    report = {"mouse": MOUSE, "seed": SEED, "held_days": HELD_DAYS, "gallery_size": len(images),
              "labeled_days_per_fold": 3, "target_repeats": 4, "archive_provenance": PROVENANCE,
              "scored_glyph_indices": sample.tolist(), "overlap_by_earlier_cohort": overlap, "formal_test_accessed": False,
              "early_window_ms": (0, 25), "evoked_window_ms": (50, 250), "source_early_axes": EARLY_AXES,
              "target_moment_weight": TARGET_MOMENT_WEIGHT, "ridge": RIDGE, "cdf_weight": BLEND,
              "primary_arm": REAL, "paired_reference": CDF, "recordings": {},
              "protocol": "Estimate a conditional population-state correction from within-query repeat residuals. "
                          "Source early activity defines twelve modes plus the global early count. The regression "
                          "for each target half pools source and anonymous target residual covariances, then scores "
                          "only the other repeats. The trial-misalignment control swaps early vectors within exactly "
                          "half of the repeated query pairs, preserving both marginals and glyph grouping. Use a "
                          "shared conditioned source gallery, the fixed original CDF whitening, and average the "
                          "two CDF score differences without changing the full four-repeat v9+25% CDF anchor. "
                          "The source-only arm leaves the target slope unchanged. Solve a full one-to-one assignment."}
    publish(report, "inputs_ready", started)
    exact = {name: [] for name in EXACT_ARMS}
    for day in HELD_DAYS:
        source = {name: values for name, values in recordings.items() if name != day}
        order = torch.from_numpy(rng.permutation(len(images)))
        query = recordings[day][order]
        split = torch.from_numpy(rng.random(query.shape[:2]).argsort(axis=1))
        publish(report, f"{day}_fitting", started)
        torch.manual_seed(MODEL_SEED)
        with torch.enable_grad():
            retrieval = fit_retrieval(as_trial_blocks(source), len(images), device=device, seed=MODEL_SEED,
                                      epochs=EPOCHS, images=images)
        hierarchy = fit_hierarchical_gallery(source, view="total")
        templates = fit_marginal_templates(source)
        hybrid = hybrid_scores(retrieval, hierarchy, query)
        fixed = cdf_scores(hybrid, query, templates)
        state = source_innovation(source, templates[2])
        deltas, validation = transport_deltas(query, split, templates, state, rng)
        scores = {V9: hybrid, CDF: fixed, **{name: fixed + BLEND * delta for name, delta in deltas.items()}}
        choices = {name: assign_glyphs(values) for name, values in scores.items()}
        if any(chosen.unique().numel() != len(images) for chosen in choices.values()):
            raise ValueError("Every endpoint must retain each glyph exactly once.")
        diagnostics = {}
        rows = torch.arange(len(order))
        for name, values in scores.items():
            density = logit_scale(values) / TEMPERATURE
            diagnostics[name] = {**row_diagnostics(values, order),
                                 "identity_nll": float((density.logsumexp(1) - density[rows, order]).mean()),
                                 "assigned_top1": float((choices[name] == order).float().mean()),
                                 "changed_vs_cdf": int((choices[name] != choices[CDF]).sum()),
                                 "distinct_outputs": int(choices[name].unique().numel())}
        report["recordings"][day] = {"source_days": tuple(source), "identity": diagnostics, "validation_halves": validation}
        publish(report, f"{day}_assignments_ready", started)
        sample_rows = order.argsort()[sample]
        measured = {name: metric.evaluate(chosen[sample_rows, None])[:, 0] for name, chosen in choices.items()}
        for name, values in measured.items():
            exact[name].append(values)
        report["recordings"][day]["ocf_vs_cdf"] = {name: summarize(values, measured[CDF]) for name, values in measured.items()}
        publish(report, f"{day}_scored", started)
    averages = {name: np.stack(values).mean(0) for name, values in exact.items()}
    report.update(ocf_vs_cdf={name: summarize(values, averages[CDF]) for name, values in averages.items()},
                  target_vs_source=summarize(averages[REAL], averages[SOURCE]),
                  target_vs_shuffled=summarize(averages[REAL], averages[SHUFFLED]),
                  ocf_rows_per_arm=len(sample) * len(HELD_DAYS), unique_exact_pairs=len(metric.values))
    publish(report, "complete", started)


if __name__ == "__main__":
    main()
