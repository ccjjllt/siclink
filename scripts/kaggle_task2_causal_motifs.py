"""Test cross-unit directed event traces on the full Task 2 assignment."""

import json
from pathlib import Path
import time

from numba import njit
import numpy as np
from scipy.signal import lfilter
import torch

from glyph_data import as_trial_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, fit_retrieval
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task2_anchored_cdf import evaluation_sample as anchored_cohort
from kaggle_task2_cdf_proxy_cycles import evaluation_sample as proxy_cohort
from kaggle_task2_classifier_ensemble import evaluation_sample as ensemble_cohort
from kaggle_task2_day_shift import cdf_scores, hybrid_scores, sample_glyphs as shift_cohorts
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics
from kaggle_task2_rank_marginals import EPOCHS, MODEL_SEED, MOUSE, TEMPERATURE, V9, evaluation_glyphs, summarize, whiten
from kaggle_task2_rank_marginals_recheck import PROVENANCE, evaluation_sample as rank_recheck, fit_marginal_templates
from kaggle_task2_repeat_precision import evaluation_sample as precision_cohort
from kaggle_task3_conditional_synchrony import read_training
from kaggle_v8_metric_validation import SESSIONS


SEED = 286523
HELD_DAYS = ("session2", "session4")
SCORE_GLYPHS, AXES = 128, 6
ONSET_MS, WINDOW_MS, COARSE_MS = 50, 250, 25
TAU_MS, WINDOWS_MS = (3, 10, 28), ((50, 125), (125, 250))
MOTIF_WEIGHT = 0.10
CDF, REAL, SHUFFLED, RATE = "stock_cdf", "real_directed_motifs", "shuffled_directed_motifs", "rate_only_motifs"
EXACT_ARMS = (V9, CDF, REAL, SHUFFLED, RATE)
OUTPUT = Path("/kaggle/working/task2_causal_motifs.json")


def evaluation_sample(size):
    first, _ = evaluation_glyphs(size)
    repeated, _ = rank_recheck(size)
    anchor, _ = anchored_cohort(size)
    followup, _ = anchored_cohort(size, followup=True)
    shift, oracle = shift_cohorts(size)
    ensemble, _ = ensemble_cohort(size)
    precision, _ = precision_cohort(size)
    proxy, _ = proxy_cohort(size)
    older = {"cdf_first": first.numpy(), "cdf_recheck": repeated.numpy(), "anchored": anchor.numpy(),
             "anchored_followup": followup.numpy(), "day_shift": shift.numpy(), "day_shift_oracle": oracle.numpy(),
             "classifier_ensemble": ensemble.numpy(), "repeat_precision": precision.numpy(), "proxy_cycles": proxy.numpy()}
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(tuple(older.values()))))
    selected = np.sort(np.random.default_rng(SEED).choice(eligible, SCORE_GLYPHS, replace=False))
    return torch.from_numpy(selected), {name: int(np.intersect1d(rows, selected).size) for name, rows in older.items()}


@njit
def permute_slots(times, groups, seed):
    """Shuffle existing millisecond slots across units within each trial's coarse bin."""
    np.random.seed(seed)
    shuffled = times.copy()
    start = 0
    while start < len(times):
        end = start + 1
        while end < len(times) and groups[end] == groups[start]:
            end += 1
        for stop in range(end - 1, start, -1):
            index = np.random.randint(start, stop + 1)
            shuffled[index], shuffled[stop] = shuffled[stop], shuffled[index]
        start = end
    return shuffled


def load_events(root, catalog):
    samples_per_ms, remainder = divmod(catalog["sample_rate_hz"], 1000)
    if remainder or catalog["window_ms"] != WINDOW_MS:
        raise ValueError("Expected exact millisecond bins through 250 ms.")
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE and entry["task"] == "train"),
                     key=lambda entry: entry["session"])
    days, counts, invariants = {}, {}, {}
    n_bins = WINDOW_MS // COARSE_MS
    for index, entry in enumerate(entries):
        day = entry["session"]
        raw = read_training(root, entry, samples_per_ms)
        n_trials, units = raw.counts.shape[:2]
        addresses = (raw.row * units + raw.units) * n_bins + raw.times // COARSE_MS
        coarse = np.bincount(addresses, minlength=n_trials * units * n_bins).reshape(n_trials, units, n_bins)
        if not np.array_equal(coarse[..., ONSET_MS // COARSE_MS:], raw.counts):
            raise ValueError("The original events and stored 25 ms counts must agree.")
        groups = raw.row * n_bins + raw.times // COARSE_MS
        order = np.argsort(groups, kind="stable")
        rows, columns, times = raw.row[order], raw.units[order], raw.times[order]
        shuffled = permute_slots(times, groups[order], SEED + 1009 * index)
        if not np.array_equal(times // COARSE_MS, shuffled // COARSE_MS):
            raise ValueError("Moving each event inside its original bin must preserve every unit count.")
        population = np.bincount(rows * WINDOW_MS + times, minlength=n_trials * WINDOW_MS)
        population_shuffled = np.bincount(rows * WINDOW_MS + shuffled, minlength=n_trials * WINDOW_MS)
        if not np.array_equal(population, population_shuffled):
            raise ValueError("The shuffle must preserve each trial's exact population PSTH.")
        days[day] = (rows, columns, times, shuffled, raw.trials, coarse.astype(np.float32))
        counts[day] = torch.from_numpy(coarse[raw.trials].astype(np.float32)).flatten(-2)
        invariants[day] = {"trials": n_trials, "units": units, "events": len(times),
                           "changed_1ms_event_slot_fraction": float(np.mean(times != shuffled)),
                           "exact_unit_counts_25ms_preserved": True, "exact_trial_population_1ms_histograms_preserved": True}
    return days, counts, invariants


def source_axes(sources):
    """Estimate stable rate coordinates from cross-day products of labeled source glyph means."""
    coordinates = []
    for counts in sources.values():
        rate = counts.unflatten(-1, (-1, 10))[..., ONSET_MS // COARSE_MS:].sum(-1)
        calibrated = (rate - rate.mean(dim=(0, 1))) / (rate.var(dim=(0, 1), unbiased=False) + 1).sqrt()
        coordinates.append(calibrated.mean(1))
    stack = torch.stack(coordinates)
    summed = stack.sum(0)
    covariance = summed.T @ summed - (stack.mT @ stack).sum(0)
    covariance /= len(stack) * (len(stack) - 1) * len(summed)
    eigenvalues, directions = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    return directions[:, -AXES:].contiguous().numpy(), eigenvalues[-AXES:].tolist()


@njit
def compressed_events(rows, units, times, projection, n_trials):
    compressed = np.zeros((n_trials, WINDOW_MS - ONSET_MS, projection.shape[1]), dtype=np.float32)
    for index in range(len(times)):
        if times[index] >= ONSET_MS:
            for coordinate in range(projection.shape[1]):
                compressed[rows[index], times[index] - ONSET_MS, coordinate] += projection[units[index], coordinate]
    return compressed


def directed_features(activities):
    upper = np.triu_indices(AXES, k=1)
    features = []
    for tau in TAU_MS:
        decay = np.float32(np.exp(-1 / tau))
        # The first numerator coefficient of zero excludes the current time bin.
        previous = lfilter(np.asarray([0, 1 - decay], dtype=np.float32), np.asarray([1, -decay], dtype=np.float32),
                           activities, axis=1)
        for start, stop in WINDOWS_MS:
            left, right = start - ONSET_MS, stop - ONSET_MS
            directed = previous[:, left:right].swapaxes(1, 2) @ activities[:, left:right]
            features.append((directed - directed.swapaxes(1, 2))[:, upper[0], upper[1]] / (stop - start))
    return np.concatenate(features, axis=1)


def motif_recording(events, directions):
    rows, units, times, shuffled, trials, coarse = events
    evoked = coarse[..., ONSET_MS // COARSE_MS:]
    scale = (evoked.sum(-1).var(0) + 1) ** 0.5
    projection = np.ascontiguousarray(directions / scale[:, None])
    uniform = np.repeat((evoked.swapaxes(1, 2) @ projection) / COARSE_MS, COARSE_MS, axis=1)
    center = uniform.mean(axis=0, keepdims=True)
    rate = directed_features(uniform - center)
    del uniform
    variants = {RATE: rate}
    for name, series in ((REAL, times), (SHUFFLED, shuffled)):
        compressed = compressed_events(rows, units, series, projection, len(coarse))
        compressed -= center
        variants[name] = directed_features(compressed) - rate
    return {name: torch.from_numpy(features[trials]) for name, features in variants.items()}


def standardized(values):
    variance = values.var(dim=(0, 1), unbiased=False)
    return ((values - values.mean(dim=(0, 1))) / (variance + 0.10 * variance.mean().clamp_min(1e-10)).sqrt()).clamp(-8, 8)


def motif_logits(sources, query):
    source = torch.stack([standardized(values) for values in sources])
    means = source.mean(2)
    center = means.mean(0)
    n_days, n_glyphs, repeats, dimensions = source.shape
    repeat = (source - means[:, :, None]).reshape(-1, dimensions)
    across = (means - center).flatten(0, 1)
    predictive = repeat.T @ repeat / (n_days * n_glyphs * (repeats - 1) * repeats)
    predictive.addmm_(across.T, across, alpha=0.35 / ((n_days - 1) * n_glyphs))
    predictive.diagonal().add_(0.20)
    factor = torch.linalg.cholesky(predictive)
    source_bank = whiten(center, factor)
    observed = whiten(standardized(query).mean(1), factor)
    return logit_scale(observed @ source_bank.T - 0.5 * source_bank.square().sum(1))


def publish(report, phase, started):
    report.update(stage=phase, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": phase, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU notebook with authorized training data.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    events, recordings, invariants = load_events(root, load_catalog(root))
    if len(images) != 2748 or tuple(recordings) != SESSIONS or any(values.shape[:2] != (len(images), 4)
                                                                  for values in recordings.values()):
        raise ValueError("Require four complete labeled recordings with four real repeats per glyph.")
    sample, overlap = evaluation_sample(len(images))
    metric = PairedOCF(official_scorer(root), images, (images[sample] >= 128).numpy(), device, 12)
    rng = np.random.default_rng(SEED + 1)
    report = {"mouse": MOUSE, "seed": SEED, "held_days": HELD_DAYS, "gallery_size": len(images),
              "labeled_days_per_fold": 3, "target_repeats": 4, "stationary_axes_per_fold": AXES,
              "trace_tau_ms": TAU_MS, "readout_windows_ms": WINDOWS_MS,
              "feature_dimensions_per_arm": len(TAU_MS) * len(WINDOWS_MS) * AXES * (AXES - 1) // 2,
              "motif_score_weight": MOTIF_WEIGHT, "archive_provenance": PROVENANCE,
              "scored_glyph_indices": sample.tolist(), "overlap_by_earlier_cohort": overlap, "formal_test_accessed": False,
              "primary_arm": REAL, "paired_reference": SHUFFLED, "control_invariants": invariants, "recordings": {},
              "protocol": "Compress real 1 ms events into six source-only cross-day stable signed population "
                          "coordinates. Extract antisymmetric strictly causal trace/current-population products. "
                          "The temporal arms subtract the same feature from a uniform 25 ms count reconstruction. "
                          "Shuffle actual timestamp slots within each trial and 25 ms window, preserving every "
                          "unit's count and the population's entire 1 ms histogram. Add each equal-dimensional "
                          "separately whitened analog to a fixed v9 plus 25% CDF score at weight 0.10. "
                          "All arms solve over the complete gallery. The directed statistic measures ordering of "
                          "subpopulation activity and does not establish synaptic connectivity."}
    publish(report, "inputs_ready", started)
    exact = {name: [] for name in EXACT_ARMS}
    for day in HELD_DAYS:
        sources = {name: counts for name, counts in recordings.items() if name != day}
        order = torch.from_numpy(rng.permutation(len(images)))
        query = recordings[day][order]
        publish(report, f"{day}_v9_fitting", started)
        torch.manual_seed(MODEL_SEED)
        with torch.enable_grad():
            retrieval = fit_retrieval(as_trial_blocks(sources), len(images), images=images, device=device,
                                      epochs=EPOCHS, seed=MODEL_SEED)
        hierarchy = fit_hierarchical_gallery(sources, view="total")
        hybrid = hybrid_scores(retrieval, hierarchy, query)
        fixed = cdf_scores(hybrid, query, fit_marginal_templates(sources))
        directions, spectrum = source_axes(sources)
        publish(report, f"{day}_extracting_causal_features", started)
        motifs = {name: motif_recording(raw, directions) for name, raw in events.items()}
        scores = {V9: hybrid, CDF: fixed}
        for arm in (REAL, SHUFFLED, RATE):
            features = tuple(motifs[name][arm] for name in sources)
            scores[arm] = fixed + MOTIF_WEIGHT * motif_logits(features, motifs[day][arm][order])
        selected = {name: assign_glyphs(values) for name, values in scores.items()}
        if any(chosen.unique().numel() != len(images) for chosen in selected.values()):
            raise ValueError("All final assignments must use one of every gallery glyph.")
        rows = torch.arange(len(order))
        diagnostics = {}
        for name, values in scores.items():
            density = logit_scale(values) / TEMPERATURE
            diagnostics[name] = {**row_diagnostics(values, order),
                                 "identity_nll_at_fixed_temperature": float((density.logsumexp(1) - density[rows, order]).mean()),
                                 "assigned_top1": float((selected[name] == order).float().mean()),
                                 "changed_vs_cdf": int((selected[name] != selected[CDF]).sum()),
                                 "distinct_outputs": int(selected[name].unique().numel())}
        report["recordings"][day] = {"labeled_sources": tuple(sources), "source_signal_eigenvalues": spectrum,
                                     "full_gallery_diagnostics": diagnostics}
        publish(report, f"{day}_assignments_ready", started)
        sample_rows = order.argsort()[sample]
        measured = {name: metric.evaluate(selected[name][sample_rows, None])[:, 0] for name in EXACT_ARMS}
        for name, values in measured.items():
            exact[name].append(values)
        report["recordings"][day]["ocf_vs_cdf"] = {name: summarize(values, measured[CDF]) for name, values in measured.items()}
        report["recordings"][day]["real_vs_shuffled"] = summarize(measured[REAL], measured[SHUFFLED])
        publish(report, f"{day}_scored", started)
    averages = {name: np.stack(values).mean(0) for name, values in exact.items()}
    report.update(ocf_vs_cdf={name: summarize(values, averages[CDF]) for name, values in averages.items()},
                  real_vs_shuffled=summarize(averages[REAL], averages[SHUFFLED]),
                  real_vs_rate=summarize(averages[REAL], averages[RATE]),
                  ocf_rows_per_arm=len(sample) * len(HELD_DAYS), unique_exact_pairs=len(metric.values))
    publish(report, "complete", started)


if __name__ == "__main__":
    main()
