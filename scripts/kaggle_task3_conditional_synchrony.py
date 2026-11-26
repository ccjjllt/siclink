"""Test repeatable glyph synchrony beyond unit timing and population activity.

Each timing null preserves all unit-by-25 ms counts and the full trial's 1 ms
population histogram. The same-glyph split-half statistic also subtracts
coincidences between distinct, latency-aligned presentations of that glyph.
The unit and pair selection precede evaluation on entirely held glyphs.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

from numba import njit
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

from glyph_data import load_catalog, locate_data


MOUSE = "mice1"
MILLISECONDS = 250
BIN_WIDTH = 25
ONSET = 50
LAG_LIMIT = 4
N_UNITS = 24
N_PAIRS = 32
NEIGHBORS = 8
LAGS = np.arange(-4, 5)
PSTH_PERIODS = np.arange(8 * BIN_WIDTH)


@dataclass
class Recording:
    trials: np.ndarray
    row: np.ndarray
    times: np.ndarray
    units: np.ndarray
    counts: np.ndarray


def read_training(root, entry, samples_per_ms):
    unit_count = entry["n_units"]
    with np.load(root / entry["path"]) as records:
        labels = records["target"]
        repeats = np.bincount(labels)
        size = np.diff(records["trial_ptr"])
        if not np.all(repeats == 4) or len(labels) != len(size) or not np.all(np.diff(records["group_ptr"]) == 1):
            raise ValueError("Expected four one-trial labeled samples per glyph.")
        times = (records["spike_samples"] // samples_per_ms).astype(np.int16)
        units = records["unit_index"].astype(np.int16)
    if times.max() >= MILLISECONDS or times.min() < 0:
        raise ValueError("Spike times must lie inside the 250 ms window.")
    row = np.repeat(np.arange(len(labels), dtype=np.int32), size)
    trials = np.argsort(labels, kind="stable").reshape(-1, 4)
    use = times >= ONSET
    address = (row[use] * unit_count + units[use]) * 8 + (times[use] - ONSET) // BIN_WIDTH
    counts = np.bincount(address, minlength=len(labels) * unit_count * 8).reshape(-1, unit_count, 8).astype(np.int16)
    return Recording(trials, row, times, units, counts)


def pick_units(recordings, fit):
    signal, noise, exposure = [], [], []
    for record in recordings.values():
        evoked = record.counts[record.trials[fit]].sum(axis=-1).astype(np.float32)
        stable = np.sqrt(evoked)
        scatter = stable.var(axis=1, ddof=1).mean(axis=0)
        signal.append(np.maximum(stable.mean(axis=1).var(axis=0) - scatter / 4, 0))
        noise.append(scatter)
        exposure.append(evoked.mean(axis=(0, 1)))
    score = np.mean(signal, axis=0) / (0.10 + np.mean(noise, axis=0))
    score *= np.sqrt(np.mean(exposure, axis=0))
    chosen = np.sort(np.argsort(score)[-N_UNITS:])
    return chosen, float(np.median(score[chosen]))


def session_propensities(record, chosen, fit):
    fit_trials = np.zeros(len(record.counts), dtype=bool)
    fit_trials[record.trials[fit].ravel()] = True
    use = fit_trials[record.row]
    units = record.counts.shape[1]
    spikes = record.units[use].astype(np.int32) * MILLISECONDS + record.times[use]
    hist = np.bincount(spikes, minlength=units * MILLISECONDS).reshape(units, MILLISECONDS).astype(np.float64)
    smoothed = gaussian_filter1d(hist, sigma=2.0, axis=1, mode="nearest")
    coarse = hist.reshape(units, 10, BIN_WIDTH).mean(axis=2).repeat(BIN_WIDTH, axis=1)
    absolute = np.maximum(0.85 * smoothed + 0.15 * coarse, 0.5)
    outside = np.ones(units, dtype=bool)
    outside[chosen] = False
    background = absolute[outside].sum(axis=0)
    onset = np.arange(25, 125)
    hypotheses = np.maximum(background[onset[None] - LAGS[:, None]], 1)
    hypotheses /= hypotheses.sum(axis=1, keepdims=True)
    background_events = outside[record.units]
    address = record.row[background_events] * MILLISECONDS + record.times[background_events]
    measured = np.bincount(address, minlength=len(record.counts) * MILLISECONDS).reshape(-1, MILLISECONDS)
    measured = measured[:, onset]
    latency_scores = measured @ np.log(hypotheses).T - 0.5 * (LAGS / 2.0)**2
    latency = np.where(measured.sum(axis=1) >= 8, LAGS[latency_scores.argmax(axis=1)], 0).astype(np.int16)
    stretched = (PSTH_PERIODS[None] + ONSET - LAGS[:, None]).clip(0, MILLISECONDS - 1)
    weights = np.log(absolute[chosen][:, stretched].transpose(1, 0, 2))
    return np.ascontiguousarray(weights), latency


def conditional_events(record, chosen, trials):
    unit_index = np.full(record.counts.shape[1], -1, dtype=np.int16)
    unit_index[chosen] = np.arange(len(chosen), dtype=np.int16)
    requested = np.zeros(len(record.counts), dtype=bool)
    requested[trials.ravel()] = True
    selected = (unit_index[record.units] >= 0) & (record.times >= ONSET) & requested[record.row]
    rows, times = record.row[selected], record.times[selected] - ONSET
    order = np.lexsort((times, rows))
    offsets = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=len(record.counts)))))
    return offsets, times[order], unit_index[record.units[selected][order]]


@njit
def randomize_labels(labels, times, weights, sweeps):
    """A symmetric slot swap with Metropolis acceptance on the two fixed margins."""
    accepted = eligible = attempted = 0
    start = 0
    while start < len(labels):
        block = times[start] // BIN_WIDTH
        stop = start + 1
        while stop < len(labels) and times[stop] // BIN_WIDTH == block:
            stop += 1
        n = stop - start
        if n > 1:
            for _ in range(sweeps * n):
                i = np.random.randint(start, stop)
                j = np.random.randint(start, stop - 1)
                if j >= i:
                    j += 1
                attempted += 1
                u, v = labels[i], labels[j]
                t, s = times[i], times[j]
                if u == v or t == s:
                    continue
                eligible += 1
                ratio = weights[u, s] + weights[v, t] - weights[u, t] - weights[v, s]
                if ratio >= 0 or np.random.random() < np.exp(ratio):
                    labels[i], labels[j] = v, u
                    accepted += 1
        start = stop
    return accepted, eligible, attempted


@njit
def count_pairs(first_t, first_u, second_t, second_u, latency, within, pair_id, values):
    """Symmetric nonzero lag counts, without crossing the original coarse-bin edges."""
    begin = 0
    for i in range(len(first_t)):
        time = first_t[i]
        lower = time + latency - LAG_LIMIT
        upper = time + latency + LAG_LIMIT
        if within:
            begin = i + 1
        while begin < len(second_t) and second_t[begin] < lower:
            begin += 1
        j = begin
        while j < len(second_t) and second_t[j] <= upper:
            other = second_t[j]
            if other != time + latency and time // BIN_WIDTH == other // BIN_WIDTH:
                pair = pair_id[first_u[i], second_u[j]]
                if pair >= 0:
                    values[pair] += 1
            j += 1


@njit
def render_repeats(trials, offsets, times, units, latency, weights, pair_id, pair_count, controls, sweeps, seed):
    np.random.seed(seed)
    width = controls + 1
    fields = np.zeros((width, len(trials), 2, 2, pair_count), dtype=np.float32)
    progress = np.zeros(5, dtype=np.int64)
    for glyph in range(len(trials)):
        for half in range(2):
            a, b = trials[glyph, 2 * half: 2 * half + 2]
            ta, tb = times[offsets[a]:offsets[a + 1]], times[offsets[b]:offsets[b + 1]]
            ua, ub = units[offsets[a]:offsets[a + 1]], units[offsets[b]:offsets[b + 1]]
            wa, wb = weights[latency[a] + 4], weights[latency[b] + 4]
            pivot_a, pivot_b = ua.copy(), ub.copy()
            randomize_labels(pivot_a, ta, wa, sweeps)
            randomize_labels(pivot_b, tb, wb, sweeps)
            for member in range(width):
                if member == 0:
                    left, right = ua, ub
                else:
                    left, right = pivot_a.copy(), pivot_b.copy()
                    accepted, eligible, attempted = randomize_labels(left, ta, wa, sweeps)
                    added, possible, tested = randomize_labels(right, tb, wb, sweeps)
                    progress[0] += accepted + added
                    progress[1] += eligible + possible
                    progress[2] += attempted + tested
                    progress[3] += (left != ua).sum() + (right != ub).sum()
                    progress[4] += len(left) + len(right)
                # The crossed term has two orientations and receives the same half factor.
                within = fields[member, glyph, half, 0]
                crossed = fields[member, glyph, half, 1]
                count_pairs(ta, left, ta, left, 0, True, pair_id, within)
                count_pairs(tb, right, tb, right, 0, True, pair_id, within)
                count_pairs(ta, left, tb, right, latency[b] - latency[a], False, pair_id, crossed)
    return fields * 0.5, progress


def rate_covariates(record, trials, latency, fit_size):
    counts = record.counts[trials].reshape(len(trials), 2, 2, -1)
    rates = np.sqrt(counts.mean(axis=2, dtype=np.float32))
    post = record.times >= ONSET
    address = record.row[post] * 40 + (record.times[post] - ONSET) // 5
    histogram = np.bincount(address, minlength=len(record.counts) * 40).reshape(-1, 40)
    population = histogram[trials].reshape(len(trials), 2, 2, 40).mean(axis=2, dtype=np.float32)
    population /= population.sum(axis=-1, keepdims=True).clip(1)
    arrival = latency[trials].reshape(len(trials), 2, 2).mean(axis=2, dtype=np.float32)[:, :, None]
    covariates = []
    for values, minimum_scale, fraction in ((rates, 0.1, 0.80), (population, 0.015, 0.15), (arrival, 1.0, 0.05)):
        scale = values[:fit_size].std(axis=(0, 1)).clip(minimum_scale)
        covariates.append(values / scale * np.sqrt(fraction / values.shape[-1]))
    return np.concatenate(covariates, axis=-1)


def matched_neighbors(covariates):
    x, y = (torch.from_numpy(rows).float() for rows in covariates.transpose(1, 0, 2))
    distance = x.square().sum(dim=1)[:, None] + y.square().sum(dim=1)[None] - 2 * (x @ y.T)
    same_distance = distance.diagonal().clone().clamp_min(0)
    distance.fill_diagonal_(torch.inf)
    ab, ba = distance.topk(NEIGHBORS, dim=1, largest=False), distance.T.topk(NEIGHBORS, dim=1, largest=False)
    report = {"mean_same_glyph_rate_distance": float(same_distance.mean()),
              "mean_matched_other_glyph_rate_distance": float((ab.values.mean() + ba.values.mean()) / 2)}
    return (ab.indices.numpy(), ba.indices.numpy()), report


def centered_features(values, fit_count):
    width = len(values) - 1
    residuals = values * (1 + 1 / width) - values.sum(axis=0, keepdims=True) / width
    training = residuals[:, :fit_count]
    means = training.mean(axis=(1, 2), keepdims=True)
    scale = training.std(axis=(0, 1, 2), keepdims=True).clip(0.20)
    return (residuals - means) / scale


def same_minus_matched(features, neighbors):
    left, right = features[:, :, 0], features[:, :, 1]
    ab, ba = neighbors
    # Do not materialize the complete glyph-by-glyph matrix for every pair.
    right_other = sum(right[:, ab[:, item]] for item in range(ab.shape[1])) / ab.shape[1]
    left_other = sum(left[:, ba[:, item]] for item in range(ba.shape[1])) / ba.shape[1]
    return (left * (right - right_other) + right * (left - left_other)) * 0.5


def summarize(values):
    contrast = values[0] - values[1:].mean(axis=0)
    observed = values[0].mean()
    conditional = values[1:].mean()
    return {"observed_similarity_gain": float(observed), "conditional_null_similarity_gain": float(conditional),
            "paired_real_minus_null": float(contrast.mean()), "paired_glyph_se": float(contrast.std(ddof=1) / np.sqrt(len(contrast))),
            "glyphs": len(contrast), "null_runs": len(values) - 1}


def run_session(record, fit, selector, outer, chosen, args, seed):
    generator = np.random.default_rng(seed)
    ordered = record.trials[np.concatenate((selector, outer))]
    order = generator.random(ordered.shape).argsort(axis=1)
    trials = np.take_along_axis(ordered, order, axis=1)
    weights, latency = session_propensities(record, chosen, fit)
    offsets, times, units = conditional_events(record, chosen, trials)
    indexes = np.triu_indices(len(chosen), k=1)
    pair_count = len(indexes[0])
    pair_id = np.full((len(chosen), len(chosen)), -1, dtype=np.int16)
    pair_id[indexes] = pair_id[indexes[::-1]] = np.arange(pair_count, dtype=np.int16)
    values, progress = render_repeats(trials, offsets, times, units, latency, weights, pair_id,
                                      pair_count, args.null_controls, args.swap_sweeps, seed)
    fitted = len(selector)
    covariates = rate_covariates(record, trials, latency, fitted)
    fit_neighbors, _ = matched_neighbors(covariates[:fitted])
    held_neighbors, rate_report = matched_neighbors(covariates[fitted:])
    conditioned = centered_features(values[..., 0, :] - values[..., 1, :], fitted)
    within = centered_features(values[..., 0, :], fitted)
    reference_pair = same_minus_matched(conditioned[:, :fitted], fit_neighbors).mean(axis=1)
    support = (values[:, :fitted, :, 0, :] > 0).any(axis=0).mean(axis=(0, 1))
    report = {**rate_report, "mean_selected_unit_spikes_per_trial": len(times) / trials.size,
              "estimated_latency_histogram_ms": dict(zip(LAGS.tolist(), np.bincount(latency[trials].ravel() + 4, minlength=9).tolist())),
              "swap_acceptance_eligible_fraction": float(progress[0] / max(progress[1], 1)),
              "eligible_swap_fraction": float(progress[1] / max(progress[2], 1)),
              "surrogate_unit_labels_changed_fraction": float(progress[3] / max(progress[4], 1))}
    held = {"within_minus_same_glyph_crossed": same_minus_matched(conditioned[:, fitted:], held_neighbors),
            "within_unadjusted_for_crossed": same_minus_matched(within[:, fitted:], held_neighbors)}
    return reference_pair[0] - reference_pair[1:].mean(axis=0), support, held, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12007)
    parser.add_argument("--held-glyphs", type=int, default=320)
    parser.add_argument("--pair-fit-glyphs", type=int, default=512)
    parser.add_argument("--null-controls", type=int, default=9)
    parser.add_argument("--swap-sweeps", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_conditional_synchrony.json"))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a Kaggle CPU Notebook with labeled competition inputs.")
    if min(args.held_glyphs, args.pair_fit_glyphs) <= NEIGHBORS or args.null_controls < 2 or args.swap_sweeps < 1:
        parser.error("Use sufficient distinct glyphs, conditional controls and swap steps.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    samples_per_ms, remainder = divmod(catalog["sample_rate_hz"], 1000)
    if remainder or catalog["window_ms"] != MILLISECONDS:
        raise ValueError("Use an integral 1 ms sample grid on the official 250 ms window.")
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE and entry["task"] == "train"),
                     key=lambda entry: entry["session"])
    recordings = {entry["session"]: read_training(root, entry, samples_per_ms) for entry in entries}
    if len(recordings) != 4:
        raise ValueError("Expected four labeled recording days on the same mouse.")
    count = len(next(iter(recordings.values())).trials)
    order = np.random.default_rng(args.seed).permutation(count)
    outer, fit = order[:args.held_glyphs], order[args.held_glyphs:]
    if len(fit) < args.pair_fit_glyphs or any(len(record.trials) != count for record in recordings.values()):
        raise ValueError("Use a disjoint training and evaluation comparison on aligned labeled glyphs.")
    chosen, reliability = pick_units(recordings, fit)
    scores, support, measured, reports = [], [], {}, {}
    print(f"Fitting conditional nulls and {N_UNITS} unit features on {len(fit)} glyphs", flush=True)
    for day, (name, record) in enumerate(recordings.items()):
        before = time.monotonic()
        pair_fit, pair_support, measured[name], reports[name] = run_session(record, fit, fit[:args.pair_fit_glyphs], outer,
                                                                         chosen, args, args.seed + 101 * day)
        scores.append(pair_fit)
        support.append(pair_support)
        reports[name]["seconds"] = time.monotonic() - before
        print(json.dumps({"session": name, **reports[name]}), flush=True)
    evidence = np.mean(scores, axis=0)
    exposure = np.mean(support, axis=0)
    viable = np.where(exposure >= 0.025)[0]
    selected = viable[np.argsort(evidence[viable])[-N_PAIRS:]]
    primary = "within_minus_same_glyph_crossed"
    aggregated = {}
    if len(selected):
        for key in measured[next(iter(measured))]:
            values = []
            for name, session in measured.items():
                result = session[key][..., selected].mean(axis=-1)
                values.append(result)
                reports[name][key] = summarize(result)
            aggregated[key] = summarize(np.mean(values, axis=0))
    output = {"mouse": MOUSE, "device": "cpu", "seed": args.seed, "fit_glyphs": len(fit), "held_glyphs": len(outer),
              "pair_fit_glyphs": args.pair_fit_glyphs, "selected_units": N_UNITS, "selected_pairs": len(selected),
              "fit_reliability_median": reliability, "eligible_pairs": len(viable),
              "fit_selected_pair_exposure_mean": float(exposure[selected].mean()) if len(selected) else None,
              "fit_selected_pair_signal_mean": float(evidence[selected].mean()) if len(selected) else None,
              "lags_ms": [1, LAG_LIMIT],
              "analysis_window_ms": [ONSET, MILLISECONDS], "nulled_margins_ms": {"per_unit": 25, "population": 1},
              "swap_sweeps": args.swap_sweeps, "primary": primary, "sessions": reports, "aggregate": aggregated,
              "protocol": "Whole-glyph holdout, two independent measured repeats per half. Fixed coarse rates, "
                          "population PSTH, and fit-only smoothed unit-specific fine rates weighted by latency "
                          "measured on non-selected units. Raw same-trial coincidences at nonzero 1–4 ms lags "
                          "subtract two crossed orientations from other presentations of the same glyph, with "
                          "estimated common latency alignment. This removes repeatable marginal signal preserved "
                          "by crossed trials. Primary statistic is the same-glyph inner product advantage over "
                          "independently rate/PSTH-matched different glyphs, compared with conditional controls. "
                          "SE averages four day effects by glyph before clustering.",
              "null_approximation": "Finite reversible Metropolis chains keep spike slots and within-25 ms cell "
                                    "counts exactly fixed. Observed and null members branch through a common random "
                                    "ancestor so they are exchangeable under the assumed weighted-allocation law "
                                    "despite short chains. This law uses fit-estimated PSTHs, omits within-unit history "
                                    "and may miss glyph-specific time profiles or fine latent state; it is not an exact "
                                    "physiological null. Small relay mobility can lower power. Effects and SE are "
                                    "reported without treating nine surrogate draws as high-resolution p-values.",
              "no_image_scores": True, "seconds": time.monotonic() - started}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"primary": aggregated.get(primary), "other": aggregated.get("within_unadjusted_for_crossed"),
                      "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
