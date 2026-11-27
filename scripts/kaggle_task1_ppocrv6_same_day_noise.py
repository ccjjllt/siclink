"""Audit a same-day repeat simulation against measured mice2 Task 1 responses."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_prior import fit_atlas_prior
from glyph_data import BIN_MS, load_catalog, load_images, locate_data
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import build_font_atlas, standardized
from kaggle_task1_mice2_textverifier_recheck import tensor_hash
from kaggle_task1_native_hanzi import RIDGE, feature_extractors, fit_response_ridge
from kaggle_task1_native_hanzi_open_atlas import image_coordinates, visual_proxy
from kaggle_task1_ppocrv6_readout import SEED as PPOCR_SEED, distinct_cohort, fit_coordinates, read_glyphs, recognizer
from kaggle_task1_ppocrv6_repeat_kernel import DIMENSIONS, SEED as KERNEL_SEED, disjoint_halves, precise_cohort
from kaggle_task1_textverifier import publish
from kaggle_v8_metric_validation import SESSIONS, summarize


SEED = 225101
EXACT_GLYPHS = 48
OCF_BLOCK = 16
SOFT_DEADLINE = 1530
SIMULATED_COUNTS = (8, 16, 20)
WITHIN_SHRINKAGE = 0.10
STOCK = "ppocr_ridge"
WHITE = "repeat_whitened_ppocr"
METHODS = (STOCK, WHITE)
MEASURED_TWO = "measured_same_day_2"
MEASURED_FOUR = "measured_same_day_4"
SIMULATED_FOUR = "simulated_same_day_4_conditioned_on_2"
OUTPUT = Path("/kaggle/working/task1_ppocrv6_same_day_noise.json")


def cohort(size, rng):
    mice2 = distinct_cohort(size, np.random.default_rng(PPOCR_SEED))[1]
    mice1 = distinct_cohort(size, np.random.default_rng(PPOCR_SEED + 1), mice2)[1]
    prior_whitening = precise_cohort(size, np.random.default_rng(KERNEL_SEED))[1]
    fit, held, overlaps = distinct_cohort(size, rng, torch.cat((mice2, mice1, prior_whitening)))
    overlaps.pop("ppocrv6_mice2")
    overlaps.update({name: int(np.intersect1d(held, earlier).size) for name, earlier in
                     {"ppocr_pilot_mice2": mice2, "ppocr_pilot_mice1": mice1, "ppocr_whitening_mice2": prior_whitening}.items()})
    if any(overlaps.values()):
        raise ValueError("The evaluation glyphs must be outside the earlier readout and v20 cohorts.")
    return fit, held, overlaps


def timing_trials(root, catalog, entry, glyphs):
    """Count the three evoked windows used by timing_view without storing all ten bins."""
    samples_per_bin = catalog["sample_rate_hz"] * BIN_MS // 1000
    if BIN_MS != 25 or catalog["window_ms"] != 250:
        raise ValueError("Direct window counting requires the original ten 25 ms bins.")
    units = entry["n_units"]
    with np.load(root / entry["path"]) as data:
        targets = data["target"]
        groups = np.diff(data["group_ptr"])
        trial_sizes = np.diff(data["trial_ptr"])
        trials = np.repeat(np.arange(len(trial_sizes)), trial_sizes)
        phase = np.asarray((0, 0, 1, 1, 2, 2, 2, 3, 3, 3), dtype=np.int8)
        phases = phase[data["spike_samples"] // samples_per_bin]
        flat_window = (trials * units + data["unit_index"]) * len(np.unique(phase)) + phases
    trial_targets = np.repeat(targets, groups)
    if (len(trial_targets) != len(trial_sizes) or len(targets) != len(groups)
            or np.bincount(trial_targets, minlength=glyphs).tolist() != [4] * glyphs):
        raise ValueError("Each training glyph needs four recorded trials in this recording.")
    hist = np.bincount(flat_window, minlength=len(trial_targets) * units * 4).reshape(-1, units, 4)
    evoked = torch.from_numpy(hist[:, :, 1:]).float()
    view = torch.cat((evoked.sum(-1, keepdim=True), evoked), dim=-1).add_(0.375).sqrt_().flatten(1)
    order = np.argsort(trial_targets, kind="stable")
    return view[order].reshape(glyphs, 4, -1)


def white_projection(training, labels, center, scale, rng):
    n, repeats, width = training.shape
    residual = (training - training.mean(1, keepdim=True)).flatten(0, 1)
    within = residual.T @ residual / (n * (repeats - 1))
    diagonal = within.diag().clamp_min(0.05)
    within.mul_(0.75).diagonal().add_(0.25 * diagonal + 0.05)
    lower = torch.linalg.cholesky(within)
    left, right = disjoint_halves(training, rng)
    left, right = torch.linalg.solve_triangular(lower, torch.cat((left, right)).T, upper=False).T.chunk(2)
    signal = (left.T @ right + right.T @ left) / (2 * n)
    eigenvalues, basis = torch.linalg.eigh(signal)
    axes = basis[:, -min(width, DIMENSIONS):]
    mean_coordinates = 0.5 * (left + right) @ axes
    mean = mean_coordinates.mean(0)
    design = mean_coordinates - mean
    contrast = design.square().sum(1).mean().clamp_min(1e-5)
    gram = design.T @ design
    gram.diagonal().add_(RIDGE * n * contrast / len(mean))
    coefficients = torch.linalg.solve(gram, design.T @ labels)
    projected = axes @ coefficients
    projection = torch.linalg.solve_triangular(lower.T, projected, upper=True) / scale[:, None]
    intercept = -(center @ projection + mean @ coefficients)
    return projection, intercept, {"reliable_dimensions": len(mean), "leading_signal_eigenvalue": float(eigenvalues[-1]),
                                  "last_retained_signal_eigenvalue": float(eigenvalues[-len(mean)])}


def projected_responses(training, query, base, pp, rng):
    ridge, center, scale = fit_response_ridge(training.mean(1))
    pp_labels, pp_bank = pp
    decoder = [ridge @ labels / scale[:, None] for labels, _ in (*base.values(), pp)]
    white, intercept, diagnostics = white_projection((training - center) / scale, pp_labels, center, scale, rng)
    all_projections = torch.cat((*decoder, white), dim=1)
    offset = torch.cat((*[-(center @ projection) for projection in decoder], intercept))
    training_codes = training @ all_projections + offset
    query_codes = query @ all_projections + offset
    banks = [bank for _, bank in base.values()] + [pp_bank, pp_bank]
    return training_codes, query_codes, banks, diagnostics


@dataclass
class SameDayNoise:
    center: torch.Tensor
    scale: torch.Tensor
    whiten: torch.Tensor
    color: torch.Tensor
    signal: torch.Tensor
    diagnostics: dict

    def encode(self, observation):
        return ((observation - self.center) / self.scale) @ self.whiten

    def decode(self, observation):
        return (observation @ self.color) * self.scale + self.center

    def predictive(self, measured, n_measured, n_future):
        statistics = self.encode(measured)
        posterior = self.signal / (self.signal + 1 / n_measured)
        future_center = statistics * posterior
        future_scale = (posterior / n_measured + 1 / n_future).sqrt()
        return future_center, future_scale

    def continue_mean(self, measured, n_measured, total, standard_normal):
        count = total - n_measured
        center, scale = self.predictive(measured, n_measured, count)
        simulated_future = self.decode(center + scale * standard_normal)
        return measured.lerp(simulated_future, count / total)

    def audit(self, first, second):
        innovations = []
        for known, reserved in ((first, second), (second, first)):
            center, scale = self.predictive(known, 2, 2)
            innovations.append((self.encode(reserved) - center) / scale)
        errors = torch.cat(innovations)
        variance = errors.square().mean(0)
        return {"forecast_error_variance_mean_expected_1": float(variance.mean()),
                "median_axis_error_variance_expected_1": float(variance.median()),
                "forecast_90_percent_interval_coverage": float((errors.abs() <= 1.6448536).float().mean()),
                "forecast_error_mean_absolute_axis_bias": float(errors.mean(0).abs().mean()),
                "held_glyphs": len(first)}


def fit_noise(training):
    glyphs, repeats, dimensions = training.shape
    means = training.mean(1)
    center = means.mean(0)
    scale = means.std(0).clamp_min(1e-4)
    projected = (training - center) / scale
    means = projected.mean(1)
    residual = (projected - means[:, None]).flatten(0, 1)
    noise = residual.T @ residual / (glyphs * (repeats - 1))
    diagonal = noise.diag().clone()
    noise.mul_(1 - WITHIN_SHRINKAGE).diagonal().add_(WITHIN_SHRINKAGE * diagonal + 1e-4)
    lower = torch.linalg.cholesky(noise)
    white_means = torch.linalg.solve_triangular(lower, means.T, upper=False).T
    eigenvalues, rotation = torch.linalg.eigh(white_means.T @ white_means / (glyphs - 1))
    # Remove finite-repeat noise from the observed between-glyph covariance.
    signal = (eigenvalues - 1 / repeats).clamp_min(0)
    whitening = torch.linalg.solve_triangular(lower.T, rotation, upper=True)
    return SameDayNoise(center, scale, whitening, rotation.T @ lower.T, signal,
                        {"projected_dimensions": dimensions, "within_covariance_shrinkage": WITHIN_SHRINKAGE,
                         "positive_signal_axes": int((signal > 0).sum()), "fit_glyphs": glyphs})


def scores(observation, banks, prior):
    features = observation.split([bank.shape[0] for bank in banks], dim=1)
    cosine = [standardized(F.normalize(values) @ bank) for values, bank in zip(features, banks, strict=True)]
    contour_hanzi = 0.5 * (cosine[0] + cosine[1])
    return {name: prior.variants(0.5 * (contour_hanzi + pp))["usage_unseen"]
            for name, pp in zip(METHODS, cosine[2:], strict=True)}


def same_day_means(query_codes, noise, random):
    first = query_codes[:, :2].mean(1)
    second = query_codes[:, 2:].mean(1)
    measured = query_codes.mean(1)
    residual = torch.randn(measured.shape, generator=random)
    # Antithetic conditional draws share each innovation across simulated budgets and methods.
    conditions = {(MEASURED_TWO, 0): first, (MEASURED_FOUR, 0): measured}
    for draw, epsilon in enumerate((residual, -residual)):
        conditions[SIMULATED_FOUR, draw] = noise.continue_mean(first, 2, 4, epsilon)
        for total in SIMULATED_COUNTS:
            conditions[f"simulated_same_day_{total}", draw] = noise.continue_mean(measured, 4, total, epsilon)
    return conditions


def exact_by_condition(choices, rows, rendered, references, score):
    keys = list(choices)
    selected = torch.stack([choices[key][rows, 0] for key in keys])
    pair_ids = torch.arange(len(rows)) + selected * len(rows)
    unique, inverse = torch.unique(pair_ids.flatten(), return_inverse=True)
    values = []
    done = 0
    for batch in unique.split(252):
        guess = rendered[batch.div(len(rows), rounding_mode="floor")] >= 128
        target = references[batch.remainder(len(rows))] >= 128
        values.extend(score(guess.numpy(), target.numpy(), device="cpu", batch_size=12))
        done += len(batch)
        print(f"Exact OCF {done}/{len(unique)}", flush=True)
    paired = np.asarray(values)[inverse.numpy()].reshape(selected.shape)
    return dict(zip(keys, paired, strict=True)), len(unique)


def group_results(choices, exact, proxy, trusted):
    results, glyph_means = {}, {}
    for condition in dict.fromkeys(endpoint for endpoint, _, _ in choices):
        report, outcomes = {}, {}
        for method in METHODS:
            keys = [key for key in choices if key[0] == condition and key[2] == method]
            aligned = torch.stack([choices[key][trusted] == proxy[trusted, None] for key in keys])
            outcomes[method] = np.stack([exact[key] for key in keys]).mean(0)
            report[method] = {"trusted_proxy_recall": {str(k): float(aligned[:, :, :k].any(2).float().mean())
                                                       for k in (1, 16, 32)},
                              "single_draw_ocf": [float(exact[key].mean()) for key in keys],
                              "glyph_mean_ocf_values": outcomes[method].tolist()}
        for method in METHODS:
            report[method]["official_ocf"] = summarize([outcomes[method]], [outcomes[STOCK]])
        results[condition] = report
        glyph_means[condition] = outcomes
    return results, glyph_means


def score_blocks(choices, eligible, atlas, references, scorer, report, started):
    pairs, previous_seconds, chunks, per_key = 0, 0, [], {}
    for block in eligible.split(OCF_BLOCK):
        if chunks and time.monotonic() - started + 1.25 * previous_seconds + 25 > SOFT_DEADLINE:
            break
        began = time.monotonic()
        values, measured = exact_by_condition(choices, block, atlas.images, references[block], scorer)
        for key, value in values.items():
            per_key.setdefault(key, []).append(value)
        previous_seconds = time.monotonic() - began
        pairs += measured
        chunks.append(block)
        report["completed_exact_ocf_blocks"] = len(chunks)
        publish(report, OUTPUT, started, f"{len(chunks) * OCF_BLOCK}_glyphs_scored")
    return {key: np.concatenate(values) for key, values in per_key.items()}, torch.cat(chunks), pairs


@torch.inference_mode()
def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    cpu = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    random = torch.Generator().manual_seed(SEED + 1)
    root = locate_data()
    images = load_images(root, cpu)
    if images.shape != (2748, 64, 64) or images.dtype != torch.uint8:
        raise ValueError("Use the complete labeled grayscale competition glyph gallery.")
    fit, held, overlaps = cohort(len(images), rng)
    stages = np.resize(np.arange(len(SESSIONS)), len(held))
    days = torch.from_numpy(rng.permutation(stages))
    selection = [rng.choice(torch.where(days == i)[0].numpy(), EXACT_GLYPHS // len(SESSIONS), replace=False)
                 for i in range(len(SESSIONS))]
    eligible = torch.from_numpy(np.asarray(selection).reshape(len(SESSIONS), -1, OCF_BLOCK // len(SESSIONS))
                                .transpose(1, 0, 2).reshape(-1))
    report = {"seed": SEED, "mouse": "mice2", "formal_queries_loaded": False,
              "gallery_sha256": tensor_hash(images), "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "fit_glyph_indices_sha256": tensor_hash(fit), "fit_glyph_count": len(fit),
              "held_gallery_indices": held.tolist(), "planned_exact_gallery_indices": held[eligible].tolist(), "overlaps": overlaps,
              "exact_ocf_budget": {"planned": EXACT_GLYPHS, "balanced_block_size": OCF_BLOCK, "soft_deadline_seconds": SOFT_DEADLINE},
              "measured_same_day_repeats_available": 4, "simulation_counts": SIMULATED_COUNTS,
              "assigned_day": {day: held[days == i].tolist() for i, day in enumerate(SESSIONS)}, "recordings": {},
              "protocol": "Freeze the public glyph bank, PP-OCRv6 recognizer, contour/Hanzi readouts and rotation of held "
                          "recordings. Compare the same standardized mixture using default PP ridge or 96-dimensional "
                          "repeat-whitened PP ridge. Estimate a joint Gaussian model in the projected readout coordinates, "
                          "using the fit glyphs' same-day residuals and deconvolved between-glyph covariance. Simulated "
                          "higher-repeat means retain their actual two or four observed trials, then average in a posterior "
                          "predictive mean of future repeats. Simulated 2 to 4 is audited against two independently "
                          "reserved actual repeats on held glyphs. Real known-correspondence 4/8/16 uses four separate days "
                          "with 1/2/4 measured repeats per day. Synthetic 8/16/20 extrapolates same-day stationarity and "
                          "Gaussian uncertainty and cannot identify actual held high-repeat scores."}
    publish(report, OUTPUT, started, "training_data_started")
    catalog = load_catalog(root)
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == "mice2" and entry["task"] == "train"),
                     key=lambda entry: entry["session"])
    if tuple(entry["session"] for entry in entries) != SESSIONS:
        raise ValueError("Use the four labeled recordings in their original day order.")
    session, metadata = recognizer()
    report["published_ocr"] = metadata
    publish(report, OUTPUT, started, "recognizer_ready")
    selected = rng.choice(fit.numpy(), size=128, replace=False)
    atlas, provenance, codes = build_font_atlas(images[selected], cpu, return_codepoints=True)
    encoded = read_glyphs(session, torch.cat((images[fit], atlas.images)), metadata)
    pp = fit_coordinates(encoded["ppocrv6_global"], len(fit))
    del encoded, session
    extractors = feature_extractors()
    extractors.pop("random_mobilenet")
    base = image_coordinates(images[fit], atlas.images, extractors)
    prior = fit_atlas_prior(atlas, codes, images[fit])
    report.update(atlas=provenance, candidates=len(codes), render_sha256=tensor_hash(atlas.images), unihan=prior.diagnostics)
    publish(report, OUTPUT, started, "frozen_representations_ready")

    choices, pooled = {}, {}
    for index, entry in enumerate(entries):
        day = entry["session"]
        trials = timing_trials(root, catalog, entry, len(images))
        order = torch.rand((len(held), 4), generator=random).argsort(1)
        query = trials[held].gather(1, order[:, :, None].expand(-1, -1, trials.shape[-1]))
        training, projected, banks, diagnostic = projected_responses(trials[fit], query, base, pp, rng)
        noise = fit_noise(training)
        del trials, training, query
        picked = torch.where(days == index)[0]
        first, second = projected[:, :2].mean(1), projected[:, 2:].mean(1)
        report["recordings"][day] = {**diagnostic, **noise.diagnostics, "conditional_2_to_4_audit": noise.audit(first, second)}
        conditions = same_day_means(projected[picked], noise, random)
        for (condition, draw), observation in conditions.items():
            for method, logits in scores(observation, banks, prior).items():
                bank = choices.setdefault((condition, draw, method), torch.empty((len(held), 32), dtype=torch.long))
                bank[picked] = logits.topk(32, dim=1).indices
        for per_day in (1, 2, 4):
            condition = f"measured_known_across_day_{per_day * len(SESSIONS)}"
            for method, logits in scores(projected[:, :per_day].mean(1), banks, prior).items():
                key = condition, 0, method
                if key in pooled:
                    pooled[key].add_(logits)
                else:
                    pooled[key] = logits
        publish(report, OUTPUT, started, f"{day}_ready")
    for key, logits in pooled.items():
        choices[key] = logits.topk(32, dim=1).indices
    del pooled

    proxy, trusted, _ = visual_proxy(atlas, images[held])
    report["trusted_proxy_count"] = int(trusted.sum())
    publish(report, OUTPUT, started, "exact_ocf_started")
    exact, scored, pairs = score_blocks(choices, eligible, atlas, images[held], official_scorer(root), report, started)
    report["scored_gallery_indices"] = held[scored].tolist()
    result, glyph_means = group_results(choices, exact, proxy, trusted)
    report["results"] = result
    actual, synthetic = glyph_means[MEASURED_FOUR], glyph_means[SIMULATED_FOUR]
    report["simulation_validation_ocf"] = {"virtual4_vs_measured4": {method: summarize([synthetic[method]], [actual[method]])
                                                                     for method in METHODS},
                                           "change_in_whitening_effect": summarize([synthetic[WHITE] - synthetic[STOCK]],
                                                                                    [actual[WHITE] - actual[STOCK]])}
    contrasts = {"known_across_day_4_vs_measured_same_day_4": ("measured_known_across_day_4", MEASURED_FOUR),
                 "known_across_day_16_vs_known_across_day_4": ("measured_known_across_day_16", "measured_known_across_day_4"),
                 "simulated_same_day_20_vs_measured_same_day_4": ("simulated_same_day_20", MEASURED_FOUR),
                 "known_across_day_16_vs_simulated_same_day_16": ("measured_known_across_day_16", "simulated_same_day_16")}
    report["paired_noise_and_domain_contrasts"] = {
        name: {method: summarize([glyph_means[left][method]], [glyph_means[right][method]]) for method in METHODS}
        for name, (left, right) in contrasts.items()}
    report["unique_exact_pairs"] = pairs
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
