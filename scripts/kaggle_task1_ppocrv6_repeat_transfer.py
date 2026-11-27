"""Test PP-OCR consistency and independent repeat prediction at the official repeat count."""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

from glyph_atlas_prior import fit_atlas_prior
from glyph_data import BIN_MS, load_catalog, load_images, locate_data
from glyph_kernel_encoding import fit_kernel_glyph_encoder
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import build_font_atlas, standardized
from kaggle_task1_native_hanzi import RIDGE, feature_extractors, fit_response_ridge
from kaggle_task1_native_hanzi_open_atlas import image_coordinates
from kaggle_task1_ppocrv6_readout import SEED as PPOCR_SEED, distinct_cohort, fit_coordinates, read_glyphs, recognizer
from kaggle_task1_ppocrv6_repeat_kernel import SEED as KERNEL_SEED, fit_repeat_whitening, precise_cohort, whitened_projection
from kaggle_task1_textverifier import publish
from kaggle_v8_metric_validation import SESSIONS, summarize


SEED, SCORE_GLYPHS, SPLITS = 225093, 112, 3
MOUSE, ORIGINAL, RIDGE_PP, WHITE = "mice2", "contour_hanzi", "ppocr_ridge", "ppocr_whitened"
MODES = (ORIGINAL, RIDGE_PP, WHITE)
OUTPUT = Path("/kaggle/working/task1_ppocrv6_repeat_transfer.json")


@dataclass
class Readout:
    ordinary: dict
    center: torch.Tensor
    scale: torch.Tensor
    whitening: tuple
    latent_center: torch.Tensor
    latent_weights: torch.Tensor
    output_center: torch.Tensor
    atlas_bank: dict

    def pick(self, mean, prior):
        normalized = (mean - self.center) / self.scale
        heads = {name: standardized(F.normalize(normalized @ weights) @ self.atlas_bank[name])
                 for name, weights in self.ordinary.items()}
        common = (heads["contour"] + heads["frozen_hanzi"]) * 0.5
        whitened = whitened_projection(mean, self.whitening)
        prediction = (whitened - self.latent_center) @ self.latent_weights + self.output_center
        white_score = standardized(F.normalize(prediction) @ self.atlas_bank["pp"])
        arms = {ORIGINAL: common, RIDGE_PP: (common + heads["pp"]) * 0.5, WHITE: (common + white_score) * 0.5}
        return {name: prior.variants(score)["usage_unseen"].topk(32, dim=1).indices for name, score in arms.items()}


def fit_readout(training, coordinates, rng):
    ridge, center, scale = fit_response_ridge(training.mean(1))
    ordinary = {name: ridge @ labels for name, (labels, _) in coordinates.items()}
    (left, right), whitening, diagnostics = fit_repeat_whitening(training, rng)
    observed = 0.5 * (left + right)
    latent_center = observed.mean(0)
    design = observed - latent_center
    label = coordinates["pp"][0]
    output_center = label.mean(0)
    gram = design.T @ design
    penalty = RIDGE * len(design) / design.shape[1]
    gram.diagonal().add_(penalty * (gram.trace() / len(design)))
    latent_weights = torch.cholesky_solve(design.T @ (label - output_center), torch.linalg.cholesky(gram))
    banks = {name: bank for name, (_, bank) in coordinates.items()}
    return Readout(ordinary, center, scale, whitening, latent_center, latent_weights, output_center, banks), diagnostics


def evoked_block(root, catalog, entry, *, expected_groups, expected_repeats, labeled):
    """Count the exact evoked windows without storing the ten-bin raw tensor or reading query IDs."""
    if BIN_MS != 25 or catalog["window_ms"] != 250:
        raise ValueError("These timing features use the original ten 25-ms bins.")
    samples_per_bin = catalog["sample_rate_hz"] * BIN_MS // 1_000
    units = entry["n_units"]
    phase_lookup = np.asarray((0, 0, 1, 1, 2, 2, 2, 3, 3, 3), dtype=np.int8)
    with np.load(root / entry["path"]) as source:
        groups = np.diff(source["group_ptr"])
        trial_sizes = np.diff(source["trial_ptr"])
        targets = np.repeat(source["target"], groups) if labeled else None
        trial = np.repeat(np.arange(len(trial_sizes), dtype=np.int32), trial_sizes)
        phase = phase_lookup[source["spike_samples"] // samples_per_bin]
        flat = (trial * units + source["unit_index"]) * 4 + phase
    if len(trial_sizes) != expected_groups * expected_repeats:
        raise ValueError("The recording does not contain the expected number of measured trials.")
    hist = np.bincount(flat, minlength=len(trial_sizes) * units * 4).reshape(-1, units, 4)
    del trial, flat
    counts = torch.from_numpy(hist[..., 1:]).float()
    if labeled:
        if np.bincount(targets, minlength=expected_groups).tolist() != [expected_repeats] * expected_groups:
            raise ValueError("Every labeled glyph must have four genuine trial repeats.")
        counts = counts[np.argsort(targets, kind="stable")]
    elif len(groups) != expected_groups or bool((groups != expected_repeats).any()):
        raise ValueError("Every anonymous group must have twenty genuine trial repeats.")
    counts = counts.reshape(expected_groups, expected_repeats, units, 3)
    totals = counts.sum(-1)
    timing = torch.cat((totals[..., None], counts), dim=-1).add_(0.375).sqrt_().flatten(-2)
    return timing, totals


def predictions(timing, counts, readout, prior, rng):
    complete = readout.pick(timing.mean(1), prior)
    partitions = []
    for _ in range(SPLITS):
        order = torch.from_numpy(rng.random(counts.shape[:2]).argsort(1))
        balanced = order.split(counts.shape[1] // 2, dim=1)
        endpoints = []
        for rows in balanced:
            select = rows[:, :, None]
            response = timing.gather(1, select.expand(-1, -1, timing.shape[-1])).mean(1)
            raw = counts.gather(1, select.expand(-1, -1, counts.shape[-1])).mean(1)
            endpoints.append((readout.pick(response, prior), raw))
        partitions.append(tuple(endpoints))
    return complete, partitions


def held_log_likelihood(observed, rates, baseline):
    reference = baseline.clamp_min(1e-4)
    expected = (0.98 * rates + 0.02 * reference).clamp_min(1e-4)
    return (observed * (expected.log() - reference.log()) - expected + reference).sum(-1)


def calculate_reliability(full, partitions, read_rates, baseline, contour):
    per_arm = {name: [] for name in MODES}
    for (left, left_trials), (right, right_trials) in partitions:
        for name in MODES:
            a, b = left[name], right[name]
            top_a, top_b = a[:, 0], b[:, 0]
            overlap = (a[:, :, None] == b[:, None]).any(2).float().mean(1)
            held = 0.5 * (held_log_likelihood(right_trials, read_rates(top_a), baseline)
                          + held_log_likelihood(left_trials, read_rates(top_b), baseline))
            same = (top_a == top_b).float()
            shape = (contour[top_a] * contour[top_b]).sum(-1)
            full_top = full[name][:, 0]
            agreement = 0.5 * ((top_a == full_top).float() + (top_b == full_top).float())
            per_arm[name].append(torch.stack((same, overlap, shape, held, agreement), dim=1))
    return {name: torch.stack(rows).mean(0) for name, rows in per_arm.items()}


def describe(stats):
    columns = ("half_top1_agreement", "half_top32_overlap", "half_shape_cosine", "independent_half_log_likelihood_nats",
               "mean_half_top1_agreement_with_full")
    report = {name: {key: float(value) for key, value in zip(columns, rows.mean(0), strict=True)}
              for name, rows in stats.items()}
    difference = stats[WHITE] - stats[RIDGE_PP]
    report["white_vs_pp_ridge"] = {key: {"mean": float(column.mean()), "paired_se": float(column.std() / len(column)**0.5)}
                                  for key, column in zip(columns, difference.T, strict=True)}
    return report


def fit_cohort(size, rng):
    _, first, _ = distinct_cohort(size, np.random.default_rng(PPOCR_SEED))
    _, second, _ = distinct_cohort(size, np.random.default_rng(PPOCR_SEED + 1), first)
    _, third, _ = precise_cohort(size, np.random.default_rng(KERNEL_SEED))
    fit, held, overlap = distinct_cohort(size, rng, torch.cat((first, second, third)))
    overlap["prior_ppocr_readout_cohorts"] = overlap.pop("ppocrv6_mice2")
    return fit, held, overlap


@torch.inference_mode()
def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    root = locate_data()
    catalog, images = load_catalog(root), load_images(root, device)
    fit, held, overlap = fit_cohort(len(images), rng)
    scored = rng.choice(len(held), size=SCORE_GLYPHS, replace=False).reshape(len(SESSIONS), -1)
    official = {entry["session"]: entry for entry in catalog["blocks"] if entry["split"] == "test"
                and entry["task"] == "task1" and entry["mouse"] == MOUSE}
    train_entries = {entry["session"]: entry for entry in catalog["blocks"] if entry["task"] == "train" and entry["mouse"] == MOUSE}
    if set(train_entries) != set(SESSIONS) or set(official) != set(SESSIONS):
        raise ValueError("Require the four labeled and four official mice2 Task 1 recordings.")
    report = {"seed": SEED, "mouse": MOUSE, "fit_count": len(fit), "labeled_held_count": len(held),
              "cohort_overlap": overlap, "official_group_count": 0, "official_labels_loaded": False,
              "official_input_diagnostic_only": True, "split_partitions": SPLITS, "recordings": {},
              "timing_count_layout": "25-ms bins compressed to baseline/early/mid/late, with exact evoked total",
              "forward_targets": "per-unit measured counts across the full 50–250 ms evoked window",
              "protocol": "Same held-glyph PP ridge and whitened PP readouts in the same atlas. Compare 2/2 disjoint "
                          "halves on independent labeled examples, and 10/10 real halves within each official group. "
                          "An independent image-to-spikes contour kernel estimates held-half spike likelihood of the "
                          "other half's selected candidate. No choices, IDs or labels from formal data are written."}
    publish(report, OUTPUT, started, "inputs_ready")
    session, metadata = recognizer()
    report["ppocrv6"] = metadata
    publish(report, OUTPUT, started, "assets_ready")
    chosen = rng.choice(fit.numpy(), size=128, replace=False)
    atlas, provenance, codepoints = build_font_atlas(images[chosen], device, return_codepoints=True)
    encoding = read_glyphs(session, torch.cat((images[fit], atlas.images)), metadata)
    pp = fit_coordinates(encoding["ppocrv6_global"], len(fit))
    del encoding, session
    extractors = feature_extractors()
    extractors.pop("random_mobilenet")
    coordinates = {**image_coordinates(images[fit], atlas.images, extractors), "pp": pp}
    prior = fit_atlas_prior(atlas, codepoints, images[fit])
    labeled = {}
    for day in SESSIONS:
        labeled[day] = evoked_block(root, catalog, train_entries[day], expected_groups=len(images), expected_repeats=4, labeled=True)
        publish(report, OUTPUT, started, f"{day}_labeled_compact")
    training_rates = {day: totals[fit] for day, (_, totals) in labeled.items()}
    contour_model = fit_kernel_glyph_encoder(training_rates, images[fit])
    scorer = official_scorer(root)
    report.update(atlas=provenance, candidates=len(codepoints))
    publish(report, OUTPUT, started, "models_ready")

    exact_by_arm = {name: [] for name in MODES}
    labeled_stats, anonymous_stats, labeled_errors = [], [], []
    pair_count = 0
    for i, day in enumerate(SESSIONS):
        formal_timing, formal_totals = evoked_block(root, catalog, official[day], expected_groups=340,
                                                  expected_repeats=20, labeled=False)
        observed_timing, observed_totals = labeled.pop(day)
        readout, diagnostic = fit_readout(observed_timing[fit], coordinates, rng)
        first = predictions(observed_timing[held], observed_totals[held], readout, prior, rng)
        second = predictions(formal_timing, formal_totals, readout, prior, rng)
        all_ids = torch.cat([values[:, 0] for _, partitions in (first, second) for split in partitions
                             for choices, _ in split for values in choices.values()]).unique(sorted=True)
        rates = contour_model.expected_counts(atlas.images[all_ids], day)
        baseline = training_rates[day].mean((0, 1))

        def read_rates(indices):
            return rates[torch.searchsorted(all_ids, indices)]

        comparison = [calculate_reliability(picks, halves, read_rates, baseline, atlas.contours)
                      for picks, halves in (first, second)]
        labeled_stats.append(comparison[0])
        anonymous_stats.append(comparison[1])
        score_rows = torch.from_numpy(scored[i])
        exact = PairedOCF(scorer, atlas.images, (images[held[score_rows]] >= 128).numpy(), device, 12)
        for name, rankings in first[0].items():
            exact_by_arm[name].append(exact.evaluate(rankings[score_rows, :1])[:, 0])
        difference = comparison[0][WHITE] - comparison[0][RIDGE_PP]
        labeled_errors.append(difference[score_rows])
        pair_count += len(exact.values)
        report["recordings"][day] = {"fit": diagnostic, "labeled": describe(comparison[0]), "official": describe(comparison[1])}
        report["official_group_count"] += len(formal_timing)
        publish(report, OUTPUT, started, f"{day}_scored")
    summary = {name: np.concatenate(ocf) for name, ocf in exact_by_arm.items()}
    report["labeled_same_day_four_ocf"] = {name: summarize([values], [summary[RIDGE_PP]]) for name, values in summary.items()}
    report["labeled_robustness"] = describe({name: torch.stack([scores[name] for scores in labeled_stats]).mean(0) for name in MODES})
    report["official_robustness"] = describe({name: torch.cat([scores[name] for scores in anonymous_stats]) for name in MODES})
    actual_delta = summary[WHITE] - summary[RIDGE_PP]
    changes = torch.cat(labeled_errors)
    measures = {"held_half_logp_difference": changes[:, 3], "top1_agreement_difference": changes[:, 0],
                "half_top32_difference": changes[:, 1]}
    report["labeled_rank_correlation_against_ocf_delta"] = {
        key: float(spearmanr(values.numpy(), actual_delta).statistic)
        if float(values.std()) > 0 and np.ptp(actual_delta) > 0 else None for key, values in measures.items()
    }
    report["unique_exact_labeled_pairs"] = pair_count
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
