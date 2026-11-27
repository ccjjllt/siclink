"""Test query-specific directional noise weights on the exact Task 2 CDF branch."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, fit_retrieval
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task2_anchored_cdf import evaluation_sample as anchored_cohort
from kaggle_task2_classifier_ensemble import evaluation_sample as ensemble_cohort
from kaggle_task2_constrained_ceiling import fresh_samples as ceiling_cohorts
from kaggle_task2_day_shift import hybrid_scores, sample_glyphs as shift_cohorts
from kaggle_task2_native_hanzi import SEED as HANZI_SEED, scoring_glyphs as hanzi_cohorts
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics
from kaggle_task2_rank_marginals import (BLEND, CEILING_CROWDED, EPOCHS, MODEL_SEED, MOUSE, TEMPERATURE, V9,
                                         evaluation_glyphs, normal_scores, responses, source_cdf, summarize, whiten)
from kaggle_task2_rank_marginals_recheck import PROVENANCE, evaluation_sample as recheck_cohort, fit_marginal_templates
from kaggle_v8_metric_validation import SESSIONS


SEED = 286391
HELD_DAYS = ("session1", "session4")
SCORING_COUNT, BANDS = 128, 8
SHRINKAGE, MIN_RATIO, MAX_RATIO = 0.35, 0.25, 3.0
CDF, DIRECTIONAL, SCALAR, SHUFFLED = "stock_cdf", "directional_precision", "scalar_precision", "shuffled_precision"
EXACT_ARMS = (V9, CDF, DIRECTIONAL, SCALAR, SHUFFLED)
OUTPUT = Path("/kaggle/working/task2_repeat_precision.json")


def evaluation_sample(size):
    first, _ = evaluation_glyphs(size)
    recheck, _ = recheck_cohort(size)
    anchor, _ = anchored_cohort(size)
    followup, _ = anchored_cohort(size, followup=True)
    shift, oracle = shift_cohorts(size)
    ensemble, _ = ensemble_cohort(size)
    ceiling, _, _ = ceiling_cohorts(size)
    hanzi, _ = hanzi_cohorts(size, np.random.default_rng(HANZI_SEED))
    older = {"rank_marginals": first.numpy(), "rank_marginals_recheck": recheck.numpy(),
             "anchored": anchor.numpy(), "anchored_followup": followup.numpy(), "day_shift": shift.numpy(),
             "day_shift_oracle": oracle.numpy(), "ensemble": ensemble.numpy(),
             "ceiling_uniform": np.concatenate(tuple(ceiling.values())), "ceiling_crowded": np.asarray(CEILING_CROWDED),
             "native_hanzi": np.concatenate(tuple(hanzi.values()))}
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(tuple(older.values()))))
    selected = np.sort(np.random.default_rng(SEED).choice(eligible, SCORING_COUNT, replace=False))
    return torch.from_numpy(selected), {name: int(np.intersect1d(rows, selected).size) for name, rows in older.items()}


def trial_counts(counts):
    return counts.unflatten(-1, (-1, 10))[..., 2:].sum(-1)


def rank_coordinates(values, factor):
    return whiten(normal_scores(values, source_cdf(values)), factor)


def trial_coordinates(counts, factor):
    ranks = rank_coordinates((counts.flatten(0, 1) + 0.375).sqrt(), factor)
    return ranks.reshape_as(counts)


def source_noise_axes(sources, factor):
    covariance = torch.zeros_like(factor)
    for spikes in sources.values():
        coordinates = trial_coordinates(trial_counts(spikes), factor)
        residual = (coordinates - coordinates.mean(1, keepdim=True)).flatten(0, 1)
        degrees = len(coordinates) * (coordinates.shape[1] - 1)
        covariance.addmm_(residual.T, residual, alpha=1 / (len(sources) * degrees))
    eigenvalues, axes = torch.linalg.eigh(covariance)
    return axes.flip(1), eigenvalues.flip(0)


def noise_energy(counts, factor, axes):
    coordinates = trial_coordinates(counts, factor)
    contrast = (coordinates[:, 0] - coordinates[:, 1]) @ axes
    return torch.stack([values.square().mean(1) * 0.5 for values in contrast.tensor_split(BANDS, dim=1)], dim=1)


def repeat_weights(energy, activity, rng):
    """Match the rowwise mean precision and shape magnitude while scrambling its source query."""
    normalized = energy / energy.mean(0).clamp_min(1e-5)
    variance = (1 - SHRINKAGE) + SHRINKAGE * normalized.clamp(MIN_RATIO, MAX_RATIO)
    precision = variance.reciprocal()
    scalar = precision.mean(1, keepdim=True)
    centered = precision.log()
    centered -= centered.mean(1, keepdim=True)
    size = centered.norm(dim=1, keepdim=True)
    permutation = torch.empty(len(activity), dtype=torch.long)
    for group in activity.argsort(stable=True).tensor_split(4):
        for stratum in group[scalar[group, 0].argsort(stable=True)].tensor_split(4):
            visit = stratum[torch.from_numpy(rng.permutation(len(stratum)))]
            permutation[visit] = visit.roll(1)
    if bool((permutation == torch.arange(len(activity))).any()):
        raise ValueError("The matched precision control must shuffle every target profile.")
    donor = centered[permutation]
    # A flat donor has no shape; roll the original bands in this degenerate case.
    control_direction = torch.where(donor.norm(dim=1, keepdim=True) > 1e-8, F.normalize(donor, dim=1),
                                    F.normalize(centered.roll(1, dims=1), dim=1))
    control = (control_direction * size).exp()
    control *= scalar / control.mean(1, keepdim=True)
    return {DIRECTIONAL: precision, SCALAR: scalar.expand_as(precision), SHUFFLED: control}, centered


def matching_evidence(rows, gallery):
    return rows @ gallery.T - 0.5 * gallery.square().sum(1)


def score_corrections(query, gallery, factor, axes, split, rng):
    """Estimate directional reliability on each target half and apply it to the other."""
    grouped = query.gather(1, split[..., None].expand_as(query)).split(2, dim=1)
    trial_halves = tuple(trial_counts(counts) for counts in grouped)
    energies = tuple(noise_energy(counts, factor, axes) for counts in trial_halves)
    bank = gallery @ axes
    deltas = {name: gallery.new_zeros((len(query), len(gallery))) for name in (DIRECTIONAL, SCALAR, SHUFFLED)}
    fitted, profiles, validations = [], [], []
    for direction, fit_energy in enumerate(energies):
        score_half = 1 - direction
        activity = trial_halves[direction].mean(1).sum(1)
        weights, shape = repeat_weights(fit_energy, activity, rng)
        profiles.append(shape)
        rank = rank_coordinates(responses(grouped[score_half]), factor) @ axes
        full = matching_evidence(rank, bank)
        rms = (full - full.mean(1, keepdim=True)).square().mean().sqrt().clamp_min(1e-6)
        del full
        for index, (rows, centers) in enumerate(zip(rank.tensor_split(BANDS, dim=1), bank.tensor_split(BANDS, dim=1))):
            evidence = matching_evidence(rows, centers) / (2 * rms)
            for name, profile in weights.items():
                deltas[name].addcmul_(evidence, profile[:, index:index + 1] - 1)
        validation = energies[score_half] / fit_energy.mean(0).clamp_min(1e-5)
        validations.append({name: float((0.5 * (profile.log() - (profile - 1) * validation)).mean())
                            for name, profile in weights.items()})
        fitted.append({"mean_precision": float(weights[DIRECTIONAL].mean()),
                       "directional_log_precision_rms": float(shape.square().mean().sqrt()),
                       "maximum_matched_row_mean_error": float((weights[SHUFFLED].mean(1) - weights[SCALAR][:, 0]).abs().max())})
    correlation = F.cosine_similarity(profiles[0].flatten(), profiles[1].flatten(), dim=0)
    report = {"split_half_shape_correlation": float(correlation), "fitted_half_weights": fitted,
              "independent_half_noise_log_likelihood_gain_per_mode": validations}
    return deltas, report


def publish(report, phase, started):
    report.update(stage=phase, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": phase, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU notebook with authorized labeled training recordings.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if len(images) != 2748 or tuple(recordings) != SESSIONS or any(part.shape[:2] != (len(images), 4)
                                                                  for part in recordings.values()):
        raise ValueError("Require four complete labeled recordings with four measured repeats per glyph.")
    sample, overlap = evaluation_sample(len(images))
    metric = PairedOCF(official_scorer(root), images, (images[sample] >= 128).numpy(), device, 12)
    rng = np.random.default_rng(SEED + 1)
    report = {"mouse": MOUSE, "seed": SEED, "held_days": HELD_DAYS, "gallery_size": len(images),
              "labeled_sources_per_fold": 3, "target_real_repeats": 4, "archive_provenance": PROVENANCE,
              "scored_glyph_indices": sample.tolist(), "ocf_overlap_by_earlier_cohort": overlap, "formal_test_accessed": False,
              "marginal_cdf_weight": BLEND, "noise_modes": BANDS, "precision_ratio_clip": (MIN_RATIO, MAX_RATIO),
              "target_repeat_variance_weight": SHRINKAGE, "primary_arm": DIRECTIONAL, "paired_reference": CDF,
              "recordings": {}, "protocol": "Keep the four-repeat v9 retrieval, empirical Bayes and target-CDF fusion "
                          "fixed. Rotate the CDF coordinates onto the eigensystem of source repeat noise, then use "
                          "eight fixed groups of consecutive axes. For each anonymous target query jointly observed "
                          "twice, estimate excess variance versus the same target half's pooled variance. Reweight "
                          "the other two actual repeats' CDF evidence by a shrunken directional precision and swap "
                          "halves. Add only the cross-fit weighted-minus-unweighted score difference to the original "
                          "CDF branch. Compare a rowwise scalar precision and a matched log-profile shuffle that "
                          "retains each query's mean precision and centered shape magnitude. Every output solves "
                          "the complete one-to-one assignment before evaluating fresh paired official OCF."}
    publish(report, "inputs_ready", started)
    exact = {name: [] for name in EXACT_ARMS}
    for day in HELD_DAYS:
        sources = {name: values for name, values in recordings.items() if name != day}
        order = torch.from_numpy(rng.permutation(len(images)))
        query = recordings[day][order]
        split = torch.from_numpy(rng.random(query.shape[:2]).argsort(axis=1))
        publish(report, f"{day}_fitting", started)
        torch.manual_seed(MODEL_SEED)
        with torch.enable_grad():
            retrieval = fit_retrieval(as_trial_blocks(sources), len(images), device=device, seed=MODEL_SEED,
                                      epochs=EPOCHS, images=images)
        hierarchy = fit_hierarchical_gallery(sources, view="total")
        _, gallery, factor, _ = fit_marginal_templates(sources)
        hybrid = hybrid_scores(retrieval, hierarchy, query)
        cdf = logit_scale(matching_evidence(rank_coordinates(responses(query), factor), gallery))
        fixed = logit_scale(hybrid).lerp(cdf, BLEND)
        axes, spectrum = source_noise_axes(sources, factor)
        deltas, noise_diagnostics = score_corrections(query, gallery, factor, axes, split, rng)
        scores = {V9: hybrid, CDF: fixed, **{name: fixed + BLEND * values for name, values in deltas.items()}}
        choices = {name: assign_glyphs(values) for name, values in scores.items()}
        if any(value.unique().numel() != len(images) for value in choices.values()):
            raise ValueError("Every full assignment must use one of each official gallery glyph.")
        identity = {}
        rows = torch.arange(len(order))
        for name, matrix in scores.items():
            normalized = logit_scale(matrix) / TEMPERATURE
            identity[name] = {**row_diagnostics(matrix, order),
                              "identity_nll": float((normalized.logsumexp(1) - normalized[rows, order]).mean()),
                              "assigned_top1": float((choices[name] == order).float().mean()),
                              "changed_vs_cdf": int((choices[name] != choices[CDF]).sum()),
                              "distinct_outputs": int(choices[name].unique().numel())}
        report["recordings"][day] = {"source_days": tuple(sources), "identity": identity, "target_noise": noise_diagnostics,
                                     "source_noise_eigenvalue_band_means": [float(part.mean()) for part in spectrum.tensor_split(BANDS)]}
        publish(report, f"{day}_assignments_ready", started)
        sample_rows = order.argsort()[sample]
        measured = {name: metric.evaluate(choices[name][sample_rows, None])[:, 0] for name in EXACT_ARMS}
        for name, values in measured.items():
            exact[name].append(values)
        report["recordings"][day]["ocf_vs_cdf"] = {name: summarize(values, measured[CDF]) for name, values in measured.items()}
        publish(report, f"{day}_scored", started)
    means = {name: np.stack(scores).mean(0) for name, scores in exact.items()}
    report.update(ocf_vs_cdf={name: summarize(values, means[CDF]) for name, values in means.items()},
                  directional_vs_scalar=summarize(means[DIRECTIONAL], means[SCALAR]),
                  directional_vs_shuffled=summarize(means[DIRECTIONAL], means[SHUFFLED]),
                  ocf_rows_per_arm=len(sample) * len(HELD_DAYS), unique_exact_pairs=len(metric.values))
    publish(report, "complete", started)


if __name__ == "__main__":
    main()
