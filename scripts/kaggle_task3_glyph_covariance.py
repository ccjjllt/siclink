"""Test stimulus-specific spike correlations at identical predictive means and variances."""

import itertools
import json
from math import log, pi
from pathlib import Path
import time

import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures, glyph_posterior
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standardized
from kaggle_posterior_utility_validation import fit_neural, v11_logits
from kaggle_task3_count_occupancy import paired_delta, posterior_metrics
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_full_balance_replication import cohorts, early_cohorts
from kaggle_task3_mice2_utility_candidate import MOUSE, SHORTLIST, TABLE_SHA, validated_utility
from kaggle_task3_native_hanzi_nomination import fresh_glyphs
from kaggle_task3_rankband_ocf_validation import NeuralRows
from kaggle_task3_relational_consensus import UTILITY_WEIGHT, choose_images
from kaggle_task3_v20_candidate_headroom import held_repeats
from kaggle_task3_v20_exact_utility import original_panel, paired_summary


SEED, RANK = 365227, 12
DAYS = ("session2", "session4")
HELD, EXACT_PER_DAY = 320, 64
WINDOWS = ((50, 100), (100, 175), (175, 250))
GLYPH_FRACTION, V20_MIX = 0.35, 0.30
V20, POOLED, VARIANCE = "v20", "pooled_covariance", "glyph_variance_only"
SHARED, GLYPH, SHUFFLED = "shared_correlation", "glyph_correlation", "within_rate_ink_shuffled_correlation"
MODELS = (POOLED, VARIANCE, SHARED, GLYPH, SHUFFLED)
DIRECT = (SHARED, GLYPH, SHUFFLED)
WORKING = Path("/kaggle/working/task3_glyph_covariance.json")


def time_features(raw):
    bins = raw.unflatten(-1, (-1, 10))
    counts = torch.stack([bins[..., start // 25:end // 25].sum(-1) for start, end in WINDOWS], dim=-1)
    return (counts.flatten(-2) + 0.375).sqrt()


def split_glyphs(size, cpu):
    visual, previous, calibration, _ = cohorts(size, cpu)
    native, _, _ = fresh_glyphs(size, cpu)
    _, early = early_cohorts(size, cpu)
    excluded = torch.unique(torch.cat((visual, previous, calibration, native, early)))
    gallery = torch.arange(size)
    eligible = gallery[~torch.isin(gallery, excluded)]
    if len(eligible) < HELD:
        raise ValueError("Need enough glyphs outside previous v20 and named control panels.")
    rng = torch.Generator().manual_seed(SEED)
    held = eligible[torch.randperm(len(eligible), generator=rng)[:HELD]]
    endpoints = held[:len(DAYS) * EXACT_PER_DAY].split(EXACT_PER_DAY)
    fit = gallery[~torch.isin(gallery, held)]
    return fit, held, dict(zip(DAYS, endpoints, strict=True)), previous, len(eligible)


@torch.no_grad()
def normalized_support(support, fit):
    scales, centers, residuals, covariance = {}, {}, {}, {}
    for day, raw in support.items():
        values = time_features(raw)
        grouped = values.mean(1)
        residual = values - grouped[:, None]
        fit_residual = residual[fit].flatten(0, 1)
        dof = len(fit) * (raw.shape[1] - 1)
        noise = (fit_residual.square().sum(0) / dof + 0.03).sqrt()
        centering = grouped[fit].mean(0)
        scales[day] = (centering, noise)
        centers[day] = (grouped - centering) / noise
        residuals[day] = residual / noise
        within = fit_residual / noise
        covariance[day] = within.T @ within / dof
    spectrum, axes = torch.linalg.eigh(torch.stack(tuple(covariance.values())).mean(0))
    basis = axes[:, -RANK:].contiguous()
    projected, global_variance, reliability_features = {}, {}, {}
    for day, values in residuals.items():
        residual = values @ basis
        individual = residual.transpose(1, 2) @ residual / (values.shape[1] - 1)
        population = individual[fit].mean(0)
        shared_sd = population.diagonal().clamp_min(0.05).sqrt()
        projected[day] = individual / (shared_sd[:, None] * shared_sd)
        global_variance[day] = population
        deviations = projected[day][fit] - projected[day][fit].mean(0)
        first, second = torch.triu_indices(RANK, RANK, offset=1)
        reliability_features[day] = deviations[:, first, second]
    reproducibility = {}
    for first, second in itertools.combinations(reliability_features, 2):
        a, b = reliability_features[first].flatten(), reliability_features[second].flatten()
        reproducibility[f"{first}_{second}"] = float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-8))
    return scales, centers, basis, projected, global_variance, {
        "mode_eigenvalues": spectrum[-RANK:].tolist(), "fit_cross_day_offdiagonal_repeatability": reproducibility,
    }


def matched_permutation(strength, ink, seed):
    """Permute class correlation matrices while retaining coarse rate and morphology."""
    rng = torch.Generator().manual_seed(seed)
    permutation = torch.arange(len(ink))
    for rate_group in strength.argsort().tensor_split(12):
        for stratum in rate_group[ink[rate_group].argsort()].tensor_split(6):
            draw = stratum[torch.randperm(len(stratum), generator=rng)]
            permutation[draw] = draw.roll(1)
    if (permutation == torch.arange(len(ink))).any() or len(permutation.unique()) != len(ink):
        raise ValueError("The rate-and-ink control must derange all covariance class labels.")
    return permutation


def correlation(covariance):
    sd = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0.05).sqrt()
    values = covariance / (sd[..., None] * sd[..., None, :])
    values = (values + values.transpose(-1, -2)) * 0.5
    values.diagonal(dim1=-2, dim2=-1).fill_(1)
    return values


@torch.no_grad()
def covariance_models(day, projected, global_variance, permutation):
    shared_cov = global_variance[day]
    shared_var = shared_cov.diagonal().clamp_min(0.05)
    identity = torch.eye(RANK, dtype=shared_cov.dtype)
    shared_corr = 0.90 * correlation(shared_cov) + 0.10 * identity
    empirical = torch.stack(tuple(projected.values())).mean(0)
    individual_var = empirical.diagonal(dim1=-2, dim2=-1).clamp(0.25, 4)
    variance = shared_var * ((1 - GLYPH_FRACTION) + GLYPH_FRACTION * individual_var)
    magnitude = variance.sqrt()
    scale = magnitude[..., None] * magnitude[..., None, :]
    glyph_corr = correlation(empirical)
    true = (1 - GLYPH_FRACTION) * shared_corr + GLYPH_FRACTION * glyph_corr
    shuffled = (1 - GLYPH_FRACTION) * shared_corr + GLYPH_FRACTION * glyph_corr[permutation]
    matrices = {POOLED: shared_corr * (shared_var[:, None] * shared_var).sqrt(),
                VARIANCE: torch.diag_embed(variance), SHARED: shared_corr * scale,
                GLYPH: true * scale, SHUFFLED: shuffled * scale}
    matrices[POOLED] = matrices[POOLED].expand_as(matrices[GLYPH])
    other = (VARIANCE, SHARED, GLYPH, SHUFFLED)
    worst = max(float((matrices[name].diagonal(dim1=-2, dim2=-1) - variance).abs().max()) for name in other)
    if worst > 1e-5:
        raise RuntimeError("The four class-variance arms must have identical projected marginal variances.")
    change = (true - shared_corr).square().mean().sqrt()
    return matrices, {"max_matched_projected_variance_difference": worst, "glyph_correlation_rms_update": float(change)}


@torch.no_grad()
def gaussian_scores(observed, means, basis, noise, matrices):
    """Exact Gaussian log density after the specified square-root count transform."""
    z_obs, z_mean = observed @ basis, means @ basis
    orth_obs, orth_mean = observed - z_obs @ basis.T, means - z_mean @ basis.T
    common = orth_obs @ orth_mean.T - 0.5 * (orth_obs.square().sum(1)[:, None] + orth_mean.square().sum(1))
    common -= noise.log().sum() + 0.5 * observed.shape[1] * log(2 * pi)
    outer = (z_obs[:, :, None] * z_obs[:, None, :]).flatten(1)
    scores = {}
    for name, covariance in matrices.items():
        factor = torch.linalg.cholesky(covariance.contiguous())
        precision = torch.cholesky_inverse(factor)
        coefficient = (precision @ z_mean[..., None])[..., 0]
        normalization = factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        energy = outer @ precision.flatten(1).T - 2 * z_obs @ coefficient.T + (coefficient * z_mean).sum(1)
        scores[name] = common - 0.5 * energy - normalization
    return scores


def near_shape_quality(scores, labels, neighbors):
    true = scores.gather(1, labels[:, None])
    close = scores.gather(1, neighbors[labels])
    delta = true - close
    return {"true_beats_visual_neighbors": float((delta > 0).float().mean()),
            "best_of_17": float((delta >= 0).all(1).float().mean())}


def publish(report, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    WORKING.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    cpu = torch.device("cpu")
    root = locate_data()
    images = load_images(root, cpu)
    masks = images >= 128
    utility, temperatures, _ = validated_utility(images, cpu)
    visual = (utility + utility.T).clone().fill_diagonal_(-torch.inf).topk(16).indices
    fit, held, references, previous, eligible = split_glyphs(len(images), cpu)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images), bin_ms=25)
    support, query, overlap = held_repeats(recordings, previous)
    del recordings
    report = {"mouse": MOUSE, "seed": SEED, "candidate_budget": SHORTLIST, "gallery_size": len(images),
              "shared_covariance_fit_glyphs": len(fit), "fresh_glyphs": len(held), "fresh_eligible": eligible,
              "queried_days": DAYS, "paired_exact_glyphs_per_day": EXACT_PER_DAY, "trial_overlap": overlap,
              "windows_ms": WINDOWS, "rank": RANK, "class_covariance_fraction": GLYPH_FRACTION,
              "v20_posterior_mix": V20_MIX, "v20_utility_weight": UTILITY_WEIGHT, "utility_sha256": TABLE_SHA,
              "protocol": "Use three independent support trials per labeled glyph per day. Learn a shared residual "
              "basis only on fit glyphs. Use the same support means in every arm and estimate shrunken glyph-level "
              "covariance from residual pairs across the four recordings. Match projected marginal variances "
              "exactly among variance-only, shared correlation, same-glyph correlation, and glyph correlations "
              "deranged within total-rate/ink strata. Score Gaussian predictive density on transformed counts, "
              "including the covariance determinant. Compute standalone and anchored ranks against the full gallery. "
              "Keep v20 actions and its utility fixed for prospective choices. Also score direct Gaussian "
              "argmax in the same seven as a diagnostic. Scoring glyphs are outside the v20 utility/temperature "
              "fits and earlier named panels. No formal data is opened.", "sessions": {}}
    publish(report, started, "split_ready")

    baseline = fit_neural(support, images, *contour_signatures(images), MOUSE, 211)
    scale, mean, basis, projected, pooled, geometry = normalized_support(support, fit)
    strength = torch.stack([block.mean(1).sum(1) for block in support.values()]).mean(0)
    permutation = matched_permutation(strength, masks.float().mean((1, 2)), SEED + 10)
    del support
    report["covariance_geometry"] = geometry
    publish(report, started, "covariance_ready")
    scorer = official_scorer(root)
    values = {}
    for day in DAYS:
        center, noise = scale[day]
        counts = query[day][held]
        observed = (time_features(counts[:, 0]) - center) / noise
        peer_centers = torch.stack([m for peer, m in mean.items() if peer != day]).mean(0)
        means = mean[day].lerp(peer_centers, 0.25)
        matrices, diagnostics = covariance_models(day, projected, pooled, permutation)
        joint = gaussian_scores(observed, means, basis, noise, matrices)
        logits = v11_logits(baseline, counts, day, images)
        temperature = temperatures[day]["temperature"]
        baseline_mass = glyph_posterior(logits, temperature)
        baseline_report, identity = posterior_metrics(baseline_mass, held)
        baseline_report.update(near_shape_quality(logits, held, visual))
        scored = torch.where(torch.isin(held, references[day]))[0]
        choice_logits = logits[scored]
        actions, original_mass, incumbent = original_panel(choice_logits, temperature, utility)
        posteriors = {V20: original_mass}
        rank_details, true_density, direct = {}, {}, {}
        for name, density in joint.items():
            score = standardized(density)
            anchor = logits + V20_MIX * score
            mass = glyph_posterior(anchor, temperature)
            shape_quality, nll = posterior_metrics(mass, held)
            shape_quality.update(near_shape_quality(anchor, held, visual))
            standalone, _ = posterior_metrics(glyph_posterior(score, temperature), held)
            standalone.update(near_shape_quality(density, held, visual))
            true_density[name] = density.gather(1, held[:, None])[:, 0]
            rank_details[name] = {"anchored": shape_quality, "standalone": standalone,
                                  "anchored_identity_nll_vs_v20": paired_delta(nll, identity),
                                  "mean_gaussian_log_density": float(true_density[name].mean())}
            posteriors[name] = mass[scored]
            if name in DIRECT:
                direct[name + "_direct7"] = actions.gather(1, score[scored].gather(1, actions).argmax(1, keepdim=True))[:, 0]
        density_differences = {name: paired_delta(true_density[GLYPH], true_density[name])
                               for name in (POOLED, VARIANCE, SHARED, SHUFFLED)}
        stock = NeuralRows(choice_logits, choice_logits, actions)
        choices = choose_images(stock, posteriors, utility)
        if not torch.equal(choices[V20], incumbent):
            raise RuntimeError("The fixed seven-action endpoint must exactly reproduce v20.")
        decisions = {name: choices[name] for name in posteriors} | direct
        report["sessions"][day] = {"v20": baseline_report, "density_models": rank_details,
                                   "matched_covariance": diagnostics, "conditional_log_density_gain": density_differences,
                                   "exact_glyph_indices": held[scored].tolist(),
                                   "changed_choices": {name: int((choice != incumbent).sum())
                                                       for name, choice in decisions.items()}}
        publish(report, started, f"{day}_likelihoods")
        candidates = torch.stack(tuple(decisions.values()))
        exact, unique = unique_pair_ocf(candidates, held[scored].expand_as(candidates), masks, scorer, cpu, 96)
        values[day] = dict(zip(decisions, exact, strict=True))
        report["sessions"][day].update(exact_pairs=unique, ocf=paired_summary(values[day]))
        publish(report, started, f"{day}_official_ocf")
    combined = {name: torch.cat([rows[name] for rows in values.values()]) for name in next(iter(values.values()))}
    report["pooled_ocf"] = paired_summary(combined)
    report["real_vs_matched_controls_ocf"] = {name: paired_delta(combined[GLYPH], combined[name])
                                             for name in (SHARED, SHUFFLED)}
    report["direct_seven_real_vs_matched_controls_ocf"] = {
        name: paired_delta(combined[GLYPH + "_direct7"], combined[name + "_direct7"]) for name in (SHARED, SHUFFLED)
    }
    publish(report, started, "complete")
    print(json.dumps(report["pooled_ocf"]), flush=True)


if __name__ == "__main__":
    main()
