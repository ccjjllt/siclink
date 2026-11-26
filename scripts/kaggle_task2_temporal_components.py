"""Test whether rate-residual temporal components improve Task 2 across days.

The supervised axes are learned independently per unit on labeled source days.
All comparison axes have the same dimensionality, use the same 50–250 ms
counts and predict the same complete one-to-one gallery on each held day.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_hierarchical_gallery_validation import validated_logits
from kaggle_task3_metric_validation import unique_pair_ocf


MICE = ("mice1", "mice2")
SESSIONS = ("session1", "session2", "session3", "session4")
BASE_BIN_MS = 5
START_MS = 50
STOP_MS = 250
FINE_BIN_MS = 10
SCALE = 0.10
TEMPERATURE = 1 / 3
BASELINE = "v9"
STABLE = "signed_supervised"
UNCONDITIONED = "unconditioned_supervised"
VARIANTS = (STABLE, UNCONDITIONED, "early_late", "whitened_pca", "count_stratified_mispair")
ARMS = (BASELINE, *VARIANTS)


@dataclass
class SessionCounts:
    count25: torch.Tensor
    evoked10: torch.Tensor


def recordings(root, catalog, mouse, device, glyphs):
    fine = grouped_blocks(root, catalog, mouse, device, glyphs, bin_ms=BASE_BIN_MS)
    if tuple(fine) != SESSIONS or any(values.shape[1] != 4 for values in fine.values()):
        raise ValueError("Expected four labeled sessions with four repeats per glyph.")
    packed = {}
    for day in SESSIONS:
        values = fine.pop(day).unflatten(-1, (-1, STOP_MS // BASE_BIN_MS))
        counts25 = values.unflatten(-1, (-1, 25 // BASE_BIN_MS)).sum(dim=-1).flatten(-2)
        evoked = values[..., START_MS // BASE_BIN_MS:STOP_MS // BASE_BIN_MS]
        counts10 = evoked.unflatten(-1, (-1, FINE_BIN_MS // BASE_BIN_MS)).sum(dim=-1)
        expected_total = counts25.unflatten(-1, (-1, STOP_MS // 25))[..., START_MS // 25:].sum(dim=-1)
        if not torch.equal(counts10.sum(dim=-1), expected_total):
            raise ValueError("The temporal and original views must have identical evoked totals.")
        packed[day] = SessionCounts(counts25, counts10)
    return packed


def proportions(counts):
    weekday = counts.sum(dim=(0, 1)) + 0.5
    return weekday / weekday.sum(dim=-1, keepdim=True)


def smooth_time(values):
    padded = 0.50 * values
    padded[..., 1:].add_(values[..., :-1], alpha=0.25)
    padded[..., :-1].add_(values[..., 1:], alpha=0.25)
    padded[..., 0].add_(values[..., 0], alpha=0.25)
    padded[..., -1].add_(values[..., -1], alpha=0.25)
    return padded


def residual_time(counts, profile):
    """Project away the evoked total and smooth without crossing the window."""
    exposure = counts.sum(dim=-1, keepdim=True)
    return smooth_time((counts - exposure * profile) / (exposure + 2).sqrt())


def symmetrize(covariance):
    return 0.5 * (covariance + covariance.mT)


def unit_moments(responses):
    """Compute within-glyph repeat noise and unlabeled trial covariance per unit."""
    means = responses.mean(dim=1)
    repeat = (responses - means[:, None]).permute(2, 0, 1, 3).flatten(1, 2)
    all_trials = (responses - responses.mean(dim=(0, 1))).permute(2, 0, 1, 3).flatten(1, 2)
    repeat_cov = (repeat.mT @ repeat) / (len(responses) * (responses.shape[1] - 1))
    trial_cov = (all_trials.mT @ all_trials) / (len(responses) * responses.shape[1])
    per_day = (means - means.mean(dim=0)).transpose(0, 1)
    return per_day, repeat_cov, trial_cov


def stable_covariance(day_means):
    """Estimate stable glyph covariance without using same-session products."""
    n_days, _, n_glyphs, _ = day_means.shape
    pooled = day_means.sum(dim=0)
    cross = pooled.mT @ pooled - (day_means.mT @ day_means).sum(dim=0)
    return symmetrize(cross / (n_days * (n_days - 1) * n_glyphs))


def mispair(day_means, raw_days, rng):
    """Preserve source day profiles and count strata, changing the paired glyphs."""
    scrambled = []
    for means, raw in zip(day_means, raw_days):
        capacity = raw.sum(dim=(1, 2, 3))
        order = capacity.argsort()
        donor = torch.empty_like(order)
        for stratum in order.tensor_split((len(order) + 63) // 64):
            permuted = stratum[torch.from_numpy(rng.permutation(len(stratum)))]
            donor[permuted] = permuted.roll(1)
        if not bool((donor != torch.arange(len(order))).all()):
            raise RuntimeError("Source mispairing must keep every donor distinct from its source glyph.")
        scrambled.append(means[:, donor])
    return torch.stack(scrambled)


def first_axis(moment, whitening):
    values, axes = torch.linalg.eigh(symmetrize(whitening @ moment @ whitening))
    return (whitening @ axes[..., -1:]).squeeze(-1), values[..., -1]


def unit_whitening(source_responses):
    means, within, spread = (torch.stack(values) for values in zip(*(unit_moments(day) for day in source_responses)))
    within = symmetrize(within.mean(dim=0))
    # A ridge guards against silent units and the exact zero-sum temporal direction.
    eigenvalue, eigenvector = torch.linalg.eigh(within)
    floor = 0.30 * within.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True) + 0.002
    whitening = (eigenvector * (eigenvalue.clamp_min(0) + floor).rsqrt()[:, None]) @ eigenvector.mT
    return means, within, spread.mean(dim=0), whitening


def normalize_axis(axis, within):
    power = (axis * (within @ axis[..., None]).squeeze(-1)).sum(dim=1).clamp_min(0)
    return axis / (power + 0.005).sqrt()[:, None]


def directions(source_responses, raw_days, rng):
    means, within, spread, whitening = unit_whitening(source_responses)
    stable_axis, stable_power = first_axis(stable_covariance(means), whitening)
    pca_axis, _ = first_axis(spread, whitening)
    null_axis, null_power = first_axis(stable_covariance(mispair(means, raw_days, rng)), whitening)
    midpoint = within.shape[-1] // 2
    early_late = torch.ones_like(stable_axis)
    early_late[:, :midpoint] = -1
    orientations = {STABLE: stable_axis, "early_late": early_late, "whitened_pca": pca_axis,
                    "count_stratified_mispair": null_axis}
    result = {name: normalize_axis(axis - axis.mean(dim=-1, keepdim=True), within)
              for name, axis in orientations.items()}
    return result, {"units": len(stable_power), "axes_per_unit": 1,
                    "mean_max_stable_eigenvalue": float(stable_power.mean()),
                    "mean_max_mispaired_eigenvalue": float(null_power.mean())}


def project(responses, axis):
    return torch.einsum("gut,ut->gu", responses.mean(dim=1), axis)


def score_temporal(source, query, axis):
    """Use the same predictive day-deviation covariance recipe for every axis."""
    source_days = torch.stack([project(day, axis) for day in source])
    source_days -= source_days.mean(dim=1, keepdim=True)
    template = source_days.mean(dim=0)
    deviations = (source_days - template).flatten(0, 1)
    predictive = deviations.T @ deviations / ((len(source_days) - 1) * len(template))
    ridge = 0.30 * predictive.diagonal().mean().clamp_min(0.002)
    predictive.diagonal().add_(ridge)
    lower = torch.linalg.cholesky(symmetrize(predictive))
    observed = project(query, axis)
    # This anonymous day mean does not learn a candidate or a temporal basis.
    observed = observed - observed.mean(dim=0)

    def whiten(features):
        return torch.linalg.solve_triangular(lower, features.T.contiguous(), upper=False).T

    query_z, template_z = whiten(observed), whiten(template)
    return query_z @ template_z.T - 0.5 * template_z.square().sum(dim=1)


def scale(matrix):
    residual = matrix - matrix.mean(dim=1, keepdim=True)
    return residual / residual.square().mean().sqrt().clamp_min(1e-6)


def identities(matrix, order):
    inverse = order.argsort()
    true_score = matrix.gather(1, order[:, None])[:, 0]
    hits = matrix.topk(32, dim=1).indices == order[:, None]
    nll = torch.logsumexp(matrix / TEMPERATURE, dim=1) - true_score / TEMPERATURE
    return {"nll": nll[inverse], "top1": hits[:, 0][inverse].float(), "top32": hits.any(dim=1)[inverse].float()}


def paired(delta):
    return {"mean": float(delta.mean()), "paired_glyph_se": float(delta.std(unbiased=True) / len(delta)**0.5)}


def temporal_identity_metrics(results):
    values = {arm: {metric: float(vector.mean()) for metric, vector in metrics.items()} for arm, metrics in results.items()}
    changes = {arm: {metric: paired(vector - results[arm][metric]) for metric, vector in results[STABLE].items()}
               for arm in ARMS if arm != STABLE}
    replication = {metric: paired(vector - results[BASELINE][metric]) for metric, vector in results[UNCONDITIONED].items()}
    return {"all": values, "supervised_minus": changes, "unconditioned_minus_v9": replication}


def exact_metrics(matrix):
    values = {arm: {"ocf": float(matrix[:, index].mean()),
                    "vs_v9": paired((matrix[:, index] - matrix[:, 0]).mean(dim=0))} for index, arm in enumerate(ARMS)}
    changes = {arm: paired((matrix[:, 1] - matrix[:, index]).mean(dim=0))
               for index, arm in enumerate(ARMS) if arm != STABLE}
    return {"n_ocf_per_arm": matrix.shape[0] * matrix.shape[-1], "unique_glyphs": matrix.shape[-1],
            "arms": values, "supervised_minus": changes}


@torch.no_grad()
def evaluate(mouse, day, blocks, images, samples, scorer, args, rng):
    started = time.monotonic()
    order = torch.as_tensor(rng.permutation(len(images)))
    sources = {name: data for name, data in blocks.items() if name != day}
    trials = [source.evoked10 for source in sources.values()]
    templates = [proportions(raw) for raw in trials]
    residuals = [residual_time(raw, profile) for raw, profile in zip(trials, templates)]
    query_counts = blocks[day].evoked10[order]
    query = residual_time(query_counts, torch.stack(templates).mean(dim=0))
    axes, axis_report = directions(residuals, trials, rng)
    temporal_logits = {name: scale(score_temporal(residuals, query, orientation)) for name, orientation in axes.items()}
    del residuals, query

    raw_time = [smooth_time(day_counts.sqrt()) for day_counts in trials]
    raw_query = smooth_time(query_counts.sqrt())
    means, repeat_cov, _, whitening = unit_whitening(raw_time)
    raw_axis, raw_power = first_axis(stable_covariance(means), whitening)
    temporal_logits[UNCONDITIONED] = scale(score_temporal(raw_time, raw_query, normalize_axis(raw_axis, repeat_cov)))
    axis_report["mean_max_unconditioned_eigenvalue"] = float(raw_power.mean())
    del raw_time, raw_query, query_counts

    count_sources = {name: data.count25 for name, data in sources.items()}
    count_query = blocks[day].count25[order]
    torch.manual_seed(args.seed)
    with torch.enable_grad():
        validated = validated_logits(mouse, count_sources, count_query, images, images.device, args)
    hierarchy = fit_hierarchical_gallery(count_sources, view="total")
    eb = hierarchy.logits(count_query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    v9 = scale(blend_hierarchical_logits(validated, eb))
    logits = {BASELINE: v9, **{name: v9 + SCALE * values for name, values in temporal_logits.items()}}
    metrics = {name: identities(scores, order) for name, scores in logits.items()}
    report = {"mouse": mouse, "held_day": day, "source_days": list(sources), "axis_fit": axis_report,
              "identity": temporal_identity_metrics(metrics),
              "standalone_temporal": {name: {key: float(field.mean()) for key, field in identities(value, order).items()}
                                      for name, value in temporal_logits.items()}}
    exact = None
    if scorer is not None:
        assignments = {name: assign_glyphs(matrix) for name, matrix in logits.items()}
        report["full_one_to_one"] = {name: {"accuracy": float((guess == order).float().mean()),
                                          "unique": len(guess.unique()),
                                          "changes_vs_v9": int((guess != assignments[BASELINE]).sum())}
                                      for name, guess in assignments.items()}
        rows = order.argsort()[samples]
        candidate = torch.stack([assignments[name][rows] for name in ARMS])
        exact, count = unique_pair_ocf(candidate, samples.expand_as(candidate), images >= 128,
                                      scorer, images.device, 384)
        report["exact_ocf"] = exact_metrics(exact[None])
        report["exact_distinct_image_pairs"] = count
    report["seconds"] = time.monotonic() - started
    return metrics, exact, report


def scoring_glyphs(count, n_score, seed, *, avoid_relational_cohort=False):
    first = np.random.default_rng(17691).permutation(count)[:160]
    second_available = np.setdiff1d(np.arange(count), first)
    second = np.random.default_rng(63729).choice(second_available, size=160, replace=False)
    available = np.setdiff1d(second_available, second)
    if avoid_relational_cohort:
        # Recreate the original temporal and relational selections in their
        # respective eligibility sets before excluding their union here.
        temporal = np.random.default_rng(77317).choice(available, size=160, replace=False)
        relational = np.random.default_rng(22177).choice(second_available, size=192, replace=False)
        available = np.setdiff1d(available, np.union1d(temporal, relational))
    sample = np.random.default_rng(seed).choice(available, size=n_score, replace=False)
    return torch.as_tensor(sample), count - len(available)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=MICE, default=("mice2",))
    parser.add_argument("--held-days", nargs=2, choices=SESSIONS, default=("session2", "session4"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--split-seed", type=int, default=77317)
    parser.add_argument("--ocf-images", type=int, default=160)
    parser.add_argument("--avoid-relational-cohort", action="store_true",
                        help="also omit the original 160 temporal and 192 relational-geometry scoring glyphs")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_temporal_components.json"))
    parser.set_defaults(encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if len(set(args.held_days)) != 2 or args.ocf_images < 2:
        parser.error("Provide distinct held recordings and a paired OCF sample.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle CPU with competition data mounted.")
    torch.set_num_threads(4)
    start = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cpu")
    images = load_images(root, device)
    sample, excluded = scoring_glyphs(len(images), args.ocf_images, args.split_seed,
                                     avoid_relational_cohort=args.avoid_relational_cohort)
    scorer = official_scorer(root) if "mice2" in args.mice else None
    output = {"paper": "https://doi.org/10.1038/s41467-025-62069-2",
              "author_code": "https://github.com/XieLuanLab/temporal_code_stability",
              "scored_glyphs_per_mice2_day": len(sample), "model_seed": args.seed, "split_seed": args.split_seed,
              "earlier_scored_glyphs_excluded": excluded,
              "cohort_exclusion_seeds": [17691, 63729, 77317, 22177] if args.avoid_relational_cohort else [17691, 63729],
              "fine_bins_ms": FINE_BIN_MS, "evoked_window_ms": [START_MS, STOP_MS], "smoothing_weights": [0.25, 0.50, 0.25],
              "fixed_logit_increment": SCALE, "nll_temperature": TEMPERATURE, "gallery_size": len(images),
              "method": "The earlier three-axis SiClink method whitened source repeat noise and estimated same-glyph "
                        "cross-day covariance on unconditional square-root counts. This pilot tests subtraction of "
                        "each unit's expected per-day timing given its observed total against a same-width one-axis "
                        "square-root-count variant using the same training, smoothing, scoring and fusion. The held "
                        "day's conditional offset uses pooled source proportions. Additional controls use early–late, "
                        "whitened PCA or count-stratified mispairing only in the source axis fit. Every arm keeps "
                        "correctly labeled source templates and the same complete gallery. Mice1 is a v9 identity "
                        "diagnostic if requested. No formal test data.",
              "mice": {}}
    for mouse in dict.fromkeys(args.mice):
        blocks = recordings(root, catalog, mouse, device, len(images))
        report = {"folds": []}
        output["mice"][mouse] = report
        all_identities, all_ocf = [], []
        for day in args.held_days:
            sequence = np.random.SeedSequence((args.split_seed, MICE.index(mouse), SESSIONS.index(day)))
            metric, values, detail = evaluate(mouse, day, blocks, images, sample, scorer if mouse == "mice2" else None,
                                              args, np.random.default_rng(sequence))
            all_identities.append(metric)
            if values is not None:
                all_ocf.append(values)
            report["folds"].append(detail)
            pooled_identity = {name: {key: torch.stack([item[name][key] for item in all_identities]).mean(dim=0)
                                      for key in all_identities[0][name]} for name in ARMS}
            report["pooled_identity"] = temporal_identity_metrics(pooled_identity)
            if all_ocf:
                report["pooled_exact_ocf"] = exact_metrics(torch.stack(all_ocf))
            output["seconds"] = time.monotonic() - start
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(detail), flush=True)
        del blocks
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
