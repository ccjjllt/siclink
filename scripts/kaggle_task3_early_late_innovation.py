"""Test a candidate-conditioned early-to-late multiplicative spike model for Task 3."""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_measured_metric_validation import paired
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf
from kaggle_task3_mice2_ocf_recheck import new_cohorts, new_trial_split
from kaggle_task3_part_discovery import standardize_rows
from kaggle_task3_rankband_ocf_recheck import gather_repeats
from kaggle_task3_rankband_ocf_validation import stage
from kaggle_task3_repeat_posterior import fit_models
from kaggle_task3_shape_capacity_validation import ACTUAL, REPEAT_SEED as CAPACITY_SEED, cohorts, predict_panel


SEED = 31247
REPEAT_SEED = 31253
OUTER = 512
OCF = 256
MIX = 0.28
MEASURED_FRACTION = 0.30
EARLY = (1, 3)
LATE = ((3, 5), (5, 7), (7, 10))
ARMS = ("marginal", "population", "unit_matched", "unit_shuffled")
OUTPUT = Path("/kaggle/working/task3_early_late_innovation.json")


@dataclass(frozen=True)
class Windows:
    early: torch.Tensor
    late: torch.Tensor


def time_counts(values):
    shaped = values.float().unflatten(-1, (-1, 10))
    early = shaped[..., EARLY[0]:EARLY[1]].sum(-1)
    late = torch.stack([shaped[..., first:last].sum(-1) for first, last in LATE], dim=-1)
    return Windows(early, late)


def innovation(observed, expected):
    difference = observed - expected
    exposure = expected + 0.75
    total = difference.sum(dim=-1, keepdim=True)
    population = (total / exposure.sum(dim=-1, keepdim=True).sqrt()).clamp(-3.5, 3.5)
    distributed = exposure / exposure.sum(dim=-1, keepdim=True)
    within = ((difference - total * distributed) / exposure.sqrt()).clamp(-3.5, 3.5)
    return population, within


def partial_gain(signal, coefficient):
    # Remove the first normal-moment bias from exponentiating a centered innovation.
    return signal[..., None] * coefficient - 0.5 * coefficient.square()


def learn_coefficient(signal, observed, expected, offset, *, ridge, seed):
    shape = observed.shape[-2:]
    theta = torch.nn.Parameter(observed.new_zeros(shape))
    optimizer = torch.optim.AdamW([theta], lr=0.035, weight_decay=0.02)
    rng = torch.Generator(device=observed.device).manual_seed(seed)
    for _ in range(10):
        for indices in torch.randperm(len(observed), device=observed.device, generator=rng).split(1024):
            beta = 0.30 * theta.tanh()
            log_gain = (offset[indices] + partial_gain(signal[indices], beta)).clamp(-0.70, 0.70)
            correction = expected[indices] * log_gain.expm1() - observed[indices] * log_gain
            loss = correction.mean() + ridge * beta.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return (0.30 * theta.detach().tanh())


def fit_dynamics(training, priors, fit_indices):
    inputs, unit_trials, shuffled_trials, observations, rates = [], [], [], [], []
    rng = torch.Generator(device=fit_indices.device).manual_seed(SEED + 17)
    same_slots, total_slots = 0, 0
    for name, block in training.items():
        counted = time_counts(block[fit_indices])
        prior = priors[name]
        repeats = counted.early.shape[1]
        loo_early = (counted.early.sum(dim=1, keepdim=True) - counted.early) / (repeats - 1)
        loo_late = (counted.late.sum(dim=1, keepdim=True) - counted.late) / (repeats - 1)
        early = prior.early[fit_indices, None].lerp(loo_early, MEASURED_FRACTION)
        late = prior.late[fit_indices, None].lerp(loo_late, MEASURED_FRACTION).clamp_min(1e-5)
        population, own = innovation(counted.early, early)
        permutation = torch.rand(own.shape[:2], device=own.device, generator=rng).argsort(dim=1)
        # Same-glyph permutations preserve the empirical early-count distribution; the random
        # permutation includes fixed points to make the expected centered pairing independent.
        replacement = own.gather(1, permutation[..., None].expand_as(own))
        same_slots += int((permutation == torch.arange(repeats, device=own.device)).sum())
        total_slots += permutation.numel()
        inputs.append(population.flatten(0, 1))
        unit_trials.append(own.flatten(0, 1))
        shuffled_trials.append(replacement.flatten(0, 1))
        observations.append(counted.late.flatten(0, 1))
        rates.append(late.flatten(0, 1))

    population, own, shuffled, observed, expected = (torch.cat(values) for values in
                                                    (inputs, unit_trials, shuffled_trials, observations, rates))
    beta_pop = learn_coefficient(population, observed, expected, torch.zeros_like(observed), ridge=0.20, seed=SEED + 43)
    fixed = partial_gain(population, beta_pop)
    beta_unit = learn_coefficient(own, observed, expected, fixed, ridge=0.20, seed=SEED + 67)
    beta_shuffle = learn_coefficient(shuffled, observed, expected, fixed, ridge=0.20, seed=SEED + 67)
    return (beta_pop, beta_unit, beta_shuffle), {"training_trials": len(observed), "shuffled_fixed_point_fraction": same_slots / total_slots,
             "mean_absolute_population_coefficients": beta_pop.abs().mean(dim=0).tolist(),
             "mean_absolute_unit_coefficients": beta_unit.abs().mean(dim=0).tolist(),
             "mean_absolute_shuffled_unit_coefficients": beta_shuffle.abs().mean(dim=0).tolist(),
             "negative_unit_coefficients_fraction": (beta_unit < 0).float().mean(dim=0).tolist()}


@torch.no_grad()
def marginal_and_conditional(query, rates, coefficients):
    beta_pop, beta_unit, beta_shuffle = coefficients
    observed = time_counts(query[:, 0])
    log_early = rates.early.clamp_min(1e-5).log()
    late_rates = rates.late.clamp_min(1e-5)
    early = observed.early @ log_early.T - rates.early.sum(dim=1)
    late = observed.late.flatten(1) @ late_rates.flatten(1).log().T - late_rates.sum(dim=(1, 2))
    baseline = early + late
    result = {name: baseline.clone() for name in ARMS}
    for begin in range(0, len(observed.early), 32):
        row = slice(begin, begin + 32)
        for start in range(0, len(rates.early), 192):
            col = slice(start, start + 192)
            p, u = innovation(observed.early[row, None], rates.early[None, col])
            group = partial_gain(p, beta_pop)
            for name, offset in (("population", group), ("unit_matched", group + partial_gain(u, beta_unit)),
                                 ("unit_shuffled", group + partial_gain(u, beta_shuffle))):
                offset = offset.clamp(-0.70, 0.70)
                extra = observed.late[row, None] * offset - late_rates[None, col] * offset.expm1()
                result[name][row, col] += extra.sum(dim=(2, 3))
    return result


def select(scores, pool):
    if pool is None:
        return scores.argmax(dim=1)
    return pool.gather(1, scores.gather(1, pool).argmax(dim=1, keepdim=True))[:, 0]


def expanded(scores, pool):
    remaining = scores.clone().scatter_(1, pool, -torch.inf)
    return torch.cat((pool, remaining.topk(9, dim=1).indices), dim=1)


def rank_quality(score, labels):
    hit = score.gather(1, labels[:, None])
    rank = 1 + (score > hit).sum(dim=1)
    return {**{f"top{depth}": float((rank <= depth).float().mean()) for depth in (1, 5, 16, 32)},
            "common_scale_nll": float((score.logsumexp(dim=1) - hit[:, 0]).mean())}


def make_cohort(n, device):
    _, _, old, _ = new_cohorts(n, device)
    _, _, first = cohorts(n, device)
    all_rows = torch.arange(n, device=device)
    valid = all_rows[~torch.isin(all_rows, torch.cat((old, first)))]
    picks = torch.as_tensor(np.random.default_rng(SEED).permutation(len(valid))[:OUTER], device=device)
    held = valid[picks]
    return all_rows[~torch.isin(all_rows, held)], held


def independent_trials(recordings):
    indices = {name: torch.arange(4, device=counts.device).reshape(1, 4, 1).expand(len(counts), -1, -1)
               for name, counts in recordings.items()}
    _, first_inner, _, first_outer = nested_repeats(indices, CAPACITY_SEED)
    _, second_inner, _, second_outer = new_trial_split(indices)
    fit, held = {}, {}
    for d, (name, counts) in enumerate(recordings.items()):
        alternatives = torch.cat((first_inner[name], second_inner[name]), dim=1)
        _, outer = split_repeats(alternatives, REPEAT_SEED + d)
        outer = outer[..., 0]
        if (outer == first_outer[name][..., 0]).any() or (outer == second_outer[name][..., 0]).any():
            raise RuntimeError(f"The new query is one of the prior evaluation repeats in {name}.")
        bank = indices[name][..., 0]
        train = bank[bank != outer].reshape(len(counts), 3)
        fit[name], held[name] = gather_repeats(counts, train), gather_repeats(counts, outer)
    return fit, held


@torch.no_grad()
def evaluate(panel, modalities, labels, undo, masks, scanner):
    pool = panel.shortlist
    v14 = standardize_rows(panel.logits)
    ranked = {ACTUAL: v14, **{name: v14.lerp(standardize_rows(score), MIX) for name, score in modalities.items()}}
    choices = {ACTUAL: select(panel.logits, pool),
               **{name: select(ranked[name], pool) for name in modalities}}
    if pool is not None:
        choices.update({name + "_expanded16": select(scores, expanded(scores, pool)) for name, scores in ranked.items()})
    scored_rows = undo[:OCF]
    reference = labels[scored_rows]
    selected = torch.stack([value[scored_rows] for value in choices.values()])
    exact, count = unique_pair_ocf(selected, reference.expand_as(selected), masks, scanner, masks.device, 384)
    scores = dict(zip(choices, exact.cpu().numpy()))
    true_logp = {name: value.gather(1, labels[:, None])[:, 0] for name, value in modalities.items()}
    diagnosis = {"rank": {name: rank_quality(value, labels) for name, value in ranked.items()},
                 "held_true_likelihood_delta_nats_per_trial": {name: float((value - true_logp["marginal"]).mean())
                                                                 for name, value in true_logp.items()},
                 "unique_exact_pairs": count,
                 "changed_vs_actual_v14": {name: int((value != choices[ACTUAL]).sum()) for name, value in choices.items()}}
    return scores, diagnosis


def summarize(data):
    control = {name: paired(value, data[ACTUAL]) for name, value in data.items()}
    for suffix in ("", "_expanded16"):
        target = "unit_matched" + suffix
        if target in data:
            for other in ("population", "unit_shuffled", "marginal"):
                control[f"unit_vs_{other}{suffix}"] = paired(data[target], data[other + suffix])
    return control


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on a private Kaggle T4 using the official labeled training repeats.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    args = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128, query_batch=64)
    root = locate_data()
    images = load_images(root, device)
    masks = images >= 128
    signatures, energy = contour_signatures(images)
    fit_indices, held_indices = make_cohort(len(images), device)
    scorer = official_scorer(root)
    catalog = load_catalog(root)
    report = {"protocol": {"sample_seed": SEED, "trial_seed": REPEAT_SEED, "outer_glyphs_per_day": OUTER,
                           "ocf_glyphs_per_day": OCF, "fit_glyphs_for_conditional_model": len(fit_indices),
                           "early_window_ms": [n * 25 for n in EARLY], "late_windows_ms": [[a * 25, b * 25] for a, b in LATE],
                           "forward_rate_measured_fraction": MEASURED_FRACTION, "frozen_component_mix": MIX,
                           "prior_capacity_and_exact_ocf_outer_trial_overlap": 0, "official_test_read": False,
                           "conditional": "Use a candidate's expected early spike count to form a standardized innovation. "
                                          "A shared population innovation and a population-adjusted same-unit innovation give "
                                          "regularized multiplicative updates to each later expected count. Fit on legal labeled repeats "
                                          "with OCF-scored glyphs excluded from the coefficient fit. The null randomly permutes early "
                                          "trial innovations within a fitting glyph, preserving the glyph's empirical marginal and frozen population term. "
                                          "All models score the same known-glyph full gallery. The exact v14 candidate policy and same-size 16 "
                                          "candidate lists for mice1 provide matched image decisions without using held labels."},
              "mice": {}}
    rng = np.random.default_rng(SEED + 1)

    for mouse in ("mice1", "mice2"):
        all_trials = grouped_blocks(root, catalog, mouse, device, len(images))
        training, held = independent_trials(all_trials)
        del all_trials
        baseline, fitted = fit_models(training, images, signatures, energy, mouse, args)
        priors = {name: time_counts(baseline.encoder.expected_counts(images, name)) for name in training}
        coefficients, dynamic_report = fit_dynamics(training, priors, fit_indices)
        details = {"v14": fitted, "conditional_fit": dynamic_report, "sessions": {}}
        report["mice"][mouse] = details
        stage("conditional_fit", started, mouse=mouse, **dynamic_report)
        observed = {}
        for name, queries in held.items():
            order = torch.as_tensor(rng.permutation(len(held_indices)), device=device)
            labels = held_indices[order]
            query = queries[labels]
            panel = predict_panel(baseline, query, images, name, mouse, args)
            train_mean = time_counts(training[name].mean(dim=1))
            candidate = Windows(priors[name].early.lerp(train_mean.early, MEASURED_FRACTION).clamp_min(1e-5),
                                priors[name].late.lerp(train_mean.late, MEASURED_FRACTION).clamp_min(1e-5))
            modalities = marginal_and_conditional(query, candidate, coefficients)
            values, diagnostic = evaluate(panel, modalities, labels, order.argsort(), masks, scorer)
            for arm, block in values.items():
                observed.setdefault(arm, []).append(block)
            details["sessions"][name] = {"paired_ocf": summarize(values), **diagnostic}
            report["seconds"] = time.monotonic() - started
            OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            stage("scored_recording", started, mouse=mouse, session=name, **details["sessions"][name])
        details["summary"] = summarize({arm: np.stack(values) for arm, values in observed.items()})
        report["seconds"] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    stage("complete", started, mice={name: values["summary"] for name, values in report["mice"].items()})


if __name__ == "__main__":
    main()
