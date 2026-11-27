"""Replicate fixed-seven repeated-rate evidence on independent mice2 recording days."""

import json
from pathlib import Path
import time

import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import _evoked_counts
from glyph_mbr import contour_signatures, glyph_posterior
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standardized
from kaggle_posterior_utility_validation import fit_neural, v11_logits
from kaggle_task3_count_occupancy import paired_delta, posterior_metrics
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_utility_candidate import MOUSE, SHORTLIST, TABLE_SHA, validated_utility
from kaggle_task3_peer_rate_transfer import (BLIND, DIAGONAL, HELD, IMAGE, PRIOR_REPEATS, V20, WEIGHT,
                                             crossfit_visual_rates, eligible_glyphs, infer_rates, new_fold)
from kaggle_task3_rankband_ocf_validation import NeuralRows
from kaggle_task3_relational_consensus import UTILITY_WEIGHT, choose_images
from kaggle_task3_v20_candidate_headroom import held_repeats
from kaggle_task3_v20_exact_utility import original_panel, paired_summary, publish


SEED, PER_DAY = 374483, 96
DAYS = ("session2", "session4")
OUTPUT = Path("/kaggle/working/task3_peer_rate_recheck.json")


def replicate_split(size, cpu):
    _, first_held, first_exact, previous_v20, _ = new_fold(size, cpu)
    eligible, _ = eligible_glyphs(size, cpu)
    unused = eligible[~torch.isin(eligible, first_held)]
    if not len(DAYS) * PER_DAY <= len(unused) <= HELD:
        raise ValueError("The fixed recheck needs 192 fresh glyphs and the identical initial fit budget.")
    rng = torch.Generator().manual_seed(SEED)
    unused = unused[torch.randperm(len(unused), generator=rng)]
    # Keep the previous fit size using old ranking glyphs that never received exact OCF.
    padding = first_held[~torch.isin(first_held, torch.cat(tuple(first_exact.values())))]
    omitted = padding[torch.randperm(len(padding), generator=rng)[:HELD - len(unused)]]
    reserved = torch.cat((unused, omitted))
    gallery = torch.arange(size)
    fitting = gallery[~torch.isin(gallery, reserved)]
    fitting = fitting[torch.randperm(len(fitting), generator=rng)]
    endpoints = dict(zip(DAYS, unused[:len(DAYS) * PER_DAY].split(PER_DAY), strict=True))
    overlaps = {day: int(torch.isin(unused, glyphs).sum()) for day, glyphs in first_exact.items()}
    if any(overlaps.values()):
        raise ValueError("No scored glyph may match the first experiment's exact endpoints.")
    return fitting, reserved, unused, endpoints, previous_v20, len(omitted), overlaps


def contrasts(data):
    return {"visual_vs_v20": paired_delta(data[IMAGE], data[V20]),
            "blind_vs_v20": paired_delta(data[BLIND], data[V20]),
            "diagonal_vs_v20": paired_delta(data[DIAGONAL], data[V20]),
            "visual_vs_blind": paired_delta(data[IMAGE], data[BLIND]),
            "diagonal_vs_visual": paired_delta(data[DIAGONAL], data[IMAGE])}


@torch.no_grad()
def fixed_seven(logits, observed, expected, truth, exact_indices, utility, temperature):
    rows = torch.where(torch.isin(truth, exact_indices))[0]
    anchor = logits[rows]
    actions, stock_mass, incumbent = original_panel(anchor, temperature, utility)
    baseline, baseline_nll = posterior_metrics(glyph_posterior(logits, temperature), truth)
    rank, density, mass = {V20: baseline}, {}, {V20: stock_mass}
    factorial = torch.lgamma(observed + 1).sum(1)
    for name, rates in expected.items():
        count_logp = observed @ rates.log().T - rates.sum(1)
        density[name] = count_logp.gather(1, truth[:, None])[:, 0] - factorial
        readout = standardized(count_logp)
        posterior = glyph_posterior(logits + WEIGHT * readout, temperature)
        anchored, nll = posterior_metrics(posterior, truth)
        standalone, _ = posterior_metrics(glyph_posterior(readout, temperature), truth)
        rank[name] = {"anchored": anchored, "standalone": standalone, "nll_vs_v20": paired_delta(nll, baseline_nll),
                      "held_true_log_probability": float(density[name].mean())}
        mass[name] = posterior[rows]
    choices = choose_images(NeuralRows(anchor, anchor, actions), mass, utility)
    if not torch.equal(choices[V20], incumbent):
        raise ValueError("Fixed-seven calibration must reproduce the exact exploratory v20 endpoint.")
    choices = {arm: choices[arm] for arm in mass}
    return choices, truth[rows], {"ranks": rank,
                                  "count_log_p_visual_vs_blind_nats": paired_delta(density[IMAGE], density[BLIND]),
                                  "count_log_p_diagonal_vs_visual_nats": paired_delta(density[DIAGONAL], density[IMAGE]),
                                  "changed_choices": {arm: int((choice != incumbent).sum())
                                                      for arm, choice in choices.items()}}


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    cpu = torch.device("cpu")
    root = locate_data()
    gallery = load_images(root, cpu)
    masks = gallery >= 128
    fit, reserved, held, endpoints, previous, omitted, overlaps = replicate_split(len(gallery), cpu)
    utility, temperatures, _ = validated_utility(gallery, cpu)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(gallery), bin_ms=25)
    support, query, trial_overlap = held_repeats(recordings, previous)
    del recordings
    if any(value.shape[1] != 3 for value in support.values()) or any(value.shape[1] != 1 for value in query.values()):
        raise ValueError("All arms require three independent support repeats and one measured query repeat.")
    report = {"mouse": MOUSE, "seed": SEED, "fit_glyphs": len(fit), "seen_gallery_size": len(gallery),
              "fresh_rank_glyphs_per_day": len(held), "exact_distinct_glyphs_per_day": PER_DAY,
              "unused_old_glyphs_reserved_to_preserve_fit_budget": omitted, "days": DAYS,
              "overlap_with_first_exact_glyphs": overlaps, "trial_overlap": trial_overlap,
              "posterior_likelihood_weight": WEIGHT, "utility_weight": UTILITY_WEIGHT,
              "prior_exposure": PRIOR_REPEATS, "candidate_count": SHORTLIST, "utility_sha256": TABLE_SHA,
              "primary_contrast": "visual_prior vs v20 on the originally specified fixed-seven decision",
              "protocol": "Complementary training days and glyphs outside all earlier protected panels and the "
              "first study's entire rank cohort provide each genuine held single trial. Apply the original likelihood, "
              "0.30 evidence mixture, temperature and fixed seven-action v20 utility. The class-blind, "
              "cross-fitted visual and same-channel peer priors share observed responses and exposure. Reserve "
              "unscored prior study glyphs to keep the exact fit sample count and fold sizes. No ranking budgets "
              "or fitting parameters are changed. No formal data is read.", "sessions": {}}
    publish(report, OUTPUT, started, "split_ready")
    model = fit_neural(support, gallery, *contour_signatures(gallery), MOUSE, 211)
    baseline = {day: v11_logits(model, query[day][held], day, gallery) for day in DAYS}
    del model
    measured = torch.stack([_evoked_counts(values.float().mean(1), "timed") for values in support.values()])
    names = tuple(support)
    del support
    visual, report["visual_folds"] = crossfit_visual_rates(gallery, fit, reserved, measured)
    publish(report, OUTPUT, started, "rate_templates_ready")

    scorer = official_scorer(root)
    measured_ocf = {}
    for day in DAYS:
        rates, rate_detail = infer_rates(names.index(day), measured, visual, fit)
        observed = _evoked_counts(query[day][held, 0], "timed")
        choices, exact_truth, details = fixed_seven(baseline[day], observed, rates, held, endpoints[day], utility,
                                                   temperatures[day]["temperature"])
        report["sessions"][day] = {"held_exact_glyphs": exact_truth.tolist(), "templates": rate_detail, **details}
        publish(report, OUTPUT, started, f"{day}_posterior")
        guessed = torch.stack(tuple(choices.values()))
        values, unique = unique_pair_ocf(guessed, exact_truth.expand_as(guessed), masks, scorer, cpu, 96)
        measured_ocf[day] = dict(zip(choices, values, strict=True))
        report["sessions"][day].update(exact_pairs=unique, official_ocf=paired_summary(measured_ocf[day]),
                                       paired_contrasts=contrasts(measured_ocf[day]))
        publish(report, OUTPUT, started, f"{day}_official_ocf")
    paired = {arm: torch.cat([daily[arm] for daily in measured_ocf.values()])
              for arm in next(iter(measured_ocf.values()))}
    report.update(pooled_ocf=paired_summary(paired), paired_contrasts=contrasts(paired))
    publish(report, OUTPUT, started, "complete")
    print(json.dumps(report["paired_contrasts"]), flush=True)


if __name__ == "__main__":
    main()
