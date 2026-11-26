"""Recheck half-balanced Task 3 posteriors against a matched exact-utility policy."""

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_exact_utility import gallery_geometry, gallery_utility
from glyph_mbr import contour_signatures, fit_temperature, glyph_posterior, similarity_from_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_early_late_innovation import make_cohort as early_cohorts
from kaggle_task3_measured_metric_validation import paired
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_microns_gamma_prior import cohorts as microns_cohorts
from kaggle_task3_mice2_ocf_recheck import new_cohorts as mice2_ocf_cohorts, new_trial_split as mice2_ocf_repeats
from kaggle_task3_rankband_ocf_recheck import PRIOR_GLYPH_SEED, gather_repeats, glyph_cohorts as rankband_cohorts
from kaggle_task3_rankband_ocf_validation import NeuralRows, fit_visual_metric, neural_rows, stage
from kaggle_task3_relational_consensus import (ACTUAL, INDEPENDENT, PARTIAL_BALANCE, UTILITY_WEIGHT, choose_images,
                                              cohorts as relational_cohorts, independent_trials as relational_repeats)
from kaggle_task3_repeat_posterior import fit_models
from kaggle_task3_shape_capacity_validation import REPEAT_SEED as CAPACITY_REPEAT, cohorts as capacity_cohorts, predict_panel
from kaggle_task3_utility_coverage import new_cohorts as utility_cohorts
from kaggle_task3_utility_nomination_recheck import independent_cohorts as nomination_cohorts


SEED = 51977
REPEAT_SEED = 51991
OUTER = 256
TEMP = 192
SHORTLIST = 7
HALF = "half_balanced_utility"
FULL = "fully_balanced_utility"
TWO_MICE = ("mice1", "mice2")
OUTPUT = Path("/kaggle/working/task3_half_balance_recheck.json")


def cohorts(size, device):
    visual, _, capacity = capacity_cohorts(size, device)
    _, early = early_cohorts(size, device)
    _, _, microns, _ = microns_cohorts(size, device)
    _, _, _, relational, _ = relational_cohorts(size, device)
    _, _, rankband = rankband_cohorts(size, device)
    prior = torch.as_tensor(np.random.default_rng(PRIOR_GLYPH_SEED).permutation(size), device=device)
    original_rankband = prior[1088:1280]
    _, _, coverage = utility_cohorts(size, device)
    _, _, mice2_previous, _ = mice2_ocf_cohorts(size, device)
    previous = {
        "mice1": {"shape_capacity": capacity, "early_late_innovation": early, "microns_gamma": microns,
                  "relational_consensus": relational, "rankband_initial": original_rankband, "rankband_recheck": rankband,
                  "utility_coverage": coverage, "utility_nomination_recheck": nomination_cohorts(size, device).outer},
        "mice2": {"shape_capacity": capacity, "early_late_innovation": early, "mice2_exact_ocf_recheck": mice2_previous,
                  "soft_exclusion": torch.as_tensor(np.random.default_rng(1531).permutation(size)[2432:2560], device=device),
                  "older_posterior_utility": torch.as_tensor(np.random.default_rng(352).permutation(size)[192:288], device=device)},
    }
    partitions, diagnostics = {}, {}
    for index, (mouse, old) in enumerate(previous.items()):
        removed = torch.unique(torch.cat(tuple(old.values())))
        order = torch.as_tensor(np.random.default_rng(SEED + index).permutation(size), device=device)
        available = order[~torch.isin(order, removed)]
        if len(available) < OUTER + TEMP:
            raise RuntimeError(f"The known earlier scored cohorts leave too few fresh glyphs for {mouse}.")
        held, temperature = available[:OUTER + TEMP].split((OUTER, TEMP))
        overlaps = {name: int(torch.isin(held, rows).sum()) for name, rows in old.items()}
        if any(overlaps.values()) or torch.isin(held, temperature).any():
            raise RuntimeError(f"A fresh OCF glyph overlaps earlier scores or current calibration for {mouse}.")
        partitions[mouse] = (held, temperature)
        diagnostics[mouse] = {"previous_distinct_scoring_glyphs_excluded": len(removed), "outer_glyph_overlap": overlaps}
    reserved = torch.cat([rows for partition in partitions.values() for rows in partition])
    visual = visual[~torch.isin(visual, reserved)]
    if len(visual) <= 512 or torch.isin(visual, reserved).any():
        raise RuntimeError("Reserve independent image-pair supervision shared by both mice.")
    return visual, partitions, diagnostics


def disjoint_trial_split(recordings, mouse):
    axis = {day: torch.arange(4, device=counts.device).reshape(1, 4, 1).expand(len(counts), -1, -1)
            for day, counts in recordings.items()}
    previous_protocol = relational_repeats if mouse == "mice1" else mice2_ocf_repeats
    _, _, _, previous = previous_protocol(axis)
    inner_fit, inner_held, outer_fit, outer_held = {}, {}, {}, {}
    for index, (day, counts) in enumerate(recordings.items()):
        _, capacity = split_repeats(axis[day], CAPACITY_REPEAT + index)
        ids = axis[day][..., 0]
        excluded = (ids == previous[day][..., 0]) | (ids == capacity[..., 0])
        generator = torch.Generator(device=counts.device).manual_seed(REPEAT_SEED + index)
        draws = torch.rand(ids.shape, device=counts.device, generator=generator).masked_fill(excluded, -1)
        held = draws.argmax(dim=1, keepdim=True)
        support = ids[ids != held].reshape(len(counts), 3)
        local_inner, inner = split_repeats(support[..., None], REPEAT_SEED + 1000 + index)
        if excluded.gather(1, held).any() or (inner[..., 0] == held).any():
            raise RuntimeError(f"A held trial overlaps a previous holdout or current calibration for {mouse} {day}.")
        inner_fit[day], inner_held[day] = gather_repeats(counts, local_inner[..., 0]), gather_repeats(counts, inner[..., 0])
        outer_fit[day], outer_held[day] = gather_repeats(counts, support), gather_repeats(counts, held)
    return inner_fit, inner_held, outer_fit, outer_held


@torch.no_grad()
def v14_panel(model, queries, gallery, day, mouse, args):
    if mouse == "mice1":
        return neural_rows(model, queries, gallery, day, args)
    scores = predict_panel(model, queries, gallery, day, mouse, args).logits
    leader = scores.argmax(dim=1, keepdim=True)
    following = scores.clone().scatter_(1, leader, -torch.inf).topk(SHORTLIST - 1, dim=1).indices
    return NeuralRows(scores, scores, torch.cat((leader, following), dim=1))


def summary(scores):
    return {**{arm: paired(values, scores[ACTUAL]) for arm, values in scores.items()},
            "half_vs_independent": paired(scores[HALF], scores[INDEPENDENT]),
            "full_vs_independent": paired(scores[FULL], scores[INDEPENDENT]),
            "half_vs_full": paired(scores[HALF], scores[FULL])}


@torch.no_grad()
def evaluate(panel, temperature, truth, held, utility, masks, scorer):
    independent = glyph_posterior(panel.mixed, temperature)
    full = glyph_posterior(panel.mixed, temperature, balance=True)
    half = independent.clamp_min(1e-25).log().lerp(full.clamp_min(1e-25).log(), PARTIAL_BALANCE).softmax(dim=1)
    posteriors = {INDEPENDENT: independent, HALF: half, FULL: full}
    choices = choose_images(panel, posteriors, utility)
    rows = truth.argsort()[held]
    if panel.action.shape != (len(truth), SHORTLIST) or not torch.equal(truth[rows], held):
        raise RuntimeError("All methods must retain the frozen seven glyphs and paired reference ordering.")
    predicted = torch.stack([choices[arm][rows] for arm in choices])
    exact, pairs = unique_pair_ocf(predicted, held.expand_as(predicted), masks, scorer, masks.device, 384)
    measured = dict(zip(choices, exact.cpu().numpy(), strict=True))
    metrics = {}
    for arm, probability in posteriors.items():
        rank = probability[rows].topk(32, dim=1).indices == held[:, None]
        metrics[arm] = {f"top{depth}": float(rank[:, :depth].any(dim=1).float().mean()) for depth in (1, 16, 32)}
        metrics[arm]["mean_absolute_column_exposure_error"] = float((probability.sum(dim=0) - 1).abs().mean())
    return measured, {"paired_ocf": summary(measured), "unique_exact_image_pairs": pairs, "full_gallery_identity": metrics,
                      "changed_scored_images_vs_v14": {arm: int((ids[rows] != choices[ACTUAL][rows]).sum())
                                                        for arm, ids in choices.items()}}


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on one Kaggle T4 with official labeled training files.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    args = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128, visual_pairs=14336, query_batch=64)
    root = locate_data()
    catalog = load_catalog(root)
    gallery = load_images(root, device)
    if len(gallery) != 2748:
        raise RuntimeError("Identity balancing requires the known gallery and complete recording banks.")
    masks = gallery >= 128
    signatures, energy = contour_signatures(gallery)
    scorer = official_scorer(root)
    visual, partitions, cohort_report = cohorts(len(gallery), device)
    geometry = gallery_geometry(masks, signatures, energy)
    proxy = similarity_from_signatures(signatures, energy)
    draw = torch.Generator(device=device).manual_seed(args.seed + 641)
    metric, _, visual_pairs = fit_visual_metric(masks, geometry, proxy, visual, scorer, draw, args)
    utility = gallery_utility(metric, geometry)
    del geometry, proxy, metric

    report = {"protocol": {"gallery_size": len(gallery), "same_candidate_budget": SHORTLIST, "outer_glyphs_per_day": OUTER,
                           "temperature_glyphs_per_source_day": TEMP, "visual_supervision_glyphs": len(visual),
                           "common_unique_exact_visual_pairs": visual_pairs, "half_balance_fraction": PARTIAL_BALANCE,
                           "fixed_exact_utility_weight": UTILITY_WEIGHT, "formal_test_accessed": False,
                           "glyph_seed": SEED, "repeat_seed": REPEAT_SEED,
                           "method": "Each full recording is independently shuffled. An independent posterior and its "
                                     "24-iteration Sinkhorn version share one calibrated scalar from three source days. "
                                     "Half balance is the normalized geometric mean. Each utility policy uses the same "
                                     "frozen visual model and incumbent actions. Mice1 uses its literal v14 v11-nominated "
                                     "seven actions and residual scoring. Mice2 uses the seven highest raw v14 contour/"
                                     "population-gain scores, retaining its exact full-gallery v14 leader. Prior mice2 "
                                     "exact-utility experiments allowed the whole gallery as the output action set. "
                                     "Every target trial is withheld from neural fitting, and raw order and identities "
                                     "are unavailable to the posterior and selection; identities enter only evaluation."},
              "cohorts": cohort_report, "mice": {}}

    def save(progress):
        report.update(stage=progress, seconds=time.monotonic() - started)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage(progress, started)

    save("shared_visual_fit")
    for index, mouse in enumerate(TWO_MICE):
        held, temperature = partitions[mouse]
        recordings = grouped_blocks(root, catalog, mouse, device, len(gallery))
        if len(recordings) != 4 or any(counts.shape[1] != 4 for counts in recordings.values()):
            raise RuntimeError(f"Expected four complete four-repeat labeled recordings for {mouse}.")
        inner_support, calibration, support, queries = disjoint_trial_split(recordings, mouse)
        del recordings
        rng = np.random.default_rng(SEED + 501 + index)
        truth = {day: torch.as_tensor(rng.permutation(len(query)), device=device) for day, query in queries.items()}
        queries = {day: query[truth[day]] for day, query in queries.items()}
        inner, _ = fit_models(inner_support, gallery, signatures, energy, mouse, args)
        panels = {day: v14_panel(inner, trials[temperature], gallery, day, mouse, args).mixed
                  for day, trials in calibration.items()}
        del inner, inner_support, calibration
        model, fit_report = fit_models(support, gallery, signatures, energy, mouse, args)
        del support
        report["mice"][mouse] = {"v14_fit": fit_report, "old_trial_overlap": 0, "sessions": {}}
        collected = {}
        for day, query in queries.items():
            source_days = [peer for peer in panels if peer != day]
            calibrated = fit_temperature([(panels[peer], temperature) for peer in source_days])
            predictions = v14_panel(model, query, gallery, day, mouse, args)
            exact, metrics = evaluate(predictions, calibrated["temperature"], truth[day], held, utility, masks, scorer)
            for arm, values in exact.items():
                collected.setdefault(arm, []).append(values)
            report["mice"][mouse]["sessions"][day] = {"source_days_for_temperature": source_days,
                                                      "temperature": calibrated, **metrics}
            report["mice"][mouse]["summary_ocf"] = summary({arm: np.stack(values) for arm, values in collected.items()})
            save(f"{mouse}_{day}_scored")
        del model, queries, panels, predictions
    save("complete")


if __name__ == "__main__":
    main()
