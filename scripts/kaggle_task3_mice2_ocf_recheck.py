"""Independently score the frozen mice2 v14 plus 20% exact-utility Task 3 policy."""

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
from kaggle_task3_measured_metric_validation import paired
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf
from kaggle_task3_rankband_ocf_recheck import gather_repeats
from kaggle_task3_rankband_ocf_validation import fit_visual_metric, stage
from kaggle_task3_repeat_posterior import fit_models
from kaggle_task3_shape_capacity_validation import ACTUAL, REPEAT_SEED as PILOT_REPEAT_SEED, cohorts, decide, predict_panel


MOUSE = "mice2"
SEED = 29417
REPEAT_SEED = 29423
OUTER = 384
TEMP = 192
FIXED_TEMPERATURE = 1.0
EXACT = "exact_calibrated"
PROXY = "contour_calibrated"
FIXED = "exact_fixed_temperature"
OUTPUT = Path("/kaggle/working/task3_mice2_ocf_recheck.json")


def new_cohorts(size, device):
    visual, pilot_temperature, pilot_outer = cohorts(size, device)
    exclusion_outer = torch.as_tensor(np.random.default_rng(1531).permutation(size)[2432:2560], device=device)
    older_utility_outer = torch.as_tensor(np.random.default_rng(211 + 141).permutation(size)[192:288], device=device)
    removed = {"original_exact_utility_fit": visual, "shape_capacity_outer": pilot_outer,
               "shape_capacity_temperature": pilot_temperature, "soft_exclusion_outer": exclusion_outer,
               "older_posterior_utility_outer": older_utility_outer}
    union = torch.unique(torch.cat(list(removed.values())))
    candidates = torch.arange(size, device=device)
    candidates = candidates[~torch.isin(candidates, union)]
    if len(candidates) < OUTER + TEMP:
        raise RuntimeError("Fewer than 384 new outer and 192 calibration glyphs are available.")
    order = torch.as_tensor(np.random.default_rng(SEED).permutation(len(candidates))[:OUTER + TEMP], device=device)
    scored, calibrated = candidates[order].split((OUTER, TEMP))
    overlaps = {name: int(torch.isin(scored, excluded).sum()) for name, excluded in removed.items()}
    if any(overlaps.values()) or torch.isin(scored, calibrated).any() or torch.isin(calibrated, visual).any():
        raise RuntimeError("Keep the new outer, temperature, and earlier utility-fitting cohorts separate.")
    return visual, calibrated, scored, {"eligible_union_complement": len(candidates), "new_outer_overlap": overlaps,
                                       "excluded_cohort_sizes": {name: len(values) for name, values in removed.items()}}


def new_trial_split(recordings):
    indices = {name: torch.arange(4, device=counts.device).reshape(1, 4, 1).expand(len(counts), -1, -1)
               for name, counts in recordings.items()}
    previous_fit, previous_inner, _, previous_outer = nested_repeats(indices, PILOT_REPEAT_SEED)
    inner_fit, inner_held, outer_fit, outer_held = {}, {}, {}, {}
    for d, (name, counts) in enumerate(recordings.items()):
        local, query = (part[..., 0] for part in split_repeats(previous_fit[name], REPEAT_SEED + d))
        previous = torch.cat((previous_inner[name], previous_outer[name]), dim=1)[..., 0]
        held = torch.cat((local, query), dim=1)
        if (held[:, :, None] == previous[:, None, :]).any() or (local == query).any():
            raise RuntimeError(f"The previous and new held trial indices overlap in {name}.")
        all_ids = indices[name][..., 0]
        remain = all_ids[all_ids != query].reshape(len(counts), 3)
        inner_fit[name] = gather_repeats(counts, previous.sort(dim=1).values)
        inner_held[name] = gather_repeats(counts, local)
        outer_fit[name] = gather_repeats(counts, remain)
        outer_held[name] = gather_repeats(counts, query)
    return inner_fit, inner_held, outer_fit, outer_held


def compare(values):
    baseline = {name: paired(score, values[ACTUAL]) for name, score in values.items()}
    return {**baseline, "exact_vs_contour_same_temperature": paired(values[EXACT], values[PROXY]),
            "exact_calibrated_vs_fixed_temperature": paired(values[EXACT], values[FIXED])}


@torch.no_grad()
def measure(panel, labels, order, temperature, table, proxy, masks, scorer):
    if panel.shortlist is not None:
        raise RuntimeError("The mice2 v14 policy must retain the full official known-glyph gallery.")
    rows = torch.arange(len(labels), device=labels.device)
    probability = glyph_posterior(panel.logits, temperature)
    fixed = glyph_posterior(panel.logits, FIXED_TEMPERATURE)
    choices = decide(panel, rows, {EXACT: probability, FIXED: fixed}, table)
    choices[PROXY] = decide(panel, rows, {PROXY: probability}, proxy)[PROXY]

    reorder = order.argsort()
    hypotheses = torch.stack([values[reorder] for values in choices.values()])
    reference = labels[reorder]
    scores, unique = unique_pair_ocf(hypotheses, reference.expand_as(hypotheses), masks, scorer, masks.device, 384)
    records = dict(zip(choices, scores.cpu().numpy()))
    reports = {"unique_paired_official_scores": unique, "mean_calibrated_top32_mass": float(probability.topk(32, dim=1).values.sum(dim=1).mean()),
               "mean_fixed_top32_mass": float(fixed.topk(32, dim=1).values.sum(dim=1).mean()),
               "changes_vs_v14": {name: int((guess != choices[ACTUAL]).sum()) for name, guess in choices.items()}}
    return records, reports


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use one private Kaggle T4 with the competition training input mounted.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    args = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128, visual_pairs=14336, query_batch=64)
    root = locate_data()
    gallery = load_images(root, device)
    masks = gallery >= 128
    signatures, energy = contour_signatures(gallery)
    visual, temp_labels, outer_labels, cohorts_report = new_cohorts(len(gallery), device)
    scorer = official_scorer(root)

    geometry = gallery_geometry(masks, signatures, energy)
    proxy = similarity_from_signatures(signatures, energy)
    draw = torch.Generator(device=device).manual_seed(args.seed + 641)
    metric, strata, unique = fit_visual_metric(masks, geometry, proxy, visual, scorer, draw, args)
    table = gallery_utility(metric, geometry)
    proxy *= 100
    del geometry, metric
    stage("original_image_utility", started, exact_unique_pairs=unique, strata=strata)

    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(gallery))
    inner_fit, inner_held, outer_fit, outer_held = new_trial_split(recordings)
    del recordings
    inner, inner_report = fit_models(inner_fit, gallery, signatures, energy, MOUSE, args)
    del inner_fit
    panels = {name: predict_panel(inner, counts[temp_labels], gallery, name, MOUSE, args).logits
              for name, counts in inner_held.items()}
    del inner, inner_held
    outer, outer_report = fit_models(outer_fit, gallery, signatures, energy, MOUSE, args)
    del outer_fit, signatures, energy
    stage("v14_models_fit", started, overlap_of_current_and_pilot_held_trials=0)

    report = {"protocol": {"mouse": MOUSE, "glyph_seed": SEED, "repeat_seed": REPEAT_SEED, "new_outer_glyphs_per_day": OUTER,
                           "temperature_glyphs_per_source_day": TEMP, "gallery_actions_in_all_arms": len(gallery),
                           "previous_shape_capacity_inner_or_outer_trial_overlap": 0, "formal_test_accessed": False,
                           "fixed_exact_ocf_weight": 0.20, "fixed_control_temperature": FIXED_TEMPERATURE,
                           "method": "The frozen mice2 v14 anchor is the equal contour-ridge and timed population-gain model. "
                                     "Each source cohort uses a different real repeat, and each target's scalar temperature is "
                                     "fit only on the other three source days. Every output is selected from the same entire known gallery. "
                                     "The exact visual utility is replicated from the original 14,336 stratified image-pair draws. "
                                     "Exact, contour proxy, and fixed unit-temperature selector use the identical 20% mixture and centering."},
              "cohorts": cohorts_report, "unique_exact_visual_fit_pairs": unique,
              "v14_fit": {"inner": inner_report, "outer": outer_report}, "sessions": {}}
    rng = np.random.default_rng(SEED + 1)
    per_arm = {}
    for name, counts in outer_held.items():
        sources = [day for day in panels if day != name]
        fitted = fit_temperature([(panels[day], temp_labels) for day in sources])
        order = torch.as_tensor(rng.permutation(len(outer_labels)), device=device)
        labels = outer_labels[order]
        panel = predict_panel(outer, counts[labels], gallery, name, MOUSE, args)
        records, diagnostics = measure(panel, labels, order, fitted["temperature"], table, proxy, masks, scorer)
        for arm, values in records.items():
            per_arm.setdefault(arm, []).append(values)
        report["sessions"][name] = {"source_days_for_temperature": sources, "temperature": fitted,
                                    "paired": compare(records), **diagnostics}
        report["seconds"] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage("scored_recording", started, session=name, paired=report["sessions"][name]["paired"])

    report["pooled_paired_ocf"] = compare({name: np.stack(values) for name, values in per_arm.items()})
    report["seconds"] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    stage("complete", started, paired=report["pooled_paired_ocf"])


if __name__ == "__main__":
    main()
