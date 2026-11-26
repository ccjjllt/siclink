"""Independently score expected-OCF nominations with frozen v12 selection."""

from dataclasses import dataclass
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
from kaggle_task3_rankband_ocf_recheck import PRIOR_GLYPH_SEED, PRIOR_REPEAT_SEED, disjoint_trial_split, gather_repeats, glyph_cohorts
from kaggle_task3_rankband_ocf_validation import fit_visual_metric, neural_rows, stage, summarize
from kaggle_task3_repeat_posterior import fit_models
from kaggle_task3_utility_coverage import (BASE_WIDTH, CONTROL, EXTRA, MEAN, MIXED_DEPTH, RESIDUAL, RESIDUAL_DEPTH,
                                           current_trials, new_cohorts, residual_nominees, select)


SEED = 18731
REPEAT_SEED = 18739
OUTER = 320
TEMPERATURE = 192
ORACLE = 48
KEY_CONTRAST = "expected_utility_vs_v12_same_residual_bank"
OUTPUT = Path("/kaggle/working/task3_utility_nomination_recheck.json")


@dataclass(frozen=True)
class Cohorts:
    visual_training: torch.Tensor
    temperature: torch.Tensor
    outer: torch.Tensor
    previous_temperature: tuple[torch.Tensor, torch.Tensor]


def independent_cohorts(size, device):
    original = torch.as_tensor(np.random.default_rng(PRIOR_GLYPH_SEED).permutation(size), device=device)
    _, first_temp, first_listwise, first_outer, visual = original.split((640, 192, 256, 192, size - 1280))
    second_visual, second_temp, second_outer = glyph_cohorts(size, device)
    third_visual, _, third_outer = new_cohorts(size, device)
    if not torch.equal(visual, second_visual) or not torch.equal(visual, third_visual):
        raise RuntimeError("All experiments must share the original exact image-pair fit glyphs.")
    excluded = torch.cat((visual, first_listwise, first_outer, second_outer, third_outer))
    available = torch.cat((first_temp, second_temp))
    if len(available) != 384 or torch.isin(available, excluded).any():
        raise RuntimeError("The expected 384 image-fit and OCF-evaluation-free glyphs were not reproduced.")

    rng = np.random.default_rng(SEED)
    order = torch.as_tensor(rng.permutation(len(available))[:OUTER], device=device)
    temperature_order = torch.as_tensor(rng.permutation(len(first_listwise))[:TEMPERATURE], device=device)
    outer, temperature = available[order], first_listwise[temperature_order]
    if torch.isin(outer, temperature).any() or torch.isin(temperature, visual).any():
        raise RuntimeError("Separate the visual fit, temperature calibration and outer OCF cohorts.")
    return Cohorts(visual, temperature, outer, (first_temp, second_temp))


def fourth_repeat_split(recordings, cohorts):
    ids = {name: torch.arange(4, device=counts.device).reshape(1, 4, 1).expand(len(counts), -1, -1)
           for name, counts in recordings.items()}
    _, first_inner, _, first_outer = nested_repeats(ids, PRIOR_REPEAT_SEED)
    _, second_inner, _, second_outer = disjoint_trial_split(ids)
    _, _, _, third_outer = current_trials(ids)
    inner_fit, inner_held, outer_fit, outer_held, past_temperature = {}, {}, {}, {}, {}
    previously_in_first = torch.isin(cohorts.outer, cohorts.previous_temperature[0])
    for day, (name, counts) in enumerate(recordings.items()):
        earlier = torch.cat((first_outer[name], second_outer[name], third_outer[name]), dim=1)[..., 0]
        by_repeat = ids[name][..., 0]
        used = (by_repeat[:, :, None] == earlier[:, None]).any(dim=2)
        if not (used.sum(dim=1) == 3).all():
            raise RuntimeError(f"The three previous outer repeats must be distinct in {name}.")
        outer_id = by_repeat[~used].reshape(len(counts), 1)
        if (outer_id == earlier).any():
            raise RuntimeError(f"A new outer repeat was previously held for OCF scoring in {name}.")
        outer_training = by_repeat[used].reshape(len(counts), 3)
        old_first_two = torch.stack((earlier[:, 0], earlier[:, 1]), dim=1)[..., None]
        _, inner_id = split_repeats(old_first_two, REPEAT_SEED + day)
        inner_id = inner_id[..., 0]
        if (inner_id == outer_id).any() or (inner_id[cohorts.temperature] == first_inner[name][cohorts.temperature, :, 0]).any():
            raise RuntimeError(f"The temperature calibration trial must be separate from the old labeled trial and current outer in {name}.")
        inner_keep = (by_repeat != inner_id) & (by_repeat != outer_id)
        inner_training = by_repeat[inner_keep].reshape(len(counts), 2)
        inner_fit[name] = gather_repeats(counts, inner_training)
        inner_held[name] = gather_repeats(counts, inner_id)
        outer_fit[name] = gather_repeats(counts, outer_training)
        outer_held[name] = gather_repeats(counts, outer_id)

        first, second = first_inner[name][cohorts.outer, 0, 0], second_inner[name][cohorts.outer, 0, 0]
        old_temp_id = torch.where(previously_in_first, first, second)
        past_temperature[name] = int((outer_id[cohorts.outer, 0] == old_temp_id).sum())
    return inner_fit, inner_held, outer_fit, outer_held, past_temperature


@torch.no_grad()
def measure(rows, labels, order, probability, utility, masks, scorer):
    expected = probability @ utility.T
    _, by_v12, by_utility = residual_nominees(rows, expected)
    pools = {CONTROL: rows.action, RESIDUAL: torch.cat((rows.action, by_v12), dim=1),
             MEAN: torch.cat((rows.action, by_utility), dim=1)}
    for name, pool in pools.items():
        sorted_ids = pool.sort(dim=1).values
        if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any() or pool.shape[1] != BASE_WIDTH + (0 if name == CONTROL else EXTRA):
            raise RuntimeError(f"{name} must have its fixed number of distinct candidates.")
    decisions = {name: select(rows.mixed, pool) for name, pool in pools.items()}
    reorder = order.argsort()
    reference = labels[reorder]
    chosen = torch.stack([value[reorder] for value in decisions.values()])
    oracle_bank = torch.cat([pool[reorder[:ORACLE]] for pool in pools.values()], dim=1)
    candidates = torch.cat((chosen.flatten(), oracle_bank.flatten()))
    targets = torch.cat((reference.expand_as(chosen).flatten(), reference[:ORACLE, None].expand_as(oracle_bank).flatten()))
    exact, n = unique_pair_ocf(candidates, targets, masks, scorer, masks.device, 384)
    selected, oracle = exact.split((chosen.numel(), oracle_bank.numel()))
    selected = dict(zip(decisions, selected.reshape_as(chosen).cpu().numpy()))
    oracle = oracle.reshape_as(oracle_bank).split([pool.shape[1] for pool in pools.values()], dim=1)
    ceilings = {name: block.amax(dim=1).cpu().numpy() for name, block in zip(pools, oracle)}
    diagnostics = {"exact_pairs": n,
                   "actual_glyph_recall": {name: float((pool == labels[:, None]).any(dim=1).float().mean()) for name, pool in pools.items()},
                   "changed_vs_v12": {name: int((chosen != decisions[CONTROL]).sum()) for name, chosen in decisions.items()},
                   "changed_expected_utility_vs_same_bank": int((decisions[MEAN] != decisions[RESIDUAL]).sum())}
    return selected, ceilings, diagnostics


def compare(data):
    return {**summarize(data), KEY_CONTRAST: paired(data[MEAN], data[RESIDUAL])}


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use one Kaggle T4 with the competition input mounted.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    args = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128, visual_pairs=14336, query_batch=64)
    root = locate_data()
    gallery = load_images(root, device)
    masks = gallery >= 128
    signatures, energy = contour_signatures(gallery)
    cohorts = independent_cohorts(len(gallery), device)
    scorer = official_scorer(root)

    geometry = gallery_geometry(masks, signatures, energy)
    proxy = similarity_from_signatures(signatures, energy)
    draw = torch.Generator(device=device).manual_seed(args.seed + 641)
    metric, strata, unique = fit_visual_metric(masks, geometry, proxy, cohorts.visual_training, scorer, draw, args)
    utility = gallery_utility(metric, geometry)
    del metric, geometry, proxy
    stage("original_visual_fit", started, exact_pairs=unique, sampled_pairs=strata)

    recordings = grouped_blocks(root, load_catalog(root), "mice1", device, len(gallery))
    inner_fit, inner_held, outer_fit, outer_held, past_temperature = fourth_repeat_split(recordings, cohorts)
    del recordings
    inner, inner_report = fit_models(inner_fit, gallery, signatures, energy, "mice1", args)
    del inner_fit
    panels = {name: neural_rows(inner, held[cohorts.temperature], gallery, name, args) for name, held in inner_held.items()}
    del inner, inner_held
    outer, outer_report = fit_models(outer_fit, gallery, signatures, energy, "mice1", args)
    del outer_fit, signatures, energy
    stage("v12_refit", started, previous_outer_repeat_overlap=0, prior_temperature_repeat_overlap=past_temperature)

    report = {"protocol": {"mouse": "mice1", "glyph_seed": SEED, "repeat_seed": REPEAT_SEED, "gallery_size": len(gallery),
                           "ocf_glyphs_per_session": len(cohorts.outer), "oracle_glyphs_per_session": ORACLE,
                           "temperature_glyphs_per_source_session": len(cohorts.temperature), "visual_fit_glyphs": len(cohorts.visual_training),
                           "residual_screen": MIXED_DEPTH, "residual_shared_bank": RESIDUAL_DEPTH,
                           "base_width": BASE_WIDTH, "expansion_width": BASE_WIDTH + EXTRA,
                           "prior_evaluation_glyph_overlap": 0, "visual_training_overlap_with_current_evaluation_or_temperature": 0,
                           "prior_outer_trial_overlap": 0, "actual_test_accessed": False,
                           "history": "The 320 outer glyphs previously served only as posterior-calibration glyphs for the three related runs. "
                                      "They were not used for visual utility fitting, OCF fitting, or OCF scoring. Each outer trial is the "
                                      "fourth held index, absent from all three previous outer allocations for that glyph and session. The "
                                      "new independent temperature cohort previously supplied listwise learning on different trials.",
                           "prediction": "Retain exact original v12 top7 in every pool. Both 16-candidate rules use the same residual bank, "
                                         "differ only in the nine added nominations, and choose output with the same unmodified v12 logits."},
              "visual_exact_pairs": unique, "v12_fits": {"inner": inner_report, "outer": outer_report}, "sessions": {}}
    scores, oracle = {}, {}
    rng = np.random.default_rng(SEED + 1)
    for name, held in outer_held.items():
        days = [day for day in panels if day != name]
        temperature = fit_temperature([(panels[day].mixed, cohorts.temperature) for day in days])
        order = torch.as_tensor(rng.permutation(len(cohorts.outer)), device=device)
        labels = cohorts.outer[order]
        rows = neural_rows(outer, held[labels], gallery, name, args)
        posterior = glyph_posterior(rows.mixed, temperature["temperature"])
        selected, ceiling, diagnostics = measure(rows, labels, order, posterior, utility, masks, scorer)
        for accumulator, block in ((scores, selected), (oracle, ceiling)):
            for arm, values in block.items():
                accumulator.setdefault(arm, []).append(values)
        report["sessions"][name] = {"ocf": compare(selected), "oracle": compare(ceiling), "temperature": temperature,
                                    "calibration_source_days": days, "prior_temperature_trial_overlap": past_temperature[name], **diagnostics}
        report["seconds"] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage("recording_scored", started, session=name, ocf=report["sessions"][name]["ocf"],
              oracle=report["sessions"][name]["oracle"])

    report["ocf"] = compare({name: np.stack(values) for name, values in scores.items()})
    report["oracle"] = compare({name: np.stack(values) for name, values in oracle.items()})
    report["seconds"] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    stage("complete", started, ocf=report["ocf"], oracle=report["oracle"])


if __name__ == "__main__":
    main()
