"""Independently recheck the fixed 20% rank-band expected OCF Task 3 policy."""

from pathlib import Path
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_exact_utility import gallery_geometry, gallery_utility
from glyph_mbr import contour_signatures, fit_temperature, similarity_from_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_measured_metric_validation import paired
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf
from kaggle_task3_rankband_ocf_validation import (decision_features, fit_bands, fit_visual_metric, fixed_utility_score,
                                                neural_rows, stage, summarize)
from kaggle_task3_repeat_posterior import fit_models


PRIOR_GLYPH_SEED = 9127
PRIOR_REPEAT_SEED = 9121
GLYPH_SEED = 14627
REPEAT_SEED = 14621
OUTPUT = Path("/kaggle/working/task3_rankband_ocf_recheck.json")


def glyph_cohorts(n, device):
    prior = torch.as_tensor(np.random.default_rng(PRIOR_GLYPH_SEED).permutation(n), device=device)
    checks, temperature, listwise, old_outer, visual_fit = prior.split((640, 192, 256, 192, n - 1280))
    new_order = torch.as_tensor(np.random.default_rng(GLYPH_SEED).permutation(len(checks)), device=device)
    new_outer, new_temperature = checks[new_order[:256]], checks[new_order[256:448]]
    excluded = torch.cat((temperature, listwise, old_outer, visual_fit))
    if torch.isin(torch.cat((new_outer, new_temperature)), excluded).any() or torch.isin(new_outer, new_temperature).any():
        raise RuntimeError("The replication cohorts overlap a prior fit, prior score, or current calibration cohort.")
    return visual_fit, new_temperature, new_outer


def gather_repeats(counts, indices):
    return counts.gather(1, indices[..., None].expand(-1, -1, counts.shape[-1]))


def disjoint_trial_split(recordings):
    # Recover original sample selection by running the shared protocol on indexed repeats.
    trial_indices = {name: torch.arange(4, device=counts.device).reshape(1, 4, 1).expand(len(counts), -1, -1)
                     for name, counts in recordings.items()}
    previous_train, previous_inner, _, previous_outer = nested_repeats(trial_indices, PRIOR_REPEAT_SEED)
    inner_fit, inner_held, outer_fit, outer_held = {}, {}, {}, {}
    for index, (name, counts) in enumerate(recordings.items()):
        if counts.shape[1] != 4:
            raise ValueError(f"Expected four real repeats in {name}.")
        current_inner, current_outer = split_repeats(previous_train[name], REPEAT_SEED + index)
        current_inner, current_outer = current_inner[..., 0], current_outer[..., 0]
        current_evidence = torch.cat((current_inner, current_outer), dim=1)
        previous_evidence = torch.cat((previous_inner[name], previous_outer[name]), dim=1)[..., 0]
        overlap = (current_evidence[:, :, None] == previous_evidence[:, None]).any()
        if overlap or (current_inner == current_outer).any():
            raise RuntimeError(f"A current inner or outer trial overlaps another reserved trial in {name}.")
        axis = trial_indices[name][..., 0]
        three_legal = axis[axis != current_outer].reshape(len(counts), 3)
        two_legal = previous_evidence.sort(dim=1).values
        inner_fit[name] = gather_repeats(counts, two_legal)
        inner_held[name] = gather_repeats(counts, current_inner)
        outer_fit[name] = gather_repeats(counts, three_legal)
        outer_held[name] = gather_repeats(counts, current_outer)
    return inner_fit, inner_held, outer_fit, outer_held


@torch.no_grad()
def score_policies(rows, features, labels, reorder, masks, scorer):
    raw = rows.mixed.gather(1, rows.action)
    choices = {"v12_top7": raw.argmax(dim=1),
               "exact_single_temperature": fixed_utility_score(features.anchor, features.unsplit_exact).argmax(dim=1),
               "exact_band_calibration": fixed_utility_score(features.anchor, features.calibrated_exact).argmax(dim=1)}
    picked = torch.stack([rows.action.gather(1, slot[:, None])[:, 0] for slot in choices.values()])
    values, pairs = unique_pair_ocf(picked, labels.expand_as(picked), masks, scorer, masks.device, 384)
    scores = dict(zip(choices, values[:, reorder].cpu().numpy()))
    switches = {name: int((slot != choices["v12_top7"]).sum()) for name, slot in choices.items()}
    return scores, {"exact_pairs": pairs, "switches_vs_v12": switches, "bands": features.band_report}


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the independent recheck on Kaggle GPU with the competition data mounted.")
    torch.set_num_threads(4)
    args = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128, visual_pairs=14336, query_batch=64)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    gallery = load_images(root, device)
    masks = gallery >= 128
    signatures, energy = contour_signatures(gallery)
    scorer = official_scorer(root)
    vision_fit, temp_labels, outer_labels = glyph_cohorts(len(gallery), device)

    # Recreate the original supervised table from its identical image cohort and pair seeds.
    geometry = gallery_geometry(masks, signatures, energy)
    proxy = similarity_from_signatures(signatures, energy)
    draw = torch.Generator(device=device).manual_seed(args.seed + 641)
    metric, strata, unique = fit_visual_metric(masks, geometry, proxy, vision_fit, scorer, draw, args)
    utility = gallery_utility(metric, geometry)
    proxy *= 100
    del geometry, metric
    stage("original_visual_fit", started, unique_exact_pairs=unique, sampled_pairs=strata)

    recordings = grouped_blocks(root, load_catalog(root), "mice1", device, len(gallery))
    inner_fit, inner_held, outer_fit, outer_held = disjoint_trial_split(recordings)
    del recordings
    inner, inner_report = fit_models(inner_fit, gallery, signatures, energy, "mice1", args)
    del inner_fit
    temp_panels = {name: neural_rows(inner, trials[temp_labels], gallery, name, args) for name, trials in inner_held.items()}
    del inner, inner_held
    outer, outer_report = fit_models(outer_fit, gallery, signatures, energy, "mice1", args)
    del outer_fit, signatures, energy
    stage("v12_refit", started, overlap_with_all_prior_reserved_repeats=0)

    report = {"protocol": {"mouse": "mice1", "gallery_size": len(gallery), "scored_days": len(outer_held),
                           "scored_glyphs_per_day": len(outer_labels), "new_temperature_glyphs_per_source_day": len(temp_labels),
                           "visual_fit_glyphs": len(vision_fit), "previous_visual_check_pool": 640,
                           "new_scoring_glyphs_overlap_prior_visual_fit_or_neural_calibration_or_scoring": 0,
                           "prior_glyph_seed": PRIOR_GLYPH_SEED, "prior_repeat_seed": PRIOR_REPEAT_SEED,
                           "glyph_seed": GLYPH_SEED, "repeat_seed": REPEAT_SEED,
                           "current_inner_or_outer_trial_overlap_prior_inner_or_outer_for_any_glyph_day": 0,
                           "v12_fixed_candidate_count": 7, "fixed_utility_weight": 0.20,
                           "actual_test_records_accessed": False,
                           "separation": "New scoring and temperature glyphs come from the prior image-only diagnostic pool, "
                                         "which supplied no parameters or ranking policy. The original visual fit is reproduced. "
                                         "Each target band's temperature and posterior correction are fit on three other days only."},
              "visual": {"original_supervised_pair_count": args.visual_pairs, "recomputed_unique_exact_pairs": unique,
                         "sampled_pairs_by_stratum": strata},
              "v12_fits": {"inner": inner_report, "outer": outer_report}, "sessions": {}}
    accum = {}
    rng = np.random.default_rng(GLYPH_SEED + 1)

    for name, trials in outer_held.items():
        sources = [previous for previous in temp_panels if previous != name]
        temperature = fit_temperature([(temp_panels[day].mixed, temp_labels) for day in sources])
        t = temperature["temperature"]
        band = fit_bands([temp_panels[day] for day in sources], temp_labels, t)

        permutation = torch.as_tensor(rng.permutation(len(outer_labels)), device=device)
        labels = outer_labels[permutation]
        rows = neural_rows(outer, trials[labels], gallery, name, args)
        features = decision_features(rows, labels, t, band, proxy, utility, args.query_batch)
        values, diagnostics = score_policies(rows, features, labels, permutation.argsort(), masks, scorer)
        for policy, scores in values.items():
            accum.setdefault(policy, []).append(scores)
        report["sessions"][name] = {"temperature": temperature, "calibration_source_days": sources,
                                    "ocf": summarize(values), **diagnostics}
        report["seconds"] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage("outer_ocf", started, session=name, **summarize(values))

    totals = {policy: np.stack(observed) for policy, observed in accum.items()}
    report["outcomes"] = summarize(totals)
    report["rank_bands_vs_single_temperature"] = paired(totals["exact_band_calibration"], totals["exact_single_temperature"])
    report["seconds"] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    stage("complete", started, outcomes=report["outcomes"], rank_bands_vs_single=report["rank_bands_vs_single_temperature"])


if __name__ == "__main__":
    main()
