"""Test equal-budget Task 3 shortlist coverage using marginal expected OCF."""

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


SEED = 16157
REPEAT_SEED = 16161
BASE_WIDTH = 7
EXTRA = 9
MIXED_DEPTH = 128
RESIDUAL_DEPTH = 64
EXACT_ORACLE_ROWS = 80
OUTPUT = Path("/kaggle/working/task3_utility_coverage.json")
CONTROL = "v12_top7"
DEEP = "v11_deep16"
RESIDUAL = "residual_v12_16"
MEAN = "residual_mean_utility16"
COVER = "residual_marginal_utility16"


def new_cohorts(n, device):
    vision, previous_temp, previous_outer = glyph_cohorts(n, device)
    first = torch.as_tensor(np.random.default_rng(PRIOR_GLYPH_SEED).permutation(n), device=device)
    checks, original_temperature, original_listwise, original_outer, _ = first.split((640, 192, 256, 192, n - 1280))
    prior = torch.cat((previous_temp, previous_outer))
    fresh = checks[~torch.isin(checks, prior)]
    order = torch.as_tensor(np.random.default_rng(SEED).permutation(len(fresh)), device=device)
    fresh = fresh[order]
    disallowed = torch.cat((prior, original_temperature, original_listwise, original_outer, vision))
    if len(fresh) != 192 or torch.isin(fresh, disallowed).any() or torch.isin(original_outer, vision).any():
        raise RuntimeError("Expected 192 fresh glyphs from the original image-only diagnostic pool.")
    # The original OCF cohort now supplies independent single-temperature calibration.
    return vision, original_outer, fresh


def current_trials(recordings):
    indices = {name: torch.arange(4, device=counts.device).reshape(1, 4, 1).expand(len(counts), -1, -1)
               for name, counts in recordings.items()}
    _, original_inner, _, original_outer = nested_repeats(indices, PRIOR_REPEAT_SEED)
    _, replicate_inner, _, replicate_outer = disjoint_trial_split(indices)
    inner_fit, inner_held, outer_fit, outer_held = {}, {}, {}, {}
    for day, (name, counts) in enumerate(recordings.items()):
        earlier_outer = torch.cat((original_outer[name], replicate_outer[name]), dim=1)[..., 0]
        available = torch.cat((original_inner[name], replicate_inner[name]), dim=1)
        inner_id, outer_id = (part[..., 0] for part in split_repeats(available, REPEAT_SEED + day))
        if (inner_id == outer_id).any() or (outer_id == earlier_outer).any() or (inner_id == earlier_outer).any():
            raise RuntimeError(f"A current held repeat overlaps a previous outer repeat in {name}.")
        all_ids = indices[name][..., 0]
        outer_training = all_ids[all_ids != outer_id].reshape(len(counts), 3)
        inner_fit[name] = gather_repeats(counts, earlier_outer.sort(dim=1).values)
        inner_held[name] = gather_repeats(counts, inner_id)
        outer_fit[name] = gather_repeats(counts, outer_training)
        outer_held[name] = gather_repeats(counts, outer_id)
    return inner_fit, inner_held, outer_fit, outer_held


def best_outside(scores, slots, amount):
    return scores.clone().scatter_(1, slots, -torch.inf).topk(amount, dim=1).indices


@torch.no_grad()
def marginal_candidates(start, possible, probability, utility):
    """Choose actions that add coverage to the best utility already represented."""
    proposed, improvements = [], []
    for original, bank, p in zip(start.split(32), possible.split(32), probability.split(32)):
        best = utility[original].amax(dim=1)
        reference_before = (p * best).sum(dim=1)
        pair = utility[bank]
        used = torch.zeros_like(bank, dtype=torch.bool)
        query = torch.arange(len(bank), device=bank.device)
        selections = []
        for _ in range(EXTRA):
            lift = (pair - best[:, None]).clamp_min_(0)
            gain = torch.bmm(lift, p[:, :, None]).squeeze(-1).masked_fill_(used, -torch.inf)
            slot = gain.argmax(dim=1)
            used[query, slot] = True
            selections.append(bank[query, slot])
            best = torch.maximum(best, pair[query, slot])
        proposed.append(torch.stack(selections, dim=1))
        improvements.append((p * best).sum(dim=1) - reference_before)
    return torch.cat(proposed), torch.cat(improvements)


@torch.no_grad()
def residual_nominees(rows, expected):
    """Return both nine-action rankings from the same residual bank."""
    eligible = best_outside(rows.mixed, rows.action, MIXED_DEPTH)
    coefficient = (rows.base * rows.mixed).mean(dim=1, keepdim=True)
    coefficient /= rows.base.square().mean(dim=1, keepdim=True).clamp_min(1e-6)
    innovation = rows.mixed - coefficient * rows.base
    bank_keep = innovation.gather(1, eligible).topk(RESIDUAL_DEPTH, dim=1).indices
    bank = eligible.gather(1, bank_keep)
    by_mixed = bank.gather(1, rows.mixed.gather(1, bank).topk(EXTRA, dim=1).indices)
    by_mean = bank.gather(1, expected.gather(1, bank).topk(EXTRA, dim=1).indices)
    return bank, by_mixed, by_mean


@torch.no_grad()
def actions(rows, temperature, utility):
    p = glyph_posterior(rows.mixed, temperature)
    expected = p @ utility.T
    bank, by_mixed, by_mean = residual_nominees(rows, expected)
    by_marginal, model_gain = marginal_candidates(rows.action, bank, p, utility)
    extras = {DEEP: best_outside(rows.base, rows.action, EXTRA), RESIDUAL: by_mixed, MEAN: by_mean, COVER: by_marginal}
    pools = {CONTROL: rows.action, **{name: torch.cat((rows.action, extra), dim=1) for name, extra in extras.items()}}
    for name, pool in pools.items():
        ordered = pool.sort(dim=1).values
        width = BASE_WIDTH if name == CONTROL else BASE_WIDTH + EXTRA
        if pool.shape[1] != width or (ordered[:, 1:] == ordered[:, :-1]).any():
            raise RuntimeError(f"Incorrect unique candidate budget for {name}.")
    return pools, expected, {"estimated_marginal_gain_relative_to_original7": float(model_gain.mean()),
                             "mean_posterior_mass_top32": float(p.topk(32, dim=1).values.sum(dim=1).mean())}


def normalize_from_base(values, original, floor):
    existing = values.gather(1, original)
    return (values - existing.mean(dim=1, keepdim=True)) / existing.std(dim=1, keepdim=True, unbiased=False).clamp_min(floor)


def select(values, pool):
    return pool.gather(1, values.gather(1, pool).argmax(dim=1, keepdim=True))[:, 0]


@torch.no_grad()
def measure(rows, expected, pools, labels, permutation, masks, scorer):
    anchored = 0.80 * normalize_from_base(rows.mixed, rows.action, 0.30)
    final_score = anchored + 0.20 * normalize_from_base(expected, rows.action, 0.50)
    choices = {name: select(rows.mixed, pool) for name, pool in pools.items()}
    choices.update({name + "_fixed_twenty_pct": select(final_score, pool) for name, pool in pools.items()})
    baseline = choices[CONTROL]
    reorder = permutation.argsort()
    predicted = torch.stack([value[reorder] for value in choices.values()])
    refs = labels[reorder]
    picked_oracles = {name: pool[reorder[:EXACT_ORACLE_ROWS]] for name, pool in pools.items()}
    oracle_bank = torch.cat(tuple(picked_oracles.values()), dim=1)
    full_candidates = torch.cat((predicted.flatten(), oracle_bank.flatten()))
    full_references = torch.cat((refs.expand_as(predicted).flatten(), refs[:EXACT_ORACLE_ROWS, None].expand_as(oracle_bank).flatten()))
    exact, unique = unique_pair_ocf(full_candidates, full_references, masks, scorer, masks.device, 384)
    top1, oracle = exact.split((predicted.numel(), oracle_bank.numel()))
    top1 = dict(zip(choices, top1.reshape_as(predicted).cpu().numpy()))
    split = oracle.reshape_as(oracle_bank).split([pool.shape[1] for pool in picked_oracles.values()], dim=1)
    best = {name: scores.amax(dim=1).cpu().numpy() for name, scores in zip(pools, split)}
    metadata = {"unique_official_pairs": unique,
                "choice_changes_vs_v12_top7": {name: int((value != baseline).sum()) for name, value in choices.items()},
                "actual_glyph_recall": {name: float((pool == labels[:, None]).any(dim=1).float().mean()) for name, pool in pools.items()},
                "novel_choice_count": {name: int((rows.action != value[:, None]).all(dim=1).sum()) for name, value in choices.items()}}
    return top1, best, metadata


def compare(rows, singles):
    scored = rows if singles else {name: np.stack(series) for name, series in rows.items()}
    result = summarize(scored)
    result["marginal_vs_same_bank_v12"] = paired(scored[COVER], scored[RESIDUAL])
    result["marginal_vs_same_bank_mean"] = paired(scored[COVER], scored[MEAN])
    result["marginal_vs_v11_depth"] = paired(scored[COVER], scored[DEEP])
    suffix = "_fixed_twenty_pct"
    if COVER + suffix in scored:
        result["marginal_vs_same_bank_with_common_twenty_pct"] = paired(scored[COVER + suffix], scored[RESIDUAL + suffix])
    return result


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle T4 and the licensed competition input.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    args = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128, visual_pairs=14336, query_batch=64)
    root = locate_data()
    gallery = load_images(root, device)
    masks = gallery >= 128
    signature, energy = contour_signatures(gallery)
    score = official_scorer(root)
    visual_fit, temp_labels, outer_labels = new_cohorts(len(gallery), device)

    geometry = gallery_geometry(masks, signature, energy)
    proxy = similarity_from_signatures(signature, energy)
    draw = torch.Generator(device=device).manual_seed(args.seed + 641)
    metric, counts, unique = fit_visual_metric(masks, geometry, proxy, visual_fit, score, draw, args)
    utility = gallery_utility(metric, geometry)
    del geometry, metric, proxy
    stage("original_visual_fit", started, unique_exact_pairs=unique, sampled_pairs=counts)

    recordings = grouped_blocks(root, load_catalog(root), "mice1", device, len(gallery))
    inner_fit, inner_held, outer_fit, outer_held = current_trials(recordings)
    del recordings
    inner, inner_report = fit_models(inner_fit, gallery, signature, energy, "mice1", args)
    del inner_fit
    calibration = {name: neural_rows(inner, trials[temp_labels], gallery, name, args) for name, trials in inner_held.items()}
    del inner, inner_held
    outer, outer_report = fit_models(outer_fit, gallery, signature, energy, "mice1", args)
    del outer_fit, signature, energy
    stage("v12_refit", started, previous_outer_repeat_overlap=0)

    report = {"protocol": {"mouse": "mice1", "gallery_size": len(gallery), "glyph_seed": SEED, "repeat_seed": REPEAT_SEED,
                           "scored_glyphs_per_day": len(outer_labels), "oracle_glyphs_per_day": EXACT_ORACLE_ROWS,
                           "core_width": BASE_WIDTH, "extra_slots": EXTRA, "residual_atlas_prepool": RESIDUAL_DEPTH,
                           "screening_v12_depth_outside_core": MIXED_DEPTH, "calibration_glyphs_per_source_day": len(temp_labels),
                           "fixed_secondary_expected_utility_weight": 0.20, "held_glyph_overlap_vs_prior_rankband_scoring_or_calibration": 0,
                           "prior_outer_repeat_overlap_per_glyph_and_day": 0, "actual_test_accessed": False,
                           "description": "All arms retain the exact original v12 seven. The residual bank removes the linear v11 "
                                          "component of v12 logits and screens novelty among the same 128 v12 candidates. The "
                                          "marginal rule greedily maximizes expected best official-fit utility for the entire pool "
                                          "under one temperature-calibrated v12 posterior. Final selection uses identical original "
                                          "v12 logits across pools; the fixed secondary utility score is normalized on the seven core "
                                          "candidates for all arms. Exact candidate oracles see labeled reference images only after choices.",
                           "cohorts": "Fresh outer glyphs previously appeared only in a pure visual diagnostic. Score repeats are "
                                      "disjoint from both earlier outer repeats; calibration reuses the first experiment's original "
                                      "outer glyphs on different measured repeats, disjoint from current outer glyphs and the visual fit."},
              "visual_fit": {"fit_glyphs": len(visual_fit), "sample_pairs": args.visual_pairs, "unique_exact_pairs": unique},
              "v12_fits": {"inner": inner_report, "outer": outer_report}, "sessions": {}}
    observed, oracles = {}, {}
    rng = np.random.default_rng(SEED + 1)

    for name, held in outer_held.items():
        sources = [session for session in calibration if session != name]
        fitted = fit_temperature([(calibration[session].mixed, temp_labels) for session in sources])
        permutation = torch.as_tensor(rng.permutation(len(outer_labels)), device=device)
        labels = outer_labels[permutation]
        rows = neural_rows(outer, held[labels], gallery, name, args)
        pools, expected, coverage = actions(rows, fitted["temperature"], utility)
        scores, oracle, details = measure(rows, expected, pools, labels, permutation, masks, score)
        for accumulator, block in ((observed, scores), (oracles, oracle)):
            for arm, values in block.items():
                accumulator.setdefault(arm, []).append(values)
        report["sessions"][name] = {"temperature": fitted, "calibration_source_days": sources, **coverage, **details,
                                    "chosen_ocf": compare(scores, True), "exact_oracle": compare(oracle, True)}
        report["seconds"] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage("scored_recording", started, session=name, chosen=report["sessions"][name]["chosen_ocf"],
              oracle=report["sessions"][name]["exact_oracle"])

    report["chosen_ocf"] = compare(observed, False)
    report["exact_oracle"] = compare(oracles, False)
    report["seconds"] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    stage("complete", started, chosen=report["chosen_ocf"], oracle=report["exact_oracle"])


if __name__ == "__main__":
    main()
