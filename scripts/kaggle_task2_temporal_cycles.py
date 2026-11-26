"""Test complete-cycle changes between Task 2's v9 and temporal assignments."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_mbr import contour_signatures, contour_utility, fit_temperature, glyph_posterior
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_hierarchical_gallery_validation import validated_logits
from kaggle_task2_temporal_components import (MICE, SCALE, SESSIONS, first_axis, normalize_axis, recordings, scale,
                                               score_temporal, smooth_time, unit_whitening, stable_covariance)
from kaggle_task3_metric_validation import unique_pair_ocf


MOUSE = "mice2"
MIN_EFFECTIVE_CYCLES = 5
MAX_CYCLE_SHARE = 0.50
MIN_CONTROL_SWITCH_ROWS = 20
MIN_EXCHANGEABLE_CYCLES = 5


@dataclass
class DayProposal:
    order: torch.Tensor
    v9_scores: torch.Tensor
    original: torch.Tensor
    temporal: torch.Tensor
    cycles: tuple


def v9_scores(sources, query, images, args):
    torch.manual_seed(args.seed)
    validated = validated_logits(MOUSE, sources, query, images, images.device, args)
    gallery = fit_hierarchical_gallery(sources, view="total")
    empirical = gallery.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    return scale(blend_hierarchical_logits(validated, empirical))


@torch.no_grad()
def temporal_scores(sources, query):
    source = [smooth_time(block.evoked10.sqrt()) for block in sources.values()]
    observed = smooth_time(query.sqrt())
    means, repeat_cov, _, white = unit_whitening(source)
    axis, _ = first_axis(stable_covariance(means), white)
    return scale(score_temporal(source, observed, normalize_axis(axis, repeat_cov)))


def assignment_cycles(first, second):
    """Decompose the difference into independent alternating cycles of rows."""
    count = len(first)
    if len(first.unique()) != count or len(second.unique()) != count:
        raise ValueError("Both proposals must be valid complete assignments.")
    next_row = first.argsort()[second].numpy()
    visited = next_row == np.arange(count)
    result = []
    for start in np.flatnonzero(~visited):
        if visited[start]:
            continue
        here = int(start)
        rows = []
        while not visited[here]:
            visited[here] = True
            rows.append(here)
            here = int(next_row[here])
        if here != start:
            raise RuntimeError("Expected every changed row to lie on a closed cycle.")
        cycle = torch.as_tensor(rows)
        if not torch.equal(first[cycle].sort().values, second[cycle].sort().values):
            raise RuntimeError("Replacing a cycle must preserve its exact candidate inventory.")
        result.append(cycle)
    if sum(map(len, result)) != int((first != second).sum()):
        raise RuntimeError("Cycle decomposition missed changed assignments.")
    return tuple(result)


def structural_report(cycles):
    sizes = np.asarray([len(cycle) for cycle in cycles], dtype=np.int64)
    lengths, counts = np.unique(sizes, return_counts=True)
    total = int(sizes.sum())
    largest = int(sizes.max(initial=0))
    effective = float(total**2 / max(1, np.square(sizes).sum()))
    exchangeable = int(counts[counts >= 2].sum())
    passed = bool(total and largest / total <= MAX_CYCLE_SHARE
                  and effective >= MIN_EFFECTIVE_CYCLES and exchangeable >= MIN_EXCHANGEABLE_CYCLES)
    return {"changed_rows": total, "cycles": len(cycles), "largest_cycle": largest,
            "largest_cycle_share": largest / max(1, total), "length_weighted_effective_cycles": effective,
            "exact_length_exchangeable_cycles": exchangeable, "exact_length_histogram": dict(zip(lengths.tolist(), counts.tolist())),
            "largest_five_lengths": sorted(sizes.tolist(), reverse=True)[:5], "structure_passed": passed}


def discover(day, blocks, images, args):
    source = {name: block for name, block in blocks.items() if name != day}
    replay_seed = np.random.SeedSequence((args.repeat_seed, MICE.index(MOUSE), SESSIONS.index(day)))
    rng = np.random.default_rng(replay_seed)
    order = torch.as_tensor(rng.permutation(len(images)))
    base = v9_scores({name: block.count25 for name, block in source.items()}, blocks[day].count25[order], images, args)
    temporal = base + SCALE * temporal_scores(source, blocks[day].evoked10[order])
    original, proposed = assign_glyphs(base), assign_glyphs(temporal)
    cycles = assignment_cycles(original, proposed)
    return DayProposal(order, base, original, proposed, cycles)


def source_temperature(held, blocks, images, args):
    """Calibrate the v9 identity temperature from other-day v9 forecasts."""
    sources = {name: block.count25 for name, block in blocks.items() if name != held}
    rng = np.random.default_rng(args.repeat_seed + 100 + SESSIONS.index(held))
    calibration = []
    for day, observed in sources.items():
        train = {name: values for name, values in sources.items() if name != day}
        order = torch.as_tensor(rng.permutation(len(images)))
        scores = v9_scores(train, observed[order], images, args)
        calibration.append((scores[:args.calibration_rows].clone(), order[:args.calibration_rows]))
    return fit_temperature(calibration)


@torch.no_grad()
def select(proposal, signature, energy, temperature):
    """Select each cycle using only a fixed pre-trained posterior and contour utility."""
    posterior = glyph_posterior(proposal.v9_scores, temperature, balance=True)
    changed = torch.cat(proposal.cycles)
    margin = energy.new_empty(len(energy))
    for rows in changed.split(128):
        first, second = proposal.original[rows], proposal.temporal[rows]
        under_v9 = contour_utility(signature[first], energy[first], signature, energy)
        under_time = contour_utility(signature[second], energy[second], signature, energy)
        under_v9.scatter_(1, first[:, None], 1.0)
        under_time.scatter_(1, second[:, None], 1.0)
        margin[rows] = ((under_time - under_v9) * posterior[rows]).sum(dim=1)
    improvement = np.asarray([float(margin[rows].sum()) for rows in proposal.cycles])
    chosen = improvement > 0
    random_probability = np.zeros(len(chosen), dtype=np.float64)
    sizes = np.asarray([len(cycle) for cycle in proposal.cycles])
    possible_switch_rows = 0
    for length in np.unique(sizes):
        matches = sizes == length
        amount = int(chosen[matches].sum())
        random_probability[matches] = amount / matches.sum()
        possible_switch_rows += int(length) * min(amount, int(matches.sum()) - amount)
    selected = proposal.original.clone()
    if bool(chosen.any()):
        switched = torch.cat([rows for rows, keep in zip(proposal.cycles, chosen) if keep])
        selected[switched] = proposal.temporal[switched]
    if len(selected.unique()) != len(proposal.original):
        raise RuntimeError("Cycle selection did not retain a complete one-to-one solution.")
    report = {"accepted_cycles": int(chosen.sum()), "accepted_rows": int(sizes[chosen].sum()),
              "positive_expected_cycle_improvement_total": float(improvement[chosen].sum()),
              "exact_length_control_switchable_rows": possible_switch_rows,
              "control_passed": possible_switch_rows >= MIN_CONTROL_SWITCH_ROWS}
    return improvement, chosen, random_probability, selected, report


def score_actions(proposal, selection, images, scorer, args, rng):
    margins, chosen, random_probability, selected, _ = selection
    sizes = np.asarray([len(cycle) for cycle in proposal.cycles])
    counts = np.minimum(sizes, np.maximum(2, np.ceil(args.changed_rows * sizes / sizes.sum()).astype(int)))
    sample = [cycle[torch.from_numpy(rng.choice(len(cycle), size=int(count), replace=False))]
              for cycle, count in zip(proposal.cycles, counts)]
    unaffected = (proposal.original == proposal.temporal).nonzero()[:, 0]
    if len(unaffected):
        keep = torch.from_numpy(rng.choice(len(unaffected), size=min(96, len(unaffected)), replace=False))
        sample.append(unaffected[keep])
    scored = torch.cat(sample)
    guesses = torch.stack((proposal.original[scored], proposal.temporal[scored]))
    exact, pairs = unique_pair_ocf(guesses, proposal.order[scored].expand_as(guesses), images >= 128,
                                  scorer, images.device, 384)
    strata = exact.double().split([len(rows) for rows in sample], dim=1)
    cells = torch.stack([values.mean(dim=1) for values in strata]).numpy()
    tracked = cells[:len(sizes)]
    per_cycle = sizes * (tracked[:, 1] - tracked[:, 0])
    total = len(proposal.order)
    baseline = float(sizes @ tracked[:, 0])
    if len(unaffected):
        baseline += len(unaffected) * float(cells[-1, 0])
    baseline /= total
    effect = {"v9": 0.0, "full_temporal": float(per_cycle.sum() / total),
              "contour_selected": float(per_cycle[chosen].sum() / total),
              "length_matched_random_mean": float(per_cycle @ random_probability / total)}
    random_draws = np.zeros(512)
    for length in np.unique(sizes):
        group = np.flatnonzero(sizes == length)
        admitted = int(chosen[group].sum())
        for index in range(len(random_draws)):
            random_draws[index] += per_cycle[rng.choice(group, admitted, replace=False)].sum() / total
    details = [{"length": int(size), "scored_rows": int(number), "accept": bool(accepted),
                "expected_contour_delta_sum": float(margin), "official_ocf_delta_per_scored_row": float(value / size),
                "full_block_contribution": float(value / total)}
               for size, number, accepted, margin, value in zip(sizes, counts, chosen, margins, per_cycle)]
    details.sort(key=lambda record: record["length"], reverse=True)
    actual = proposal.order
    return {"weighted_ocf_estimate": {name: baseline + difference for name, difference in effect.items()},
            "paired_delta_vs_v9": effect,
            "selected_minus_full_temporal": effect["contour_selected"] - effect["full_temporal"],
            "selected_minus_length_matched_random": effect["contour_selected"] - effect["length_matched_random_mean"],
            "conditional_random_interval": np.quantile(random_draws, (0.10, 0.90)).tolist(),
            "exact_pairs": pairs, "scored_changed_rows": int(counts.sum()),
            "scored_unchanged_rows": len(scored) - int(counts.sum()),
            "identity_top1": {name: float((guess == actual).float().mean())
                              for name, guess in (("v9", proposal.original), ("full_temporal", proposal.temporal),
                                                  ("contour_selected", selected))},
            "detail_by_cycle": details}


def save(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-days", choices=SESSIONS, nargs=2, default=("session1", "session3"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, default=23971)
    parser.add_argument("--calibration-rows", type=int, default=384)
    parser.add_argument("--changed-rows", type=int, default=224)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_temporal_cycles.json"))
    parser.set_defaults(encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if len(set(args.held_days)) != 2 or min(args.changed_rows, args.calibration_rows) < 2:
        parser.error("Use two distinct held days and valid sampling counts.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle CPU with official training data mounted.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    device = torch.device("cpu")
    images = load_images(root, device)
    blocks = recordings(root, load_catalog(root), MOUSE, device, len(images))
    result = {"mouse": MOUSE, "model_seed": args.seed, "recheck_base_sampling_seed": args.repeat_seed,
              "held_days": args.held_days, "gallery_size": len(images), "temporal_weight": SCALE,
              "min_effective_cycles": MIN_EFFECTIVE_CYCLES, "max_cycle_share": MAX_CYCLE_SHARE,
              "min_exact_length_exchangeable_cycles": MIN_EXCHANGEABLE_CYCLES,
              "min_selected_random_control_switchable_rows": MIN_CONTROL_SWITCH_ROWS,
              "protocol": "The original assignment and fixed timing proposal are complete. Score only whole cycles "
                          "with a frozen source-day-calibrated v9 identity posterior and a fixed contour proxy. "
                          "The random contrast matches the exact selected length histogram and therefore both cycle "
                          "count and changed rows. Exact OCF estimates use uniform within-cycle samples weighted by "
                          "cycle size. The same sample is used for every policy. Unchanged rows are sampled once for "
                          "the absolute score and cancel in every difference. No glyph-level independence SE is used. "
                          "Only competition training data are accessed.", "days": {}}
    proposals = {}
    for day in args.held_days:
        proposals[day] = discover(day, blocks, images, args)
        result["days"][day] = structural_report(proposals[day].cycles)
        result["days"][day]["original_unique_assignments"] = len(proposals[day].original.unique())
        result["days"][day]["temporal_unique_assignments"] = len(proposals[day].temporal.unique())
        result["seconds"] = time.monotonic() - started
        save(args.output, result)
        print(json.dumps({"held_day": day, "structure": result["days"][day]}), flush=True)

    if not all(day["structure_passed"] for day in result["days"].values()):
        result["stopped_after"] = "cycle_structure"
        save(args.output, result)
        return

    signature, energy = contour_signatures(images)
    selections = {}
    for day in args.held_days:
        temperature = source_temperature(day, blocks, images, args)
        selections[day] = select(proposals[day], signature, energy, temperature["temperature"])
        result["days"][day].update({"source_calibrated_temperature": temperature, **selections[day][-1]})
        result["seconds"] = time.monotonic() - started
        save(args.output, result)
        print(json.dumps({"held_day": day, "selection": selections[day][-1]}), flush=True)

    if not all(day["control_passed"] for day in result["days"].values()):
        result["stopped_after"] = "length_matched_control_support"
        save(args.output, result)
        return

    scorer = official_scorer(root)
    for day in args.held_days:
        rng = np.random.default_rng(args.repeat_seed + 200 + SESSIONS.index(day))
        scores = score_actions(proposals[day], selections[day], images, scorer, args, rng)
        result["days"][day]["official_ocf"] = scores
        result["seconds"] = time.monotonic() - started
        save(args.output, result)
        print(json.dumps({"held_day": day, "official_ocf": scores["paired_delta_vs_v9"]}), flush=True)
    daily = [day["official_ocf"] for day in result["days"].values()]
    result["mean_paired_delta_vs_v9_across_held_days"] = {name: float(np.mean([day["paired_delta_vs_v9"][name] for day in daily]))
                                                          for name in daily[0]["paired_delta_vs_v9"]}
    result["stopped_after"] = "exact_ocf_completed"
    save(args.output, result)


if __name__ == "__main__":
    main()
