"""Replace the fitted v20 Task 3 utility estimate with a sampled exact correction."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures, glyph_posterior
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standard_error
from kaggle_posterior_utility_validation import fit_neural, v11_logits
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_full_balance_replication import cohorts
from kaggle_task3_mice2_utility_candidate import (GALLERY_SIZE, MOUSE, REPORT_SHA, SHORTLIST, TABLE_SHA,
                                                  validated_utility)
from kaggle_task3_rankband_ocf_validation import NeuralRows, scale_actions
from kaggle_task3_relational_consensus import INDEPENDENT, UTILITY_WEIGHT, choose_images
from kaggle_task3_repeat_posterior import expected_utility
from kaggle_task3_v20_candidate_headroom import DAYS, GLYPHS_PER_DAY, SEED, held_repeats


V20 = "v20"
FULL = "exact_control_variate"
HALVES = ("first_half", "second_half")
DRAW_SEED = SEED + 2000
PANEL_SEED = SEED + 1000
PROFILES = {"full": (GLYPHS_PER_DAY, 32), "pilot": (16, 32), "small": (6, 8)}
OUTPUT_NAME = "task3_v20_exact_utility.json"


def args_for_run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/input"),
                        help="competition mount, extracted competition directory, or raw dataset root")
    parser.add_argument("--utility-root", type=Path,
                        help="directory containing the pinned report and table, or a parent directory (default: --data-root)")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working") / OUTPUT_NAME)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:0")
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--small", dest="profile", action="store_const", const="small",
                          help="six preselected glyphs per day and eight draws per query")
    profiles.add_argument("--pilot", dest="profile", action="store_const", const="pilot",
                          help="16 preselected glyphs per day and 32 draws per query")
    profiles.add_argument("--full", dest="profile", action="store_const", const="full",
                          help="48 glyphs per day and 32 draws per query (default)")
    parser.set_defaults(profile="full")
    return parser.parse_args()


def held_panels(previous, fit_glyphs, glyph_count):
    """Reproduce the headroom cohort, then optionally take a separate seeded subset."""
    order_rng = np.random.default_rng(SEED)
    selection_rng = np.random.default_rng(PANEL_SEED)
    sampled = torch.from_numpy(order_rng.choice(previous.numpy(), len(DAYS) * GLYPHS_PER_DAY, replace=False))
    if torch.isin(sampled, fit_glyphs).any():
        raise RuntimeError("Scoring glyphs overlap the pinned utility or temperature fits.")
    references, orders = {}, {}
    for day, panel in zip(DAYS, sampled.split(GLYPHS_PER_DAY), strict=True):
        if glyph_count < len(panel):
            draw = np.sort(selection_rng.choice(len(panel), glyph_count, replace=False))
            panel = panel[torch.from_numpy(draw)]
        references[day] = panel
        orders[day] = torch.from_numpy(order_rng.permutation(GALLERY_SIZE))
    return references, orders


def gallery_draws(posterior, draw_count, seed):
    """Use independent stratified inverse-CDF draws per half and no reference labels."""
    measure = posterior.detach().to(device="cpu", dtype=torch.float64)
    uniform = 1.0 / measure.shape[1]
    proposal = 0.5 * (measure + uniform)
    proposal /= proposal.sum(dim=1, keepdim=True)
    cdf = proposal.cumsum(dim=1)
    cdf[:, -1] = 1.0
    half = draw_count // 2
    rng = torch.Generator().manual_seed(seed)
    location = (torch.arange(half) + torch.rand((len(measure), 2, half), dtype=measure.dtype, generator=rng)) / half
    sampled = torch.searchsorted(cdf, location.flatten(1), right=True)
    weight = (measure.gather(1, sampled) - uniform) / proposal.gather(1, sampled)
    return sampled.reshape_as(location).to(posterior.device), weight.reshape_as(location).to(posterior)


def original_panel(logits, temperature, utility):
    posterior = glyph_posterior(logits, temperature, balance=False)
    first = logits.argmax(dim=1, keepdim=True)
    remaining = logits.clone().scatter_(1, first, -torch.inf).topk(SHORTLIST - 1, dim=1).indices
    actions = torch.cat((first, remaining), dim=1)
    current = choose_images(NeuralRows(logits, logits, actions), {INDEPENDENT: posterior}, utility)[INDEPENDENT]
    return actions, posterior, current


def control_variate(logits, actions, posterior, stock_choice, utility, masks, scorer, device, draws, seed):
    gallery, weight = gallery_draws(posterior, draws, seed)
    candidate = actions[:, :, None, None].expand(-1, -1, gallery.shape[1], gallery.shape[2])
    references = gallery[:, None].expand_as(candidate)
    measured, unique = unique_pair_ocf(candidate, references, masks, scorer, device, 384)
    residual = measured - utility[candidate, references]
    correction = (residual * weight[:, None]).mean(dim=-1)
    base_excess = expected_utility(posterior, actions, utility) - utility.mean(dim=1)[actions]
    fixed_anchor = (1 - UTILITY_WEIGHT) * scale_actions(logits.gather(1, actions), 0.30)

    def choose(excess):
        return (fixed_anchor + UTILITY_WEIGHT * scale_actions(excess, 0.50)).argmax(dim=1)

    baseline = choose(base_excess)
    if not torch.equal(actions.gather(1, baseline[:, None])[:, 0], stock_choice):
        raise RuntimeError("The uncorrected score must reproduce the inherited v20 action.")
    first = choose(base_excess + correction[..., 0])
    second = choose(base_excess + correction[..., 1])
    combined = choose(base_excess + correction.mean(dim=-1))
    arms = {V20: baseline, FULL: combined, HALVES[0]: first, HALVES[1]: second}
    first_change, second_change = first != baseline, second != baseline
    either = first_change | second_change
    centered = correction - correction.mean(dim=1, keepdim=True)
    left, right = centered.unbind(dim=-1)
    cross = float((left * right).sum())
    powers = float(left.square().sum()), float(right.square().sum())
    diagnostics = {"unique_sampled_ocf_pairs": unique, "exact_pair_draws_before_deduplication": candidate.numel(),
                   "sampled_utility_residual_mae": float(residual.abs().mean()),
                   "max_abs_importance_weight": float(weight.abs().max()),
                   "mean_posterior_mass_top32": float(posterior.topk(32, dim=1).values.sum(1).mean()),
                   "both_halves_same_pick": int((first == second).sum()),
                   "either_half_changes": int(either.sum()),
                   "both_halves_same_replacement": int(((first == second) & either).sum()),
                   "half_correction_cosine": cross / max((powers[0] * powers[1]) ** 0.5, 1e-12),
                   "half_difference_rms": float((left - right).square().mean().sqrt()),
                   "full_centered_correction_rms": float(centered.mean(dim=-1).square().mean().sqrt()),
                   "changed_from_v20": {name: int((choice != baseline).sum()) for name, choice in arms.items()}}
    return arms, diagnostics


def paired_summary(ocf):
    base = ocf[V20]
    result = {}
    for name, scores in ocf.items():
        delta = (scores - base).cpu().numpy()
        result[name] = {"ocf": float(scores.mean()), "paired_delta_vs_v20": float(delta.mean()),
                        "paired_se": standard_error(delta), "scored_glyphs": len(delta)}
    return result


def publish(report, path, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


@torch.no_grad()
def main():
    args = args_for_run()
    glyphs_per_day, draw_count = PROFILES[args.profile]
    chosen = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device, cpu = torch.device(chosen), torch.device("cpu")
    torch.set_num_threads(4)
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = args.data_root if (args.data_root / "catalog.json").is_file() else locate_data(args.data_root)
    images_cpu = load_images(root, cpu)
    utility, fitted_temperatures, _ = validated_utility(images_cpu, device, args.utility_root or args.data_root)
    temperatures = {day: fitted_temperatures[day]["temperature"] for day in DAYS}
    visual, previous, calibration, _ = cohorts(len(images_cpu), cpu)
    references, orders = held_panels(previous, torch.cat((visual, calibration)), glyphs_per_day)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images_cpu), bin_ms=25)
    support, query, overlap = held_repeats(recordings, previous)
    del recordings
    images = images_cpu.to(device)
    del images_cpu
    masks = images >= 128
    report = {"device": str(device), "profile": args.profile, "cohort_seed": SEED, "panel_seed": PANEL_SEED,
              "draw_seed": DRAW_SEED, "mouse": MOUSE, "days": list(DAYS), "glyphs_per_day": glyphs_per_day,
              "gallery": len(images), "candidate_budget": SHORTLIST, "draws_per_query": draw_count,
              "utility_weight": UTILITY_WEIGHT, "fixed_temperature_by_day": temperatures,
              "utility_sha256": TABLE_SHA, "utility_report_sha256": REPORT_SHA, "trial_overlap": overlap,
              "protocol": "Retain the actual v20-inherited seven actions, temperature, 80/20 mixture and 0.30/0.50 "
                          "scale floors. The two independently stratified half draws sample a half-posterior, "
                          "half-uniform known gallery proposal. Every candidate shares a query's full draw. "
                          "Subtract the pinned image-pair utility from official OCF, weigh by posterior-minus-uniform "
                          "over proposal, and add this estimate to the pinned expected excess utility. The original "
                          "headroom reference identities and disjoint held trial indices are reused in full mode. "
                          "Pilot and small modes sample fixed random subsets before any scoring. Scored reference identity "
                          "only enters the actual paired endpoints after all decisions. Neural data are from "
                          "labeled training, with three support repeats and one held single query.",
              "recordings": {}}
    publish(report, args.output, started, "inputs_verified")
    with torch.enable_grad():
        model = fit_neural({day: values.to(device) for day, values in support.items()}, images,
                           *contour_signatures(images), MOUSE, 211)
    del support
    scorer = official_scorer(root)
    all_scores = []
    for offset, day in enumerate(DAYS):
        order = orders[day]
        truth_cpu = references[day]
        rows = order.argsort()[truth_cpu].to(device)
        logits = v11_logits(model, query[day][order].to(device), day, images)[rows]
        actions, posterior, current = original_panel(logits, temperatures[day], utility)
        ranks, diagnostics = control_variate(logits, actions, posterior, current, utility, masks, scorer, device,
                                              draw_count, DRAW_SEED + offset)
        report["recordings"][day] = {"selected_glyph_indices": truth_cpu.tolist(), "diagnostics": diagnostics}
        publish(report, args.output, started, f"{day}_decisions")

        truth = truth_cpu.to(device)
        exact, unique = unique_pair_ocf(actions, truth[:, None].expand_as(actions), masks, scorer, device, 384)
        scores = {name: exact.gather(1, position[:, None])[:, 0].cpu() for name, position in ranks.items()}
        scores["oracle_seven"] = exact.amax(1).cpu()
        all_scores.append(scores)
        report["recordings"][day].update(official_eval_pairs=unique, exact_ocf=paired_summary(scores))
        publish(report, args.output, started, f"{day}_ocf")
        print(json.dumps(report["recordings"][day]["exact_ocf"]), flush=True)
    report["pooled_ocf"] = paired_summary({name: torch.cat([day[name] for day in all_scores]) for name in all_scores[0]})
    publish(report, args.output, started, "complete")
    print(json.dumps(report["pooled_ocf"]), flush=True)


if __name__ == "__main__":
    main()
