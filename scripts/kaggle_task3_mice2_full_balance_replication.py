"""Replicate fixed seven-candidate mice2 Task 3 balancing on fresh labeled trials."""

import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_exact_utility import fit_exact_utility, gallery_geometry, gallery_utility
from glyph_gain_likelihood import gain_template_logits
from glyph_mbr import contour_signatures, fit_temperature, similarity_from_signatures
from kaggle_diagnostics import official_scorer
from kaggle_posterior_utility_validation import fit_neural
from kaggle_task3_anchor_coverage import novel_repeats, scoring_glyphs
from kaggle_task3_early_late_innovation import make_cohort as early_cohorts
from kaggle_task3_half_balance_recheck import (OUTER, SHORTLIST, TEMP, cohorts as prior_cohorts,
                                              disjoint_trial_split, evaluate, summary)
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_ocf_recheck import new_cohorts as prior_ocf_cohorts, new_trial_split
from kaggle_task3_rankband_ocf_recheck import gather_repeats
from kaggle_task3_rankband_ocf_validation import NeuralRows, stage, stratified_pairs
from kaggle_task3_relational_consensus import PARTIAL_BALANCE, UTILITY_WEIGHT
from kaggle_task3_shape_capacity_validation import REPEAT_SEED as CAPACITY_SEED, cohorts as capacity_cohorts


MOUSE = "mice2"
SEED = 95731
REPEAT_SEED = 95747
OUTPUT = Path("/kaggle/working/task3_mice2_full_balance_replication.json")
UTILITY_TABLE = OUTPUT.with_name("task3_mice2_full_balance_utility.npy")


def cohorts(size, device):
    visual, prior, _ = prior_cohorts(size, device)
    _, _, capacity = capacity_cohorts(size, device)
    _, early = early_cohorts(size, device)
    _, _, exact, _ = prior_ocf_cohorts(size, device)
    _, old_balance_temperature = prior[MOUSE]
    previous = {"half_balance": prior[MOUSE][0], "half_balance_temperature": old_balance_temperature,
                "anchor_coverage": scoring_glyphs(size, 128)[0], "prior_exact_ocf": exact,
                "shape_capacity": capacity, "early_late": early,
                "soft_exclusion": torch.from_numpy(np.random.default_rng(1531).permutation(size)[2432:2560]),
                "older_posterior_utility": torch.from_numpy(np.random.default_rng(352).permutation(size)[192:288])}
    excluded = torch.unique(torch.cat((visual, *previous.values())))
    order = torch.from_numpy(np.random.default_rng(SEED).permutation(size))
    choices = order[~torch.isin(order, excluded)]
    if len(choices) < OUTER + TEMP:
        raise ValueError(f"Only {len(choices)} cohorts remain outside the frozen visual fit and prior evaluations.")
    held, temperature = choices[:OUTER + TEMP].split((OUTER, TEMP))
    overlap = {name: int(torch.isin(held, old).sum()) for name, old in previous.items()}
    return visual, held, temperature, {"eligible_glyphs": len(choices), "earlier_outer_glyph_overlap": overlap,
                                       "visual_and_scored_glyph_overlap": int(torch.isin(held, visual).sum())}


def split_trials(recordings, scored):
    index = {day: torch.arange(4).reshape(1, 4, 1).expand(len(counts), -1, -1)
             for day, counts in recordings.items()}
    prior_ocf = new_trial_split(index)[3]
    prior_balance = disjoint_trial_split(index, MOUSE)[3]
    prior_anchor = novel_repeats(index, MOUSE)[1]
    inner_fit, inner_query, outer_fit, outer_query, diagnostics = {}, {}, {}, {}, {}
    for offset, (day, counts) in enumerate(recordings.items()):
        _, capacity = split_repeats(index[day], CAPACITY_SEED + offset)
        ids = index[day][..., 0]
        earlier = {"prior_exact_ocf": prior_ocf[day][:, 0, 0], "shape_capacity": capacity[:, 0, 0],
                   "half_balance": prior_balance[day][:, 0, 0], "anchor_coverage": prior_anchor[day][:, 0, 0]}
        blocked = (ids == earlier["prior_exact_ocf"][:, None]) | (ids == earlier["shape_capacity"][:, None])
        rng = torch.Generator().manual_seed(REPEAT_SEED + offset)
        penalty = (ids == earlier["half_balance"][:, None]).float() + (ids == earlier["anchor_coverage"][:, None])
        random = torch.rand(ids.shape, generator=rng) + penalty
        held = random.masked_fill(blocked, torch.inf).argmin(dim=1, keepdim=True)
        support = ids[ids != held].reshape(len(counts), 3)
        keep, calibration = split_repeats(support[..., None], REPEAT_SEED + 1000 + offset)
        inner_fit[day], inner_query[day] = gather_repeats(counts, keep[..., 0]), gather_repeats(counts, calibration[..., 0])
        outer_fit[day], outer_query[day] = gather_repeats(counts, support), gather_repeats(counts, held)
        equal = {name: ids == held[:, 0] for name, ids in earlier.items()}
        diagnostics[day] = {"same_repeat_index_all_glyphs": {name: int(hit.sum()) for name, hit in equal.items()},
                            "same_repeat_index_new_scoring_glyphs": {name: int(hit[scored].sum()) for name, hit in equal.items()},
                            "inner_and_outer_held_overlap": int((calibration[:, 0, 0] == held[:, 0]).sum())}
    return inner_fit, inner_query, outer_fit, outer_query, diagnostics


@torch.no_grad()
def panel(model, counts, gallery, day, batch_size):
    encoder, contour, gain = model
    rates = encoder.expected_counts(gallery, day)
    scores = torch.cat([blend_contour_poisson(contour_retrieval_scores(contour, batch, day)["contour_ridge"],
                                              gain_template_logits(batch, rates, gain))
                        for batch in counts.split(batch_size)])
    anchor = scores.argmax(dim=1, keepdim=True)
    remaining = scores.clone().scatter_(1, anchor, -torch.inf).topk(SHORTLIST - 1, dim=1).indices
    return NeuralRows(scores, scores, torch.cat((anchor, remaining), dim=1))


def main():
    if not Path("/kaggle/input").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Run this replication on a Kaggle T4 with official labeled data.")
    started = time.monotonic()
    torch.set_num_threads(4)
    cpu, device = torch.device("cpu"), torch.device("cuda:0")
    root = locate_data()
    scorer = official_scorer(root)
    args = SimpleNamespace(seed=211, visual_pairs=14336, query_batch=64)
    gallery_cpu = load_images(root, cpu)
    if len(gallery_cpu) != 2748:
        raise ValueError("Expected all 2,748 known Task 3 glyphs.")
    # Keep CPU RNG, the glyph axes and the exact visual pair draw from the CPU run.
    visual, held_cpu, temperature_cpu, cohort_report = cohorts(len(gallery_cpu), cpu)
    signatures_cpu, energy_cpu = contour_signatures(gallery_cpu)
    proxy_cpu = similarity_from_signatures(signatures_cpu, energy_cpu)
    draw = torch.Generator().manual_seed(args.seed + 641)
    action, reference, strata = stratified_pairs(visual, proxy_cpu, args.visual_pairs, draw)
    del proxy_cpu

    gallery = gallery_cpu.to(device)
    masks = gallery >= 128
    signatures, energy = signatures_cpu.to(device), energy_cpu.to(device)
    held, temperature = held_cpu.to(device), temperature_cpu.to(device)
    report = {"protocol": {"mouse": MOUSE, "device": str(device), "cohort_and_trial_rng_device": "cpu",
                           "gallery_size": len(gallery), "same_candidate_budget": SHORTLIST,
                           "outer_glyphs_per_day": len(held), "temperature_glyphs_per_source_day": len(temperature),
                           "visual_supervision_glyphs": len(visual), "requested_exact_visual_pairs": args.visual_pairs,
                           "glyph_seed": SEED, "repeat_seed": REPEAT_SEED, "formal_test_accessed": False,
                           "fixed_exact_utility_weight": UTILITY_WEIGHT, "half_balance_fraction": PARTIAL_BALANCE,
                           "balance_iterations": 24, "encoder_epochs": 18,
                           "method": "Reuse the previous visual fit cohort, stratified pair sampling rule "
                                     "and exact utility model architecture. "
                                     "Reproduce raw v14 contour and temporal gain scores. Calibrate temperature on three "
                                     "other sessions. Shuffle full observed blocks independently, compare original v14 and "
                                     "fixed seven-candidate 20% exact utility under independent, half and full Sinkhorn mass. "
                                     "Only labeled training recordings are read."}, "cohorts": cohort_report,
              "sessions": {}}

    def save(progress):
        report.update(stage=progress, seconds=time.monotonic() - started)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage(progress, started)

    save("ready")
    geometry = gallery_geometry(masks, signatures, energy)
    action, reference = action.to(device), reference.to(device)
    observed, pairs = unique_pair_ocf(action, reference, masks, scorer, device, 384)
    metric = fit_exact_utility(geometry, action, reference, observed, seed=args.seed + 643)
    utility = gallery_utility(metric, geometry)
    table = utility.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    np.save(UTILITY_TABLE, table, allow_pickle=False)
    original_gallery = gallery_cpu.contiguous().numpy()
    report["visual_fit"] = {"exact_unique_pairs": pairs, "strata": strata, "ocf_batch_size": 12,
                            "utility_table": {"file": UTILITY_TABLE.name, "dtype": str(table.dtype),
                                              "shape": list(table.shape),
                                              "sha256": hashlib.sha256(UTILITY_TABLE.read_bytes()).hexdigest(),
                                              "ordered_gallery_sha256": hashlib.sha256(original_gallery.tobytes()).hexdigest(),
                                              "ordered_gallery_shape": list(original_gallery.shape),
                                              "ordered_gallery_dtype": str(original_gallery.dtype)}}
    del geometry, metric, table, original_gallery, gallery_cpu, signatures_cpu, energy_cpu
    save("shared_visual_fit")

    recordings = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(gallery))
    if len(recordings) != 4 or any(counts.shape[1] != 4 for counts in recordings.values()):
        raise ValueError("Expected four complete mice2 training sessions with four repeats.")
    inner, calibration, support, queries, diagnostics = split_trials(recordings, held_cpu)
    report["trial_exclusion_checks"] = diagnostics
    del recordings
    inner, calibration, support, queries = ({day: tensor.to(device) for day, tensor in arrays.items()}
                                             for arrays in (inner, calibration, support, queries))
    inner_model = fit_neural(inner, gallery, signatures, energy, MOUSE, args.seed)
    calibration = {day: panel(inner_model, trials[temperature], gallery, day, args.query_batch).mixed
                   for day, trials in calibration.items()}
    del inner, inner_model
    model = fit_neural(support, gallery, signatures, energy, MOUSE, args.seed)
    del support
    save("neural_fit")

    rng = np.random.default_rng(SEED + 501)
    measured = {}
    for day, counts in queries.items():
        order = torch.from_numpy(rng.permutation(len(counts))).to(device)
        source_days = [peer for peer in calibration if peer != day]
        calibrated = fit_temperature([(calibration[peer], temperature) for peer in source_days])
        predictions = panel(model, counts[order], gallery, day, args.query_batch)
        values, detail = evaluate(predictions, calibrated["temperature"], order, held, utility, masks, scorer)
        for arm, scores in values.items():
            measured.setdefault(arm, []).append(scores)
        report["sessions"][day] = {"source_days_for_temperature": source_days, "temperature": calibrated, **detail}
        report["summary_ocf"] = summary({arm: np.stack(scores) for arm, scores in measured.items()})
        save(day + "_scored")
    save("complete")


if __name__ == "__main__":
    main()
