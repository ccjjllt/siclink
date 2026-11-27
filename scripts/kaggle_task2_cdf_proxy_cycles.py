"""Test transfer of CDF-induced assignment cycles with the archived model's scores hidden."""

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, fit_retrieval
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task2_anchored_cdf import ADAPTED, FROZEN, evaluation_sample as anchored_cohort, rank_scores
from kaggle_task2_classifier_ensemble import evaluation_sample as ensemble_cohort
from kaggle_task2_constrained_ceiling import fresh_samples as ceiling_cohorts
from kaggle_task2_day_shift import hybrid_scores, sample_glyphs as shift_cohorts
from kaggle_task2_native_hanzi import SEED as HANZI_SEED, scoring_glyphs as hanzi_cohorts
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics
from kaggle_task2_rank_marginals import (BLEND, CEILING_CROWDED, EPOCHS, MODEL_SEED, MOUSE, TEMPERATURE, V9,
                                         evaluation_glyphs, responses, source_cdf, summarize)
from kaggle_task2_rank_marginals_recheck import PROVENANCE, evaluation_sample as recheck_cohort, fit_marginal_templates
from kaggle_task2_repeat_precision import evaluation_sample as precision_cohort
from kaggle_task2_temporal_cycles import assignment_cycles
from kaggle_v8_metric_validation import SESSIONS


SEED, PROXY_SEED = 286439, 587
HELD_DAYS = ("session3", "session4")
OCF_GLYPHS = 112
IDEAL, PROXY, PROXY_CDF, TRANSFER = "stock_cdf", "proxy", "proxy_cdf", "supported_cycle_transfer"
ARMS = (V9, IDEAL, PROXY, PROXY_CDF, TRANSFER)
OUTPUT = Path("/kaggle/working/task2_cdf_proxy_cycles.json")


def evaluation_sample(size):
    first, _ = evaluation_glyphs(size)
    recheck, _ = recheck_cohort(size)
    anchor, _ = anchored_cohort(size)
    followup, _ = anchored_cohort(size, followup=True)
    shift, oracle = shift_cohorts(size)
    ensemble, _ = ensemble_cohort(size)
    precision, _ = precision_cohort(size)
    ceiling, _, _ = ceiling_cohorts(size)
    hanzi, _ = hanzi_cohorts(size, np.random.default_rng(HANZI_SEED))
    earlier = {"rank_marginals": first.numpy(), "rank_marginals_recheck": recheck.numpy(),
               "anchored": anchor.numpy(), "anchored_followup": followup.numpy(), "day_shift": shift.numpy(),
               "day_shift_oracle": oracle.numpy(), "ensemble": ensemble.numpy(), "repeat_precision": precision.numpy(),
               "ceiling_uniform": np.concatenate(tuple(ceiling.values())), "ceiling_crowded": np.asarray(CEILING_CROWDED),
               "native_hanzi": np.concatenate(tuple(hanzi.values()))}
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(tuple(earlier.values()))))
    selected = np.sort(np.random.default_rng(SEED).choice(eligible, OCF_GLYPHS, replace=False))
    return torch.from_numpy(selected), {name: int(np.intersect1d(rows, selected).size) for name, rows in earlier.items()}


def identity_diagnostics(matrix, chosen, order):
    rows = torch.arange(len(order))
    density = logit_scale(matrix) / TEMPERATURE
    return {**row_diagnostics(matrix, order),
            "identity_nll": float((density.logsumexp(1) - density[rows, order]).mean()),
            "assigned_top1": float((chosen == order).float().mean()),
            "distinct_outputs": int(chosen.unique().numel())}


def supported_cycle_transfer(archived, proxy_model, hierarchy, templates, query, split):
    """Map out-of-half-confirmed proxy adaptation cycles to the frozen owner's glyphs."""
    halves = query.gather(1, split[..., None].expand_as(query)).split(2, dim=1)
    matrices = []
    for counts in halves:
        proxy = logit_scale(hybrid_scores(proxy_model, hierarchy, counts))
        ranks = rank_scores(counts, templates)
        matrices.append(((1 - BLEND) * proxy, BLEND * (ranks[ADAPTED] - ranks[FROZEN])))
    owner = archived.argsort()
    proposed, supported = [], []
    for index, (discovery_proxy, discovery_delta) in enumerate(matrices):
        original = assign_glyphs(discovery_proxy)
        modified = assign_glyphs(discovery_proxy + discovery_delta)
        diagnostic_proxy, diagnostic_delta = matrices[1 - index]
        cycles = assignment_cycles(original, modified)
        details = []
        for cycle in cycles:
            old, new = original[cycle], modified[cycle]
            rows = owner[old]
            proxy_gain = float((diagnostic_proxy[rows, new] - diagnostic_proxy[rows, old]).double().sum())
            delta_gain = float((diagnostic_delta[rows, new] - diagnostic_delta[rows, old]).double().sum())
            combined_gain = proxy_gain + delta_gain
            accepted = proxy_gain <= 0 < combined_gain and delta_gain > 0
            details.append({"size": len(rows), "held_proxy_gain": proxy_gain, "held_delta_gain": delta_gain,
                            "held_combined_gain": combined_gain, "passes": accepted})
            if accepted:
                supported.append((combined_gain / len(rows), index, rows, new))
        proposed.append({"discovery_half": index, "proposed_rows": sum(map(len, cycles)),
                         "proxy_discovery_agrees_archived": int((original == archived).sum()),
                         "cycles": details})
    assigned = archived.clone()
    used = torch.zeros(len(archived), dtype=torch.bool)
    accepted = []
    for mean_gain, half, rows, new in sorted(supported, key=lambda item: item[0], reverse=True):
        if bool(used[rows].any()):
            continue
        if not torch.equal(archived[rows].sort().values, new.sort().values):
            raise ValueError("Each transferred cycle must keep the exact glyph inventory of its archived owners.")
        assigned[rows] = new
        used[rows] = True
        accepted.append({"size": len(rows), "held_mean_gain": mean_gain, "discovery_half": half})
    return assigned, {"proposed_halves": proposed, "supported_length_counts": dict(Counter(len(row) for _, _, row, _ in supported)),
                      "accepted_cycles": accepted, "accepted_rows": int(used.sum())}


def publish(report, stage, started):
    report.update(stage=stage, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": stage, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU notebook with authorized labeled training recordings.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if len(images) != 2748 or tuple(recordings) != SESSIONS or any(counts.shape[:2] != (len(images), 4)
                                                                  for counts in recordings.values()):
        raise ValueError("Require four complete labeled recordings and four real repeats per glyph.")
    sample, overlap = evaluation_sample(len(images))
    metric = PairedOCF(official_scorer(root), images, (images[sample] >= 128).numpy(), device, 12)
    rng = np.random.default_rng(SEED + 1)
    report = {"mouse": MOUSE, "seed": SEED, "held_days": HELD_DAYS, "labeled_source_days_per_fold": 3,
              "gallery_size": len(images), "real_repeats": 4, "stock_seed": MODEL_SEED, "proxy_seed": PROXY_SEED,
              "classifier_epochs": EPOCHS, "cdf_weight": BLEND, "archive_provenance": PROVENANCE,
              "half_trial_frozen_cdf_source_pairs_per_glyph_day": 6,
              "scored_glyph_indices": sample.tolist(), "overlap_by_earlier_cohort": overlap, "formal_test_accessed": False,
              "primary_arm": TRANSFER, "paired_reference": V9, "recordings": {},
              "protocol": "Simulate a frozen archive with the stock 211 classifier. Only comparator assignments use "
                          "stock logits. Fit an independent seed-587 classifier. On each two-repeat half compare its "
                          "full proxy assignment with its target-minus-frozen CDF score correction. Freeze the half's "
                          "source CDF on all six source repeat pairs to match the target repeat count. Transfer only "
                          "these implied glyph cycles to the archived glyph owners. Score on the other repeats. "
                          "A cycle must gain under the held combined score and lose or tie under the held proxy "
                          "without the CDF delta. Consider both directions in descending held combined gain per row "
                          "and accept only disjoint cycles. The frozen archive's own permutation is the no-update "
                          "control. Stock+CDF and proxy+CDF use all four repeats and the same full target CDF fusion."}
    publish(report, "inputs_ready", started)
    exact = {name: [] for name in ARMS}
    for day in HELD_DAYS:
        source = {name: counts for name, counts in recordings.items() if day != name}
        blocks = as_trial_blocks(source)
        order = torch.from_numpy(rng.permutation(len(images)))
        query = recordings[day][order]
        split = torch.from_numpy(rng.random(query.shape[:2]).argsort(axis=1))
        hierarchy = fit_hierarchical_gallery(source, view="total")
        templates = fit_marginal_templates(source)
        target_cdf = rank_scores(query, templates)[ADAPTED]
        source_pairs = [responses(counts[:, pair]) for counts in source.values() for pair in combinations(range(4), 2)]
        half_templates = (*templates[:3], source_cdf(torch.cat(source_pairs)))
        del source_pairs
        choices, identity = {}, {}
        for seed, base_arm, cdf_arm in ((MODEL_SEED, V9, IDEAL), (PROXY_SEED, PROXY, PROXY_CDF)):
            publish(report, f"{day}_seed_{seed}_fitting", started)
            torch.manual_seed(seed)
            with torch.enable_grad():
                model = fit_retrieval(blocks, len(images), device=device, seed=seed, epochs=EPOCHS, images=images)
            base = hybrid_scores(model, hierarchy, query)
            blended = logit_scale(base).lerp(target_cdf, BLEND)
            for arm, matrix in ((base_arm, base), (cdf_arm, blended)):
                choices[arm] = assign_glyphs(matrix)
                identity[arm] = identity_diagnostics(matrix, choices[arm], order)
            del base, blended, matrix
        choices[TRANSFER], transfer = supported_cycle_transfer(choices[V9], model, hierarchy, half_templates, query, split)
        identity[TRANSFER] = {"assigned_top1": float((choices[TRANSFER] == order).float().mean()),
                              "distinct_outputs": int(choices[TRANSFER].unique().numel())}
        if any(value.unique().numel() != len(images) for value in choices.values()):
            raise ValueError("Every assignment must use each glyph exactly once.")
        for arm in ARMS:
            changed = choices[arm] != choices[V9]
            nonreplay = choices[PROXY] != choices[V9]
            identity[arm]["changed_vs_archive"] = int(changed.sum())
            identity[arm]["changed_where_proxy_missed_archive"] = int((changed & nonreplay).sum())
        report["recordings"][day] = {"labeled_source_days": tuple(source), "identity": identity,
                                     "full_proxy_agrees_archive": int((choices[PROXY] == choices[V9]).sum()),
                                     "transfer": transfer}
        publish(report, f"{day}_assignments_ready", started)
        rows = order.argsort()[sample]
        measured = {name: metric.evaluate(choices[name][rows, None])[:, 0] for name in ARMS}
        for name, values in measured.items():
            exact[name].append(values)
        report["recordings"][day]["ocf_vs_archive"] = {name: summarize(values, measured[V9]) for name, values in measured.items()}
        publish(report, f"{day}_scored", started)
    averages = {name: np.stack(values).mean(0) for name, values in exact.items()}
    report.update(ocf_vs_archive={name: summarize(values, averages[V9]) for name, values in averages.items()},
                  proxy_cdf_vs_proxy=summarize(averages[PROXY_CDF], averages[PROXY]),
                  transfer_vs_full_proxy_cdf=summarize(averages[TRANSFER], averages[PROXY_CDF]),
                  ocf_rows_per_arm=len(sample) * len(HELD_DAYS), unique_exact_pairs=len(metric.values))
    publish(report, "complete", started)


if __name__ == "__main__":
    main()
