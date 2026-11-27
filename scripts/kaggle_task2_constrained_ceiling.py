"""Measure full one-to-one and relaxed exact OCF ceilings on matched v9 shortlists."""

import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import torch

from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, combine_logits, fit_retrieval, retrieval_score_components
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standard_error
from kaggle_task2_local_cycles import held_glyphs as cycle_cohort
from kaggle_task2_mice2_ranker_scale import previous_mice2_cohorts
from kaggle_task2_native_hanzi import SEED as HANZI_SEED, scoring_glyphs as hanzi_cohorts
from kaggle_task2_peer_mouse import logit_scale


MOUSE = "mice2"
SOURCE_DAYS, HELD_DAYS = ("session1", "session3"), ("session2", "session4")
SEED, MODEL_SEED, EPOCHS = 286019, 211, 12
CORE, EXTRA = 8, 4
WEIGHT = 0.15
ACTION_SAMPLE, ORACLE_UNIFORM, ORACLE_CROWDED = 64, 32, 16
V9, DEEP, RECIPROCAL = "v9", "v9_depth", "reciprocal_nomination"
OUTPUT = Path("/kaggle/working/task2_constrained_ceiling.json")


def fresh_samples(size):
    all_glyphs = np.arange(size)
    earlier = previous_mice2_cohorts(size)
    scaling = np.random.default_rng(41721).choice(np.setdiff1d(all_glyphs, earlier), size=256, replace=False)
    cycles = cycle_cohort(size, 52873, 768, torch.device("cpu"))[0].numpy()
    hanzi, _ = hanzi_cohorts(size, np.random.default_rng(HANZI_SEED))
    prior = np.unique(np.concatenate((earlier, scaling, cycles, *(rows.numpy() for rows in hanzi.values()))))
    eligible = np.setdiff1d(all_glyphs, prior)
    sample = np.random.default_rng(SEED).choice(eligible, len(HELD_DAYS) * ACTION_SAMPLE, replace=False)
    panel = dict(zip(HELD_DAYS, torch.from_numpy(sample).split(ACTION_SAMPLE), strict=True))
    return panel, torch.from_numpy(np.setdiff1d(eligible, sample)), len(prior)


def candidate_graphs(v9, baseline):
    # Column rank is query-specific. A mere additive column bias would cancel under one-to-one assignment.
    preferences = v9.argsort(dim=0, descending=True, stable=True)
    rank = torch.empty_like(preferences)
    rank.scatter_(0, preferences, torch.arange(1, len(v9) + 1)[:, None].expand_as(preferences))
    reciprocal = logit_scale(-rank.float().log())
    objective = v9 + WEIGHT * reciprocal
    keep = v9.topk(CORE, dim=1).indices
    missing = (keep != baseline[:, None]).all(1)
    keep[missing, -1] = baseline[missing]
    graphs = {name: torch.cat((keep, score.clone().scatter_(1, keep, -torch.inf).topk(EXTRA, dim=1).indices), dim=1)
              for name, score in ((DEEP, v9), (RECIPROCAL, objective))}
    for edges in graphs.values():
        if bool((edges.sort(1).values.diff(dim=1) == 0).any()) or not bool((edges == baseline[:, None]).any(1).all()):
            raise ValueError("Every row must have exactly 12 distinct edges including its full v9 assignment.")
    return graphs, objective


def solve(edges, objective):
    count, slots = edges.shape
    weights = objective.double().numpy()
    if not np.isfinite(weights).all():
        raise ValueError("Every allowed edge must have a finite objective.")
    costs = 1 + weights.max() - weights.ravel()
    graph = csr_matrix((costs, edges.numpy().ravel(), np.arange(0, (count + 1) * slots, slots)), shape=(count, count))
    rows, columns = min_weight_full_bipartite_matching(graph)
    chosen = torch.empty(count, dtype=torch.long)
    chosen[torch.from_numpy(rows)] = torch.from_numpy(columns).long()
    if chosen.unique().numel() != count:
        raise ValueError("Every constrained solution must use all 2,748 glyphs exactly once.")
    return chosen


def paired(values, control):
    gain = values - control
    return {"ocf": float(values.mean()), "paired_delta": float(gain.mean()), "paired_glyph_se": standard_error(gain),
            "scored_glyphs": len(gain)}


def publish(report, started, phase):
    report.update(stage=phase, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": phase, "seconds": report["seconds"]}), flush=True)


def oracle_report(graph, row_ids, metric, baseline, stock):
    """Score sampled rows exactly and leave the remaining rows as feasibility constraints."""
    pool = graph[row_ids]
    labels = metric.evaluate(pool)
    relaxed_index = torch.from_numpy(labels.argmax(axis=1)).long()
    relaxed = labels.max(axis=1)
    edges = graph.new_zeros(graph.shape, dtype=torch.float64)
    edges[row_ids] = torch.from_numpy(labels)
    matched = solve(graph, edges)
    constrained = metric.evaluate(matched[row_ids, None])[:, 0]
    if constrained.sum() < stock.sum() - 1e-7 or (constrained > relaxed + 1e-7).any():
        raise ValueError("The exact constrained solution must lie between feasible v9 and the relaxed oracle.")
    strata = {"uniform": slice(0, ORACLE_UNIFORM), "crowded": slice(ORACLE_UNIFORM, None)}
    report = {}
    for name, subset in strata.items():
        independent_choices = pool.gather(1, relaxed_index[:, None])[subset, 0]
        report[name] = {"v9_ocf": float(stock[subset].mean()),
                        "rowwise_oracle_vs_v9": paired(relaxed[subset], stock[subset]),
                        "full_one_to_one_oracle_vs_v9": paired(constrained[subset], stock[subset]),
                        "one_to_one_minus_rowwise": paired(constrained[subset], relaxed[subset]),
                        "rowwise_duplicate_glyphs": int(len(independent_choices) - independent_choices.unique().numel())}
    unlabeled = torch.ones(len(graph), dtype=torch.bool)
    unlabeled[row_ids] = False
    report["unscored_rows_changed_vs_v9"] = int((matched[unlabeled] != baseline[unlabeled]).sum())
    report["oracle_target_total"] = paired(constrained, stock)
    return report, {name: (relaxed[subset], constrained[subset]) for name, subset in strata.items()}


@torch.no_grad()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Run on a private Kaggle CPU instance with the official labeled training input.")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(MODEL_SEED)
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED + 1)
    root = locate_data()
    images = load_images(root, device)
    masks = images >= 128
    if len(images) != 2748:
        raise ValueError("Expected all 2,748 known Task 2 glyphs.")
    uniform, stress_eligible, previous = fresh_samples(len(images))
    scorer = official_scorer(root)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if any(recordings[day].shape[:2] != (len(images), 4) for day in (*SOURCE_DAYS, *HELD_DAYS)):
        raise ValueError("Use each labeled day with its four measured repeats per glyph.")
    sources = {day: recordings[day] for day in SOURCE_DAYS}
    with torch.enable_grad():
        retrieval = fit_retrieval(as_trial_blocks(sources), len(images), images=images, device=device, seed=MODEL_SEED, epochs=EPOCHS)
    hierarchy = fit_hierarchical_gallery(sources, view="total")
    report = {"mouse": MOUSE, "seed": SEED, "v9_model_seed": MODEL_SEED, "source_days": SOURCE_DAYS, "held_days": HELD_DAYS,
              "gallery_size": len(images), "previous_glyphs_excluded": previous, "formal_inputs_read": False,
              "v9_classifier_epochs": EPOCHS, "edge_budget": CORE + EXTRA, "common_edges": CORE,
              "reciprocal_column_rank_weight": WEIGHT, "action_uniform_glyphs_per_day": ACTION_SAMPLE,
              "oracle_uniform_glyphs_per_day": ORACLE_UNIFORM, "oracle_crowded_glyphs_per_day": ORACLE_CROWDED,
              "protocol": "Fit v9 on two labeled source days and independently permute each held day. Compare two 12-edge candidate "
                          "graphs that share eight v9 leaders including the original assigned edge. Add four ordinary v9 or four "
                          "reciprocal nominations. The reciprocal signal is the negative log within-column query rank with row "
                          "centering and global RMS normalization. Both graph interventions use the identical v9 + 0.15 reciprocal "
                          "objective. Crowded samples come from rows with high v9 top-choice contention, conditional on disjoint "
                          "fresh glyph eligibility. The constrained oracle maximizes exact OCF jointly on both sampled strata under a "
                          "complete permutation; every unscored row has zero utility and still consumes one distinct glyph. "
                          "The relaxed oracle scores the same shortlist independently by row. These ceilings apply to sampled rows "
                          "only. Exact held OCF never enters non-oracle decisions.", "recordings": {}}
    publish(report, started, "v9_fit_ready")

    action, oracles = [], []
    for day in HELD_DAYS:
        publish(report, started, f"{day}_started")
        order = torch.from_numpy(rng.permutation(len(images)))
        query = recordings[day][order]
        validated = combine_logits(retrieval_score_components(retrieval, query, None, task="task2"), "task2")
        empirical = hierarchy.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
        raw = blend_hierarchical_logits(validated, empirical)
        baseline = assign_glyphs(raw)
        v9 = logit_scale(raw)
        graphs, score = candidate_graphs(v9, baseline)
        rows = torch.arange(len(images))
        sparse_v9 = solve(graphs[DEEP], v9.gather(1, graphs[DEEP]))
        gap = float((v9[rows, sparse_v9].double() - v9[rows, baseline].double()).sum())
        if abs(gap) > 5e-4:
            raise ValueError("The sparse depth graph must retain the exact dense v9 objective.")
        choices = {V9: baseline, **{name: solve(graph, score.gather(1, graph)) for name, graph in graphs.items()}}
        lead = v9.argmax(1)
        eligible_rows = order.argsort()[stress_eligible]
        crowded_rows = eligible_rows[lead.bincount(minlength=len(images))[lead[eligible_rows]].topk(ORACLE_CROWDED).indices]
        crowded = order[crowded_rows]
        stress_eligible = stress_eligible[~torch.isin(stress_eligible, crowded)]
        labels = torch.cat((uniform[day], crowded))
        labeled_rows = order.argsort()[labels]
        oracle_ids = torch.cat((uniform[day][:ORACLE_UNIFORM], crowded))
        oracle_rows = order.argsort()[oracle_ids]
        oracle_metric = PairedOCF(scorer, images, masks[oracle_ids].numpy(), device, 12)
        tail_metric = PairedOCF(scorer, images, masks[uniform[day][ORACLE_UNIFORM:]].numpy(), device, 12)
        outcomes = {}
        for name, choice in choices.items():
            main_prefix = oracle_metric.evaluate(choice[oracle_rows, None])[:ORACLE_UNIFORM, 0]
            tail = tail_metric.evaluate(choice[labeled_rows[ORACLE_UNIFORM:ACTION_SAMPLE], None])[:, 0]
            outcomes[name] = np.concatenate((main_prefix, tail))
        selected_scores = {name: paired(scores, outcomes[V9]) for name, scores in outcomes.items()}
        details = {"uniform_glyphs": uniform[day].tolist(), "crowded_glyphs": crowded.tolist(), "v9_sparse_objective_gap": gap,
                   "uniform_selected_ocf": selected_scores, "reciprocal_minus_depth": paired(outcomes[RECIPROCAL], outcomes[DEEP]),
                   "assigned_top1": {name: float((choice == order).float().mean()) for name, choice in choices.items()},
                   "full_gallery_candidate_truth_recall": {name: float((pool == order[:, None]).any(1).float().mean())
                                                           for name, pool in graphs.items()}, "oracle": {}}
        report["recordings"][day] = details
        publish(report, started, f"{day}_selections_scored")
        v9_oracle = oracle_metric.evaluate(baseline[oracle_rows, None])[:, 0]
        oracle_values = {}
        for name, graph in graphs.items():
            details["oracle"][name], oracle_values[name] = oracle_report(graph, oracle_rows, oracle_metric, baseline, v9_oracle)
        details["unique_exact_pairs"] = len(oracle_metric.values) + len(tail_metric.values)
        action.append(outcomes)
        oracles.append(oracle_values)
        publish(report, started, f"{day}_ceilings_scored")

    report["uniform_selected_ocf"] = {name: paired(np.concatenate([day[name] for day in action]),
                                                  np.concatenate([day[V9] for day in action])) for name in (V9, DEEP, RECIPROCAL)}
    report["reciprocal_minus_depth"] = paired(np.concatenate([day[RECIPROCAL] for day in action]),
                                              np.concatenate([day[DEEP] for day in action]))
    report["oracle_comparison"] = {}
    for stratum in ("uniform", "crowded"):
        depth, reverse = [tuple(np.concatenate([day[name][stratum][index] for day in oracles]) for index in range(2))
                          for name in (DEEP, RECIPROCAL)]
        report["oracle_comparison"][stratum] = {"reciprocal_rowwise_minus_depth": paired(reverse[0], depth[0]),
                                                "reciprocal_one_to_one_minus_depth": paired(reverse[1], depth[1]),
                                                "depth_one_to_one_minus_rowwise": paired(depth[1], depth[0]),
                                                "reciprocal_one_to_one_minus_rowwise": paired(reverse[1], reverse[0])}
    publish(report, started, "complete")


if __name__ == "__main__":
    main()
