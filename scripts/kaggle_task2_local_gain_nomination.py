"""Test candidate recall after conditioning on correlated neuron-group counts."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_task2_mice2_ranker_scale import previous_mice2_cohorts
from kaggle_task2_peer_mouse import logit_scale, task2_scores
from kaggle_task2_peer_nomination import EDGE_BONUS, KEEP, PROPOSE, fixed_candidates, sparse_match
from kaggle_task3_metric_validation import unique_pair_ocf


MOUSE = "mice2"
SESSIONS = ("session1", "session2", "session3", "session4")
WINDOWS = ((50, 100), (100, 250))
CLUSTERS = 6
NOMINATION_MIX = 0.35
LOCAL = "local_conditions"
V9 = "v9"
DEEPER = "v9_deeper"
ARMS = (DEEPER, LOCAL, "shuffled_clusters", "global_conditions", "poisson_same_template")


def window_counts(counts):
    bins = counts.unflatten(-1, (-1, 10))
    return torch.stack([bins[..., left // 25:right // 25].sum(dim=-1) for left, right in WINDOWS], dim=-1)


def residual_correlation(natives):
    """Estimate spontaneous rate covariation after conditioning on each glyph."""
    total = natives.sum(dim=-1)
    expected = total.mean(dim=1, keepdim=True)
    residual = ((total - expected) / (expected + 0.50).sqrt()).flatten(0, 1)
    # Fit one common-drive nuisance per unit from all other units. This avoids
    # giving the tested unit its own counts as a noise regressor.
    background = (residual.sum(dim=1, keepdim=True) - residual) / (residual.shape[1] - 1)**0.5
    slope = (background * residual).mean(dim=0) / (background.square().mean(dim=0) + 0.03)
    repaired = residual - background * slope
    repaired /= (repaired.square().mean(dim=0) + 0.03).sqrt()
    similarity = repaired.T @ repaired / len(repaired)
    return similarity.fill_diagonal_(0)


def cluster_units(correlation, rng):
    """Fit equally sized spectral groups to source residual correlations."""
    affinity = correlation.clamp_min(0)
    degree = affinity.sum(dim=1).clamp_min(1e-3)
    normalized = affinity / (degree[:, None] * degree[None, :]).sqrt()
    _, axes = torch.linalg.eigh(normalized)
    embedding = torch.nn.functional.normalize(axes[:, -CLUSTERS:], dim=1)
    starts = [int(degree.argmax())]
    for _ in range(CLUSTERS - 1):
        distance = torch.cdist(embedding, embedding[starts]).amin(dim=1)
        distance[starts] = -1
        starts.append(int(distance.argmax()))
    centers = embedding[starts]
    sizes = torch.tensor([len(group) for group in torch.arange(len(embedding)).tensor_split(CLUSTERS)])
    quotas = torch.arange(CLUSTERS).repeat_interleave(sizes)
    tie = torch.from_numpy(rng.random((len(embedding), len(embedding)), dtype=np.float32)) * 1e-6
    labels = torch.empty(len(embedding), dtype=torch.long)
    for _ in range(5):
        costs = torch.cdist(embedding, centers[quotas]) + tie
        rows, selected = linear_sum_assignment(costs.numpy())
        labels[torch.from_numpy(rows)] = quotas[torch.from_numpy(selected)]
        centers = torch.stack([embedding[labels == label].mean(dim=0) for label in range(CLUSTERS)])
    return labels


def shuffled_clusters(labels, response, rng):
    shuffled = labels.clone()
    for group in response.argsort().tensor_split(6):
        order = group[torch.from_numpy(rng.permutation(len(group)))]
        shuffled[order] = labels[order.roll(1)]
    if not torch.equal(shuffled.bincount(), labels.bincount()):
        raise RuntimeError("Shuffling must preserve the group sizes.")
    return shuffled


def within_noise(correlation, membership):
    diagonal = torch.eye(len(membership), dtype=torch.bool)
    same = (membership[:, None] == membership[None, :]) & ~diagonal
    return float(correlation[same].mean())


def rate_templates(training, query):
    """Pool day-relative stimulus rates and update only anonymous unit means."""
    activity = torch.stack([block.mean(dim=(0, 1)) for block in training])
    relative = torch.stack([(block.mean(dim=1) + 0.10) / (one + 0.10)
                            for block, one in zip(training, activity)]).mean(dim=0)
    mean_rate = 0.50 * activity.mean(dim=0) + 0.50 * query.mean(dim=(0, 1))
    return (mean_rate * (0.90 * relative + 0.10)).clamp_min(0.0001), activity.mean(dim=0)


def log_likelihood(observed, expected, membership=None):
    log_rate = expected.log()
    if membership is None:
        cross = observed.flatten(1) @ log_rate.flatten(1).T
        return cross - expected.sum(dim=(1, 2))
    for label in membership.unique():
        group = membership == label
        log_rate[:, group] -= log_rate[:, group].logsumexp(dim=1, keepdim=True)
    return observed.flatten(1) @ log_rate.flatten(1).T


def cohort(number, count, seed):
    earlier = previous_mice2_cohorts(number)
    eligible = np.setdiff1d(np.arange(number), earlier)
    prior_scale = np.random.default_rng(41721).choice(eligible, size=256, replace=False)
    eligible = np.setdiff1d(eligible, prior_scale)
    return torch.as_tensor(np.random.default_rng(seed).choice(eligible, size=count, replace=False))


def fit_proposals(day, recordings, images, opts, rng):
    source = {name: block for name, block in recordings.items() if name != day}
    order = torch.as_tensor(rng.permutation(len(images)))
    observed = recordings[day][order]
    torch.manual_seed(opts.seed)
    raw_v9 = task2_scores(MOUSE, source, observed, images, opts)
    baseline = assign_glyphs(raw_v9)
    v9 = logit_scale(raw_v9)

    training = [window_counts(counts) for counts in source.values()]
    query = window_counts(observed)
    expected, mean_rate = rate_templates(training, query)
    noise = torch.stack([residual_correlation(one) for one in training]).mean(dim=0)
    labels = cluster_units(noise, rng)
    random_labels = shuffled_clusters(labels, mean_rate.sum(dim=1), rng)
    held_noise = residual_correlation(query)
    ordinary = query.mean(dim=1)
    choices = {LOCAL: labels, "shuffled_clusters": random_labels,
               "global_conditions": torch.zeros_like(labels), "poisson_same_template": None}
    scores = {name: logit_scale(log_likelihood(ordinary, expected, membership)) for name, membership in choices.items()}
    enabled = torch.ones(len(v9), dtype=torch.bool)
    proposal_logits = {DEEPER: v9, **{name: v9 + NOMINATION_MIX * score for name, score in scores.items()}}
    graphs = {name: fixed_candidates(v9, baseline, score, enabled) for name, score in proposal_logits.items()}
    sparse_unperturbed = sparse_match(graphs[DEEPER], v9, enabled, 0.0)
    rows = torch.arange(len(v9))
    difference = float((v9[rows, sparse_unperturbed].double() - v9[rows, baseline].double()).sum())
    if abs(difference) > 5e-4:
        raise RuntimeError("The unchanged v9 optimum must be feasible in the common-edge graph.")

    # All candidate sets are frozen before the held correspondence is used.
    coverage = {name: (edges == order[:, None]).any(dim=1) for name, edges in graphs.items()}
    source_corr = {"genuine": within_noise(noise, labels), "shuffled": within_noise(noise, random_labels)}
    held_corr = {"genuine": within_noise(held_noise, labels), "shuffled": within_noise(held_noise, random_labels)}
    report = {"day": day, "source_days": list(source), "group_sizes": labels.bincount().tolist(),
              "source_group_mean_noise_correlation": source_corr,
              "held_group_mean_noise_correlation": held_corr,
              "fixed_graph_budget": KEEP + PROPOSE, "sparse_v9_objective_gap": difference,
              "equivalent_sparse_v9_changed_ties": int((sparse_unperturbed != baseline).sum()),
              "same_budget_pool": {name: {"truth_recall": float(hit.float().mean()),
                                          "true_rows_lost_vs_depth": int((~hit & coverage[DEEPER]).sum()),
                                          "true_rows_added_vs_depth": int((hit & ~coverage[DEEPER]).sum()),
                                          "new_nomination_edges": int((edges[:, -PROPOSE:, None]
                                                                        != graphs[DEEPER][:, None]).all(dim=2).sum())}
                                   for (name, edges), hit in zip(graphs.items(), coverage.values())},
              "coverage_passed": all(int(coverage[LOCAL].sum()) > int(hit.sum())
                                     for name, hit in coverage.items() if name != LOCAL)}
    return order, v9, baseline, graphs, report


def paired(delta):
    return {"mean": float(delta.mean()), "glyph_se": float(delta.std(unbiased=True) / len(delta)**0.5)}


def exact_comparison(fold, images, sample, scorer, opts):
    order, v9, baseline, graphs, _ = fold
    active = torch.ones(len(v9), dtype=torch.bool)
    predictions = {V9: baseline, **{name: sparse_match(edges, v9, active, EDGE_BONUS) for name, edges in graphs.items()}}
    if any(len(values.unique()) != len(images) for values in predictions.values()):
        raise RuntimeError("A candidate nomination broke the complete assignment constraint.")
    rows = order.argsort()[sample]
    selected = torch.stack([values[rows] for values in predictions.values()])
    bank = images >= 128
    exact, image_pairs = unique_pair_ocf(selected, sample.expand_as(selected), bank, scorer, images.device, 384)
    values = dict(zip(predictions, exact))
    details = {name: {"ocf": float(value.mean()), "vs_v9": paired(value - values[V9]),
                      "identity_top1": float((predictions[name] == order).float().mean()),
                      "distinct_assignments": len(predictions[name].unique())} for name, value in values.items()}

    small = sample[:opts.oracle_glyphs]
    graph = torch.stack([edges[rows[:len(small)]] for edges in graphs.values()])
    oracle, oracle_pairs = unique_pair_ocf(graph, small[None, :, None].expand_as(graph), bank, scorer, images.device, 384)
    oracles = dict(zip(graphs, oracle.amax(dim=2)))
    return {"paired": details, "local_minus_controls": {name: paired(values[LOCAL] - reference)
                                                       for name, reference in values.items() if name != LOCAL},
            "oracle": {name: {"ocf": float(value.mean()), "local_minus_arm": paired(oracles[LOCAL] - value)}
                       for name, value in oracles.items()}, "oracle_glyphs": len(small),
            "unique_exact_pairs": image_pairs, "unique_exact_oracle_pairs": oracle_pairs}, values


def save(path, result, started):
    result["seconds"] = time.monotonic() - started
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--split-seed", type=int, default=44357)
    parser.add_argument("--held-days", nargs=2, choices=SESSIONS, default=("session1", "session4"))
    parser.add_argument("--ocf-glyphs", type=int, default=128)
    parser.add_argument("--oracle-glyphs", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_local_gain_nomination.json"))
    parser.set_defaults(encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if len(set(args.held_days)) != 2 or not 2 <= args.oracle_glyphs <= args.ocf_glyphs:
        parser.error("Provide two distinct training days and a valid paired OCF and oracle sample.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle CPU with official training data mounted.")
    started = time.monotonic()
    torch.set_num_threads(4)
    root = locate_data()
    images = load_images(root, torch.device("cpu"))
    blocks = grouped_blocks(root, load_catalog(root), MOUSE, images.device, len(images))
    if tuple(blocks) != SESSIONS or any(values.shape[1] != 4 for values in blocks.values()):
        raise RuntimeError("Need all four labeled days and four repeats per class.")
    result = {"mouse": MOUSE, "seed": args.seed, "split_seed": args.split_seed, "gallery_size": len(images),
              "evoked_windows_ms": WINDOWS, "cluster_count": CLUSTERS,
              "shared_base_edges": KEEP, "added_edge_budget": PROPOSE, "nomination_weight": NOMINATION_MIX,
              "same_tail_bonus_on_every_graph": EDGE_BONUS,
              "method": "Source-days only determine neuron groups from within-glyph repeat residual correlation. "
                        "Given each group and window's observed spike total, the conditional Poisson likelihood "
                        "removes any common multiplicative gain in that group. Source measured class rates are held "
                        "fixed across local, count-stratified cluster shuffle, whole-population, and unconditional "
                        "models. Only the candidate tails change. All graphs keep the original v9 matching and "
                        "use the same number of edges. Evaluate OCF only when the local nominations improve "
                        "truth coverage over every control in both days. The same tail boost is used on each "
                        "graph for exact assignment. Formal test data are unused.",
              "physiology": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC4534383/", "https://www.nature.com/articles/nature14273"],
              "days": {}}
    folds = {}
    for day in args.held_days:
        rng = np.random.default_rng(np.random.SeedSequence((args.split_seed, SESSIONS.index(day))))
        fold = fit_proposals(day, blocks, images, args, rng)
        folds[day] = fold
        result["days"][day] = fold[-1]
        print(json.dumps(fold[-1]), flush=True)
        save(args.output, result, started)
    result["ocf_performed"] = all(fold[-1]["coverage_passed"] for fold in folds.values())
    if result["ocf_performed"]:
        selection = cohort(len(images), args.ocf_glyphs, args.split_seed + 1)
        scorer = official_scorer(root)
        exact = []
        for day, fold in folds.items():
            report, scores = exact_comparison(fold, images, selection, scorer, args)
            result["days"][day]["official_exact"] = report
            exact.append(scores)
            save(args.output, result, started)
        result["pooled_paired_ocf_vs_v9"] = {name: {"ocf": float(torch.stack([day[name] for day in exact]).mean()),
                                                   "vs_v9": paired(torch.stack([day[name] - day[V9] for day in exact]).mean(0))}
                                              for name in exact[0]}
    save(args.output, result, started)


if __name__ == "__main__":
    main()
