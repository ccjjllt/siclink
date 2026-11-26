"""Evaluate selective cross-mouse edge nomination at a fixed Task 2 budget.

Fit match reliability using only source recordings and score two complete
held recordings with independently permuted mice1 and mice2 query rows.
All sparse arms retain the complete original v9 matching, use the same
number of candidate edges and promote exactly four edges per eligible row.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_task2_peer_mouse import MICE, SESSIONS, alignment, crossmouse_scores, logit_scale, task2_scores
from kaggle_task3_metric_validation import unique_pair_ocf


PARTNERS = 8
IDENTITY_TOP = 64
IDENTITY_TEMPERATURE = 0.35
MAX_COVERAGE = 0.25
KEEP = 12
PROPOSE = 4
NOMINATION_WEIGHT = 0.65
EDGE_BONUS = 0.45
ARMS = ("v9", "v9_deeper", "peer_nominated", "stratified_shuffle")


def identity_evidence(fit, query):
    gallery = fit_hierarchical_gallery(fit, view="total")
    score = gallery.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    return (logit_scale(score) / IDENTITY_TEMPERATURE).softmax(dim=1)


def sparse_identity(probability):
    values, labels = probability.topk(IDENTITY_TOP, dim=1)
    count = len(probability)
    return csr_matrix((values.flatten().numpy(), labels.flatten().numpy(),
                       np.arange(0, (count + 1) * IDENTITY_TOP, IDENTITY_TOP)), shape=(count, count))


@dataclass
class PairEvidence:
    raw: torch.Tensor
    overlap: torch.Tensor

    def __post_init__(self):
        self.raw_best = self.raw.amax(dim=0)
        self.overlap_best = self.overlap.amax(dim=0)

    def shortlist(self, selected):
        raw = self.raw.gather(1, selected)
        overlap = self.overlap.gather(1, selected)
        agreement = raw.clamp(0, 4) * overlap.clamp(0, 4) / 4
        return torch.stack((raw, overlap, (raw - self.raw_best[selected]).clamp(-4, 0),
                            (overlap - self.overlap_best[selected]).clamp(-4, 0), agreement), dim=-1)

    def dense_score(self, weights):
        agreement = self.raw.clamp(0, 4) * self.overlap.clamp(0, 4) / 4
        return (self.raw * weights[0] + self.overlap * weights[1]
                + (self.raw - self.raw_best).clamp(-4, 0) * weights[2]
                + (self.overlap - self.overlap_best).clamp(-4, 0) * weights[3] + agreement * weights[4])


def pair_evidence(sources, queries):
    raw = crossmouse_scores(alignment(sources), queries).clamp(-8, 8)
    left, right = (sparse_identity(identity_evidence(sources[mouse], queries[mouse])) for mouse in MICE)
    overlap = torch.from_numpy((right @ left.T).toarray()).mul_(len(raw)).log1p_()
    return PairEvidence(raw, logit_scale(overlap).clamp(-8, 8))


def independent_queries(recordings, day, rng):
    orders = {mouse: torch.as_tensor(rng.permutation(len(recordings[mouse][day]))) for mouse in MICE}
    return orders, {mouse: recordings[mouse][day][orders[mouse]] for mouse in MICE}


def calibration_pairs(evidence, pairing, rng):
    random_controls = torch.from_numpy(rng.integers(len(pairing), size=(len(pairing), 32)))
    selected = torch.cat((evidence.raw.topk(64, dim=1).indices, evidence.overlap.topk(64, dim=1).indices,
                          random_controls), dim=1).sort(dim=1).values
    keep = torch.ones_like(selected, dtype=torch.bool)
    keep[:, 1:] = selected[:, 1:] != selected[:, :-1]
    keep &= selected != pairing[:, None]
    return evidence.shortlist(torch.cat((pairing[:, None], selected), dim=1)), F.pad(keep, (1, 0), value=True)


def fit_pair_score(episodes):
    values, valid = (torch.cat(items) for items in zip(*episodes))
    start = values.new_tensor((1.0, 0.65, 0.20, 0.14, 0.15))
    weight = start.clone().requires_grad_()
    target = torch.zeros(len(values), dtype=torch.long)
    optimizer = torch.optim.Adam((weight,), lr=0.04)
    for _ in range(52):
        optimizer.zero_grad(set_to_none=True)
        logits = (values @ weight).masked_fill(~valid, -torch.inf)
        loss = F.cross_entropy(logits, target) + 0.04 * (weight - start).square().sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weight.clamp_(0, 3)
    return weight.detach()


def partner_rows(evidence, weights):
    score = evidence.dense_score(weights)
    ceiling = score.topk(PARTNERS + 1, dim=1)
    members = ceiling.indices[:, :PARTNERS]
    top_score = ceiling.values[:, :PARTNERS]
    raw_agree = (members == evidence.raw.argmax(dim=1, keepdim=True)).any(dim=1).float()
    identity_agree = (members == evidence.overlap.argmax(dim=1, keepdim=True)).any(dim=1).float()
    reciprocal_rank = score.gather(1, members[:, :1])[:, 0] - score.amax(dim=0)[members[:, 0]]
    features = torch.stack((torch.logsumexp(top_score, dim=1) - torch.logsumexp(score, dim=1),
                            top_score[:, 0] - ceiling.values[:, -1], top_score[:, 0],
                            raw_agree, identity_agree, reciprocal_rank), dim=1)
    return members, top_score.softmax(dim=1), features


@dataclass
class TrustModel:
    location: torch.Tensor
    scale: torch.Tensor
    weights: torch.Tensor
    intercept: torch.Tensor
    threshold: float

    def probability(self, features):
        return (((features - self.location) / self.scale) @ self.weights + self.intercept).sigmoid()


def fit_trust(features, labels):
    location = features.mean(dim=0)
    scale = (features.var(dim=0, unbiased=False) + 0.01).sqrt()
    model = TrustModel(location, scale, torch.zeros_like(location, requires_grad=True),
                       labels.mean().clamp(1e-4, 1 - 1e-4).logit().requires_grad_(), 0.0)
    optimizer = torch.optim.Adam((model.weights, model.intercept), lr=0.03)
    for _ in range(85):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy(model.probability(features), labels) + 0.02 * model.weights.square().sum()
        loss.backward()
        optimizer.step()
    model.weights = model.weights.detach()
    model.intercept = model.intercept.detach()
    model.threshold = float(model.probability(features).quantile(1 - MAX_COVERAGE))
    return model


def calibrate(sources, rng):
    episodes, training, truth = [], [], []
    for day in sources["mice1"]:
        fit = {mouse: {key: value for key, value in rows.items() if key != day} for mouse, rows in sources.items()}
        order, query = independent_queries(sources, day, rng)
        pairing = order["mice1"].argsort()[order["mice2"]]
        evidence = pair_evidence(fit, query)
        truth.append(pairing)
        training.append(calibration_pairs(evidence, pairing, rng))
        episodes.append(evidence)

    checks, outcomes = [], []
    for index, (evidence, pairing) in enumerate(zip(episodes, truth)):
        held_ranker = fit_pair_score([data for offset, data in enumerate(training) if offset != index])
        partners, _, features = partner_rows(evidence, held_ranker)
        checks.append(features)
        outcomes.append((partners == pairing[:, None]).any(dim=1).float())
    features, labels = torch.cat(checks), torch.cat(outcomes)
    trust = fit_trust(features, labels)
    selected = trust.probability(features) >= trust.threshold
    report = {"source_episodes": list(sources["mice1"]), "source_top8_hit": float(labels.mean()),
              "source_selected_coverage": float(selected.float().mean()),
              "source_selected_top8_hit": float(labels[selected].mean()), "absolute_threshold": trust.threshold}
    return fit_pair_score(training), trust, report


def stratified_peer_shuffle(probability, peer_query, rng):
    """Shuffle the peer identity payload within concentration and rate strata."""
    peak = probability.amax(dim=1).argsort()
    total = peer_query.sum(dim=(1, 2))
    shuffled = torch.empty_like(peak)
    for confidence in peak.tensor_split(8):
        for stratum in confidence[total[confidence].argsort()].tensor_split(4):
            indices = stratum[torch.from_numpy(rng.permutation(len(stratum)))]
            shuffled[indices] = indices.roll(1)
    return shuffled


def borrowed_probability(probability, links, weights):
    return torch.cat([(probability[part] * mass[..., None]).sum(dim=1)
                      for part, mass in zip(links.split(96), weights.split(96))])


def fixed_candidates(base, original, proposal, eligible):
    leaders = base.topk(KEEP, dim=1).indices
    needs_original = (leaders != original[:, None]).all(dim=1)
    leaders[needs_original, -1] = original[needs_original]
    score = torch.where(eligible[:, None], proposal, base).clone()
    score.scatter_(1, leaders, -torch.inf)
    added = score.topk(PROPOSE, dim=1).indices
    return torch.cat((leaders, added), dim=1)


def sparse_match(edges, base, eligible, bonus):
    objective = base.gather(1, edges).double()
    objective[:, -PROPOSE:] += bonus * eligible[:, None]
    cost = 1.0 + objective.max().item() - objective.numpy().ravel()
    count, slots = edges.shape
    graph = csr_matrix((cost, edges.numpy().ravel(), np.arange(0, slots * (count + 1), slots)), shape=(count, count))
    rows, choices = min_weight_full_bipartite_matching(graph)
    result = torch.empty(count, dtype=torch.long)
    result[torch.from_numpy(rows)] = torch.from_numpy(choices).long()
    return result


def stats(values, anchor):
    delta = (values - anchor).mean(dim=0)
    return {"ocf": float(values.mean()), "paired_delta": float(delta.mean()),
            "paired_glyph_se": float(delta.std(unbiased=True) / len(delta)**0.5)}


def evaluate(held, recordings, images, samples, scorer, args):
    started = time.monotonic()
    rng = np.random.default_rng(args.split_seed + 1009 * SESSIONS.index(held))
    sources = {mouse: {day: counts for day, counts in rows.items() if day != held} for mouse, rows in recordings.items()}
    weights, trust, calibration = calibrate(sources, rng)
    print(f"{held}: source pairing fit in {time.monotonic() - started:.1f}s", flush=True)
    order, queries = independent_queries(recordings, held, rng)
    evidence = pair_evidence(sources, queries)
    links, link_weights, confidence_features = partner_rows(evidence, weights)
    confidence = trust.probability(confidence_features)
    eligible = torch.zeros(len(confidence), dtype=torch.bool)
    high = confidence.topk(int(MAX_COVERAGE * len(confidence))).indices
    eligible[high] = confidence[high] >= trust.threshold

    estimates = {}
    for mouse in MICE:
        torch.manual_seed(args.seed)
        estimates[mouse] = task2_scores(mouse, sources[mouse], queries[mouse], images, args)
    base = logit_scale(estimates["mice2"])
    actual_v9 = assign_glyphs(estimates["mice2"])
    peer = (logit_scale(estimates["mice1"]) / IDENTITY_TEMPERATURE).softmax(dim=1)
    shuffled_peer = peer[stratified_peer_shuffle(peer, queries["mice1"], rng)]
    borrowed = {name: logit_scale(borrowed_probability(payload, links, link_weights).clamp_min(1e-12).log())
                for name, payload in (("peer_nominated", peer), ("stratified_shuffle", shuffled_peer))}
    evidence_logits = {"v9_deeper": base, **{arm: base + NOMINATION_WEIGHT * values for arm, values in borrowed.items()}}
    graphs = {name: fixed_candidates(base, actual_v9, score, eligible) for name, score in evidence_logits.items()}
    sparse_v9 = sparse_match(graphs["v9_deeper"], base, eligible, 0.0)
    rows = torch.arange(len(base))
    objective_gap = float((base[rows, sparse_v9].double() - base[rows, actual_v9].double()).sum())
    if abs(objective_gap) > 5e-4:
        raise RuntimeError(f"Sparse v9 lost the original feasible matching: objective gap {objective_gap}")
    decisions = {"v9": actual_v9, **{name: sparse_match(edges, base, eligible, EDGE_BONUS) for name, edges in graphs.items()}}

    # The independently permuted correspondences enter only post-decision diagnostics.
    truth = order["mice2"]
    true_partner = order["mice1"].argsort()[truth]
    partner_hit = links == true_partner[:, None]
    raw_hit = evidence.raw.topk(PARTNERS, dim=1).indices == true_partner[:, None]
    identity_hit = evidence.overlap.topk(PARTNERS, dim=1).indices == true_partner[:, None]
    sample_rows = truth.argsort()[samples]
    guesses = torch.stack([decisions[name][sample_rows] for name in ARMS])
    exact, pairs = unique_pair_ocf(guesses, samples.expand_as(guesses), images >= 128, scorer, images.device, 384)
    metrics = {name: {**stats(exact[index][None], exact[0][None]),
                      "identity_accuracy": float((decisions[name] == truth).float().mean()),
                      "changed_vs_v9": int((decisions[name] != actual_v9).sum()),
                      "unique_outputs": len(decisions[name].unique())} for index, name in enumerate(ARMS)}
    match_report = {"raw_top8": float(raw_hit.any(dim=1).float().mean()),
                    "identity_overlap_top8": float(identity_hit.any(dim=1).float().mean()),
                    "joint_top8": float(partner_hit.any(dim=1).float().mean()),
                    "joint_top1": float(partner_hit[:, 0].float().mean()),
                    "eligible_joint_top8": float(partner_hit[eligible].any(dim=1).float().mean()) if eligible.any() else None,
                    "eligible_joint_top1": float(partner_hit[eligible, 0].float().mean()) if eligible.any() else None,
                    "eligible_rows": int(eligible.sum()), "pair_coefficients": weights.tolist()}
    pool_report = {}
    for name, edges in graphs.items():
        hit = (edges == truth[:, None]).any(dim=1)
        added = edges[:, -PROPOSE:]
        novel = (added[:, :, None] != graphs["v9_deeper"][:, None, :]).all(dim=2)
        used = (decisions[name][:, None] == added).any(dim=1) & eligible
        pool_report[name] = {"edges_per_row": edges.shape[1], "top16_with_assignment_edge": float(hit.float().mean()),
                             "eligible_pool_hit": float(hit[eligible].float().mean()) if eligible.any() else None,
                             "novel_edges": int(novel.sum()), "true_glyph_among_novel_edges": int((novel & (added == truth[:, None])).sum()),
                             "boosted_edge_used": int(used.sum())}
    info = {"held_day": held, "sources": list(sources["mice1"]), "calibration": calibration, "matching": match_report,
            "full_gallery_size": len(images), "sparse_v9_objective_gap": objective_gap,
            "sparse_v9_equivalent_tied_rows": int((sparse_v9 != actual_v9).sum()),
            "pool": pool_report, "actions": metrics, "peer_minus_deeper": stats(exact[2][None], exact[1][None]),
            "peer_minus_shuffle": stats(exact[2][None], exact[3][None]),
            "unique_exact_pairs": pairs, "seconds": time.monotonic() - started}
    return exact, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-days", nargs=2, choices=SESSIONS, default=("session2", "session4"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--split-seed", type=int, default=63729)
    parser.add_argument("--ocf-images", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_peer_nomination.json"))
    parser.set_defaults(encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if args.held_days[0] == args.held_days[1] or args.ocf_images < 2:
        parser.error("Provide two distinct days and at least two OCF images per day.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run with Kaggle competition input.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, torch.device("cpu"))
    catalog = load_catalog(root)
    recordings = {mouse: grouped_blocks(root, catalog, mouse, images.device, len(images)) for mouse in MICE}
    if any(set(days) != set(SESSIONS) or any(data.shape[1] != 4 for data in days.values()) for days in recordings.values()):
        raise ValueError("Each mouse needs four labeled days and four repeats per glyph.")
    prior_sample = np.random.default_rng(17691).permutation(len(images))[:160]
    eligible = np.setdiff1d(np.arange(len(images)), prior_sample)
    samples = torch.from_numpy(np.random.default_rng(args.split_seed).choice(eligible, size=args.ocf_images, replace=False))
    scorer = official_scorer(root)
    result = {"arms": ARMS, "seed": args.seed, "split_seed": args.split_seed, "paired_glyphs_per_day": len(samples),
              "gallery": "All labeled classes; independently permuted outer query rows; no formal test.",
              "comparison": "Actual v9 full matching against three equal-budget, identically boosted graphs. "
                            "The shuffled arm changes the peer payload within confidence and count strata.",
              "max_eligible_coverage": MAX_COVERAGE, "top_partner_rows": PARTNERS, "preserved_edges": KEEP,
              "proposed_edges": PROPOSE, "nomination_weight": NOMINATION_WEIGHT, "same_edge_bonus": EDGE_BONUS,
              "days": []}
    all_values = []
    for day in args.held_days:
        values, info = evaluate(day, recordings, images, samples, scorer, args)
        all_values.append(values)
        result["days"].append(info)
        measured = torch.stack(all_values)
        result["summary"] = {name: stats(measured[:, index], measured[:, 0]) for index, name in enumerate(ARMS)}
        result["peer_minus_deeper"] = stats(measured[:, 2], measured[:, 1])
        result["peer_minus_shuffle"] = stats(measured[:, 2], measured[:, 3])
        result["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
