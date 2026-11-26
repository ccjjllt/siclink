"""Test if anonymous single-trial neighborhood relations transfer across Task 3 recordings."""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_encoding import _evoked_counts
from glyph_exact_utility import gallery_geometry, gallery_utility
from glyph_mbr import contour_signatures, fit_temperature, glyph_posterior, similarity_from_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_early_late_innovation import make_cohort as early_late_cohorts
from kaggle_task3_measured_metric_validation import paired
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_microns_gamma_prior import REPEAT_SEED as MICRONS_REPEAT, cohorts as microns_cohorts
from kaggle_task3_rankband_ocf_recheck import gather_repeats
from kaggle_task3_rankband_ocf_validation import NeuralRows, fit_visual_metric, neural_rows, scale_actions, stage
from kaggle_task3_repeat_posterior import expected_utility, fit_models
from kaggle_task3_shape_capacity_validation import REPEAT_SEED as CAPACITY_REPEAT, cohorts as capacity_cohorts


SEED = 90821
REPEAT_SEED = 90823
OUTER = 320
TEMP_GLYPHS = 192
QUERY_NEIGHBORS = 16
CLASS_NEIGHBORS = 32
GEOMETRY_DIMENSIONS = 72
PARTIAL_BALANCE = 0.5
PEER_WEIGHT = 0.45
RELATION_WEIGHT = 0.8
UTILITY_WEIGHT = 0.2
ACTUAL = "v14_top7"
INDEPENDENT = "independent_utility"
BALANCED = "partially_balanced_utility"
STATIC = "static_relation"
JOINT = "peer_relation"
SHUFFLED = "shuffled_peer_relation"
OUTPUT = Path("/kaggle/working/task3_relational_consensus.json")


@dataclass(frozen=True)
class Neighbors:
    indices: torch.Tensor
    weight: torch.Tensor

    def aggregate(self, values):
        blocks = zip(self.indices.split(96), self.weight.split(96), strict=True)
        return torch.cat([(values[index] * weight[..., None]).sum(dim=1) for index, weight in blocks])


@dataclass(frozen=True)
class SpikeGeometry:
    center: torch.Tensor
    count_center: torch.Tensor
    count_slope: torch.Tensor
    projection: torch.Tensor

    def transform(self, counts):
        raw = _evoked_counts(counts.float(), "timed")
        gain = raw.sum(dim=-1, keepdim=True).log1p() - self.count_center
        residual = (raw + 0.1).sqrt() - self.center - gain * self.count_slope
        return F.normalize(residual @ self.projection, dim=-1)


@dataclass(frozen=True)
class RecordingEvidence:
    query: torch.Tensor
    panel: NeuralRows
    geometry: SpikeGeometry
    support_vectors: torch.Tensor
    graph: Neighbors
    shuffled_graph: Neighbors
    independent: torch.Tensor
    partial: torch.Tensor
    relation: torch.Tensor
    shuffled_relation: torch.Tensor


def cohorts(size, device):
    visual, _, capacity = capacity_cohorts(size, device)
    _, early_late = early_late_cohorts(size, device)
    _, _, microns, _ = microns_cohorts(size, device)
    previous = torch.unique(torch.cat((capacity, early_late, microns)))
    order = torch.as_tensor(np.random.default_rng(SEED).permutation(size), device=device)
    available = order[~torch.isin(order, previous)]
    if len(available) < TEMP_GLYPHS + OUTER:
        raise RuntimeError("Insufficient independent scoring and temperature glyphs.")
    temperature, held = available[:TEMP_GLYPHS], available[TEMP_GLYPHS:TEMP_GLYPHS + OUTER]
    reserved = torch.cat((temperature, held))
    relation_fit = order[~torch.isin(order, reserved)]
    visual_fit = visual[~torch.isin(visual, reserved)]
    if len(visual_fit) <= 512 or torch.isin(visual_fit, reserved).any() or torch.isin(held, previous).any():
        raise RuntimeError("The visual fit, score, and prior evaluation glyph exclusions must be respected.")
    return visual_fit, relation_fit, temperature, held, len(previous)


def independent_trials(recordings):
    inner_fit, inner_query, outer_fit, outer_query = {}, {}, {}, {}
    for index, (day, counts) in enumerate(recordings.items()):
        if counts.shape[1] != 4:
            raise RuntimeError("Expected four independently recorded training repeats.")
        axis = torch.arange(4, device=counts.device).expand(len(counts), -1)
        _, microns = split_repeats(axis[..., None], MICRONS_REPEAT + index)
        _, capacity = split_repeats(axis[..., None], CAPACITY_REPEAT + index)
        previous = (axis == microns[..., 0]) | (axis == capacity[..., 0])
        draw = torch.Generator(device=counts.device).manual_seed(REPEAT_SEED + index)
        priority = torch.rand(axis.shape, device=counts.device, generator=draw).masked_fill(previous, -1)
        query = priority.argmax(dim=1, keepdim=True)
        training = axis[axis != query].reshape(len(counts), 3)
        inner, temp = split_repeats(training[..., None], REPEAT_SEED + 1000 + index)
        if previous.gather(1, query).any() or (temp[..., 0] == query).any():
            raise RuntimeError(f"A held or calibration repeat overlaps an excluded trial in {day}.")
        inner_fit[day] = gather_repeats(counts, inner[..., 0])
        inner_query[day] = gather_repeats(counts, temp[..., 0])
        outer_fit[day] = gather_repeats(counts, training)
        outer_query[day] = gather_repeats(counts, query)
    return inner_fit, inner_query, outer_fit, outer_query


@torch.no_grad()
def fit_geometry(counts, *, seed):
    timed = _evoked_counts(counts.float(), "timed")
    values = (timed + 0.1).sqrt()
    center = values.mean(dim=(0, 1))
    log_count = timed.sum(dim=-1, keepdim=True).log1p()
    count_center = log_count.mean()
    gain = log_count - count_center
    slope = (gain * (values - center)).mean(dim=(0, 1)) / (gain.square().mean() + 0.2)
    residual = values - center - gain * slope
    within = residual.var(dim=1, unbiased=True).mean(dim=0)
    noise = (0.85 * within + 0.15 * within.mean()).clamp_min(0.02).sqrt()
    residual /= noise

    rng = torch.Generator(device=counts.device).manual_seed(seed)
    held = torch.randint(3, (len(counts),), generator=rng, device=counts.device)
    left = residual[torch.arange(len(counts), device=counts.device), held]
    right = (residual.sum(dim=1) - left) / 2
    cross = (left - left.mean(dim=0)).T @ (right - right.mean(dim=0)) / len(counts)
    eigenvalues, axes = torch.linalg.eigh((0.5 * (cross + cross.T)).cpu())
    weights = eigenvalues[-GEOMETRY_DIMENSIONS:].clamp_min(1e-3).sqrt()
    projection = (axes[:, -GEOMETRY_DIMENSIONS:] * weights).to(counts.device) / noise[:, None]
    return SpikeGeometry(center, count_center, slope, projection)


@torch.no_grad()
def neighbor_graph(values, width, temperature):
    similarities = (values @ values.T).fill_diagonal_(-torch.inf)
    strength, indices = similarities.topk(width, dim=1)
    return Neighbors(indices, (strength / temperature).softmax(dim=1))


@torch.no_grad()
def shuffle_graph(graph, query, posterior, seed):
    activity = query.sum(dim=(-1, -2)).log1p()
    entropy = -(posterior * posterior.clamp_min(1e-25).log()).sum(dim=1)
    slices = torch.linspace(0, 1, 9, device=activity.device)[1:-1]
    rate_bin = torch.bucketize(activity.contiguous(), torch.quantile(activity, slices))
    confidence_bin = torch.bucketize(entropy.contiguous(), torch.quantile(entropy, slices[1::2]))
    stratum = rate_bin * 4 + confidence_bin
    original = torch.arange(len(query), device=query.device)
    permutation = original.clone()
    rng = torch.Generator(device=query.device).manual_seed(seed)
    for band in stratum.unique():
        indices = (stratum == band).nonzero()[:, 0]
        permutation[indices] = indices[torch.randperm(len(indices), device=query.device, generator=rng)]
    inverse = permutation.argsort()
    relabeled = Neighbors(inverse[graph.indices[permutation]], graph.weight[permutation])
    return relabeled, float((permutation != original).float().mean())


@torch.no_grad()
def peer_coupling(graph, posterior):
    columns = posterior.sqrt()
    centered = F.normalize(columns - columns.mean(dim=0), dim=0)
    relation = centered.T @ graph.aggregate(centered)
    return 0.5 * (relation + relation.T)


@torch.no_grad()
def peer_graph(base, relation):
    within_edges = relation.gather(1, base.indices)
    standard = (within_edges - within_edges.mean(dim=1, keepdim=True)) / within_edges.std(dim=1, keepdim=True, unbiased=False).clamp_min(0.01)
    return Neighbors(base.indices, (base.weight.log() + PEER_WEIGHT * standard.clamp(-2, 2)).softmax(dim=1))


@torch.no_grad()
def conditional_posterior(posterior, neighbors, class_graph):
    transition = torch.zeros_like(posterior).scatter_(1, class_graph.indices, class_graph.weight)
    neighborhood = neighbors @ transition.T
    ambient = posterior.mean(dim=0) @ transition.T
    evidence = ((neighborhood + 0.25 * ambient) / (1.25 * ambient).clamp_min(1e-20)).clamp_min(1e-10).log()
    return (posterior.clamp_min(1e-25).log() + RELATION_WEIGHT * evidence.clamp(-0.95, 0.95)).softmax(dim=1)


@torch.no_grad()
def choose_images(panel, posterior, utility):
    pool = panel.action
    anchor = panel.mixed.gather(1, pool)
    choices = {ACTUAL: pool.gather(1, anchor.argmax(dim=1, keepdim=True))[:, 0]}
    anchor = (1 - UTILITY_WEIGHT) * scale_actions(anchor, 0.30)
    common = utility.mean(dim=1)[pool]
    for arm, mass in posterior.items():
        value = expected_utility(mass, pool, utility) - common
        score = anchor + UTILITY_WEIGHT * scale_actions(value, 0.50)
        choices[arm] = pool.gather(1, score.argmax(dim=1, keepdim=True))[:, 0]
    return choices


def rank_quality(mass, truth, rows):
    observed = mass[rows]
    targets = truth[rows, None]
    ranked = observed.topk(32, dim=1).indices == targets
    return {**{f"top{width}": float(ranked[:, :width].any(dim=1).float().mean()) for width in (1, 16, 32)},
            "nll": float(-observed.gather(1, targets).clamp_min(1e-25).log().mean())}


def paired_results(measured):
    output = {arm: paired(score, measured[ACTUAL]) for arm, score in measured.items()}
    for arm in (INDEPENDENT, BALANCED, STATIC, SHUFFLED):
        output[f"peer_vs_{arm}"] = paired(measured[JOINT], measured[arm])
    return output


@torch.no_grad()
def evaluate_day(evidence, posterior, class_graph, truth, held, alternate, utility, masks, scorer):
    decisions = choose_images(evidence.panel, posterior, utility)
    rows = truth.argsort()[held]
    if not torch.equal(truth[rows], held) or evidence.panel.action.shape[1] != 7:
        raise RuntimeError("The counterpart scores need the same held glyphs and original v14 candidate budget.")
    picks = torch.stack([ids[rows] for ids in decisions.values()])
    exact, pairs = unique_pair_ocf(picks, held.expand_as(picks), masks, scorer, masks.device, 384)
    scores = dict(zip(decisions, exact.cpu().numpy(), strict=True))

    # Both graphs use independent measured repeats of the same held class set here only.
    first = neighbor_graph(evidence.geometry.transform(evidence.query[rows, 0]), QUERY_NEIGHBORS, 0.12).indices
    second = neighbor_graph(evidence.geometry.transform(alternate[held, 0]), QUERY_NEIGHBORS, 0.12).indices
    exact_neighbor_overlap = float((first[:, :, None] == second[:, None, :]).any(dim=-1).float().mean())
    candidates = class_graph.indices[held]
    match = {}
    for name, graph in (("observed", evidence.graph), ("matched_shuffle", evidence.shuffled_graph)):
        neighbors = truth[graph.indices[rows]]
        match[name] = float((neighbors[:, :, None] == candidates[:, None, :]).any(dim=2).float().mean())
    return scores, {"paired_ocf": paired_results(scores), "unique_exact_pairs": pairs,
                    "outer_full_gallery_ranking": {arm: rank_quality(mass, truth, rows) for arm, mass in posterior.items()},
                    "true_neighbor_fraction_in_peer_support_graph": match,
                    "disjoint_measured_repeat_neighbor_overlap_on_outer_glyphs": exact_neighbor_overlap,
                    "uniform_neighbor_overlap_on_outer_glyphs": QUERY_NEIGHBORS / (len(held) - 1),
                    "selected_image_changes_vs_v14": {arm: int((choice[rows] != decisions[ACTUAL][rows]).sum())
                                                       for arm, choice in decisions.items()}}


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use one Kaggle T4 with the official labeled competition input attached.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    args = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128, visual_pairs=14336, query_batch=64)
    root = locate_data()
    images = load_images(root, device)
    masks = images >= 128
    signatures, energy = contour_signatures(images)
    scorer = official_scorer(root)
    visual, relation_fit, temperature_ids, held, previous = cohorts(len(images), device)
    recordings = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if len(recordings) != 4 or len(images) != 2748:
        raise RuntimeError("The partial balancing control requires the four complete known-gallery recordings.")
    inner_support, inner_query, support, queries = independent_trials(recordings)
    del recordings
    order = np.random.default_rng(SEED + 7)
    truth = {day: torch.as_tensor(order.permutation(len(query)), device=device) for day, query in queries.items()}
    permuted = {day: query[truth[day]] for day, query in queries.items()}
    del queries

    geometry = gallery_geometry(masks, signatures, energy)
    proxy = similarity_from_signatures(signatures, energy)
    draw = torch.Generator(device=device).manual_seed(args.seed + 641)
    metric, _, visual_pairs = fit_visual_metric(masks, geometry, proxy, visual, scorer, draw, args)
    utility = gallery_utility(metric, geometry)
    del geometry, proxy, metric
    report = {"protocol": {"mouse": "mice1", "gallery_size": len(images), "held_glyphs_per_day": len(held),
                           "temperature_calibration_glyphs": len(temperature_ids), "visual_fit_glyphs": len(visual),
                           "neural_geometry_fit_glyphs": len(relation_fit), "prior_evaluation_glyphs_excluded": previous,
                           "same_candidate_budget": 7, "seed": SEED, "repeat_seed": REPEAT_SEED,
                           "query_neighbors": QUERY_NEIGHBORS, "class_neighbors": CLASS_NEIGHBORS,
                           "geometry_dimensions": GEOMETRY_DIMENSIONS, "partial_balance_fraction": PARTIAL_BALANCE,
                           "peer_edge_reweighting": PEER_WEIGHT, "relation_weight": RELATION_WEIGHT,
                           "exact_utility_weight": UTILITY_WEIGHT, "formal_test_inputs_read": False,
                           "description": "First independently shuffle each complete held single-repeat recording. The shared "
                                          "gallery can balance expected class counts, but no exact output assignment is imposed. "
                                          "Remove gain associated with total count and fit each day's embedding on independent "
                                          "labeled supports excluding the evaluation and calibration glyphs. Estimate each peer "
                                          "day's class relation with centered square-root posterior columns and observed query "
                                          "neighbors. Reweight the same support-graph edges by the other days' relations, and use "
                                          "one local neighbor update. The matched control conjugates each peer graph by an "
                                          "activity/entropy-stratified node permutation. Raw row order, identifiers and query "
                                          "labels never enter matching, posterior updates or image decisions. Held labels enter "
                                          "after image selection, for ranking, exact OCF and cross-repeat geometry diagnostics."},
              "exact_visual_fit_pairs": visual_pairs, "sessions": {}}

    def save(progress):
        report.update(stage=progress, seconds=time.monotonic() - started)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage(progress, started)

    save("visual_fit")
    inner, _ = fit_models(inner_support, images, signatures, energy, "mice1", args)
    del inner_support
    inner_logits = {day: neural_rows(inner, array[temperature_ids], images, day, args).mixed
                    for day, array in inner_query.items()}
    temperatures = {day: fit_temperature([(score, temperature_ids) for peer, score in inner_logits.items() if day != peer])
                    for day in inner_logits}
    del inner, inner_logits

    baseline, report["v14_fit"] = fit_models(support, images, signatures, energy, "mice1", args)
    evidence = {}
    for index, (day, query) in enumerate(permuted.items()):
        panel = neural_rows(baseline, query, images, day, args)
        independent = glyph_posterior(panel.mixed, temperatures[day]["temperature"])
        balanced = glyph_posterior(panel.mixed, temperatures[day]["temperature"], balance=True)
        partial = independent.clamp_min(1e-25).log().lerp(balanced.clamp_min(1e-25).log(), PARTIAL_BALANCE).softmax(dim=1)
        spike_geometry = fit_geometry(support[day][relation_fit], seed=SEED + 71 + index)
        support_vectors = F.normalize(spike_geometry.transform(support[day]).mean(dim=1), dim=-1)
        graph = neighbor_graph(spike_geometry.transform(query[:, 0]), QUERY_NEIGHBORS, 0.12)
        shuffled, shuffled_fraction = shuffle_graph(graph, query, partial, SEED + 151 + index)
        evidence[day] = RecordingEvidence(query, panel, spike_geometry, support_vectors, graph, shuffled, independent, partial,
                                          peer_coupling(graph, partial), peer_coupling(shuffled, partial))
        report["sessions"][day] = {"temperature": temperatures[day], "graph_node_relabel_fraction": shuffled_fraction}
        save(f"{day}_anonymous_graph")
    del baseline, support, permuted

    collected = {}
    for day, target in evidence.items():
        peers = {name: peer for name, peer in evidence.items() if name != day}
        shared_geometry = torch.cat([peer.support_vectors for peer in peers.values()], dim=1) / len(peers)**0.5
        static = neighbor_graph(shared_geometry, CLASS_NEIGHBORS, 0.09)
        shared = torch.stack([peer.relation for peer in peers.values()]).mean(dim=0)
        shuffled = torch.stack([peer.shuffled_relation for peer in peers.values()]).mean(dim=0)
        joint_graph, control_graph = peer_graph(static, shared), peer_graph(static, shuffled)
        neighbor_mass = target.graph.aggregate(target.partial)
        posteriors = {INDEPENDENT: target.independent, BALANCED: target.partial,
                      STATIC: conditional_posterior(target.partial, neighbor_mass, static),
                      JOINT: conditional_posterior(target.partial, neighbor_mass, joint_graph),
                      SHUFFLED: conditional_posterior(target.partial, neighbor_mass, control_graph)}
        measured, diagnostics = evaluate_day(target, posteriors, static, truth[day], held, inner_query[day], utility, masks, scorer)
        for arm, values in measured.items():
            collected.setdefault(arm, []).append(values)
        edge_shift = {name: float((graph.weight - static.weight).abs().sum(dim=1).mean()) / 2
                      for name, graph in ((JOINT, joint_graph), (SHUFFLED, control_graph))}
        report["sessions"][day].update(diagnostics, peer_recordings=list(peers), class_graph_weight_change=edge_shift)
        report["summary_ocf"] = paired_results({arm: np.stack(values) for arm, values in collected.items()})
        save(f"{day}_scored")
    save("complete")


if __name__ == "__main__":
    main()
