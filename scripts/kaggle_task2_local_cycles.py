"""Learn exact-OCF local exchanges of the complete mice2 Task 2 assignment."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import fit_hierarchical_gallery
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task2_metric_validation import SESSIONS, score_block, shortlist
from kaggle_task2_mice2_ranker_scale import Gallery, MOUSE, paired, permute, previous_mice2_cohorts
from kaggle_task3_metric_validation import unique_pair_ocf


BASE = "v14_mice2_v9"
CYCLE = "cycle_head"
EDGE = "additive_edge_head"
GAP = "smallest_v9_objective_loss"
ARMS = (BASE, CYCLE, EDGE, GAP)
MAX_CALIBRATION_CYCLES = {2: 280, 3: 280}
SWAP_LIMITS = {2: 90, 3: 60}
OUTPUT = Path("/kaggle/working/task2_local_cycles.json")


@dataclass
class Panel:
    rows: torch.Tensor
    labels_old: torch.Tensor
    labels_new: torch.Tensor
    edge: torch.Tensor
    context: torch.Tensor
    stock_loss: torch.Tensor


@dataclass
class CycleGallery(Gallery):
    shape_agreement: torch.Tensor


@dataclass
class Ridge:
    center: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    intercept: torch.Tensor

    def predict(self, values):
        return (((values - self.center) / self.scale).clamp(-6, 6) @ self.weight + self.intercept)


def normalize(matrix):
    centered = matrix - matrix.mean(1, keepdim=True)
    return centered / centered.square().mean().sqrt().clamp_min(1e-6)


def fit_ridge(features, targets, penalty=0.07):
    center = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(0.04)
    values = ((features - center) / scale).clamp(-6, 6)
    mean = targets.mean()
    cov = values.T @ values / len(values)
    cov.diagonal().add_(penalty)
    weights = torch.linalg.solve(cov, values.T @ (targets - mean) / len(values))
    return Ridge(center, scale, weights, mean)


@torch.no_grad()
def cycle_rows(graph, baseline):
    """Enumerate two- and three-row rotations on the same candidate graph."""
    n, k = graph.indices.shape
    device = baseline.device
    rows = torch.arange(n, device=device)
    destination = baseline.argsort()[graph.indices]
    edges = graph.valid & (destination != rows[:, None])
    active_rows, slots = edges.nonzero(as_tuple=True)
    lookup = torch.full((n, n), -1, dtype=torch.int16, device=device)
    lookup[active_rows, destination[active_rows, slots]] = slots.to(lookup.dtype)
    target = destination[active_rows, slots]
    pair = (active_rows < target) & (lookup[target, active_rows] >= 0)
    cycles = {2: torch.stack((active_rows[pair], target[pair]), dim=1)}

    # A minimal first vertex deduplicates rotations. Opposite directions remain distinct.
    final = destination[target]
    triple = ((active_rows[:, None] < target[:, None]) & (active_rows[:, None] < final)
              & edges[target] & (lookup[final, active_rows[:, None]] >= 0))
    pair_index, end_slot = triple.nonzero(as_tuple=True)
    cycles[3] = torch.stack((active_rows[pair_index], target[pair_index], final[pair_index, end_slot]), dim=1)
    return cycles, lookup


@torch.no_grad()
def half_scores(source, query, seed):
    """Measure conditional support for both disjoint halves of the real query repeats."""
    device = query.device
    rng = torch.Generator(device=device).manual_seed(seed)
    order = torch.rand(query.shape[:2], device=device, generator=rng).argsort(1)
    separated = query.gather(1, order[:, :, None].expand_as(query))
    gallery = fit_hierarchical_gallery(source, view="total")
    return tuple(normalize(gallery.logits(part, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"])
                 for part in separated.split(2, dim=1))


@torch.no_grad()
def feature_panel(rows, lookup, graph, baseline, stock, halves, gallery):
    destinations = rows.roll(-1, dims=1)
    new, old = baseline[destinations], baseline[rows]
    alternative_slots = lookup[rows, destinations].long()
    if not bool((alternative_slots >= 0).all()):
        raise ValueError("Every selected rotation must use an available edge.")
    base_slots = ((graph.indices == baseline[:, None]) & graph.valid).long().argmax(dim=1)
    present = graph.features[rows, base_slots[rows]]
    alternative = graph.features[rows, alternative_slots]
    difference = alternative - present
    loss = stock[rows, old] - stock[rows, new]
    least, most = torch.stack([values[rows, new] - values[rows, old] for values in halves]).sort(dim=0).values
    shape = gallery.shape_agreement[old, new]
    narrow = torch.stack((loss, least, most, shape), dim=-1)
    edge = torch.cat((difference, present, narrow), dim=-1)
    moments = (edge.mean(1), edge.amin(1), edge.amax(1))
    half_agreement = ((least > 0) & (most > 0)).float().mean(1)
    weak_votes = (most < 0).float().mean(1)
    context = torch.cat((*moments, torch.stack((half_agreement, weak_votes), dim=1),
                         edge.new_full((len(rows), 1), rows.shape[1] - 2)), dim=1)
    return Panel(rows, old, new, edge, context, loss.mean(1))


def score_gallery(source, query, gallery, args):
    fits, baseline = score_block(MOUSE, source, query, gallery.images, gallery.signatures,
                                 gallery.energy, gallery.device, args)
    graph = shortlist(fits, baseline, gallery.energy, gallery.ink)
    stock = normalize(fits["v9"])
    del fits
    halves = half_scores(source, query, args.seed + 5)
    cycles, lookup = cycle_rows(graph, baseline)
    return graph, baseline, stock, halves, cycles, lookup


def stratified_cycles(rows, baseline, stock, cap, rng):
    # Sample each v9-cost quartile for an OCF fit with both plausible and weak swaps.
    original = stock[rows, baseline[rows]]
    swapped = stock[rows, baseline[rows.roll(-1, dims=1)]]
    ordered = (original - swapped).mean(1).argsort().cpu().numpy()
    take = [rng.choice(group, size=min(len(group), cap // 4), replace=False)
            for group in np.array_split(ordered, 4)]
    return rows[torch.as_tensor(np.concatenate(take), device=rows.device)]


def calibration_day(day, source, gallery, forbidden, args, rng):
    query, order = permute(source[day], rng, gallery.device)
    inner = {name: values for name, values in source.items() if name != day}
    graph, baseline, stock, halves, cycles, lookup = score_gallery(inner, query, gallery, args)
    report = {"training_days": list(inner), "all_cycles": {size: len(rows) for size, rows in cycles.items()},
              "calibrated": {}}
    parts = []
    used = []
    for size, all_rows in cycles.items():
        allowed = all_rows[~forbidden[order[all_rows]].any(1)]
        rows = stratified_cycles(allowed, baseline, stock, MAX_CALIBRATION_CYCLES[size], rng)
        if not len(rows):
            continue
        panel = feature_panel(rows, lookup, graph, baseline, stock, halves, gallery)
        truth = order[rows]
        predictions = torch.stack((panel.labels_old, panel.labels_new))
        exact, count = unique_pair_ocf(predictions, truth.expand_as(predictions), gallery.masks,
                                       gallery.scorer, gallery.device, 384)
        delta = exact[1] - exact[0]
        parts.append((panel, delta))
        used.append(truth.flatten())
        report["calibrated"][size] = {"cycles": len(rows), "exact_ordered_pairs": count,
                                      "positive_net_fraction": float((delta.sum(1) > 0).float().mean()),
                                      "mean_v9_objective_loss": float(panel.stock_loss.mean())}
    if not parts:
        raise ValueError("A source day needs labeled standalone exchange examples.")
    return parts, torch.cat(used).unique(), report


def fit_heads(source, gallery, forbidden, args, rng):
    panels, particular, report = [], [], {}
    for day in source:
        data, identities, report[day] = calibration_day(day, source, gallery, forbidden, args, rng)
        panels.extend(data)
        particular.append(identities)
        print(json.dumps({"calibration_day": day, **report[day]}), flush=True)
    edge = torch.cat([panel.edge.flatten(0, 1) for panel, _ in panels])
    edge_target = torch.cat([delta.flatten() for _, delta in panels])
    context = torch.cat([panel.context for panel, _ in panels])
    context_target = torch.cat([delta.mean(1) for _, delta in panels])
    edge_head = fit_ridge(edge, edge_target)
    cycle_head = fit_ridge(context, context_target)
    report["fit"] = {"exact_labeled_edges": len(edge_target), "exact_labeled_cycles": len(context_target),
                     "edge_features": edge.shape[1], "cycle_features": context.shape[1]}
    return edge_head, cycle_head, torch.cat(particular).unique(), report


def greedy(rows, scores, used, limit, *, positive_only=False):
    values, order = scores.sort(descending=True, stable=True)
    values, order = values.cpu().numpy(), order.cpu().numpy()
    occupied = used.copy()
    taken = []
    for index, score in zip(order, values, strict=True):
        if len(taken) >= limit or (positive_only and score <= 0):
            break
        vertices = rows[index]
        if not occupied[vertices].any():
            occupied[vertices] = True
            taken.append(index)
    return np.asarray(taken, dtype=np.int64)


def select(baseline, panels, edge_head, cycle_head):
    # The arms spend exactly the same number of 2- and 3-row swaps on the same pool.
    # Accept positive expected changes from the proposed method and match its quota in the controls.
    predictions = {name: baseline.clone() for name in ARMS}
    occupied = {name: np.zeros(len(baseline), dtype=bool) for name in (CYCLE, EDGE, GAP)}
    picked = {name: {} for name in occupied}
    for size in (3, 2):
        panel = panels[size]
        host_rows = panel.rows.cpu().numpy()
        scores = {CYCLE: cycle_head.predict(panel.context), EDGE: edge_head.predict(panel.edge).mean(1),
                  GAP: -panel.stock_loss}
        proposals = {name: greedy(host_rows, matrix, occupied[name], SWAP_LIMITS[size], positive_only=(name == CYCLE))
                     for name, matrix in scores.items()}
        matched_count = min(map(len, proposals.values()))
        for name, index in proposals.items():
            chosen = index[:matched_count]
            picked[name][size] = chosen
            occupied[name][host_rows[chosen]] = True
            predictions[name][panel.rows[chosen]] = panel.labels_new[chosen]
    if any(len(values.unique()) != len(baseline) for values in predictions.values()):
        raise ValueError("Local rotations must preserve every glyph exactly once.")
    return predictions, picked


def comparisons(values):
    return {**{name: paired(scores, values[BASE]) for name, scores in values.items()},
            "cycle_vs_additive": paired(values[CYCLE], values[EDGE]),
            "cycle_vs_v9_gap": paired(values[CYCLE], values[GAP])}


def evaluate(held, source, observed, gallery, primary, forbidden, args, rng):
    edge_head, cycle_head, used_fit, training = fit_heads(source, gallery, forbidden, args, rng)
    query, order = permute(observed, rng, gallery.device)
    graph, baseline, stock, halves, cycles, lookup = score_gallery(source, query, gallery, args)
    panels = {size: feature_panel(rows, lookup, graph, baseline, stock, halves, gallery) for size, rows in cycles.items()}
    predictions, choices = select(baseline, panels, edge_head, cycle_head)

    # The held-day correspondence is revealed to the scorer only after the complete assignments are fixed.
    all_glyphs = torch.arange(len(baseline), device=gallery.device)
    unique_labeled = all_glyphs[~torch.isin(all_glyphs, used_fit)]
    primary_rows = order.argsort()[primary]
    secondary_rows = order.argsort()[unique_labeled]
    all_scores = torch.stack([value[secondary_rows] for value in predictions.values()])
    values, num_pairs = unique_pair_ocf(all_scores, unique_labeled.expand_as(all_scores), gallery.masks,
                                       gallery.scorer, gallery.device, 384)
    positions = torch.searchsorted(unique_labeled, primary)
    if not torch.equal(unique_labeled[positions], primary):
        raise ValueError("Primary held glyphs must exclude every exact-OCF calibration reference.")
    scored = {name: row[positions].cpu() for name, row in zip(predictions, values, strict=True)}
    secondary = {name: row[None].cpu() for name, row in zip(predictions, values, strict=True)}
    score_rows = torch.full_like(order, -1)
    score_rows[secondary_rows] = torch.arange(len(secondary_rows), device=order.device)
    details = {}
    for index, (name, selected) in enumerate(predictions.items()):
        moved = selected != baseline
        finished = {}
        for size, selection in choices.get(name, {}).items():
            observed_rows = score_rows[panels[size].rows[selection]]
            complete = observed_rows[(observed_rows >= 0).all(dim=1)]
            gains = (values[index] - values[0])[complete].sum(dim=1)
            finished[size] = {"complete_unlabeled_in_calibration_cycles": len(complete),
                              "mean_net_ocf_gain_per_cycle": float(gains.mean()) if len(gains) else None,
                              "positive_gain_fraction": float((gains > 0).float().mean()) if len(gains) else None}
        details[name] = {"moved_rows": int(moved.sum()), "moved_primary_rows": int(moved[primary_rows].sum()),
                         "full_gallery_identity_top1": float((selected == order).float().mean()),
                         "distinct_glyphs": int(selected.unique().numel()),
                         "selected_cycles": {size: len(selection) for size, selection in choices.get(name, {}).items()},
                         "complete_cycle_realized_gains": finished}
    report = {"held_day": held, "source_days": list(source), "calibration": training,
              "calibration_reference_count": len(used_fit), "source_untouched_scoring_glyphs": len(unique_labeled),
              "eligible_cycles": {size: len(rows) for size, rows in cycles.items()},
              "stock_cycle_loss_minimum": {size: float(panel.stock_loss.min()) if len(panel.rows) else None
                                           for size, panel in panels.items()}, "assignments": details,
              "exact_unique_ordered_pairs": num_pairs,
              "primary": comparisons({name: row[None] for name, row in scored.items()}),
              "all_exact_calibration_disjoint_glyphs": comparisons(secondary)}
    return report, scored


def held_glyphs(size, seed, count, device):
    original = previous_mice2_cohorts(size)
    before_scale = np.setdiff1d(np.arange(size), original)
    multi_previous = np.random.default_rng(41721).choice(before_scale, size=256, replace=False)
    excluded = np.union1d(original, multi_previous)
    eligible = np.setdiff1d(np.arange(size), excluded)
    if len(eligible) < count:
        raise ValueError("Earlier Task 2 cohorts must leave enough fresh primary scoring glyphs.")
    selected = np.sort(np.random.default_rng(seed).choice(eligible, size=count, replace=False))
    return torch.as_tensor(selected, device=device), len(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-days", nargs="+", choices=SESSIONS, default=("session1", "session3"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--split-seed", type=int, default=52873)
    parser.add_argument("--ocf-glyphs", type=int, default=768)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.set_defaults(encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a private Kaggle T4 with labeled competition recordings mounted.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    device = torch.device("cuda:0")
    images = load_images(root, device)
    masks = images >= 128
    signature, energy = contour_signatures(images)
    gallery = CycleGallery(images, masks, signature, energy, masks.float().mean((1, 2)), official_scorer(root),
                           signature @ signature.T)
    blocks = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(blocks) != SESSIONS or any(block.shape[1] != 4 for block in blocks.values()):
        raise ValueError("Require four complete labeled days of four measured repeats per glyph.")
    primary, excluded = held_glyphs(len(images), args.split_seed, args.ocf_glyphs, device)
    forbidden = torch.zeros(len(images), dtype=torch.bool, device=device)
    forbidden[primary] = True
    report = {"mouse": MOUSE, "seed": args.seed, "split_seed": args.split_seed,
              "library_size": len(images), "repeats": 4, "primary_glyphs_per_recording": len(primary),
              "prior_primary_glyphs_excluded": excluded, "calibration_cycle_limit_per_day": MAX_CALIBRATION_CYCLES,
              "maximum_swaps_per_length": SWAP_LIMITS, "formal_test_used": False,
              "protocol": "Open the same known closed gallery in each anonymized labeled recording. Propose only two- "
                          "and three-row rotations using labels currently owned by other v14 v9 assignment rows. "
                          "Fit edge and one-cycle OCF models on the same exact labeled exchanges from inner source "
                          "rotations. The four query repeats supply the full v9 scores and two independent half EB "
                          "views. The cycle model may only spend on positive predicted gains. Additive and low-v9-loss "
                          "controls spend exactly the same number of two- and three-row exchanges. All arms retain "
                          "the same 2,748-glyph inventory. Source calibration excludes primary reference glyphs. All "
                          "scoring uses held-day labels only after the complete assignments are locked.",
              "recordings": {}}
    pooled = {name: [] for name in ARMS}
    for held in dict.fromkeys(args.held_days):
        source = {day: values for day, values in blocks.items() if day != held}
        rng = np.random.default_rng(np.random.SeedSequence((args.split_seed, SESSIONS.index(held))))
        details, measured = evaluate(held, source, blocks[held], gallery, primary, forbidden, args, rng)
        for name, pair in measured.items():
            pooled[name].append(pair)
        report["recordings"][held] = details
        report["primary_paired_by_glyph"] = comparisons({name: torch.stack(rows) for name, rows in pooled.items()})
        report["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"held": held, "compared": details["primary"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
