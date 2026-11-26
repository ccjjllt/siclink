"""Test whether local glyph parts nominate useful new Task 3 candidates.

Learn visual codebooks on overlapping patches of the known training gallery.
Both the contour control and the spike-to-part readouts use three repeats per
glyph from each source session. The fourth repeat is reserved before any fit.
Compare equal-size shortlists, with part codes shuffled within ink and contour
energy strata as a control. Exact OCF references are consulted after ranking.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_contour_retrieval import (_fit_readout, _response_features, contour_retrieval_scores,
                                     fit_contour_retrieval)
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_mbr import contour_signatures
from glyph_retrieval import _standardize
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import unique_pair_ocf
from ocf_training import directional_fields


SESSIONS = ("session1", "session2", "session3", "session4")
PATCHES = ((slice(None), slice(0, 40)), (slice(None), slice(24, 64)),
           (slice(0, 40), slice(None)), (slice(24, 64), slice(None)))
METHODS = ("contour", "part_union", "shuffled_union")


def patch_descriptors(masks):
    output = [[] for _ in PATCHES]
    for masks_batch in masks.split(96):
        canvas = masks_batch.float().unsqueeze(1)
        fields = [directional_fields(canvas, sigma) for sigma in (1, 2, 4)]
        for values, (height, width) in zip(output, PATCHES):
            ink = F.adaptive_avg_pool2d(canvas[..., height, width], (8, 8)).flatten(1)
            parts = [0.35 * F.normalize(ink)]
            for field in fields:
                crop = field[..., height, width]
                signed = F.adaptive_avg_pool2d(crop, (8, 8)).flatten(1)
                energy = F.adaptive_avg_pool2d(crop.square(), (6, 6)).sqrt().flatten(1)
                parts.extend((0.50 * F.normalize(signed), 0.80 * F.normalize(energy)))
            values.append(F.normalize(torch.cat(parts, dim=1)))
    return [torch.cat(values) for values in output]


def fit_codewords(descriptors, size, generator):
    initial = torch.randperm(len(descriptors), generator=generator)[:size].to(descriptors.device)
    centers = descriptors[initial].clone()
    for _ in range(12):
        labels = (descriptors @ centers.T).argmax(dim=1)
        updates = torch.zeros_like(centers).index_add_(0, labels, descriptors)
        occupied = updates.square().sum(dim=1, keepdim=True) > 0
        centers = torch.where(occupied, F.normalize(updates), centers)
    return ((descriptors @ centers.T) / 0.07).softmax(dim=1)


def permute_part_codes(codes, ink, energy, generator):
    """Keep per-region code frequencies, ink strata and overall contour energy."""
    ink_edges = torch.quantile(ink, torch.arange(1, 8, device=ink.device) / 8)
    energy_edges = torch.quantile(energy, torch.arange(1, 4, device=energy.device) / 4)
    strata = 4 * torch.bucketize(ink, ink_edges) + torch.bucketize(energy, energy_edges)
    shuffled = torch.empty_like(codes)
    for region in range(len(PATCHES)):
        for group in strata.unique():
            rows = (strata == group).nonzero()[:, 0]
            permutation = torch.randperm(len(rows), generator=generator).to(rows.device)
            ordered = rows[permutation]
            shuffled[ordered, region] = codes[ordered.roll(1), region]
    return shuffled


def project_counts(contour, counts, session):
    features = _response_features(counts, contour.view, contour.n_bins)
    return _standardize(features, *contour.moments[session]) @ contour.projection


def fit_part_readouts(contour, training, codes):
    projected = {name: project_counts(contour, counts, name) for name, counts in training.items()}
    targets = codes.sqrt().flatten(1)
    pooled = _fit_readout(tuple(projected.values()), targets, ridge=0.12)
    local = {name: _fit_readout((values,), targets, ridge=0.25) for name, values in projected.items()}
    return pooled, local


def part_logits(contour, readout, counts, session, codes, shuffled):
    pooled, local = readout
    projected = project_counts(contour, counts, session).mean(dim=1)
    estimate = 0.80 * pooled.predict(projected) + 0.20 * local[session].predict(projected)
    mass = estimate.reshape(len(counts), len(PATCHES), codes.shape[-1]).clamp_min(0.01).square()
    probabilities = mass / mass.sum(dim=2, keepdim=True)
    base = codes.mean(dim=0).clamp_min(1e-5)
    evidence = 0.25 + 0.75 * (probabilities / base)
    outputs = []
    for gallery in (codes, shuffled):
        local_matches = evidence.transpose(0, 1) @ gallery.permute(1, 2, 0)
        outputs.append(local_matches.clamp_min(1e-5).log().mean(dim=0))
    return outputs


def standardize_rows(values):
    mean, deviation = values.mean(dim=1, keepdim=True), values.std(dim=1, keepdim=True, unbiased=False)
    return (values - mean) / deviation.clamp_min(1e-5)


def nominate(contour_scores, real_parts, shuffled_parts, budget, additional):
    baseline = contour_scores.topk(budget, dim=1).indices
    score = standardize_rows(contour_scores)
    retained = baseline[:, :budget - additional]
    pools = {"contour": baseline}
    decisions = {"contour": baseline[:, 0]}
    for name, parts in zip(METHODS[1:], (real_parts, shuffled_parts)):
        part_score = standardize_rows(parts)
        discovery = 0.35 * score + 0.65 * part_score
        discovery.scatter_(1, baseline, -torch.inf)
        novel = discovery.topk(additional, dim=1).indices
        pool = torch.cat((retained, novel), dim=1)
        ranking = (0.80 * score + 0.20 * part_score).gather(1, pool)
        pools[name] = pool
        decisions[name] = pool.gather(1, ranking.argmax(dim=1, keepdim=True))[:, 0]
    return pools, decisions


def compare_measurements(values, baseline):
    delta = (values - baseline).mean(axis=0)
    return {"ocf": float(values.mean()), "paired_delta_vs_contour": float(delta.mean()),
            "paired_glyph_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "scored_rows": int(values.size)}


def calculate_exact(pools, decisions, sample_rows, glyphs, masks, scorer, oracle_count):
    prediction = torch.stack([decisions[name][sample_rows] for name in METHODS])
    best_targets = glyphs.expand_as(prediction)
    shortlist = torch.stack([pools[name][sample_rows[:oracle_count]] for name in METHODS])
    shortlist_targets = glyphs[None, :oracle_count, None].expand_as(shortlist)
    values, unique_pairs = unique_pair_ocf(torch.cat((prediction.flatten(), shortlist.flatten())),
                                          torch.cat((best_targets.flatten(), shortlist_targets.flatten())),
                                          masks, scorer, torch.device("cpu"), 256)
    best, oracle = values.split((prediction.numel(), shortlist.numel()))
    best = best.reshape(prediction.shape).numpy()
    oracle = oracle.reshape(shortlist.shape).max(dim=2).values.numpy()
    return dict(zip(METHODS, best)), dict(zip(METHODS, oracle)), unique_pairs


def ranking_statistics(pools, decisions, labels, additional):
    truth = labels[:, None]
    statistics = {}
    for name in METHODS:
        matched = pools[name] == truth
        statistics[name] = {"top1": float((decisions[name] == labels).float().mean()),
                            "shortlist_recall": float(matched.any(dim=1).float().mean()),
                            "tail_hit_rate": float(matched[:, -additional:].any(dim=1).float().mean()),
                            "shortlist_changed_top1_fraction": float((decisions[name] != decisions["contour"])
                                                                    .float().mean())}
    return statistics


def block_summary(results):
    stacked = {name: np.stack(results[name]) for name in METHODS}
    scores = {name: compare_measurements(values, stacked["contour"]) for name, values in stacked.items()}
    delta = (stacked["part_union"] - stacked["shuffled_union"]).mean(axis=0)
    scores["part_union"].update(paired_delta_vs_shuffled=float(delta.mean()),
                                paired_se_vs_shuffled=float(delta.std(ddof=1) / np.sqrt(len(delta))))
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=SESSIONS, default=("session2", "session4"))
    parser.add_argument("--ocf-images", type=int, default=64)
    parser.add_argument("--oracle-images", type=int, default=8)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--part-quota", type=int, default=4)
    parser.add_argument("--codes-per-part", type=int, default=32)
    parser.add_argument("--repeat-seed", type=int, default=419)
    parser.add_argument("--seed", type=int, default=6143)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_part_discovery.json"))
    args = parser.parse_args()
    if not 2 <= args.oracle_images <= args.ocf_images or not 0 < args.part_quota < args.budget:
        parser.error("Reserve valid shortlist quotas and at least two references for each OCF comparison.")
    if args.codes_per_part < 2:
        parser.error("Choose at least two visual codes per patch.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the pilot on Kaggle CPU with competition data mounted.")
    torch.set_num_threads(4)
    device = torch.device("cpu")
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    masks = images >= 128
    if max(args.budget + args.part_quota, args.codes_per_part, args.ocf_images) > len(masks):
        parser.error("The requested codebook, candidate and metric settings must fit the known gallery.")
    signature, energy = contour_signatures(masks)
    ink = masks.float().mean(dim=(1, 2))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    codes = torch.stack([fit_codewords(region, args.codes_per_part, generator)
                         for region in patch_descriptors(masks)], dim=1)
    shuffled = permute_part_codes(codes, ink, energy, generator)
    rng = np.random.default_rng(args.seed + 1)
    glyphs = torch.as_tensor(rng.choice(len(masks), args.ocf_images, replace=False))
    scorer = official_scorer(root)
    print(f"Prepared {len(masks)} gallery masks and {args.codes_per_part} codes per region", flush=True)

    output = {"seed": args.seed, "repeat_seed": args.repeat_seed, "device": str(device),
              "complete_gallery_size": len(masks), "code_counts_per_patch": args.codes_per_part,
              "budget": args.budget, "part_quota": args.part_quota, "ocf_images_per_block": len(glyphs),
              "oracle_images_per_block": args.oracle_images, "folds": [],
              "control": "Contour-only stable-axis ridge; Task 3 same-session pooled and local blend.",
              "method": "Part compositions are fit from seen training glyph images and separate measured repeats. "
                        "Replace the contour tail with equally sized nominations outside its original shortlist. "
                        "Shuffled part assignments preserve ink and energy strata. OCF-best is diagnostic only."}
    top1_results = {name: [] for name in METHODS}
    oracle_results = {name: [] for name in METHODS}
    for mouse in dict.fromkeys(args.mice):
        recordings = grouped_blocks(root, catalog, mouse, device, len(images))
        training, held = {}, {}
        for index, (name, counts) in enumerate(recordings.items()):
            if counts.shape[1] != 4:
                raise ValueError("The Task 3 proxy requires the four original labeled repeats in each session.")
            training[name], held[name] = split_repeats(counts, args.repeat_seed + index)
        del recordings
        print(f"{mouse}: fit contour and part responses on three trials per glyph", flush=True)
        contour = fit_contour_retrieval(training, signature, energy, task="task3")
        readout = fit_part_readouts(contour, training, codes)
        del training
        for session in dict.fromkeys(args.sessions):
            fold_start = time.monotonic()
            permutation = torch.as_tensor(rng.permutation(len(masks)))
            query = held[session][permutation]
            stable = contour_retrieval_scores(contour, query, session)["contour_ridge"]
            real, randomized = part_logits(contour, readout, query, session, codes, shuffled)
            pools, choices = nominate(stable, real, randomized, args.budget, args.part_quota)
            rows = permutation.argsort()[glyphs]
            top1, best_of_pool, pairs = calculate_exact(pools, choices, rows, glyphs, masks, scorer, args.oracle_images)
            for name in METHODS:
                top1_results[name].append(top1[name])
                oracle_results[name].append(best_of_pool[name])
            fold = {"mouse": mouse, "session": session, "exact_pairs": pairs,
                    "full_gallery": ranking_statistics(pools, choices, permutation, args.part_quota),
                    "exact_top1": block_summary({name: [items] for name, items in top1.items()}),
                    "reference_best_of_fixed_budget": block_summary({name: [items] for name, items in best_of_pool.items()}),
                    "seconds": time.monotonic() - fold_start}
            print(json.dumps(fold), flush=True)
            output["folds"].append(fold)
            output["summary"] = {"exact_top1": block_summary(top1_results),
                                 "reference_best_of_fixed_budget": block_summary(oracle_results)}
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            del stable, real, randomized, query
        del contour, readout, held
    print(json.dumps(output["summary"]), flush=True)


if __name__ == "__main__":
    main()
