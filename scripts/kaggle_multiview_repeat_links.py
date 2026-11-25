"""Measure reproducible spike-only cross-block links at real Task 1 repeat counts."""

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials, split_glyphs
from glyph_multiview_alignment import fit_multiview, pair_similarity, reciprocal_matches


SESSIONS = ("session1", "session2", "session3", "session4")
HALF_PAIRS = ("aa", "bb", "ab", "ba")
REPLICATED = ("aa_and_bb", "ab_and_ba", "either_disjoint_pair", "all_four")


@dataclass
class RepeatEmbeddings:
    a: torch.Tensor
    b: torch.Tensor
    pooled: torch.Tensor


def encode_block(model, block, counts, repeats, rng):
    """Use one independent query permutation and two non-overlapping repeat halves."""
    if counts.ndim != 3 or counts.shape[1] != repeats or repeats % 2 or len(counts) < 2:
        raise ValueError(f"Expected {repeats} grouped repeats and at least two queries in {block}")
    permutation = torch.as_tensor(rng.permutation(len(counts)), device=counts.device)
    order = torch.as_tensor(rng.random((len(counts), repeats)).argsort(axis=1), device=counts.device)
    shuffled = counts[permutation].gather(1, order[:, :, None].expand(-1, -1, counts.shape[2]))
    a, b = shuffled.split(repeats // 2, dim=1)
    return RepeatEmbeddings(model.encode(block, a), model.encode(block, b), model.encode(block, shuffled)), permutation


def reciprocal_map(left, right):
    scores = pair_similarity(left, right)
    rows, columns, _ = reciprocal_matches(scores)
    result = torch.full((len(left),), -1, dtype=torch.long, device=left.device)
    result[rows] = columns
    return result


def intersect(left, right):
    return torch.where((left >= 0) & (left == right), left, -1)


def merge_confirmed(direct, crossed, n_target):
    """Retain either disjoint confirmation and abstain on conflicting endpoints."""
    conflict = (direct >= 0) & (crossed >= 0) & (direct != crossed)
    merged = torch.where(direct >= 0, direct, crossed)
    merged[conflict] = -1
    columns = merged[merged >= 0]
    ambiguous = torch.bincount(columns, minlength=n_target) > 1
    merged[(merged >= 0) & ambiguous[merged.clamp_min(0)]] = -1
    return merged, {"conflicting_source_rows": int(conflict.sum()), "ambiguous_target_rows": int(ambiguous.sum())}


def measure_links(mapping, n_target, left_labels, right_labels):
    valid = mapping >= 0
    links = int(valid.sum())
    result = {"links": links, "source_coverage": links / len(mapping), "target_coverage": links / n_target}
    if left_labels is not None:
        correct = int((left_labels[valid] == right_labels[mapping[valid]]).sum())
        result.update(correct=correct, precision=correct / links if links else None)
    return result


def link_reproducibility(results, left, right, intersection):
    a, b, both = (results[name]["links"] for name in (left, right, intersection))
    union = a + b - both
    return {"intersection": both, "union": union, "jaccard": both / union if union else None,
            "fraction_first_confirmed": both / a if a else None,
            "fraction_second_confirmed": both / b if b else None}


def compare(left_block, right_block, left, right, left_labels, right_labels):
    mappings = {name: reciprocal_map(getattr(left, name[0]), getattr(right, name[1])) for name in HALF_PAIRS}
    direct = intersect(mappings["aa"], mappings["bb"])
    crossed = intersect(mappings["ab"], mappings["ba"])
    confirmed, conflicts = merge_confirmed(direct, crossed, len(right.a))
    mappings.update(aa_and_bb=direct, ab_and_ba=crossed, either_disjoint_pair=confirmed,
                    all_four=intersect(direct, crossed), pooled=reciprocal_map(left.pooled, right.pooled))
    results = {name: measure_links(value, len(right.a), left_labels, right_labels)
               for name, value in mappings.items()}
    row = {"left": left_block, "right": right_block, "same_mouse": left_block[0] == right_block[0],
           "source_rows": len(left.a), "target_rows": len(right.a), "rules": results, "abstention": conflicts,
           "direct_reproducibility": link_reproducibility(results, "aa", "bb", "aa_and_bb"),
           "cross_reproducibility": link_reproducibility(results, "ab", "ba", "ab_and_ba")}
    return row, {name: mappings[name] for name in REPLICATED}


def summarize_pairs(pairs, labeled):
    summary = {}
    groups = ("all", pairs), ("same_mouse", [row for row in pairs if row["same_mouse"]]), (
        "cross_mouse", [row for row in pairs if not row["same_mouse"]])
    for name, group in groups:
        if not group:
            continue
        n_source = sum(row["source_rows"] for row in group)
        n_target = sum(row["target_rows"] for row in group)
        info = {"block_pairs": len(group), "rules": {}}
        for rule in (*HALF_PAIRS, *REPLICATED, "pooled"):
            links = sum(row["rules"][rule]["links"] for row in group)
            record = {"links": links, "source_coverage": links / n_source, "target_coverage": links / n_target}
            if labeled:
                correct = sum(row["rules"][rule]["correct"] for row in group)
                record.update(correct=correct, precision=correct / links if links else None)
            info["rules"][rule] = record
        info["direct_reproducibility"] = link_reproducibility(info["rules"], "aa", "bb", "aa_and_bb")
        info["cross_reproducibility"] = link_reproducibility(info["rules"], "ab", "ba", "ab_and_ba")
        summary[name] = info
    return summary


def evaluate(embeddings, labels=None):
    pairs = []
    coverage = {block: {name: value.a.new_zeros((len(value.a),), dtype=torch.long) for name in REPLICATED}
                for block, value in embeddings.items()}
    for (left_key, left), (right_key, right) in combinations(embeddings.items(), 2):
        left_labels, right_labels = (labels[left_key], labels[right_key]) if labels is not None else (None, None)
        result, replicated = compare(left_key, right_key, left, right, left_labels, right_labels)
        pairs.append(result)
        for rule, mapping in replicated.items():
            valid = mapping >= 0
            coverage[left_key][rule][valid] += 1
            coverage[right_key][rule][mapping[valid]] += 1
    by_block = [{"block": block, "rules": {
        name: {"queries_with_link": int((values > 0).sum()), "queries": len(values),
               "coverage": float((values > 0).float().mean()), "mean_linked_peers": float(values.float().mean())}
        for name, values in rules.items()}}
        for block, rules in coverage.items()]
    return {"pairs": pairs, "summary": summarize_pairs(pairs, labels is not None), "by_block": by_block}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=SESSIONS, default=SESSIONS)
    parser.add_argument("--view", choices=("evoked-total", "evoked-multiscale", "early-late"), default="evoked-total")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.35)
    parser.add_argument("--held-glyphs", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/multiview_repeat_links.json"))
    args = parser.parse_args()
    if (len(set(args.mice)) * len(set(args.sessions)) < 2 or not 2 <= args.held_glyphs <= 2745
            or args.dimension < 1 or args.ridge <= 0):
        parser.error("Choose at least two blocks, positive model settings and a valid held split.")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Use Kaggle CPU or CUDA.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    n_glyphs = len(load_images(root, torch.device("cpu")))
    fit, held = split_glyphs(n_glyphs, args.held_glyphs, args.seed, device)
    recordings = {}
    for mouse in dict.fromkeys(args.mice):
        groups = grouped_blocks(root, catalog, mouse, device, n_glyphs)
        recordings.update({(mouse, session): counts for session, counts in groups.items() if session in args.sessions})
    model = fit_multiview({block: counts[fit] for block, counts in recordings.items()}, view=args.view,
                          dimension=args.dimension, ridge=args.ridge)
    print(f"Fitted {len(recordings)} GCCA views on {len(fit):,} training glyphs", flush=True)

    held_rng = np.random.default_rng(args.seed + 1)
    held_blocks, held_labels = {}, {}
    for block, counts in recordings.items():
        held_blocks[block], held_labels[block] = encode_block(model, block, counts[held], 4, held_rng)
    del recordings, groups, counts
    held_result = evaluate(held_blocks, held_labels)
    output = {"device": str(device), "seed": args.seed, "view": args.view, "dimensions": len(model.spectrum),
              "ridge": args.ridge, "fit_glyphs": len(fit), "held_glyphs": len(held),
              "protocol": "GCCA and its feature normalization use fit glyphs only. Each block is independently row-shuffled, and each row's repeats are partitioned at random. AA versus BB and AB versus BA each use disjoint observations on both sides. The four comparisons are not jointly independent. Official Task 1 reports reproducibility and coverage only, with no inferred correctness, images, labels, IDs or ordering features.",
              "held_training_2_2": held_result, "seconds": time.monotonic() - started}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"held_training_2_2": held_result["summary"]}), flush=True)

    test_rng = np.random.default_rng(args.seed + 2)
    test_blocks = {}
    for entry in catalog["blocks"]:
        block = (entry["mouse"], entry["session"])
        if entry["task"] == "task1" and block in model.projections:
            counts = read_trials(root, catalog, entry, device).counts
            test_blocks[block], _ = encode_block(model, block, counts, 20, test_rng)
            print(f"{block}: encoded two disjoint ten-repeat halves for {len(counts):,} official queries", flush=True)
            del counts
    if set(test_blocks) != set(held_blocks):
        raise ValueError("Expected official Task 1 blocks for every held training block")
    test_blocks = {block: test_blocks[block] for block in held_blocks}
    test_result = evaluate(test_blocks)
    output["official_task1_10_10"] = test_result
    output["seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"official_task1_10_10": test_result["summary"]}), flush=True)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
