"""Measure v11/v12 mice1 Task 1 stability on the official grouped repeats."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_metric_ranker import pack_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_mbr import contour_signatures
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_exact_rerank_validation import split_calibration
from kaggle_task1_test_repeatability import fit_signal_space
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank


MOUSE = "mice1"
MODEL_SEED = 211
ANCHOR_OCF = 4.0
METRIC_GLYPHS = 96
NOISE_GLYPHS = 384


@dataclass
class Reading:
    source_nominations: dict
    candidates: object
    v8_scores: torch.Tensor
    reranker_scores: torch.Tensor
    decisions: dict


@torch.no_grad()
def decode(model, ranker, likelihood, counts, session, rates):
    scores = atlas_scores(model, likelihood, counts, session, rates=rates)
    proposals = pack_atlas_candidates(scores, NOMINATIONS, baseline="v8")
    source = {name: scores[name].topk(n, dim=1).indices for name, n in NOMINATIONS.items()}
    reranked = ranker.rank(proposals, anchor_ocf=ANCHOR_OCF)
    result = {"v8_anchor": scores["v8"].argmax(dim=1),
              "v11_v12_reranker": proposals.indices.gather(1, reranked.argmax(dim=1, keepdim=True))[:, 0]}
    return Reading(source, proposals, scores["v8"], reranked, result)


def paired_indices(left, right, left_valid=None, right_valid=None):
    same = left[:, :, None] == right[:, None, :]
    if left_valid is not None:
        same &= left_valid[:, :, None]
    if right_valid is not None:
        same &= right_valid[:, None, :]
    return same


def overlap(same, left_size, right_size):
    intersection = same.any(dim=-1).sum(dim=1)
    union = left_size + right_size - intersection
    return {"mean_intersection": float(intersection.float().mean()), "mean_union": float(union.float().mean()),
            "mean_jaccard": float((intersection / union.clamp_min(1)).mean()),
            "rows_with_any_overlap": float((intersection > 0).float().mean())}


def candidate_overlap(left, right, same):
    first, second = left.candidates, right.candidates
    result = {name: overlap(paired_indices(left.source_nominations[name], right.source_nominations[name]), n, n)
              for name, n in NOMINATIONS.items()}
    result["deduplicated_union"] = overlap(same, first.valid.sum(dim=1), second.valid.sum(dim=1))
    return result


def correlations(left, right, valid=None):
    if valid is None:
        a, b = left - left.mean(dim=1, keepdim=True), right - right.mean(dim=1, keepdim=True)
    else:
        size = valid.sum(dim=1, keepdim=True).clamp_min(1)
        first, second = (item.masked_fill(~valid, 0) for item in (left, right))
        first_mean, second_mean = first.sum(dim=1, keepdim=True) / size, second.sum(dim=1, keepdim=True) / size
        a, b = (first - first_mean).masked_fill(~valid, 0), (second - second_mean).masked_fill(~valid, 0)
    scale = a.norm(dim=1) * b.norm(dim=1)
    usable = scale > 1e-10
    scores = (a * b).sum(dim=1)[usable] / scale[usable]
    return {"mean": float(scores.mean()) if len(scores) else None, "groups": len(scores)}


def common_rank_scores(left, right, match):
    support = match.any(dim=-1)
    right_columns = match.long().argmax(dim=-1)
    first = left.candidates
    arrays = {"v8_anchor": (left.v8_scores.gather(1, first.indices), right.v8_scores.gather(1, first.indices)),
              "v11_v12_reranker": (left.reranker_scores, right.reranker_scores.gather(1, right_columns))}
    result = {}
    for name, (a, b) in arrays.items():
        a, b = a.masked_fill(~support, -torch.inf), b.masked_fill(~support, -torch.inf)
        # Compute ranks within the common proposal set only. Ties use fixed column order.
        a_rank, b_rank = (scores.argsort(dim=1, descending=True, stable=True).argsort(dim=1).float()
                          for scores in (a, b))
        usable = support.sum(dim=1) >= 2
        result[name] = {"common_candidate_score_pearson": correlations(a, b, support),
                        "common_candidate_rank_spearman": correlations(a_rank, b_rank, support),
                        "restricted_to_common_winner_agreement": float(
                            (a[usable].argmax(dim=1) == b[usable].argmax(dim=1)).float().mean()) if bool(usable.any()) else None,
                        "groups_with_two_common_candidates": int(usable.sum())}
    result["v8_anchor"]["complete_atlas_score_pearson"] = correlations(left.v8_scores, right.v8_scores)
    return result


def contains(proposals, selected):
    return ((proposals.indices == selected[:, None]) & proposals.valid).any(dim=1)


@torch.no_grad()
def output_agreement(left, right, full, rendered, control):
    readings = (left, right, full)
    names = tuple(full.decisions)
    picked = torch.stack([torch.stack(tuple(reading.decisions.values())) for reading in readings])
    unique, inverse = torch.unique(picked, return_inverse=True)
    signatures, energy = contour_signatures(rendered[unique])
    features, magnitude = signatures[inverse], energy[inverse]

    def similarity(a, b, method, other_rows=None):
        first, second = features[a, method], features[b, method]
        mass_a, mass_b = magnitude[a, method], magnitude[b, method]
        if other_rows is not None:
            second, mass_b = second[other_rows], mass_b[other_rows]
        cosine = (first * second).sum(dim=-1).clamp(0, 1)
        ratio = torch.minimum(mass_a, mass_b) / torch.maximum(mass_a, mass_b)
        shared = cosine * ratio
        return 100 * shared / (1 + ratio - shared)

    result = {}
    for method, name in enumerate(names):
        choices = picked[:, method]
        left_equal, right_equal = choices[:2] == choices[2]
        half_equal = choices[0] == choices[1]
        first, second = contains(left.candidates, choices[2]), contains(right.candidates, choices[2])
        pair_similarity = similarity(0, 1, method)
        result[name] = {"half_top1_agreement": float(half_equal.float().mean()),
                        "half_vs_full20_agreement_left": float(left_equal.float().mean()),
                        "half_vs_full20_agreement_right": float(right_equal.float().mean()),
                        "matches_full20_in_both_halves": float((left_equal & right_equal).float().mean()),
                        "full20_winner_proposed_by_both_halves": float((first & second).float().mean()),
                        "full20_winner_proposed_by_either_half": float((first | second).float().mean()),
                        "halves_output_directional_proxy": float(pair_similarity.mean()),
                        "halves_output_proxy_when_different": float(pair_similarity[~half_equal].mean()) if bool((~half_equal).any()) else None,
                        "shuffled_groups_output_directional_proxy": float(similarity(0, 1, method, control).mean()),
                        "half_vs_full20_output_proxy_left": float(similarity(0, 2, method).mean()),
                        "half_vs_full20_output_proxy_right": float(similarity(1, 2, method).mean())}
    return result


@torch.no_grad()
def neural_fidelity(counts, trial_order, signal_space, rng):
    calibration, projection, variance = signal_space
    normalized, _ = standardize(feature_view(counts.cpu(), "evoked-total"), calibration)
    embedded = normalized @ projection
    order = trial_order.cpu()
    points = embedded.gather(1, order[:, :, None].expand_as(embedded))
    left, right = points[:, :10].mean(1), points[:, 10:].mean(1)
    left, right = left - left.mean(0), right - right.mean(0)
    corr = (left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-10)
    shrink = variance / (variance + 0.1)
    uncertainty = (1 + shrink) * 0.1
    report = {"split_half_correlation": float(corr),
              "same_group_identity_chance": 1 / len(points)}
    for name, a, b in (("left_to_right", left, right), ("right_to_left", right, left)):
        order = torch.as_tensor(rng.permutation(len(points)))
        truth = order.argsort()
        centers = b[order] * shrink
        gaussian = (a / uncertainty) @ centers.T - 0.5 * (centers.square() / uncertainty).sum(dim=1)
        cosine = F.normalize(a) @ F.normalize(b[order]).T
        comparisons = {}
        for key, values in (("gaussian", gaussian), ("cosine", cosine)):
            predicted = values.topk(5, dim=1).indices
            comparisons[key] = {"top1": float((predicted[:, 0] == truth).float().mean()),
                                "top5": float((predicted == truth[:, None]).any(dim=1).float().mean())}
        report[name] = comparisons
    return report


def split_groups(counts, rng):
    trial_order = torch.as_tensor(rng.random(counts.shape[:2]).argsort(axis=1), device=counts.device)
    split = counts.gather(1, trial_order[:, :, None].expand_as(counts))
    return trial_order, split[:, :10], split[:, 10:]


def inspect_session(model, ranker, likelihood, entry, counts, space, rendered, args):
    session = entry["session"]
    with torch.autocast(device_type="cuda"):
        rates = model.encoder.expected_counts(model.atlas.images, session, batch_size=192).float()
    full = decode(model, ranker, likelihood, counts, session, rates)
    output = {"groups": len(counts), "repeats": counts.shape[1], "partitions": [],
              "full20_mean_proposal_count": float(full.candidates.valid.sum(dim=1).float().mean()),
              "full20_ranker_changes_anchor": float((full.decisions["v11_v12_reranker"] != full.decisions["v8_anchor"]).float().mean())}
    for partition in range(args.partitions):
        sequence = np.random.SeedSequence((args.split_seed, SESSIONS.index(session), partition))
        rng = np.random.default_rng(sequence)
        order, half_a, half_b = split_groups(counts, rng)
        left = decode(model, ranker, likelihood, half_a, session, rates)
        right = decode(model, ranker, likelihood, half_b, session, rates)
        a, b = left.candidates, right.candidates
        matches = paired_indices(a.indices, b.indices, a.valid, b.valid)
        # A randomly ordered cycle makes the shape baseline an explicit derangement.
        random = torch.as_tensor(rng.permutation(len(counts)), device=counts.device)
        control = torch.empty_like(random)
        control[random] = random.roll(1)
        row = {"partition": partition, "neural_fidelity": neural_fidelity(counts, order, space, rng),
               "proposal_overlap": candidate_overlap(left, right, matches),
               "score_rank_stability": common_rank_scores(left, right, matches),
               "half_ranker_changes_anchor": [float((reading.decisions["v11_v12_reranker"] !=
                                                      reading.decisions["v8_anchor"]).float().mean())
                                               for reading in (left, right)],
               "output_stability": output_agreement(left, right, full, rendered, control)}
        output["partitions"].append(row)
    return output


def rollup(sessions):
    rows = [p for session in sessions.values() for p in session["partitions"]]
    directions = ("left_to_right", "right_to_left")
    result = {"neural_same_group_top1_gaussian": float(np.mean(
        [r["neural_fidelity"][d]["gaussian"]["top1"] for r in rows for d in directions])),
        "neural_whitened_correlation": float(np.mean([r["neural_fidelity"]["split_half_correlation"] for r in rows])),
        "v8_top16_jaccard": float(np.mean([r["proposal_overlap"]["v8"]["mean_jaccard"] for r in rows])),
        "proposal_union_jaccard": float(np.mean([r["proposal_overlap"]["deduplicated_union"]["mean_jaccard"] for r in rows]))}
    for name in ("v8_anchor", "v11_v12_reranker"):
        keys = rows[0]["output_stability"][name]
        observations = [row["output_stability"][name] for row in rows]
        result[name] = {key: float(np.mean([row[key] for row in observations if row[key] is not None]))
                        for key in keys if any(row[key] is not None for row in observations)}
        ranks = [row["score_rank_stability"][name]["common_candidate_rank_spearman"]["mean"] for row in rows]
        usable = [value for value in ranks if value is not None]
        result[name]["common_candidate_rank_spearman"] = float(np.mean(usable)) if usable else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seed", type=int, default=3811)
    parser.add_argument("--partitions", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_formal_rank_stability.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the frozen mice1 Task 1 stability comparison on Kaggle T4.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cuda:0")
    images = load_images(root, device)
    training = grouped_blocks(root, catalog, MOUSE, device, len(images))
    split = split_calibration(len(images), METRIC_GLYPHS, NOISE_GLYPHS, device, MODEL_SEED)
    print("Fit the exact mice1 v11/v12 training-only v8 atlas and anchored ranker", flush=True)
    ranker, likelihoods, nested, codepoints = nested_ranker(training, images, split, official_scorer(root), MODEL_SEED)
    model = fit_v8(training, images, MODEL_SEED)
    if not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("Expected identical final and nested public atlas ordering.")
    rendered = rendered_bank(model)
    space = {name: fit_signal_space(feature_view(counts.cpu(), "evoked-total")) for name, counts in training.items()}
    del training
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE
                      and entry["split"] == "test" and entry["task"] == "task1"), key=lambda x: x["session"])
    if tuple(entry["session"] for entry in entries) != SESSIONS:
        raise ValueError("The atlas stability study needs all four official mice1 Task 1 recordings.")
    output = {"mouse": MOUSE, "model_seed": MODEL_SEED, "split_seed": args.split_seed,
              "partitions_per_session": args.partitions, "metric_glyphs": METRIC_GLYPHS, "noise_glyphs": NOISE_GLYPHS,
              "anchor_ocf": ANCHOR_OCF, "nominations": NOMINATIONS, "atlas_size": len(rendered),
              "nested_fit": nested, "ranker": ranker.diagnostics, "sessions": {},
              "protocol": "Frozen training-only v11/v12 mice1 Task 1 policy, with grouped official responses split "
                          "within each anonymous sample. Independently shuffled candidate rows measure same-recording "
                          "neural fidelity using only the provided within-group relation. No cross-session correspondence "
                          "or sample IDs enter the diagnostics. The full group contains both halves. Two partitions "
                          "reuse observations and are descriptive. Contour utility compares model outputs and provides "
                          "no reconstruction OCF or hidden-label accuracy. This reruns the release policy without "
                          "reading or matching archived submission PNG bytes."}
    for entry in entries:
        session = entry["session"]
        counts = read_trials(root, catalog, entry, device).counts
        if counts.shape[:2] != (340, 20):
            raise ValueError("Expected 340 anonymously grouped examples with 20 trials per recording.")
        report = inspect_session(model, ranker, likelihoods[session], entry, counts, space[session], rendered, args)
        output["sessions"][session] = report
        output["summary"] = rollup(output["sessions"])
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, "summary": output["summary"]}), flush=True)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
