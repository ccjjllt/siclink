"""Test cross-recording response geometry for complete Task 2 matching."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import glyph_posterior
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics, task2_scores
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_v8_metric_validation import summarize


MOUSE = "mice2"
HELD_DAYS = ("session1", "session3")
SPLIT_SEED = 22177
DIMENSION = 48
NOISE_SHRINKAGE = 0.35
TEMPERATURE = 0.35
GEOMETRY_WEIGHT = 0.35
ARMS = ("v9", "geometry", "stratified_shuffle", "known_neighbor_identity")


def evoked_responses(counts):
    return standardize(feature_view(counts, "early-late"))[0]


def response_halves(counts):
    """Fit anonymous response geometry using repeated presentations in one domain."""
    residual = (counts - counts.mean(1, keepdim=True)).flatten(0, 1)
    noise = residual.T @ residual / (len(counts) * (counts.shape[1] - 1))
    noise.diagonal().add_(NOISE_SHRINKAGE * noise.diagonal().mean().clamp_min(0.02))
    factor = torch.linalg.cholesky(noise)
    # Alternating halves give every source recording the same weight.
    first = counts[:, ::2].mean(1)
    second = counts[:, 1::2].mean(1)
    whitened = torch.linalg.solve_triangular(factor, torch.cat((first, second)).T.contiguous(), upper=False).T
    first, second = whitened.chunk(2, dim=0)
    first, second = first - first.mean(0), second - second.mean(0)
    stable = first.T @ second
    _, axes = torch.linalg.eigh(stable + stable.T)
    leading = axes[:, -DIMENSION:]
    return first @ leading, second @ leading


def graph_vote(query_halves, gallery_halves, correspondence):
    """Compute the two symmetric Gram products while excluding self edges."""
    query_left, query_right = query_halves
    glyph_left, glyph_right = gallery_halves
    query_diagonal = (query_left * query_right).sum(1)
    source_diagonal = (glyph_left * glyph_right).sum(1)
    mapped = query_left @ (query_right.T @ correspondence)
    mapped.add_(query_right @ (query_left.T @ correspondence)).mul_(0.5)
    mapped.sub_(query_diagonal[:, None] * correspondence)
    vote = (mapped @ glyph_left) @ glyph_right.T
    vote.add_((mapped @ glyph_right) @ glyph_left.T).mul_(0.5)
    vote.sub_(mapped * source_diagonal)
    vote.sub_(vote.mean(0, keepdim=True))
    return logit_scale(vote)


def shuffle_columns(gallery_halves, masks, rng):
    """Break glyph correspondence in matched ink and response-energy strata."""
    response = (gallery_halves[0] + gallery_halves[1]).square().sum(1)
    ink = masks.float().mean(dim=(1, 2))
    permutation = torch.empty(len(ink), dtype=torch.long)
    for group in ink.argsort().tensor_split(8):
        for stratum in group[response[group].argsort()].tensor_split(4):
            order = stratum[torch.from_numpy(rng.permutation(len(stratum)))]
            permutation[order] = order.roll(1)
    if bool((permutation == torch.arange(len(ink))).any()):
        raise RuntimeError("The correspondence control must shuffle each glyph.")
    return permutation


def evaluate(held, recordings, images, masks, scorer, scored, args):
    started = time.monotonic()
    count = len(images)
    fit = {day: counts for day, counts in recordings.items() if day != held}
    rng = np.random.default_rng(SPLIT_SEED + int(held[-1]))
    order = torch.from_numpy(rng.permutation(count))
    query = recordings[held][order]
    torch.manual_seed(args.seed)
    base = logit_scale(task2_scores(MOUSE, fit, query, images, args))
    posterior = glyph_posterior(base, TEMPERATURE, balance=True)
    query_halves = response_halves(evoked_responses(query))
    gallery_halves = response_halves(torch.cat([evoked_responses(counts) for counts in fit.values()], dim=1))
    permuted = shuffle_columns(gallery_halves, masks, rng)
    scores = {"v9": base}
    scores["geometry"] = base + GEOMETRY_WEIGHT * graph_vote(query_halves, gallery_halves, posterior)
    scores["stratified_shuffle"] = base + GEOMETRY_WEIGHT * graph_vote(query_halves, gallery_halves,
                                                                      posterior[:, permuted])

    # Retrospective control. The query cannot vote for its own match even here.
    known = torch.zeros_like(posterior).scatter_(1, order[:, None], 1.0)
    scores["known_neighbor_identity"] = base + GEOMETRY_WEIGHT * graph_vote(query_halves, gallery_halves, known)
    choices = {name: assign_glyphs(logits) for name, logits in scores.items()}
    sample = order.argsort()[scored]
    outputs = torch.stack([choices[name][sample] for name in ARMS])
    exact, pairs = unique_pair_ocf(outputs, scored.expand_as(outputs), masks, scorer, images.device, 384)
    raw = dict(zip(ARMS, exact.numpy()))
    results = {
        "source_days": list(fit), "unique_ocf_pairs": pairs,
        "posterior_top1": float((posterior.argmax(1) == order).float().mean()),
        "seconds": time.monotonic() - started,
        "arms": {
            name: {**row_diagnostics(scores[name], order), **summarize([raw[name]], [raw["v9"]]),
                   "assigned_top1": float((choice == order).float().mean()),
                   "changed_assignments": int((choice != choices["v9"]).sum()),
                   "unique_assignments": int(choice.unique().numel())}
            for name, choice in choices.items()
        },
        "paired_ocf": {name: values.tolist() for name, values in raw.items()},
    }
    return results, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_relational_geometry.json"))
    parser.set_defaults(seed=211, encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Use Kaggle CPU with the mounted competition data.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    device = torch.device("cpu")
    images = load_images(root, device)
    masks = images >= 128
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(masks))
    scorer = official_scorer(root)

    # Exclude the paired OCF sample from the previous cross-mouse study on these days.
    prior_scored = np.random.default_rng(17691).permutation(len(masks))[:160]
    eligible = np.setdiff1d(np.arange(len(masks)), prior_scored)
    scored = torch.from_numpy(np.random.default_rng(SPLIT_SEED).choice(eligible, size=192, replace=False))
    report = {
        "mouse": MOUSE, "split_seed": SPLIT_SEED, "dimension": DIMENSION, "temperature": TEMPERATURE,
        "quadratic_geometry_weight": GEOMETRY_WEIGHT, "scored_glyphs": scored.tolist(), "days": {},
        "protocol": "One geometry update from frozen v9 identity probabilities on each complete held recording. "
                    "Source and target Gram matrices use split trial means, within-glyph noise whitening and "
                    "independent top signal eigenspaces. The diagonal is removed in both domains. All arms use "
                    "the complete gallery and a one-to-one assignment. Known correspondence appears only in "
                    "the retrospective neighbor-identity control and metric computation. No formal test input.",
    }
    all_scores = {name: [] for name in ARMS}
    for held in HELD_DAYS:
        day, scores = evaluate(held, recordings, images, masks, scorer, scored, args)
        report["days"][held] = day
        for name, values in scores.items():
            all_scores[name].append(values)
        report["summary"] = {name: summarize(rows, all_scores["v9"]) for name, rows in all_scores.items()}
        report["real_vs_shuffle"] = summarize(all_scores["geometry"], all_scores["stratified_shuffle"])
        report["seconds"] = time.monotonic() - started
        staged = args.output.with_suffix(".tmp")
        staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.replace(args.output)
        print(json.dumps({"session": held, "arms": day["arms"], "seconds": day["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
