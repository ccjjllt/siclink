"""Compare latent identity balancing and contour utility in Task 2."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, fit_atlas_metric_ranker
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures, fit_temperature, glyph_posterior, similarity_from_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task2_metric_validation import SESSIONS, metric_labels, score_block, shortlist, shortlisted_v9_score, sparse_assignment
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_v8_metric_validation import summarize


V11_ANCHOR = 8.0
SINKHORN_ITERATIONS = 32
ACTION_VARIANTS = (
    "v11", "v11_independent", "row_contour", "balanced_identity",
    "balanced_contour", "balanced_contour_independent",
)


def normalize(logits):
    centered = logits - logits.mean(dim=1, keepdim=True)
    return centered / centered.square().mean().sqrt().clamp_min(1e-6)


def calibration_sets(count, seed, ocf_size, posterior_size, evaluation_size):
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    return np.split(order[:ocf_size + posterior_size + evaluation_size],
                    (ocf_size, ocf_size + posterior_size))


def fit_on_source_rotations(sources, bank, ocf_labels, probability_labels, args):
    images, masks, signature, energy, ink, _, scorer = bank
    rng = np.random.default_rng(args.seed)
    candidates, exact, calibrate, blocks = [], [], [], {}
    for session, counts in sources.items():
        fit = {other: source for other, source in sources.items() if other != session}
        order = torch.as_tensor(rng.permutation(len(masks)), device=images.device)
        inverse = order.argsort()
        evidence, v9 = score_block("mice1", fit, counts[order], images, signature, energy, images.device, args)
        rows = inverse[ocf_labels]
        sampled = shortlist({name: values[rows] for name, values in evidence.items()}, v9[rows], energy, ink)
        values, pairs = metric_labels(sampled, ocf_labels, masks, scorer, images.device, 384)
        candidates.append(sampled)
        exact.append(values)
        scores = normalize(evidence["v9"])
        calibrate.append((scores[inverse[probability_labels]], probability_labels))
        blocks[session] = {"source_days": list(fit), "exact_image_pairs": pairs}
        print(json.dumps({"posterior_calibration": session, **blocks[session]}), flush=True)
    ranker = fit_atlas_metric_ranker(concat_atlas_candidates(candidates), torch.cat(exact), seed=args.seed)
    temperature = fit_temperature(calibrate)
    return ranker, temperature, blocks


def posterior_diagnostics(probability, order):
    probability = probability.clamp_min(1e-12)
    exact = probability.gather(1, order[:, None])[:, 0]
    leaders = probability.topk(32, dim=1)
    agreement = leaders.indices == order[:, None]
    return {"identity_nll": float(-exact.log().mean()), "top1": float(agreement[:, 0].float().mean()),
            "top5": float(agreement[:, :5].any(1).float().mean()), "top32": float(agreement.any(1).float().mean()),
            "mean_top32_mass": float(leaders.values.sum(1).mean()),
            "mean_abs_column_mass_error": float((probability.sum(0) - 1).abs().mean())}


@torch.no_grad()
def decide(matrices, v9, ranker, energy, ink, utility, temperature):
    candidates = shortlist(matrices, v9, energy, ink)
    base = ranker.predict_ocf(candidates) + V11_ANCHOR * shortlisted_v9_score(matrices["v9"], candidates)
    normal = normalize(matrices["v9"])
    unbalanced = glyph_posterior(normal, temperature)
    balanced = glyph_posterior(normal, temperature, balance=True, iterations=SINKHORN_ITERATIONS)

    # Raw expected OCF tends to favor glyphs that resemble many unrelated
    # glyphs. Subtract the value under a uniform nuisance identity.
    image_mean = utility.mean(dim=0)
    uniform_self = 100 / utility.shape[0]
    row_shape = unbalanced @ utility - image_mean
    balanced_shape = balanced @ utility - image_mean
    identity = 100 * balanced - uniform_self
    decision_scores = {
        "v11": base,
        "v11_independent": base,
        "row_contour": base + row_shape.gather(1, candidates.indices),
        "balanced_identity": base + identity.gather(1, candidates.indices),
        "balanced_contour": base + balanced_shape.gather(1, candidates.indices),
        "balanced_contour_independent": base + balanced_shape.gather(1, candidates.indices),
    }
    choices, fallback = {}, {}
    for name, objective in decision_scores.items():
        if name.endswith("_independent"):
            position = objective.masked_fill(~candidates.valid, -torch.inf).argmax(dim=1, keepdim=True)
            choices[name] = candidates.indices.gather(1, position)[:, 0]
            fallback[name] = False
        else:
            choices[name], fallback[name] = sparse_assignment(candidates, objective, v9)
    offsets = {name: float((values - base)[candidates.valid].abs().mean()) for name, values in decision_scores.items()}
    return choices, fallback, offsets, unbalanced, balanced


def evaluate(held_day, recordings, bank, ocf_labels, probability_labels, scored, args):
    started = time.monotonic()
    images, masks, signature, energy, ink, utility, scorer = bank
    sources = {session: values for session, values in recordings.items() if session != held_day}
    ranker, temperature, rotation_reports = fit_on_source_rotations(sources, bank, ocf_labels, probability_labels, args)
    rng = np.random.default_rng(args.split_seed + SESSIONS.index(held_day))
    order = torch.as_tensor(rng.permutation(len(masks)), device=images.device)
    matrices, v9 = score_block("mice1", sources, recordings[held_day][order], images, signature, energy, images.device, args)
    choices, fallback, offsets, row_matrix, balanced = decide(matrices, v9, ranker, energy, ink, utility,
                                                              temperature["temperature"])
    sample = order.argsort()[scored]
    predictions = torch.stack([values[sample] for values in choices.values()])
    exact, pairs = unique_pair_ocf(predictions, scored.expand_as(predictions), masks, scorer, images.device, 384)
    by_mode = dict(zip(choices, exact.cpu().numpy()))
    report = {"session": held_day, "source_days": list(sources), "temperature": temperature,
              "ocf_calibration": rotation_reports, "unique_scored_pairs": pairs,
              "latent_posterior": {"row": posterior_diagnostics(row_matrix, order),
                                   "balanced": posterior_diagnostics(balanced, order)},
              "actions": {name: {**summarize([by_mode[name]], [by_mode["v11"]]),
                                  "identity_top1": float((values == order).float().mean()),
                                  "changed_from_v11": int((values != choices["v11"]).sum()),
                                  "distinct_glyphs": int(values.unique().numel()), "fallback": fallback[name],
                                  "mean_absolute_offset": offsets[name]}
                          for name, values in choices.items()},
              "seconds": time.monotonic() - started}
    return report, by_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-days", nargs="+", choices=SESSIONS, default=("session2", "session4"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--split-seed", type=int, default=15803)
    parser.add_argument("--ocf-calibration-images", type=int, default=72)
    parser.add_argument("--posterior-calibration-images", type=int, default=384)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_posterior_assignment.json"))
    args = parser.parse_args()
    sizes = (args.ocf_calibration_images, args.posterior_calibration_images, args.ocf_images)
    if min(sizes) < 2:
        parser.error("Use at least two glyphs in each independent partition.")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Use Kaggle with the official competition input.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    device = torch.device("cpu")
    images = load_images(root, device)
    if sum(sizes) > len(images):
        parser.error("The calibration and evaluation partitions must fit inside the gallery.")
    masks = images >= 128
    signatures, energy = contour_signatures(images)
    ink = masks.float().mean(dim=(1, 2))
    visual = 100 * similarity_from_signatures(signatures, energy)
    scorer = official_scorer(root)
    recordings = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    partitions = calibration_sets(len(images), args.split_seed, args.ocf_calibration_images,
                                  args.posterior_calibration_images, args.ocf_images)
    ocf_fit, probability_fit, scored = (torch.as_tensor(rows, device=device) for rows in partitions)
    bank = images, masks, signatures, energy, ink, visual, scorer
    measurements = {name: [] for name in ACTION_VARIANTS}
    report = {"mouse": "mice1", "model_seed": args.seed, "split_seed": args.split_seed,
              "ocf_fit_glyphs": len(ocf_fit), "probability_calibration_glyphs": len(probability_fit),
              "ocf_rows_per_recording": len(scored), "v11_anchor": V11_ANCHOR, "sinkhorn_steps": SINKHORN_ITERATIONS,
              "protocol": "Only complete anonymous Task 2 training-day holdouts with four actual repeats. "
                          "The latent identity posterior is either row normalized or balanced jointly because each "
                          "gallery glyph occurs once. Fixed v11 output edges receive excess expected contour "
                          "agreement or the matched balanced identity-only score. Every rule uses the same outer "
                          "query rows and disjoint source-day calibrations. No official test input or sample indices.", "days": {}}
    for held_day in dict.fromkeys(args.held_days):
        result, values = evaluate(held_day, recordings, bank, ocf_fit, probability_fit, scored, args)
        report["days"][held_day] = result
        for name, scores in values.items():
            measurements[name].append(scores)
        report["summary"] = {name: summarize(rows, measurements["v11"]) for name, rows in measurements.items()}
        report["balanced_vs_row_contour"] = summarize(measurements["balanced_contour"], measurements["row_contour"])
        report["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
