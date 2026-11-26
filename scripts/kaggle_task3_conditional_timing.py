"""Test candidate-specific spike timing after conditioning on each 25 ms count."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores, fit_contour_retrieval
from glyph_data import load_catalog, load_images, locate_data, read_trials, split_repeats, training_groups
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model, gain_template_logits
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import unique_pair_ocf


BIN_MS = 5
COUNTS_PER_WINDOW = 5
N_COARSE = 10
PRIOR_COUNTS = (2.0, 8.0, 32.0, 128.0)
TIMING_WEIGHTS = (0.5, 1.0)


def window_view(counts):
    return counts.unflatten(-1, (-1, N_COARSE, COUNTS_PER_WINDOW))


def coarse_counts(counts):
    return window_view(counts).sum(dim=-1).flatten(-2)


def evoked(counts):
    return window_view(counts)[..., 2:, :]


def split_recordings(root, catalog, mouse, image_count, calibrate, evaluate, repeat_seed):
    coarse, queries, inner, outer = {}, {}, {}, {}
    entries = sorted((item for item in catalog["blocks"] if item["mouse"] == mouse and item["task"] == "train"),
                     key=lambda item: item["session"])
    for index, entry in enumerate(entries):
        session = entry["session"]
        block = read_trials(root, catalog, entry, "cpu", bin_ms=BIN_MS)
        grouped = training_groups(block, image_count)
        del block
        fitting, held = split_repeats(grouped, repeat_seed + index)
        del grouped
        inside_fit, inside_held = split_repeats(fitting[calibrate], repeat_seed + 1000 + index)
        # Retain fine histograms only for the aggregates and sampled held queries.
        inner[session] = (evoked(inside_fit).sum(dim=1), evoked(inside_held)[:, 0].contiguous())
        outer[session] = (evoked(fitting).sum(dim=1), evoked(held[evaluate])[:, 0].contiguous())
        coarse[session] = coarse_counts(fitting)
        queries[session] = coarse_counts(held[evaluate])
        del fitting, held, inside_fit, inside_held
    return coarse, queries, inner, outer


def timing_terms(local, others, strength):
    """Dirichlet-smoothed within-window profiles with the same coarse counts."""
    background = local.sum(dim=0) + 0.5
    background = background / background.sum(dim=-1, keepdim=True)
    support = local + 0.5 * others
    posterior = support + strength * background
    probabilities = posterior / posterior.sum(dim=-1, keepdim=True)
    return probabilities.log() - background.log()


def mean_other_recordings(recordings, session):
    return torch.stack([counts for name, (counts, _) in recordings.items() if name != session]).mean(dim=0)


def calibrate_strength(recordings):
    fits = {value: [] for value in PRIOR_COUNTS}
    for session, (counts, observed) in recordings.items():
        others = mean_other_recordings(recordings, session)
        for value in PRIOR_COUNTS:
            delta = (observed * timing_terms(counts, others, value)).flatten(1).sum(dim=1)
            fits[value].append(delta)
    mean_gain = {value: float(torch.cat(gains).mean()) for value, gains in fits.items()}
    return max(mean_gain, key=mean_gain.get), mean_gain


def zscore(values):
    centered = values - values.mean(dim=1, keepdim=True)
    return centered / centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)


def rank_metrics(score, labels):
    listing = score.topk(32, dim=1).indices
    matches = listing == labels[:, None]
    distribution = 1.5 * zscore(score)
    nll = distribution.logsumexp(dim=1) - distribution.gather(1, labels[:, None])[:, 0]
    return {"top1": float(matches[:, 0].float().mean()), "top5": float(matches[:, :5].any(1).float().mean()),
            "top32": float(matches.any(1).float().mean()), "nll_fixed_temperature": float(nll.mean())}


def summarize(scores):
    control = np.stack(scores["v11"])
    summary = {}
    for name, measured in scores.items():
        values = np.stack(measured)
        delta = (values - control).mean(axis=0)
        summary[name] = {"ocf": float(values.mean()), "paired_delta": float(delta.mean()),
                         "paired_glyph_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
                         "scored_rows": values.size}
    return summary


@torch.no_grad()
def score_session(session, query, fine_query, timing_tables, images, masks, scorer, labels, permutation,
                  order, encode, contour, gain, count):
    rates = encode.expected_counts(images, session)
    raw = count_template_logits(query, rates, "timed") if gain is None else gain_template_logits(query, rates, gain)
    direction = contour_retrieval_scores(contour, query, session)["contour_ridge"]
    baseline = blend_contour_poisson(direction, raw)
    evidence = fine_query[order].flatten(1) @ timing_tables.flatten(1).T
    scale = raw.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    methods = {"v11": baseline, "timing_alone": evidence}
    for weight in TIMING_WEIGHTS:
        methods[f"timing_{weight:g}"] = baseline + weight * evidence / scale
        methods[f"permuted_{weight:g}"] = baseline + weight * evidence[:, permutation] / scale
    ranked = {name: rank_metrics(values, labels) for name, values in methods.items()}
    paired_rows = order.argsort()[:count]
    choices = {name: values[paired_rows].argmax(dim=1) for name, values in methods.items() if name != "timing_alone"}
    predictions = torch.stack(tuple(choices.values()))
    references = labels[paired_rows].expand_as(predictions)
    measured, pairs = unique_pair_ocf(predictions, references, masks, scorer, images.device, 128)
    paired = dict(zip(choices, measured.numpy()))
    evidence_on_target = evidence.gather(1, labels[:, None])[:, 0]
    permuted_on_target = evidence[:, permutation].gather(1, labels[:, None])[:, 0]
    report = {"rank": ranked, "ocf": summarize({name: [score] for name, score in paired.items()}),
              "likelihood_gain_nats_true_class": float(evidence_on_target.mean()),
              "permuted_likelihood_gain_nats_true_class": float(permuted_on_target.mean()),
              "mean_candidate_timing_to_coarse_sd": float((evidence.std(dim=1, keepdim=True) / scale).mean()),
              "unique_exact_pairs": pairs}
    return paired, report


def run_mouse(mouse, root, catalog, images, masks, shape, scorer, args, calibration, sampled, rng, report, save):
    training, coarse_query, inner, outer = split_recordings(root, catalog, mouse, len(images), calibration,
                                                           sampled, args.repeat_seed)
    strength, timing_calibration = calibrate_strength(inner)
    del inner
    report["finite_timing_prior_count"] = strength
    report["calibration_nats_vs_global_timing"] = timing_calibration
    save()
    print(json.dumps({"mouse": mouse, "timing_prior_count": strength, "calibration": timing_calibration}), flush=True)

    encode = fit_glyph_encoder(training, images, epochs=18, seed=args.seed)
    contour = fit_contour_retrieval(training, *shape, task="task3", seed=args.seed)
    gain = fit_gain_model(training, "timed", temporal=True) if mouse == "mice2" else None
    del training
    donor_order = torch.as_tensor(rng.permutation(len(images)))
    permutation = torch.empty_like(donor_order)
    permutation[donor_order] = donor_order.roll(1)
    measurements = {}
    for session, counts in coarse_query.items():
        order = torch.as_tensor(rng.permutation(len(sampled)))
        target_tables, fine_query = outer[session]
        templates = timing_terms(target_tables, mean_other_recordings(outer, session), strength)
        paired, block = score_session(session, query=counts[order], fine_query=fine_query, timing_tables=templates,
                                      images=images, masks=masks,
                                      scorer=scorer, labels=sampled[order], permutation=permutation, order=order,
                                      encode=encode, contour=contour, gain=gain,
                                      count=args.ocf_images)
        report["sessions"][session] = block
        for name, values in paired.items():
            measurements.setdefault(name, []).append(values)
        report["summary_ocf"] = summarize(measurements)
        save()
        print(json.dumps({"mouse": mouse, "session": session, **block}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, default=3977)
    parser.add_argument("--calibration-glyphs", type=int, default=384)
    parser.add_argument("--evaluation-glyphs", type=int, default=512)
    parser.add_argument("--ocf-images", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_conditional_timing.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_images <= args.evaluation_glyphs or args.calibration_glyphs < 2:
        parser.error("Choose positive distinct calibration and evaluation allocations.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the competition-data timing experiment on Kaggle CPU.")
    started = time.monotonic()
    torch.set_num_threads(4)
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, "cpu")
    masks = images >= 128
    shape = contour_signatures(images)
    rng = np.random.default_rng(args.repeat_seed + 3001)
    order = torch.as_tensor(rng.permutation(len(images)))
    end = args.calibration_glyphs + args.evaluation_glyphs
    if end > len(order):
        parser.error("Reserve separate calibration and evaluation glyph groups.")
    calibration, sampled = order[:args.calibration_glyphs], order[args.calibration_glyphs:end]
    scorer = official_scorer(root)
    output = {"device": "cpu", "repeat_seed": args.repeat_seed, "model_seed": args.seed,
              "reference_glyph_count": len(images), "scored_queries_per_session": len(sampled),
              "calibration_queries_per_session": len(calibration), "ocf_images_per_session": args.ocf_images,
              "protocol": "Condition on the existing 25 ms totals. In each 25 ms window from 50 to 250 ms, "
                          "use trial spike positions in five 5 ms bins and smooth each glyph's timing distribution "
                          "toward a global per-unit and window distribution. Calibrate finite shrinkage on "
                          "different inner repeats and glyphs than the outer scoring rows. The shuffled control "
                          "permutes only the glyph timing tables. Baseline measures actual v11 Task 3 policies.",
              "mice": {}}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    for mouse in ("mice1", "mice2"):
        report = {"sessions": {}}
        output["mice"][mouse] = report
        run_mouse(mouse, root, catalog, images, masks, shape, scorer, args, calibration, sampled, rng, report, save)


if __name__ == "__main__":
    main()
