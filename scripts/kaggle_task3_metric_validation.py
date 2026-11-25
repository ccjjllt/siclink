"""Evaluate a nested exact-OCF reranker for Task 3's closed glyph gallery."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, fit_atlas_metric_ranker, pack_atlas_candidates
from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_early_noise import fit_evoked_templates
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model, gain_template_logits
from glyph_mbr import contour_signatures
from glyph_negative_binomial import fit_count_dispersion, negative_binomial_template_logits
from kaggle_diagnostics import feature_view, official_scorer, standardize


NOMINATORS = ("v6", "contour_gain", "negative_binomial", "count_template")


@dataclass
class Task3ScoreModels:
    encoder: object
    contours: object
    gain: object
    dispersion: torch.Tensor
    count_templates: dict
    count_calibration: dict


def fit_score_models(training, images, signature, energy, args):
    expected = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
    contours = fit_contour_retrieval(training, signature, energy, task="task3", seed=args.seed)
    gain = fit_gain_model(training, "timed", temporal=True)
    dispersion = fit_count_dispersion(training, "timed")
    total_count, calibration = {}, {}
    for name, counts in training.items():
        total_count[name], calibration[name] = standardize(feature_view(counts, "evoked-total"))
    return Task3ScoreModels(expected, contours, gain, dispersion, fit_evoked_templates(total_count), calibration)


@torch.no_grad()
def score_candidates(model, query, images, session, mouse, energy, ink, nominations):
    contour = contour_retrieval_scores(model.contours, query, session)["contour_ridge"]
    rates = model.encoder.expected_counts(images, session)
    poisson = count_template_logits(query, rates, "timed")
    gain = gain_template_logits(query, rates, model.gain)
    negative_binomial = negative_binomial_template_logits(query, rates, "timed", model.dispersion)
    totals, _ = standardize(feature_view(query, "evoked-total"), model.count_calibration[session])
    measured = model.count_templates[session].logits(totals)
    v6 = blend_contour_poisson(contour, poisson) if mouse == "mice1" else contour
    scores = {"v6": v6, "contour_gain": blend_contour_poisson(contour, gain),
              "negative_binomial": negative_binomial, "count_template": measured,
              "poisson": poisson, "contour": contour, "gain": gain,
              "log_contour_energy": energy.log().expand_as(v6), "ink_fraction": ink.expand_as(v6)}
    candidates = pack_atlas_candidates(scores, nominations, baseline="v6")
    top1 = {name: scores[name].argmax(dim=1) for name in ("v6", "contour_gain")}
    return candidates, top1


@torch.no_grad()
def unique_pair_ocf(predicted, targets, masks, scorer, device, chunk_size):
    """Score each ordered candidate/target image pair once and restore the input order."""
    n_glyphs = len(masks)
    encoded = targets * n_glyphs + predicted
    unique, inverse = torch.unique(encoded.reshape(-1), return_inverse=True)
    scores = []
    for part in unique.split(chunk_size):
        guess = masks[part.remainder(n_glyphs)].float().cpu().numpy()
        reference = masks[part.div(n_glyphs, rounding_mode="floor")].cpu().numpy()
        measured = scorer(guess, reference, device=str(device), batch_size=12)
        scores.append(torch.as_tensor(measured, dtype=torch.float32, device=device))
    return torch.cat(scores)[inverse].reshape_as(encoded), len(unique)


def nested_repeats(recordings, repeat_seed):
    """Reserve the outer repeat before selecting any inner calibration repeat."""
    inner_fit, inner_metric, outer_fit, outer_query = {}, {}, {}, {}
    for index, (session, counts) in enumerate(recordings.items()):
        if counts.shape[1] != 4:
            raise ValueError(f"Task 3 metric validation needs four labeled repeats in {session}.")
        outer_fit[session], outer_query[session] = split_repeats(counts, repeat_seed + index)
        inner_fit[session], inner_metric[session] = split_repeats(outer_fit[session], repeat_seed + 1000 + index)
    return inner_fit, inner_metric, outer_fit, outer_query


def paired_statistics(values, baseline):
    delta = values - baseline
    error = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None
    return {"ocf": float(values.mean()), "paired_delta_vs_v6": float(delta.mean()), "paired_se": error}


def aggregate_measurements(measurements):
    baseline = np.stack(measurements["v6"])
    results = {}
    for name, sampled in measurements.items():
        values = np.stack(sampled)
        results[name] = {**paired_statistics(values.mean(axis=0), baseline.mean(axis=0)),
                         "ocf_n": values.size, "unique_held_glyphs": values.shape[1]}
    return results


def run_mouse(mouse, root, catalog, images, masks, signature, energy, ink, setup, scorer, device, args, started, output):
    metric_glyphs, held_glyphs, nominations, repeat_seed = setup
    raw = grouped_blocks(root, catalog, mouse, device, len(images))
    inner_fit, inner_metric, outer_fit, outer_query = nested_repeats(raw, repeat_seed)
    del raw
    print(f"{mouse}: fit score models on two repeats for inner calibration", flush=True)
    inner = fit_score_models(inner_fit, images, signature, energy, args)
    batches = [score_candidates(inner, inner_metric[session][metric_glyphs], images, session,
                                mouse, energy, ink, nominations)[0] for session in args.sessions]
    calibration = concat_atlas_candidates(batches)
    del batches, inner, inner_fit, inner_metric
    row, col = calibration.valid.nonzero(as_tuple=True)
    references = metric_glyphs[row.remainder(len(metric_glyphs))]
    exact, unique_pairs = unique_pair_ocf(calibration.indices[row, col], references,
                                         masks, scorer, device, args.ocf_chunk_size)
    target = calibration.features.new_full(calibration.valid.shape, torch.nan)
    target[row, col] = exact
    ranker = fit_atlas_metric_ranker(calibration, target, seed=args.seed, epochs=args.ranker_epochs)
    oracle = target.masked_fill(~calibration.valid, -torch.inf).max(dim=1).values
    measurement_summary = {"queries": len(calibration.valid), "labeled_candidates": len(exact),
                           "unique_exact_pairs": unique_pairs, "mean_candidate_ocf": float(exact.mean()),
                           "mean_shortlist_oracle_ocf": float(oracle.mean()), "ranker": ranker.diagnostics}
    print(json.dumps({"mouse": mouse, "calibration": measurement_summary}), flush=True)
    record = {"calibration": measurement_summary, "sessions": {}}
    output["mice"][mouse] = record
    del calibration, target, exact

    print(f"{mouse}: fit score models on three repeats for the outer fold", flush=True)
    outer = fit_score_models(outer_fit, images, signature, energy, args)
    del outer_fit
    measurements = {}
    for session in args.sessions:
        candidates, chosen = score_candidates(outer, outer_query[session][held_glyphs], images, session,
                                            mouse, energy, ink, nominations)
        chosen.update(metric_ocf=ranker.select(candidates, anchor_ocf=0),
                      metric_ocf_anchored=ranker.select(candidates, anchor_ocf=0.75),
                      metric_ocf_strong_anchor=ranker.select(candidates, anchor_ocf=2.0))
        stacked = torch.stack(tuple(chosen.values()))
        measured, _ = unique_pair_ocf(stacked, held_glyphs.expand_as(stacked), masks,
                                      scorer, device, args.ocf_chunk_size)
        paired = dict(zip(chosen, measured.cpu().numpy()))
        baseline = paired["v6"]
        rows = {}
        for name, values in paired.items():
            measurements.setdefault(name, []).append(values)
            rows[name] = {**paired_statistics(values, baseline), "ocf_n": len(values),
                          "top1_on_sample": float((chosen[name] == held_glyphs).float().mean()),
                          "changed_fraction": float((chosen[name] != chosen["v6"]).float().mean())}
        record["sessions"][session] = rows
        record["summary"] = aggregate_measurements(measurements)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"mouse": mouse, "session": session, "variants": rows}), flush=True)
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session1", "session2", "session3", "session4"))
    parser.add_argument("--calibration-images", type=int, default=64, help="exact metric training glyphs per session")
    parser.add_argument("--ocf-images", type=int, default=64, help="paired outer held glyphs per session")
    parser.add_argument("--candidates-per-source", type=int, default=6)
    parser.add_argument("--ocf-chunk-size", type=int, default=384)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--ranker-epochs", type=int, default=80)
    parser.add_argument("--allow-shared-glyphs", action="store_true",
                        help="still separate repeats while permitting a glyph in both metric subsets")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_metric_validation.json"))
    args = parser.parse_args()
    if min(args.calibration_images, args.ocf_images) < 2 or min(args.candidates_per_source, args.ocf_chunk_size,
                                                                args.encoder_epochs, args.ranker_epochs) < 1:
        parser.error("Choose positive fit and scoring settings, with at least two sampled glyphs per subset.")
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    masks = images >= 128
    signature, energy = contour_signatures(images)
    ink = masks.float().mean(dim=(1, 2))
    scorer = official_scorer(root)
    rng = np.random.default_rng(args.seed + 1800)
    if max(args.calibration_images, args.ocf_images, args.candidates_per_source) > len(images):
        parser.error("Calibration, evaluation and candidate counts cannot exceed the gallery.")
    if not args.allow_shared_glyphs and args.calibration_images + args.ocf_images > len(images):
        parser.error("The sum of two disjoint metric subsets cannot exceed the gallery.")
    ordering = rng.permutation(len(images))
    metric = ordering[:args.calibration_images]
    if args.allow_shared_glyphs:
        held = rng.choice(len(images), args.ocf_images, replace=False)
    else:
        held = ordering[args.calibration_images:args.calibration_images + args.ocf_images]
    nominations = dict.fromkeys(NOMINATORS, args.candidates_per_source)
    repeat_seed = args.seed if args.repeat_seed is None else args.repeat_seed
    setup = (torch.as_tensor(metric, device=device), torch.as_tensor(held, device=device), nominations, repeat_seed)
    output = {"device": str(device), "seed": args.seed, "repeat_seed": repeat_seed,
              "calibration_glyphs_per_session": len(metric), "scored_outer_glyphs_per_session": len(held),
              "disjoint_metric_glyphs": not args.allow_shared_glyphs, "nominations": nominations,
              "encoder_epochs": args.encoder_epochs, "ranker_epochs": args.ranker_epochs,
              "protocol": "Two inner fit repeats, one exact-OCF calibration repeat, one untouched outer repeat. "
                          "Outer score models are refitted on all three non-outer repeats.",
              "v6_reference": {"mice1": "contour plus timed Poisson", "mice2": "contour ridge"}, "mice": {}}
    all_measurements = {}
    for mouse in args.mice:
        measured = run_mouse(mouse, root, catalog, images, masks, signature, energy, ink, setup,
                             scorer, device, args, started, output)
        for name, blocks in measured.items():
            all_measurements.setdefault(name, []).extend(blocks)
        output["summary"] = aggregate_measurements(all_measurements)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
