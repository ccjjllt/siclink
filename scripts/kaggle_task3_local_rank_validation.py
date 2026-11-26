"""Validate candidate-local directional evidence against the frozen Task 3 policy."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, pack_atlas_candidates
from glyph_contour_retrieval import _agreement, _response_features, blend_contour_poisson
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits
from glyph_gain_likelihood import gain_template_logits
from glyph_local_ocf_ranker import directional_tiles, fit_relative_ranker, local_evidence
from glyph_mbr import contour_signatures
from glyph_negative_binomial import negative_binomial_template_logits
from glyph_retrieval import _standardize
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_task3_metric_validation import (fit_score_models, nested_repeats, paired_statistics,
                                             unique_pair_ocf)


NOMINATIONS = {"v11": 6, "contour": 4, "other_blend": 4, "negative_binomial": 3, "count_template": 3}
ANCHORS = (4, 8)


@torch.no_grad()
def score_candidates(model, counts, images, gallery, energy, ink, session, mouse):
    contours = model.contours
    features = _response_features(counts, contours.view, contours.n_bins)
    projected = _standardize(features, *contours.moments[session]).mean(dim=1) @ contours.projection
    estimate = 0.80 * contours.readout.predict(projected) + 0.20 * contours.local[session].predict(projected)
    contour = _agreement(estimate, contours)
    rates = model.encoder.expected_counts(images, session)
    poisson = count_template_logits(counts, rates, "timed")
    gain = gain_template_logits(counts, rates, model.gain)
    poisson_blend = blend_contour_poisson(contour, poisson)
    gain_blend = blend_contour_poisson(contour, gain)
    v11, other = (poisson_blend, gain_blend) if mouse == "mice1" else (gain_blend, poisson_blend)
    measured, _ = standardize(feature_view(counts, "evoked-total"), model.count_calibration[session])
    scores = {"v11": v11, "other_blend": other, "contour": contour, "poisson": poisson, "gain": gain,
              "negative_binomial": negative_binomial_template_logits(counts, rates, "timed", model.dispersion),
              "count_template": model.count_templates[session].logits(measured),
              "ink_fraction": ink.expand_as(v11), "log_contour_energy": energy.log().expand_as(v11)}
    candidates = pack_atlas_candidates(scores, NOMINATIONS, baseline="v11")
    anchor = v11.argmax(dim=1)
    matches = candidates.valid & (candidates.indices == anchor[:, None])
    if not bool(matches.any(dim=1).all()):
        raise ValueError("Expected the v11 leader to be present for every query.")
    local = local_evidence(estimate[:, :-1], gallery, candidates.indices)
    return candidates, local, matches.long().argmax(dim=1), anchor


def compare_rows(measured, chosen, targets):
    return {name: {**compare(scores, measured["v11"]),
                   "ocf_n": len(scores), "top1_on_sample": float((chosen[name] == targets).float().mean()),
                   "changed_fraction": float((chosen[name] != chosen["v11"]).float().mean())}
            for name, scores in measured.items()}


def compare(values, baseline):
    result = paired_statistics(values, baseline)
    result["paired_delta_vs_v11"] = result.pop("paired_delta_vs_v6")
    return result


def summarize(measurements):
    base = np.stack(measurements["v11"])
    result = {}
    for name, values in measurements.items():
        values = np.stack(values)
        row = compare(values.mean(axis=0), base.mean(axis=0))
        row.update(ocf_n=values.size, unique_held_glyphs=values.shape[1])
        result[name] = row
    for weight in ANCHORS:
        local, scalar = f"local_anchor{weight}", f"scalar_anchor{weight}"
        comparison = paired_statistics(np.stack(measurements[local]).mean(axis=0),
                                       np.stack(measurements[scalar]).mean(axis=0))
        result[local].update(paired_delta_vs_scalar=comparison["paired_delta_vs_v6"],
                             paired_se_vs_scalar=comparison["paired_se"])
    return result


def run_mouse(mouse, data, calibration, held, output, args, save):
    root, catalog, images, masks, signature, energy, gallery, ink, scorer = data
    started = time.monotonic()
    device = images.device
    raw = grouped_blocks(root, catalog, mouse, device, len(images))
    inner_fit, inner_query, outer_fit, outer_query = nested_repeats(raw, args.repeat_seed)
    del raw
    print(f"{mouse}: fit two repeats for calibration", flush=True)
    inner = fit_score_models(inner_fit, images, signature, energy, args)
    compiled = [score_candidates(inner, inner_query[session][calibration], images, gallery, energy, ink, session, mouse)
                for session in args.sessions]
    candidates, local, anchors, _ = zip(*compiled)
    candidates = concat_atlas_candidates(candidates)
    local, anchors = torch.cat(local), torch.cat(anchors)
    del inner, inner_fit, inner_query, compiled
    row, col = candidates.valid.nonzero(as_tuple=True)
    exact, pairs = unique_pair_ocf(candidates.indices[row, col], calibration[row.remainder(len(calibration))],
                                  masks, scorer, device, args.ocf_chunk_size)
    target = candidates.features.new_full(candidates.valid.shape, torch.nan)
    target[row, col] = exact
    scalar = fit_relative_ranker(candidates, None, anchors, target)
    direction = fit_relative_ranker(candidates, local, anchors, target)
    inner_leader_ocf = target.gather(1, anchors[:, None]).mean()
    inner_oracle = target.masked_fill(~candidates.valid, -torch.inf).amax(dim=1).mean()
    record = {"calibration": {"queries": len(anchors), "unique_exact_pairs": pairs,
                              "v11_ocf": float(inner_leader_ocf), "shortlist_oracle_ocf": float(inner_oracle),
                              "scalar": scalar.diagnostics, "local": direction.diagnostics}, "sessions": {}}
    output["mice"][mouse] = record
    save()
    del candidates, local, anchors, exact, target
    print(json.dumps({"mouse": mouse, "calibration": record["calibration"]}), flush=True)

    print(f"{mouse}: fit three repeats for outer evaluation", flush=True)
    outer = fit_score_models(outer_fit, images, signature, energy, args)
    del outer_fit
    measurements = {}
    for session in args.sessions:
        candidates, local, anchors, v11 = score_candidates(outer, outer_query[session][held], images, gallery,
                                                           energy, ink, session, mouse)
        chosen = {"v11": v11}
        for weight in ANCHORS:
            chosen[f"scalar_anchor{weight}"] = scalar.select(candidates, None, anchors, weight)
            chosen[f"local_anchor{weight}"] = direction.select(candidates, local, anchors, weight)
        stack = torch.stack(tuple(chosen.values()))
        values, pairs = unique_pair_ocf(stack, held.expand_as(stack), masks, scorer, device, args.ocf_chunk_size)
        paired = dict(zip(chosen, values.cpu().numpy()))
        for name, scores in paired.items():
            measurements.setdefault(name, []).append(scores)
        record["sessions"][session] = {"unique_exact_pairs": pairs, "methods": compare_rows(paired, chosen, held)}
        record["summary"] = summarize(measurements)
        record["seconds"] = time.monotonic() - started
        save()
        print(json.dumps({"mouse": mouse, "session": session, "metrics": record["sessions"][session]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session1", "session2", "session3", "session4"))
    parser.add_argument("--calibration-images", type=int, default=96)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--ocf-chunk-size", type=int, default=384)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=211, help="v11 encoder and contour seed")
    parser.add_argument("--repeat-seed", type=int, default=419)
    parser.add_argument("--glyph-seed", type=int, default=3011)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_local_rank_validation.json"))
    args = parser.parse_args()
    if min(args.calibration_images, args.ocf_images) < 2 or min(args.ocf_chunk_size, args.encoder_epochs) < 1:
        parser.error("Choose positive chunk and epoch counts and at least two glyphs per fold.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the nested exact OCF validation on Kaggle GPU.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, "cuda:0")
    masks = images >= 128
    signature, energy = contour_signatures(images)
    gallery = directional_tiles(signature)
    ink = masks.float().mean(dim=(1, 2))
    scorer = official_scorer(root)
    order = torch.as_tensor(np.random.default_rng(args.glyph_seed).permutation(len(images)), device=images.device)
    stop = args.calibration_images + args.ocf_images
    if stop > len(images):
        parser.error("The disjoint glyph samples must fit inside the training gallery.")
    calibration, held = order[:args.calibration_images], order[args.calibration_images:stop]
    output = {"model_seed": args.seed, "repeat_seed": args.repeat_seed, "glyph_seed": args.glyph_seed,
              "calibration_glyphs": len(calibration), "evaluated_glyphs": len(held), "sessions": args.sessions,
              "nominations": NOMINATIONS, "anchors": ANCHORS, "primary_anchor": 4,
              "v11": {"mice1": "contour + timed Poisson", "mice2": "contour + temporal gain"},
              "protocol": "The outer single response is held out before any model fitting. Rankers learn exact "
                          "within-query OCF differences on different glyphs from responses excluded from the inner "
                          "two-repeat fits. Scalar and local policies use the same candidates and exact labels.", "mice": {}}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    data = root, catalog, images, masks, signature, energy, gallery, ink, scorer
    for mouse in args.mice:
        run_mouse(mouse, data, calibration, held, output, args, save)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
