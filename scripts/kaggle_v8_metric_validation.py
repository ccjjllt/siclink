"""Validate exact OCF reranking of full v8 Task 1 candidates on held mice1 glyphs."""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, fit_atlas_metric_ranker, pack_atlas_candidates
from glyph_atlas_prior import fit_atlas_prior
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_encoding import fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_mbr import contour_signatures
from glyph_predictive_likelihood import fit_predictive_likelihood
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_exact_rerank_validation import fit_direct, oracle_scores, split_calibration
from kaggle_gallery_validation import (align_training_glyphs, build_font_atlas, font_scores,
                                       score_block, standard_error, standardized)
from kaggle_task1_contour_update import ATLAS_WEIGHT, CONTOUR_WEIGHT, ENCODER_EPOCHS


SESSIONS = ("session1", "session2", "session3", "session4")
NOMINATIONS = {"v8": 16, "ridge_contour": 6, "v4": 6, "noise": 6}
ANCHORS = (2.0, 4.0)


@dataclass
class V8Fit:
    atlas: object
    codepoints: torch.Tensor
    style: object
    prior: object
    contour: object
    encoder: object
    decoder: object
    normalization: dict
    report: dict


def fit_v8(training, images, seed):
    device = images.device
    rows = torch.as_tensor(np.random.default_rng(seed + 2).choice(len(images), 128, replace=False), device=device)
    atlas, provenance, codepoints = build_font_atlas(images[rows], device, return_codepoints=True)
    aligned, agreement = align_training_glyphs(atlas, images)
    prior = fit_atlas_prior(atlas, codepoints, images)
    style, style_fit = fit_aligned_font_style(aligned, images, seed=seed)
    contour = fit_contour_retrieval(training, *contour_signatures(aligned), task="task2", seed=seed)
    signature, energy = contour_signatures(atlas.images)
    contour = replace(contour, signature=signature, energy=energy)
    encoder = fit_glyph_encoder(training, aligned, epochs=ENCODER_EPOCHS, seed=seed)
    decoder, normalization = fit_direct(training, images, seed)
    report = {"glyphs": len(images), "atlas": provenance, "style_fit": style_fit,
              "alignment_agreement": agreement, "prior": prior.diagnostics}
    return V8Fit(atlas, codepoints, style, prior, contour, encoder, decoder, normalization, report)


@torch.no_grad()
def atlas_scores(model, likelihood, counts, session, *, rates=None):
    """Calculate the complete atlas scores used by the v8 proposal policy."""
    normalized, _ = standardize(feature_view(counts, "evoked-total"), model.normalization[session])
    direct = model.decoder.predict_variants(session, normalized.mean(dim=1))
    if rates is None:
        with torch.autocast(device_type="cuda"):
            rates = model.encoder.expected_counts(model.atlas.images, session, batch_size=192).float()
    base = font_scores(model.atlas, rates, counts, direct)
    v4 = standardized(base["font_fused"])
    ridge = contour_retrieval_scores(model.contour, counts, session)["contour_ridge"]
    v8 = ATLAS_WEIGHT * v4 + CONTOUR_WEIGHT * standardized(ridge)
    noise = likelihood.logits(counts, rates, include_model_error=False)
    predictive = likelihood.logits(counts, rates, include_model_error=True)
    noise_blend = 0.5 * (standardized(base["font_contour"]) + standardized(noise))
    scores = {name: model.prior.variants(raw)["usage_unseen"] for name, raw in (
        ("v8", v8), ("ridge_contour", ridge), ("v4", v4), ("noise", noise_blend))}
    scores.update({"poisson_rate": base["font_poisson_rate"], "poisson_timed": base["font_poisson_timed"],
                   "decoded_contour": base["font_contour"], "ridge_contour_raw": ridge,
                   "predictive_error": predictive, "noise_raw": noise,
                   "lexical": model.prior.lexical_log.expand_as(v8), "unseen": model.prior.unseen_log.expand_as(v8)})
    return scores


@torch.no_grad()
def shortlist(model, likelihood, counts, session):
    scores = atlas_scores(model, likelihood, counts, session)
    # Preserve the complete v8 argmax before the ranker's clipped feature scale.
    baseline = scores["v8"].argmax(dim=1)
    return pack_atlas_candidates(scores, NOMINATIONS, baseline="v8"), baseline


@torch.no_grad()
def rendered_bank(model):
    batches = (model.style.apply(batch[:, None].float() / 255)[:, 0] >= 0.5
               for batch in model.atlas.images.split(1024))
    return torch.cat([mask.to(torch.uint8) * 255 for mask in batches])


def nested_ranker(training, images, split, scorer, seed):
    metric_rows, noise_rows, inner_rows = split
    inner = {session: counts[inner_rows] for session, counts in training.items()}
    model = fit_v8(inner, images[inner_rows], seed)
    noise_glyphs, _ = align_training_glyphs(model.atlas, images[noise_rows])
    likelihoods = {name: fit_predictive_likelihood(training[name][noise_rows],
                                                 model.encoder.expected_counts(noise_glyphs, name))
                   for name in SESSIONS}
    rendered = rendered_bank(model)
    reference = (images[metric_rows] >= 128).cpu().numpy()
    candidates, labels = [], []
    for name in SESSIONS:
        shortlist_rows, _ = shortlist(model, likelihoods[name], training[name][metric_rows], name)
        candidates.append(shortlist_rows)
        labels.append(oracle_scores(shortlist_rows, rendered, reference, scorer, images.device))
        print(f"{name}: scored {int(shortlist_rows.valid.sum()):,} nested v8 candidate pairs", flush=True)
    ranker = fit_atlas_metric_ranker(concat_atlas_candidates(candidates), torch.cat(labels), seed=seed)
    return ranker, likelihoods, model.report, model.codepoints.cpu()


def summarize(values, baseline):
    values = np.stack(values)
    delta = (values - np.stack(baseline)).mean(axis=0)
    return {"ocf": float(values.mean()), "paired_delta": float(delta.mean()), "paired_se": standard_error(delta),
            "ocf_n": values.size, "independent_held_glyphs": len(delta)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=961)
    parser.add_argument("--avoid-held-seed", type=int, help="exclude a previous 320-glyph holdout from this holdout")
    parser.add_argument("--metric-glyphs", type=int, default=96)
    parser.add_argument("--noise-glyphs", type=int, default=384)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--oracle-rows", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/v8_metric_validation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle GPU and its mounted competition data.")
    if not 0 <= args.oracle_rows <= args.ocf_images <= 320 or min(args.ocf_images, args.metric_glyphs, args.noise_glyphs) < 2:
        parser.error("Choose valid nested calibration counts and a paired held subset.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    if args.avoid_held_seed is None:
        fit, held = split_glyphs(len(images), 320, args.seed, device)
    else:
        previous = np.random.default_rng(args.avoid_held_seed).permutation(len(images))[:320]
        order = np.random.default_rng(args.seed).permutation(len(images))
        evaluation = np.sort(order[~np.isin(order, previous)][:320])
        fit = torch.as_tensor(np.setdiff1d(np.arange(len(images)), evaluation), device=device)
        held = torch.as_tensor(evaluation, device=device)
    if args.metric_glyphs + args.noise_glyphs > len(fit) - 512:
        parser.error("Reserve at least 512 glyphs for the inner v8 fit.")
    split = split_calibration(len(fit), args.metric_glyphs, args.noise_glyphs, device, args.seed)
    rows = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_images, replace=False)
    queries = held[torch.as_tensor(rows, device=device)]
    data = grouped_blocks(root, catalog, "mice1", device, len(images))
    training = {name: counts[fit] for name, counts in data.items()}
    scorer = official_scorer(root)
    print("Fitting the nested v8 anchor and independent exact OCF calibration", flush=True)
    ranker, likelihoods, nested, codepoints = nested_ranker(training, images[fit], split, scorer, args.seed)
    print("Refitting the complete v8 anchor on outer-fit glyphs", flush=True)
    outer = fit_v8(training, images[fit], args.seed)
    if not torch.equal(codepoints, outer.codepoints.cpu()):
        raise ValueError("Nested and outer atlas need identical public codepoint order.")
    rendered = rendered_bank(outer)
    output = {"mouse": "mice1", "seed": args.seed, "avoid_held_seed": args.avoid_held_seed,
              "held_glyphs": len(held), "ocf_images_per_session": len(queries),
              "atlas_weight": ATLAS_WEIGHT, "directional_contour_weight": CONTOUR_WEIGHT,
              "nominations": NOMINATIONS, "anchors": ANCHORS, "noise_calibration_glyphs": len(split[1]),
              "metric_calibration_glyphs": len(split[0]), "nested_fit": nested, "outer_fit": outer.report,
              "ranker": ranker.diagnostics, "protocol": "Each nested scoring source, including the atlas geometry, style and prior, uses only inner-fit glyphs. Noise and exact OCF metric calibration use separate labeled training glyphs. The full 60/40 v8 control is refit on all outer-fit glyphs; untouched outer glyphs are scored only after model ranking. Oracle shortlist OCF is diagnostic only.",
              "blocks": []}
    accumulation, oracle = {}, {}
    reference = (images[queries] >= 128).cpu().numpy()
    for session in SESSIONS:
        candidates, baseline = shortlist(outer, likelihoods[session], data[session][queries], session)
        selected = {"v8": baseline, **{f"anchor_{anchor:g}": ranker.select(candidates, anchor_ocf=anchor)
                                     for anchor in ANCHORS}}
        predictions = {name: rendered[indices] for name, indices in selected.items()}
        values = score_block(scorer, predictions, reference, device)
        report = {"session": session, "noise_fit": likelihoods[session].diagnostics,
                  "mean_union_size": float(candidates.valid.sum(dim=1).float().mean()), "metrics": {}}
        for name, measured in values.items():
            accumulation.setdefault(name, []).append(measured)
            report["metrics"][name] = {**summarize([measured], [values["v8"]]),
                                       "changed": int((selected[name] != baseline).sum())}
        if args.oracle_rows:
            held_candidates = candidates.take_rows(slice(args.oracle_rows))
            exact = oracle_scores(held_candidates, rendered, reference[:args.oracle_rows], scorer, device)
            exact = exact.masked_fill(~held_candidates.valid, -torch.inf)
            best_v8 = exact[:, :NOMINATIONS["v8"]].amax(dim=1).cpu().numpy()
            best_union = exact.amax(dim=1).cpu().numpy()
            oracle.setdefault("v8_top16", []).append(best_v8)
            oracle.setdefault("union", []).append(best_union)
            report["oracle"] = {"n": len(best_v8), "v8_top16": float(best_v8.mean()),
                                "union": float(best_union.mean())}
            output["oracle_summary"] = {name: summarize(rows, oracle["v8_top16"]) for name, rows in oracle.items()}
        output["blocks"].append(report)
        output["summary"] = {name: summarize(rows, accumulation["v8"]) for name, rows in accumulation.items()}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
