"""Fit an OCF-aware Task 1 reranker using nested training glyphs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, fit_atlas_metric_ranker, pack_atlas_candidates
from glyph_atlas_prior import fit_atlas_prior
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_encoding import fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_predictive_likelihood import fit_predictive_likelihood
from glyph_reconstruction import fit_glyph_reconstructor
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_gallery_validation import (align_training_glyphs, build_font_atlas, font_scores,
                                       score_block, standard_error, standardized)


SHORTLIST = {"v4": 12, "noise": 12, "blend": 8}


def fit_direct(training, images, seed):
    prototypes, stats = {}, {}
    for name, counts in training.items():
        features, stats[name] = standardize(feature_view(counts, "evoked-total"))
        prototypes[name] = features.mean(1)
    model = fit_glyph_reconstructor(prototypes, images, seed=seed, refine_epochs=12, contour_epochs=3)
    return model, stats


@torch.no_grad()
def shortlist(atlas, prior, encoder, decoder, stats, likelihood, counts, session):
    features, _ = standardize(feature_view(counts, "evoked-total"), stats[session])
    direct = decoder.predict_variants(session, features.mean(1))
    with torch.autocast(device_type="cuda"):
        rates = encoder.expected_counts(atlas.images, session, batch_size=256).float()
    base = font_scores(atlas, rates, counts, direct)
    shape = standardized(base["font_contour"])
    noise = standardized(likelihood.logits(counts, rates, include_model_error=False))
    modeled = standardized(likelihood.logits(counts, rates, include_model_error=True))
    v4 = base["font_fused"]
    blend = 0.62 * v4 + 0.19 * (shape + modeled)
    scores = {"v4": prior.variants(v4)["usage_unseen"],
              "noise": prior.variants(0.5 * (shape + noise))["usage_unseen"],
              "blend": prior.variants(blend)["usage_unseen"],
              "shape": base["font_contour"], "poisson_rate": base["font_poisson_rate"],
              "poisson_timed": base["font_poisson_timed"], "pred_error": modeled,
              "lexical": prior.lexical_log.expand_as(v4), "unseen": prior.unseen_log.expand_as(v4)}
    return pack_atlas_candidates(scores, SHORTLIST, baseline="v4")


def oracle_scores(candidates, bank, refs, scorer, device):
    rows, columns = candidates.valid.nonzero(as_tuple=True)
    glyphs = candidates.indices[rows, columns]
    examples = (bank[glyphs] >= 128).cpu().numpy()
    labels = scorer(examples, refs[rows.cpu().numpy()], device=str(device), batch_size=12)
    result = torch.full(candidates.indices.shape, float("nan"), device=device)
    result[rows, columns] = torch.as_tensor(labels, dtype=result.dtype, device=device)
    return result


def split_calibration(image_count, metric_count, noise_count, device, seed):
    order = torch.as_tensor(np.random.default_rng(seed + 18).permutation(image_count), device=device)
    boundary = metric_count + noise_count
    return order[:metric_count], order[metric_count:boundary], order[boundary:]


def fit_calibrated_ranker(atlas, codepoints, fit_images, aligned, training, split, rendered, scorer, sessions, seed):
    """Fit neural, noise, and metric stages on disjoint groups of training glyphs."""
    metric, noise, inner = split
    device = fit_images.device
    prior = fit_atlas_prior(atlas, codepoints, fit_images[inner])
    nested = {name: counts[inner] for name, counts in training.items()}
    encoder = fit_glyph_encoder(nested, aligned[inner], epochs=18, seed=seed)
    decoder, stats = fit_direct(nested, fit_images[inner], seed)
    likelihoods = {name: fit_predictive_likelihood(training[name][noise], encoder.expected_counts(aligned[noise], name))
                   for name in sessions}
    reference = (fit_images[metric] >= 128).cpu().numpy()
    batches, labels = [], []
    for name in sessions:
        candidates = shortlist(atlas, prior, encoder, decoder, stats, likelihoods[name], training[name][metric], name)
        batches.append(candidates)
        labels.append(oracle_scores(candidates, rendered, reference, scorer, device))
        print(f"{name}: scored {int(candidates.valid.sum())} fit-only glyph pairs", flush=True)
    packed = concat_atlas_candidates(batches)
    calibration = torch.cat(labels)
    ranker = fit_atlas_metric_ranker(packed, calibration, seed=seed)
    diagnostics = {"calibration_pairs": int(packed.valid.sum()), "calibration_queries": len(packed.valid),
                   "calibration_candidate_mean_ocf": float(calibration[packed.valid].mean()),
                   "ranker": ranker.diagnostics}
    return ranker, likelihoods, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session1", "session4"))
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--metric-glyphs", type=int, default=96)
    parser.add_argument("--noise-glyphs", type=int, default=384)
    parser.add_argument("--ocf-images", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/exact_atlas_validation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a Kaggle GPU Notebook with the mounted competition input.")
    if min(args.metric_glyphs, args.noise_glyphs, args.ocf_images) < 2 or args.ocf_images > 320:
        parser.error("Provide positive disjoint metric and noise calibration counts, and at most 320 held images.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    fit, held = split_glyphs(len(images), 320, args.seed, device)
    if args.metric_glyphs + args.noise_glyphs > len(fit) - 512:
        parser.error("Leave at least 512 glyphs to fit the nested encoders.")
    split = split_calibration(len(fit), args.metric_glyphs, args.noise_glyphs, device, args.seed)
    metric, noise, inner = split
    style_rows = torch.as_tensor(np.random.default_rng(args.seed + 2).choice(len(inner), 128, replace=False), device=device)
    held_rows = torch.as_tensor(np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_images, replace=False), device=device)
    atlas, provenance, codepoints = build_font_atlas(images[fit[inner[style_rows]]], device, return_codepoints=True)
    aligned, agreement = align_training_glyphs(atlas, images[fit])
    style, style_stats = fit_aligned_font_style(aligned[inner], images[fit[inner]], seed=args.seed)
    prior = fit_atlas_prior(atlas, codepoints, images[fit])
    rendered = torch.cat([(style.apply(batch[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                          for batch in atlas.images.split(1024)])
    scorer = official_scorer(root)
    output = {"atlas": provenance, "training_alignment_agreement": agreement, "style_fit": style_stats,
              "seed": args.seed, "inner_glyphs": len(inner), "metric_calibration_glyphs": len(metric),
              "noise_calibration_glyphs": len(noise), "held_glyphs": len(held), "shortlist": SHORTLIST,
              "protocol": "Disjoint nested prediction, covariance, and exact OCF calibration glyphs. Outer held glyphs do not fit any model or prior.",
              "mice": {}, "blocks": []}
    results = {}
    for mouse in args.mice:
        train = grouped_blocks(root, catalog, mouse, device, len(images))
        outer = {name: counts[fit] for name, counts in train.items()}
        ranker, likelihoods, output["mice"][mouse] = fit_calibrated_ranker(
            atlas, codepoints, images[fit], aligned, outer, split, rendered, scorer, args.sessions, args.seed)

        encoder = fit_glyph_encoder(outer, aligned, epochs=18, seed=args.seed)
        decoder, stats = fit_direct(outer, images[fit], args.seed)
        for name in args.sessions:
            candidates = shortlist(atlas, prior, encoder, decoder, stats, likelihoods[name],
                                   train[name][held[held_rows]], name)
            baseline = candidates.indices.gather(1, candidates.baseline_z.masked_fill(~candidates.valid, -torch.inf)
                                                .argmax(dim=1, keepdim=True))[:, 0]
            selected = {"v4": baseline, "exact_ocf_rank": ranker.select(candidates, anchor_ocf=0.75),
                        "exact_ocf_anchored": ranker.select(candidates, anchor_ocf=2.0)}
            predictions = {key: rendered[glyphs] for key, glyphs in selected.items()}
            reference = (images[held[held_rows]] >= 128).cpu().numpy()
            measured = score_block(scorer, predictions, reference, device)
            row = {"mouse": mouse, "session": name, "variants": {}}
            for key, scores in measured.items():
                delta = scores - measured["v4"]
                results.setdefault(key, []).append(scores)
                row["variants"][key] = {"ocf": float(scores.mean()), "paired_delta": float(delta.mean()),
                                        "paired_se": standard_error(delta), "n": len(scores)}
            output["blocks"].append(row)
            print(json.dumps(row), flush=True)
            output["summary"] = {}
            for key, scores in results.items():
                values = np.stack(scores)
                paired = (values - np.stack(results["v4"])).mean(0)
                output["summary"][key] = {"ocf": float(values.mean()), "paired_delta": float(paired.mean()),
                                          "paired_se": standard_error(paired), "n": values.size}
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        del train, outer, encoder, decoder


if __name__ == "__main__":
    main()
