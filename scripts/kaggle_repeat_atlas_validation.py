"""Compare repeat-aware Task 1 atlas scoring and unlabeled split-half stability."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials, split_glyphs
from glyph_encoding import fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_predictive_likelihood import fit_predictive_likelihood
from glyph_reconstruction import fit_glyph_reconstructor
from glyph_repeat_atlas_likelihood import RepeatAtlasLikelihood
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, font_scores, standard_error, standardized


SESSIONS = ("session1", "session2", "session3", "session4")
CONTROL = "candidate_v4"


def permute_repeats(counts, rng):
    order = torch.as_tensor(rng.random(counts.shape[:2]).argsort(axis=1), device=counts.device)
    return counts.gather(1, order[:, :, None].expand_as(counts))


@torch.no_grad()
def rank_counts(raw, expected, session, decoder, normalization, likelihood, atlas, prior, depth):
    response, _ = standardize(feature_view(raw, "evoked-total"), normalization)
    direct = decoder.predict_variants(session, response.mean(dim=1))
    scores = font_scores(atlas, expected, raw, direct)
    original = scores["font_fused"]
    shape = standardized(scores["font_contour"])
    contenders = {CONTROL: original}
    contenders.update({name: 0.62 * original + 0.38 * (0.5 * (shape + standardized(values)))
                       for name, values in likelihood.logits(raw, expected).items()})
    return {name: prior.variants(values)["usage_unseen"].topk(depth, dim=1).indices
            for name, values in contenders.items()}


def exact_held(ranked, rendered, reference, scorer, device):
    keys = [(half, name) for half, variants in ranked.items() for name in variants]
    selected = torch.stack([ranked[half][name][:, 0] for half, name in keys], dim=1).cpu().numpy()
    pairs = np.column_stack((np.repeat(np.arange(len(reference)), len(keys)), selected.ravel()))
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    prediction = rendered[torch.as_tensor(unique[:, 1], device=device)].cpu().numpy()
    scores = scorer(prediction, reference[unique[:, 0]], device=str(device), batch_size=12)
    values = np.asarray(scores)[inverse].reshape(selected.shape)
    return {key: values[:, position] for position, key in enumerate(keys)}


def paired_summary(scores, baseline):
    observations = np.stack(scores)
    delta = (observations - np.stack(baseline)).mean(axis=0)
    return {"ocf": float(observations.mean()), "baseline_ocf": float(np.mean(baseline)), "n": observations.size,
            "independent_glyphs": len(delta), "paired_delta": float(delta.mean()), "paired_se": standard_error(delta)}


def stability(left, right):
    k = left.shape[1]
    matches = left[:, :, None] == right[:, None, :]
    overlap = matches.any(dim=2).sum(dim=1).float()
    symmetric = 0.5 * (matches[:, 0, :min(k, 5)].any(1).float() + matches[:, :min(k, 5), 0].any(1).float())
    return {"queries": len(left), "shortlist": k, "top1_agreement": float((left[:, 0] == right[:, 0]).float().mean()),
            "winner_in_peer_top5": float(symmetric.mean()), "mean_shortlist_overlap": float((overlap / k).mean()),
            "mean_shortlist_jaccard": float((overlap / (2 * k - overlap)).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", choices=SESSIONS, default=("session1", "session4"))
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--calibration-glyphs", type=int, default=384)
    parser.add_argument("--ocf-images", type=int, default=48)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--refine-epochs", type=int, default=12)
    parser.add_argument("--contour-epochs", type=int, default=3)
    parser.add_argument("--shortlist", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/repeat_atlas_validation.json"))
    args = parser.parse_args()
    if (not 1 <= args.ocf_images <= 320 or not 2 <= args.calibration_glyphs <= 2300
            or min(args.encoder_epochs, args.refine_epochs, args.shortlist) < 1
            or not 0 <= args.contour_epochs <= args.refine_epochs):
        parser.error("Choose valid fit, calibration and held evaluation sizes.")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Use Kaggle CPU or CUDA for this diagnostic.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    fit, held = split_glyphs(len(images), 320, args.seed, device)
    fit_images = images[fit]
    metric_rows = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_images, replace=False)
    queries = held[torch.as_tensor(metric_rows, device=device)]
    style_rows = np.random.default_rng(args.seed + 2).choice(len(fit), 128, replace=False)
    calibration = np.random.default_rng(args.seed + 17).choice(len(fit), args.calibration_glyphs, replace=False)
    calibration = torch.as_tensor(calibration, device=device)
    remaining = torch.ones(len(fit), dtype=torch.bool, device=device)
    remaining[calibration] = False
    atlas, provenance, codepoints = build_font_atlas(fit_images[style_rows], device, return_codepoints=True)
    if args.shortlist > len(atlas.images):
        parser.error("Shortlist depth exceeds the public atlas size.")
    aligned, agreement = align_training_glyphs(atlas, fit_images)
    style, style_fit = fit_aligned_font_style(aligned, fit_images, seed=args.seed)
    prior = fit_atlas_prior(atlas, codepoints, fit_images)
    with torch.no_grad():
        rendered = torch.cat([style.apply(batch[:, None].float() / 255)[:, 0] >= 0.5
                              for batch in atlas.images.split(1024)])
    score = official_scorer(root)
    test = {(item["mouse"], item["session"]): item for item in catalog["blocks"] if item["task"] == "task1"}
    output = {"seed": args.seed, "fit_glyphs": len(fit), "calibration_glyphs": len(calibration),
              "held_glyphs": len(held), "ocf_images": len(queries), "atlas": provenance, "style": style_fit,
              "prior": prior.diagnostics, "atlas_fit_agreement": agreement, "blocks": [],
              "protocol": "Aligned public-atlas encoder and prior use fit training glyphs only. Trial variance and irreducible code residual are fitted with nested predictions made without noise-calibration glyphs. Held labels score official OCF for two disjoint two-repeat variants and all four repeats. Official Task 1 uses two random non-overlapping ten-repeat halves and reports only candidate agreement, with no test accuracy or quality inference."}
    held_rng = np.random.default_rng(args.seed + 21)
    test_rng = np.random.default_rng(args.seed + 27)
    accumulated = {2: {}, 4: {}}
    for mouse in ("mice1", "mice2"):
        groups = grouped_blocks(root, catalog, mouse, device, len(images))
        training = {block: counts[fit] for block, counts in groups.items()}
        subset = {block: counts[remaining] for block, counts in training.items()}
        nested = fit_glyph_encoder(subset, aligned[remaining], epochs=args.encoder_epochs, seed=args.seed)
        likelihood = {block: RepeatAtlasLikelihood(fit_predictive_likelihood(
            training[block][calibration], nested.expected_counts(aligned[calibration], block)))
            for block in args.sessions}
        del subset, nested
        encoder = fit_glyph_encoder(training, aligned, epochs=args.encoder_epochs, seed=args.seed)
        prototypes, statistics = {}, {}
        for block, counts in training.items():
            normalized, statistics[block] = standardize(feature_view(counts, "evoked-total"))
            prototypes[block] = normalized.mean(dim=1)
        decoder = fit_glyph_reconstructor(prototypes, fit_images, seed=args.seed,
                                          refine_epochs=args.refine_epochs, contour_epochs=args.contour_epochs)
        for block in dict.fromkeys(args.sessions):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                expected = encoder.expected_counts(atlas.images, block, batch_size=256).float()
            observed = permute_repeats(groups[block][queries], held_rng)
            held_rank = {name: rank_counts(part, expected, block, decoder, statistics[block], likelihood[block], atlas, prior, 1)
                         for name, part in (("2a", observed[:, :2]), ("2b", observed[:, 2:]), ("4", observed))}
            reference = (images[queries] >= 128).cpu().numpy()
            exact = exact_held(held_rank, rendered, reference, score, device)
            block_report = {"mouse": mouse, "session": block, "noise_fit": likelihood[block].calibration.diagnostics,
                            "held": {}, "official_stability": {}}
            for repeat_count, condition in ((2, ("2a", "2b")), (4, ("4",))):
                block_report["held"][repeat_count] = {}
                baseline = [exact[half, CONTROL] for half in condition]
                for method in held_rank["4"]:
                    values = [exact[half, method] for half in condition]
                    block_report["held"][repeat_count][method] = paired_summary(values, baseline)
                    accumulated[repeat_count].setdefault(method, []).extend(values)

            counts = read_trials(root, catalog, test[mouse, block], device).counts
            if counts.shape[1] != 20:
                raise ValueError("Expected the official twenty-repeat Task 1 groups")
            order = torch.as_tensor(test_rng.permutation(len(counts)), device=device)
            left, right = permute_repeats(counts[order], test_rng).chunk(2, dim=1)
            test_rank = [rank_counts(half, expected, block, decoder, statistics[block], likelihood[block], atlas, prior,
                                     args.shortlist) for half in (left, right)]
            block_report["official_stability"] = {method: stability(test_rank[0][method], test_rank[1][method])
                                                  for method in test_rank[0]}
            output["blocks"].append(block_report)
            output["held_paired_summary"] = {repeats: {method: paired_summary(values, methods[CONTROL])
                                                      for method, values in methods.items()}
                                              for repeats, methods in accumulated.items()}
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(block_report), flush=True)
        del groups, training, encoder, decoder
    output["official_stability_mean"] = {
        name: {key: float(np.mean([block["official_stability"][name][key] for block in output["blocks"]]))
               for key in ("top1_agreement", "winner_in_peer_top5", "mean_shortlist_overlap", "mean_shortlist_jaccard")}
        for name in output["blocks"][0]["official_stability"]}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
