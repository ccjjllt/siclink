"""Compare repeat weighting with exact held-glyph OCF and spike-only split-half retrieval."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_mbr import contour_signatures
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, standardized
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_task1_test_repeatability import fit_signal_space, inspect_split
from kaggle_v8_metric_validation import summarize


PREVIOUS_OUTER = (961, 2209, 613, 3307, 8723, 5573, 6379)
MODES = ("uniform", "central", "shuffled")
DIMS = 64
MODEL_SEED = 211


def encode_trials(counts, space):
    normalization, projection, _ = space
    values, _ = standardize(feature_view(counts, "evoked-total"), normalization)
    return values @ projection[:, -DIMS:]


def repeat_weights(values, generator):
    repeats = values.shape[1]
    off_diagonal = ~torch.eye(repeats, dtype=torch.bool, device=values.device)
    distance = torch.cdist(values, values)[:, off_diagonal].view(len(values), repeats, repeats - 1)
    centrality = distance.median(dim=-1).values
    typical = centrality.quantile(0.5, dim=1, keepdim=True).clamp_min(1e-6)
    relative = (centrality / typical - 1).clamp_min(0)
    attenuation = (-2 * relative).exp().clamp_min(0.4)
    central = attenuation / attenuation.sum(dim=1, keepdim=True)
    permutation = torch.rand(attenuation.shape, generator=generator, device=values.device).argsort(dim=1)
    shuffled = central.gather(1, permutation)
    return {"uniform": torch.full_like(central, 1 / repeats), "central": central, "shuffled": shuffled}


def pool(values, weights):
    return (values * weights[..., None]).sum(dim=1)


def neural_readout(query, rates, contour, session, weights):
    shape = standardized(contour_retrieval_scores(contour, query, session, repeat_weights=weights)["contour_ridge"])
    pooled_counts = pool(query, weights)[:, None]
    rate, timed = [standardized(count_template_logits(pooled_counts, rates, view)) for view in ("rate", "timed")]
    signal = standardized(0.5 * (rate + timed))
    return 0.65 * shape + 0.35 * signal


@torch.no_grad()
def eval_gallery(root, images, sample, groups, spaces, contour, encoder, atlas, prior, rendered, seed):
    rng = torch.Generator().manual_seed(seed + 149)
    scorer = PairedOCF(official_scorer(root), rendered, (images[sample] >= 128).numpy(), images.device, 12)
    session_scores = {mode: [] for mode in MODES}
    pooled_order = {}
    report = {}
    for session, counts in groups.items():
        query = counts[sample]
        rates = encoder.expected_counts(atlas.images, session, batch_size=192)
        weights = repeat_weights(encode_trials(query, spaces[session]), rng)
        current = {}
        for name in MODES:
            score = neural_readout(query, rates, contour, session, weights[name])
            choice = prior.variants(score)["usage_unseen"].argmax(dim=1)
            session_scores[name].append(scorer.evaluate(choice[:, None])[:, 0])
            current[name] = session_scores[name][-1]
            if name in pooled_order:
                pooled_order[name].add_(score)
            else:
                pooled_order[name] = score
        report[session] = {name: summarize([values], [current["uniform"]]) for name, values in current.items()}
        report[session]["mean_effective_repeats"] = float(weights["central"].square().sum(1).reciprocal().mean())
        print(json.dumps({"session": session, "held_ocf": report[session]}), flush=True)
    # The cross-session order is known for training-only held glyphs. Each
    # recording is scored independently before accumulating its evidence.
    pooled_scores = {}
    for name, score in pooled_order.items():
        choice = prior.variants(score / len(groups))["usage_unseen"].argmax(dim=1)
        pooled_scores[name] = scorer.evaluate(choice[:, None])[:, 0]
    return {"sessions": report,
            "four_repeats": {name: summarize(values, session_scores["uniform"]) for name, values in session_scores.items()},
            "four_repeats_central_vs_shuffled": summarize(session_scores["central"], session_scores["shuffled"]),
            "sixteen_known_aligned_repeats": {name: summarize([values], [pooled_scores["uniform"]])
                                              for name, values in pooled_scores.items()},
            "exact_pairs": len(scorer.values)}


@torch.no_grad()
def formal_split_diagnostic(root, catalog, sample_count, fit, seed, device):
    result = {}
    rng = torch.Generator(device=device).manual_seed(seed + 433)
    entries = [entry for entry in catalog["blocks"] if entry["task"] == "task1" and entry["split"] == "test"]
    for mouse in ("mice1", "mice2"):
        groups = grouped_blocks(root, catalog, mouse, device, sample_count)
        sessions = {}
        for entry in sorted((item for item in entries if item["mouse"] == mouse), key=lambda item: item["session"]):
            session = entry["session"]
            space = fit_signal_space(feature_view(groups[session][fit], "evoked-total"))
            counts = read_trials(root, catalog, entry, device).counts
            if counts.shape[1] != 20:
                raise ValueError("This comparison requires the supplied 20-repeat Task 1 groups.")
            values = encode_trials(counts, space)
            reports = []
            for _ in range(2):
                order = torch.rand(values.shape[:2], generator=rng, device=device).argsort(dim=1)
                shuffled = values.gather(1, order[..., None].expand_as(values))
                left, right = shuffled[:, :10], shuffled[:, 10:]
                l_weight, r_weight = repeat_weights(left, rng), repeat_weights(right, rng)
                outcomes = {name: inspect_split(pool(left, l_weight[name]), pool(right, r_weight[name]),
                                                space[-1][-DIMS:], 10) for name in MODES}
                effective = torch.cat([weights["central"].square().sum(1).reciprocal() for weights in (l_weight, r_weight)])
                outcomes["mean_effective_repeats"] = float(effective.mean())
                reports.append(outcomes)
            sessions[session] = reports
        result[mouse] = sessions
        del groups
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7079)
    parser.add_argument("--ocf-images", type=int, default=112)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_robust_repeat_pool.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_images <= 320:
        parser.error("Choose a valid paired OCF sample size.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run against the mounted Kaggle competition input.")
    torch.set_num_threads(4)
    torch.manual_seed(MODEL_SEED)
    started = time.monotonic()
    root = locate_data()
    device = torch.device("cpu")
    images, catalog = load_images(root, device), load_catalog(root)
    fit, held = outer_split(len(images), args.seed, PREVIOUS_OUTER, device)
    rng = np.random.default_rng(args.seed)
    formal = formal_split_diagnostic(root, catalog, len(images), fit, args.seed, device)
    print(json.dumps({"official_grouped_repeat_halves": formal}), flush=True)

    groups = grouped_blocks(root, catalog, "mice1", device, len(images))
    training = {session: counts[fit] for session, counts in groups.items()}
    spaces = {session: fit_signal_space(feature_view(counts, "evoked-total")) for session, counts in training.items()}
    sample = held[torch.as_tensor(rng.permutation(len(held))[:args.ocf_images])]
    train_images = images[fit]
    atlas, provenance, codepoints = build_font_atlas(train_images[rng.choice(len(fit), 128, replace=False)],
                                                    device, return_codepoints=True)
    aligned, _ = align_training_glyphs(atlas, train_images)
    style, _ = fit_aligned_font_style(aligned, train_images, seed=MODEL_SEED)
    rendered = torch.cat([(style.apply(batch[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                          for batch in atlas.images.split(1024)])
    prior = fit_atlas_prior(atlas, codepoints, train_images)
    contour = fit_contour_retrieval(training, *contour_signatures(aligned), task="task2", seed=MODEL_SEED)
    signature, energy = contour_signatures(atlas.images)
    contour = replace(contour, signature=signature, energy=energy)
    encoder = fit_glyph_encoder(training, aligned, epochs=18, seed=MODEL_SEED)
    del training
    gallery = eval_gallery(root, images, sample, groups, spaces, contour, encoder, atlas, prior, rendered, args.seed)
    result = {"seed": args.seed, "fit_glyphs": len(fit), "held_glyphs": len(held), "atlas": provenance,
              "ocf_per_session": len(sample), "signal_space_dimensions": DIMS,
              "protocol": "Public atlas and weights fit only on labeled fitting glyphs. Relative within-group "
                          "neural distances downweight less representative repeated trials. Shuffle trial weights "
                          "within each group to preserve effective repeat count. Official Task 1 uses only provided "
                          "within-example grouping to match independent halves, no sample identifiers or images.",
              "official_grouped_repeat_halves": formal, "held_open_gallery": gallery,
              "seconds": time.monotonic() - started}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {result['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
