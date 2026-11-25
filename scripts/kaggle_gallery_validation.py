"""Validate neural ranking of a fixed public font atlas against direct Task 1 reconstruction."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_encoding import count_template_logits, evaluate_glyph_encoder, fit_glyph_encoder
from glyph_font_oracle import FONT_SOURCES, descriptor, fit_font_transform, render_font, thumbnails, warp
from glyph_reconstruction import fit_glyph_reconstructor
from kaggle_diagnostics import feature_view, official_scorer, standardize


def image_descriptors(images):
    return descriptor(F.avg_pool2d(images[:, None].float() / 255, 4))


@dataclass
class FontAtlas:
    images: torch.Tensor
    contours: torch.Tensor


@torch.no_grad()
def build_font_atlas(training_glyphs, device, *, return_codepoints=False):
    """Render the fixed Noto Sans repertoire and fit one global transform on fit glyphs."""
    font, provenance, *codepoints = render_font(*FONT_SOURCES[0], return_codepoints=return_codepoints)
    print(f"Rendered {len(font):,} independent public glyphs", flush=True)
    fitting = (training_glyphs.cpu() >= 128).to(torch.uint8) * 255
    transform, similarity = fit_font_transform(thumbnails(font), thumbnails(fitting))
    images = torch.empty((len(font), 64, 64), dtype=torch.uint8, device=device)
    for offset in range(0, len(font), 512):
        batch = font[offset:offset + 512, None].to(device=device, dtype=torch.float32) / 255
        images[offset:offset + len(batch)] = (warp(batch, transform)[:, 0].clamp(0, 1) * 255).round().to(torch.uint8)
    contours = torch.cat([image_descriptors((part >= 128).to(torch.uint8) * 255)
                          for part in images.split(1024)])
    provenance.update(global_transform=asdict(transform), training_similarity=similarity,
                      style_fit_glyphs=len(training_glyphs))
    return FontAtlas(images, contours), provenance, *codepoints


def standardized(scores):
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)


@torch.no_grad()
def font_scores(atlas, encoded_rates, query_trials, direct_images):
    """Rank public glyphs using only query spikes and the decoder's output."""
    contours = 0.75 * direct_images["refined"].float() + 0.25 * direct_images["refined_gray"].float()
    shape = image_descriptors(contours) @ atlas.contours.T
    rate = count_template_logits(query_trials, encoded_rates, view="rate")
    timed = count_template_logits(query_trials, encoded_rates, view="timed")
    rate_z, timed_z = standardized(rate), standardized(timed)
    neural_z = standardized(0.5 * (rate_z + timed_z))
    fused = 0.5 * (standardized(shape) + neural_z)
    return {"font_contour": shape, "font_poisson_rate": rate, "font_poisson_timed": timed,
            "font_fused": fused}


@torch.no_grad()
def rank_font_atlas(atlas, encoded_rates, query_trials, direct_images, top_k):
    rankings = font_scores(atlas, encoded_rates, query_trials, direct_images)
    predictions = {name: (atlas.images[ranking.argmax(dim=1)] >= 128).to(torch.uint8) * 255
                   for name, ranking in rankings.items()}

    values, shortlist = rankings["font_fused"].topk(top_k, dim=1)
    posterior = (values / 0.5).softmax(dim=1)
    layouts = atlas.contours[shortlist]
    agreement = layouts @ layouts.transpose(1, 2)
    expected_agreement = (agreement @ posterior[:, :, None])[:, :, 0]
    representative = (expected_agreement + 0.15 * posterior).argmax(dim=1)
    candidates = atlas.images[shortlist] >= 128
    rows = torch.arange(len(candidates), device=candidates.device)
    predictions["font_fused_medoid"] = candidates[rows, representative].to(torch.uint8) * 255
    average = (candidates.float() * posterior[:, :, None, None]).sum(dim=1)
    softened = 0.75 * average + 0.25 * direct_images["refined"].float() / 255
    predictions["font_fused_consensus"] = (softened >= 0.42).to(torch.uint8) * 255
    return predictions


def standard_error(values):
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None


@torch.no_grad()
def align_training_glyphs(atlas, images):
    """Pair labeled training images to the nearest public rendering by contour."""
    binary = (images >= 128).to(torch.uint8) * 255
    matches, agreement = [], []
    for batch in binary.split(128):
        similarity = image_descriptors(batch) @ atlas.contours.T
        best, indices = similarity.max(dim=1)
        matches.append(atlas.images[indices])
        agreement.append(best)
    return torch.cat(matches), float(torch.cat(agreement).mean())


def score_block(score, predictions, references, device):
    paired = {}
    for name, images in predictions.items():
        predictions_float = images.detach().cpu().numpy().astype(np.float32) / 255
        paired[name] = np.asarray(score(predictions_float, references, device=str(device), batch_size=4))
    return paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session1", "session4"))
    parser.add_argument("--ocf-per-session", type=int, default=64)
    parser.add_argument("--style-fit-glyphs", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--encoder", choices=("receptive-field", "kernel"), default="receptive-field")
    parser.add_argument("--ranking-weight", type=float, default=0.0,
                        help="compare the receptive-field baseline with additional training-glyph ranking loss")
    parser.add_argument("--receptive-init", choices=("random", "sta"), default="random")
    parser.add_argument("--align-training-to-atlas", action="store_true",
                        help="fit the visual encoder on public renderings matched to labeled fit glyphs")
    parser.add_argument("--refine-epochs", type=int, default=12)
    parser.add_argument("--contour-epochs", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=8, help="candidates for posterior consensus")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/gallery_validation.json"))
    args = parser.parse_args()
    if not 1 <= args.ocf_per_session <= 320 or min(args.style_fit_glyphs, args.encoder_epochs,
                                                    args.refine_epochs, args.top_k) < 1:
        parser.error("Choose positive sample counts and epochs, with at most 320 OCF images per block.")
    if not 0 <= args.contour_epochs <= args.refine_epochs:
        parser.error("Contour epochs must be between zero and the total refinement epochs.")
    if args.ranking_weight < 0 or ((args.ranking_weight or args.receptive_init != "random")
                                   and args.encoder != "receptive-field"):
        parser.error("Use a nonnegative ranking weight with the receptive-field encoder.")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run the public atlas download and neural evaluation in a Kaggle Notebook.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle GPU for the paired gallery experiment.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    score = official_scorer(root)
    train, held = split_glyphs(len(images), 320, args.seed, device)
    rng = np.random.default_rng(args.seed + 1)
    held_queries = rng.choice(len(held), args.ocf_per_session, replace=False)
    queries = held[torch.as_tensor(held_queries, device=device)]
    if args.style_fit_glyphs > len(train):
        parser.error("Style calibration needs fewer glyphs than the training split size.")
    style = np.random.default_rng(args.seed + 2).choice(len(train), args.style_fit_glyphs, replace=False)
    atlas, provenance = build_font_atlas(images[train[torch.as_tensor(style, device=device)]], device)
    if args.top_k > len(atlas.images):
        parser.error("Choose a candidate count no larger than the public atlas.")
    print(json.dumps(provenance), flush=True)
    encoder_images = images[train]
    alignment = None
    if args.align_training_to_atlas:
        encoder_images, alignment = align_training_glyphs(atlas, encoder_images)
        print(f"Mean training contour agreement with the public atlas: {alignment:.4f}", flush=True)

    output = {"device": str(device), "seed": args.seed, "held_glyphs": len(held),
              "ocf_per_session": len(queries), "encoder": args.encoder, "encoder_epochs": args.encoder_epochs,
              "ranking_weight": args.ranking_weight, "receptive_init": args.receptive_init,
              "ranked_held_response": {},
              "aligned_training_contour_agreement": alignment,
              "refine_epochs": args.refine_epochs, "contour_epochs": args.contour_epochs,
              "top_k": args.top_k, "atlas": provenance, "rows": [], "held_response": {}}
    scores_by_variant = {}
    for mouse in args.mice:
        grouped = grouped_blocks(root, catalog, mouse, device, len(images))
        training = {session: trials[train] for session, trials in grouped.items()}
        prototypes, calibration = {}, {}
        for session, trials in training.items():
            normalized, calibration[session] = standardize(feature_view(trials, "evoked-total"))
            prototypes[session] = normalized.mean(dim=1)
        if args.encoder == "kernel":
            from glyph_kernel_encoding import fit_kernel_glyph_encoder

            encoder = fit_kernel_glyph_encoder(training, encoder_images)
        else:
            encoder = fit_glyph_encoder(training, encoder_images, epochs=args.encoder_epochs,
                                        seed=args.seed, receptive_init=args.receptive_init)
        print(f"{mouse}: {args.encoder} encoder trained", flush=True)
        ranked_encoder = None
        if args.ranking_weight:
            ranked_encoder = fit_glyph_encoder(training, encoder_images, epochs=args.encoder_epochs,
                                               seed=args.seed, ranking_weight=args.ranking_weight,
                                               receptive_init=args.receptive_init)
            print(f"{mouse}: ranking-trained encoder ready", flush=True)
        decoder = fit_glyph_reconstructor(prototypes, images[train], seed=args.seed,
                                          refine_epochs=args.refine_epochs, contour_epochs=args.contour_epochs)
        print(f"{mouse}: evoked-total reference decoder trained", flush=True)
        for session in args.sessions:
            trials = grouped[session][queries]
            normalized, _ = standardize(feature_view(trials, "evoked-total"), calibration[session])
            decoded = decoder.predict_variants(session, normalized.mean(dim=1))
            with torch.autocast(device_type=device.type, enabled=args.encoder == "receptive-field"):
                rates = encoder.expected_counts(atlas.images, session, batch_size=256)
            estimated = rank_font_atlas(atlas, rates.float(), trials, decoded, args.top_k)
            if ranked_encoder is not None:
                with torch.autocast(device_type=device.type):
                    ranked_rates = ranked_encoder.expected_counts(atlas.images, session, batch_size=256)
                ranked = font_scores(atlas, ranked_rates.float(), trials, decoded)
                original = font_scores(atlas, rates.float(), trials, decoded)
                for name, values in (("ranked_font_fused", ranked["font_fused"]),
                                     ("ranked_font_poisson_timed", ranked["font_poisson_timed"]),
                                     ("font_encoder_blend", 0.5 * (ranked["font_fused"] + original["font_fused"]))):
                    estimated[name] = (atlas.images[values.argmax(1)] >= 128).to(torch.uint8) * 255
                del ranked_rates
            estimated["evoked_total_refined"] = decoded["refined"]
            # Ground truth enters the experiment only after neural-only selection.
            reference = (images[queries] >= 128).cpu().numpy()
            paired = score_block(score, estimated, reference, device)
            baseline = paired["evoked_total_refined"]
            for variant, values in paired.items():
                delta = values - baseline
                result = {"mouse": mouse, "session": session, "variant": variant,
                          "ocf": float(values.mean()), "ocf_n": len(values),
                          "paired_delta": float(delta.mean()), "paired_delta_se": standard_error(delta)}
                print(json.dumps(result), flush=True)
                output["rows"].append(result)
                scores_by_variant.setdefault(variant, []).append(values)

            block_baseline = np.stack(scores_by_variant["evoked_total_refined"])
            output["paired_summary"] = {}
            for variant, values in scores_by_variant.items():
                blocks = np.stack(values)
                glyph_delta = (blocks - block_baseline).mean(axis=0)
                output["paired_summary"][variant] = {"ocf": float(blocks.mean()), "ocf_n": blocks.size,
                                                      "paired_delta": float(glyph_delta.mean()),
                                                      "paired_delta_se": standard_error(glyph_delta),
                                                      "unique_held_glyphs": len(glyph_delta)}
                if variant in ("ranked_font_fused", "font_encoder_blend"):
                    delta = (blocks - np.stack(scores_by_variant["font_fused"])).mean(axis=0)
                    output["paired_summary"][variant].update(delta_vs_font_fused=float(delta.mean()),
                                                             delta_vs_font_fused_se=standard_error(delta))
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            del rates
        held_trials = {session: grouped[session][held] for session in args.sessions}
        output["held_response"][mouse] = evaluate_glyph_encoder(encoder, images[held], held_trials)
        if ranked_encoder is not None:
            output["ranked_held_response"][mouse] = evaluate_glyph_encoder(ranked_encoder, images[held], held_trials)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        del held_trials
        del encoder, ranked_encoder, decoder, grouped, training


if __name__ == "__main__":
    main()
