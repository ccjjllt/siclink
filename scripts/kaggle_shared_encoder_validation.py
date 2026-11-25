"""Compare single-mouse encoding with a shared visual core on unseen glyphs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_exact_rerank_validation import fit_direct
from kaggle_gallery_validation import (align_training_glyphs, build_font_atlas, font_scores,
                                       image_descriptors, score_block, standard_error)


MICE = ("mice1", "mice2")


def joined_responses(training):
    """Align training stimulus means while giving every recorded unit its own head."""
    widths = {mouse: next(iter(training[mouse].values())).shape[-1] for mouse in MICE}
    by_day = {session: torch.cat([training[mouse][session].mean(dim=1) for mouse in MICE], dim=-1)[:, None]
              for session in training[MICE[0]]}
    positions = {MICE[0]: slice(0, widths[MICE[0]]), MICE[1]: slice(widths[MICE[0]], sum(widths.values()))}
    return by_day, positions


@torch.no_grad()
def response_accuracy(encoder, positions, images, session, observed):
    expected = encoder.expected_counts(images, session, batch_size=256)[:, positions].float()
    scores = count_template_logits(observed, expected, "rate")
    candidates = scores.topk(5, dim=1).indices
    labels = torch.arange(len(images), device=images.device)
    return {"top1": float((candidates[:, 0] == labels).float().mean()),
            "top5": float((candidates == labels[:, None]).any(dim=1).float().mean()), "n": len(images)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session1", "session4"))
    parser.add_argument("--ocf-images", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--joint-fields", type=int, default=4)
    parser.add_argument("--single-wide-control", action="store_true",
                        help="also fit the same number of spatial readouts on each mouse separately")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/shared_encoder_validation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a Kaggle GPU Notebook with competition input.")
    if not 1 <= args.ocf_images <= 320 or min(args.epochs, args.joint_fields) < 1:
        parser.error("Choose positive encoding parameters and 1–320 evaluation glyphs.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    fit, held = split_glyphs(len(images), 320, args.seed, device)
    fit_images = images[fit]
    selected = torch.as_tensor(np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_images, replace=False),
                               device=device)
    style_rows = torch.as_tensor(np.random.default_rng(args.seed + 2).choice(len(fit), 128, replace=False), device=device)
    atlas, provenance, codepoints = build_font_atlas(fit_images[style_rows], device, return_codepoints=True)
    aligned, agreement = align_training_glyphs(atlas, fit_images)
    prior = fit_atlas_prior(atlas, codepoints, fit_images)
    style, style_fit = fit_aligned_font_style(aligned, fit_images, seed=args.seed)
    scorer = official_scorer(root)
    all_sessions = {mouse: grouped_blocks(root, catalog, mouse, device, len(images)) for mouse in MICE}
    fit_by_mouse = {mouse: {session: counts[fit] for session, counts in blocks.items()}
                    for mouse, blocks in all_sessions.items()}
    joined, positions = joined_responses(fit_by_mouse)
    shared = {"shared_2_fields": fit_glyph_encoder(joined, aligned, epochs=args.epochs, seed=args.seed)}
    if args.joint_fields != 2:
        shared[f"shared_{args.joint_fields}_fields"] = fit_glyph_encoder(joined, aligned, epochs=args.epochs,
                                                                       fields_per_unit=args.joint_fields, seed=args.seed)
    del joined
    output = {"atlas": provenance, "training_alignment_agreement": agreement, "style_fit": style_fit,
              "seed": args.seed, "epochs": args.epochs, "fit_glyphs": len(fit), "held_glyphs": len(held),
              "joint_fields": args.joint_fields, "single_wide_control": args.single_wide_control,
              "fit_protocol": "All visual encoders fit the same training glyphs. Joint cores see both mice's stimulus means with separate unit parameters. Held images enter only OCF and the auxiliary 320-known-image accuracy.",
              "prior": prior.diagnostics, "blocks": []}
    accumulated = {}
    for mouse in MICE:
        training = fit_by_mouse[mouse]
        separate = fit_glyph_encoder(training, aligned, epochs=args.epochs, seed=args.seed)
        decoder, stats = fit_direct(training, fit_images, args.seed)
        encoders = {"single_mouse": (separate, slice(None)),
                    **{name: (encoder, positions[mouse]) for name, encoder in shared.items()}}
        if args.single_wide_control and args.joint_fields != 2:
            wide = fit_glyph_encoder(training, aligned, epochs=args.epochs, fields_per_unit=args.joint_fields,
                                     seed=args.seed)
            encoders[f"single_{args.joint_fields}_fields"] = (wide, slice(None))
        for session in args.sessions:
            responses = all_sessions[mouse][session][held]
            queries = responses[selected]
            normalized, _ = standardize(feature_view(queries, "evoked-total"), stats[session])
            direct = decoder.predict_variants(session, normalized.mean(dim=1))
            ranks, closed = {}, {}
            for name, (encoder, columns) in encoders.items():
                with torch.autocast(device_type="cuda"):
                    rates = encoder.expected_counts(atlas.images, session, batch_size=192)[:, columns].float()
                ranks[name] = font_scores(atlas, rates, queries, direct)["font_fused"]
                closed[name] = response_accuracy(encoder, columns, images[held], session, responses)
                del rates
            for name in list(encoders)[1:]:
                ranks[f"{name}_blend"] = 0.60 * ranks["single_mouse"] + 0.40 * ranks[name]
            choices = {name: prior.variants(scores)["usage_unseen"].topk(32, dim=1).indices
                       for name, scores in ranks.items()}
            top = {name: (style.apply(atlas.images[candidates[:, 0], None].float() / 255)[:, 0] >= 0.5)
                   .to(torch.uint8) * 255 for name, candidates in choices.items()}
            reference = images[held[selected]] >= 128
            visual = (image_descriptors(reference.to(torch.uint8) * 255) @ atlas.contours.T).argmax(dim=1)
            result = score_block(scorer, top, reference.cpu().numpy(), device)
            block = {"mouse": mouse, "session": session, "closed_gallery": closed, "methods": {}}
            for name, values in result.items():
                delta = values - result["single_mouse"]
                recall = (choices[name] == visual[:, None]).any(dim=1).float().mean()
                block["methods"][name] = {"ocf": float(values.mean()), "paired_delta": float(delta.mean()),
                                           "paired_se": standard_error(delta), "visual_neighbor_in_top32": float(recall),
                                           "n": len(values)}
                accumulated.setdefault(name, []).append(values)
            output["blocks"].append(block)
            baseline = np.stack(accumulated["single_mouse"])
            output["paired_summary"] = {}
            for name, values in accumulated.items():
                aggregate = np.stack(values)
                delta = (aggregate - baseline).mean(axis=0)
                output["paired_summary"][name] = {"ocf": float(aggregate.mean()), "paired_delta": float(delta.mean()),
                                                   "paired_se": standard_error(delta), "n": aggregate.size}
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(block), flush=True)
        del separate, decoder


if __name__ == "__main__":
    main()
