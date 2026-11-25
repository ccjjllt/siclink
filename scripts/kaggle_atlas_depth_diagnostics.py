"""Measure exact OCF headroom inside spike-ranked public Task 1 glyph pools."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import load_catalog, load_images, locate_data, split_glyphs
from glyph_font_oracle import aggregate
from kaggle_atlas_prior_validation import neural_blocks
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import (align_training_glyphs, build_font_atlas, image_descriptors,
                                       standard_error)


DEPTHS = (1, 4, 16, 32)


class PairedOCF:
    """Reuse exact scores for the same held image and candidate across blocks."""

    def __init__(self, scorer, atlas_images, reference, device, batch_size):
        self.scorer = scorer
        self.atlas_images = atlas_images
        self.reference = reference
        self.device = str(device)
        self.batch_size = batch_size
        self.values = {}

    def evaluate(self, shortlist):
        indices = shortlist.detach().cpu().numpy()
        missing = [(row, int(candidate)) for row, candidates in enumerate(indices) for candidate in candidates
                   if (row, int(candidate)) not in self.values]
        if missing:
            rows, candidates = zip(*missing)
            candidates = torch.as_tensor(candidates, device=self.atlas_images.device)
            predictions = (self.atlas_images[candidates] >= 128).cpu().numpy()
            references = self.reference[np.asarray(rows)]
            scores = self.scorer(predictions, references, device=self.device, batch_size=self.batch_size)
            self.values.update((pair, float(score)) for pair, score in zip(missing, scores))
        return np.asarray([[self.values[row, int(candidate)] for candidate in candidates]
                           for row, candidates in enumerate(indices)])


def depth_summary(exact, visual_best):
    baseline = exact[..., 0]
    result = {}
    for depth in DEPTHS:
        best = exact[..., :depth].max(axis=-1)
        delta = best - baseline
        # The same held glyphs are reused in every block. Collapse blocks before the paired SE.
        glyph_delta = delta if delta.ndim == 1 else delta.mean(axis=0)
        result[str(depth)] = {**aggregate(best.ravel()), "paired_gain_vs_top1": float(glyph_delta.mean()),
                              "paired_gain_se": standard_error(glyph_delta),
                              "visual_pool_gap": float((visual_best - best).mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session1", "session4"))
    parser.add_argument("--ocf-per-session", type=int, default=16)
    parser.add_argument("--visual-depth", type=int, default=16)
    parser.add_argument("--style-fit-glyphs", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--refine-epochs", type=int, default=12)
    parser.add_argument("--contour-epochs", type=int, default=3)
    parser.add_argument("--device", default="auto", help="atlas and model device; defaults to available CUDA or CPU")
    parser.add_argument("--metric-device", help="official OCF device; defaults to the model device")
    parser.add_argument("--metric-batch", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/atlas_depth_diagnostics.json"))
    args = parser.parse_args()
    if not 1 <= args.ocf_per_session <= 320 or min(args.visual_depth, args.metric_batch, args.style_fit_glyphs,
                                                   args.encoder_epochs, args.refine_epochs) < 1:
        parser.error("Use positive experiment settings and at most 320 metric glyphs per block.")
    if not 0 <= args.contour_epochs <= args.refine_epochs:
        parser.error("Choose a valid contour epoch count.")
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    metric_device = torch.device(args.metric_device) if args.metric_device else device
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run in a Kaggle Notebook with competition input and Internet enabled.")
    if any(item.type not in ("cuda", "cpu") for item in (device, metric_device)):
        parser.error("Use CPU or CUDA for fitting and OCF.")

    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    fit, held = split_glyphs(len(images), 320, args.seed, device)
    if args.style_fit_glyphs > len(fit):
        parser.error("Style calibration cannot exceed the training fold size.")
    style_rows = np.random.default_rng(args.seed + 2).choice(len(fit), args.style_fit_glyphs, replace=False)
    metric_rows = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_per_session, replace=False)
    style_rows = torch.as_tensor(style_rows, device=device)
    metric_rows = torch.as_tensor(metric_rows, device=device)
    fit_images = images[fit]
    atlas, provenance = build_font_atlas(fit_images[style_rows], device)
    if max(args.visual_depth, max(DEPTHS)) > len(atlas.images):
        parser.error("The shortlist cannot exceed the number of public atlas images.")
    font_images, contour_agreement = align_training_glyphs(atlas, fit_images)
    score = official_scorer(root)
    output = {"device": str(device), "metric_device": str(metric_device), "seed": args.seed,
              "held_glyphs": len(held), "ocf_per_session": len(metric_rows), "depths": DEPTHS,
              "atlas": provenance, "training_alignment_agreement": contour_agreement,
              "encoder_epochs": args.encoder_epochs, "refine_epochs": args.refine_epochs,
              "contour_epochs": args.contour_epochs, "blocks": [],
              "protocol": "The model ranks the full atlas from held neural responses. Exact OCF chooses the best candidate only for diagnostics. The visual shortlist uses held images for evaluation after model ranking and is an achievable reference-image search score, not an exhaustive atlas bound."}
    print(json.dumps({"atlas": provenance, "held_glyphs": len(held), "scored_per_block": len(metric_rows)}), flush=True)

    evaluator = None
    exact_blocks = []
    for (mouse, session), logits in neural_blocks(atlas, root, catalog, len(images), fit_images, font_images,
                                                fit, held, args, device):
        block_start = time.monotonic()
        shortlist = logits[metric_rows].topk(max(DEPTHS), dim=1).indices
        if evaluator is None:
            # No reference image is supplied to the neural scorer or train-only alignment.
            masks = images[held[metric_rows]] >= 128
            evaluator = PairedOCF(score, atlas.images, masks.cpu().numpy(), metric_device, args.metric_batch)
            visual_similarity = image_descriptors(masks.to(torch.uint8) * 255) @ atlas.contours.T
            visual = visual_similarity.topk(args.visual_depth, dim=1).indices
            visual_exact = evaluator.evaluate(visual)
            visual_best = visual_exact.max(axis=1)
            output["visual_pool"] = {"depth": args.visual_depth, "top1": aggregate(visual_exact[:, 0]),
                                     "best": aggregate(visual_best)}
            print(json.dumps({"visual_pool": output["visual_pool"]}), flush=True)

        print(f"{mouse}/{session}: scoring {len(metric_rows)} x {max(DEPTHS)} ranked pairs", flush=True)
        before = len(evaluator.values)
        exact = evaluator.evaluate(shortlist)
        exact_blocks.append(exact)
        visual_hit = (shortlist[:, :, None] == visual[:, None, :]).any(dim=2)
        row = {"mouse": mouse, "session": session, "depth": depth_summary(exact, visual_best),
               "visual_overlap": {str(depth): float(visual_hit[:, :depth].any(dim=1).float().mean())
                                  for depth in DEPTHS},
               "new_ocf_pairs": len(evaluator.values) - before, "seconds": time.monotonic() - block_start}
        output["blocks"].append(row)
        output["summary"] = {"depth": depth_summary(np.stack(exact_blocks), visual_best),
                             "unique_held_glyphs": len(metric_rows), "exact_cached_pairs": len(evaluator.values)}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
