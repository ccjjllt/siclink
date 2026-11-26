"""Compare public repertoire restrictions using held Task 1 spike responses.

Freeze the audited visual matching thresholds on a new training-only fold,
and infer the residual TGH Level 1 atlas from fit images. The same neural
scores rank the full atlas, an equal-size lexical pool, and the TGH variants.
Only measured four-repeat day groups are queried. Held images enter only OCF.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_prior import _usage_log_enrichment, seen_glyph_evidence
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_mbr import contour_signatures
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_font_family_coverage import BASE, rasterize, style_geometry
from kaggle_gallery_validation import (FontAtlas, align_training_glyphs, image_descriptors,
                                       standard_error, standardized)
from kaggle_task1_repertoire_coverage import fixed_budget, independent_cohorts, shape_matches, standard_repertoires


MOUSE = "mice1"
CALIBRATION_SEED = 7933
CALIBRATION_SOURCE = "https://www.kaggle.com/code/ccjjllt/siclink-task1-repertoire-coverage"
MATCH_QUALITY = 0.55
MATCH_MARGIN = 0.01
DIRECTION_WEIGHT = 0.65
LEXICAL_WEIGHT = 0.55
SOFT_PENALTY = 1.50
ANCHOR = "full"
LEXICAL = "same_size_lexical_residual"
RAW = "hard_tgh_raw"
HARD = "hard_tgh_residual"
SOFT = "soft_tgh_residual"


def fit_atlas(training, seed):
    font, provenance, codes = rasterize(BASE)
    aligned, matches, reference, geometry = style_geometry(font, training, seed)
    del font
    style, diagnostics = fit_aligned_font_style(matches, reference, seed=seed)
    raw_atlas = FontAtlas(aligned, torch.cat([image_descriptors((part >= 128).to(torch.uint8) * 255)
                                             for part in aligned.split(512)]))
    with torch.no_grad():
        rendered = torch.cat([(style.apply(part[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                              for part in aligned.split(1024)])
    report = {**provenance, **geometry, "style": asdict(style), "style_fit": diagnostics}
    return raw_atlas, rendered, codes, report


def fit_pools(atlas, rendered, codes, training):
    masks, tiers, frequency, provenance = standard_repertoires(codes)
    styled = FontAtlas(rendered, torch.cat([image_descriptors(part) for part in rendered.split(512)]))
    fit_matches = shape_matches(styled, training)
    confident = fit_matches.confident(MATCH_QUALITY, MATCH_MARGIN)
    removed = torch.zeros(len(codes), dtype=torch.bool)
    removed[fit_matches.closest[confident]] = True
    del styled
    seen, match_report = seen_glyph_evidence(atlas, training)
    lexical, _ = _usage_log_enrichment(tiers, frequency, seen)
    fixed_prior = LEXICAL_WEIGHT * lexical + torch.log1p(-0.94 * seen)
    raw = masks["tgh_level1"]
    residual = raw & ~removed
    size = int(residual.sum())
    if size < 1:
        raise ValueError("The residual TGH Level 1 pool is empty.")
    control = fixed_budget(fixed_prior, ~removed, size)
    pools = {LEXICAL: control, RAW: raw, HARD: residual}
    overlap = {"equal_budgets": {name: int(mask.sum()) for name, mask in pools.items()},
               "fit_confident_exclusions_full_atlas": int(removed.sum()),
               "fit_confident_exclusions_in_tgh": int((removed & raw).sum()),
               "residual_shared_tgh_and_lexical": int((residual & control).sum()),
               "residual_tgh_lexical_jaccard": float((residual & control).sum() / (residual | control).sum()),
               "soft_prior_fit": match_report}
    return pools, fixed_prior, provenance, overlap


@torch.no_grad()
def rank_session(atlas, encoder, decoder, query, session, prior, pools):
    shape = standardized(contour_retrieval_scores(decoder, query, session)["contour_ridge"])
    rate = encoder.expected_counts(atlas.images, session, batch_size=192)
    total = standardized(count_template_logits(query, rate, "rate"))
    timed = standardized(count_template_logits(query, rate, "timed"))
    neural = standardized(0.5 * (total + timed))
    scores = standardized(DIRECTION_WEIGHT * shape + (1 - DIRECTION_WEIGHT) * neural) + prior
    choices = {ANCHOR: scores.argmax(1),
               **{name: scores.masked_fill(~pool, -torch.inf).argmax(1) for name, pool in pools.items()},
               SOFT: (scores - SOFT_PENALTY * (~pools[HARD])).argmax(1)}
    return choices


def compare(values, anchor):
    scores, base = np.stack(values), np.stack(anchor)
    delta = (scores - base).mean(axis=0)
    return {"ocf": float(scores.mean()), "paired_delta": float(delta.mean()), "glyph_se": standard_error(delta),
            "score_pairs": scores.size, "unique_glyphs": len(delta)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=8291)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_repertoire_ocf.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_images <= 320:
        parser.error("Score between 2 and 320 fresh glyphs per recording.")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Use a Kaggle CPU Notebook with the competition input and Internet enabled.")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    core, inner, held, previous = independent_cohorts(len(images), args.seed, exclude_coverage_seed=CALIBRATION_SEED)
    fit = torch.cat((core, inner)).sort().values
    sample = held[:args.ocf_images]
    training = images[fit]
    print(f"Fit common atlas and neural maps on {len(fit)} labeled glyphs", flush=True)
    atlas, rendered, codes, font = fit_atlas(training, args.seed)
    pools, prior, standards, overlap = fit_pools(atlas, rendered, codes, training)
    print(json.dumps({"pool_overlap": overlap}), flush=True)
    aligned, agreement = align_training_glyphs(atlas, training)
    groups = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if len(groups) != 4 or any(values.shape[1] != 4 for values in groups.values()):
        raise ValueError("The comparison needs four held repeats from each of four training days.")
    train_counts = {name: values[fit] for name, values in groups.items()}
    decoder = fit_contour_retrieval(train_counts, *contour_signatures(aligned), task="task2", seed=args.seed)
    signature, energy = contour_signatures(atlas.images)
    decoder = replace(decoder, signature=signature, energy=energy)
    encoder = fit_glyph_encoder(train_counts, aligned, epochs=18, seed=args.seed)
    del train_counts
    predictions = {}
    for session, counts in groups.items():
        choices = rank_session(atlas, encoder, decoder, counts[sample], session, prior, pools)
        predictions[session] = choices
        print(json.dumps({"session": session, "changed_choices": {name: int((choice != choices[ANCHOR]).sum())
                                                                  for name, choice in choices.items()}}), flush=True)
    del groups, encoder, decoder, atlas

    score = PairedOCF(official_scorer(root), rendered, (images[sample] >= 128).numpy(), device, 12)
    measured, blocks = {}, []
    report = {"seed": args.seed, "mouse": MOUSE, "fit_glyphs": len(fit), "held_glyphs": len(held),
              "paired_glyphs_per_session": len(sample), "prior_holdout_glyphs_excluded_from_held": previous,
              "font": font, "fit_alignment_agreement": agreement, "standards": standards, "pools": overlap,
              "public_confidence_source": {"notebook": CALIBRATION_SOURCE, "version": 1, "quality": MATCH_QUALITY,
                                           "margin": MATCH_MARGIN, "requires_cosine_overlap_agreement": True,
                                           "requires_unique_atlas_assignment": True},
              "weights": {"directional": DIRECTION_WEIGHT, "poisson": 1 - DIRECTION_WEIGHT,
                          "lexical": LEXICAL_WEIGHT, "out_of_tgh_residual_penalty": SOFT_PENALTY},
              "primary_comparison": "Hard TGH residual versus full atlas. Equal-budget original lexical residual "
                                    "is the main comparator for the provenance of any benefit.",
              "protocol": "CPU directional contour and Poisson comparator with fixed atlas and fit-only rendering. "
                          "Fit-image matches use unbiased full-atlas shape search before membership selection and "
                          "do not use held images. Each query uses only the four measured repeats from its own day. "
                          "The same 128 held glyphs by default are exact-scored on every day. Glyph-clustered paired "
                          "SE averages day differences before estimating uncertainty. No official test records, "
                          "CNN decoder, or exact-OCF ranker are used.", "sessions": blocks}
    for session, rows in predictions.items():
        exact = {key: score.evaluate(choice[:, None])[:, 0] for key, choice in rows.items()}
        block = {"session": session, "vs_full": {key: compare([values], [exact[ANCHOR]])
                                                 for key, values in exact.items()},
                 "tgh_residual_vs_equal_budget_lexical": compare([exact[HARD]], [exact[LEXICAL]]),
                 "tgh_residual_vs_tgh_raw": compare([exact[HARD]], [exact[RAW]]),
                 "changed_vs_full": {key: int((choice != rows[ANCHOR]).sum()) for key, choice in rows.items()},
                 "hard_residual_vs_lexical_changed": int((rows[HARD] != rows[LEXICAL]).sum())}
        for key, values in exact.items():
            measured.setdefault(key, []).append(values)
        blocks.append(block)
        report["summary"] = {"vs_full": {key: compare(values, measured[ANCHOR]) for key, values in measured.items()},
                             "tgh_residual_vs_equal_budget_lexical": compare(measured[HARD], measured[LEXICAL]),
                             "tgh_residual_vs_tgh_raw": compare(measured[HARD], measured[RAW]),
                             "unique_exact_ocf_pairs": len(score.values)}
        report["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(block), flush=True)
    print(json.dumps({"summary": report["summary"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
