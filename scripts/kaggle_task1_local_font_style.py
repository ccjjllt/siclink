"""Compare restricted local font calibration at fixed public Task 1 glyph choices."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_local_font_style import MAX_DISPLACEMENT, MAX_EDIT_FRACTION, fit_local_font_style, render_local_style
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import FontAtlas, align_training_glyphs, build_font_atlas, image_descriptors, standardized
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_task1_gallery_gap import split_glyphs as gallery_gap_split
from kaggle_task1_repertoire_coverage import independent_cohorts, shape_matches
from kaggle_task1_robust_repeat_pool import PREVIOUS_OUTER
from kaggle_task1_stroke_edit_validation import exact_score


MOUSE = "mice1"
CONTROL = "global_font_style"
LOCAL = "local_font_style"
MORPHOLOGY = "same_count_morphology"
MIN_MATCH_QUALITY = 0.55
MIN_MATCH_MARGIN = 0.01
ENCODER_SEED = 211
FIT_PAIR_BUDGET = 768
FORCED_FRACTIONS = (0.0, 0.005, 0.01, 0.02)


def fresh_outer(size, seed):
    _, _, gap = gallery_gap_split(size, 7529)
    _, robust = outer_split(size, 7079, PREVIOUS_OUTER, torch.device("cpu"))
    _, _, edits, _ = independent_cohorts(size, 9103, exclude_coverage_seed=7933)
    old = np.unique(torch.cat((gap, robust, edits)).numpy())
    labels = np.arange(size)
    available = np.setdiff1d(labels, old)
    if len(available) < 320:
        raise ValueError("The previous cohorts leave too few unscored glyphs.")
    outer = np.random.default_rng(seed).permutation(available)[:320]
    fitting = np.setdiff1d(labels, outer)
    return torch.as_tensor(fitting), torch.as_tensor(outer), len(old)


@torch.no_grad()
def atlas_bits(atlas, style):
    return torch.cat([style.apply(part[:, None].float() / 255)[:, 0] >= 0.5 for part in atlas.images.split(1024)])


@torch.no_grad()
def rank_open(atlas, encoder, contour, prior, groups, sample):
    choices = {}
    for session, counts in groups.items():
        query = counts[sample]
        shape = standardized(contour_retrieval_scores(contour, query, session)["contour_ridge"])
        rates = encoder.expected_counts(atlas.images, session, batch_size=192)
        summed = standardized(count_template_logits(query, rates, "rate"))
        timed = standardized(count_template_logits(query, rates, "timed"))
        neural = standardized(0.5 * (summed + timed))
        choices[session] = prior.variants(0.65 * shape + 0.35 * neural)["usage_unseen"].argmax(dim=1)
        print(json.dumps({"session": session, "fixed_open_choices": len(query)}), flush=True)
    return choices


def paired(values, reference, eligible=None):
    values, reference = np.stack(values), np.stack(reference)
    if eligible is None:
        eligible = np.ones_like(values, dtype=bool)
    used = eligible.any(axis=0)
    pairs, glyphs = int(eligible.sum()), int(used.sum())
    if not pairs:
        return {"ocf": None, "paired_delta": None, "glyph_clustered_se": None, "pairs": 0, "glyphs": 0}
    delta = values - reference
    average = delta[eligible].mean()
    centered = ((delta - average) * eligible).sum(axis=0)[used]
    error = float(np.sqrt(glyphs / (glyphs - 1) * np.sum(centered**2)) / pairs) if glyphs > 1 else None
    return {"ocf": float(values[eligible].mean()), "paired_delta": float(average), "glyph_clustered_se": error,
            "pairs": pairs, "glyphs": glyphs}


def compare(values, eligible=None):
    result = {name: paired(scores, values[CONTROL], eligible) for name, scores in values.items()}
    result["local_vs_morphology"] = paired(values[LOCAL], values[MORPHOLOGY], eligible)
    return result


@torch.inference_mode()
def applied_variants(model, bank, choices, forced_fraction):
    indices, inverse = torch.unique(choices, return_inverse=True)
    rendered = {}
    for rows in indices.split(64):
        for name, values in render_local_style(model, bank[rows], forced_fraction=forced_fraction).items():
            rendered.setdefault(name, []).append(values)
    variants = {name: torch.cat(bitmaps, dim=0)[inverse] for name, bitmaps in rendered.items()}
    base = variants[CONTROL]
    for region in (base, ~base):
        local_count = ((variants[LOCAL] != base) & region).sum(dim=(1, 2))
        control_count = ((variants[MORPHOLOGY] != base) & region).sum(dim=(1, 2))
        if not torch.equal(local_count, control_count):
            raise RuntimeError("The two edge editors must have identical addition and removal counts.")
    return variants


def measure(model, bank, choices, reference, scorer, cache, fit_codes, *, forced_fraction=None):
    variants = applied_variants(model, bank, choices, forced_fraction)
    exact = {name: exact_score(bitmaps, reference, scorer, cache) for name, bitmaps in variants.items()}
    count = (variants[LOCAL] != variants[CONTROL]).sum(dim=(1, 2))
    fit_match = torch.isin(choices, fit_codes).numpy()
    return exact, fit_match, {"glyphs_edited": int((count > 0).sum()), "mean_pixels_edited": float(count.float().mean()),
                              "selected_public_masks_used_in_local_fit": int(fit_match.sum())}


def calibrate_budget(model, bank, choices, references, scorer, fit_ids):
    cache, trials, contenders = {}, {}, []
    for fraction in FORCED_FRACTIONS:
        measured, _, edits = measure(model, bank, choices, references, scorer, cache, fit_ids, forced_fraction=fraction)
        metrics = compare({name: [values] for name, values in measured.items()})
        local, control = metrics[LOCAL], metrics["local_vs_morphology"]
        support = min(local["paired_delta"] - local["glyph_clustered_se"],
                      control["paired_delta"] - control["glyph_clustered_se"])
        trials[str(fraction)] = {"paired": metrics, "edits": edits, "minimum_gain_minus_se": support}
        if fraction > 0 and support > 0:
            contenders.append((support, -fraction))
        print(json.dumps({"independent_calibration_budget": fraction, **trials[str(fraction)]}), flush=True)
    selected = -max(contenders)[1] if contenders else 0.0
    return selected, {"tested_fractions": FORCED_FRACTIONS, "results": trials, "unique_exact_pairs": len(cache),
                      "selected_fraction": selected,
                      "rule": "Require a positive gain minus one paired standard error against the unchanged "
                              "global rendering and same-count morphology. Maximize the smaller margin, then "
                              "choose the smaller budget if tied. Zero is eligible by default."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=15919)
    parser.add_argument("--visual-glyphs", type=int, default=128)
    parser.add_argument("--open-glyphs", type=int, default=96)
    parser.add_argument("--calibrate-forced-edits", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_local_font_style.json"))
    args = parser.parse_args()
    if min(args.visual_glyphs, args.open_glyphs) < 2 or max(args.visual_glyphs, args.open_glyphs) > 320:
        parser.error("Choose positive whole-glyph paired samples within the held cohort.")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Use Kaggle CPU with the official training data and public font access.")
    torch.set_num_threads(4)
    torch.manual_seed(ENCODER_SEED)
    device = torch.device("cpu")
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, device)
    fit, held, prior_excluded = fresh_outer(len(images), args.seed)
    random = np.random.default_rng(args.seed + 2)
    train = images[fit]
    style_rows = random.choice(len(train), 128, replace=False)
    atlas, font, codes = build_font_atlas(train[style_rows], device, return_codepoints=True)
    aligned, align_info = align_training_glyphs(atlas, train)
    style, global_fit = fit_aligned_font_style(aligned, train, seed=ENCODER_SEED)
    bank = atlas_bits(atlas, style)
    public_style = FontAtlas(bank.to(torch.uint8) * 255,
                             torch.cat([image_descriptors(part.to(torch.uint8) * 255) for part in bank.split(512)]))
    training_matches = shape_matches(public_style, train)
    high = torch.where(training_matches.confident(MIN_MATCH_QUALITY, MIN_MATCH_MARGIN))[0]
    if len(high) < 192:
        raise ValueError("The fixed confidence screen needs at least 192 independent visual pairs.")
    chosen = high[random.permutation(len(high))[:FIT_PAIR_BUDGET]]
    fit_ids = training_matches.closest[chosen]
    local, local_report = fit_local_font_style(bank[fit_ids], train[chosen] >= 128, training_matches.scores[chosen, 0],
                                              seed=ENCODER_SEED)
    print(json.dumps({"fit_high_confidence_pairs": len(high), "local_style": local_report}), flush=True)

    paired_match = shape_matches(public_style, images[held])
    del public_style
    held_confident = paired_match.confident(MIN_MATCH_QUALITY, MIN_MATCH_MARGIN)
    fit_overlap_with_held = held_confident & torch.isin(paired_match.closest, fit_ids)
    visible = torch.where(held_confident & ~fit_overlap_with_held)[0][:args.visual_glyphs]
    if len(visible) < 2:
        raise ValueError("Too few confidently paired held glyphs with independent public shapes.")
    scorer = official_scorer(root)
    result = {"seed": args.seed, "model_seed": ENCODER_SEED, "mouse": MOUSE, "fit_glyphs": len(fit),
              "held_glyphs": len(held), "prior_cohort_union_excluded": prior_excluded,
              "fit_high_confidence_pairs": len(high), "local_style_fit": local_report,
              "matching": {"minimum_quality": MIN_MATCH_QUALITY, "minimum_margin": MIN_MATCH_MARGIN,
                           "one_to_one": True, "source": "audited fixed confidence rule from the training-image repertoire study"},
              "font": font, "global_font_style": asdict(style), "global_style_fit": global_fit,
              "font_alignment": align_info, "edge_displacement_px": MAX_DISPLACEMENT,
              "maximum_edit_fraction": MAX_EDIT_FRACTION,
              "protocol": "Existing global style renders all original glyphs as binary masks. Local model works "
                          "from those binary masks only. Visual exact OCF uses independently matched held images, "
                          "with model-seen public indices excluded from this diagnostic. Public identities are "
                          "confident visual correspondences rather than given competition codepoints. Open-atlas "
                          "choices come from the same CPU directional/Poisson decoder and Unihan prior, unchanged "
                          "between renders. Each held query is four observed repeats from one labeled recording "
                          "day. Exact OCF uses paired glyph-clustered uncertainty, with subgroup comparisons by "
                          "whether the chosen public bitmap appeared in the local model fit. No official tests."}

    forced_fraction = None
    if args.calibrate_forced_edits:
        reserved = torch.zeros(len(held), dtype=torch.bool)
        reserved[visible] = True
        reserved[:args.open_glyphs] = True
        calibration = torch.where(held_confident & ~fit_overlap_with_held & ~reserved)[0][:64]
        if len(calibration) < 48:
            raise ValueError("The independent budget calibration needs at least 48 other held glyphs.")
        calibration_refs = (images[held[calibration]] >= 128).numpy()
        forced_fraction, calibration_report = calibrate_budget(local, bank, paired_match.closest[calibration],
                                                              calibration_refs, scorer, fit_ids)
        result["independent_calibration"] = {**calibration_report, "glyphs": len(calibration),
                                             "overlap_with_either_outer_score_cohort": 0,
                                             "stimulus": "Fixed high-confidence public image matches."}
        if forced_fraction == 0:
            result["open_confirmation"] = "The predeclared independent calibration gate selected zero editing."
            result["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"selected_fraction": 0, "seconds": result["seconds"]}), flush=True)
            return

    visual_references = (images[held[visible]] >= 128).numpy()
    visual, _, visual_edits = measure(local, bank, paired_match.closest[visible], visual_references, scorer, {}, fit_ids,
                                      forced_fraction=forced_fraction)
    visual_report = {"pairs": len(visible), "eligible_high_confidence": int(held_confident.sum()),
                     "excluded_public_fit_overlap": int(fit_overlap_with_held.sum()),
                     "mean_frozen_visual_match_quality": float(paired_match.scores[visible, 0].mean()),
                     "edits": visual_edits, "paired": compare({name: [scores] for name, scores in visual.items()})}
    result["fixed_visual_match"] = visual_report
    print(json.dumps({"fixed_visual_match": visual_report}), flush=True)

    groups = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if len(groups) != 4 or any(values.shape[1] != 4 for values in groups.values()):
        raise ValueError("The frozen open comparison needs four repeats from each labeled day.")
    training = {session: values[fit] for session, values in groups.items()}
    prior = fit_atlas_prior(atlas, codes, train)
    contour = fit_contour_retrieval(training, *contour_signatures(aligned), task="task2", seed=ENCODER_SEED)
    signature, energy = contour_signatures(atlas.images)
    contour = replace(contour, signature=signature, energy=energy)
    encoder = fit_glyph_encoder(training, aligned, epochs=18, seed=ENCODER_SEED)
    del training
    sample = held[:args.open_glyphs]
    choices = rank_open(atlas, encoder, contour, prior, groups, sample)
    del atlas, encoder, contour, groups
    references = (images[sample] >= 128).numpy()
    readings, cohorts, blocks, cache = {}, [], {}, {}
    for session, ids in choices.items():
        exact, membership, details = measure(local, bank, ids, references, scorer, cache, fit_ids,
                                             forced_fraction=forced_fraction)
        cohorts.append(membership)
        one_day = {name: [values] for name, values in exact.items()}
        blocks[session] = {"edits": details, "paired": compare(one_day),
                           "public_mask_in_local_fit": compare(one_day, membership[None]),
                           "public_mask_not_in_local_fit": compare(one_day, ~membership[None])}
        for name, values in exact.items():
            readings.setdefault(name, []).append(values)
        used = np.stack(cohorts)
        result["fixed_open_top1"] = {"sessions": blocks, "paired": compare(readings),
                                     "public_mask_in_local_fit": compare(readings, used),
                                     "public_mask_not_in_local_fit": compare(readings, ~used),
                                     "unique_exact_open_pairs": len(cache)}
        result["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, "fixed_open_top1": blocks[session]}), flush=True)
    print(json.dumps({"open_paired": result["fixed_open_top1"]["paired"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
