"""Replicate exact-OCF calibration scaling using four full-size v11 training folds."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, fit_atlas_metric_ranker
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_predictive_likelihood import fit_predictive_likelihood
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_gallery_validation import align_training_glyphs
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_task1_metric_scale import (ANCHOR, HISTORICAL_ORDER, MATCHED_ROWS_PER_EPOCH, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE,
                                      RANKER_BATCH, RANKER_EPOCHS, calibration_labels, independent_holdout)
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, rendered_bank, shortlist, summarize


PILOT_SEED = 17641
OLDER_COHORTS = (961, 2209, 4931, 5147, 12457, 16633)
CALIBRATION_FOLDS = 4


def fresh_outer(image_count, seed, device):
    cpu = torch.device("cpu")
    _, pilot = independent_holdout(image_count, PILOT_SEED, cpu)
    older = [outer_split(image_count, prior, HISTORICAL_ORDER[:HISTORICAL_ORDER.index(prior)], cpu)[1].numpy()
             for prior in OLDER_COHORTS]
    excluded = np.unique(np.concatenate([pilot.numpy(), *older]))
    available = np.setdiff1d(np.arange(image_count), excluded)
    if len(available) < 320:
        raise ValueError("Expect 320 or more glyphs outside the predeclared older cohorts.")
    picked = np.sort(np.random.default_rng(seed).permutation(available)[:320])
    if np.intersect1d(pilot.numpy(), picked).size:
        raise ValueError("The recheck must exclude every glyph from the prior scale experiment.")
    train = np.setdiff1d(np.arange(image_count), picked)
    return torch.as_tensor(train, device=device), torch.as_tensor(picked, device=device)


def fold_indices(image_count, device, *, fold_count=CALIBRATION_FOLDS):
    """Keep fold zero's exact v11 metric, noise, and upstream training order."""
    original, noise, literal_inner = split_calibration(image_count, ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    eligible = torch.cat((original, literal_inner))
    capacity = fold_count * ORIGINAL_SIZE
    if len(eligible) < capacity + 512:
        raise ValueError("Keep a substantial inner encoder fit after reserving whole-glyph calibration folds.")
    masks = torch.ones((fold_count, len(eligible)), dtype=torch.bool, device=device)
    for fold in range(fold_count):
        masks[fold, fold * ORIGINAL_SIZE:(fold + 1) * ORIGINAL_SIZE] = False
    folds = list(zip(eligible[:capacity].split(ORIGINAL_SIZE), [eligible[mask] for mask in masks]))
    if not torch.equal(folds[0][0], original) or not torch.equal(folds[0][1], literal_inner):
        raise ValueError("Fold zero must follow the literal v11 calibration split and order.")
    return noise, folds


def calibrated_fold(training, images, metric, noise, inner, scorer):
    began = time.monotonic()
    fit = {name: counts[inner] for name, counts in training.items()}
    model = fit_v8(fit, images[inner], MODEL_SEED)
    noise_glyphs, _ = align_training_glyphs(model.atlas, images[noise])
    uncertainty = {name: fit_predictive_likelihood(training[name][noise], model.encoder.expected_counts(noise_glyphs, name))
                   for name in SESSIONS}
    rendered = rendered_bank(model)
    reference = images[metric] >= 128
    parts = [shortlist(model, uncertainty[name], training[name][metric], name)[0] for name in SESSIONS]
    candidates = concat_atlas_candidates(parts)
    target, measured = calibration_labels(candidates, rendered, reference, scorer)
    report = {"calibration_glyphs": len(metric), "noise_glyphs": len(noise), "neural_and_font_fit_glyphs": len(inner),
              "candidate_edges": int(candidates.valid.sum()), "unique_exact_calibration_pairs": measured,
              "training_queries": len(candidates.valid), "atlas_and_font_style": model.report,
              "seconds": time.monotonic() - began}
    return candidates, target, uncertainty, model.codepoints.cpu(), report


def score_report(observations):
    report = {name: summarize(values, observations["v11"]) for name, values in observations.items()}
    for fold in range(1, CALIBRATION_FOLDS):
        name = f"single_fold_{fold}_96"
        report[f"crossfit_384_vs_{name}"] = summarize(observations["crossfit_384"], observations[name])
    singles = [np.stack(observations[f"single_fold_{index}_96"]) for index in range(1, CALIBRATION_FOLDS)]
    report["crossfit_384_vs_mean_new_single_folds"] = summarize(observations["crossfit_384"], np.stack(singles).mean(0))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=18313)
    parser.add_argument("--ocf-images", type=int, default=192)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_crossfit_metric_scale.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_images <= 320:
        parser.error("Use two to 320 held glyphs for paired OCF on each recording.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on a private Kaggle T4 worker with official training and public font assets.")
    start = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held = fresh_outer(len(images), args.seed, device)
    rng = np.random.default_rng(args.seed + 1)
    selected = held[torch.as_tensor(rng.choice(len(held), args.ocf_images, replace=False), device=device)]
    recording = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if tuple(recording) != SESSIONS or any(values.shape[1] != 4 for values in recording.values()):
        raise ValueError("Expect all four labeled mice1 recordings with four repeats.")
    training = {name: counts[fit] for name, counts in recording.items()}
    scorer = official_scorer(root)
    noise, folds = fold_indices(len(fit), device)
    output = {"mouse": "mice1", "seed": args.seed, "model_seed": MODEL_SEED, "outer_fit_glyphs": len(fit),
              "outer_holdout_glyphs": len(held), "avoided_whole_cohorts": (*OLDER_COHORTS, PILOT_SEED),
              "paired_glyphs_per_recording": len(selected), "repeats_per_glyph": 4, "v11_anchor": ANCHOR,
              "v11_nominations": NOMINATIONS, "fold_count": len(folds), "fixed_noise_glyphs": len(noise),
              "optimizer_steps_per_head": RANKER_EPOCHS * MATCHED_ROWS_PER_EPOCH // RANKER_BATCH,
              "rows_seen_per_epoch": MATCHED_ROWS_PER_EPOCH, "calibration_folds": {}, "sessions": {},
              "protocol": "Fold zero preserves exactly the v11 nested glyph split and training order, with the same "
                          "ranker defaults. Each of four folds fits its neural encoder, decoder, font geometry and "
                          "style only on the equally sized inner set that excludes its own 96 label glyphs and the "
                          "same shared 384 noise glyphs. Scores from the same rendered candidate/reference pair are "
                          "deduplicated only within a fold because font calibration can differ across folds. All "
                          "folds and the outer model must share precisely the same public atlas column order. "
                          "A single full outer v11 graph, font rendering and anchor are reused for every head, with "
                          "fold-zero noise parameters as used by v11. The pooled head uses 384 independently held "
                          "calibration glyphs, taking 384 randomized rows per epoch, the same Adam budget as a 96-glyph "
                          "head. New single fold one is the predeclared equal-budget comparison. Evaluation uses "
                          "the same paired held glyphs across four recordings and no ranks rely on held target images "
                          "or sample identifiers. No formal test or submission input is accessed."}

    parts, targets, rankers = [], [], {}
    stock_noise, atlas_order = None, None
    for index, (metric, inner) in enumerate(folds):
        print(f"Fitting calibration fold {index} on {len(inner):,} upstream glyphs", flush=True)
        candidates, labels, fitted_noise, codepoints, report = calibrated_fold(training, images[fit], metric, noise, inner, scorer)
        if index == 0:
            stock_noise, atlas_order = fitted_noise, codepoints
        elif not torch.equal(atlas_order, codepoints):
            raise ValueError("Calibration fold atlas columns changed order.")
        name = "v11" if index == 0 else f"single_fold_{index}_96"
        rankers[name] = fit_atlas_metric_ranker(candidates, labels, seed=MODEL_SEED)
        report["ranker"] = rankers[name].diagnostics
        output["calibration_folds"][str(index)] = report
        parts.append(candidates)
        targets.append(labels)
        output["seconds"] = time.monotonic() - start
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    rankers["crossfit_384"] = fit_atlas_metric_ranker(concat_atlas_candidates(parts), torch.cat(targets), seed=MODEL_SEED,
                                                     epochs=RANKER_EPOCHS, batch_rows=RANKER_BATCH,
                                                     rows_per_epoch=MATCHED_ROWS_PER_EPOCH)
    output["pooled_ranker"] = rankers["crossfit_384"].diagnostics
    output["unique_calibration_pairs_total"] = sum(row["unique_exact_calibration_pairs"] for row in output["calibration_folds"].values())
    outer = fit_v8(training, images[fit], MODEL_SEED)
    del training
    if not torch.equal(atlas_order, outer.codepoints.cpu()):
        raise ValueError("Calibrated heads and outer v11 graph must have the same public atlas indices.")
    output["outer_font_and_atlas"] = outer.report
    output["atlas_order_verified"] = True
    rendered = rendered_bank(outer)
    observed_by_arm = {}
    exact = PairedOCF(scorer, rendered, (images[selected] >= 128).cpu().numpy(), device, 12)
    for name in SESSIONS:
        candidates, _ = shortlist(outer, stock_noise[name], recording[name][selected], name)
        choices = {arm: ranker.select(candidates, anchor_ocf=ANCHOR) for arm, ranker in rankers.items()}
        measured = {arm: exact.evaluate(predictions[:, None])[:, 0] for arm, predictions in choices.items()}
        for arm, values in measured.items():
            observed_by_arm.setdefault(arm, []).append(values)
        row = {"metrics": score_report({arm: [values] for arm, values in measured.items()}),
               "shared_outer_edges": int(candidates.valid.sum()),
               "changed_vs_v11": {arm: int((prediction != choices["v11"]).sum()) for arm, prediction in choices.items()}}
        output["sessions"][name] = row
        output["summary"] = score_report(observed_by_arm)
        output["unique_exact_outer_pairs"] = len(exact.values)
        output["seconds"] = time.monotonic() - start
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": name, **row}), flush=True)
    output["drop_one_recording"] = {removed: score_report({arm: [v for i, v in enumerate(values) if i != index]
                                                           for arm, values in observed_by_arm.items()})
                                    for index, removed in enumerate(SESSIONS)}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
