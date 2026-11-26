"""Compare four-fold mice2 Task 1 calibration with the incumbent per recording.

Session1 uses the modern v10 ranking path. Sessions 2–4 use the inherited v6
ranking path. Fit on the same glyphs and use the same final public rendering
for every arm. Evaluate only labeled training glyphs reserved from every fit.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import (SHORTLIST as LEGACY_NOMINATIONS, fit_calibrated_ranker,
                                           shortlist as legacy_shortlist, split_calibration)
from kaggle_gallery_validation import align_training_glyphs
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_task1_atlas_shift_validation import ROOT_HELD_SEED, outer_indices
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_metric_scale import ANCHOR, MATCHED_ROWS_PER_EPOCH, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank, shortlist, summarize


MOUSE = "mice2"
MODERN_DAY = "session1"
LEGACY_ANCHOR = 2.0
WHOLE_GLYPHS = 320
TRANSFER_HOLDOUTS = (961, 2209, 613)
EARLY_WHOLE_HOLDOUTS = (211, 509)
SHIFT_HOLDOUT = 3541
MODERN_SINGLE = "modern_96"
MODERN_CROSSFIT = "modern_crossfit_384"
LEGACY = "legacy_96"
INCUMBENT = "incumbent_by_session"


def independent_split(count, seed, device):
    cpu = torch.device("cpu")
    previous = [outer_split(count, old, TRANSFER_HOLDOUTS[:idx], cpu)[1].numpy()
                for idx, old in enumerate(TRANSFER_HOLDOUTS)]
    previous.extend(np.random.default_rng(old).permutation(count)[:WHOLE_GLYPHS] for old in EARLY_WHOLE_HOLDOUTS)
    previous.append(np.random.default_rng(ROOT_HELD_SEED).permutation(count)[-WHOLE_GLYPHS:])
    previous.append(outer_indices(count, SHIFT_HOLDOUT, WHOLE_GLYPHS, cpu)[1].numpy())
    excluded = np.unique(np.concatenate(previous))
    available = np.setdiff1d(np.arange(count), excluded)
    if len(available) < WHOLE_GLYPHS or seed in (*TRANSFER_HOLDOUTS, *EARLY_WHOLE_HOLDOUTS, ROOT_HELD_SEED, SHIFT_HOLDOUT):
        raise ValueError("Need a new seed and enough glyphs outside the historical evaluation cohorts.")
    held = np.sort(np.random.default_rng(seed).choice(available, size=WHOLE_GLYPHS, replace=False))
    fitted = np.setdiff1d(np.arange(count), held)
    return torch.as_tensor(fitted, device=device), torch.as_tensor(held, device=device), len(excluded)


def outcomes(measured):
    return {"ocf_by_arm": {name: summarize(values, measured[INCUMBENT]) for name, values in measured.items()},
            "crossfit_vs_modern_96": summarize(measured[MODERN_CROSSFIT], measured[MODERN_SINGLE]),
            "crossfit_vs_actual_incumbent": summarize(measured[MODERN_CROSSFIT], measured[INCUMBENT]),
            "modern_96_vs_legacy_96": summarize(measured[MODERN_SINGLE], measured[LEGACY]),
            "crossfit_vs_legacy_96": summarize(measured[MODERN_CROSSFIT], measured[LEGACY])}


@torch.no_grad()
def score_session(session, query, order, outer, rendered, exact, single, expanded, legacy, modern_noise, legacy_noise):
    observed = query[order]
    candidates, _ = shortlist(outer, modern_noise[session], observed, session)
    inherited = legacy_shortlist(outer.atlas, outer.prior, outer.encoder, outer.decoder, outer.normalization,
                                 legacy_noise[session], observed, session)
    inverse = order.argsort()
    chosen = {MODERN_SINGLE: single.select(candidates, anchor_ocf=ANCHOR)[inverse],
              MODERN_CROSSFIT: expanded.select(candidates, anchor_ocf=ANCHOR)[inverse],
              LEGACY: legacy.select(inherited, anchor_ocf=LEGACY_ANCHOR)[inverse]}
    baseline = MODERN_SINGLE if session == MODERN_DAY else LEGACY
    chosen[INCUMBENT] = chosen[baseline]
    if any(bool(((choice < 0) | (choice >= len(rendered))).any()) for choice in chosen.values()):
        raise RuntimeError("Every held query must select a valid atlas rendering.")
    scores = {name: exact.evaluate(index[:, None])[:, 0] for name, index in chosen.items()}
    summaries = outcomes({name: [value] for name, value in scores.items()})
    summaries.update({"incumbent_arm": baseline, "modern_candidate_pairs": int(candidates.valid.sum()),
                      "legacy_candidate_pairs": int(inherited.valid.sum()),
                      "rows_changed_vs_modern_96": {name: int((value != chosen[MODERN_SINGLE]).sum())
                                                    for name, value in chosen.items()},
                      "rows_changed_vs_actual_incumbent": {name: int((value != chosen[INCUMBENT]).sum())
                                                           for name, value in chosen.items()}})
    return scores, summaries


def save(path, result):
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=19391, help="independent whole-glyph and query-order seed")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_mice2_incumbent_crossfit.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on Kaggle GPU with competition data and public atlas access.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(recordings) != SESSIONS or any(counts.shape[1] != 4 for counts in recordings.values()):
        raise ValueError("Expected four complete mice2 training sessions with four repeats per glyph.")
    fitted, held, excluded = independent_split(len(images), args.seed, device)
    train_images = images[fitted]
    training = {session: counts[fitted] for session, counts in recordings.items()}
    query = {session: counts[held] for session, counts in recordings.items()}
    del recordings
    scorer = official_scorer(root)
    split = split_calibration(len(fitted), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    result = {"mouse": MOUSE, "outer_seed": args.seed, "production_model_seed": MODEL_SEED,
              "outer_fit_glyphs": len(fitted), "outer_held_glyphs": len(held), "earlier_glyphs_excluded": excluded,
              "historical_calibration_outside_held": {"modern_transfer": TRANSFER_HOLDOUTS,
                                                       "earlier_legacy": EARLY_WHOLE_HOLDOUTS,
                                                       "atlas_shift": (ROOT_HELD_SEED, SHIFT_HOLDOUT)},
              "scored_repeats_per_glyph_per_session": 4, "retained_noise_glyphs": len(split[1]),
              "literal_96_glyphs_per_head": len(split[0]), "modern_crossfit_glyphs": 4 * ORIGINAL_SIZE,
              "modern_nominations": NOMINATIONS, "legacy_nominations": LEGACY_NOMINATIONS,
              "modern_anchor_ocf": ANCHOR, "legacy_anchor_ocf": LEGACY_ANCHOR,
              "matched_calibration_rows_per_epoch": MATCHED_ROWS_PER_EPOCH,
              "incumbent": {name: MODERN_SINGLE if name == MODERN_DAY else LEGACY for name in SESSIONS},
              "protocol": "Outer held glyphs are excluded from every stage of both pipelines. Identical outer atlas "
                          "columns, styles, images, and four measured trials are used. Legacy retains the v6 outer-fit "
                          "calibration atlas and v4 shortlist. Both modern heads use the v10 outer shortlist and noise model. "
                          "The single head preserves the v10 batch order for exact calibration labels. The four-fold head "
                          "reuses the v13 deduplicated fold calibration and fixed optimization budget. These two calibration "
                          "procedures may have slight exact-OCF batch-order differences. The four-day contrast clusters by "
                          "glyph and the three legacy days have a separately paired report. No formal queries were accessed.",
              "sessions": {}}

    print("Fit the literal mice2 session1 v10 head", flush=True)
    single, modern_noise, fit_report, codes = nested_ranker(training, train_images, split, scorer, MODEL_SEED)
    result[MODERN_SINGLE] = {"ranker": single.diagnostics, "fit": fit_report, "time_to_stage": time.monotonic() - started}
    save(args.output, result)
    print("Fit the same modern calibration over four independent 96-glyph folds", flush=True)
    expanded, folds = fit_crossfit_head(training, train_images, scorer, codes)
    result[MODERN_CROSSFIT] = {"ranker": expanded.diagnostics, "folds": folds, "time_to_stage": time.monotonic() - started}
    save(args.output, result)

    print("Fit the shared final atlas and the original v6 ranker", flush=True)
    outer = fit_v8(training, train_images, MODEL_SEED)
    if not torch.equal(codes, outer.codepoints.cpu()):
        raise RuntimeError("The modern calibration folds and common inference bank must share exact atlas columns.")
    rendered = rendered_bank(outer)
    aligned, _ = align_training_glyphs(outer.atlas, train_images)
    legacy, legacy_noise, legacy_fit = fit_calibrated_ranker(outer.atlas, outer.codepoints, train_images, aligned,
                                                            training, split, rendered, scorer, SESSIONS, MODEL_SEED)
    del training, aligned
    result[LEGACY] = {"fit": legacy_fit, "time_to_stage": time.monotonic() - started}
    result["shared_outer_rendering"] = outer.report
    save(args.output, result)

    exact = PairedOCF(scorer, rendered, (images[held] >= 128).cpu().numpy(), device, 12)
    history = {name: [] for name in (MODERN_SINGLE, MODERN_CROSSFIT, LEGACY, INCUMBENT)}
    rng = np.random.default_rng(args.seed + 1)
    for session in SESSIONS:
        order = torch.as_tensor(rng.permutation(len(held)), device=device)
        scores, detail = score_session(session, query[session], order, outer, rendered, exact, single, expanded,
                                       legacy, modern_noise, legacy_noise)
        for name, value in scores.items():
            history[name].append(value)
        result["sessions"][session] = detail
        result["all_days"] = outcomes(history)
        if len(history[INCUMBENT]) > 1:
            result["incumbent_legacy_days"] = outcomes({name: values[1:] for name, values in history.items()})
        result["unique_exact_outer_pairs"] = len(exact.values)
        result["seconds"] = time.monotonic() - started
        save(args.output, result)
        print(json.dumps({"session": session, **detail}), flush=True)
    print(json.dumps({"all_days": result["all_days"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
