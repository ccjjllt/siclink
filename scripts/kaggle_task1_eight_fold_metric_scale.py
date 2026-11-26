"""Compare the current four-fold calibration head with eight folds at equal updates."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, fit_atlas_metric_ranker
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_task1_crossfit_metric_scale import calibrated_fold, fold_indices, fresh_outer, PILOT_SEED
from kaggle_task1_metric_scale import (ANCHOR, HISTORICAL_ORDER, MATCHED_ROWS_PER_EPOCH, MODEL_SEED,
                                      RANKER_BATCH, RANKER_EPOCHS, independent_holdout)
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, rendered_bank, shortlist, summarize


REPLICATION_SEED = 18313
OLDER_HOLDOUTS = (961, 2209, 4931, 12457, 16633)
CURRENT_FOLDS = 4
EXPANDED_FOLDS = 8


def new_outer(size, seed, device):
    cpu = torch.device("cpu")
    _, pilot = independent_holdout(size, PILOT_SEED, cpu)
    _, replicated = fresh_outer(size, REPLICATION_SEED, cpu)
    history = [outer_split(size, old, HISTORICAL_ORDER[:HISTORICAL_ORDER.index(old)], cpu)[1].numpy()
               for old in OLDER_HOLDOUTS]
    excluded = np.unique(np.concatenate([pilot.numpy(), replicated.numpy(), *history]))
    eligible = np.setdiff1d(np.arange(size), excluded)
    if len(eligible) < 320:
        raise ValueError("Expect enough previously unused glyphs for a complete holdout.")
    held = np.sort(np.random.default_rng(seed).choice(eligible, 320, replace=False))
    if np.intersect1d(held, pilot.numpy()).size or np.intersect1d(held, replicated.numpy()).size:
        raise ValueError("Overlap with an earlier metric scale evaluation holdout.")
    fit = np.setdiff1d(np.arange(size), held)
    return torch.as_tensor(fit, device=device), torch.as_tensor(held, device=device)


def collect_calibration(folds, training, images, noise, scorer, save, atlas_order=None):
    parts, targets = [], []
    for number, (metric, inner) in folds:
        print(f"Fold {number + 1}: fitting {len(inner):,} upstream glyphs", flush=True)
        candidates, exact, uncertainty, order, information = calibrated_fold(training, images, metric, noise, inner, scorer)
        if atlas_order is None:
            atlas_order = order
        elif not torch.equal(atlas_order, order):
            raise ValueError("The public atlas column ordering must match in every fold.")
        parts.append(candidates)
        targets.append(exact)
        save(number, information, uncertainty)
    return parts, targets, atlas_order


def combined_head(parts, targets):
    return fit_atlas_metric_ranker(concat_atlas_candidates(parts), torch.cat(targets), seed=MODEL_SEED,
                                   epochs=RANKER_EPOCHS, batch_rows=RANKER_BATCH, rows_per_epoch=MATCHED_ROWS_PER_EPOCH)


def summarize_arms(values):
    result = {arm: summarize(blocks, values["fourfold_384"]) for arm, blocks in values.items()}
    result["fourfold_384_vs_fold0_96"] = summarize(values["fourfold_384"], values["fold0_96"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=19661)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_eight_fold_metric_scale.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on one private Kaggle T4, with the competition and public font assets available.")
    begin = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held = new_outer(len(images), args.seed, device)
    order = torch.as_tensor(np.random.default_rng(args.seed + 1).permutation(len(held)), device=device)
    selected = held[order]
    recording = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if tuple(recording) != SESSIONS or any(values.shape[1] != 4 for values in recording.values()):
        raise ValueError("Expect four labeled mice1 recordings and four measured repeats per glyph.")
    training = {name: values[fit] for name, values in recording.items()}
    scorer = official_scorer(root)
    noise, folds = fold_indices(len(fit), device, fold_count=EXPANDED_FOLDS)
    original_noise, original_folds = fold_indices(len(fit), device)
    if not torch.equal(noise, original_noise) or any(
            not torch.equal(a, b) for before, current in zip(original_folds, folds[:CURRENT_FOLDS])
            for a, b in zip(before, current)):
        raise ValueError("Expansion must retain precisely the four-fold protocol's original folds and noise split.")
    output = {"mouse": "mice1", "seed": args.seed, "model_seed": MODEL_SEED,
              "avoided_held_cohorts": (PILOT_SEED, REPLICATION_SEED, *OLDER_HOLDOUTS),
              "outer_fit_glyphs": len(fit), "outer_holdout_glyphs": len(held),
              "paired_glyphs_per_recording": len(selected), "repeats_per_glyph": 4,
              "fixed_noise_glyphs": len(noise), "v11_anchor": ANCHOR, "nominations": NOMINATIONS,
              "first_four_folds_identical_to_original_protocol": True,
              "equal_ranker_optimizer_steps": RANKER_EPOCHS * MATCHED_ROWS_PER_EPOCH // RANKER_BATCH,
              "rows_per_epoch": MATCHED_ROWS_PER_EPOCH, "calibration_folds": {}, "sessions": {},
              "protocol": "Four-fold 384 and eight-fold 768 heads use the same initial four calibration "
                          "folds and same 640 optimizer updates. Each fold's neural encoding, decoder, "
                          "corpus prior and font style exclude that fold's metric glyphs and the identical "
                          "fixed noise glyphs. Exact labels are deduplicated within each fold across "
                          "recordings. Fold zero supplies the common outer uncertainty model. Both heads "
                          "are evaluated using the same retained outer v11 proposal graph, same rendering "
                          "and anchor. All 320 completely held glyphs are paired in four recordings and "
                          "the two prior scale evaluation holdout groups do not overlap. Standard errors "
                          "cluster by glyph. No official test records or submission packages are used."}
    stock_noise = None

    def checkpoint(stage):
        output["stage"] = stage
        output["seconds"] = time.monotonic() - begin
        staging = args.output.with_suffix(".tmp")
        staging.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        staging.replace(args.output)

    def save_fold(index, information, uncertainty):
        nonlocal stock_noise
        if index == 0:
            stock_noise = uncertainty
        output["calibration_folds"][str(index + 1)] = information
        output["unique_exact_calibration_pairs_so_far"] = sum(row["unique_exact_calibration_pairs"]
                                                               for row in output["calibration_folds"].values())
        checkpoint(f"calibration_fold_{index + 1}_complete")

    first_parts, first_targets, atlas_order = collect_calibration(enumerate(folds[:CURRENT_FOLDS]), training, images[fit],
                                                                 noise, scorer, save_fold)
    four = combined_head(first_parts, first_targets)
    stock = fit_atlas_metric_ranker(first_parts[0], first_targets[0], seed=MODEL_SEED)
    output["fourfold_ranker"] = four.diagnostics
    output["fold0_ranker"] = stock.diagnostics
    outer = fit_v8(training, images[fit], MODEL_SEED)
    if not torch.equal(outer.codepoints.cpu(), atlas_order):
        raise ValueError("The common outer graph must have the same public atlas column ordering.")
    output["outer_fit"] = outer.report
    rendered = rendered_bank(outer)
    exact = PairedOCF(scorer, rendered, (images[selected] >= 128).cpu().numpy(), device, 12)
    graphs, original_choices = {}, {}
    observations = {"fold0_96": [], "fourfold_384": []}
    for name in SESSIONS:
        graph, _ = shortlist(outer, stock_noise[name], recording[name][selected], name)
        graphs[name] = graph
        original_choices[name] = {"fold0_96": stock.select(graph, anchor_ocf=ANCHOR),
                                  "fourfold_384": four.select(graph, anchor_ocf=ANCHOR)}
        values = {arm: exact.evaluate(indices[:, None])[:, 0] for arm, indices in original_choices[name].items()}
        for arm, result in values.items():
            observations[arm].append(result)
        output["sessions"][name] = {"metrics": summarize_arms({arm: [result] for arm, result in values.items()}),
                                    "fixed_candidate_edges": int(graph.valid.sum())}
        checkpoint(f"fourfold_outer_{name}_complete")
    output["fourfold_baseline"] = summarize_arms(observations)
    checkpoint("fourfold_baseline_complete")
    print(json.dumps({"stage": output["stage"], "seconds": output["seconds"], "fourfold_baseline": output["fourfold_baseline"]}), flush=True)
    del outer

    rest_parts, rest_targets, _ = collect_calibration(enumerate(folds[CURRENT_FOLDS:], CURRENT_FOLDS), training,
                                                       images[fit], noise, scorer, save_fold, atlas_order)
    eight = combined_head(first_parts + rest_parts, first_targets + rest_targets)
    output["eightfold_ranker"] = eight.diagnostics
    observations["eightfold_768"] = []
    for index, name in enumerate(SESSIONS):
        chosen = eight.select(graphs[name], anchor_ocf=ANCHOR)
        observations["eightfold_768"].append(exact.evaluate(chosen[:, None])[:, 0])
        row = output["sessions"][name]
        row["metrics"] = summarize_arms({arm: [values[index]] for arm, values in observations.items()})
        row["changed_vs_fourfold"] = int((chosen != original_choices[name]["fourfold_384"]).sum())
        print(json.dumps({"recording": name, **row}), flush=True)
        checkpoint(f"eightfold_outer_{name}_complete")
    output["summary"] = summarize_arms(observations)
    output["drop_one_recording"] = {omitted: summarize_arms({arm: [r for index, r in enumerate(values) if index != i]
                                                           for arm, values in observations.items()})
                                    for i, omitted in enumerate(SESSIONS)}
    output["unique_exact_outer_pairs"] = len(exact.values)
    checkpoint("complete")
    print(json.dumps({"seconds": output["seconds"], "summary": output["summary"]}), flush=True)


if __name__ == "__main__":
    main()
