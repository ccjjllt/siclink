"""Measure the value of more exact OCF calibration glyphs for the fixed v11 ranker."""

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
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank, shortlist, summarize


MODEL_SEED = 211
ANCHOR = 4.0
ORIGINAL_SIZE = 96
NOISE_SIZE = 384
ADDITIONAL_SIZE = 384
NESTED_SIZES = (96, 192, ADDITIONAL_SIZE)
RANKER_EPOCHS = 80
RANKER_BATCH = 48
MATCHED_ROWS_PER_EPOCH = ORIGINAL_SIZE * len(SESSIONS)

# Recreate the actual recursive historical cohorts to exclude seven complete groups.
HISTORICAL_ORDER = (961, 2209, 3931, 4931, 5147, 8081, 12457, 16633)
AVOIDED_HELD = (961, 2209, 3931, 4931, 5147, 12457, 16633)


def independent_holdout(count, seed, device):
    historical = {old: outer_split(count, old, HISTORICAL_ORDER[:index], torch.device("cpu"))[1].numpy()
                  for index, old in enumerate(HISTORICAL_ORDER) if old in AVOIDED_HELD}
    excluded = np.concatenate(list(historical.values()))
    left = np.setdiff1d(np.arange(count), excluded)
    if len(left) < 320:
        raise ValueError("Earlier whole-glyph cohorts leave fewer than 320 new validation glyphs.")
    selected = np.sort(np.random.default_rng(seed).permutation(left)[:320])
    fit = np.setdiff1d(np.arange(count), selected)
    return tuple(torch.as_tensor(rows, device=device) for rows in (fit, selected))


@torch.no_grad()
def calibration_labels(packed, rendered, references, scorer):
    """Deduplicate image-reference pairs across the four calibration recordings."""
    rows, columns = packed.valid.nonzero(as_tuple=True)
    encoded = rows.remainder(len(references)) * len(rendered) + packed.indices[rows, columns]
    unique, inverse = torch.unique(encoded, return_inverse=True)
    batch_scores = []
    scored = 0
    for part in unique.split(768):
        candidate = rendered[part.remainder(len(rendered))] >= 128
        reference = references[part.div(len(rendered), rounding_mode="floor")]
        values = scorer(candidate.cpu().numpy(), reference.cpu().numpy(), device=str(rendered.device), batch_size=12)
        batch_scores.append(torch.as_tensor(values, device=rendered.device, dtype=torch.float32))
        scored += len(part)
        if scored % (768 * 5) == 0 or scored == len(unique):
            print(f"Scored {scored:,} of {len(unique):,} unique calibration pairs", flush=True)
    labels = torch.full_like(packed.baseline_z, torch.nan)
    labels[rows, columns] = torch.cat(batch_scores)[inverse]
    return labels, len(unique)


def matched_calibration(training, images, split, scorer):
    """Fit common neural proposals and score a shared pool of labeled OCF judgments."""
    original, noise, literal_inner = split
    new = literal_inner[:ADDITIONAL_SIZE]
    inner = literal_inner[ADDITIONAL_SIZE:]
    metric = torch.cat((original, new))
    neural = {session: counts[inner] for session, counts in training.items()}
    model = fit_v8(neural, images[inner], MODEL_SEED)
    noise_glyphs, _ = align_training_glyphs(model.atlas, images[noise])
    uncertainty = {name: fit_predictive_likelihood(training[name][noise], model.encoder.expected_counts(noise_glyphs, name))
                   for name in SESSIONS}
    bank = rendered_bank(model)
    candidates = concat_atlas_candidates([shortlist(model, uncertainty[name], training[name][metric], name)[0]
                                         for name in SESSIONS])
    labels, measured = calibration_labels(candidates, bank, images[metric] >= 128, scorer)
    by_session = torch.arange(len(SESSIONS), device=images.device)[:, None] * len(metric)
    subsets = {"matched_original_96": (0, ORIGINAL_SIZE),
               **{f"matched_new_{size}": (ORIGINAL_SIZE, ORIGINAL_SIZE + size) for size in NESTED_SIZES}}
    rankers, diagnostics = {}, {}
    all_valid_rows = candidates.valid.shape[0]
    for name, (first, last) in subsets.items():
        rows = (by_session + torch.arange(first, last, device=images.device)).flatten()
        own = candidates.take_rows(rows)
        rankers[name] = fit_atlas_metric_ranker(own, labels[rows], seed=MODEL_SEED, epochs=RANKER_EPOCHS,
                                               batch_rows=RANKER_BATCH, rows_per_epoch=MATCHED_ROWS_PER_EPOCH)
        diagnostics[name] = {**rankers[name].diagnostics, "independent_calibration_glyphs": last - first,
                             "rows_per_epoch": MATCHED_ROWS_PER_EPOCH,
                             "optimizer_updates": RANKER_EPOCHS * MATCHED_ROWS_PER_EPOCH // RANKER_BATCH}
    report = {"original_calibration_glyphs": len(original), "new_calibration_glyphs": len(new), "noise_glyphs": len(noise),
              "shared_neural_fit_glyphs": len(inner), "exact_labels_after_cross_recording_pair_deduplication": measured,
              "labeled_candidate_rows": all_valid_rows, "labeled_candidate_edges": int(candidates.valid.sum()),
              "head_training": diagnostics, "shared_neural_fit": model.report}
    return rankers, report, model.codepoints.cpu()


def comparisons(observations):
    result = {name: summarize(values, observations["v11"]) for name, values in observations.items()}
    for name in ("matched_original_96", "matched_new_192", "matched_new_384"):
        result[f"{name}_vs_matched_new_96"] = summarize(observations[name], observations["matched_new_96"])
    result["matched_new_384_vs_matched_original_96"] = summarize(observations["matched_new_384"], observations["matched_original_96"])
    result["matched_new_384_vs_matched_new_192"] = summarize(observations["matched_new_384"], observations["matched_new_192"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17641)
    parser.add_argument("--ocf-images", type=int, default=192)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_metric_scale.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_images <= 320:
        parser.error("Select two to 320 held glyphs for paired exact OCF.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a private Kaggle GPU with official training input and public fonts available.")
    began = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held = independent_holdout(len(images), args.seed, device)
    rng = np.random.default_rng(args.seed + 1)
    selection = held[torch.as_tensor(rng.choice(len(held), args.ocf_images, replace=False), device=device)]
    recording = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if tuple(recording) != SESSIONS or any(values.shape[1] != 4 for values in recording.values()):
        raise ValueError("Require four training recordings with four repeats per glyph.")
    training = {name: values[fit] for name, values in recording.items()}
    scorer = official_scorer(root)
    split = split_calibration(len(fit), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    print("Fitting the literal v11 96-glyph reference", flush=True)
    v11, uncertainty, stock_fit, stock_codes = nested_ranker(training, images[fit], split, scorer, MODEL_SEED)
    print("Fitting common matched calibration features and exact labels", flush=True)
    rankers, matched_fit, matched_codes = matched_calibration(training, images[fit], split, scorer)
    print("Fitting the one common outer model and candidate generator", flush=True)
    outer = fit_v8(training, images[fit], MODEL_SEED)
    del training
    outer_codes = outer.codepoints.cpu()
    if not (torch.equal(outer_codes, stock_codes) and torch.equal(outer_codes, matched_codes)):
        raise ValueError("All fitted heads and the outer proposal generator must use the same public atlas order.")
    rendered = rendered_bank(outer)
    exact = PairedOCF(scorer, rendered, (images[selection] >= 128).cpu().numpy(), device, 12)
    output = {"mouse": "mice1", "seed": args.seed, "v11_model_seed": MODEL_SEED, "excludes_whole_cohorts": AVOIDED_HELD,
              "outer_fit_glyphs": len(fit), "outer_holdout_glyphs": len(held), "paired_per_recording": len(selection),
              "repeats_per_glyph": 4, "v11_nominations": NOMINATIONS, "v11_anchor": ANCHOR,
              "stock_v11_ranker": v11.diagnostics, "stock_v11_neural_fit": stock_fit,
              "v11_noise_glyphs": len(split[1]), "matched_calibration": matched_fit, "sessions": {},
              "protocol": "Use the literal v11 calibration procedure with 96 label glyphs as the primary control. "
                          "The auxiliary controlled heads share the same neural fit and noise glyphs, and predict "
                          "the same score features. This shared neural fit excludes the original 96 and 384 "
                          "additional label glyphs from all upstream model training. The new 96, 192 and 384 are "
                          "nested within the independent additional set. All heads use 80 epochs and the same "
                          "number of queries and Adam updates per epoch. Larger pools sample a fixed 384 query "
                          "rows without replacement per epoch. Exact calibration pairs are deduplicated across "
                          "recordings, retaining the normal per-query loss. All arms, including the literal v11 "
                          "reference, use identical outer v11 candidate edges, frozen score features, anchor and "
                          "rendered font. Held images enter exact OCF only after ranking. No official test data "
                          "or submission access."}
    all_observations = {}
    models = {"v11": v11, **rankers}
    for session in SESSIONS:
        candidates, _ = shortlist(outer, uncertainty[session], recording[session][selection], session)
        choices = {name: head.select(candidates, anchor_ocf=ANCHOR) for name, head in models.items()}
        observed = {name: exact.evaluate(prediction[:, None])[:, 0] for name, prediction in choices.items()}
        for name, values in observed.items():
            all_observations.setdefault(name, []).append(values)
        report = {"metrics": comparisons({name: [values] for name, values in observed.items()}),
                  "winner_changes_vs_v11": {name: int((chosen != choices["v11"]).sum()) for name, chosen in choices.items()},
                  "fixed_outer_candidate_edges": int(candidates.valid.sum())}
        output["sessions"][session] = report
        output["summary"] = comparisons(all_observations)
        output["unique_outer_exact_image_pairs"] = len(exact.values)
        output["seconds"] = time.monotonic() - began
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, **report}), flush=True)
    output["drop_one_recording"] = {omitted: comparisons({name: [v for i, v in enumerate(values) if i != removed]
                                                          for name, values in all_observations.items()})
                                    for removed, omitted in enumerate(SESSIONS)}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
