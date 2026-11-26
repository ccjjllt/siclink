"""Replicate deeper full-evidence Task 1 nominations on two untouched glyph cohorts."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_task1_half_proposal_validation import (ANCHOR_OCF, MODEL_SEED, choose_variants, deeper_prepool,
                                                   expanded_choice, extra_scores, twos)
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank, summarize


EARLIER_SEEDS = (961, 2209, 3931)
REPLICATION_SEEDS = (4931, 5147)
HALF_SEED = 7517


def metrics(observations):
    result = {name: summarize(values, observations["v11"]) for name, values in observations.items()}
    result["deeper_full_only_vs_matched"] = summarize(observations["deeper_full_only"],
                                                       observations["deeper_matched"])
    return result


@torch.no_grad()
def choose_deeper(model, ranker, likelihood, counts, session):
    a, b = twos(counts, HALF_SEED + SESSIONS.index(session))
    with torch.autocast(device_type="cuda"):
        rates = model.encoder.expected_counts(model.atlas.images, session, batch_size=192).float()
    full = atlas_scores(model, likelihood, counts, session, rates=rates)
    halves = tuple(atlas_scores(model, likelihood, spikes, session, rates=rates) for spikes in (a, b))
    base, previous, _, counts = choose_variants(full, halves, ranker)
    prepool_ids, prepool_valid = deeper_prepool(full, base)
    proposed_ids, proposed_valid, extra = extra_scores(prepool_ids, prepool_valid, full, base, ranker)
    base_scores = ranker.rank(base, anchor_ocf=ANCHOR_OCF)
    choices = {"v11": previous["v11"], "deeper_matched": previous["deeper_full_additions"],
               "deeper_full_only": expanded_choice(base, base_scores, proposed_ids, extra)}
    info = {"matched_prepool_mean": float(counts["matched_prepool"].float().mean()),
            "full_only_prepool_mean": float(prepool_valid.sum(dim=1).float().mean()),
            "matched_extra_count": int(counts["added_to_each_arm"].sum()),
            "full_only_extra_count": int(proposed_valid.sum()),
            "matched_extra_histogram": counts["added_to_each_arm"].bincount(minlength=5).tolist(),
            "full_only_extra_histogram": proposed_valid.sum(dim=1).bincount(minlength=5).tolist()}
    return choices, info


def run_fold(seed, earlier, images, recording, scorer, args, result, save):
    started = time.monotonic()
    device = images.device
    fit, held = outer_split(len(images), seed, earlier, device)
    chosen = torch.as_tensor(np.random.default_rng(seed + 1).choice(len(held), args.ocf_images, replace=False), device=device)
    sample = held[chosen]
    fit_images = images[fit]
    training = {name: counts[fit] for name, counts in recording.items()}
    split = split_calibration(len(fit), 96, 384, device, MODEL_SEED)
    print(f"seed {seed}: fit exact v11 atlas and nested ranker on {len(fit)} glyphs", flush=True)
    ranker, likelihoods, fit_report, codepoints = nested_ranker(training, fit_images, split, scorer, MODEL_SEED)
    model = fit_v8(training, fit_images, MODEL_SEED)
    del training
    if not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("Nested and outer models must retain identical atlas order.")
    exact = PairedOCF(scorer, rendered_bank(model), (images[sample] >= 128).cpu().numpy(), device, 12)
    result.update(outer_seed=seed, recursively_excluded_seeds=earlier, fit_glyphs=len(fit), held_glyphs=len(held),
                  paired_glyphs_per_session=len(sample), ranker=ranker.diagnostics, nested_fit=fit_report, sessions={})
    observations = {}
    for session in SESSIONS:
        counts = recording[session][sample]
        choices, matched = choose_deeper(model, ranker, likelihoods[session], counts, session)
        measurements = {name: exact.evaluate(indices[:, None])[:, 0] for name, indices in choices.items()}
        for name, values in measurements.items():
            observations.setdefault(name, []).append(values)
        report = {"metrics": metrics({name: [values] for name, values in measurements.items()}), **matched}
        report["winners_changed_vs_v11"] = {name: int((indices != choices["v11"]).sum())
                                             for name, indices in choices.items() if name != "v11"}
        report["full_only_vs_matched_winner_changes"] = int((choices["deeper_full_only"] != choices["deeper_matched"]).sum())
        result["sessions"][session] = report
        result["summary"] = metrics(observations)
        result["seconds"] = time.monotonic() - started
        result["unique_exact_pairs"] = len(exact.values)
        save()
        print(json.dumps({"seed": seed, "session": session, **report}), flush=True)
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocf-images", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_deeper_replication.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle GPU with official training data and Internet enabled.")
    if not 2 <= args.ocf_images <= 320:
        parser.error("Use between two and 320 paired OCF images per recording.")

    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, "cuda:0")
    recording = grouped_blocks(root, load_catalog(root), "mice1", images.device, len(images))
    if any(counts.shape[1] != 4 for counts in recording.values()):
        raise ValueError("The unseen-glyph proxy uses the four labeled repeats in each training group.")
    scorer = official_scorer(root)
    output = {"mouse": "mice1", "model_seed": MODEL_SEED, "replication_seeds": REPLICATION_SEEDS,
              "historical_seeds": EARLIER_SEEDS, "half_seed": HALF_SEED, "query_repeats": 4,
              "anchor_ocf": ANCHOR_OCF, "original_nominations": NOMINATIONS, "precommitted_method": "deeper_matched",
              "protocol": "Two disjoint new 320-glyph outer cohorts exclude all three previous cohort holds. The "
                          "primary replays the original deeper-full arm literally, including the same dependence "
                          "on split-repeat prepool size to equalize available candidates. Full-only sensitivity "
                          "keeps all valid candidates from the fixed full evidence prepool. Both use the same "
                          "full-count schema and unchanged v11 ranker and base scores. Only official training "
                          "records are used. Four measured repeats serve as a proxy for formal twenty-repeat queries.",
              "folds": []}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    combined = []
    for index, seed in enumerate(REPLICATION_SEEDS):
        record = {}
        output["folds"].append(record)
        earlier = (*EARLIER_SEEDS, *REPLICATION_SEEDS[:index])
        combined.append(run_fold(seed, earlier, images, recording, scorer, args, record, save))
        torch.cuda.empty_cache()
    pooled = {name: [np.concatenate([fold[name][session] for fold in combined]) for session in range(len(SESSIONS))]
              for name in combined[0]}
    output["pooled_summary"] = metrics(pooled)
    save()
    print(json.dumps({"pooled": output["pooled_summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
