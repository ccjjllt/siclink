"""Independently recheck the precommitted mice1 Task 3 density top-seven policy."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_part_discovery import SESSIONS
from kaggle_task3_repeat_posterior import CLASSIFIER_WEIGHT, fit_models, score_sources


PRIMARY = "density_within_v11_top7"
N_SCORED = 192


def top7_predictions(v11, density):
    anchor = v11.argmax(dim=1, keepdim=True)
    remainder = v11.clone().scatter_(1, anchor, -torch.inf).topk(15, dim=1).indices[:, :6]
    shortlist = torch.cat((anchor, remainder), dim=1)
    elected = shortlist.gather(1, density.gather(1, shortlist).argmax(dim=1, keepdim=True))
    return anchor[:, 0], elected[:, 0]


def new_sample(gallery_size, seed):
    excluded = np.zeros(gallery_size, dtype=bool)
    earlier = {"density_ratio_evaluation": np.random.default_rng(1901).choice(gallery_size, 64, replace=False),
               "posterior_calibration_and_evaluation": np.random.default_rng(2729).permutation(gallery_size)[:320],
               "selector_temperature_ocf_calibration_and_evaluation": np.random.default_rng(3727).permutation(gallery_size)[:400]}
    for rows in earlier.values():
        excluded[rows] = True
    remaining = np.flatnonzero(~excluded)
    sampled = np.random.default_rng(seed).choice(remaining, N_SCORED, replace=False)
    return sampled, {"excluded_counts_by_experiment": {name: len(rows) for name, rows in earlier.items()},
                     "excluded_union": int(excluded.sum()), "eligible": len(remaining)}


def summarize(observations):
    baseline = np.stack(observations["v11"])
    result = {}
    for name, samples in observations.items():
        values = np.stack(samples)
        delta = (values - baseline).mean(axis=0)
        result[name] = {"ocf": float(values.mean()), "delta_vs_v11": float(delta.mean()),
                        "paired_glyph_se": float(delta.std(ddof=1) / np.sqrt(len(delta))), "scored_rows": int(values.size)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_density_top7_recheck.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on Kaggle T4 with competition training data mounted.")
    # These fit and decision settings match the independent previous pilot.
    args.seed, args.encoder_epochs, args.epochs, args.dimensions = 211, 18, 8, 128
    repeat_seed, sample_seed = 4813, 4817
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    masks = images >= 128
    signature, energy = contour_signatures(images)
    scorer = official_scorer(root)
    sampled, mask = new_sample(len(images), sample_seed)
    sample = torch.as_tensor(sampled, device=device)
    recordings = grouped_blocks(root, catalog, "mice1", device, len(images))
    training, held = {}, {}
    for index, (session, counts) in enumerate(recordings.items()):
        if counts.shape[1] != 4:
            raise ValueError("The independent proxy requires four measured training repeats.")
        training[session], held[session] = split_repeats(counts, repeat_seed + index)
    del recordings
    print(f"mice1: refit on three repeats, paired cohort {len(sample)} of {mask['eligible']} eligible glyphs", flush=True)
    models, fit = fit_models(training, images, signature, energy, "mice1", args)
    del training
    result = {"primary_precommitted": PRIMARY, "mouse": "mice1", "gallery_size": len(images),
              "repeat_seed": repeat_seed, "scored_glyph_seed": sample_seed, "scored_glyphs_per_block": len(sample),
              "classifier_mix": CLASSIFIER_WEIGHT, "selector_width": 7, "fit": fit, "eligible_mask": mask,
              "protocol": "Three non-held training repeats, 192 previously unscored references per session. All 2,748 known glyphs "
                          "remain in training and gallery. Shuffled measured outer rows influence predictions only through spikes. "
                          "Rank the fixed 65/35 score directly since positive temperature scaling preserves its order.",
              "folds": []}
    rng = np.random.default_rng(sample_seed + 1)
    accumulated = {}
    for session in SESSIONS:
        begin = time.monotonic()
        ordering = torch.as_tensor(rng.permutation(len(images)), device=device)
        v11, _, density = score_sources(models, held[session][ordering], images, session, "mice1", args.dimensions)
        anchor, top7_label = top7_predictions(v11, density)
        chosen = {"v11": anchor, "density_full_gallery": density.argmax(dim=1), PRIMARY: top7_label}
        rows = ordering.argsort()[sample]
        predictions = torch.stack([value[rows] for value in chosen.values()])
        exact, pairs = unique_pair_ocf(predictions, sample.expand_as(predictions), masks, scorer, device, 384)
        block = dict(zip(chosen, exact.cpu().numpy()))
        for name, samples in block.items():
            accumulated.setdefault(name, []).append(samples)
        metrics = {name: {"gallery_top1": float((labels == ordering).float().mean()),
                          "changed_vs_v11": float((labels != anchor).float().mean())}
                   for name, labels in chosen.items()}
        fold = {"session": session, "unique_ocf_pairs": pairs, "gallery": metrics,
                "paired": summarize({name: [values] for name, values in block.items()}),
                "seconds": time.monotonic() - begin}
        result["folds"].append(fold)
        result["summary"] = summarize(accumulated)
        result["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(fold), flush=True)
    print(json.dumps({"summary": result["summary"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
