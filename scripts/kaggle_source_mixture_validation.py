"""Compare a held-repeat-calibrated source mixture against v7 Task 2 assignment."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits
from glyph_retrieval import assign_glyphs
from glyph_source_mixture import fit_source_mixture
from kaggle_diagnostics import official_scorer
from kaggle_hierarchical_gallery_validation import paired_delta, validated_logits


def evaluate(scores, rows, selected, masks, score, device):
    """Use known labels only for metrics after independent one-to-one assignments."""
    chosen_rows = rows.argsort()[selected]
    references = masks[selected].cpu().numpy()
    measurements, summary = {}, {}
    for name, matrix in scores.items():
        assignment = assign_glyphs(matrix)
        predictions = masks[assignment[chosen_rows]].float().cpu().numpy()
        values = np.asarray(score(predictions, references, device=str(device), batch_size=4))
        measurements[name] = values
        leader = matrix.topk(min(5, len(masks)), dim=1).indices
        delta, error = paired_delta(values, measurements["v7"])
        summary[name] = {"ocf": float(values.mean()), "ocf_n": len(values),
                         "paired_delta_vs_v7": delta, "paired_se": error,
                         "one_to_one_top1": float((assignment == rows).float().mean()),
                         "map_top5": float((leader == rows[:, None]).any(dim=1).float().mean()),
                         "top_n": len(rows)}
    return summary, measurements


def aggregate(measurements):
    baseline = np.stack(measurements["v7"])
    summary = {}
    for name, parts in measurements.items():
        values = np.stack(parts)
        delta, error = paired_delta(values.mean(axis=0), baseline.mean(axis=0))
        summary[name] = {"ocf": float(values.mean()), "ocf_n": values.size,
                         "unique_scored_glyphs": values.shape[1], "paired_delta_vs_v7": delta, "paired_se": error}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--held-sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session2", "session4"))
    parser.add_argument("--ocf-images", type=int, default=96)
    parser.add_argument("--evidence-temperature", type=float, default=0.12)
    parser.add_argument("--row-batch-size", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--retrieval-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/source_mixture_validation.json"))
    args = parser.parse_args()
    if (args.ocf_images < 2 or min(args.row_batch_size, args.encoder_epochs, args.retrieval_epochs) < 1
            or args.evidence_temperature <= 0):
        parser.error("Choose valid fit, scoring and source evidence settings.")
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    masks = images >= 128
    score = official_scorer(root)
    sample = np.random.default_rng(args.seed).choice(len(images), min(args.ocf_images, len(images)), replace=False)
    selected = torch.as_tensor(sample, device=device)
    rng = np.random.default_rng(args.seed + 1000)
    output = {"seed": args.seed, "device": str(device), "ocf_images_per_fold": len(selected),
              "evidence_temperature": args.evidence_temperature, "query_repeat_mix": 0.5,
              "encoder_epochs": args.encoder_epochs, "retrieval_epochs": args.retrieval_epochs,
              "v7": "65% validated per-mouse logits and 35% empirical-Bayes repeat-noise logits, then one-to-one",
              "blend": "65% v7 logits and 35% predictive source-mixture logits, after row centering and global RMS",
              "protocol": "Held training recordings proxy session5. Each complete four-repeat query block is "
                          "shuffled. Marginal held-repeat evidence uses no glyph correspondences or assigned labels.",
              "mice": {}}
    pooled = {}
    for mouse in args.mice:
        grouped = grouped_blocks(root, catalog, mouse, device, len(images))
        record = {}
        output["mice"][mouse] = record
        for session in args.held_sessions:
            fold_started = time.monotonic()
            training = {name: counts for name, counts in grouped.items() if name != session}
            permutation = torch.as_tensor(rng.permutation(len(images)), device=device)
            query = grouped[session][permutation]
            print(f"{mouse}/{session}: fitting v7 and source styles on {list(training)}", flush=True)
            mixture = fit_source_mixture(training)
            reference = validated_logits(mouse, training, query, images, device, args)
            prediction = mixture.hierarchy.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
            v7 = blend_hierarchical_logits(reference, prediction)
            del reference, prediction
            alternatives, calibration = mixture.logits(query, query_repeat_mix=0.5,
                                                        temperature=args.evidence_temperature,
                                                        row_batch_size=args.row_batch_size)
            blended = blend_hierarchical_logits(v7, alternatives["predictive_source_mixture"])
            trials = {"v7": v7, **alternatives, "v7_source_mixture": blended}
            results, measured = evaluate(trials, permutation, selected, masks, score, device)
            for name, values in measured.items():
                pooled.setdefault(name, []).append(values)
            record[session] = {"training_sessions": list(training), "source_fit": mixture.diagnostics,
                               "calibration": calibration, "metrics": results,
                               "seconds": time.monotonic() - fold_started}
            output["summary"] = aggregate(pooled)
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"mouse": mouse, "session": session, **record[session]}), flush=True)
            del trials, v7, blended, alternatives, mixture, query
        del grouped


if __name__ == "__main__":
    main()
