"""Validate source-constrained latent glyph loadings on complete Task 2 holdouts."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import VIEWS, blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs
from glyph_shared_stimulus import fit_shared_stimulus
from kaggle_diagnostics import official_scorer
from kaggle_hierarchical_gallery_validation import paired_delta, validated_logits


SESSIONS = ("session1", "session2", "session3", "session4")


def evaluate(matrices, permutation, selected, masks, scorer, device):
    """Rank complete blocks before consulting any held glyph correspondences."""
    sample_rows = permutation.argsort()[selected]
    chosen, top5 = {}, {}
    for name, logits in matrices.items():
        chosen[name] = assign_glyphs(logits)
        leaders = logits.topk(5, dim=1).indices
        top5[name] = float((leaders == permutation[:, None]).any(dim=1).float().mean())
    picked = torch.stack([labels[sample_rows] for labels in chosen.values()], dim=1).cpu().numpy()
    pairs = np.column_stack((np.repeat(np.arange(len(selected)), len(matrices)), picked.ravel()))
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    references = masks[selected].cpu().numpy()
    predictions = masks[torch.as_tensor(unique[:, 1], device=device)].cpu().numpy()
    measurements = np.asarray(scorer(predictions, references[unique[:, 0]], device=str(device), batch_size=12))
    values = measurements[inverse].reshape(picked.shape)
    scores = {name: values[:, i] for i, name in enumerate(matrices)}
    metrics = {}
    for name, observed in scores.items():
        delta, error = paired_delta(observed, scores["v8_task2"])
        metrics[name] = {"ocf": float(observed.mean()), "ocf_n": len(observed), "paired_delta": delta,
                         "paired_delta_se": error, "map_top5": top5[name],
                         "one_to_one_top1": float((chosen[name] == permutation).float().mean()),
                         "ranked_queries": len(permutation)}
    return scores, metrics


def summarize(measurements):
    expected = np.stack(measurements["v8_task2"])
    summary = {}
    for name, observed in measurements.items():
        values = np.stack(observed)
        delta, error = paired_delta(values.mean(axis=0), expected.mean(axis=0))
        summary[name] = {"ocf": float(values.mean()), "ocf_n": values.size, "paired_delta": delta,
                         "paired_delta_se": error, "unique_scored_glyphs": values.shape[1]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--held-sessions", nargs="+", choices=SESSIONS, default=SESSIONS)
    parser.add_argument("--ocf-images", type=int, default=96)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--view", choices=tuple(VIEWS), default="total")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--retrieval-epochs", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/shared_stimulus_validation.json"))
    args = parser.parse_args()
    if not (2 <= args.ocf_images <= 2748 and min(args.rank, args.encoder_epochs, args.retrieval_epochs) > 0):
        parser.error("Choose valid ranks, epochs and a paired scoring count.")
    torch.set_num_threads(4)
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    score = official_scorer(root)
    images = load_images(root, device)
    masks = images >= 128
    selected = torch.as_tensor(np.random.default_rng(args.seed).choice(len(images), args.ocf_images, replace=False),
                               device=device)
    rng = np.random.default_rng(args.seed + 1000)
    output = {"seed": args.seed, "device": str(device), "rank": args.rank, "windows_ms": VIEWS[args.view],
              "ocf_images_per_fold": len(selected), "encoder_epochs": args.encoder_epochs,
              "retrieval_epochs": args.retrieval_epochs, "target_noise_mix": 0.50, "maximum_template_change": 0.15,
              "paper": "https://www.nature.com/articles/s41467-025-59652-y",
              "v8_task2": "Original validated per-mouse logits (65%) plus repeat-noise empirical Bayes (35%).",
              "v8_adapted_blend": "Row-centered global-RMS blend of v8 Task 2 (65%) and adapted loading (35%).",
              "protocol": "Whole training recording holdouts proxy the independent session5. Source labels fit shared stimulus codes and recording loadings. Target rows are shuffled; only their unlabeled repeat covariance and denoised population moments adapt the loadings. Held correspondence is used solely for paired OCF and retrieval accuracy.",
              "folds": []}
    accumulated = {}
    for mouse in dict.fromkeys(args.mice):
        grouped = grouped_blocks(root, catalog, mouse, device, len(images))
        for session in dict.fromkeys(args.held_sessions):
            fold_started = time.monotonic()
            training = {name: values for name, values in grouped.items() if name != session}
            permutation = torch.as_tensor(rng.permutation(len(images)), device=device)
            held = grouped[session][permutation]
            print(f"{mouse}/{session}: fitting latent loadings and exact v8 control", flush=True)
            model = fit_shared_stimulus(training, rank=args.rank, view=args.view)
            alternatives, diagnostics = model.logits(held)
            reference = validated_logits(mouse, training, held, images, device, args)
            hierarchy = fit_hierarchical_gallery(training, view="total")
            eb = hierarchy.logits(held, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
            v8 = blend_hierarchical_logits(reference, eb)
            matrices = {"v8_task2": v8, **alternatives,
                        "v8_adapted_blend": blend_hierarchical_logits(v8, alternatives["adapted_loading"])}
            measured, metrics = evaluate(matrices, permutation, selected, masks, score, device)
            for name, values in measured.items():
                accumulated.setdefault(name, []).append(values)
            fold = {"mouse": mouse, "session": session, "training": model.diagnostics, "adaptation": diagnostics,
                    "metrics": metrics, "seconds": time.monotonic() - fold_started}
            output["folds"].append(fold)
            output["summary"] = summarize(accumulated)
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(fold), flush=True)
            del matrices, alternatives, v8, reference, eb, hierarchy
        del grouped
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
