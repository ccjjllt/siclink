"""Compare supervised episodic assignment with the actual v7 Task 2 policy."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_episodic_assignment import fit_episodic_assignment
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_hierarchical_gallery_validation import paired_delta, validated_logits


def measure(score_map, labels, selected, masks, scorer, device, baseline):
    sample_rows = labels.argsort()[selected]
    references = masks[selected].cpu().numpy()
    values, metrics = {}, {}
    for name, matrix in score_map.items():
        chosen = assign_glyphs(matrix)
        guesses = masks[chosen[sample_rows]].float().cpu().numpy()
        values[name] = np.asarray(scorer(guesses, references, device=str(device), batch_size=4))
        top5 = matrix.topk(5, dim=1).indices
        metrics[name] = {"ocf": float(values[name].mean()), "ocf_n": len(selected),
                         "one_to_one_top1": float((chosen == labels).float().mean()),
                         "map_top5": float((top5 == labels[:, None]).any(dim=1).float().mean()),
                         "top_n": len(labels)}
    for name, scores in values.items():
        delta, error = paired_delta(scores, values[baseline])
        metrics[name].update(paired_delta=delta, paired_se=error)
    return metrics, values


def summary(values, baseline):
    original = np.stack(values[baseline])
    metrics = {}
    for name, parts in values.items():
        scores = np.stack(parts)
        delta, error = paired_delta(scores.mean(axis=0), original.mean(axis=0))
        metrics[name] = {"ocf": float(scores.mean()), "ocf_n": scores.size,
                         "paired_delta": delta, "paired_se": error, "unique_scored_glyphs": scores.shape[1]}
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--held-sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session2", "session4"))
    parser.add_argument("--ocf-images", type=int, default=96)
    parser.add_argument("--episodic-epochs", type=int, default=14)
    parser.add_argument("--episode-size", type=int, default=160)
    parser.add_argument("--sinkhorn-weight", type=float, default=0.10)
    parser.add_argument("--view", choices=("evoked-total", "evoked-multiscale"), default="evoked-multiscale")
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--retrieval-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--skip-v7", action="store_true", help="run only the paired episodic and frozen controls")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/episodic_assignment_validation.json"))
    args = parser.parse_args()
    if (args.ocf_images < 2 or min(args.episodic_epochs, args.episode_size, args.encoder_epochs, args.retrieval_epochs) < 1
            or not 0 <= args.sinkhorn_weight <= 0.25):
        parser.error("Choose valid scoring, epoch and episode settings.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    masks = images >= 128
    scorer = official_scorer(root)
    sample = np.random.default_rng(args.seed).choice(len(images), min(len(images), args.ocf_images), replace=False)
    selected = torch.as_tensor(sample, device=device)
    rng = np.random.default_rng(args.seed + 1000)
    baseline = "frozen_prototype" if args.skip_v7 else "v7"
    output = {"device": str(device), "seed": args.seed, "baseline": baseline, "view": args.view,
              "ocf_images_per_fold": len(selected), "episodic_epochs": args.episodic_epochs,
              "episode_size": args.episode_size, "sinkhorn_weight": args.sinkhorn_weight,
              "protocol": "Independent outer recording holdout, four repeats per query, randomized query order. "
                          "One-to-one inference uses scores only. Known outer labels enter only metrics.",
              "mice": {}}
    all_values = {}
    for mouse in args.mice:
        grouped = grouped_blocks(root, catalog, mouse, device, len(images))
        output["mice"][mouse] = {}
        for session in args.held_sessions:
            fold_started = time.monotonic()
            training = {name: counts for name, counts in grouped.items() if name != session}
            permutation = torch.as_tensor(rng.permutation(len(images)), device=device)
            query = grouped[session][permutation]
            print(f"{mouse}/{session}: supervised source episodes on {list(training)}, held block randomized", flush=True)
            model = fit_episodic_assignment(training, view=args.view, epochs=args.episodic_epochs,
                                            batch_size=args.episode_size, sinkhorn_weight=args.sinkhorn_weight,
                                            seed=args.seed)
            scores = model.scores(query)
            fold = {"estimated_episode_model": model.diagnostics, "training_sessions": list(training)}
            output["mice"][mouse][session] = fold
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            if not args.skip_v7:
                print(f"{mouse}/{session}: episodic scores ready, fitting v7 reference", flush=True)
                validated = validated_logits(mouse, training, query, images, device, args)
                hierarchy = fit_hierarchical_gallery(training)
                eb = hierarchy.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
                scores["v7"] = blend_hierarchical_logits(validated, eb)
                scores["v7_episodic"] = blend_hierarchical_logits(scores["v7"], scores["episodic_prototype"])
                del validated, hierarchy, eb
            results, values = measure(scores, permutation, selected, masks, scorer, device, baseline)
            fold.update(metrics=results, seconds=time.monotonic() - fold_started)
            for name, scores_per_glyph in values.items():
                all_values.setdefault(name, []).append(scores_per_glyph)
            output["summary"] = summary(all_values, baseline)
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"mouse": mouse, "session": session, **fold}), flush=True)
            del scores, model, query
        del grouped


if __name__ == "__main__":
    main()
