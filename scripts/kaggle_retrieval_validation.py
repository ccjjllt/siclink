"""Run labeled Task 2 and 3 retrieval comparisons on Kaggle."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import BIN_MS, as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_retrieval import assign_glyphs, combine_logits, fit_retrieval, posterior_images, retrieval_score_components
from kaggle_diagnostics import official_scorer, summarize_metric


def calibration_for(training, images, task, device, args, n_bins):
    from glyph_mbr import fit_temperature

    if task == "task2":
        last = tuple(training)[-1]
        fit = {session: counts for session, counts in training.items() if session != last}
        held = {last: training[last]}
    else:
        folds = {session: split_repeats(counts, args.seed + 1000 + index)
                 for index, (session, counts) in enumerate(training.items())}
        fit = {name: fold[0] for name, fold in folds.items()}
        held = {name: fold[1] for name, fold in folds.items()}

    model = fit_retrieval(as_trial_blocks(fit), len(images), device=device, seed=args.seed + 1,
                          epochs=args.epochs, images=images, n_bins=n_bins, verbose=False)
    labels = torch.arange(len(images), device=device)
    pairs = [(combine_logits(retrieval_score_components(model, counts, name, task=task), task), labels)
             for name, counts in held.items()]
    return {"sessions": list(held), **fit_temperature(pairs)}


def evaluate(model, query, images, *, task, device, score, size, seed, similarity=None,
             temperatures=None, balance=False):
    if similarity is not None:
        from glyph_mbr import glyph_medoid, glyph_posterior

    random = np.random.default_rng(seed)
    masks = (images >= 128).to(torch.uint8) * 255
    labels = torch.arange(len(images), device=device)
    references, predictions, hits, unique = [], {}, {}, {}
    sample_size = min(len(images), size // len(query))
    for session, counts in query.items():
        components = retrieval_score_components(model, counts, session, task=task)
        fused = combine_logits(components, task)
        selection = torch.as_tensor(random.choice(len(images), sample_size, replace=False), device=device)
        references.append(images[selection])

        for name, logits in {**components, "fused": fused}.items():
            top = logits.topk(5, dim=1).indices
            hits.setdefault(name, []).append((int((top[:, 0] == labels).sum()),
                                             int((top == labels[:, None]).any(dim=1).sum())))
            predictions.setdefault(name, []).append(masks[top[selection, 0]])

        assigned = assign_glyphs(fused)
        hits.setdefault("fused_assignment", []).append((int((assigned == labels).sum()), None))
        predictions.setdefault("fused_assignment", []).append(masks[assigned[selection]])
        predictions.setdefault("fused_posterior", []).append(posterior_images(fused[selection], images))
        if similarity is not None:
            for scale, temperature in temperatures.items():
                for balanced in ((False, True) if balance else (False,)):
                    method = "mbr_balanced" if balanced else "mbr"
                    name = f"{method}_{scale:g}x"
                    posterior = glyph_posterior(fused, temperature, balance=balanced)
                    chosen = glyph_medoid(posterior, similarity)
                    hits.setdefault(name, []).append((int((chosen == labels).sum()), None))
                    predictions.setdefault(name, []).append(masks[chosen[selection]])
                    unique.setdefault(name, []).append(chosen.unique().numel() / len(chosen))

    reference = torch.cat(references)
    n = len(query) * len(images)
    metrics = {}
    for name, batches in predictions.items():
        metric = summarize_metric(score, torch.cat(batches), reference, device)
        if name in hits:
            top1, top5 = zip(*hits[name])
            metric.update(top1=sum(top1) / n, top1_n=n)
            if top5[0] is not None:
                metric["top5"] = sum(top5) / n
        if name in unique:
            metric["unique_fraction"] = sum(unique[name]) / len(unique[name])
        metrics[name] = metric
        print(json.dumps({"device": str(device), "task": task, "decoder": name, **metric}), flush=True)
    return metrics


def run_mouse(mouse, device, root, catalog, score, args):
    started = time.monotonic()
    images = load_images(root, device)
    groups = grouped_blocks(root, catalog, mouse, device, len(images))
    n_bins = catalog["window_ms"] // BIN_MS
    similarity = None
    if args.mbr:
        from glyph_mbr import gallery_similarity

        similarity = gallery_similarity(images)
    print(f"{mouse} on {device}: loaded {len(groups)} sessions", flush=True)
    task2_train = {session: counts for session, counts in groups.items() if session != "session4"}
    task2_query = {"session4": groups["session4"]}
    task3_folds = {session: split_repeats(counts, args.seed + offset)
                   for offset, (session, counts) in enumerate(groups.items())}
    task3_train = {session: fold[0] for session, fold in task3_folds.items()}
    task3_query = {session: fold[1] for session, fold in task3_folds.items()}

    result = {"mouse": mouse, "device": str(device), "tasks": {}}
    for task, training, query in (("task2", task2_train, task2_query), ("task3", task3_train, task3_query)):
        task_started = time.monotonic()
        print(f"{mouse} {task}: fitting {list(training)} and evaluating {list(query)}", flush=True)
        calibration = temperatures = None
        if args.mbr:
            calibration = calibration_for(training, images, task, device, args, n_bins)
            temperatures = {scale: calibration["temperature"] * scale for scale in args.mbr_temperature_scales}
            print(json.dumps({"mouse": mouse, "task": task, "mbr_calibration": calibration}), flush=True)
        model = fit_retrieval(as_trial_blocks(training), len(images), device=device, seed=args.seed,
                              epochs=args.epochs, images=images, n_bins=n_bins)
        metrics = evaluate(model, query, images, task=task, device=device, score=score,
                           size=args.ocf_images, seed=args.seed, similarity=similarity,
                           temperatures=temperatures, balance=args.mbr_balance)
        result["tasks"][task] = {"seconds": time.monotonic() - task_started, "metrics": metrics}
        if calibration is not None:
            result["tasks"][task]["mbr_calibration"] = calibration
    result["seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocf-images", type=int, default=48, help="Paired OCF images per mouse, task and decoder")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--mbr", action="store_true", help="Test contour-aware hard gallery predictions")
    parser.add_argument("--mbr-balance", action="store_true", help="Also test the complete-block Sinkhorn posterior")
    parser.add_argument("--mbr-temperature-scales", nargs="+", type=float, default=(1.0,),
                        help="Multipliers for the nested-calibrated temperature")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/retrieval_validation.json"))
    args = parser.parse_args()
    if not 4 <= args.ocf_images <= 128:
        parser.error("--ocf-images must be between 4 and 128")
    if args.mbr_balance and not args.mbr:
        parser.error("--mbr-balance requires --mbr")
    if not all(np.isfinite(scale) and scale > 0 for scale in args.mbr_temperature_scales):
        parser.error("--mbr-temperature-scales requires positive finite scales")
    torch.set_num_threads(2)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    score = official_scorer(root)
    result = {"seed": args.seed, "epochs": args.epochs, "ocf_images": args.ocf_images, "results": []}
    if args.mbr:
        result.update(mbr_balance=args.mbr_balance, mbr_temperature_scales=args.mbr_temperature_scales)
    devices = [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())]
    devices = devices or [torch.device("cpu")]
    print(f"Validation devices: {devices}", flush=True)
    with ThreadPoolExecutor(max_workers=min(2, len(devices))) as executor:
        jobs = [executor.submit(run_mouse, mouse, devices[index % len(devices)], root, catalog, score, args)
                for index, mouse in enumerate(("mice1", "mice2"))]
        for job in as_completed(jobs):
            result["results"].append(job.result())
            result["results"].sort(key=lambda row: row["mouse"])
            result["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {result['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
