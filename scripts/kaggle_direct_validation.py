"""Measure continuous reconstruction on the labeled Task 2 and 3 validation folds."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_direct import _render, direct_images, fit_direct
from kaggle_diagnostics import official_scorer, summarize_metric


def evaluate(model, query, images, *, task, device, score, size, seed):
    random = np.random.default_rng(seed)
    reference, predictions = [], {}
    sample_size = min(len(images), size // len(query))
    mask = (images >= 128).float()
    prior = _render(mask.mean(dim=0, keepdim=True), model.foreground, fixed_threshold=0.39, mix=0.30)
    for session, counts in query.items():
        selection = torch.as_tensor(random.choice(len(images), sample_size, replace=False), device=device)
        reference.append(images[selection])
        candidates = direct_images(model, counts, session, selection)
        candidates["static_prior"] = prior.expand(len(selection), -1, -1)
        for name, glyphs in candidates.items():
            predictions.setdefault(name, []).append(glyphs)

    reference = torch.cat(reference)
    metrics = {}
    for name, groups in predictions.items():
        metric = summarize_metric(score, torch.cat(groups), reference, device)
        metrics[name] = metric
        print(json.dumps({"device": str(device), "task": task, "decoder": name, **metric}), flush=True)
    return metrics


def run_mouse(mouse, device, root, catalog, score, args):
    started = time.monotonic()
    images = load_images(root, device)
    groups = grouped_blocks(root, catalog, mouse, device, len(images))
    print(f"{mouse} on {device}: loaded {len(groups)} training sessions", flush=True)
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
        model = fit_direct(training, images, task, seed=args.seed, epochs=args.epochs,
                           contour_epochs=args.contour_epochs, batch_size=args.batch_size)
        metrics = evaluate(model, query, images, task=task, device=device, score=score,
                           size=args.ocf_images, seed=args.seed)
        result["tasks"][task] = {"seconds": time.monotonic() - task_started, "metrics": metrics}
    result["seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocf-images", type=int, default=48, help="Paired OCF images per mouse, task and decoder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--contour-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/direct_validation.json"))
    args = parser.parse_args()
    if not 4 <= args.ocf_images <= 128:
        parser.error("--ocf-images must be between 4 and 128")
    torch.set_num_threads(2)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    score = official_scorer(root)
    result = {"seed": args.seed, "epochs": args.epochs, "contour_epochs": args.contour_epochs,
              "ocf_images": args.ocf_images, "results": []}
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
