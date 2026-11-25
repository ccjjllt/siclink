"""Compare 25 ms counts with compact fast spike timing on Task 3 held repeats."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import load_catalog, load_images, locate_data, split_repeats
from glyph_event_timing import event_contour_scores, event_features, fit_event_contours, read_event_block
from glyph_mbr import contour_signatures
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer


def read_mouse(root, catalog, mouse, n_glyphs, views):
    entries = sorted((entry for entry in catalog["blocks"] if entry["task"] == "train" and entry["mouse"] == mouse),
                     key=lambda entry: entry["session"])
    return {entry["session"]: read_event_block(root, catalog, entry, n_glyphs, views) for entry in entries}


def score_view(view, raw, signature, energy, choice, orders, masks, score, device, repeat_seed):
    training, held = {}, {}
    for index, (session, block) in enumerate(raw.items()):
        features = event_features(block, view, device)
        training[session], held[session] = split_repeats(features, repeat_seed + index)
    model = fit_event_contours(training, signature, energy)
    labels = torch.arange(len(masks), device=device)
    references, predictions, hits = [], {}, {}
    hits5 = 0
    for session, counts in held.items():
        logits = event_contour_scores(model, counts, session)
        ranks = logits.topk(5, dim=1).indices
        hits5 += int((ranks == labels[:, None]).any(dim=1).sum())
        decisions = {"map": ranks[:, 0]}
        if orders:
            order = orders[session]
            assigned = assign_glyphs(logits[order])
            restored = torch.empty_like(assigned)
            restored[order] = assigned
            decisions["balanced"] = restored
        sampled = choice[session]
        references.append(masks[sampled.cpu().numpy()])
        for rule, indices in decisions.items():
            predictions.setdefault(rule, []).append(masks[indices[sampled].cpu().numpy()])
            hits[rule] = hits.get(rule, 0) + int((indices == labels).sum())

    reference = np.concatenate(references)
    measurements = {rule: np.asarray(score(np.concatenate(blocks), reference, device=str(device), batch_size=4))
                    for rule, blocks in predictions.items()}
    n = len(held) * len(masks)
    return measurements, {rule: {"top1": hits[rule] / n, "map_top5": hits5 / n, "top_n": n} for rule in hits}


def run_mouse(mouse, device, root, catalog, scorer, args):
    started = time.monotonic()
    images = load_images(root, device)
    masks = (images >= 128).cpu().numpy()
    signature, energy = contour_signatures(images)
    views = ("count25", *dict.fromkeys(view for view in args.views if view != "count25"))
    raw = read_mouse(root, catalog, mouse, len(images), views)
    print(f"{mouse} on {device}: loaded sparse events from {list(raw)}", flush=True)

    sample_per_session = min(len(images), args.ocf_images // len(raw))
    random = np.random.default_rng(args.seed)
    choice = {session: torch.as_tensor(random.choice(len(images), sample_per_session, replace=False), device=device)
              for session in raw}
    assignment_random = np.random.default_rng(args.seed + 3000)
    orders = {session: torch.as_tensor(assignment_random.permutation(len(images)), device=device)
              for session in raw} if args.compare_balanced else {}
    repeat_seed = args.seed if args.repeat_seed is None else args.repeat_seed
    baseline, results = {}, []
    for view in views:
        view_started = time.monotonic()
        scores, hits = score_view(view, raw, signature, energy, choice, orders, masks, scorer, device, repeat_seed)
        if view == "count25":
            baseline = scores
        for rule, values in scores.items():
            diff = values - baseline[rule]
            sessions = diff.reshape(len(raw), sample_per_session).mean(axis=1)
            row = {"view": view, "rule": rule, "ocf": float(values.mean()), "ocf_n": len(values),
                   "paired_delta_vs_count25": float(diff.mean()),
                   "paired_se_vs_count25": float(diff.std(ddof=1) / len(diff)**0.5),
                   "paired_delta_by_session": dict(zip(raw, sessions.tolist())),
                   "seconds": time.monotonic() - view_started, **hits[rule]}
            print(json.dumps({"mouse": mouse, **row}), flush=True)
            results.append(row)
    return {"mouse": mouse, "device": str(device), "sessions": list(raw), "metrics": results,
            "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", nargs="+", choices=("count25", "phase5", "event"), default=("phase5", "event"))
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ocf-images", type=int, default=384, help="Paired OCF sample per mouse")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int)
    parser.add_argument("--compare-balanced", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/event_timing_validation.json"))
    args = parser.parse_args()
    if args.ocf_images < 4:
        parser.error("Use at least four paired OCF images")

    devices = [] if args.device == "cpu" else [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if args.device == "cuda" and not devices:
        parser.error("The requested CUDA device is unavailable")
    devices = devices or [torch.device("cpu")]
    torch.set_num_threads(4 if devices[0].type == "cpu" else 2)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    scorer = official_scorer(root)
    result = {"seed": args.seed, "repeat_seed": args.seed if args.repeat_seed is None else args.repeat_seed,
              "ocf_images": args.ocf_images, "compare_balanced": args.compare_balanced, "results": []}
    with ThreadPoolExecutor(max_workers=min(len(args.mice), len(devices))) as pool:
        jobs = [pool.submit(run_mouse, mouse, devices[index % len(devices)], root, catalog, scorer, args)
                for index, mouse in enumerate(args.mice)]
        for job in as_completed(jobs):
            result["results"].append(job.result())
            result["results"].sort(key=lambda row: row["mouse"])
            result["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {result['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
