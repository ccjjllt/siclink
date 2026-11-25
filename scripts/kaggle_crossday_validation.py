"""Evaluate cross-fitted session adaptation without changing the Task 2 initializer."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_crossday_adaptation import crossday_logits
from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import fit_glyph_encoder, make_count_templates
from glyph_retrieval import assign_glyphs, fit_retrieval
from kaggle_diagnostics import official_scorer


def evaluate(baseline, adapted, labels, selected_glyphs, bank, scorer, device):
    sample_rows = labels.argsort()[selected_glyphs]
    reference = bank[selected_glyphs.cpu().numpy()]
    metrics, recorded, baseline_choices = [], {}, {}
    for method, logits in (("initial", baseline), ("adapted", adapted)):
        top = logits.topk(5, dim=1).indices
        top5 = float((top == labels[:, None]).any(dim=1).float().mean())
        for rule, chosen in (("map", top[:, 0]), ("balanced", assign_glyphs(logits))):
            predictions = bank[chosen[sample_rows].cpu().numpy()]
            values = np.asarray(scorer(predictions, reference, device=str(device), batch_size=4))
            if method == "initial":
                recorded[rule] = values
                baseline_choices[rule] = chosen
            delta = values - recorded[rule]
            metrics.append({"method": method, "rule": rule, "ocf": float(values.mean()), "ocf_n": len(values),
                            "paired_ocf_delta": float(delta.mean()),
                            "paired_ocf_se": float(delta.std(ddof=1) / len(values)**0.5) if len(values) > 1 else None,
                            "top1": float((chosen == labels).float().mean()), "map_top5": top5, "top_n": len(labels),
                            "changed_fraction": float((chosen != baseline_choices[rule]).float().mean())})
    return metrics


def run_mouse(mouse, device, root, catalog, scorer, args):
    started = time.monotonic()
    images = load_images(root, device)
    bank = (images >= 128).cpu().numpy()
    counts = grouped_blocks(root, catalog, mouse, device, len(images))
    selected_glyphs = np.random.default_rng(args.seed).choice(len(images), min(len(images), args.ocf_images),
                                                              replace=False)
    selected_glyphs = torch.as_tensor(selected_glyphs, device=device)
    rng = np.random.default_rng(args.seed + 1000)
    result = {"mouse": mouse, "device": str(device), "folds": {}}
    use_poisson = args.initializer == "poisson"
    for session in args.held_sessions:
        fold_started = time.monotonic()
        training = {name: values for name, values in counts.items() if name != session}
        print(f"{mouse} {session} on {device}: training {args.initializer} on {list(training)}", flush=True)
        retrieval = fit_retrieval(as_trial_blocks(training), len(images), device=device, seed=args.seed,
                                  epochs=0 if use_poisson else args.retrieval_epochs,
                                  images=None if use_poisson else images)
        poisson = None
        if use_poisson:
            encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
            poisson = make_count_templates(encoder, images, training)

        # The decoder sees only spike arrays in randomized query-row order.
        # The inverse ordering is used exclusively for scoring known references.
        permutation = torch.as_tensor(rng.permutation(len(images)), device=device)
        queries = counts[session][permutation]
        options = {"poisson_bank": poisson, "rank": args.rank, "temperature": args.temperature,
                   "shrinkage": args.shrinkage, "maximum_change": args.maximum_change,
                   "correction_weight": args.correction_weight}
        baseline, adapted, details = crossday_logits(retrieval, queries, **options)
        if args.check_permutation:
            second = torch.as_tensor(rng.permutation(len(images)), device=device)
            _, shuffled, _ = crossday_logits(retrieval, queries[second], **options)
            details["permutation_max_abs_error"] = float((shuffled - adapted[second]).abs().max())
        metrics = evaluate(baseline, adapted, permutation, selected_glyphs, bank, scorer, device)
        result["folds"][session] = {"training": list(training), "adaptation": details, "metrics": metrics,
                                    "seconds": time.monotonic() - fold_started}
        print(json.dumps({"mouse": mouse, "session": session, **result["folds"][session]}), flush=True)
    result["seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initializer", choices=("poisson", "fused"), default="poisson")
    parser.add_argument("--held-sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session2", "session4"))
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--retrieval-epochs", type=int, default=12)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--shrinkage", type=float, default=2.0)
    parser.add_argument("--maximum-change", type=float, default=0.20)
    parser.add_argument("--correction-weight", type=float, default=0.35)
    parser.add_argument("--check-permutation", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/crossday_validation.json"))
    args = parser.parse_args()
    if args.ocf_images < 2:
        parser.error("Choose at least two images for the paired OCF comparison")

    torch.set_num_threads(2)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    scorer = official_scorer(root)
    devices = [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())] or [torch.device("cpu")]
    output = {"initializer": args.initializer, "seed": args.seed, "ocf_images": args.ocf_images,
              "held_sessions": args.held_sessions, "encoder_epochs": args.encoder_epochs,
              "retrieval_epochs": args.retrieval_epochs, "rank": args.rank, "temperature": args.temperature,
              "shrinkage": args.shrinkage, "maximum_change": args.maximum_change,
              "correction_weight": args.correction_weight, "results": []}
    with ThreadPoolExecutor(max_workers=min(len(args.mice), len(devices))) as pool:
        jobs = [pool.submit(run_mouse, mouse, devices[index % len(devices)], root, catalog, scorer, args)
                for index, mouse in enumerate(args.mice)]
        for job in as_completed(jobs):
            output["results"].append(job.result())
            output["results"].sort(key=lambda result: result["mouse"])
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
