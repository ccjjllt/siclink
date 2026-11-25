"""Compare population-gain and Poisson likelihoods on matched seen-glyph folds."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder, make_count_templates
from glyph_gain_likelihood import fit_gain_model, gain_template_logits
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_encoding_validation import seen_folds


def evaluate(training, queries, images, bank, score, device, mouse, task, args):
    name, view = ("shrinkage", args.task2_view) if task == "task2" else ("forward", "timed")
    compare_contour = args.compare_contour and task == "task3"
    gains = {} if compare_contour else {"population_gain": fit_gain_model(training, view)}
    if compare_contour or (args.temporal_gain and view == "timed"):
        gains["temporal_gain"] = fit_gain_model(training, view, temporal=True)
    prior_policy = None
    if compare_contour:
        from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores, fit_contour_retrieval
        from glyph_mbr import contour_signatures

        contours = fit_contour_retrieval(training, *contour_signatures(images), task="task3", seed=args.seed)
        prior_policy = "contour_poisson" if mouse == "mice1" else "contour_ridge"

    labels = torch.arange(len(images), device=device)
    masks = (images >= 128).cpu().numpy()
    random = np.random.default_rng(args.seed)
    assignment_random = np.random.default_rng(args.seed + 3000)
    sample_size = min(len(images), args.ocf_images // len(queries))
    references, predictions, top1, top5 = [], {}, {}, {}
    for session, observed in queries.items():
        selection = random.choice(len(images), sample_size, replace=False)
        selected = torch.as_tensor(selection, device=device)
        references.append(selection)
        order = None
        if task == "task3" and args.compare_balanced:
            # Fix a label- and spike-independent ordering for tie breaking, shared
            # across methods. The Hungarian result is put back in query order.
            order = torch.as_tensor(assignment_random.permutation(len(images)), device=device)
        rates = bank.for_recording(session, observed)[name]
        logits = {"poisson": count_template_logits(observed, rates, view)}
        logits.update({label: gain_template_logits(observed, rates, model) for label, model in gains.items()})
        if compare_contour:
            ridge = contour_retrieval_scores(contours, observed, session)["contour_ridge"]
            logits["contour_ridge"] = ridge
            logits["contour_poisson"] = blend_contour_poisson(ridge, logits["poisson"])
            logits["contour_temporal_gain"] = blend_contour_poisson(ridge, logits["temporal_gain"])
        for label, values in logits.items():
            ranked = values.topk(5, dim=1).indices
            top5[label] = top5.get(label, 0) + int((ranked == labels[:, None]).any(dim=1).sum())
            decisions = {"map": ranked[:, 0]}
            if task == "task2":
                decisions["balanced"] = assign_glyphs(values)
            elif order is not None:
                assigned = assign_glyphs(values[order])
                balanced = torch.empty_like(assigned)
                balanced[order] = assigned
                decisions["balanced"] = balanced
            for rule, chosen in decisions.items():
                key = (label, rule)
                predictions.setdefault(key, []).append(chosen[selected].cpu().numpy())
                top1[key] = top1.get(key, 0) + int((chosen == labels).sum())

    n = len(queries) * len(images)
    predicted_ids = np.stack([np.concatenate(blocks) for blocks in predictions.values()])
    reference_ids = np.broadcast_to(np.concatenate(references), predicted_ids.shape)
    pairs = np.stack((predicted_ids, reference_ids), axis=-1).reshape(-1, 2)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    pair_scores = np.asarray(score(masks[unique[:, 0]], masks[unique[:, 1]], device=str(device), batch_size=8))
    scored = dict(zip(predictions, pair_scores[inverse].reshape(predicted_ids.shape)))
    rows = []
    for (likelihood, rule), values in scored.items():
        differences = values - scored[("poisson", rule)]
        row = {"likelihood": likelihood, "rule": rule, "template": name, "view": view,
               "ocf": float(values.mean()), "ocf_n": len(values),
               "paired_delta": float(differences.mean()),
               "paired_se": float(differences.std(ddof=1) / len(differences)**0.5) if len(values) > 1 else None,
               "top1": top1[(likelihood, rule)] / n, "map_top5": top5[likelihood] / n, "top_n": n}
        if prior_policy is not None:
            prior_delta = values - scored[(prior_policy, "map")]
            per_session = prior_delta.reshape(len(queries), sample_size).mean(axis=1)
            row.update(paired_prior_delta=float(prior_delta.mean()),
                       paired_prior_se=float(prior_delta.std(ddof=1) / len(prior_delta)**0.5),
                       paired_prior_delta_by_session=dict(zip(queries, per_session.tolist())))
        if task == "task3" and rule == "balanced":
            map_delta = values - scored[(likelihood, "map")]
            row.update(paired_same_method_map_delta=float(map_delta.mean()),
                       paired_same_method_map_se=float(map_delta.std(ddof=1) / len(map_delta)**0.5))
        rows.append(row)
    calibration = {label: {"windows_ms": [[25 * start, 25 * end] for start, end in model.windows],
                           "alpha": model.alpha.tolist()} for label, model in gains.items()}
    result = {"calibration": calibration, "metrics": rows}
    if prior_policy is not None:
        result.update(prior_policy=prior_policy, prior_policy_rule="map", contour_likelihood_weights=[0.5, 0.5])
    return result


def run_mouse(mouse, device, root, catalog, score, args):
    started = time.monotonic()
    images = load_images(root, device)
    raw = grouped_blocks(root, catalog, mouse, device, len(images))
    result = {"mouse": mouse, "device": str(device), "tasks": {}}
    repeat_seed = args.seed if args.repeat_seed is None else args.repeat_seed
    for task, (fit, queries) in seen_folds(raw, repeat_seed, args.tasks).items():
        task_started = time.monotonic()
        print(f"{mouse} {task}: fitting {list(fit)}, evaluating {list(queries)} on {device}", flush=True)
        encoder = fit_glyph_encoder(fit, images, epochs=args.encoder_epochs, seed=args.seed)
        bank = make_count_templates(encoder, images, fit)
        metrics = evaluate(fit, queries, images, bank, score, device, mouse, task, args)
        result["tasks"][task] = {"training": list(fit), "queries": list(queries),
                                 "seconds": time.monotonic() - task_started, **metrics}
        print(json.dumps({"mouse": mouse, "task": task, **metrics}), flush=True)
    result["seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("task2", "task3"), default=("task2", "task3"))
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--ocf-images", type=int, default=128, help="Paired OCF sample per mouse and task")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, help="Choose an alternate held-repeat split at fixed OCF draw")
    parser.add_argument("--task2-view", choices=("rate", "timed"), default="rate")
    parser.add_argument("--temporal-gain", action="store_true", help="Also test onset/late gains for timed features")
    parser.add_argument("--compare-contour", action="store_true",
                        help="Compare Task 3 contour, temporal gain and predeclared row-normalized blends")
    parser.add_argument("--compare-balanced", action="store_true",
                        help="Also evaluate one-to-one assignments for complete Task 3 blocks")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/gain_validation.json"))
    args = parser.parse_args()
    if args.ocf_images < 4 or args.encoder_epochs < 1:
        parser.error("Use at least four paired OCF samples and a positive epoch count")
    if args.compare_contour and "task3" not in args.tasks:
        parser.error("--compare-contour requires task3")

    torch.set_num_threads(2)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    score = official_scorer(root)
    devices = [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())] or [torch.device("cpu")]
    output = {"seed": args.seed, "repeat_seed": args.seed if args.repeat_seed is None else args.repeat_seed,
              "encoder_epochs": args.encoder_epochs, "ocf_images": args.ocf_images, "task2_view": args.task2_view,
              "temporal_gain": args.temporal_gain or args.compare_contour, "compare_contour": args.compare_contour,
              "compare_balanced": args.compare_balanced, "results": []}
    # CUDA initializes its linear algebra library lazily and must do so before
    # multiple mouse workers can enter the contour projection concurrently.
    if args.compare_contour and len(args.mice) > 1:
        for device in devices:
            if device.type == "cuda":
                torch.linalg.cholesky(torch.eye(2, device=device))
    with ThreadPoolExecutor(max_workers=min(len(args.mice), len(devices))) as pool:
        jobs = [pool.submit(run_mouse, mouse, devices[index % len(devices)], root, catalog, score, args)
                for index, mouse in enumerate(args.mice)]
        for job in as_completed(jobs):
            output["results"].append(job.result())
            output["results"].sort(key=lambda result: result["mouse"])
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
