"""Validate learned image-to-spike encoding and inverse reconstruction on unseen glyphs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs, split_repeats
from glyph_encoding import (count_template_logits, evaluate_glyph_encoder, fit_glyph_encoder,
                            make_count_templates, poisson_logits)
from glyph_reconstruction import fit_glyph_reconstructor
from kaggle_diagnostics import feature_view, official_scorer, standardize


def evaluate_seen(encoder, training, queries, images, scorer, device, size, seed):
    from glyph_retrieval import assign_glyphs

    random = np.random.default_rng(seed)
    reference = (images >= 128).cpu().numpy()
    n_per_session = min(len(images), max(1, size // len(queries)))
    results = []
    expected_indices = torch.arange(len(images), device=device)
    templates = make_count_templates(encoder, images, training)
    for session, observed in queries.items():
        selection = random.choice(len(images), n_per_session, replace=False)
        selection_index = torch.as_tensor(selection, device=device)
        for name, rates in templates.for_recording(session, observed).items():
            for view in ("rate", "timed"):
                logits = count_template_logits(observed, rates, view)
                top = logits.topk(5, dim=1).indices
                predictions = {"map": top[:, 0]}
                if session not in training and name == "shrinkage":
                    predictions["balanced"] = assign_glyphs(logits)
                for rule, indices in predictions.items():
                    reconstructed = reference[indices[selection_index].cpu().numpy()]
                    ocf = scorer(reconstructed, reference[selection], device=str(device), batch_size=4)
                    results.append({"session": session, "template": name, "view": view, "decision": rule,
                                    "ocf": float(ocf.mean()), "ocf_n": len(ocf),
                                    "top1": float((indices == expected_indices).float().mean()),
                                    "top5": float((top == expected_indices[:, None]).any(1).float().mean()),
                                    "top_n": len(indices)})
    return results


def seen_folds(raw, seed, tasks=("task2", "task3")):
    folds = {}
    if "task2" in tasks:
        folds["task2"] = ({session: counts for session, counts in raw.items() if session != "session4"},
                          {"session4": raw["session4"]})
    if "task3" in tasks:
        repeats = {session: split_repeats(counts, seed + index)
                   for index, (session, counts) in enumerate(raw.items())}
        folds["task3"] = ({session: pair[0] for session, pair in repeats.items()},
                          {session: pair[1] for session, pair in repeats.items()})
    return folds


def run_seen(args, root, catalog, images, scorer, device, started):
    output = {"device": str(device), "seed": args.seed, "epochs": args.epochs, "mice": {}}
    for mouse in ("mice1", "mice2"):
        raw = grouped_blocks(root, catalog, mouse, device, len(images))
        output["mice"][mouse] = {}
        for task, (fit, queries) in seen_folds(raw, args.seed).items():
            encoder = fit_glyph_encoder(fit, images, epochs=args.epochs, seed=args.seed)
            rows = evaluate_seen(encoder, fit, queries, images, scorer, device, args.ocf_per_session, args.seed)
            for row in rows:
                print(json.dumps({"mouse": mouse, "task": task, **row}), flush=True)
            output["mice"][mouse][task] = rows
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n")


def adaptation_fold(templates, observed, images, score, device, args):
    from glyph_retrieval import assign_glyphs
    from glyph_session_adaptation import session_adaptation_logits

    reference = (images >= 128).cpu().numpy()
    labels = torch.arange(len(images), device=device)
    selection = np.random.default_rng(args.seed).choice(len(images), min(len(images), args.ocf_per_session),
                                                        replace=False)
    sampled = torch.as_tensor(selection, device=device)
    rows, calibration = [], {}
    for view in args.adapt_views:
        variants, calibration[view] = session_adaptation_logits(observed, templates, view)
        baseline = {}
        for name, logits in variants.items():
            top = logits.topk(5, dim=1).indices
            top5 = float((top == labels[:, None]).any(dim=1).float().mean())
            for rule, chosen in (("map", top[:, 0]), ("balanced", assign_glyphs(logits))):
                prediction = reference[chosen[sampled].cpu().numpy()]
                values = np.asarray(score(prediction, reference[selection], device=str(device), batch_size=4))
                if name == "baseline":
                    baseline[rule] = values
                delta = values - baseline[rule]
                rows.append({"view": view, "template": name, "decision": rule,
                             "ocf": float(values.mean()), "ocf_n": len(values),
                             "paired_ocf_delta": float(delta.mean()),
                             "paired_ocf_se": float(delta.std(ddof=1) / len(delta)**0.5) if len(delta) > 1 else None,
                             "top1": float((chosen == labels).float().mean()), "top_n": len(labels),
                             "map_top5": top5})
    return rows, calibration


def run_adaptation(args, root, catalog, images, scorer, device, started):
    output = {"mode": "adaptation", "device": str(device), "seed": args.seed, "epochs": args.epochs,
              "ocf_per_session": args.ocf_per_session, "mice": {}}
    for mouse in args.adapt_mice:
        raw = grouped_blocks(root, catalog, mouse, device, len(images))
        output["mice"][mouse] = {}
        for session in args.adapt_sessions:
            fit = {name: counts for name, counts in raw.items() if name != session}
            observed = raw[session]
            encoder = fit_glyph_encoder(fit, images, epochs=args.epochs, seed=args.seed)
            templates = make_count_templates(encoder, images, fit).for_recording(session, observed)["shrinkage"]
            rows, calibration = adaptation_fold(templates, observed, images, scorer, device, args)
            for row in rows:
                print(json.dumps({"mouse": mouse, "session": session, **row}), flush=True)
            output["mice"][mouse][session] = {"training": list(fit), "calibration": calibration, "metrics": rows}
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def evaluate_negative_binomial(templates, training, queries, images, scorer, device, task, size, seed):
    from glyph_negative_binomial import fit_count_dispersion, negative_binomial_template_logits
    from glyph_retrieval import assign_glyphs

    template, view = ("shrinkage", "rate") if task == "task2" else ("forward", "timed")
    alpha = fit_count_dispersion(training, view)
    labels = torch.arange(len(images), device=device)
    bank = (images >= 128).cpu().numpy()
    random = np.random.default_rng(seed)
    sample_size = min(len(images), max(1, size // len(queries)))
    references, predictions, hits, top5 = [], {}, {}, {}
    for session, observed in queries.items():
        selection = random.choice(len(images), sample_size, replace=False)
        selected = torch.as_tensor(selection, device=device)
        references.append(bank[selection])
        rates = templates.for_recording(session, observed)[template]
        scores = {"poisson": count_template_logits(observed, rates, view),
                  "negative_binomial": negative_binomial_template_logits(observed, rates, view, alpha)}
        for name, logits in scores.items():
            top = logits.topk(5, dim=1).indices
            top5[name] = top5.get(name, 0) + int((top == labels[:, None]).any(dim=1).sum())
            decisions = {"map": top[:, 0]}
            if task == "task2":
                decisions["balanced"] = assign_glyphs(logits)
            for rule, chosen in decisions.items():
                key = (name, rule)
                predictions.setdefault(key, []).append(bank[chosen[selected].cpu().numpy()])
                hits[key] = hits.get(key, 0) + int((chosen == labels).sum())

    reference = np.concatenate(references)
    n = len(queries) * len(images)
    rows, baseline = [], {}
    for (name, rule), blocks in predictions.items():
        values = np.asarray(scorer(np.concatenate(blocks), reference, device=str(device), batch_size=4))
        if name == "poisson":
            baseline[rule] = values
        difference = values - baseline[rule]
        rows.append({"likelihood": name, "template": template, "view": view, "decision": rule,
                     "ocf": float(values.mean()), "ocf_n": len(values),
                     "paired_ocf_delta": float(difference.mean()),
                     "paired_ocf_se": float(difference.std(ddof=1) / len(values)**0.5) if len(values) > 1 else None,
                     "top1": hits[(name, rule)] / n, "map_top5": top5[name] / n, "top_n": n})
    summary = {"features": len(alpha), "nonzero_fraction": float((alpha > 0).float().mean()),
               "mean_alpha": float(alpha.mean()), "median_alpha": float(alpha.median()), "max_alpha": float(alpha.max())}
    return {"dispersion": summary, "metrics": rows}


def run_negative_binomial(args, root, catalog, images, scorer, device, started):
    output = {"mode": "negative-binomial", "device": str(device), "seed": args.seed, "epochs": args.epochs,
              "ocf_images_per_task": args.nb_ocf_images, "mice": {}}
    for mouse in args.nb_mice:
        raw = grouped_blocks(root, catalog, mouse, device, len(images))
        output["mice"][mouse] = {}
        for task, (fit, queries) in seen_folds(raw, args.seed, args.nb_tasks).items():
            print(f"{mouse} {task}: fitting encoder on {list(fit)} and evaluating {list(queries)}", flush=True)
            encoder = fit_glyph_encoder(fit, images, epochs=args.epochs, seed=args.seed)
            templates = make_count_templates(encoder, images, fit)
            result = evaluate_negative_binomial(templates, fit, queries, images, scorer, device,
                                                task, args.nb_ocf_images, args.seed)
            output["mice"][mouse][task] = {"training": list(fit), "queries": list(queries), **result}
            output["seconds"] = time.monotonic() - started
            print(json.dumps({"mouse": mouse, "task": task, **result}), flush=True)
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--ocf-per-session", type=int, default=32)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--mode", choices=("unseen", "seen", "adaptation", "negative-binomial"), default="unseen")
    parser.add_argument("--adapt-mice", nargs="+", choices=("mice1", "mice2"), default=("mice2",))
    parser.add_argument("--adapt-sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session2", "session4"))
    parser.add_argument("--adapt-views", nargs="+", choices=("rate", "timed"), default=("rate",))
    parser.add_argument("--nb-mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--nb-tasks", nargs="+", choices=("task2", "task3"), default=("task2", "task3"))
    parser.add_argument("--nb-ocf-images", type=int, default=128, help="Paired OCF sample per mouse and task")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/encoding.json"))
    args = parser.parse_args()
    if args.ocf_per_session < 1:
        parser.error("--ocf-per-session must be positive")
    if args.mode == "negative-binomial" and args.nb_ocf_images < 4:
        parser.error("--nb-ocf-images must be at least 4")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    task_score = official_scorer(root)
    if args.mode == "seen":
        run_seen(args, root, catalog, images, task_score, device, started)
        return
    if args.mode == "adaptation":
        run_adaptation(args, root, catalog, images, task_score, device, started)
        return
    if args.mode == "negative-binomial":
        run_negative_binomial(args, root, catalog, images, task_score, device, started)
        return
    train, hold = split_glyphs(len(images), 320, args.seed, device)
    rng = np.random.default_rng(args.seed + 1)
    indices = torch.as_tensor(rng.choice(len(hold), args.ocf_per_session, replace=False), device=device)
    queries = hold[indices]
    reference = (images[queries] >= 128).cpu().numpy()
    target_columns = torch.arange(len(hold), device=device)
    output = {"device": str(device), "seed": args.seed, "epochs": args.epochs, "mice": {}}

    for mouse in ("mice1", "mice2"):
        raw = grouped_blocks(root, catalog, mouse, device, len(images))
        responses = {session: counts[train] for session, counts in raw.items()}
        held_responses = {session: counts[hold] for session, counts in raw.items()}
        encoder = fit_glyph_encoder(responses, images[train], epochs=args.epochs, seed=args.seed)
        encoding = evaluate_glyph_encoder(encoder, images[hold], held_responses)
        for session, counts in held_responses.items():
            # Oracle gallery is a train-set diagnostic. Test Task 1 images are unavailable.
            scores = poisson_logits(counts, encoder.expected_counts(images[hold], session))
            ranks = scores.topk(5, dim=1).indices == target_columns[:, None]
            encoding[session].update(oracle_gallery_top1=ranks[:, 0].float().mean().item(),
                                      oracle_gallery_top5=ranks.any(1).float().mean().item())
        print(json.dumps({"mouse": mouse, "encoding": encoding}), flush=True)

        prototype, stats = {}, {}
        for session, counts in responses.items():
            normalized, stats[session] = standardize(feature_view(counts, "evoked-multiscale"))
            prototype[session] = normalized.mean(1)
        decoder = fit_glyph_reconstructor(prototype, images[train], seed=args.seed, refine_epochs=0)
        inversion = {}
        for session in ("session1", "session4"):
            counts = raw[session][queries]
            query, _ = standardize(feature_view(counts, "evoked-multiscale"), stats[session])
            _, continuous = decoder.readout.predict(session, query.mean(1))
            initial = continuous.clamp(0.05, 0.95)
            refined = encoder.invert(counts, initial, session, steps=30, prior_weight=0.075)
            foreground = decoder.foreground
            candidates = {"kernel_gray": initial, "inverted_gray": refined}
            for name, estimate in tuple(candidates.items()):
                threshold = 0.75 * torch.quantile(estimate.flatten(1), 1.0 - foreground, dim=1) + 0.25 * 0.45
                candidates[f"{name}_binary"] = (estimate >= threshold[:, None, None]).float()
            inversion[session] = {
                name: float(task_score(pred.detach().cpu().numpy(), reference, device=str(device), batch_size=4).mean())
                for name, pred in candidates.items()
            }
            print(json.dumps({"mouse": mouse, "session": session, "inversion_ocf": inversion[session]}), flush=True)
        output["mice"][mouse] = {"encoding": encoding, "inversion_ocf": inversion, "ocf_n": len(indices)}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
