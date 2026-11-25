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


def run_seen(args, root, catalog, images, scorer, device, started):
    output = {"device": str(device), "seed": args.seed, "epochs": args.epochs, "mice": {}}
    for mouse in ("mice1", "mice2"):
        raw = grouped_blocks(root, catalog, mouse, device, len(images))
        task2_fit = {session: counts for session, counts in raw.items() if session != "session4"}
        task2_queries = {"session4": raw["session4"]}
        task3_folds = {session: split_repeats(counts, args.seed + index)
                       for index, (session, counts) in enumerate(raw.items())}
        task3_fit = {session: pair[0] for session, pair in task3_folds.items()}
        task3_queries = {session: pair[1] for session, pair in task3_folds.items()}
        output["mice"][mouse] = {}
        for task, fit, queries in (("task2", task2_fit, task2_queries), ("task3", task3_fit, task3_queries)):
            encoder = fit_glyph_encoder(fit, images, epochs=args.epochs, seed=args.seed)
            rows = evaluate_seen(encoder, fit, queries, images, scorer, device, args.ocf_per_session, args.seed)
            for row in rows:
                print(json.dumps({"mouse": mouse, "task": task, **row}), flush=True)
            output["mice"][mouse][task] = rows
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--ocf-per-session", type=int, default=32)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--mode", choices=("unseen", "seen"), default="unseen")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/encoding.json"))
    args = parser.parse_args()
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
