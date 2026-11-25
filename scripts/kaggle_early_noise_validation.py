"""Test early post-onset conditioning on paired Task 3 held-repeat folds."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_early_noise import (CONDITIONERS, EVOKED_WINDOWS, fit_early_conditioner, fit_evoked_templates,
                               split_repeat_variance)
from kaggle_diagnostics import official_scorer


def rank_metrics(scores, masks, selected, scorer, device):
    ranked = scores.topk(min(5, len(masks)), dim=1).indices
    labels = torch.arange(len(masks), device=device)
    targets = masks[selected].cpu().numpy()
    predictions = masks[ranked[selected, 0]].float().cpu().numpy()
    values = np.asarray(scorer(predictions, targets, device=str(device), batch_size=4))
    counts = {"top1_count": int((ranked[:, 0] == labels).sum()),
              "top5_count": int((ranked == labels[:, None]).any(dim=1).sum()), "top_n": len(labels)}
    return counts, values


def compare_metrics(counts, values, baseline):
    delta = values - baseline
    return {"top1": counts["top1_count"] / counts["top_n"], "top5": counts["top5_count"] / counts["top_n"],
            "top_n": counts["top_n"], "ocf": float(values.mean()), "ocf_n": len(values),
            "paired_ocf_delta": float(delta.mean()),
            "paired_ocf_se": float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None}


@torch.inference_mode()
def evaluate(training, queries, selected, masks, scorer, device, args, view, early_ms, contours):
    nuisance = fit_early_conditioner(training, early_ms=early_ms, view=view, ridge=args.ridge)
    original_fit = {session: nuisance.transform(counts, session, "unconditioned") for session, counts in training.items()}
    original_queries = {session: nuisance.transform(counts, session, "unconditioned") for session, counts in queries.items()}
    results = {}
    for variant in CONDITIONERS:
        if variant == "unconditioned":
            fit, held = original_fit, original_queries
        else:
            fit = {session: nuisance.transform(counts, session, variant) for session, counts in training.items()}
            held = {session: nuisance.transform(counts, session, variant) for session, counts in queries.items()}
        templates = fit_evoked_templates(fit)
        contour_readout = None
        if contours is not None:
            from glyph_contour_retrieval import _fit_readout

            signature, energy = contours
            log_energy = energy.log()
            contour_readout = _fit_readout(tuple(fit.values()), torch.cat((signature, log_energy[:, None]), dim=1), ridge=0.12)
        for session, features in held.items():
            trimmed = templates[session].logits(features)
            statistics, ocf = rank_metrics(trimmed, masks, selected[session], scorer, device)
            noise = split_repeat_variance(original_fit[session], original_queries[session], fit[session], features)
            results[("template", variant, session)] = (statistics, ocf, noise)
            del trimmed
            if contour_readout is not None:
                from glyph_mbr import contour_utility

                inferred = contour_readout.predict(features.mean(dim=1))
                inferred_energy = inferred[:, -1].clamp(log_energy.min(), log_energy.max()).exp()
                shape = contour_utility(inferred[:, :-1], inferred_energy, signature, energy)
                statistics, ocf = rank_metrics(shape, masks, selected[session], scorer, device)
                results[("contour", variant, session)] = (statistics, ocf, noise)
                del shape

    rows = []
    summaries = {}
    for (decoder, variant, session), (counts, values, noise) in results.items():
        baseline = results[(decoder, "unconditioned", session)][1]
        row = {"decoder": decoder, "variant": variant, "session": session,
               **compare_metrics(counts, values, baseline), **noise}
        rows.append(row)
        summaries.setdefault((decoder, variant), []).append((counts, values, baseline, noise))
    pooled = []
    for (decoder, variant), blocks in summaries.items():
        counts = {name: sum(block[0][name] for block in blocks) for name in blocks[0][0]}
        values = np.concatenate([block[1] for block in blocks])
        baseline = np.concatenate([block[2] for block in blocks])
        prior_variance = np.mean([block[3]["held_repeat_variance_before"] for block in blocks])
        reduced_variance = np.mean([block[3]["held_repeat_variance_after"] for block in blocks])
        pooled.append({"decoder": decoder, "variant": variant, **compare_metrics(counts, values, baseline),
                       "held_repeat_variance_before": float(prior_variance),
                       "held_repeat_variance_after": float(reduced_variance),
                       "held_repeat_variance_explained": float(1 - reduced_variance / max(prior_variance, 1e-8))})
    return {"early_window_ms_after_onset": [0, early_ms], "evoked_windows_ms": EVOKED_WINDOWS[view],
            "session_metrics": rows, "summary": pooled}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--early-windows", nargs="+", type=int, choices=(25, 50), default=(25,))
    parser.add_argument("--views", nargs="+", choices=tuple(EVOKED_WINDOWS), default=("total",))
    parser.add_argument("--ridge", type=float, default=0.45, help="relative nuisance-regression regularization")
    parser.add_argument("--compare-contour", action="store_true")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, help="change the held-repeat split with the same sampled OCF glyphs")
    parser.add_argument("--ocf-images", type=int, default=128, help="total paired OCF sample per mouse")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/early_noise_validation.json"))
    args = parser.parse_args()
    if args.ridge <= 0 or args.ocf_images < 4:
        parser.error("Use positive nuisance regularization and at least four OCF samples per mouse.")
    torch.set_num_threads(4)
    started = time.monotonic()
    chosen_device = args.device
    if chosen_device == "auto":
        chosen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(chosen_device)
    root = locate_data()
    catalog = load_catalog(root)
    scorer = official_scorer(root)
    repeat_seed = args.seed if args.repeat_seed is None else args.repeat_seed
    output = {"device": str(device), "seed": args.seed, "repeat_seed": repeat_seed, "ridge": args.ridge,
              "ocf_images_per_mouse_requested": args.ocf_images, "compare_contour": args.compare_contour,
              "variance_protocol": "held evoked repeat minus the mean of that glyph's remaining fit repeats",
              "mice": {}}
    for mouse in args.mice:
        images = load_images(root, device)
        masks = images >= 128
        counts = grouped_blocks(root, catalog, mouse, device, len(images))
        folds = {session: split_repeats(values, repeat_seed + index)
                 for index, (session, values) in enumerate(counts.items())}
        training = {session: pair[0] for session, pair in folds.items()}
        queries = {session: pair[1] for session, pair in folds.items()}
        sample_size = min(len(images), max(1, args.ocf_images // len(queries)))
        rng = np.random.default_rng(args.seed)
        selected = {name: torch.as_tensor(rng.choice(len(images), sample_size, replace=False), device=device)
                    for name in queries}
        contours = None
        if args.compare_contour:
            from glyph_mbr import contour_signatures

            contours = contour_signatures(images)
        records = []
        output["mice"][mouse] = records
        for view in args.views:
            for early_ms in args.early_windows:
                result = evaluate(training, queries, selected, masks, scorer, device, args, view, early_ms, contours)
                records.append(result)
                output["seconds"] = time.monotonic() - started
                args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({"mouse": mouse, "view": view, "early_ms": early_ms,
                                  "summary": result["summary"]}), flush=True)
        del counts, folds, training, queries, contours


if __name__ == "__main__":
    main()
