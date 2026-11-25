"""Validate a hierarchical gallery on complete Task 2 recording holdouts."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import EB_BLEND_WEIGHT, VIEWS, blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, combine_logits, fit_retrieval, retrieval_score_components
from kaggle_diagnostics import official_scorer


def validated_logits(mouse, training, query, images, device, args):
    """Re-fit the current per-mouse decision policy on the identical training fold."""
    if mouse == "mice1":
        from glyph_encoding import count_template_logits, fit_glyph_encoder, make_count_templates

        encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
        rates = make_count_templates(encoder, images, training).for_recording(None, query)["shrinkage"]
        return count_template_logits(query, rates, "rate")
    model = fit_retrieval(as_trial_blocks(training), len(images), device=device, epochs=args.retrieval_epochs,
                          images=images, seed=args.seed)
    return combine_logits(retrieval_score_components(model, query, None, task="task2"), "task2")


def paired_ocf(score, chosen, sample_rows, masks, selected, device):
    chosen_masks = masks[chosen[sample_rows]].float().cpu().numpy()
    references = masks[selected].cpu().numpy()
    return np.asarray(score(chosen_masks, references, device=str(device), batch_size=4))


def paired_delta(values, baseline):
    difference = values - baseline
    error = float(difference.std(ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else None
    return float(difference.mean()), error


def evaluate(logits, permutation, selected, masks, score, device, *, balanced_only=False):
    sample_rows = permutation.argsort()[selected]
    rows, measurements = [], {}
    for name, matrix in logits.items():
        leading = matrix.topk(min(5, len(masks)), dim=1).indices
        top5 = float((leading == permutation[:, None]).any(dim=1).float().mean())
        decisions = {} if balanced_only else {"map": leading[:, 0]}
        decisions["one_to_one"] = assign_glyphs(matrix)
        for rule, choices in decisions.items():
            values = paired_ocf(score, choices, sample_rows, masks, selected, device)
            rows.append({"model": name, "rule": rule, "ocf": float(values.mean()), "ocf_n": len(values),
                         "top1": float((choices == permutation).float().mean()),
                         "map_top5": top5, "top_n": len(permutation)})
            measurements[(name, rule)] = values
    for row in rows:
        name, rule = row["model"], row["rule"]
        chosen = measurements[name, rule]
        delta, error = paired_delta(chosen, measurements["constant", rule])
        row.update(paired_delta_vs_constant=delta, paired_se_vs_constant=error)
        if "validated" in logits:
            delta, error = paired_delta(chosen, measurements["validated", rule])
            row.update(paired_delta_vs_validated=delta, paired_se_vs_validated=error)
    return rows, measurements


def aggregate(measurements):
    rows = []
    for (view, name, rule), blocks in measurements.items():
        values = np.stack(blocks)
        baseline = np.stack(measurements[view, "constant", rule])
        delta, error = paired_delta(values.mean(axis=0), baseline.mean(axis=0))
        row = {"view": view, "model": name, "rule": rule, "ocf": float(values.mean()),
               "ocf_n": values.size, "unique_scored_glyphs": values.shape[1],
               "paired_delta_vs_constant": delta, "paired_se_vs_constant": error}
        if (view, "validated", rule) in measurements:
            validated = np.stack(measurements[view, "validated", rule])
            delta, error = paired_delta(values.mean(axis=0), validated.mean(axis=0))
            row.update(paired_delta_vs_validated=delta, paired_se_vs_validated=error)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session2", "session4"))
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--views", nargs="+", choices=tuple(VIEWS), default=("total",))
    parser.add_argument("--ocf-images", type=int, default=64, help="same scored glyph sample in each recording")
    parser.add_argument("--query-repeat-mix", type=float, default=0.5,
                        help="optional noise-only update using the within-sample repeats of the query recording")
    parser.add_argument("--compare-validated", action="store_true")
    parser.add_argument("--blend-validated", action="store_true",
                        help="also score a fixed 35%% empirical-Bayes blend with the validated policy")
    parser.add_argument("--balanced-only", action="store_true", help="skip independent MAP OCF scoring")
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--retrieval-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/hierarchical_gallery_validation.json"))
    args = parser.parse_args()
    if args.ocf_images < 2 or not 0 <= args.query_repeat_mix <= 1 or min(args.encoder_epochs, args.retrieval_epochs) < 1:
        parser.error("Provide valid scoring, repeat covariance and comparison settings.")
    chosen_device = args.device
    if chosen_device == "auto":
        chosen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(chosen_device)
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    score = official_scorer(root)
    images = load_images(root, device)
    masks = images >= 128
    selected = np.random.default_rng(args.seed).choice(len(images), min(len(images), args.ocf_images), replace=False)
    selected = torch.as_tensor(selected, device=device)
    rng = np.random.default_rng(args.seed + 1000)
    compare_validated = args.compare_validated or args.blend_validated
    blend_source = "empirical_bayes_repeat_noise" if args.query_repeat_mix else "empirical_bayes"
    output = {"device": str(device), "seed": args.seed, "ocf_images_per_fold": len(selected),
              "query_repeat_mix": args.query_repeat_mix, "compare_validated": compare_validated,
              "training_normalization": "session-wise fit moments; query uses the pooled fit moments",
              "session_protocol": "held training recordings proxy the independent-day session5 test",
              "mice": {}}
    if args.blend_validated:
        output["blend"] = {"source": blend_source, "eb_weight": EB_BLEND_WEIGHT,
                           "normalization": "global RMS after subtracting each query's mean score"}
    if args.balanced_only:
        output["balanced_only"] = True
    aggregate_scores = {}
    for mouse in args.mice:
        grouped = grouped_blocks(root, catalog, mouse, device, len(images))
        output["mice"][mouse] = {}
        for session in args.held_sessions:
            fold_started = time.monotonic()
            training = {name: raw for name, raw in grouped.items() if name != session}
            permutation = torch.as_tensor(rng.permutation(len(images)), device=device)
            query = grouped[session][permutation]
            validated = None
            if compare_validated:
                validated = validated_logits(mouse, training, query, images, device, args)
            fold = {"training_sessions": list(training), "views": {}}
            output["mice"][mouse][session] = fold
            for view in args.views:
                model = fit_hierarchical_gallery(training, view=view)
                logits = model.logits(query, query_repeat_mix=args.query_repeat_mix)
                if validated is not None:
                    logits["validated"] = validated
                    if args.blend_validated:
                        logits["validated_eb_blend"] = blend_hierarchical_logits(validated, logits[blend_source])
                rows, measured = evaluate(logits, permutation, selected, masks, score, device,
                                          balanced_only=args.balanced_only)
                fold["views"][view] = {"windows_ms": VIEWS[view], "fit": model.diagnostics, "metrics": rows}
                for (name, rule), values in measured.items():
                    aggregate_scores.setdefault((view, name, rule), []).append(values)
                fold["seconds"] = time.monotonic() - fold_started
                output["summary"] = aggregate(aggregate_scores)
                output["seconds"] = time.monotonic() - started
                print(json.dumps({"mouse": mouse, "session": session, "view": view,
                                  "fit": model.diagnostics, "metrics": rows}), flush=True)
                args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
                del logits
            del validated
        del grouped


if __name__ == "__main__":
    main()
