"""Validate unlabeled Task 2 covariance transport on held training sessions."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import feature_view, official_scorer, standardize


SESSIONS = ("session1", "session2", "session3", "session4")
CORAL = "https://doi.org/10.1609/AAAI.V30I1.10306"


@dataclass
class RepeatCovariances:
    halves: torch.Tensor
    repeat_noise: torch.Tensor
    baseline_noise: torch.Tensor


def covariance(values):
    centered = values - values.mean(dim=0)
    return centered.T @ centered / (len(centered) - 1)


def repeat_covariances(counts):
    """Estimate one-trial noise by differencing normalized two-repeat means."""
    values, _ = standardize(feature_view(counts, "evoked-total"))
    if values.shape[1] != 4:
        raise ValueError("The split-half covariance needs four repeats per query")
    a, b, c, d = values.unbind(dim=1)
    differences = ((a + b - c - d) / 2, (a + c - b - d) / 2, (a + d - b - c) / 2)
    noise = sum(covariance(difference) for difference in differences) / len(differences)
    halves = torch.stack(((a + b) / 2, (c + d) / 2))
    residual = (values - values.mean(dim=1, keepdim=True)).flatten(0, 1)
    # Match the original sqrt/z/LDA residual covariance normalization.
    return RepeatCovariances(halves, noise, residual.T @ residual / len(residual))


def clipped_signal(halves, mean_trial_noise):
    """Subtract noise in the pooled mean and report independent-half stability."""
    left, right = halves
    left_signal = covariance(left) - mean_trial_noise / 2
    right_signal = covariance(right) - mean_trial_noise / 2
    agreement = float((left_signal * right_signal).sum() /
                      (left_signal.norm() * right_signal.norm()).clamp_min(1e-10))
    signal = covariance(halves.mean(dim=0)) - mean_trial_noise / 4
    eigenvalues, basis = torch.linalg.eigh(signal)
    negative_fraction = float(eigenvalues.clamp_max(0).abs().sum() / eigenvalues.abs().sum().clamp_min(1e-10))
    positive = eigenvalues.clamp_min(0)
    effective_rank = float(positive.sum().square() / positive.square().sum().clamp_min(1e-10))
    stable = agreement >= 0.35 and negative_fraction <= 0.25 and effective_rank >= 2
    return (basis * positive) @ basis.T, {
        "half_covariance_agreement": agreement, "negative_eigenmass_fraction": negative_fraction,
        "effective_rank": effective_rank, "used": stable,
    }


def sym_power(matrix, power):
    spectrum, axes = torch.linalg.eigh(matrix)
    return (axes * spectrum.clamp_min(1e-8).pow(power)) @ axes.T


def transport_templates(templates, source, target, shrinkage, strength, maximum_change):
    """Move source templates a short distance toward target covariance."""
    axis_variance = 0.5 * (source.diagonal().mean() + target.diagonal().mean())
    identity = torch.eye(len(source), dtype=source.dtype, device=source.device)
    common = shrinkage * axis_variance.clamp_min(1e-6) * identity
    regularized_source = (1 - shrinkage) * source + common
    regularized_target = (1 - shrinkage) * target + common
    transport = sym_power(regularized_source, -0.5) @ sym_power(regularized_target, 0.5)
    change = strength * (transport - identity)
    operator_change = torch.linalg.matrix_norm(change, ord=2)
    change *= (maximum_change / operator_change.clamp_min(1e-10)).clamp_max(1)
    centered = templates - templates.mean(dim=0)
    delta = centered @ change
    stats = {"proposed_operator_change": float(operator_change),
             "operator_change": min(float(operator_change), maximum_change),
             "relative_template_rms_change": float(delta.norm() / centered.norm().clamp_min(1e-10))}
    return templates + delta, stats


def linalg_logits(query, templates, noise):
    regularized = noise.clone()
    regularized.diagonal().add_(0.35)
    weights = torch.cholesky_solve(templates.T, torch.linalg.cholesky(regularized)).T
    return query @ weights.T - 0.5 * (templates * weights).sum(dim=1)


def fold_logits(source, target, args):
    source_halves = torch.stack([block.halves for block in source]).mean(dim=0)
    source_noise = torch.stack([block.repeat_noise for block in source]).mean(dim=0)
    baseline_noise = torch.stack([block.baseline_noise for block in source]).mean(dim=0)
    templates, query = source_halves.mean(0), target.halves.mean(0)
    # Convert target one-trial noise to the legacy within-group covariance convention.
    target_precision_noise = baseline_noise.lerp(0.75 * target.repeat_noise, args.precision_weight)
    transported, noise_stats = transport_templates(templates, source_noise, target.repeat_noise,
                                                    args.shrinkage, args.strength, args.maximum_change)
    outputs = {
        "no_correction": linalg_logits(query, templates, baseline_noise),
        "target_precision_only": linalg_logits(query, templates, target_precision_noise),
        "noise_transport": linalg_logits(query, transported, baseline_noise),
        "noise_transport_and_target_precision": linalg_logits(query, transported, target_precision_noise),
    }
    source_signal, source_stats = clipped_signal(source_halves, source_noise / len(source))
    target_signal, target_stats = clipped_signal(target.halves, target.repeat_noise)
    stats = {"noise_transport": noise_stats, "source_signal": source_stats, "target_signal": target_stats}
    if source_stats["used"] and target_stats["used"]:
        transported_signal, signal_stats = transport_templates(templates, source_signal, target_signal,
                                                               args.shrinkage, args.strength, args.maximum_change)
        outputs["signal_transport"] = linalg_logits(query, transported_signal, baseline_noise)
        stats["signal_transport"] = signal_stats
    return outputs, stats


def evaluate_fold(logits, labels, selected, bank, score, device, cache, args):
    queries = labels.argsort()[selected]
    references = selected.cpu().numpy()
    fold, deltas, baseline_assignments = {}, {}, {}
    baseline_scores = {}
    for key, scores in logits.items():
        top = scores.topk(5, dim=1).indices
        row = {"map_top5": float((top == labels[:, None]).any(dim=1).float().mean()), "top_n": len(labels)}
        for rule, predicted in (("map", top[:, 0]), ("balanced", assign_glyphs(scores))):
            if key == "no_correction":
                baseline_assignments[rule] = predicted
            measured = {"top1": float((predicted == labels).float().mean()),
                        "changed_fraction": float((predicted != baseline_assignments[rule]).float().mean())}
            if score is not None and (rule == "balanced" or args.map_ocf):
                pairs = list(zip(references, predicted[queries].cpu().numpy()))
                missing = list(dict.fromkeys(pair for pair in pairs if pair not in cache))
                if missing:
                    names, choices = map(np.asarray, zip(*missing))
                    values = score(bank[choices], bank[names], device=str(device), batch_size=args.metric_batch)
                    cache.update(zip(missing, values.tolist()))
                values = np.asarray([cache[pair] for pair in pairs])
                if key == "no_correction":
                    baseline_scores[rule] = values
                delta = values - baseline_scores[rule]
                measured.update(ocf=float(values.mean()), ocf_n=len(values), paired_delta=float(delta.mean()))
                deltas[key, rule] = delta
            row[rule] = measured
        fold[key] = row
    return fold, deltas


def aggregate_rows(folds, deltas):
    result = {}
    for method in sorted({name for fold in folds for name in fold["rows"]}):
        measured = [fold["rows"][method] for fold in folds if method in fold["rows"]]
        result[method] = {"n_folds": len(measured), "map_top5": float(np.mean([row["map_top5"] for row in measured]))}
        for rule in ("map", "balanced"):
            records = [row[rule] for row in measured]
            record = {"top1": float(np.mean([row["top1"] for row in records]))}
            if (method, rule) in deltas:
                # Each fold evaluates the same selected glyphs. Average their paired deltas first.
                per_glyph = np.stack(deltas[method, rule]).mean(axis=0)
                n = len(per_glyph)
                record.update(ocf=float(np.mean([row["ocf"] for row in records])),
                              paired_delta=float(per_glyph.mean()), unique_ocf_glyphs=n,
                              paired_delta_se=float(per_glyph.std(ddof=1) / n**0.5) if n > 1 else None)
            result[method][rule] = record
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--held-sessions", nargs="+", choices=SESSIONS, default=SESSIONS)
    parser.add_argument("--ocf-images", type=int, default=24, help="Paired glyphs per fold; zero runs ranking only")
    parser.add_argument("--map-ocf", action="store_true", help="also score the independent MAP retrieval")
    parser.add_argument("--metric-batch", type=int, default=8)
    parser.add_argument("--shrinkage", type=float, default=0.65)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--maximum-change", type=float, default=0.15)
    parser.add_argument("--precision-weight", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/coral_validation.json"))
    args = parser.parse_args()
    if not (1 <= args.metric_batch and 0 <= args.ocf_images <= 2748 and 0 < args.shrinkage <= 1
            and 0 <= args.strength <= 1 and 0 <= args.maximum_change <= 0.25 and 0 <= args.precision_weight <= 1):
        parser.error("Choose valid metric counts, covariance shrinkage and a correction cap up to 0.25.")
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Use Kaggle CPU or CUDA for this experiment.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    bank = (load_images(root, torch.device("cpu")) >= 128).numpy()
    selected = np.random.default_rng(args.seed).choice(len(bank), args.ocf_images, replace=False)
    selected = torch.as_tensor(selected, device=device)
    rng = np.random.default_rng(args.seed + 1000)
    score = official_scorer(root) if len(selected) else None
    output = {"paper": CORAL, "device": str(device), "seed": args.seed, "ocf_images": len(selected),
              "covariance_shrinkage": args.shrinkage, "transport_strength": args.strength,
              "maximum_operator_change": args.maximum_change, "target_precision_weight": args.precision_weight,
              "protocol": "Leave-one-training-session-out Task 2 proxy. Randomized target query rows supply only unlabeled repeat moments. Labeled source repeats form glyph templates. Held labels are used for evaluation only; no competition test recordings are accessed.",
              "folds": []}
    cache, accumulated = {}, {}
    for mouse in args.mice:
        recordings = grouped_blocks(root, catalog, mouse, device, len(bank))
        fitted = {name: repeat_covariances(counts) for name, counts in recordings.items()}
        for session in args.held_sessions:
            fold_start = time.monotonic()
            labels = torch.as_tensor(rng.permutation(len(bank)), device=device)
            target = repeat_covariances(recordings[session][labels])
            source = [values for name, values in fitted.items() if name != session]
            logits, stability = fold_logits(source, target, args)
            rows, paired = evaluate_fold(logits, labels, selected, bank, score, device, cache, args)
            for key, delta in paired.items():
                accumulated.setdefault(key, []).append(delta)
            result = {"mouse": mouse, "session": session, "stability": stability, "rows": rows,
                      "seconds": time.monotonic() - fold_start}
            output["folds"].append(result)
            output["summary"] = aggregate_rows(output["folds"], accumulated)
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result), flush=True)
        del recordings, fitted
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
