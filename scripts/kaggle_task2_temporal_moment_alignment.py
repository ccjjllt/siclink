"""Identify a cross-day frame from several unlabeled repeat cross-moments."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_hierarchical_gallery_validation import validated_logits
from kaggle_task3_metric_validation import unique_pair_ocf


SESSIONS = ("session1", "session2", "session3", "session4")
WINDOWS = ((50, 250), (50, 100), (100, 150), (150, 250))
SCORE_DIM = 48
MAP_DIM = 12
MAP_ROTATION_CAP = 0.25
CALIBRATION_GLYPHS = 512
FUSION = 0.20
ARMS = ("frozen", "coral", "multi_moment", "mismatched_windows", "labeled_oracle")


def sym(matrix):
    return 0.5 * (matrix + matrix.mT)


def views(counts):
    counts = counts.unflatten(-1, (-1, 10))
    return torch.stack([(counts[..., left // 25:right // 25].sum(dim=-1) + 0.375).sqrt()
                        for left, right in WINDOWS], dim=-2)


def fit_space(source, query):
    days = [views(values) for values in source]
    totals = torch.stack([day[:, :, 0] for day in days])
    repeat_error = (totals - totals.mean(dim=2, keepdim=True)).flatten(0, 2)
    within = repeat_error.T @ repeat_error / (len(source) * totals.shape[1] * (totals.shape[2] - 1))
    within.diagonal().add_(0.25 * within.diagonal().mean().clamp_min(0.02))
    lower = torch.linalg.cholesky(sym(within))

    centers = totals.mean(dim=2)
    centers = centers - centers.mean(dim=1, keepdim=True)
    flat = centers.flatten(0, 1)
    whitened = torch.linalg.solve_triangular(lower, flat.T.contiguous(), upper=False).T.reshape_as(centers)
    pooled = whitened.sum(dim=0)
    stability = pooled.T @ pooled - whitened.flatten(0, 1).T @ whitened.flatten(0, 1)
    _, axes = torch.linalg.eigh(sym(stability))
    projection = torch.linalg.solve_triangular(lower.T, axes[:, -SCORE_DIM:].flip(1), upper=True)

    def project(block):
        return (block - block.mean(dim=(0, 1), keepdim=True)) @ projection

    return [project(day) for day in days], project(views(query))


def repeat_operators(values):
    """Covariances of stimulus means estimated with different repeat indices."""
    glyphs, repeats, windows, units = values.shape
    values = values - values.mean(dim=0, keepdim=True)
    flat = values.flatten(-2)
    pooled = flat.sum(dim=1)
    individual = flat.flatten(0, 1)
    cross = (pooled.T @ pooled - individual.T @ individual) / (glyphs * repeats * (repeats - 1))
    return cross.reshape(windows, units, windows, units).permute(0, 2, 1, 3).contiguous()


def powers(matrix):
    dim = len(matrix)
    spectrum, axes = torch.linalg.eigh(sym(matrix))
    stable = spectrum.clamp_min(0)
    stable = 0.85 * stable + 0.15 * stable.mean().clamp_min(0.02)
    return (axes * stable.sqrt()) @ axes.T, (axes * stable.rsqrt()) @ axes.T


def whitened_operators(estimate, whitening):
    indices = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))
    raw = torch.stack([estimate[i, j] for i, j in indices])
    return whitening @ raw @ whitening


def rotate_to_match(source, target):
    """Fit a regularized, bounded orthogonal correction in whitened coordinates."""
    n = source.shape[-1]
    first, second = torch.triu_indices(n, n, offset=1)
    tangent = source.new_zeros((len(first), n, n))
    rows = torch.arange(len(first))
    tangent[rows, first, second], tangent[rows, second, first] = 1.0, -1.0
    weight = source.flatten(1).norm(dim=1).clamp_min(0.15).reciprocal()
    target = target * weight[:, None, None]
    source = source * weight[:, None, None]
    derivatives = torch.einsum("mab,kbc->mkac", target, tangent) - torch.einsum("kab,mbc->mkac", tangent, target)
    jacobian = derivatives.transpose(0, 1).flatten(1).T.contiguous()
    normal = jacobian.T @ jacobian
    spectrum = torch.linalg.eigvalsh(sym(normal)).clamp_min(0)
    regularization = 0.30 * normal.diagonal().mean().clamp_min(1e-5)
    normal.diagonal().add_(regularization)
    coefficients = torch.linalg.solve(normal, jacobian.T @ (source - target).flatten())
    correction = torch.einsum("k,kab->ab", coefficients, tangent)
    amplitude = torch.linalg.matrix_norm(correction, ord=2)
    correction *= (MAP_ROTATION_CAP / amplitude.clamp_min(1e-6)).clamp_max(1)
    rotation = torch.matrix_exp(correction)
    return rotation, {"unbounded_rotation_operator_norm": float(amplitude),
                      "relative_smallest_angular_information": float(spectrum[0] / spectrum[-1].clamp_min(1e-8)),
                      "informative_angle_fraction": float((spectrum > 0.01 * spectrum[-1]).float().mean())}


def moment_error(source, observed, rotation):
    discrepancy = source - rotation.T @ observed @ rotation
    scale = source.flatten(1).norm(dim=1).clamp_min(0.15)
    return float((discrepancy.flatten(1).norm(dim=1) / scale).square().mean().sqrt())


def oracle_rotation(query_half, template, order, calibration, inverse_target, inverse_source):
    """Fit a supervised rotation as a reference on a disjoint evaluation set."""
    rows = order.argsort()[calibration]
    q = query_half[rows, :, 0, :MAP_DIM].mean(dim=1) @ inverse_target
    t = template[calibration, :MAP_DIM] @ inverse_source
    cross = q.T @ t / len(q)
    cross.diagonal().add_(0.25 * cross.abs().diagonal().mean().clamp_min(0.10))
    left, _, right = torch.linalg.svd(cross)
    left[:, -1] *= torch.linalg.det(left @ right).sign()
    return left @ right


def logits(query, templates, lower):
    q = torch.linalg.solve_triangular(lower, query.T.contiguous(), upper=False).T
    t = torch.linalg.solve_triangular(lower, templates.T.contiguous(), upper=False).T
    return q @ t.T - 0.5 * t.square().sum(dim=1)


def normalized(score):
    centered = score - score.mean(dim=1, keepdim=True)
    return centered / centered.square().mean().sqrt().clamp_min(1e-6)


def retrieval_metrics(scores, order, evaluation):
    rows = order.argsort()[evaluation]
    scores = scores[rows]
    true = scores.gather(1, evaluation[:, None])[:, 0]
    hit = scores.topk(32, dim=1).indices == evaluation[:, None]
    nll = torch.logsumexp(3 * scores, dim=1) - 3 * true
    return {"nll": nll, "top32": hit.any(dim=1).float(), "top1": hit[:, 0].float()}


def mean_and_se(values):
    return {"mean": float(values.mean()), "glyph_se": float(values.std(unbiased=True) / len(values)**0.5)}


def compare_identity(metrics):
    return {"per_arm": {name: {key: float(value.mean()) for key, value in parts.items()}
                        for name, parts in metrics.items()},
            "multi_moment_minus": {name: {key: mean_and_se(metrics["multi_moment"][key] - measured)
                                            for key, measured in parts.items()}
                                     for name, parts in metrics.items() if name != "multi_moment"}}


@dataclass
class Diagnostic:
    order: torch.Tensor
    scores: dict
    report: dict


@torch.no_grad()
def evaluate(day, blocks, calibration, evaluation, rng):
    begin = time.monotonic()
    source = [counts for other, counts in blocks.items() if other != day]
    order = torch.as_tensor(rng.permutation(len(blocks[day])))
    query = blocks[day][order]
    perm = torch.from_numpy(rng.random(query.shape[:2]).argsort(axis=1))
    query = query.gather(1, perm[..., None].expand_as(query))
    source, anonymous = fit_space(source, query)
    day_templates = torch.stack([values[:, :, 0].mean(dim=1) for values in source])
    template = day_templates.mean(dim=0)
    changes = (day_templates - template).flatten(0, 1)
    uncertainty = changes.T @ changes / ((len(source) - 1) * len(template))
    uncertainty.diagonal().add_(0.30 * uncertainty.diagonal().mean().clamp_min(0.02))
    factor = torch.linalg.cholesky(sym(uncertainty))
    reference = torch.stack([repeat_operators(values[..., :MAP_DIM]) for values in source]).mean(dim=0)
    color, inverse_source = powers(reference[0, 0])
    source_ops = whitened_operators(reference, inverse_source)
    identity = torch.eye(MAP_DIM)
    sums = {name: None for name in ARMS}
    geometry = []
    for fitting, scoring in ((anonymous[:, :2], anonymous[:, 2:]), (anonymous[:, 2:], anonymous[:, :2])):
        estimate = repeat_operators(fitting[..., :MAP_DIM])
        held = repeat_operators(scoring[..., :MAP_DIM])
        _, inverse_target = powers(estimate[0, 0])
        target_ops = whitened_operators(estimate, inverse_target)
        opposite = torch.tensor([0, 3, 2, 1])
        wrong = whitened_operators(estimate[opposite][:, opposite], inverse_target)
        joint, angular = rotate_to_match(source_ops, target_ops)
        shuffled, _ = rotate_to_match(source_ops, wrong)
        oracle = oracle_rotation(fitting, template, order, calibration, inverse_target, inverse_source)
        rotations = {"coral": identity, "multi_moment": joint, "mismatched_windows": shuffled, "labeled_oracle": oracle}
        held_ops = whitened_operators(held, inverse_target)
        report = {**angular, "same_axes_held_moment_error": {name: moment_error(source_ops, held_ops, rotation)
                                                             for name, rotation in rotations.items()},
                  "rotation_operator_change": {name: float(torch.linalg.matrix_norm(rotation - identity, ord=2))
                                               for name, rotation in rotations.items()}}
        observed = scoring[:, :, 0].mean(dim=1)
        scores = {"frozen": logits(observed, template, factor)}
        for name, rotation in rotations.items():
            predicted = observed.clone()
            predicted[:, :MAP_DIM] = observed[:, :MAP_DIM] @ inverse_target @ rotation @ color
            scores[name] = logits(predicted, template, factor)
        for name, value in scores.items():
            value = normalized(value) * 0.5
            sums[name] = value if sums[name] is None else sums[name] + value
        geometry.append(report)
    metrics = {name: retrieval_metrics(value, order, evaluation) for name, value in sums.items()}
    reports = compare_identity(metrics)
    comparisons = [reports["multi_moment_minus"][tag] for tag in ("coral", "mismatched_windows")]
    repeat_geometry_gain = all(item["same_axes_held_moment_error"]["multi_moment"]
                               < item["same_axes_held_moment_error"]["coral"] for item in geometry)
    passes = bool(repeat_geometry_gain and all(row["nll"]["mean"] < 0 and row["top32"]["mean"] > 0 for row in comparisons))
    report = {"source_days": [name for name in blocks if name != day], "paired_identity_proxies": reports,
              "split_geometries": geometry, "passed_neural_diagnostic": passes, "seconds": time.monotonic() - begin}
    del sums["labeled_oracle"]
    return Diagnostic(order, sums, report)


def exact_evaluation(day, diagnostic, blocks, images, sample, scorer, args):
    sources = {name: block for name, block in blocks.items() if name != day}
    query = blocks[day][diagnostic.order]
    torch.manual_seed(args.seed)
    validated = validated_logits(args.mouse, sources, query, images, images.device, args)
    eb = fit_hierarchical_gallery(sources, view="total").logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    base = normalized(blend_hierarchical_logits(validated, eb))
    matrices = {"v9": base}
    matrices.update({name: base + FUSION * (value - diagnostic.scores["frozen"])
                     for name, value in diagnostic.scores.items() if name != "frozen"})
    choices = {name: assign_glyphs(scores) for name, scores in matrices.items()}
    subset = diagnostic.order.argsort()[sample]
    guesses = torch.stack([indices[subset] for indices in choices.values()])
    exact, pair_count = unique_pair_ocf(guesses, sample.expand_as(guesses), images >= 128, scorer, images.device, 384)
    measured = dict(zip(choices, exact))
    packed = {name: {"ocf": float(value.mean()), "vs_v9": mean_and_se(value - measured["v9"]),
                     "changes_vs_v9": int((choices[name] != choices["v9"]).sum()),
                     "distinct_assignments": len(choices[name].unique())} for name, value in measured.items()}
    return {"paired": packed, "exact_pairs": pair_count}, measured


def save(path, report, began):
    report["seconds"] = time.monotonic() - began
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", choices=("mice1", "mice2"), default="mice1")
    parser.add_argument("--held-days", nargs=2, choices=SESSIONS, default=("session1", "session3"))
    parser.add_argument("--split-seed", type=int, default=35797)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--ocf-glyphs", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_temporal_moment_alignment.json"))
    parser.set_defaults(encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if len(set(args.held_days)) != 2 or args.ocf_glyphs < 2:
        parser.error("Choose two distinct days and a paired OCF sample.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle CPU with official training data mounted.")
    torch.set_num_threads(4)
    start = time.monotonic()
    root = locate_data()
    images = load_images(root, torch.device("cpu"))
    blocks = grouped_blocks(root, load_catalog(root), args.mouse, images.device, len(images))
    if tuple(blocks) != SESSIONS or any(counts.shape[1] != 4 for counts in blocks.values()):
        raise ValueError("Need four training sessions with four measured repeats per glyph.")
    sequence = np.random.default_rng(args.split_seed).permutation(len(images))
    calibration = torch.from_numpy(sequence[:CALIBRATION_GLYPHS])
    evaluation = torch.from_numpy(sequence[CALIBRATION_GLYPHS:])
    sample = evaluation[:args.ocf_glyphs]
    report = {"mouse": args.mouse, "held_days": args.held_days, "seed": args.seed, "split_seed": args.split_seed,
              "windows_ms": WINDOWS, "score_dimension": SCORE_DIM, "adapted_dimension": MAP_DIM,
              "orientation_cap": MAP_ROTATION_CAP, "calibration_glyphs_for_labeled_oracle_only": len(calibration),
              "untouched_neural_metric_glyphs": len(evaluation), "v9_score_correction_weight": FUSION,
              "method": "The allowed mapping uses only holdout repeat count moments, with one 2-repeat half to fit "
                        "and the other to measure response geometry and decode. The reverse direction averages in. "
                        "Both CORAL and joint alignment share zero-order whitening and recoloring. Only the joint "
                        "rotation uses multiple time operators. The control reverses target time-operator identities. "
                        "The separate labeled rotation is a supervised reference, not a mathematical upper bound. "
                        "It never contributes to blind adaptation, OCF, or candidate scores. "
                        "Only if both outer days improve held-half moments against CORAL and identity vs both alignment "
                        "controls, compute paired exact OCF on complete one-to-one v9 assignments. For mice1 this is "
                        "the matched v9 neural reference. The current exact-OCF reranker and official test are outside "
                        "this experiment.",
              "related_method": "https://doi.org/10.1109/78.554307", "days": {}}
    results = {}
    for day in args.held_days:
        rng = np.random.default_rng(np.random.SeedSequence((args.split_seed, SESSIONS.index(day))))
        result = evaluate(day, blocks, calibration, evaluation, rng)
        report["days"][day] = result.report
        results[day] = result
        print(json.dumps({"held_day": day, **result.report}), flush=True)
        save(args.output, report, start)
    if all(result.report["passed_neural_diagnostic"] for result in results.values()):
        scorer = official_scorer(root)
        exact_trials = []
        for day, result in results.items():
            exact, measured = exact_evaluation(day, result, blocks, images, sample, scorer, args)
            report["days"][day]["official_ocf"] = exact
            exact_trials.append(measured)
            save(args.output, report, start)
        report["official_ocf_summary"] = {name: mean_and_se(torch.stack([day[name] - day["v9"]
                                                                       for day in exact_trials]).mean(dim=0))
                                          for name in exact_trials[0]}
    report["exact_ocf_performed"] = "official_ocf_summary" in report
    save(args.output, report, start)


if __name__ == "__main__":
    main()
