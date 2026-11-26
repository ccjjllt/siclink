"""Measure a trial-count-matched recording gap and its paired calibration headroom."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_diagnostics import feature_view, standardize


CALIBRATION = 512
DIMENSION = 48
ROTATION_RANK = 16
ROTATION_PRIOR = 0.75
PARTITIONS = 2
SESSIONS = ("session1", "session2", "session3", "session4")


def responses_in_source_space(recordings, query_day, fit):
    """Fit one coordinate system on source days, with label-free day means."""
    sources = [values[fit] for day, values in recordings.items() if day != query_day]
    stats = [standardize(values)[1] for values in sources]
    normalization = tuple(torch.stack(items).mean(0) for items in zip(*stats))
    centered = {}
    for day, values in recordings.items():
        normalized, _ = standardize(values, normalization)
        centered[day] = normalized - normalized[fit].mean(dim=(0, 1))
    fit_sources = torch.stack([values[fit] for day, values in centered.items() if day != query_day])
    templates = fit_sources.mean(dim=2)
    residual = (fit_sources - templates[:, :, None]).flatten(0, 2)
    noise = residual.T @ residual / (len(templates) * len(fit) * (fit_sources.shape[2] - 1))
    noise.diagonal().add_(0.20 * noise.diagonal().mean().clamp_min(0.02))
    whitener = torch.linalg.cholesky(noise)
    flat = templates.flatten(0, 1)
    matrices = torch.linalg.solve_triangular(whitener, flat.T.contiguous(), upper=False).T.reshape_as(templates)
    pooled = matrices.sum(dim=0)
    stacked = matrices.flatten(0, 1)
    stable = pooled.T @ pooled - stacked.T @ stacked
    _, axes = torch.linalg.eigh(0.5 * (stable + stable.T))
    axes = axes[:, -DIMENSION:].flip(1).contiguous()
    project = torch.linalg.solve_triangular(whitener.T, axes, upper=True)
    return {day: values @ project for day, values in centered.items()}


def paired_rotation(source, target):
    """On the leading coordinates, fit a norm-preserving map near identity."""
    source, target = source[:, :ROTATION_RANK], target[:, :ROTATION_RANK]
    cross = source.T @ target / len(source)
    variance = 0.5 * (source.square().mean() + target.square().mean())
    cross.diagonal().add_(ROTATION_PRIOR * variance)
    u, _, vt = torch.linalg.svd(cross, full_matrices=False)
    # Restrict to a proper rotation, including under the randomly paired control.
    u[:, -1] *= torch.linalg.det(u @ vt).sign()
    return u @ vt


def rotated(values, rotation):
    leading = values[:, :ROTATION_RANK] @ rotation
    return torch.cat((leading, values[:, ROTATION_RANK:]), dim=1)


def match(query, templates, index, truth):
    """Match anonymous rows in the same complete evaluation gallery."""
    score = query @ F.normalize(templates[index], dim=1).T
    winners = score.topk(32, dim=1).indices
    hit = (winners == truth[:, None])
    positive = score.gather(1, truth[:, None])[:, 0]
    negative = (score.sum(1) - positive) / (score.shape[1] - 1)
    return {"top1": hit[:, 0].float().numpy(), "top32": hit.any(1).float().numpy(),
            "true_minus_mean_false_cosine": (positive - negative).numpy()}


def moment_summary(values):
    return {"mean": float(values.mean()), "glyph_cluster_se": float(values.std(ddof=1) / len(values)**0.5)}


def average(parts):
    return {field: np.stack([row[field] for row in parts]).mean(0) for field in parts[0]}


def comparison_summary(arms, moments):
    result = {name: {key: moment_summary(values) for key, values in arm.items()} for name, arm in arms.items()}
    contrasts = (("same_minus_cross", "same", "cross"), ("paired_minus_cross", "paired", "cross"),
                 ("shuffled_minus_cross", "shuffled", "cross"), ("paired_minus_shuffled", "paired", "shuffled"))
    result["contrasts"] = {name: {key: moment_summary(arms[left][key] - arms[right][key]) for key in arms[left]}
                           for name, left, right in contrasts}
    result["neural_moments"] = {key: moment_summary(values) for key, values in moments.items()}
    sampling = moments["two_repeat_query_noise"].mean() + moments["two_repeat_source_noise"].mean()
    result["neural_moments"]["drift_over_total_two_repeat_noise"] = float(moments["deconvolved_drift"].mean() / sampling)
    result["n_evaluation_glyphs"] = len(next(iter(moments.values())))
    return result


def blocked_halves(values, selection):
    order = selection[..., None].expand(-1, -1, values.shape[-1])
    arranged = values.gather(1, order)
    return arranged[:, :2].mean(1), arranged[:, 2:].mean(1)


def compare_recording(recordings, query_day, fit, held, permutations, rng):
    projected = responses_in_source_space(recordings, query_day, fit)
    query_means = projected[query_day][fit].mean(1)
    sources = [day for day in recordings if day != query_day]
    calibration_shuffle = torch.as_tensor(rng.permutation(len(fit)))
    rotations = {}
    for day in sources:
        fit_means = projected[day][fit].mean(1)
        rotations[day] = {"paired": paired_rotation(fit_means, query_means),
                          "shuffled": paired_rotation(fit_means, query_means[calibration_shuffle])}
    by_source = {day: [] for day in sources}
    arm_rows = {name: [] for name in ("same", "cross", "paired", "shuffled")}
    all_moments = []
    for partition in permutations:
        halves = {day: blocked_halves(values[held], partition[day]) for day, values in projected.items()}
        query_a, query_b = halves[query_day]
        row_order = torch.as_tensor(rng.permutation(len(held)))
        gallery_order = torch.as_tensor(rng.permutation(len(held)))
        row_inverse = row_order.argsort().numpy()
        outcome = gallery_order.argsort()[row_order]
        query = F.normalize(query_a[row_order], dim=1)
        same = match(query, query_b, gallery_order, outcome)
        same = {name: values[row_inverse] for name, values in same.items()}
        query_noise = 0.5 * (query_a - query_b).square().sum(1).numpy()
        for day in sources:
            source_a, source_b = halves[day]
            arms = {"same": same}
            candidates = {"cross": source_b, **{name: rotated(source_b, rotation)
                                                for name, rotation in rotations[day].items()}}
            for name, template in candidates.items():
                measured = match(query, template, gallery_order, outcome)
                arms[name] = {field: values[row_inverse] for field, values in measured.items()}

            # Labels enter these matched moments only as a retrospective diagnostic.
            source_noise = 0.5 * (source_a - source_b).square().sum(1)
            drift = ((query_a - source_a) * (query_b - source_b)).sum(1)
            moments = {"two_repeat_query_noise": query_noise, "two_repeat_source_noise": source_noise.numpy(),
                       "deconvolved_drift": drift.numpy()}
            by_source[day].append((arms, moments))
            all_moments.append(moments)
            for name, values in arms.items():
                arm_rows[name].append(values)

    combined = {name: average(values) for name, values in arm_rows.items()}
    moments = average(all_moments)
    summary = comparison_summary(combined, moments)
    pairs = {}
    for day, parts in by_source.items():
        pair_arms = {name: average([part[0][name] for part in parts]) for name in combined}
        pairs[day] = comparison_summary(pair_arms, average([part[1] for part in parts]))
        pairs[day]["rotation_rms_from_identity"] = {name: float((r - torch.eye(len(r))).norm() / len(r)**0.5)
                                                    for name, r in rotations[day].items()}
    return summary, pairs, combined, moments


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7313)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_repeat_day_gap.json"))
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    root = locate_data()
    catalog = load_catalog(root)
    n_glyphs = len(load_images(root, "cpu"))
    if n_glyphs != 2748:
        raise ValueError("Expect the 2,748 labeled gallery glyphs in the official training data.")
    rng = np.random.default_rng(args.seed)
    glyphs = torch.as_tensor(rng.permutation(n_glyphs))
    fit, held = glyphs[:CALIBRATION], glyphs[CALIBRATION:]
    output = {"seed": args.seed, "calibration_glyphs": len(fit), "evaluation_glyphs": len(held),
              "repeat_partitions": PARTITIONS, "feature_window_ms": [50, 250], "source_dimensions": DIMENSION,
              "rotation_rank": ROTATION_RANK, "rotation_prior": ROTATION_PRIOR,
              "protocol": "For every ordered cross-recording pair, the no-adaptation, same-recording and "
                          "calibrated arms use the identical query half, held glyphs, common source whitening "
                          "and cosine scorer. Fit coordinates only on 512 non-evaluated glyphs. Anonymous day "
                          "centering uses the same calibration rows. Diagnostic paired rotation uses the "
                          "target day's known glyph labels on those rows; the shuffled arm permutes those "
                          "labels. Retrospective score and drift labels never enter fitting. Glyph-cluster "
                          "SE averages repeats, partners and days within each glyph before inference. "
                          "Only labeled training recordings are read. No OCF, formal test block or sample "
                          "ID is used for fitting or matching.", "mice": {}}

    for mouse in ("mice1", "mice2"):
        mouse_start = time.monotonic()
        recordings = {day: feature_view(values, "evoked-total") for day, values in
                      grouped_blocks(root, catalog, mouse, "cpu", n_glyphs).items()}
        if tuple(recordings) != SESSIONS or any(values.shape[:2] != (n_glyphs, 4) for values in recordings.values()):
            raise ValueError("Each mouse requires four labeled recordings with four repeats per glyph.")
        permutations = [{day: torch.as_tensor(rng.random((len(held), 4)).argsort(1)) for day in SESSIONS}
                        for _ in range(PARTITIONS)]
        day_reports, arm_parts, moment_parts = {}, [], []
        for day in SESSIONS:
            summary, pairs, arms, moments = compare_recording(recordings, day, fit, held, permutations, rng)
            day_reports[day] = {"summary": summary, "by_source_session": pairs}
            arm_parts.append(arms)
            moment_parts.append(moments)
            print(json.dumps({"mouse": mouse, "query_day": day, "summary": summary}), flush=True)
        averaged = {name: average([item[name] for item in arm_parts]) for name in arm_parts[0]}
        summary = comparison_summary(averaged, average(moment_parts))
        output["mice"][mouse] = {"summary": summary, "recordings": day_reports,
                                 "seconds": time.monotonic() - mouse_start}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
