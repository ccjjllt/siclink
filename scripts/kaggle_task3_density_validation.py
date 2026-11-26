"""Measure how much spike-derived glyph density can add to a fixed Task 3 shortlist."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_contour_retrieval import _fit_readout, _response_features, contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_mbr import contour_signatures
from glyph_retrieval import _moments, _standardize
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import unique_pair_ocf


BASELINE = "contour_only"


def image_descriptors(masks):
    """Log ink fraction and horizontal/vertical mass in four strips each."""
    tiles = masks.reshape(-1, 4, 16, 4, 16).sum(dim=(2, 4)).float()
    area = tiles.sum(dim=(1, 2)).clamp_min(1)
    mass = torch.cat((tiles.sum(dim=1), tiles.sum(dim=2)), dim=1) / area[:, None]
    return torch.cat(((area / 4096).log()[:, None], mass), dim=1)


def response_views(contour, groups, session):
    raw = _response_features(groups, contour.view, contour.n_bins)
    projected = _standardize(raw, *contour.moments[session]) @ contour.projection
    total_count = groups.unflatten(-1, (-1, contour.n_bins))[..., 2:].sum(dim=(-1, -2)).sqrt()
    return projected, total_count.unsqueeze(-1)


def fit_descriptors(contour, training, descriptors):
    views = {session: response_views(contour, counts, session) for session, counts in training.items()}
    count_moments = {session: _moments(count) for session, (_, count) in views.items()}
    count_views = tuple(_standardize(count, *count_moments[session]) for session, (_, count) in views.items())
    full_views = tuple(projected for projected, _ in views.values())
    return (_fit_readout(full_views, descriptors, ridge=0.09),
            _fit_readout(count_views, descriptors, ridge=0.09), count_moments)


def descriptor_reliability(prediction, count_prediction, truth):
    error = (prediction - truth).square().mean(dim=0)
    count_error = (count_prediction - truth).square().mean(dim=0)
    reference = truth.var(dim=0, unbiased=False).clamp_min(1e-8)
    correlation = torch.corrcoef(torch.stack((prediction[:, 0], count_prediction[:, 0], truth[:, 0])))
    return {"rows": len(truth), "log_ink_pearson": float(correlation[0, 2]),
            "count_only_log_ink_pearson": float(correlation[1, 2]),
            "ink_fraction_rmse_percentage_points": float(100 * (prediction[:, 0].exp() - truth[:, 0].exp()).square().mean().sqrt()),
            "count_only_ink_fraction_rmse_percentage_points": float(100 * (count_prediction[:, 0].exp() - truth[:, 0].exp()).square().mean().sqrt()),
            "log_ink_r2": float(1 - error[0] / reference[0]),
            "count_only_log_ink_r2": float(1 - count_error[0] / reference[0]),
            "marginal_r2": float(1 - error[1:].sum() / reference[1:].sum()),
            "count_only_marginal_r2": float(1 - count_error[1:].sum() / reference[1:].sum())}


def density_compatibility(candidates, query, scale):
    # A fixed scale derived from the labeled image bank. It gives log ink
    # and the eight normalized spatial bins comparable dimensionless units.
    discrepancy = ((candidates - query[:, None]) / scale).square().clamp_max(16)
    return -0.5 * discrepancy[..., 0], -0.5 * discrepancy[..., 1:].mean(dim=-1)


def shuffle_by_count(prediction, totals, generator):
    """Break query identity within count octiles while retaining the count trend."""
    order = totals.argsort()
    shuffled = prediction.clone()
    for stratum in order.tensor_split(8):
        shuffled[stratum] = prediction[stratum[torch.randperm(len(stratum), generator=generator)]]
    return shuffled


def make_choices(shortlist, contour_score, prediction, count_prediction, truth, shuffled, candidate_features, scale):
    incoming = contour_score - contour_score.mean(dim=1, keepdim=True)
    anchor = incoming / incoming.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    area, marginal = density_compatibility(candidate_features, prediction, scale)
    count_area, _ = density_compatibility(candidate_features, count_prediction, scale)
    shuffled_area, shuffled_marginal = density_compatibility(candidate_features, shuffled, scale)
    true_area, _ = density_compatibility(candidate_features, truth, scale)
    proposed = {BASELINE: anchor, "count_only_ink": anchor + count_area,
                "predicted_ink": anchor + area, "predicted_ink_and_marginals": anchor + area + marginal,
                "shuffled_within_count_ink_and_marginals": anchor + shuffled_area + shuffled_marginal,
                "true_ink_assisted_diagnostic": anchor + true_area,
                "nearest_true_ink_diagnostic": true_area}
    return {method: shortlist.gather(1, scores.argmax(dim=1, keepdim=True)).squeeze(1)
            for method, scores in proposed.items()}


def paired_metrics(scored, baseline, references):
    delta = np.asarray(scored - baseline)
    _, groups = np.unique(references, return_inverse=True)
    totals = np.bincount(groups, weights=delta - delta.mean())
    standard_error = float(np.sqrt(len(totals) / (len(totals) - 1) * (totals**2).sum()) / len(delta))
    return {"ocf": float(scored.mean()), "ocf_n": len(delta), "paired_delta": float(delta.mean()),
            "paired_glyph_se": standard_error}


@torch.inference_mode()
def run_mouse(mouse, root, catalog, masks, signatures, energy, descriptors, scale, score, args):
    started = time.monotonic()
    device = masks.device
    groups = grouped_blocks(root, catalog, mouse, device, len(masks))
    available = set(groups)
    if any(session not in available for session in args.sessions):
        raise ValueError(f"Requested sessions missing from {mouse}: {args.sessions}")
    training, queries = {}, {}
    for index, (session, counts) in enumerate(groups.items()):
        if session in args.sessions:
            training[session], queries[session] = split_repeats(counts, args.repeat_seed + index)
    del groups
    print(f"{mouse}: fitting contour and ink readouts on other repeats", flush=True)
    contour = fit_contour_retrieval(training, signatures, energy, task="task3")
    full_model, count_model, count_moments = fit_descriptors(contour, training, descriptors)
    del training

    rng = np.random.default_rng(args.seed)
    shuffle = torch.Generator().manual_seed(args.seed)
    choices, references, oracle_choices, reliability = {}, [], [], {}
    all_predicted, all_counts = [], []
    correct_top1, correct_topk = 0, 0
    for session, query in queries.items():
        projected, counts = response_views(contour, query, session)
        predicted = full_model.predict(projected.squeeze(1))
        count_prediction = count_model.predict(_standardize(counts.squeeze(1), *count_moments[session]))
        shifted = shuffle_by_count(predicted, counts.flatten(), shuffle)
        contour_scores = contour_retrieval_scores(contour, query, session)["contour_ridge"]
        shortlist_scores, shortlist = contour_scores.topk(args.top_k, dim=1)
        destinations = torch.arange(len(masks), device=device)
        selected = torch.as_tensor(rng.choice(len(masks), args.ocf_per_session, replace=False), device=device)
        candidates = shortlist[selected]
        options = make_choices(candidates, shortlist_scores[selected], predicted[selected], count_prediction[selected],
                               descriptors[selected], shifted[selected], descriptors[candidates], scale)
        for method, index in options.items():
            choices.setdefault(method, []).append(index)
        references.append(selected)
        oracle_choices.append(candidates[:args.oracle_per_session])
        correct_top1 += int((shortlist[:, 0] == destinations).sum())
        correct_topk += int((shortlist == destinations[:, None]).any(dim=1).sum())
        reliability[session] = descriptor_reliability(predicted, count_prediction, descriptors)
        all_predicted.append(predicted)
        all_counts.append(count_prediction)

    reference = torch.cat(references)
    selected_choices = torch.stack([torch.cat(rows) for rows in choices.values()])
    measurements, unique = unique_pair_ocf(selected_choices, reference.expand_as(selected_choices), masks, score, device, 128)
    measured = measurements.cpu().numpy()
    samples = reference.numpy()
    summary = {}
    for index, method in enumerate(choices):
        summary[method] = paired_metrics(measured[index], measured[0], samples)
        by_session = (measured[index] - measured[0]).reshape(len(queries), args.ocf_per_session).mean(axis=1)
        summary[method]["paired_delta_by_session"] = dict(zip(queries, by_session.tolist()))
        summary[method]["changed_fraction"] = float((selected_choices[index] != selected_choices[0]).float().mean())

    oracle = torch.cat(oracle_choices)
    oracle_references = torch.cat([rows[:args.oracle_per_session] for rows in references])
    oracle_scores, unique_oracle = unique_pair_ocf(oracle, oracle_references[:, None].expand_as(oracle),
                                                  masks, score, device, 128)
    ceiling = paired_metrics(oracle_scores.max(dim=1).values.numpy(), oracle_scores[:, 0].numpy(),
                             oracle_references.numpy())
    ceiling["contour_only_ocf_on_oracle_rows"] = float(oracle_scores[:, 0].mean())
    result = {"mouse": mouse, "sessions": list(queries), "top1": correct_top1 / (len(masks) * len(queries)),
              "top_k_hit": correct_topk / (len(masks) * len(queries)), "metrics": summary,
              "fixed_shortlist_exact_ocf_oracle": ceiling, "unique_exact_pairs": unique,
              "unique_oracle_pairs": unique_oracle,
              "reliability": descriptor_reliability(torch.cat(all_predicted), torch.cat(all_counts), descriptors.repeat(len(queries), 1)),
              "reliability_by_session": reliability, "seconds": time.monotonic() - started}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session1", "session2", "session3", "session4"))
    parser.add_argument("--ocf-per-session", type=int, default=16)
    parser.add_argument("--oracle-per-session", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--repeat-seed", type=int, default=419)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_density_validation.json"))
    args = parser.parse_args()
    if (len(set(args.sessions)) < 2 or args.ocf_per_session * len(set(args.sessions)) < 48
            or not 2 <= args.oracle_per_session <= args.ocf_per_session or args.top_k < 2):
        parser.error("Use at least 48 OCF rows, at least two sessions, and two or more oracle rows and candidates")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, torch.device("cpu"))
    if max(args.top_k, args.ocf_per_session) > len(images):
        parser.error("Candidate and sample counts must fit inside the gallery")
    masks = images >= 128
    signature, energy = contour_signatures(images)
    descriptors = image_descriptors(masks)
    scale = descriptors.std(dim=0, unbiased=False).clamp_min(0.02)
    scorer = official_scorer(root)
    output = {"baseline": BASELINE, "seed": args.seed, "repeat_seed": args.repeat_seed,
              "ocf_per_session": args.ocf_per_session, "oracle_per_session": args.oracle_per_session,
              "top_k": args.top_k, "results": []}
    for mouse in args.mice:
        output["results"].append(run_mouse(mouse, root, catalog, masks, signature, energy, descriptors, scale, scorer, args))
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
