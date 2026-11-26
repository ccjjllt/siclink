"""Check for cross-day stimulus information in repeat-count clustering.

Use native spike counts from four same-glyph repeats, holding out each query
recording entirely. The conditional null preserves each unit's total spikes
and each trial's population total in each evoked window. Under independent
Poisson counts with a separable population gain, the conditional distribution
is exactly a random assignment of those unit labels to the repeat slots.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_hierarchical_gallery_validation import validated_logits


SESSIONS = ("session1", "session2", "session3", "session4")
WINDOWS = ((50, 100), (100, 250))
ARMS = ("v9", "collision", "count_matched_shuffle", "fixed_margin_null")
SCORE_WEIGHT = 0.08
TEMPERATURE = 1 / 3


def window_counts(raw, start, stop):
    return raw.unflatten(-1, (-1, 10))[..., start // 25:stop // 25].sum(dim=-1)


def fixed_margin_draw(counts, rng):
    """Sample repeat allocations without replacing event slots or unit labels."""
    by_unit = counts.sum(dim=1).numpy().astype(np.int64)
    by_repeat = counts.sum(dim=2).numpy().astype(np.int64)
    shuffled = np.empty(counts.shape, dtype=np.int32)
    for row, (pool, sizes) in enumerate(zip(by_unit, by_repeat)):
        if not pool.sum():
            shuffled[row].fill(0)
            continue
        remaining = pool.copy()
        for repeat, size in enumerate(sizes[:-1]):
            trial = rng.multivariate_hypergeometric(remaining, size, method="marginals")
            shuffled[row, repeat] = trial
            remaining -= trial
        shuffled[row, -1] = remaining
    if not (np.array_equal(shuffled.sum(axis=1), by_unit) and np.array_equal(shuffled.sum(axis=2), by_repeat)):
        raise RuntimeError("The exact null changed a count margin.")
    return torch.from_numpy(shuffled)


def falling(number, order):
    product = torch.ones_like(number)
    for offset in range(order):
        product *= (number - offset).clamp_min(0)
    return product


def conditional_moments(counts):
    """Return the exact null mean and variance of same-repeat ordered pairs."""
    counts = counts.double()
    unit = counts.sum(dim=1)
    slots = counts.sum(dim=2)
    total = slots.sum(dim=1, keepdim=True)
    slot_pairs = falling(slots, 2)
    pair_total = slot_pairs.sum(dim=1, keepdim=True)
    p2 = pair_total / falling(total, 2).clamp_min(1)
    p3 = falling(slots, 3).sum(dim=1, keepdim=True) / falling(total, 3).clamp_min(1)
    both = falling(slots, 4).sum(dim=1, keepdim=True) + pair_total.square() - slot_pairs.square().sum(dim=1, keepdim=True)
    p22 = both / falling(total, 4).clamp_min(1)
    center = falling(unit, 2) * p2
    second = 2 * center + 4 * falling(unit, 3) * p3 + falling(unit, 4) * p22
    variance = (second - center.square()).clamp_min(0)
    observed = falling(counts, 2).sum(dim=1)
    return observed, center, variance, unit, total, p2


def collision_features(counts):
    observed, center, variance, unit, total, p2 = conditional_moments(counts)
    valid = (unit >= 2).float()
    residual = ((observed - center) / (variance + 4).sqrt()).float().clamp(-8, 8) * valid
    own = unit.log1p().float() / 4
    population = (total.log1p().float() - 6) / 3
    imbalance = ((p2.float() - 0.25) * 5).expand_as(own)
    population = population.expand_as(own)
    nuisance = torch.stack((torch.ones_like(own), own, own.square(), population, imbalance, own * imbalance), dim=-1)
    return residual, nuisance, valid, float(valid.mean()), float((observed - center).mean())


def remove_marginal_trend(sources, query):
    """Fit a separate low-order count/exposure correction per unit on source days."""
    response = torch.cat([one[0] for one in sources], dim=0).T.unsqueeze(-1)
    design = torch.cat([one[1] * one[2][..., None] for one in sources], dim=0).transpose(0, 1)
    fit_size = design.shape[1]
    precision = design.transpose(1, 2) @ design / fit_size
    precision.diagonal(dim1=-2, dim2=-1).add_(0.002)
    rhs = design.transpose(1, 2) @ response / fit_size
    beta = torch.linalg.solve(precision, rhs).squeeze(-1)
    residuals = [(value - (matrix * beta[None]).sum(dim=-1)) * keep
                 for value, matrix, keep, _, _ in (*sources, query)]
    return torch.stack(residuals[:-1]), residuals[-1]


def normalized_logits(scores):
    centered = scores - scores.mean(dim=1, keepdim=True)
    return centered / centered.square().mean().sqrt().clamp_min(1e-6)


def conditional_retrieval(sources, query):
    mean = sources.mean(dim=(0, 1))
    sources = sources - mean
    query = query - mean
    scale = (sources.square().mean(dim=(0, 1)) + 0.05).sqrt()
    sources, query = sources / scale, query / scale
    across = (sources.sum(dim=0).square() - sources.square().sum(dim=0)).mean(dim=0)
    across /= len(sources) * (len(sources) - 1)
    evidence = across.clamp_min(0)
    weight = (0.25 + 0.75 * evidence / (0.04 + evidence)).sqrt()
    return F.normalize(query * weight, dim=1), F.normalize(sources.mean(dim=0) * weight, dim=1), across


def count_matched_permutation(source_counts, rng):
    """Derange within count strata while keeping multineuron count profiles close."""
    means = source_counts.mean(dim=0)
    window_total = means.sum(dim=-1)
    ordered = window_total.sum(dim=1).argsort()
    profiles = means.sqrt().flatten(1)
    profiles = F.normalize(profiles - profiles.mean(dim=0), dim=1)
    replacement = torch.empty(len(means), dtype=torch.long)
    similarities = []
    strata = ordered.tensor_split((len(means) + 95) // 96)
    for group in strata:
        similarity = profiles[group] @ profiles[group].T
        jitter = torch.from_numpy(rng.random(similarity.shape, dtype=np.float32)) * 0.01
        objective = (similarity + jitter).fill_diagonal_(-torch.inf)
        assignment = assign_glyphs(objective)
        replacement[group] = group[assignment]
        similarities.append(similarity.gather(1, assignment[:, None])[:, 0])
    if len(replacement.unique()) != len(means) or bool((replacement == torch.arange(len(means))).any()):
        raise RuntimeError("The source count-matched permutation needs distinct non-self glyphs.")
    distance = ((window_total - window_total[replacement]).abs() / window_total.clamp_min(1)).mean()
    return replacement, {"paired_profile_cosine": float(torch.cat(similarities).mean()),
                         "mean_relative_window_total_difference": float(distance)}


def identity_metrics(logits, truth):
    positive = logits.gather(1, truth[:, None])[:, 0]
    top = logits.topk(32, dim=1).indices == truth[:, None]
    inverse = truth.argsort()
    nll = torch.logsumexp(logits / TEMPERATURE, dim=1) - positive / TEMPERATURE
    gap = positive - (logits.sum(dim=1) - positive) / (logits.shape[1] - 1)
    return {"nll": nll[inverse], "top1": top[:, 0][inverse].float(),
            "top32": top.any(dim=1)[inverse].float(), "target_over_mean_false": gap[inverse]}


def mean_and_se(values):
    return {"mean": float(values.mean()), "glyph_se": float(values.std(unbiased=True) / len(values)**0.5)}


def summarize_identity(metrics):
    contrasts = {control: {key: mean_and_se(metrics["collision"][key] - metrics[control][key])
                           for key in metrics["collision"]} for control in ARMS if control != "collision"}
    values = {arm: {key: mean_and_se(vector) for key, vector in row.items()} for arm, row in metrics.items()}
    return {"values": values, "collision_minus": contrasts}


def evaluate(held, recordings, images, args):
    started = time.monotonic()
    rng = np.random.default_rng(args.seed + 1009 * SESSIONS.index(held))
    order = torch.as_tensor(rng.permutation(len(images)))
    sources = {day: counts for day, counts in recordings.items() if day != held}
    query = recordings[held][order]
    torch.manual_seed(args.seed)
    validated = validated_logits(args.mouse, sources, query, images, images.device, args)
    model = fit_hierarchical_gallery(sources, view="total")
    eb = model.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    base = normalized_logits(blend_hierarchical_logits(validated, eb))
    print(f"{held}: v9 fit and query in {time.monotonic() - started:.1f}s", flush=True)

    real_scores, null_scores, counts_for_matching, exposure = [], [], [], []
    for start, stop in WINDOWS:
        windows = [window_counts(raw, start, stop) for raw in (*sources.values(), query)]
        counts_for_matching.append(torch.stack([raw.sum(dim=1) for raw in windows[:-1]]))
        real = [collision_features(raw) for raw in windows]
        null = [collision_features(fixed_margin_draw(raw, rng)) for raw in windows]
        s, q = remove_marginal_trend(real[:-1], real[-1])
        u, v = remove_marginal_trend(null[:-1], null[-1])
        x, y, stability = conditional_retrieval(s, q)
        a, b, null_stability = conditional_retrieval(u, v)
        real_scores.append((x, y))
        null_scores.append(a @ b.T)
        exposure.append({"band_ms": [start, stop], "query_active_unit_fraction": real[-1][3],
                         "mean_query_excess_pairs": real[-1][4], "mean_null_excess_pairs": null[-1][4],
                         "mean_source_cross_day_stability": float(stability.mean()),
                         "mean_null_source_cross_day_stability": float(null_stability.mean())})
    permutation, match = count_matched_permutation(torch.stack(counts_for_matching, dim=2), rng)
    margins = {"collision": sum(x @ y.T for x, y in real_scores),
               "count_matched_shuffle": sum(x @ y[permutation].T for x, y in real_scores),
               "fixed_margin_null": sum(null_scores)}
    margins = {tag: normalized_logits(matrix) for tag, matrix in margins.items()}
    scores = {"v9": base, **{tag: base + SCORE_WEIGHT * matrix for tag, matrix in margins.items()}}
    identities = {tag: identity_metrics(matrix, order) for tag, matrix in scores.items()}
    residual_only = {tag: mean_and_se(identity_metrics(matrix, order)["target_over_mean_false"])
                     for tag, matrix in margins.items()}
    diagnostics = {"sources": list(sources), "held": held, "count_match": match, "exposure": exposure,
                   "full_gallery_size": len(images), "identity": summarize_identity(identities),
                   "residual_only_target_over_mean_false": residual_only,
                   "seconds_before_exact_ocf": time.monotonic() - started}
    choices = {tag: assign_glyphs(matrix)[order.argsort()] for tag, matrix in scores.items()}
    diagnostics["assignment_accuracy"] = {tag: float((values == torch.arange(len(images))).float().mean())
                                          for tag, values in choices.items()}
    diagnostics["assignment_changes_vs_v9"] = {tag: int((values != choices["v9"]).sum())
                                              for tag, values in choices.items() if tag != "v9"}
    return diagnostics, identities, choices


def exact_ocf(all_choices, selected, masks, scorer):
    ordered = torch.stack([one[tag][selected] for one in all_choices for tag in ARMS])
    n_glyphs = len(masks)
    keys = selected * n_glyphs + ordered
    distinct, inverse = torch.unique(keys.flatten(), return_inverse=True)
    guess = masks[distinct % n_glyphs].float().numpy()
    reference = masks[distinct // n_glyphs].numpy()
    measured = torch.as_tensor(scorer(guess, reference, device="cpu", batch_size=12), dtype=torch.float32)
    scores = measured[inverse].reshape(len(all_choices), len(ARMS), len(selected))
    outcome = {}
    for index, arm in enumerate(ARMS):
        paired = scores[:, index] - scores[:, 0]
        outcome[arm] = {"mean": float(scores[:, index].mean()), "paired_minus_v9": mean_and_se(paired.mean(dim=0)),
                        "per_day": [{"mean": float(values.mean()), "paired_minus_v9": mean_and_se(delta)}
                                    for values, delta in zip(scores[:, index], paired)]}
    contrast = {tag: mean_and_se((scores[:, 1] - scores[:, index]).mean(dim=0))
                for index, tag in enumerate(ARMS) if tag != "collision"}
    return {"paired_glyphs_per_day": len(selected), "unique_exact_pairs": len(distinct),
            "arms": outcome, "collision_minus": contrast}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", choices=("mice1", "mice2"), default="mice1")
    parser.add_argument("--held-sessions", choices=SESSIONS, nargs=2, default=("session2", "session4"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--ocf-images", type=int, default=96)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_repeat_collision.json"))
    parser.set_defaults(encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if args.held_sessions[0] == args.held_sessions[1] or args.ocf_images < 2:
        parser.error("Provide two different held days and a paired OCF sample.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this validation on Kaggle CPU with the competition data attached.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, torch.device("cpu"))
    recordings = grouped_blocks(root, load_catalog(root), args.mouse, images.device, len(images))
    if set(recordings) != set(SESSIONS) or any(values.shape[1] != 4 for values in recordings.values()):
        raise ValueError("Expected four training recordings with four trials per labeled glyph.")
    output = {"mouse": args.mouse, "seed": args.seed, "windows_ms": WINDOWS, "arms": ARMS,
              "baseline": "v9: 65/35 neural validated plus EB repeat covariance. One-to-one on all arms.",
              "nll_temperature_predeclared": TEMPERATURE, "collision_weight_predeclared": SCORE_WEIGHT,
              "primary": "Lower full-gallery NLL and higher top32 than every control on BOTH held days. "
                         "Fixed combination of two bands. Other metrics and glyph SEs are descriptive. "
                         "Omit OCF when the primary criteria fail.",
              "folds": []}
    identity, decisions = [], []
    for held in args.held_sessions:
        fold, identities, choices = evaluate(held, recordings, images, args)
        identity.append(identities)
        decisions.append(choices)
        output["folds"].append(fold)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(fold), flush=True)
    pooled = {tag: {key: torch.stack([day[tag][key] for day in identity]).mean(dim=0) for key in identity[0][tag]}
              for tag in ARMS}
    output["pooled_identity"] = summarize_identity(pooled)
    passes = all(delta["nll"]["mean"] < 0 and delta["top32"]["mean"] > 0
                 for day in output["folds"] for delta in day["identity"]["collision_minus"].values())
    output["primary_passes"] = passes
    if passes:
        selection = np.random.default_rng(args.seed + 101).choice(len(images), args.ocf_images, replace=False)
        output["exact_ocf"] = exact_ocf(decisions, torch.as_tensor(selection), images >= 128, official_scorer(root))
    output["seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"primary_passes": passes, "exact_ocf": output.get("exact_ocf"), "seconds": output["seconds"]}),
          flush=True)


if __name__ == "__main__":
    main()
