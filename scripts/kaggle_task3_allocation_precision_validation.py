"""Test stimulus-dependent temporal allocation precision against the frozen mice1 v12 policy."""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_density_top7_recheck import new_sample, top7_predictions
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_repeat_posterior import fit_models, score_sources


WINDOWS = ((2, 4, 10), (2, 3, 4), (4, 6, 10), (4, 5, 6), (6, 8, 10), (6, 7, 8), (8, 9, 10))
WEIGHT = 0.25


@torch.no_grad()
def functional_groups(training):
    means = torch.stack([counts.unflatten(-1, (-1, 10))[..., 2:].sum(dim=-1).mean(dim=1)
                         for counts in training.values()])
    contrasts = (means - means.mean(dim=1, keepdim=True)) / means.std(dim=1, keepdim=True).clamp_min(0.10)
    fingerprint = F.normalize(contrasts.mean(dim=0).T, dim=1)
    _, axes = torch.linalg.eigh(fingerprint @ fingerprint.T)
    groups = [torch.arange(len(axes), device=axes.device)]
    for axis in range(3):
        groups = [child for group in groups for child in group[axes[group, -(axis + 1)].argsort()].tensor_split(2)]
    membership = torch.zeros((len(axes), len(groups)), device=axes.device)
    for index, group in enumerate(groups):
        membership[group, index] = 1
    return membership, [len(group) for group in groups]


def tree_counts(counts, membership):
    values = counts.unflatten(-1, (-1, 10)).transpose(-2, -1) @ membership
    left = torch.stack([values[..., start:split, :].sum(dim=-2) for start, split, _ in WINDOWS], dim=-1)
    total = torch.stack([values[..., start:stop, :].sum(dim=-2) for start, _, stop in WINDOWS], dim=-1)
    return left, total


def binomial_log_kernel(left, total, probability):
    return left * probability.log() + (total - left) * (-probability).log1p()


def beta_binomial_log_kernel(left, total, probability, precision):
    alpha = precision * probability
    beta = precision * (1 - probability)
    beta_term = (torch.lgamma(left + alpha) + torch.lgamma(total - left + beta) - torch.lgamma(total + precision)
                 - torch.lgamma(alpha) - torch.lgamma(beta) + torch.lgamma(precision))
    return torch.where(total >= 2, beta_term, binomial_log_kernel(left, total, probability))


def observed_excess(left, total, probability, pairs):
    first, second = zip(*pairs)
    early_left, late_left = left[:, :, first], left[:, :, second]
    early_total, late_total = total[:, :, first], total[:, :, second]
    exposure = early_total.clamp_min(1).reciprocal() + late_total.clamp_min(1).reciprocal()
    squared = (early_left / early_total.clamp_min(1) - late_left / late_total.clamp_min(1)).square()
    binomial_var = (probability * (1 - probability)).clamp_min(0.04)[:, :, None]
    excess = ((squared / binomial_var - exposure) / (2 - exposure).clamp_min(0.25)).clamp(-0.6, 1.5)
    valid = (early_total >= 2) & (late_total >= 2)
    # Finite support shrinkage applies both during leave-one-repeat-out fitting and inference.
    return (excess * valid).sum(dim=2) / (2 + valid.sum(dim=2)), valid.float().mean(dim=2)


def features(probability, expected_total, left, total, ink, energy, *, leave_one_out):
    repeats = left.shape[2]
    pairs = tuple(combinations(range(repeats), 2))
    if leave_one_out:
        by_repeat = [observed_excess(left, total, probability, tuple(pair for pair in pairs if index not in pair))
                     for index in range(repeats)]
        observed, support = (torch.stack(columns, dim=2) for columns in zip(*by_repeat))
    else:
        observed, support = (tensor[:, :, None] for tensor in observed_excess(left, total, probability, pairs))
    p = probability[:, :, None].expand_as(observed)
    level = expected_total.log1p()[:, :, None].expand_as(observed)
    return torch.stack((p, p * (1 - p), level, observed, support,
                        ink[None, :, None, None, None].expand_as(observed),
                        energy[None, :, None, None, None].expand_as(observed)), dim=-1)


class Concentration(nn.Module):
    def __init__(self, sessions, groups):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(sessions, groups, len(WINDOWS)))
        self.rates = nn.Sequential(nn.Linear(7, 12), nn.SiLU(), nn.Linear(12, 1))
        nn.init.zeros_(self.rates[-1].weight)
        nn.init.zeros_(self.rates[-1].bias)

    def modulation(self, features):
        return 0.90 * self.rates(features).squeeze(-1).tanh()

    def precision(self, features=None):
        log_precision = 1 + 4 * self.base.sigmoid()
        if features is not None:
            log_precision = (log_precision[:, None, None] + self.modulation(features)).clamp(math.log(2), math.log(160))
        return log_precision.exp()


def fit_precision(left, total, probability, inputs, seed, epochs):
    torch.manual_seed(seed)
    model = Concentration(left.shape[0], left.shape[-2]).to(left.device)
    generator = torch.Generator(device=left.device).manual_seed(seed + 1)
    optimizer = torch.optim.Adam((model.base,), lr=0.045)
    eligible = total >= 2
    for _ in range(epochs):
        for rows in torch.randperm(left.shape[1], generator=generator, device=left.device).split(384):
            n, k, p = total[:, rows], left[:, rows], probability[:, rows, None]
            used = eligible[:, rows]
            loss = -beta_binomial_log_kernel(k, n, p, model.precision()[:, None, None])[used].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.base.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.rates.parameters(), lr=0.004, weight_decay=0.02)
    for _ in range(epochs):
        for rows in torch.randperm(left.shape[1], generator=generator, device=left.device).split(384):
            n, k, p, data = total[:, rows], left[:, rows], probability[:, rows, None], inputs[:, rows]
            used = eligible[:, rows]
            loss = -beta_binomial_log_kernel(k, n, p, model.precision(data))[used].mean()
            loss = loss + 0.02 * model.modulation(data).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def permute_modulation(modulation, probability, expected, generator):
    shuffled = modulation.clone()
    for group in range(modulation.shape[1]):
        for node in range(modulation.shape[2]):
            p = probability[:, group, node]
            rate = expected[:, group, node]
            for comparable_p in p.argsort().tensor_split(8):
                for cell in comparable_p[rate[comparable_p].argsort()].tensor_split(8):
                    order = torch.randperm(len(cell), generator=generator, device=p.device)
                    shuffled[cell, group, node] = modulation[cell[order], group, node]
    return shuffled


@torch.no_grad()
def full_log_likelihood(left, total, probability, precisions):
    scores = left.new_zeros((1 + len(precisions), len(left), len(probability)))
    for group in range(left.shape[1]):
        for node in range(left.shape[2]):
            n, k = total[:, group, node].long(), left[:, group, node].long()
            base = int(n.max()) + 1
            observed, inverse = torch.unique(n * base + k, return_inverse=True)
            counts = observed.div(base, rounding_mode="floor")[:, None]
            early = observed.remainder(base)[:, None]
            p = probability[:, group, node]
            scores[0].add_(binomial_log_kernel(early, counts, p)[inverse])
            for index, precision in enumerate(precisions, 1):
                known = precision[:, group, node]
                table = beta_binomial_log_kernel(early, counts, p, known)
                scores[index].add_(table[inverse])
    return scores


def ranks(scores, labels):
    actual = scores.gather(1, labels[:, None])
    rank = 1 + (scores > actual).sum(dim=1)
    return {f"top{cut}": float((rank <= cut).float().mean()) for cut in (1, 7, 32)}


def ocf_summary(scores, baseline):
    changes = (scores - baseline).reshape(-1, scores.shape[-1]).mean(axis=0)
    return {"ocf": float(scores.mean()), "paired_delta": float(changes.mean()),
            "paired_glyph_se": float(changes.std(ddof=1) / np.sqrt(len(changes))), "rows": int(scores.size)}


def independent_sample(gallery, seed):
    previous, _ = new_sample(gallery, 4817)
    previous = set(previous.tolist())
    selected, report = new_sample(gallery, seed)
    backup, _ = new_sample(gallery, seed + 1)
    candidates = tuple(dict.fromkeys(np.concatenate((selected, backup)).tolist()))
    fresh = np.asarray([glyph for glyph in candidates if glyph not in previous])[:len(selected)]
    if len(fresh) != len(selected):
        raise ValueError("Expected a full cohort disjoint from the prior v12 calibration and OCF references.")
    return fresh, {**report, "excluded_v12_recheck_glyphs": len(previous)}


@torch.no_grad()
def compare_session(models, query, order, sample, images, masks, membership, session, model_data, scorer, args):
    probability, expected, log_precision, modulation = model_data
    generator = torch.Generator(device=query.device).manual_seed(args.sample_seed + int(session[-1]))
    shuffled = permute_modulation(modulation, probability, expected, generator)
    fixed = log_precision.exp()[None].expand_as(probability)
    conditional, shuffled_conditional = ((log_precision[None] + delta).clamp(math.log(2), math.log(160)).exp()
                                        for delta in (modulation, shuffled))
    left, total = tree_counts(query[:, 0], membership)
    names = ("binomial", "fixed", "conditional", "shuffle")
    evidence = full_log_likelihood(left, total, probability, (fixed, conditional, shuffled_conditional))
    true = evidence.gather(2, order.expand(len(evidence), -1)[:, :, None]).squeeze(2)
    log_combination = (torch.lgamma(total + 1) - torch.lgamma(left + 1) - torch.lgamma(total - left + 1)).sum((1, 2))
    absolute_nll = -(true + log_combination[None])
    v11, _, v12 = score_sources(models, query, images, session, "mice1", args.dimensions)
    _, elected = top7_predictions(v11, v12)
    leader = v11.argmax(dim=1, keepdim=True)
    runner_up = v11.clone().scatter_(1, leader, -torch.inf).topk(15, dim=1).indices[:, :6]
    pool = torch.cat((leader, runner_up), dim=1)
    residual = evidence[1:] - evidence[0]
    residual -= residual.mean(dim=2, keepdim=True)
    scale = residual[0].std(dim=1, keepdim=True, unbiased=False).clamp_min(0.25)
    adjusted = v12[None] + WEIGHT * (residual / scale[None]).clamp(-2, 2)
    choices = {"v12": elected}
    for name, scores in zip(names[1:], adjusted):
        position = scores.gather(1, pool).argmax(dim=1, keepdim=True)
        choices[name] = pool.gather(1, position).squeeze(1)
    rows = order.argsort()[sample]
    chosen = torch.stack([values[rows] for values in choices.values()])
    exact, pairs = unique_pair_ocf(chosen, sample.expand_as(chosen), masks, scorer, query.device, 384)
    scored = dict(zip(choices, exact.cpu().numpy()))
    informative = total >= 2
    per_row = informative.sum(dim=(1, 2)).float()
    nll_difference = (absolute_nll[2] - absolute_nll[1], absolute_nll[2] - absolute_nll[3])
    report = {"paired_ocf": {name: ocf_summary(values, scored["v12"]) for name, values in scored.items()},
              "conditional_vs_fixed_ocf": ocf_summary(scored["conditional"], scored["fixed"]),
              "conditional_vs_shuffle_ocf": ocf_summary(scored["conditional"], scored["shuffle"]),
              "full_gallery_ranks": {"v12_unrestricted": ranks(v12, order), **{name: ranks(values, order)
                                    for name, values in zip(names[1:], adjusted)}},
              "standalone_likelihood_ranks": {name: ranks(values, order) for name, values in zip(names, evidence)},
              "v12_top7_pool_coverage": float((pool == order[:, None]).any(dim=1).float().mean()),
              "count_allocation_nll": {name: float(values.mean()) for name, values in zip(names, absolute_nll)},
              "conditional_nll_comparisons": {name: {"delta": float(delta.mean()),
                                                      "paired_glyph_se": float(delta.std() / len(delta)**0.5)}
                                             for name, delta in zip(("fixed", "shuffle"), nll_difference)},
              "actual_v12_changed": {name: int((item != elected).sum()) for name, item in choices.items()},
              "informative_event_fraction": float(informative.float().mean()),
              "informative_event_fraction_by_node": informative.float().mean(dim=(0, 1)).tolist(),
              "average_informative_events_per_trial": float(per_row.mean()),
              "queries_with_informative_event": float((per_row > 0).float().mean()),
              "mean_abs_conditional_log_precision_delta": float(modulation.abs().mean()), "unique_ocf_pairs": pairs}
    return scored, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-seed", type=int, default=6733)
    parser.add_argument("--sample-seed", type=int, default=6747)
    parser.add_argument("--precision-epochs", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_allocation_precision_validation.json"))
    parser.set_defaults(seed=211, encoder_epochs=18, epochs=8, dimensions=128)
    args = parser.parse_args()
    if args.precision_epochs < 1:
        parser.error("Fit at least one precision epoch.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this validation against labeled inputs on one Kaggle T4.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    device = torch.device("cuda:0")
    images = load_images(root, device)
    masks = images >= 128
    signatures, energy = contour_signatures(images)
    catalog = load_catalog(root)
    groups = grouped_blocks(root, catalog, "mice1", device, len(images))
    training, held = {}, {}
    for i, (name, counts) in enumerate(groups.items()):
        training[name], held[name] = split_repeats(counts, args.repeat_seed + i)
    del groups
    models, baseline_fit = fit_models(training, images, signatures, energy, "mice1", args)
    membership, sizes = functional_groups(training)
    observed = torch.stack([counts for counts in training.values()])
    left, total = tree_counts(observed, membership)
    del observed, training
    means = torch.stack([models.encoder.expected_counts(images, session) for session in held])
    expected_left, expected_total = tree_counts(means, membership)
    probability = (expected_left / expected_total.clamp_min(1e-5)).clamp(0.003, 0.997)
    ink = masks.float().mean(dim=(1, 2))
    ink = (ink - ink.mean()) / ink.std().clamp_min(1e-5)
    log_energy = energy.log()
    log_energy = (log_energy - log_energy.mean()) / log_energy.std().clamp_min(1e-5)
    source = features(probability, expected_total, left, total, ink, log_energy, leave_one_out=True)
    future = features(probability, expected_total, left, total, ink, log_energy, leave_one_out=False)
    model = fit_precision(left, total, probability, source, args.seed, args.precision_epochs)
    fixed = model.precision().detach().log()
    correction = model.modulation(future).detach().squeeze(2)
    sample_ids, sample_report = independent_sample(len(images), args.sample_seed)
    sample = torch.as_tensor(sample_ids, device=device)
    scorer = official_scorer(root)
    rng = np.random.default_rng(args.sample_seed + 1)
    output = {"mouse": "mice1", "baseline": "exact v12 topology refit on three disjoint labeled repeats",
              "training_repeats": 3, "test_repeat_proxy": 1, "windows_start_split_stop_ms": [[25 * bin for bin in node] for node in WINDOWS],
              "functional_group_sizes": sizes, "baseline_fit": baseline_fit, "scored_glyphs_per_block": len(sample),
              "sample": sample_report, "weight": WEIGHT, "repeat_seed": args.repeat_seed, "sample_seed": args.sample_seed,
              "precision_epochs": args.precision_epochs, "sessions": {}}
    all_scores = {}
    for i, (session, counts) in enumerate(held.items()):
        order = torch.as_tensor(rng.permutation(len(images)), device=device)
        values, report = compare_session(models, counts[order], order, sample, images, masks, membership, session,
                                         (probability[i], expected_total[i], fixed[i], correction[i]), scorer, args)
        output["sessions"][session] = report
        for name, scores in values.items():
            all_scores.setdefault(name, []).append(scores)
        stacked = {name: np.stack(values) for name, values in all_scores.items()}
        output["aggregate"] = {name: ocf_summary(scores, stacked["v12"]) for name, scores in stacked.items()}
        output["conditional_vs_fixed"] = ocf_summary(stacked["conditional"], stacked["fixed"])
        output["conditional_vs_shuffle"] = ocf_summary(stacked["conditional"], stacked["shuffle"])
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, **report}), flush=True)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
