"""Compare joint and pairwise cross-recording alignment on shuffled held glyphs."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_multiview_alignment import fit_multiview, pair_similarity
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import feature_view, standardize


def pair_scores(views):
    scores = {}
    for left, right in combinations(views, 2):
        values = pair_similarity(views[left], views[right])
        values = values - values.mean(dim=1, keepdim=True)
        values = values - values.mean(dim=0, keepdim=True)
        values = values / values.square().mean().sqrt().clamp_min(1e-6)
        scores[left, right], scores[right, left] = values, values.T
    return scores


def plans_from_scores(scores, temperature):
    plans = {}
    for key, matrix in scores.items():
        prior = matrix / temperature
        for _ in range(24):
            prior = prior - prior.logsumexp(dim=1, keepdim=True)
            prior = prior - prior.logsumexp(dim=0, keepdim=True)
        plans[key] = prior.softmax(dim=1)
    return plans


def cycle_scores(scores, names, temperature=0.45):
    """Add indirect correspondence evidence from both independent intermediate recordings."""
    plans = plans_from_scores(scores, temperature)
    corrected = {}
    for left, right in combinations(names, 2):
        paths = [plans[left, other] @ plans[other, right] for other in names if other not in (left, right)]
        indirect = torch.stack(paths).mean(dim=0).clamp_min(1e-9).log()
        indirect = indirect - indirect.mean()
        indirect = indirect / indirect.square().mean().sqrt().clamp_min(1e-6)
        corrected[left, right] = 0.65 * scores[left, right] + 0.35 * indirect
        corrected[right, left] = corrected[left, right].T
    return corrected


def total_agreement(scores, alignments):
    return float(sum(scores[left, right][alignments[left], alignments[right]].mean()
                     for left, right in combinations(alignments, 2)))


def joint_assignment(scores, initialization, names):
    """Use several anchored starts and optimize one complete permutation at a time."""
    best, best_score = None, -float("inf")
    example = next(iter(scores.values()))
    identity = torch.arange(len(example), device=example.device)
    for anchor in names:
        positions = {name: (identity if name == anchor
                            else assign_glyphs(initialization[anchor, name])) for name in names}
        for _ in range(8):
            changes = 0
            for name in names:
                if name == anchor:
                    continue
                combined = sum(scores[peer, name][positions[peer]] for peer in names if peer != name)
                updated = assign_glyphs(combined)
                changes += int((positions[name] != updated).sum())
                positions[name] = updated
            if changes == 0:
                break
        objective = total_agreement(scores, positions)
        if objective > best_score:
            best, best_score = positions, objective
    return {(left, right): best[right][best[left].argsort()] for left, right in scores}, best_score


def encode(model, data, glyphs, permutations, repeats):
    return {name: model.encode(name, counts[glyphs[permutations[name]]][:, repeats])
            for name, counts in data.items()}


def accuracy(matches, ground_truth):
    return float(torch.cat([choice == ground_truth[key] for key, choice in matches.items()]).float().mean())


def fit_pair_evidence(data, fit):
    """Fit a regularized Gaussian same-glyph likelihood in paired canonical axes."""
    settings, whitened, axes = {}, {}, {}
    for name, blocks in data.items():
        counts, moments = standardize(feature_view(blocks[fit], "early-late"))
        means = counts.mean(dim=1)
        center = means.mean(dim=0)
        centered = means - center
        covariance = centered.T @ centered / len(centered)
        residual = (counts - means[:, None]).flatten(0, 1)
        covariance += 0.10 * (residual.T @ residual / (len(counts) * (counts.shape[1] - 1) * counts.shape[1]))
        covariance.diagonal().add_(0.25 * covariance.diagonal().mean().clamp_min(1e-5))
        lower = torch.linalg.cholesky(covariance)
        settings[name] = (moments, center, lower)
        whitened[name] = torch.linalg.solve_triangular(lower, centered.T, upper=False).T
    for left, right in combinations(data, 2):
        cross = whitened[left].T @ whitened[right] / len(fit)
        u, values, vt = torch.linalg.svd(cross, full_matrices=False)
        correlations = (values[:64] * 0.90).clamp(0.0, 0.94)
        axes[left, right] = (u[:, :len(correlations)].contiguous(), vt[:len(correlations)].T.contiguous(), correlations)
    return settings, axes


def conditional_evidence(model, data, held, permutations):
    settings, axes = model
    whitened = {}
    for name, blocks in data.items():
        moments, center, lower = settings[name]
        observed, _ = standardize(feature_view(blocks[held[permutations[name]]], "early-late"), moments)
        whitened[name] = torch.linalg.solve_triangular(lower, (observed.mean(dim=1) - center).T,
                                                       upper=False).T
    likelihood, cosine = {}, {}
    for (left, right), (u, v, rho) in axes.items():
        x, y = whitened[left] @ u, whitened[right] @ v
        denominator = 1 - rho.square()
        evidence = (x * (rho / denominator)) @ y.T
        squares = rho.square() / denominator
        norm_x, norm_y = x.square() @ squares, y.square() @ squares
        likelihood[left, right] = evidence - 0.50 * (norm_x[:, None] + norm_y)
        likelihood[right, left] = likelihood[left, right].T
        similarity = torch.nn.functional.normalize(x, dim=1) @ torch.nn.functional.normalize(y, dim=1).T
        cosine[left, right], cosine[right, left] = similarity, similarity.T
    return likelihood, cosine


def compare_conditional(model, data, held, permutations):
    truth = {(left, right): permutations[right].argsort()[permutations[left]]
             for left in data for right in data if left != right}
    likelihood, cosine = conditional_evidence(model, data, held, permutations)
    row_scores, normalized = {}, {}
    for key, values in likelihood.items():
        row_scores[key] = values.argmax(dim=1)
        zero = values - values.mean(dim=1, keepdim=True)
        zero = zero - zero.mean(dim=0, keepdim=True)
        normalized[key] = zero / zero.square().mean().sqrt().clamp_min(1e-6)
    cycles = cycle_scores(normalized, tuple(data))
    assignments = {key: assign_glyphs(matrix) for key, matrix in normalized.items()}
    joint, _ = joint_assignment(normalized, cycles, tuple(data))
    return {"canonical_cosine_top1": accuracy({key: values.argmax(1) for key, values in cosine.items()}, truth),
            "conditional_top1": accuracy(row_scores, truth), "conditional_assignment": accuracy(assignments, truth),
            "conditional_two_hop": accuracy({key: assign_glyphs(matrix) for key, matrix in cycles.items()}, truth),
            "conditional_joint": accuracy(joint, truth)}


def subset_metrics(matched, selected, ground_truth):
    used = torch.cat(list(selected.values()))
    correct = torch.cat([result == ground_truth[key] for key, result in matched.items()])
    selected_n = int(used.sum())
    verified = int((used & correct).sum())
    return {"agreed": selected_n, "correct": verified, "pairs": len(used), "coverage": selected_n / len(used),
            "precision": verified / selected_n if selected_n else None}


def compare(model, data, held, permutations):
    truth = {(left, right): permutations[right].argsort()[permutations[left]]
             for left in data for right in data if left != right}
    embedding = encode(model, data, held, permutations, slice(None))
    direct = pair_scores(embedding)
    corrected = cycle_scores(direct, tuple(embedding))
    nearest = {key: matrix.argmax(dim=1) for key, matrix in direct.items()}
    direct_assignment = {key: assign_glyphs(matrix) for key, matrix in direct.items()}
    cycle_assignment = {key: assign_glyphs(matrix) for key, matrix in corrected.items()}
    joint, joint_objective = joint_assignment(direct, corrected, tuple(embedding))
    votes = {(left, right): sum(direct_assignment[left, peer] == direct_assignment[right, peer][match]
                                for peer in data if peer not in (left, right))
             for (left, right), match in joint.items()}
    group_agreement = subset_metrics(joint, {key: value == 2 for key, value in votes.items()}, truth)

    splits = []
    for half in (slice(0, 2), slice(2, 4)):
        values = pair_scores(encode(model, data, held, permutations, half))
        corrected_half = cycle_scores(values, tuple(data))
        halves, _ = joint_assignment(values, corrected_half, tuple(data))
        splits.append(halves)
    first, second = splits
    split_agreement = subset_metrics(first, {key: value == second[key] for key, value in first.items()}, truth)
    return {"glyphs": len(held), "direct_top1": accuracy(nearest, truth),
            "pairwise_assignment": accuracy(direct_assignment, truth),
            "two_hop_assignment": accuracy(cycle_assignment, truth),
            "joint_assignment": accuracy(joint, truth), "joint_objective": joint_objective,
            "joint_first_half": accuracy(first, truth), "joint_second_half": accuracy(second, truth),
            "both_intermediate_recordings_agree": group_agreement, "repeat_halves_agree": split_agreement}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=961)
    parser.add_argument("--conditional-pairs", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/cycle_alignment_validation.json"))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use the official data in a Kaggle Notebook.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    output = {"seed": args.seed, "mice": {}, "protocol": "Four independently shuffled recording blocks share the "
              "same 320 withheld training glyphs for this diagnostic. GCCA uses only other labeled training glyphs. "
              "Joint objectives and initialization use only recorded counts and full-block permutation constraints. "
              "No official test recording or held image is used to fit or select a match."}
    seed2 = args.seed + 692
    first_fit, first_held = split_glyphs(len(images), 320, args.seed, device)
    order = np.random.default_rng(seed2).permutation(len(images))
    held_second = np.sort(order[~np.isin(order, first_held.numpy())][:320])
    fit_second = np.setdiff1d(np.arange(len(images)), held_second)
    folds = ((args.seed, first_fit, first_held), (seed2, torch.as_tensor(fit_second), torch.as_tensor(held_second)))
    for mouse in ("mice1", "mice2"):
        grouped = {(mouse, name): block for name, block in grouped_blocks(root, catalog, mouse, device, len(images)).items()}
        output["mice"][mouse] = []
        for seed, fit, held in folds:
            model = fit_multiview({name: values[fit] for name, values in grouped.items()})
            rng = np.random.default_rng(seed + 910)
            permutations = {name: torch.as_tensor(rng.permutation(len(held)), device=device) for name in grouped}
            result = {"split_seed": seed, **compare(model, grouped, held, permutations)}
            if args.conditional_pairs:
                result["pair_likelihood"] = compare_conditional(fit_pair_evidence(grouped, fit), grouped, held, permutations)
            output["mice"][mouse].append(result)
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"mouse": mouse, **result}), flush=True)
        del grouped


if __name__ == "__main__":
    main()
