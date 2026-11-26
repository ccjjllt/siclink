"""Assess soft capacity on latent glyphs while leaving v9 output actions unconstrained."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.special import logsumexp
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import oracle_scores, split_calibration
from kaggle_gallery_validation import score_block
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank, shortlist, summarize


V9_ANCHOR = 4.0
V9_SEED = 211
DUSTBIN = 0.25
TEMPERATURE = 0.45
CAPACITY_WEIGHT = 3.0
DIRECT_PENALTY = 2.0
LATENT_CLIPS = (1.0, 3.0)


def latent_probabilities(local_indices, valid, scores, strength):
    """Solve the squared excess-occupancy penalty in the nonnegative dual."""
    n_candidates = int(local_indices.max()) + 1
    latent = np.where(valid, scores / TEMPERATURE, -np.inf)
    rows, slots = np.nonzero(valid)
    columns = local_indices[rows, slots]
    latent += np.log1p(-DUSTBIN) - logsumexp(latent, axis=1, keepdims=True)
    log_unknown = np.log(DUSTBIN)

    def posterior(discount):
        discounted = latent - discount[local_indices]
        normalizer = np.logaddexp(logsumexp(discounted, axis=1), log_unknown)
        mass = np.exp(discounted - normalizer[:, None])
        occupancy = np.bincount(columns, weights=mass[rows, slots], minlength=n_candidates)
        return mass, occupancy, normalizer

    zero = np.zeros(n_candidates, dtype=np.float64)
    independent, before, _ = posterior(zero)
    if strength == 0 or not np.any(before > 1):
        return independent, independent, {"active_candidates": 0, "converged": True,
                                          "peak_occupancy_before": float(before.max()),
                                          "peak_occupancy_after": float(before.max()),
                                          "mean_query_total_variation": 0.0, "mean_dustbin_after": DUSTBIN}

    def objective(discount):
        _, occupancy, normalizer = posterior(discount)
        value = normalizer.sum() + discount.sum() + (discount @ discount) / (2 * strength)
        gradient = 1 + discount / strength - occupancy
        return value, gradient

    fit = minimize(objective, zero, jac=True, bounds=[(0, None)] * n_candidates,
                   method="L-BFGS-B", options={"ftol": 1e-11, "gtol": 1e-7, "maxiter": 120})
    adjusted, after, _ = posterior(fit.x)
    delta = adjusted - independent
    variation = (np.abs(delta).sum(axis=1) + np.abs(delta.sum(axis=1))) * 0.5
    diagnostics = {"active_candidates": int((fit.x > 1e-7).sum()), "converged": bool(fit.success),
                   "solver_iterations": int(fit.nit), "candidates_over_capacity_initial": int((before > 1).sum()),
                   "peak_occupancy_before": float(before.max()), "peak_occupancy_after": float(after.max()),
                   "mean_query_total_variation": float(variation.mean()),
                   "rows_with_tv_over_1pct": int((variation > 0.01).sum()),
                   "mean_dustbin_after": float(1 - adjusted.sum(axis=1).mean())}
    return independent, adjusted, diagnostics


def direct_assignment(local_indices, valid, utility, penalty):
    """Assign private copies at a marginal collision cost, or one copy in hard mode."""
    n_rows = len(utility)
    rows, slots = np.nonzero(valid)
    columns = local_indices[rows, slots]
    choices = local_indices[np.arange(n_rows), utility.argmax(axis=1)]
    counts = np.bincount(columns)
    copies = counts if penalty is not None else np.ones_like(counts)
    starts = np.cumsum(copies) - copies
    expanded_size = copies[columns]
    expanded = np.repeat(np.arange(len(rows)), expanded_size)
    copy = np.arange(len(expanded)) - np.repeat(np.cumsum(expanded_size) - expanded_size, expanded_size)
    expanded_columns = starts[columns[expanded]] + copy
    expanded_rows = rows[expanded]
    cost = 1 + utility.max(axis=1)[expanded_rows] - utility[expanded_rows, slots[expanded]]
    public_labels = np.repeat(np.arange(len(copies)), copies)
    if penalty is not None:
        cost += penalty * copy
        fallback = 0
    else:
        # Finite private edges expose infeasibility instead of widening shortlists.
        expanded_columns = np.concatenate((expanded_columns, len(public_labels) + np.arange(n_rows)))
        expanded_rows = np.concatenate((expanded_rows, np.arange(n_rows)))
        cost = np.concatenate((cost, np.full(n_rows, 10_000.0)))
        public_labels = np.concatenate((public_labels, choices))
    graph = csr_matrix((cost, (expanded_rows, expanded_columns)), shape=(n_rows, len(public_labels)))
    matched_rows, assigned = min_weight_full_bipartite_matching(graph)
    if penalty is None:
        fallback = int((assigned >= copies.sum()).sum())
    selected = np.empty(n_rows, dtype=np.int64)
    selected[matched_rows] = public_labels[assigned]
    return selected, fallback


@torch.no_grad()
def contour_utility_delta(rendered, labels, local_indices, valid, adjustment):
    """Propagate changes in latent identity to the utility of possible hard glyphs."""
    signatures, energies = contour_signatures(rendered[labels])
    correction = []
    for indices, mask, diff in zip(local_indices.split(24), valid.split(24), adjustment.split(24)):
        glyphs = signatures[indices]
        contour = torch.bmm(glyphs, glyphs.transpose(1, 2)).clamp_(0, 1)
        energy = energies[indices]
        left, right = energy[:, :, None], energy[:, None, :]
        ratio = torch.minimum(left, right) / torch.maximum(left, right)
        agreement = ratio * contour
        similarity = agreement / (1 + ratio - agreement).clamp_min(1e-7)
        change = 100 * torch.bmm(similarity, diff.unsqueeze(-1)).squeeze(-1)
        center = change.masked_fill(~mask, 0).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
        correction.append((change - center).masked_fill(~mask, 0))
    return torch.cat(correction)


@torch.no_grad()
def decisions(candidates, ranker, rendered):
    utility = ranker.rank(candidates, anchor_ocf=V9_ANCHOR)
    baseline = ranker.select(candidates, anchor_ocf=V9_ANCHOR)
    valid = candidates.valid
    labels, inverse = torch.unique(candidates.indices[valid], sorted=True, return_inverse=True)
    local = torch.zeros_like(candidates.indices)
    local[valid] = inverse
    valid_numpy, local_numpy = valid.cpu().numpy(), local.cpu().numpy()
    utility_numpy = utility.float().cpu().numpy()
    scores = candidates.baseline_z.double().cpu().numpy()
    p, q, latent = latent_probabilities(local_numpy, valid_numpy, scores, CAPACITY_WEIGHT)
    _, q_reverse, _ = latent_probabilities(local_numpy[::-1], valid_numpy[::-1], scores[::-1], CAPACITY_WEIGHT)
    discrepancy = float(np.max(np.abs(q - q_reverse[::-1])))
    if discrepancy > 1e-5:
        raise ValueError(f"Latent update changed after row permutation: {discrepancy}")
    latent["reversed_rows_max_probability_difference"] = discrepancy
    null_before, null_after, _ = latent_probabilities(local_numpy, valid_numpy, scores, 0)
    if not np.array_equal(null_before, null_after):
        raise ValueError("Zero occupancy regularization must leave the posterior unchanged.")

    delta = torch.as_tensor(q - p, device=utility.device, dtype=utility.dtype)
    correction = contour_utility_delta(rendered, labels, local, valid, delta)
    chosen = {"v9": baseline}
    for clip in LATENT_CLIPS:
        choice = (utility + correction.clamp(-clip, clip)).argmax(dim=1, keepdim=True)
        chosen[f"latent_clip{clip:g}"] = candidates.indices.gather(1, choice).squeeze(1)
    for name, penalty in (("direct_soft2", DIRECT_PENALTY), ("direct_hard", None)):
        selected, fallbacks = direct_assignment(local_numpy, valid_numpy, utility_numpy, penalty)
        chosen[name] = labels[torch.as_tensor(selected, device=labels.device)]
        if name == "direct_hard":
            latent["hard_unmatched_rows"] = fallbacks
    occupancy = torch.bincount(baseline, minlength=len(rendered))
    collision = occupancy[baseline] > 1
    latent["v9_duplicate_rows"] = int(collision.sum())
    latent["v9_duplicate_atlas_glyphs"] = int((occupancy > 1).sum())
    latent["max_absolute_action_correction_ocf"] = float(correction.abs().max())
    latent["rows_with_correction_above_025_ocf"] = int((correction.abs().amax(dim=1) > 0.25).sum())
    latent["choices_changed_full_block"] = {name: int((choice != baseline).sum()) for name, choice in chosen.items()}
    return chosen, collision, latent


def score_choices(scorer, choices, rows, rendered, references, device):
    choices_matrix = torch.stack([choice[rows] for choice in choices.values()])
    pairs = (len(rendered) * torch.arange(len(rows), device=device))[None] + choices_matrix
    keys, inverse = torch.unique(pairs, return_inverse=True)
    reference_rows = keys.div(len(rendered), rounding_mode="floor").cpu().numpy()
    predictions = rendered[keys.remainder(len(rendered))]
    scores = score_block(scorer, {"all": predictions}, references[reference_rows], device)["all"]
    return dict(zip(choices, scores[inverse.cpu().numpy()])), len(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3307)
    parser.add_argument("--avoid-held-seed", type=int, nargs="+", default=(961, 2209, 613))
    parser.add_argument("--ocf-images", type=int, default=160)
    parser.add_argument("--oracle-collisions", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_latent_repertoire_validation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle GPU with the mounted labeled competition data.")
    if not 160 <= args.ocf_images <= 320 or not 0 <= args.oracle_collisions <= args.ocf_images:
        parser.error("Compare at least 160 and at most 320 paired held glyphs per block.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    fitting, held = outer_split(len(images), args.seed, args.avoid_held_seed, device)
    split = split_calibration(len(fitting), 96, 384, device, V9_SEED)
    groups = grouped_blocks(root, catalog, "mice1", device, len(images))
    training = {session: counts[fitting] for session, counts in groups.items()}
    scorer = official_scorer(root)
    print("Fitting the frozen v9 policy on the nested labeled folds", flush=True)
    ranker, likelihoods, nested, codepoints = nested_ranker(training, images[fitting], split, scorer, V9_SEED)
    model = fit_v8(training, images[fitting], V9_SEED)
    if not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("Expected strictly aligned nested and outer public atlas columns.")
    del training
    rendered = rendered_bank(model)
    rng = np.random.default_rng(args.seed + 1)
    sample = torch.as_tensor(rng.choice(len(held), args.ocf_images, replace=False), device=device)
    output = {"mouse": "mice1", "outer_seed": args.seed, "avoided_outer_seeds": args.avoid_held_seed,
              "v9_seed": V9_SEED, "v9_anchor": V9_ANCHOR, "nominations": NOMINATIONS,
              "held_glyphs": len(held), "oracle_collisions_per_session": args.oracle_collisions,
              "scored_images_per_session": len(sample), "repeats": 4, "per_session": {},
              "latent_parameters": {"temperature": TEMPERATURE, "dustbin_mass": DUSTBIN,
                                    "capacity_weight": CAPACITY_WEIGHT, "action_clips_ocf": LATENT_CLIPS},
              "calibration": {"ranker": ranker.diagnostics, "nested": nested},
              "protocol": "All 320 held rows in each session are independently permuted before the set policies. "
                          "Only the public atlas indices and spike-derived scores enter optimization. Held references "
                          "are used afterward for official OCF. No actual competition test examples are accessed."}
    measured = {}
    for session in SESSIONS:
        permutation = torch.as_tensor(rng.permutation(len(held)), device=device)
        shuffled, reverse = held[permutation], permutation.argsort()
        candidates, _ = shortlist(model, likelihoods[session], groups[session][shuffled], session)
        choices, collision, diagnostics = decisions(candidates, ranker, rendered)
        rows = reverse[sample]
        ref = (images[shuffled[rows]] >= 128).cpu().numpy()
        observed, exact_pairs = score_choices(scorer, choices, rows, rendered, ref, device)
        paired_collision = collision[rows].cpu().numpy()
        scored = {}
        for method, scores in observed.items():
            measured.setdefault(method, []).append(scores)
            delta = scores - observed["v9"]
            scored[method] = {**summarize([scores], [observed["v9"]]),
                              "mean_delta_on_v9_collision_rows": float(delta[paired_collision].mean()) if paired_collision.any() else None}
        report = {"diagnostics": diagnostics, "metrics": scored, "paired_collision_rows": int(paired_collision.sum()),
                  "unique_exact_pairs": exact_pairs}
        if args.oracle_collisions and paired_collision.any():
            sample_collision = np.flatnonzero(paired_collision)[:args.oracle_collisions]
            oracle_rows = rows[torch.as_tensor(sample_collision, device=device)]
            oracle_candidates = candidates.take_rows(oracle_rows)
            exact = oracle_scores(oracle_candidates, rendered, ref[sample_collision], scorer, device)
            best = exact.masked_fill(~oracle_candidates.valid, -torch.inf).amax(dim=1).cpu().numpy()
            report["collision_subset_oracle"] = summarize([best], [observed["v9"][sample_collision]])
        output["per_session"][session] = report
        output["summary"] = {method: summarize(scores, measured["v9"]) for method, scores in measured.items()}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, **report}), flush=True)
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
