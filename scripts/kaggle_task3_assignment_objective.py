"""Learn a same-day single-trial metric with an exact assignment training objective."""

import json
from pathlib import Path
import time

from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_episodic_assignment import ResidualProjection, _initialize, _projection
from glyph_mbr import contour_signatures, glyph_posterior
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_gallery_validation import standardized
from kaggle_posterior_utility_validation import fit_neural, v11_logits
from kaggle_task3_count_occupancy import paired_delta, posterior_metrics
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_utility_candidate import MOUSE, SHORTLIST, TABLE_SHA, validated_utility
from kaggle_task3_peer_rate_recheck import replicate_split
from kaggle_task3_peer_rate_transfer import eligible_glyphs, new_fold
from kaggle_task3_rankband_ocf_validation import NeuralRows
from kaggle_task3_relational_consensus import choose_images
from kaggle_task3_v20_candidate_headroom import held_repeats
from kaggle_task3_v20_exact_utility import original_panel, paired_summary, publish


SEED, DAYS, PER_DAY = 376291, ("session1", "session3"), 96
VIEW, DIMENSION, EPISODE, EPOCHS = "evoked-multiscale", 96, 128, 10
HEAD_SCALE, HINGE_SHARE, REGRET_SCALE, MIX = 10.0, 0.45, 2.0 / 100, 0.30
V20, BASE_BALANCED = "v20", "v20_posthoc_balance"
INDEPENDENT, ASSIGNMENT = "independent_training", "assignment_training"
OUTPUT = Path("/kaggle/working/task3_assignment_objective.json")


def new_cohort(size, cpu):
    _, _, first, earlier, _ = new_fold(size, cpu)
    _, _, _, second, _, _, _ = replicate_split(size, cpu)
    eligible, _ = eligible_glyphs(size, cpu)
    excluded = torch.cat((*first.values(), *second.values()))
    fresh = eligible[~torch.isin(eligible, excluded)]
    if len(fresh) < len(DAYS) * PER_DAY:
        raise ValueError("Choose 192 evaluation glyphs outside all named exact-score cohorts.")
    rng = torch.Generator().manual_seed(SEED)
    fresh = fresh[torch.randperm(len(fresh), generator=rng)]
    targets = dict(zip(DAYS, fresh[:len(DAYS) * PER_DAY].split(PER_DAY), strict=True))
    gallery = torch.arange(size)
    fit = gallery[~torch.isin(gallery, fresh)]
    return fit, fresh, targets, earlier, {"previous_rate_study_endpoints": int(torch.isin(fresh, excluded).sum())}


@torch.no_grad()
def population_features(support, query, fit):
    axes, _ = _projection({day: values[fit] for day, values in support.items()}, VIEW, DIMENSION)
    projected = {}
    for day, values in support.items():
        trials = feature_view(values, VIEW)
        _, stats = standardize(trials[fit])
        fitted, _ = standardize(trials, stats)
        measured, _ = standardize(feature_view(query[day], VIEW), stats)
        projected[day] = (fitted.mean(1) @ axes, measured[:, 0] @ axes)
    return projected


def training_episodes(fit, visual, rng, clustered):
    remaining = fit[torch.randperm(len(fit), generator=rng)]
    if not clustered:
        yield from remaining.split(EPISODE)
        return
    while len(remaining) > EPISODE:
        pivot, rest = remaining[0], remaining[1:]
        rivals = visual[pivot, rest].topk(EPISODE - 1, sorted=False).indices
        yield torch.cat((remaining[:1], rest[rivals]))
        keep = torch.ones(len(rest), dtype=torch.bool)
        keep[rivals] = False
        remaining = rest[keep]
    yield remaining


def loss_augmented_hinge(logits, labels, regret, objective):
    row = torch.arange(len(logits))
    augmented = logits + regret
    if objective == ASSIGNMENT:
        _, assignment = linear_sum_assignment(augmented.detach().numpy(), maximize=True)
        value = augmented[row, torch.from_numpy(assignment)]
    else:
        value = augmented.amax(dim=1)
    return value.mean() - logits[row, labels].mean()


def train_metric(projected, fit, utility, objective):
    rng = torch.Generator().manual_seed(SEED + 41)
    network = ResidualProjection(DIMENSION)
    _initialize(network, rng)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0015, weight_decay=0.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0002)
    equations, updates = [], 0
    examples = 0
    visual = (utility + utility.T) * 0.5
    for epoch in range(EPOCHS):
        history = []
        for prototype, query in projected.values():
            for batch in training_episodes(fit, visual, rng, clustered=bool(epoch % 2)):
                order = torch.randperm(len(batch), generator=rng)
                columns = batch[order]
                labels = order.argsort()
                raw = query[batch]
                masked = raw.masked_fill(torch.rand(raw.shape, generator=rng) < 0.06, 0)
                logits = HEAD_SCALE * (network(masked) @ network(prototype[columns]).T)
                gains = utility[columns[None], batch[:, None]]
                regret = ((utility[batch, batch][:, None] - gains) * REGRET_SCALE).clamp_min(0)
                pointwise = F.cross_entropy(logits, labels)
                assignment = loss_augmented_hinge(logits, labels, regret, objective)
                loss = (1 - HINGE_SHARE) * pointwise + HINGE_SHARE * assignment
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                history.append(torch.stack((pointwise.detach(), assignment.detach())))
                updates += 1
                examples += len(batch)
        scheduler.step()
        ce, hinge = torch.stack(history).mean(0).tolist()
        equations.append({"cross_entropy": ce, "regret_hinge": hinge})
        print(f"{objective} epoch {epoch + 1}/{EPOCHS}: cross entropy {ce:.4f}, margin {hinge:.4f}", flush=True)
    return network.eval(), {"loss": equations, "updates": updates, "single_query_examples": examples,
                           "parameters": sum(value.numel() for value in network.parameters())}


@torch.no_grad()
def compare_day(models, projected, ordering, logits, held, exact, utility, temperature):
    prototype, query = projected
    matrices = {V20: logits}
    for name, network in models.items():
        metric = network(query[ordering]) @ network(prototype).T
        matrices[name] = logits + MIX * standardized(metric)
    fresh = torch.isin(ordering, held)
    rows = torch.where(torch.isin(ordering, exact))[0]
    actions, _, incumbent = original_panel(logits[rows], temperature, utility)
    reports, for_actions = {}, {}
    for objective, values in matrices.items():
        _, assignment = linear_sum_assignment(values.numpy(), maximize=True)
        labels = torch.from_numpy(assignment)
        matching = labels == ordering
        matched = {"all": float(matching.float().mean()), "held": float(matching[fresh].float().mean())}
        for balanced in (False, True):
            name = (BASE_BALANCED if balanced else V20) if objective == V20 else f"{objective}_{'balanced' if balanced else 'row'}"
            posterior = glyph_posterior(values, temperature, balance=balanced)
            for_actions[name] = posterior[rows]
            rank = {"full_gallery": posterior_metrics(posterior, ordering)[0],
                    "held_glyphs_full_gallery": posterior_metrics(posterior[fresh], ordering[fresh])[0],
                    "hungarian_identity_diagnostic": matched}
            if balanced:
                rank["mean_class_mass_error"] = float((posterior.sum(0) - 1).abs().mean())
            reports[name] = rank
    choices = choose_images(NeuralRows(logits[rows], logits[rows], actions), for_actions, utility)
    choices = {name: choices[name] for name in for_actions}
    if not torch.equal(choices[V20], incumbent):
        raise ValueError("Only the posterior may change the original v20 fixed-seven choice.")
    return choices, ordering[rows], {"ranks": reports, "switched_vs_v20": {
        name: int((selected != incumbent).sum()) for name, selected in choices.items()
    }}


def comparisons(scores):
    comparisons = {}
    for inference in ("row", "balanced"):
        structured = f"{ASSIGNMENT}_{inference}"
        independent = f"{INDEPENDENT}_{inference}"
        comparisons[f"assignment_vs_independent_{inference}"] = paired_delta(scores[structured], scores[independent])
    for training in (INDEPENDENT, ASSIGNMENT):
        comparisons[f"{training}_posthoc_vs_row"] = paired_delta(scores[f"{training}_balanced"], scores[f"{training}_row"])
    return comparisons


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    cpu = torch.device("cpu")
    root = locate_data()
    images = load_images(root, cpu)
    masks = images >= 128
    utility, temperatures, _ = validated_utility(images, cpu)
    fitting, held, targets, previous, overlaps = new_cohort(len(images), cpu)
    grouped = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images), bin_ms=25)
    support, query, repeat_overlap = held_repeats(grouped, previous)
    del grouped
    if (any(block.shape[:2] != (len(images), 3) for block in support.values())
            or any(block.shape[:2] != (len(images), 1) for block in query.values())):
        raise ValueError("Task 3 proxy blocks must contain one real query and three independent supports per glyph.")
    report = {"mouse": MOUSE, "seed": SEED, "scoring_days": DAYS, "source_days": list(support),
              "held_ranking_glyphs_per_day": len(held), "scored_glyphs_per_day": PER_DAY,
              "fit_glyphs": len(fitting), "full_gallery": len(images), "repeats_per_exact_query": 1,
              "same_day_support_repeats": 3, "overlap": overlaps, "repeat_overlap": repeat_overlap,
              "episode_size": EPISODE, "epochs": EPOCHS, "margin_weight": HINGE_SHARE, "head_scale": HEAD_SCALE,
              "fixed_v20_candidates": SHORTLIST, "posterior_mixture": MIX, "utility_sha256": TABLE_SHA,
              "protocol": "The recording proxy has one real anonymous query per training glyph plus its separate "
              "three support repeats. Fit the same embedding with exactly the same episodes, budget, initial "
              "weights, cross entropy and shape regret from the fixed utility table. The row model finds its "
              "margin violations per query, and the structured model finds them under exact one-to-one Hungarian "
              "constraints. Alternate uniform and visually clustered class episodes. All rankings use the full "
              "recording and gallery. Compare row and post-hoc balanced inference factorially; all actual outputs "
              "use the original v20 seven candidates, anchor and utility. Full-gallery summaries include model-fit "
              "glyphs with disjoint repeated spikes; report independent glyph results separately. Score 192 fresh "
              "query/glyph cases with exact official OCF. No formal inputs are read.", "sessions": {}}
    publish(report, OUTPUT, started, "split_ready")

    projected = population_features(support, query, fitting)
    anchor = fit_neural(support, images, *contour_signatures(images), MOUSE, 211)
    rng = torch.Generator().manual_seed(SEED + 24)
    order = {day: torch.randperm(len(images), generator=rng) for day in DAYS}
    baseline = {day: v11_logits(anchor, query[day][order[day]], day, images) for day in DAYS}
    del anchor, support, query
    models, report["fit"] = {}, {}
    for mode in (INDEPENDENT, ASSIGNMENT):
        models[mode], report["fit"][mode] = train_metric(projected, fitting, utility, mode)
        publish(report, OUTPUT, started, f"{mode}_ready")
    first, second = map(report["fit"].get, (INDEPENDENT, ASSIGNMENT))
    for field in ("updates", "single_query_examples", "parameters"):
        if first[field] != second[field]:
            raise RuntimeError(f"Unmatched training budget for {field}.")
    scorer = official_scorer(root)

    results = {}
    for day in DAYS:
        choices, truth, metrics = compare_day(models, projected[day], order[day], baseline[day], held, targets[day],
                                              utility, temperatures[day]["temperature"])
        report["sessions"][day] = {"exact_glyph_indices": truth.tolist(), **metrics}
        publish(report, OUTPUT, started, f"{day}_assignments")
        guesses = torch.stack(tuple(choices.values()))
        exact, unique = unique_pair_ocf(guesses, truth.expand_as(guesses), masks, scorer, cpu, 96)
        results[day] = dict(zip(choices, exact, strict=True))
        report["sessions"][day].update(exact_pairs=unique, ocf=paired_summary(results[day]),
                                       paired_contrasts=comparisons(results[day]))
        publish(report, OUTPUT, started, f"{day}_official_ocf")
    pooled = {arm: torch.cat([daily[arm] for daily in results.values()]) for arm in next(iter(results.values()))}
    report.update(pooled_ocf=paired_summary(pooled), paired_contrasts=comparisons(pooled))
    publish(report, OUTPUT, started, "complete")
    print(json.dumps({"OCF": report["pooled_ocf"], "paired": report["paired_contrasts"]}), flush=True)


if __name__ == "__main__":
    main()
