"""Recheck the MICrONS and 1.4× Poisson atlas policies on fresh real glyphs."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import describe_atlas_candidates, pack_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_task1_bipartite_validation import image_proxy, recall, split_whole
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_ids_poisson_recheck import new_holdout as ids_poisson_holdout
from kaggle_task1_ids_retrieval import split_glyphs as ids_holdout
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE
from kaggle_task1_microns_open_atlas import ATLAS_CHUNK, HELD_COUNT, RIDGE, SEED as OPEN_SEED, fit_kernel_readout
from kaggle_task1_microns_open_atlas import outer_split as open_holdout
from kaggle_task1_microns_open_atlas import publish, stream_atlas
from kaggle_task1_microns_transfer import BLACK_LEVEL, FNN_VERSION, FRAMES, PUBLIC_ARCHIVE, SEED as CLOSED_SEED, STREAM, load_released_core
from kaggle_task1_poisson_bridge import fresh_split as bridge_holdout
from kaggle_task1_repeat_agreement import splits as agreement_holdout
from kaggle_task1_repeat_weight_validation import independent_split as weight_holdout
from kaggle_task1_v11_repeat_weight_transfer import (POISSON_AMPLIFICATION, decisions, matched_pool, normalized_sources,
                                                      pooled_scores, weighted_fusion)
from kaggle_task1_v13_repeat_weight_recheck import independent_outer as v13_weight_holdout
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank, summarize


SEED = 91801
STOCK = "v14"
V15 = "v15_poisson_1p4"
CORE = "microns_poisson_1p0"
CORE_14 = "microns_poisson_1p4"
ZERO = "no_direct_poisson"
OUTPUT = Path("/kaggle/working/task1_microns_repeat_recheck.json")


def fresh_cohort(count, device):
    cpu = torch.device("cpu")
    _, bipartite, _ = split_whole(count, 36467, cpu)
    _, ids = ids_holdout(count, cpu)
    closed_pool = np.setdiff1d(np.arange(count), np.union1d(bipartite.numpy(), ids.numpy()))
    closed = np.random.default_rng(CLOSED_SEED).choice(closed_pool, size=HELD_COUNT, replace=False)
    required = {"microns_closed_v2": closed, "microns_open_v1": open_holdout(count, OPEN_SEED, cpu)[1].numpy(),
                "ids_v1": ids.numpy(), "ids_poisson_recheck_v1": ids_poisson_holdout(count, cpu)[1].numpy(),
                "bipartite_v1": bipartite.numpy(), "poisson_bridge_v1": bridge_holdout(count, cpu)[1].numpy(),
                "v13_poisson_repeat_recheck": v13_weight_holdout(count, cpu)[1].numpy()}
    extra = {"poisson_6109": weight_holdout(count, 6109)[1].numpy(),
             "poisson_6947": weight_holdout(count, 6947, exclude_first_pilot=True)[1].numpy(),
             "poisson_agreement_18917": agreement_holdout(count, 18917, 144)[2].numpy()}
    excluded = np.unique(np.concatenate(tuple(required.values())))
    accepted, skipped = {}, {}
    for name, older in extra.items():
        union = np.union1d(excluded, older)
        if count - len(union) >= HELD_COUNT:
            accepted[name] = older
            excluded = union
        else:
            skipped[name] = older
    eligible = np.setdiff1d(np.arange(count), excluded)
    held = np.sort(np.random.default_rng(SEED).choice(eligible, size=HELD_COUNT, replace=False))
    fitting = np.setdiff1d(np.arange(count), held)
    overlap = {name: int(np.intersect1d(held, cohort).size) for name, cohort in {**required, **accepted, **skipped}.items()}
    if any(overlap[name] for name in {**required, **accepted}):
        raise RuntimeError("A holdout overlaps an excluded cohort.")
    independence = {"eligible": len(eligible), "excluded_glyphs": len(excluded), "optional_exclusions_skipped": list(skipped),
                    "overlap_by_previous_cohort": overlap}
    return torch.as_tensor(fitting, device=device), torch.as_tensor(held, device=device), independence


@torch.no_grad()
def single_day_scores(model, stock, query, core_rates):
    original_sources = normalized_sources(stock)
    released = {**stock, **weighted_fusion(model, original_sources, POISSON_AMPLIFICATION)}
    core_poisson = {f"poisson_{name}": count_template_logits(query, core_rates, name) for name in ("rate", "timed")}
    replacement = {**stock, **core_poisson}
    new_sources = normalized_sources(replacement)
    microns = {**replacement, **weighted_fusion(model, new_sources, 1.0)}
    microns_amplified = {**replacement, **weighted_fusion(model, new_sources, POISSON_AMPLIFICATION)}
    zero = zero_poisson(model, stock, original_sources)
    return {STOCK: stock, V15: released, CORE: microns, CORE_14: microns_amplified, ZERO: zero}, original_sources, new_sources


@torch.no_grad()
def zero_poisson(model, score, sources):
    direct = {name: torch.full_like(score[name], -torch.inf) for name in ("poisson_rate", "poisson_timed")}
    return {**score, **direct, **weighted_fusion(model, sources, 0.0)}


@torch.no_grad()
def pooled_maps(model, stock, core):
    original, released = pooled_scores(model, stock)
    microns, amplified = pooled_scores(model, core)
    return {STOCK: original, V15: released, CORE: microns, CORE_14: amplified, ZERO: zero_poisson(model, original, stock)}


def accumulate(total, sources, index):
    for key, values in sources.items():
        ordered = values[index]
        if key in total:
            total[key].add_(ordered)
        else:
            total[key] = ordered


@torch.no_grad()
def choose(policies, ranker, proxy, trusted):
    stock = pack_atlas_candidates(policies[STOCK], NOMINATIONS, baseline="v8")
    predictions = {STOCK: ranker.select(stock, anchor_ocf=ANCHOR)}
    details = {STOCK: {"recall": recall(policies[STOCK], stock, proxy, trusted)},
               "mean_v14_distinct_candidates": float(stock.valid.sum(1).float().mean()), "trusted_proxy_glyphs": int(trusted.sum())}
    for arm, ranking in policies.items():
        if arm == STOCK:
            continue
        proposed = pack_atlas_candidates(ranking, NOMINATIONS, baseline=stock.baseline)
        indices, support, novel = matched_pool(stock, proposed)
        balanced = describe_atlas_candidates(ranking, NOMINATIONS, stock.baseline, stock.rank_depth, indices, support)
        predictions[arm] = ranker.select(balanced, anchor_ocf=ANCHOR)
        details[arm] = {"recall": recall(ranking, balanced, proxy, trusted), "same_distinct_budget_for_every_row": True,
                        "mean_novel_candidates": float(novel.float().mean()),
                        "changed_predictions": int((predictions[arm] != predictions[STOCK]).sum())}
    release, _ = decisions(policies[STOCK], policies[V15], ranker)
    if not torch.equal(release["v11"], predictions[STOCK]) or not torch.equal(release["poisson_same_budget"], predictions[V15]):
        raise RuntimeError("The matched v15 policy differs from its original packaging decisions.")
    details["v15_exact_policy_match"] = True
    return predictions, details


def measurements(actual, baseline, contrast):
    result = {arm: summarize(values, actual[baseline]) for arm, values in actual.items()}
    result.update({name: summarize(actual[left], actual[right]) for name, (left, right) in contrast.items()})
    return result


def compare(actual):
    return measurements(actual, STOCK, {"microns_1p0_vs_v15": (CORE, V15), "microns_1p4_vs_v15": (CORE_14, V15),
                                       "microns_1p4_vs_1p0": (CORE_14, CORE)})


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the repeat-budget experiment on one private Kaggle GPU.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held, independence = fresh_cohort(len(images), device)
    groups = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if tuple(groups) != SESSIONS or any(source.shape[1] != 4 for source in groups.values()):
        raise ValueError("Expected four days of measured four-repeat labeled glyphs.")
    training = {day: counts[fit] for day, counts in groups.items()}
    queries = {day: counts[held] for day, counts in groups.items()}
    del groups
    labeled, references = images[fit], images[held]
    rates = torch.stack([counts.mean(1) for counts in training.values()])
    feature_count = rates.shape[-1]
    stabilized = (rates + 0.1).sqrt().transpose(0, 1).flatten(1)
    del rates
    cores, archive_sha = load_released_core()
    head = cores["published_core"]
    del cores
    report = {"held_seed": SEED, "seed_model": MODEL_SEED, "mouse": "mice1", "fit_glyphs": len(fit), "held_glyphs": len(held),
              "cohort_independence": independence, "ocf_glyphs_per_condition": len(held), "anchor_ocf": ANCHOR,
              "original_nominations": NOMINATIONS, "poisson_amplification": POISSON_AMPLIFICATION, "ridge": RIDGE,
              "microns": {"FNN_version": FNN_VERSION, "archive_url": PUBLIC_ARCHIVE, "archive_sha256": archive_sha,
                          "dtype": "float32", "black_level": BLACK_LEVEL, "stream": STREAM, "frames": list(FRAMES)},
              "repeat_conditions": {"per_day": 4, "four_days_known_correspondence": 16},
              "protocol": "An independent whole-glyph holdout is excluded from every fit and all OCF calibration. Use fixed "
                          "four-fold v14 ranker, uncertainty, contours and rendered atlas. All alternatives share the exact "
                          "initial v14 distinct candidate count in each row. The v15 setting matches the original packager. "
                          "The zero control masks the two direct-rate score features and sets the direct Poisson mixture to zero while "
                          "retaining the stock contours and uncertainty. Pooled 16 averages corresponding held glyph's four "
                          "independently standardized daily score matrices as in the earlier repeat-transfer protocol and "
                          "adds the character prior once. This higher-repeat proxy uses known correspondence over four days. "
                          "There were no same-day 20-repeat trials or formal queries. The trusted atlas proxy and actual OCF "
                          "use held references only to measure fixed outcomes.", "recordings": {}}
    publish(report, OUTPUT, started, "source_verified")

    scorer = official_scorer(root)
    calibration = split_calibration(len(fit), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    _, noise, nested, columns = nested_ranker(training, labeled, calibration, scorer, MODEL_SEED)
    ranker, folds = fit_crossfit_head(training, labeled, scorer, columns)
    model = fit_v8(training, labeled, MODEL_SEED)
    del training
    if not torch.equal(columns, model.codepoints.cpu()):
        raise ValueError("The outer and four calibration atlases use different public columns.")
    report["v14_fit"] = {"model": model.report, "nested": nested, "folds": folds, "ranker": ranker.diagnostics,
                         "atlas_glyphs": len(model.atlas.images)}
    publish(report, OUTPUT, started, "v14_ready")

    readout = fit_kernel_readout(head, labeled, stabilized)
    rate_bank, milestones = stream_atlas(head, readout, model.atlas.images, feature_count, ATLAS_CHUNK)
    del head, readout, stabilized, labeled
    report["microns"]["streaming_progress"] = milestones
    report["microns"]["atlas_rate_shape"] = list(rate_bank.shape)
    report["microns"]["kernel_solves"] = 1
    publish(report, OUTPUT, started, "published_core_atlas_ready")

    proxy, trusted = image_proxy(model, references)
    exact = PairedOCF(scorer, rendered_bank(model), (references >= 128).cpu().numpy(), device, 12)
    rng = np.random.default_rng(SEED + 1)
    total_original, total_core = {}, {}
    day_scores = {}
    for day_index, (day, query) in enumerate(queries.items()):
        order = torch.as_tensor(rng.permutation(len(held)), device=device)
        inverse = order.argsort()
        shuffled = query[order]
        stock = atlas_scores(model, noise[day], shuffled, day)
        policies, original_sources, new_sources = single_day_scores(model, stock, shuffled, rate_bank[day_index].to(device))
        predictions, detail = choose(policies, ranker, proxy[order], trusted[order])
        scores = {arm: exact.evaluate(picks[inverse, None])[:, 0] for arm, picks in predictions.items()}
        for arm, observed in scores.items():
            day_scores.setdefault(arm, []).append(observed)
        detail["paired_exact"] = compare({arm: [observed] for arm, observed in scores.items()})
        report["recordings"][day] = detail
        report["same_day_four_repeats_clustered_by_glyph"] = compare(day_scores)
        accumulate(total_original, original_sources, inverse)
        accumulate(total_core, new_sources, inverse)
        del stock, policies, original_sources, new_sources
        publish(report, OUTPUT, started, day)

    stock = {key: value.div_(len(SESSIONS)) for key, value in total_original.items()}
    core = {key: value.div_(len(SESSIONS)) for key, value in total_core.items()}
    policies = pooled_maps(model, stock, core)
    choices, detail = choose(policies, ranker, proxy, trusted)
    scores = {arm: [exact.evaluate(picks[:, None])[:, 0]] for arm, picks in choices.items()}
    detail["paired_exact"] = compare(scores)
    report["pooled16_known_correspondence_proxy"] = detail
    report["pooled16_increment_over_same_day_four_repeat_gains"] = {
        arm: summarize([scores[arm][0] - scores[STOCK][0]], [(np.stack(day_scores[arm]) - np.stack(day_scores[STOCK])).mean(0)])
        for arm in scores if arm != STOCK}
    report["unique_exact_pairs"] = len(exact.values)
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
