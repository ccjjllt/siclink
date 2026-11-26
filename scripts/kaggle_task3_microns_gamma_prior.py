"""Test whether a frozen mouse V1 core improves same-day single-trial retrieval."""

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_encoding import _evoked_counts
from glyph_gamma_template import GammaCountTemplates, gamma_template_logits, gamma_templates, paired_count_log_probability
from glyph_kernel_encoding import descriptors, image_kernel
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task1_bipartite_validation import split_whole
from kaggle_task1_ids_retrieval import split_glyphs as ids_split
from kaggle_task1_microns_transfer import (BLACK_LEVEL, FNN_SOURCE, FNN_VERSION, FRAMES, PUBLIC_ARCHIVE,
                                          SEED as CLOSED_SEED, core_descriptors, load_released_core,
                                          normalized_views, template_rates)
from kaggle_task3_measured_metric_validation import paired
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_part_discovery import standardize_rows
from kaggle_task3_rankband_ocf_validation import neural_rows
from kaggle_task3_repeat_posterior import fit_models


SEED = 87149
REPEAT_SEED = 87163
FOLDS = 4
CALIBRATION = 128
OUTER = 320
OCF = 128
EXPOSURE = 3.0
RATE_CENTER_WEIGHT = 0.02
ANCHOR_WEIGHT = 0.75
PUBLISHED = "published_core"
ACTUAL = "v14_top7"
OUTPUT = Path("/kaggle/working/task3_microns_gamma_prior.json")


def cohorts(size, device):
    _, bipartite, _ = split_whole(size, 36467, torch.device("cpu"))
    _, ids = ids_split(size, torch.device("cpu"))
    previous_population = np.setdiff1d(np.arange(size), np.union1d(bipartite.numpy(), ids.numpy()))
    previous = np.random.default_rng(CLOSED_SEED).choice(previous_population, size=320, replace=False)
    rng = np.random.default_rng(SEED)
    eligible = rng.permutation(np.setdiff1d(np.arange(size), previous))
    if len(eligible) < CALIBRATION + OUTER or OCF > OUTER:
        raise RuntimeError("The independent calibration, outer, and OCF cohorts must fit the labeled gallery.")
    calibration, outer = eligible[:CALIBRATION], eligible[CALIBRATION:CALIBRATION + OUTER]
    scored = rng.choice(OUTER, OCF, replace=False)
    reserved = np.concatenate((calibration, outer))
    fit = rng.permutation(np.setdiff1d(np.arange(size), reserved))
    folds = []
    for train_bucket, reserve_bucket in zip(np.array_split(fit, FOLDS), np.array_split(reserved, FOLDS), strict=True):
        sources = np.setdiff1d(fit, train_bucket)
        targets = np.concatenate((train_bucket, reserve_bucket))
        if np.intersect1d(sources, targets).size or np.intersect1d(sources, reserved).size:
            raise RuntimeError("A gallery class or reserved class entered its prior fit.")
        folds.append(tuple(torch.as_tensor(rows, device=device) for rows in (sources, targets)))
    if not torch.equal(torch.cat([targets for _, targets in folds]).sort().values, torch.arange(size, device=device)):
        raise RuntimeError("Every gallery class must receive exactly one out-of-fold prior.")
    calibration, outer, scored = (torch.as_tensor(rows, device=device) for rows in (calibration, outer, scored))
    return folds, calibration, outer, scored


@torch.no_grad()
def out_of_fold_prior(views, folds, measured):
    days, classes, features = measured.shape
    stabilized = (measured + 0.1).sqrt().transpose(0, 1).reshape(classes, -1)
    prior = torch.empty_like(measured)
    for train, gallery in folds:
        vector = normalized_views(views, train)
        basis = vector[train]
        rates = template_rates(image_kernel(basis, basis), image_kernel(vector[gallery], basis), stabilized[train])
        rates = rates.reshape(len(gallery), days, features).transpose(0, 1)
        center = measured[:, train].mean(dim=1, keepdim=True)
        prior[:, gallery] = rates.lerp(center, RATE_CENTER_WEIGHT).clamp_min_(1e-5)
    if not bool(prior.isfinite().all()):
        raise RuntimeError("A cross-fitted timed count prior is nonfinite.")
    return prior


def pick(scores, shortlist):
    return shortlist.gather(1, scores.gather(1, shortlist).argmax(dim=1, keepdim=True))[:, 0]


def nominate(scores, keep):
    extra = scores.clone().scatter_(1, keep, -torch.inf).topk(3, dim=1).indices
    return torch.cat((keep, extra), dim=1)


def identity(scores, labels):
    hits = scores.topk(32, dim=1).indices == labels[:, None]
    return {f"top{depth}": float(hits[:, :depth].any(dim=1).float().mean()) for depth in (1, 16, 32)}


@torch.no_grad()
def evaluate_day(panel, query, local, prior, check, labels, sample, masks, scorer):
    if panel.action.shape != (len(labels), 7) or query.shape[1] != 1:
        raise RuntimeError("Expected the exact mice1 top-seven policy and one held trial per query.")
    anchor = panel.mixed
    shortlist = panel.action
    position = anchor.gather(1, shortlist).argsort(dim=1, descending=True, stable=True)[:, :4]
    keep = shortlist.gather(1, position)
    new_anchor = nominate(anchor, keep)
    choices = {ACTUAL: pick(anchor, shortlist), "v14_renominated7": pick(anchor, new_anchor)}
    ranking = {ACTUAL: identity(anchor, labels)}
    pool_recall = {ACTUAL: float((shortlist == labels[:, None]).any(dim=1).float().mean()),
                   "v14_renominated7": float((new_anchor == labels[:, None]).any(dim=1).float().mean())}
    y = _evoked_counts(query[:, 0], "timed")
    local_rates = _evoked_counts(local.mean(1), "timed").clamp_min_(1e-5)
    evidence = {"same_day_poisson": y @ local_rates.log().T - local_rates.sum(dim=1)}
    calibration = {}
    check_ids, check_trials = check
    for name, rates in prior.items():
        templates = gamma_templates(rates, local, EXPOSURE)
        poisson, gamma = gamma_template_logits(query[:, 0], templates)
        evidence[name] = gamma
        if name == PUBLISHED:
            evidence["published_matched_poisson"] = poisson
        subset = GammaCountTemplates(templates.shape[check_ids], templates.rate)
        check_gamma, check_poisson = paired_count_log_probability(check_trials[:, 0], subset)
        calibration[name] = {"true_count_gamma_log_p": float(check_gamma.mean()),
                             "true_count_matched_poisson_log_p": float(check_poisson.mean())}

    for name, logits in evidence.items():
        blended = ANCHOR_WEIGHT * anchor + (1 - ANCHOR_WEIGHT) * standardize_rows(logits)
        candidates = nominate(blended, keep)
        ranking[f"{name}_standalone"] = identity(logits, labels)
        ranking[f"{name}_with_v14"] = identity(blended, labels)
        pool_recall[f"{name}_renominated7"] = float((candidates == labels[:, None]).any(dim=1).float().mean())
        choices[f"{name}_fixed7"] = pick(blended, shortlist)
        choices[f"{name}_renominated7"] = pick(blended, candidates)

    predictions = torch.stack([choice[sample] for choice in choices.values()])
    scores, pairs = unique_pair_ocf(predictions, labels[sample].expand_as(predictions), masks, scorer, masks.device, 384)
    return dict(zip(choices, scores.cpu().numpy(), strict=True)), {
        "full_gallery_identity": ranking, "exact_candidate_recall": pool_recall,
        "independent_calibration_glyph_count_log_p": calibration, "unique_exact_pairs": pairs,
        "changed_outer_choices_vs_v14": {name: int((guess != choices[ACTUAL]).sum()) for name, guess in choices.items()},
    }


def summarize(values):
    score = {arm: paired(values[arm], values[ACTUAL]) for arm in values}
    for selection in ("fixed7", "renominated7"):
        arm = values[f"{PUBLISHED}_{selection}"]
        for control in ("contour", "random_core", "same_day_poisson", "published_matched_poisson"):
            score[f"{PUBLISHED}_vs_{control}_{selection}"] = paired(arm, values[f"{control}_{selection}"])
        if selection == "renominated7":
            score[f"{PUBLISHED}_vs_v14_{selection}"] = paired(arm, values["v14_renominated7"])
    return score


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this experiment on Kaggle T4 using only labeled competition training input.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    args = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128, query_batch=64)
    root = locate_data()
    images = load_images(root, device)
    if len(images) != 2748:
        raise RuntimeError("Expected the full official 2,748-glyph training gallery.")
    masks = images >= 128
    scorer = official_scorer(root)
    folds, calibration, outer, scored = cohorts(len(images), device)
    recordings = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if len(recordings) != 4 or any(counts.shape[1] != 4 for counts in recordings.values()):
        raise RuntimeError("Expected four labeled mice1 recordings with four trials per glyph.")
    support, held = {}, {}
    for index, (day, counts) in enumerate(recordings.items()):
        support[day], held[day] = split_repeats(counts, REPEAT_SEED + index)
    del recordings, counts

    cores, sha256 = load_released_core()
    report = {"seed": SEED, "repeat_seed": REPEAT_SEED, "mouse": "mice1", "gallery_size": len(images),
              "prior_fold_sizes": [{"fit": len(fit), "gallery": len(gallery)} for fit, gallery in folds],
              "calibration_glyphs": len(calibration), "outer_glyphs": len(outer), "scored_glyphs_per_day": len(scored),
              "exposure": EXPOSURE, "rate_center_weight": RATE_CENTER_WEIGHT, "anchor_weight": ANCHOR_WEIGHT,
              "support_repeats": 3, "query_repeats": 1, "actual_test_accessed": False,
              "code": {"source": FNN_SOURCE, "version": FNN_VERSION, "license": "MIT"},
              "weights": {"source": PUBLIC_ARCHIVE, "sha256": sha256, "member": "params_core.pt",
                          "license": "CC BY 4.0 per the BossDB project listing"},
              "frozen_core_frames": FRAMES, "black_level": BLACK_LEVEL,
              "protocol": "Cross-fitted image-to-rate regressions omit the candidate glyph across all days and omit all "
                          "calibration and scoring glyphs from every fold. Candidate image descriptors are legal known-gallery "
                          "images. Only three disjoint same-day labeled trials update each Gamma prior. No trial elsewhere "
                          "enters its candidate update. Calibration counts are reported for diagnosis and select no parameters. "
                          "The production v14 mice1 selection rule refits on the same three legal repeats. All six score "
                          "sources use the same gallery and candidate images. Renomination keeps the four best incumbent "
                          "actions, adds three unused actions by the corresponding full-gallery score, and selects by that score. "
                          "Each recording's queries are independently shuffled; truth is used to evaluate ranks and image pairs.",
              "sessions": {}}

    def save(stage):
        report.update(stage=stage, seconds=time.monotonic() - started)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{stage} after {report['seconds']:.1f} s", flush=True)

    save("source_verified")
    measured = torch.stack([_evoked_counts(counts.mean(dim=1), "timed") for counts in support.values()])
    priors = {}
    for name, core in cores.items():
        if core is None:
            views = [torch.cat(view) for view in zip(*(descriptors(batch) for batch in images.split(128)), strict=True)]
        else:
            views = core_descriptors(core, images)
        priors[name] = out_of_fold_prior(views, folds, measured)
        del views
        save(f"{name}_prior_fitted")
    del measured, cores

    signature, energy = contour_signatures(images)
    baseline, report["v14_fit"] = fit_models(support, images, signature, energy, "mice1", args)
    save("v14_fitted")
    results = {}
    rng = np.random.default_rng(SEED + 7)
    for index, (day, query) in enumerate(held.items()):
        permutation = torch.as_tensor(rng.permutation(len(outer)), device=device)
        labels = outer[permutation]
        sample = permutation.argsort()[scored]
        if not torch.equal(labels[sample], outer[scored]):
            raise RuntimeError("The paired OCF rows must retain the same glyph order across recordings.")
        trials = query[labels]
        panel = neural_rows(baseline, trials, images, day, args)
        exact, diagnostics = evaluate_day(panel, trials, support[day], {name: rates[index] for name, rates in priors.items()},
                                          (calibration, query[calibration]), labels, sample, masks, scorer)
        for name, score in exact.items():
            results.setdefault(name, []).append(score)
        report["sessions"][day] = {"exact_ocf": summarize(exact), **diagnostics}
        report["summary_exact_ocf"] = summarize({name: np.stack(scores) for name, scores in results.items()})
        ranks = [record["full_gallery_identity"] for record in report["sessions"].values()]
        report["summary_full_gallery_identity"] = {
            arm: {name: float(np.mean([session[arm][name] for session in ranks])) for name in metrics}
            for arm, metrics in ranks[0].items()
        }
        save(f"{day}_scored")
    save("complete")


if __name__ == "__main__":
    main()
