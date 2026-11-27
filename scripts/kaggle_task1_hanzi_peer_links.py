"""Test whether frozen-Hanzi scores improve held Task 1 cross-day links."""

import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch.nn import functional as F

from glyph_bipartite_encoding import timing_view
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import build_font_atlas, standard_error, standardized
from kaggle_task1_joint_atlas_pairing import assignment, holdout as joint_holdout, split_trials
from kaggle_task1_microns_hanzi_recheck import fresh_cohort as direct_holdout
from kaggle_task1_native_hanzi import (OCR_CHECKSUM, OCR_REVISION, OCR_SOURCE, PC_COUNT, RIDGE, SEED as PILOT_SEED,
                                       feature_extractors, fit_response_ridge, microns_scope, split_glyphs as pilot_holdout)
from kaggle_task1_native_hanzi_open_atlas import image_coordinates, independent_split as atlas_holdout
from kaggle_task1_peer_transport import choose_split as transport_holdout
from kaggle_v8_metric_validation import SESSIONS


MOUSE = "mice1"
SEED = 222503
HOLDOUT, OCF_SAMPLE, TOP = 320, 64, 64
TEMPERATURE = 1.0
TRANSFER_PAIRS = (("session1", "session3"), ("session2", "session4"))
CONTOUR, OCR = "contour", "frozen_hanzi"
LINK_ARMS = (CONTOUR, OCR)
OWN2, OWN4, KNOWN = "own_two", "own_four", "known_link"
OUTPUT = Path("/kaggle/working/task1_hanzi_peer_links.json")


def fresh_cohort(size):
    device = torch.device("cpu")
    older = {"joint_atlas_pairing": joint_holdout(size, device)[1], "peer_transport": transport_holdout(size, device)[1],
             "native_hanzi_pilot": pilot_holdout(size, np.random.default_rng(PILOT_SEED))[1],
             "native_hanzi_full_atlas": atlas_holdout(size)[1], "mice1_v16_hanzi": direct_holdout(size)[1],
             "microns_repeat_recheck": microns_scope(size, device)[1]}
    excluded = np.unique(np.concatenate(tuple(older.values())))
    eligible = np.setdiff1d(np.arange(size), excluded)
    held = np.sort(np.random.default_rng(SEED).choice(eligible, HOLDOUT, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    return torch.from_numpy(fit), torch.from_numpy(held), {
        "eligible": len(eligible), "overlap_by_cohort": {name: int(np.intersect1d(held, rows).size) for name, rows in older.items()}}


def posterior(scores):
    values, columns = scores.topk(TOP, dim=1)
    mass = (values / TEMPERATURE).softmax(dim=1).sqrt().numpy().ravel()
    offsets = np.arange(0, (len(scores) + 1) * TOP, TOP)
    return csr_matrix((mass, columns.numpy().ravel(), offsets), shape=tuple(scores.shape))


def true_rank_recall(score, truth):
    value = score.gather(1, truth[:, None])
    better, tied = (score > value).sum(1), (score == value).sum(1)
    recall = {str(depth): float(((depth - better).clamp_min(0) / tied).clamp_max(1).mean()) for depth in (1, 5, 16)}
    return {"tie_adjusted_recall": recall, "positive_true_overlap_fraction": float((value > 0).float().mean())}


def paired(delta):
    return {"paired_delta": float(delta.mean()), "paired_glyph_se": standard_error(delta), "paired_glyphs": len(delta)}


def publish(report, started, phase):
    report.update(stage=phase, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": phase, "seconds": report["seconds"]}), flush=True)


@torch.inference_mode()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a Kaggle CPU Notebook with the labeled training input.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    rng = np.random.default_rng(SEED + 1)
    images = load_images(root, device)
    if len(images) != 2748:
        raise ValueError("Expected the 2,748 labeled training glyphs.")
    fit, held, independence = fresh_cohort(len(images))
    sample = torch.from_numpy(np.sort(np.random.default_rng(SEED + 2).choice(HOLDOUT, OCF_SAMPLE, replace=False)))
    groups = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(groups) != SESSIONS or any(counts.shape[:2] != (len(images), 4) for counts in groups.values()):
        raise ValueError("Expected four training days with four measured repeats per glyph.")
    queries, orders = split_trials(groups, held, rng)
    inverse = {day: rows.argsort() for day, rows in orders.items()}
    train_images = images[fit]
    atlas, provenance = build_font_atlas(train_images[rng.choice(len(fit), 128, replace=False)], device)
    if len(atlas.images) != 27_550:
        raise ValueError("Both representations must use the complete 27,550-column public atlas.")
    report = {"mouse": MOUSE, "seed": SEED, "fit_glyphs": len(fit), "held_glyphs": len(held),
              "held_gallery_indices": held.tolist(), "cohort_independence": independence, "ocf_glyphs": OCF_SAMPLE,
              "ocf_sample_gallery_indices": held[sample].tolist(), "atlas": provenance, "atlas_columns": len(atlas.images),
              "matching_candidates": TOP, "posterior_temperature": TEMPERATURE, "pca_components": PC_COUNT,
              "ridge_per_glyph": RIDGE, "formal_inputs_read": False, "transfer_pairs": TRANSFER_PAIRS,
              "ocr": {"source": OCR_SOURCE, "revision": OCR_REVISION, "sha256": OCR_CHECKSUM},
              "protocol": "Each held day and each held glyph's four repeats are independently shuffled. Fit contour and frozen "
                          "Hanzi neural score maps with the same 128-dimensional pipeline outside every held glyph. Match two real "
                          "repeats via top-64 posterior Bhattacharyya overlap and the same one-to-one solver for both arms. "
                          "Freeze the decoder family to Hanzi for all transferred images. Average two standardized score maps "
                          "from the complementary repeats in each linked day, exchange the halves and repeat. Own-four uses four "
                          "real repeats at one day. Held identity is used only for link recall, oracle transfer and OCF pairing."}
    publish(report, started, "atlas_ready")
    image_codes = image_coordinates(train_images, atlas.images, {OCR: feature_extractors()[OCR]})
    marginal, half_ocr, own_four = {}, {}, {}
    for day, counts in groups.items():
        observed = timing_view(counts[fit].unflatten(-1, (-1, 10))).mean(1)
        ridge, center, scale = fit_response_ridge(observed)
        timing = timing_view(queries[day].unflatten(-1, (-1, 10)))
        blocks = timing.split(2, dim=1)
        design = tuple((half.mean(1) - center) / scale for half in blocks)
        marginal[day] = {}
        for arm, (fitting, prototypes) in image_codes.items():
            projection = ridge @ fitting
            scores = tuple(standardized(F.normalize(half @ projection) @ prototypes) for half in design)
            marginal[day][arm] = tuple(posterior(matrix) for matrix in scores)
            if arm == OCR:
                half_ocr[day] = scores
                own = (timing.mean(1) - center) / scale
                own_four[day] = (F.normalize(own @ projection) @ prototypes).argmax(1)
    del queries, groups, image_codes
    publish(report, started, "day_scores_ready")

    rows = torch.arange(HOLDOUT)
    link_stats, true_links, learned = {}, {}, {}
    correct = {arm: [] for arm in LINK_ARMS}
    for left_index, left in enumerate(SESSIONS):
        for right in SESSIONS[left_index + 1:]:
            pair = (left, right)
            truth = inverse[right][orders[left]]
            label = f"{left}-{right}"
            true_links[pair] = truth
            link_stats[label], learned[pair] = [], []
            for half in range(2):
                detail, links = {}, {}
                for arm in LINK_ARMS:
                    overlap = marginal[left][arm][half] @ marginal[right][arm][half].T
                    score = torch.from_numpy(overlap.toarray())
                    links[arm] = assignment(score)
                    hit = links[arm] == truth
                    correct[arm].append(hit[inverse[left]].numpy())
                    detail[arm] = {"true_one_to_one_recall": float(hit.float().mean()),
                                   "zero_overlap_selected_fraction": float((score[rows, links[arm]] == 0).float().mean()),
                                   "left_to_right": true_rank_recall(score, truth),
                                   "right_to_left": true_rank_recall(score.T, truth.argsort())}
                link_stats[label].append(detail)
                learned[pair].append(links)
    report["matching_halves"] = link_stats
    glyph_match = {arm: np.stack(measured).mean(0) for arm, measured in correct.items()}
    report["true_one_to_one_recall"] = {**{arm: float(values.mean()) for arm, values in glyph_match.items()},
                                       "hanzi_vs_contour": paired(glyph_match[OCR] - glyph_match[CONTOUR])}
    del marginal
    publish(report, started, "links_ready")

    score = PairedOCF(official_scorer(root), atlas.images, (images[held[sample]] >= 128).numpy(), device, 12)
    exact = {arm: [] for arm in (*LINK_ARMS, OWN2, OWN4, KNOWN)}
    day_scores = {day: {arm: [] for arm in exact} for day in SESSIONS}
    for left, right in TRANSFER_PAIRS:
        pair = (left, right)
        for half, links in enumerate(learned[pair]):
            selected = {day: half_ocr[day][1 - half] for day in pair}
            choices = {OWN2: {day: selected[day].argmax(1) for day in pair}, OWN4: own_four}
            for arm, partner in {**links, KNOWN: true_links[pair]}.items():
                both = ((selected[left] + selected[right][partner]) / 2).argmax(1)
                choices[arm] = {left: both, right: both[partner.argsort()]}
            for arm, picks in choices.items():
                for day in pair:
                    selected_sample = picks[day][inverse[day][sample]]
                    measured = score.evaluate(selected_sample[:, None])[:, 0]
                    exact[arm].append(measured)
                    day_scores[day][arm].append(measured)
        report[f"{left}-{right}_exact_pairs_so_far"] = len(score.values)
        publish(report, started, f"{left}_{right}_scored")

    glyph_scores = {arm: np.stack(measured).mean(0) for arm, measured in exact.items()}
    contrasts = ((OCR, CONTOUR), (OCR, OWN4), (CONTOUR, OWN4), (KNOWN, OWN4), (OCR, OWN2))
    report["exact_ocf"] = {arm: float(glyphs.mean()) for arm, glyphs in glyph_scores.items()}
    report["paired_exact_ocf"] = {f"{left}_vs_{right}": paired(glyph_scores[left] - glyph_scores[right]) for left, right in contrasts}
    report["exact_ocf_by_day_and_matching_half"] = {day: {arm: np.stack(panel).mean(1).tolist() for arm, panel in arms.items()}
                                                     for day, arms in day_scores.items()}
    report["unique_exact_pairs_scored"] = len(score.values)
    publish(report, started, "complete")


if __name__ == "__main__":
    main()
