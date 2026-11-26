"""Check whether fixed public Han repertoires cover whole held training glyphs.

This image-only experiment measures visual evidence for repertoire inclusion.
Public-codepoint confidence is calibrated on a second public font. Competition
codepoints are unavailable, so image matches provide a proxy and can abstain.
Only aggregate counts are saved. No spike or official test files are read.
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import torch
from torch.nn import functional as F

from glyph_atlas_prior import (PINLU_COUNT, UNIHAN_LICENSE, UNIHAN_SOURCE, UNIHAN_VERSION, _aligned_overlap,
                               _usage_log_enrichment, seen_glyph_evidence)
from glyph_data import load_images, locate_data
from kaggle_font_family_coverage import BASE, fit_bank
from kaggle_gallery_validation import FontAtlas, image_descriptors
from kaggle_lexical_support_validation import fit_density, morphology
from kaggle_task1_repeat_weight_validation import independent_split


OUTER_GLYPHS = 320
INNER_GLYPHS = 256
DEPTH = 24
GB_STANDARD = "https://std.samr.gov.cn/gb/search/gbDetailed?id=71F772D79E19D3A7E05397BE0A0AB82A&review=true"
TGH_STANDARD = "https://www.moe.gov.cn/jyb_xwfb/moe_2082/siju/yywzxxgls/ztzl_yywzgfbz/201306/t20130601_186002.html"
UNIHAN_DOCS = "https://www.unicode.org/reports/tr38/tr38-39.html#kTGH"


def standard_repertoires(codepoints):
    gb = set()
    for row in range(16, 56):
        for cell in range(1, 95):
            try:
                letter = bytes((0xA0 + row, 0xA0 + cell)).decode("gb2312")
            except UnicodeDecodeError:
                continue
            gb.add(ord(letter))
    if len(gb) != 3755:
        raise ValueError("GB2312 Level 1 should contain 3,755 codepoints.")

    with urlopen(UNIHAN_SOURCE, timeout=90) as response:
        payload = response.read()
    tgh, pinlu, unicode_gb = {}, {}, set()
    with ZipFile(BytesIO(payload)) as archive:
        for name in archive.namelist():
            if not name.endswith(".txt"):
                continue
            with archive.open(name) as records:
                for line in records:
                    if not line.startswith(b"U+"):
                        continue
                    value, field, detail = line.decode("utf-8").strip().split("\t", 2)
                    codepoint = int(value[2:], 16)
                    if field == "kTGH" and detail.startswith("2013:"):
                        tgh[codepoint] = int(detail.partition(":")[2])
                    elif field == "kHanyuPinlu":
                        pinlu[codepoint] = sum(int(count) for count in PINLU_COUNT.findall(detail))
                    elif field == "kGB0" and 16 <= int(detail[:2]) <= 55:
                        unicode_gb.add(codepoint)
    tgh_first = {codepoint for codepoint, index in tgh.items() if 1 <= index <= 3500}
    if len(tgh_first) != 3500 or len(tgh) != 8105:
        raise ValueError("Pinned Unihan kTGH should contain all 8,105 published ideographs.")
    full = {"gb2312_level1": gb, "tgh_level1": tgh_first, "gb2312_tgh_intersection": gb & tgh_first}
    codes = codepoints.tolist()
    tiers = torch.tensor([0 if not (index := tgh.get(codepoint)) else 1 + (index > 3500) + (index > 6500)
                          for codepoint in codes])
    frequency = torch.tensor([pinlu.get(codepoint, 0) for codepoint in codes], dtype=torch.float32)
    present = frequency > 0
    edges = torch.quantile(frequency[present], torch.tensor([1.0 / 3, 2.0 / 3]))
    frequency_bins = torch.zeros_like(tiers)
    frequency_bins[present] = torch.bucketize(frequency[present], edges) + 1
    masks = {name: torch.tensor([codepoint in members for codepoint in codes]) for name, members in full.items()}
    provenance = {"standard_urls": {"gb2312": GB_STANDARD, "tgh": TGH_STANDARD},
                  "gb2312_method": "Python gb2312 codec, EUC row-cell positions 16–55.",
                  "gb2312_codec_documentation": "https://docs.python.org/3/library/codecs.html#standard-encodings",
                  "gb2312_codec_license": "https://docs.python.org/3/license.html",
                  "gb2312_symmetric_difference_vs_unihan_kGB0": len(gb ^ unicode_gb),
                  "unihan": {"version": UNIHAN_VERSION, "source": UNIHAN_SOURCE, "sha256": sha256(payload).hexdigest(),
                             "license": "Unicode License V3", "license_url": UNIHAN_LICENSE,
                             "documentation": UNIHAN_DOCS},
                  "public_codepoint_counts": {name: len(members) for name, members in full.items()},
                  "atlas_codepoint_counts": {name: int(mask.sum()) for name, mask in masks.items()}}
    return masks, tiers, frequency_bins, provenance


def independent_cohorts(count, seed, *, exclude_coverage_seed=None):
    random = np.random.default_rng(seed)
    labels = np.arange(count)
    previous = np.random.default_rng(613).permutation(count)[:OUTER_GLYPHS]
    other_lexical = np.random.default_rng(1283).permutation(count)
    other_lexical = other_lexical[~np.isin(other_lexical, previous)][:OUTER_GLYPHS]
    _, weight_pilot, _ = independent_split(count, 6109)
    _, weight_recheck, _ = independent_split(count, 6947, exclude_first_pilot=True)
    excluded = np.unique(np.concatenate((previous, other_lexical, weight_pilot.numpy(), weight_recheck.numpy())))
    eligible = np.setdiff1d(labels, excluded)
    if exclude_coverage_seed is not None:
        prior_coverage = np.random.default_rng(exclude_coverage_seed).permutation(eligible)[:OUTER_GLYPHS]
        eligible = np.setdiff1d(eligible, prior_coverage)
        excluded = np.union1d(excluded, prior_coverage)
    if len(eligible) < OUTER_GLYPHS:
        raise ValueError("Prior coverage and weighting experiments leave too few fresh glyphs.")
    held = random.permutation(eligible)[:OUTER_GLYPHS]
    fit = random.permutation(np.setdiff1d(labels, held))
    return torch.as_tensor(fit[INNER_GLYPHS:]), torch.as_tensor(fit[:INNER_GLYPHS]), torch.as_tensor(held), len(excluded)


@dataclass
class Matches:
    candidates: torch.Tensor
    scores: torch.Tensor
    agree: torch.Tensor
    assignments: torch.Tensor

    @property
    def closest(self):
        return self.candidates[:, 0]

    def confident(self, quality, margin):
        return (self.agree & (self.assignments == self.closest) & (self.scores[:, 0] >= quality)
                & (self.scores[:, 0] - self.scores[:, 1] >= margin))


def shape_matches(atlas, images):
    all_candidates, all_scores, agreements = [], [], []
    for batch in images.split(32):
        binary = batch >= 128
        cosine, candidates = (image_descriptors(binary.to(torch.uint8) * 255) @ atlas.contours.T).topk(DEPTH, dim=1)
        overlap = _aligned_overlap(binary[:, None], atlas.images[candidates] >= 128)
        combined, order = (0.65 * cosine + 0.35 * overlap).sort(dim=1, descending=True)
        all_candidates.append(candidates.gather(1, order))
        all_scores.append(combined)
        agreements.append((order[:, 0] == 0) & (order[:, 0] == overlap.argmax(dim=1)))

    candidates, scores, agree = torch.cat(all_candidates), torch.cat(all_scores), torch.cat(agreements)
    size, n_codes = len(scores), len(atlas.images)
    rows = np.arange(size)
    repeated = np.repeat(rows, DEPTH)
    # One private abstention slot per image keeps the sparse assignment feasible.
    costs = np.concatenate(((2 - scores).numpy().ravel(), np.full(size, 2 - 0.50)))
    edges = coo_matrix((costs, (np.concatenate((repeated, rows)),
                               np.concatenate((candidates.numpy().ravel(), n_codes + rows)))),
                       shape=(size, n_codes + size)).tocsr()
    permutation, chosen = min_weight_full_bipartite_matching(edges)
    assigned = torch.full((size,), -1, dtype=torch.long)
    matched = chosen < n_codes
    assigned[permutation[matched]] = torch.from_numpy(chosen[matched]).long()
    return Matches(candidates, scores, agree, assigned)


def wilson_lower(correct, total):
    if total == 0:
        return 0.0
    probability = correct / total
    z = 1.6448536269514722
    center = probability + z**2 / (2 * total)
    radius = z * np.sqrt(probability * (1 - probability) / total + z**2 / (4 * total**2))
    return float((center - radius) / (1 + z**2 / total))


def public_precision(matches, actual_codes, atlas_codes, thresholds):
    accepted = matches.confident(*thresholds)
    correct = (atlas_codes[matches.closest] == actual_codes) & accepted
    size, hit = int(accepted.sum()), int(correct.sum())
    return {"matched": size, "correct": hit, "sample": len(actual_codes),
            "precision": hit / size if size else None, "one_sided_95_lower": wilson_lower(hit, size)}


def calibrate(atlas, atlas_codes, core, random, seed):
    public_font, public_codes, provenance = fit_bank("simxihei", core, seed + 1)
    common = torch.where(torch.isin(public_codes, atlas_codes))[0]
    if len(common) < 2048:
        raise ValueError("The independent public face needs at least 2,048 shared codepoints.")
    calibration, audit = common[random.permutation(len(common))[:2048]].split(1024)
    fits = shape_matches(atlas, public_font[calibration].to(torch.uint8) * 255)
    tests = shape_matches(atlas, public_font[audit].to(torch.uint8) * 255)
    grid = [(quality, margin) for quality in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
            for margin in (0.01, 0.02, 0.03, 0.05)]
    feasible = []
    for threshold in grid:
        report = public_precision(fits, public_codes[calibration], atlas_codes, threshold)
        if report["matched"] >= 64 and report["one_sided_95_lower"] >= 0.95:
            feasible.append((threshold, report))
    threshold, fit_report = max(feasible, key=lambda item: item[1]["matched"],
                                default=((float("inf"), float("inf")), None))
    audit_report = public_precision(tests, public_codes[audit], atlas_codes, threshold)
    passed = audit_report["matched"] >= 64 and audit_report["one_sided_95_lower"] >= 0.95
    chosen = threshold if passed else (float("inf"), float("inf"))
    return chosen, {"font": provenance, "fit": fit_report, "audit": audit_report, "audit_passed": passed,
                    "tested_thresholds": {"quality": threshold[0], "margin": threshold[1]} if feasible else None}


def proxy_coverage(matches, eligible, confident):
    closest = eligible[matches.closest]
    close = matches.scores >= matches.scores[:, :1] - 0.035
    reachable = (eligible[matches.candidates] & close).any(dim=1)
    matches_in = int((closest & confident).sum())
    unknown = int((~confident).sum())
    return {"pool_size": int(eligible.sum()), "nearest_shape_in_pool": float(closest.float().mean()),
            "near_shape_in_top24": float(reachable.float().mean()), "confident_in_pool": matches_in,
            "confident_outside_pool": int((~closest & confident).sum()), "confidence_abstentions": unknown,
            "confident_coverage": matches_in / int(confident.sum()) if confident.any() else None,
            "proxy_membership_interval_over_abstentions": [matches_in / len(closest),
                                                            (matches_in + unknown) / len(closest)]}


def fixed_budget(score, allowed, size):
    selected = torch.zeros_like(allowed)
    columns = torch.where(allowed)[0]
    selected[columns[score[columns].topk(size).indices]] = True
    return selected


def shuffled_pool(target, allowed, strata, random):
    result = torch.zeros_like(allowed)
    for group in strata.unique():
        rows = torch.where(allowed & (strata == group))[0]
        size = int(target[rows].sum())
        result[rows[random.choice(len(rows), size, replace=False)]] = True
    return result


def evaluate_pools(matches, confidence, raw_pools, removed, lexical, strata, seed):
    random = np.random.default_rng(seed)
    result = {}
    raw_eligible = torch.ones_like(removed)
    for name, raw_pool in raw_pools.items():
        result[name] = {}
        for condition, allowed in (("raw", raw_eligible), ("residual", ~removed)):
            pool = raw_pool & allowed
            size = int(pool.sum())
            baseline = fixed_budget(lexical, allowed, size)
            shuffled = shuffled_pool(pool, allowed, strata, random)
            report = {"public_list": proxy_coverage(matches, pool, confidence),
                      "morphology_unihan_same_size": proxy_coverage(matches, baseline, confidence),
                      "shuffled_same_size_and_strata": proxy_coverage(matches, shuffled, confidence)}
            if condition == "residual":
                lost = raw_pool[matches.closest] & removed[matches.closest]
                report["possible_false_exclusions"] = int(lost.sum())
                report["confident_possible_false_exclusions"] = int((lost & confidence).sum())
            result[name][condition] = report
    return result


def choose_pool(inner_metrics):
    supported = []
    for name, outcomes in inner_metrics.items():
        residual = outcomes["residual"]["public_list"]
        credible = residual["confident_coverage"]
        accepted = residual["confident_in_pool"] + residual["confident_outside_pool"]
        if accepted >= 64 and credible >= .95 and residual["near_shape_in_top24"] >= .95:
            supported.append((residual["pool_size"], name))
    return min(supported)[1] if supported else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7933)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_repertoire_coverage.json"))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on Kaggle CPU with competition input and public Internet downloads enabled.")
    torch.set_num_threads(4)
    started = time.monotonic()
    random = np.random.default_rng(args.seed + 2)
    images = load_images(locate_data(), torch.device("cpu"))
    core, inner, held, previous = independent_cohorts(len(images), args.seed)
    print(f"Fit common render geometry and style on {len(core)} training glyphs", flush=True)
    bank, codes, font = fit_bank(BASE, images[core], args.seed)
    bitmaps = bank.to(torch.uint8) * 255
    del bank
    atlas = FontAtlas(bitmaps, torch.cat([image_descriptors(part) for part in bitmaps.split(512)]))
    raw_pools, tiers, frequencies, standards = standard_repertoires(codes)
    print(json.dumps({"public_atlas_pools": standards["atlas_codepoint_counts"]}), flush=True)
    thresholds, calibration = calibrate(atlas, codes, images[core], random, args.seed)
    print(json.dumps({"public_fit": calibration["fit"], "public_audit": calibration["audit"]}), flush=True)

    fit_matches = shape_matches(atlas, images[core])
    confidently_seen = fit_matches.confident(*thresholds)
    removed = torch.zeros(len(codes), dtype=torch.bool)
    removed[fit_matches.closest[confidently_seen]] = True
    fit_shapes = morphology(bitmaps)
    features = torch.cat((fit_shapes, F.one_hot(tiers, 4).float(), F.one_hot(frequencies, 4).float()), dim=1)
    seen, _ = seen_glyph_evidence(atlas, images[core])
    lexical = 0.55 * fit_density(features, seen) + torch.log1p(-0.94 * seen)
    current_lexical, _ = _usage_log_enrichment(tiers, frequencies, seen)
    ink = fit_shapes[:, 0]
    strata = 4 * frequencies + torch.bucketize(ink, torch.quantile(ink, torch.tensor([0.25, 0.5, 0.75])))

    inner_matches = shape_matches(atlas, images[inner])
    inner_confidence = inner_matches.confident(*thresholds)
    inner_metrics = evaluate_pools(inner_matches, inner_confidence, raw_pools, removed, lexical, strata, args.seed + 5)
    chosen = choose_pool(inner_metrics)
    print(json.dumps({"fit_only_choice": chosen, "fit_codepoints_removed": int(removed.sum())}), flush=True)
    # This independently assigned block is never involved in style, priors, exclusions, or list choice.
    held_matches = shape_matches(atlas, images[held])
    held_confidence = held_matches.confident(*thresholds)
    outer = evaluate_pools(held_matches, held_confidence, raw_pools, removed, lexical, strata, args.seed + 5)
    for name, conditions in outer.items():
        for condition, report in conditions.items():
            eligible = ~removed if condition == "residual" else torch.ones_like(removed)
            current = fixed_budget(0.55 * current_lexical + torch.log1p(-0.94 * seen), eligible,
                                   report["public_list"]["pool_size"])
            report["original_lexical_same_size"] = proxy_coverage(held_matches, current, held_confidence)
    result = {"seed": args.seed, "fit_glyphs": len(core), "inner_glyphs": len(inner), "held_glyphs": len(held),
              "prior_cohorts_excluded_from_outer": previous, "font": font, "standards": standards,
              "public_precision_calibration": calibration, "fit_codepoints_removed": int(removed.sum()),
              "fit_only_choice": chosen, "fit_only_pool_metrics": inner_metrics, "held_pool_metrics": outer,
              "fit_match_quality_median": float(fit_matches.scores[:, 0].median()),
              "held_match_quality_median": float(held_matches.scores[:, 0].median()),
              "held_assignment_disagreed_with_nearest": int((held_matches.assignments != held_matches.closest).sum()),
              "protocol": "Two visual similarities nominate codepoints on the entire fixed atlas before any "
                          "repertoire restriction. Sparse matching independently assigns fit, inner and outer "
                          "blocks with abstention. It removes only confidently unique fit matches. Confidence "
                          "thresholds use disjoint subsets of a public comparison font and may transfer imperfectly. "
                          "An inner visual cohort picks the smallest residual pool if it retains at least 95% of "
                          "both confident matches and near-best image shapes, with at least 64 accepted inner rows. "
                          "Removed outer matches are possible false exclusions measured on an independent full-atlas "
                          "image query. This is visual proxy coverage, with no competition codepoint ground truth and "
                          "no access to official test records. Shuffled control preserves counts by ink quartile and "
                          "Pinlu frequency bucket.", "seconds": time.monotonic() - started}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    condensed = {name: {condition: report["public_list"] for condition, report in conditions.items()}
                 for name, conditions in outer.items()}
    print(json.dumps({"selected": chosen, "held_public_lists": condensed, "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
