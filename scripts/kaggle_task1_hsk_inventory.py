"""Test public HSK char inventories against fit-only Unihan on held Task 1 glyphs."""

from collections import Counter
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_prior import _usage_log_enrichment, seen_glyph_evidence
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_font_family_coverage import BASE as FONT, fit_bank
from kaggle_gallery_validation import FontAtlas, image_descriptors, standardized, standard_error
from kaggle_task1_mice2_textverifier_recheck import fresh_cohort, tensor_hash
from kaggle_task1_microns_hanzi_recheck import fit_ocr_logits
from kaggle_task1_native_hanzi import response_views
from kaggle_task1_repertoire_coverage import calibrate, independent_cohorts, shape_matches, standard_repertoires
from kaggle_task1_repertoire_prior_transfer import SEED as PRIOR_TRANSFER_SEED, matched_membership
from kaggle_task1_textverifier import publish
from kaggle_v8_metric_validation import SESSIONS, summarize


SEED = 224177
CALIBRATION, HELD, EXACT = 256, 320, 64
BUDGETS = (340, 500, 700, 1000)
NEURAL_BUDGET = 700
OUTPUT = Path("/kaggle/working/task1_hsk_inventory.json")
WHEEL = "https://files.pythonhosted.org/packages/60/f8/4cb12e94f5ee403f9be370e2d864384be4554e7f1981e204df47bb543c48/hsk30-0.3.0-py3-none-any.whl"
WHEEL_SHA = "4374a20fcf10305762b5d253b0abc22325eb8df6f136e4f5f7a91c496b6e5ac6"
HSK2025 = "https://download.chinesetest.cn/newhsk-site/Syllabus/H_DG_1110.pdf"
HSK2021 = "https://www.moe.gov.cn/jyb_xwfb/gzdt_gzdt/s5987/202103/W020210329527301787356.pdf"
UNI, NEW, OLD, CTRL, FITTED, FULL = "unihan", "hsk_2025", "hsk_2021", "matched_hsk_control", "fit_repertoire", "full_unihan"


def independent_split(size):
    eligible, direct, _ = fresh_cohort(size)
    prior = {
        "published_coverage": independent_cohorts(size, 7933)[2],
        "public_list_neural": independent_cohorts(size, 8291, exclude_coverage_seed=7933)[2],
        "published_membership_transfer": np.random.default_rng(PRIOR_TRANSFER_SEED).choice(eligible.numpy(), size=320, replace=False),
        "textverifier_direct": direct,
    }
    excluded = np.unique(np.concatenate(tuple(prior.values())))
    rng = np.random.default_rng(SEED)
    held = np.sort(rng.choice(np.setdiff1d(np.arange(size), excluded), size=HELD, replace=False))
    training = np.setdiff1d(np.arange(size), held)
    calibration = np.sort(rng.choice(training, size=CALIBRATION, replace=False))
    fit = np.setdiff1d(training, calibration)
    overlap = {name: int(np.intersect1d(held, rows).size) for name, rows in prior.items()}
    return tuple(torch.from_numpy(rows) for rows in (fit, calibration, held)), overlap


def hsk_levels(codepoints):
    with urlopen(WHEEL, timeout=90) as response:
        contents = response.read()
    if sha256(contents).hexdigest() != WHEEL_SHA:
        raise ValueError("The public character table wheel differs from its pinned hash.")
    levels, provenance = {}, {"source": WHEEL, "sha256": WHEEL_SHA, "tables": {}}
    definitions = ((NEW, HSK2025, "hsk2025_chars.tsv", (246, 125, 284, 441, 431, 413, 1148)),
                   (OLD, HSK2021, "hsk30_chars.tsv", (300, 300, 300, 300, 300, 300, 1200)))
    with ZipFile(BytesIO(contents)) as archive:
        for name, standard, filename, counts in definitions:
            member = f"hsk30/data/{filename}"
            raw = archive.read(member)
            lines = (line for line in raw.decode("utf-8-sig").splitlines() if line and not line.startswith("#"))
            reader = csv.reader(lines, delimiter="\t")
            next(reader)
            rows = [(ord(row[0]), int(row[1])) for row in reader]
            grades = dict(rows)
            if len(grades) != len(rows) or Counter(grades.values()) != Counter(dict(enumerate(counts, 1))):
                raise ValueError(f"The {name} character counts do not match the published extraction.")
            levels[name] = torch.tensor([grades.get(point, 0) for point in codepoints.tolist()])
            provenance["tables"][name] = {"official_standard": standard, "member": member,
                                          "sha256": sha256(raw).hexdigest(), "distinct_characters": len(grades),
                                          "atlas_characters": int((levels[name] > 0).sum()), "per_level": counts}
    return levels, provenance


def fit_membership(usage, frequency, hsk, gb_first, seen):
    """Estimate observed training-set occupancy with a shrunk public-inventory partition."""
    parent = usage * 4 + frequency
    global_rate = seen.mean()
    base = (torch.bincount(parent, weights=seen, minlength=16) + 64 * global_rate) / (torch.bincount(parent, minlength=16) + 64)
    bands = {name: torch.bucketize(level, torch.tensor([0, 3, 6])) for name, level in hsk.items()}
    child = ((parent * 4 + bands[NEW]) * 4 + bands[OLD]) * 2 + gb_first.long()
    size = torch.bincount(child, minlength=512)
    mass = torch.bincount(child, weights=seen, minlength=512)
    posterior = (mass + 32 * base.repeat_interleave(32)) / (size + 32)
    return posterior[child].clamp_min(1e-8).log(), {"nonempty_groups": int((size > 0).sum()), "shrinkage": 32}


def raw_coverage(matches, confidence, support):
    closest = support[matches.closest]
    close = support[matches.candidates] & (matches.scores >= matches.scores[:, :1] - 0.035)
    return {"nearest": closest.numpy(), "near": close.any(1).numpy(), "trusted": closest[confidence].numpy()}


def paired_coverage(measured, reference):
    report = {}
    for key, outcome in measured.items():
        offset = outcome.astype(float) - reference[key]
        report[key] = {"coverage": float(outcome.mean()) if len(outcome) else None, "glyphs": len(outcome),
                       "difference_vs_unihan": float(offset.mean()) if len(offset) else None,
                       "paired_se": standard_error(offset)}
    return report


def limited_pools(scores, allowed, rng, budgets):
    # Break exact prior ties independently of Unicode order.
    allowed = torch.where(allowed)[0]
    shuffle = torch.from_numpy(rng.permutation(len(allowed)))
    allowed = allowed[shuffle]
    pools = {}
    for budget in budgets:
        pools[budget] = {}
        for arm, score in scores.items():
            mask = torch.zeros(len(score), dtype=torch.bool)
            mask[allowed[score[allowed].argsort(descending=True, stable=True)[:budget]]] = True
            pools[budget][arm] = mask
    return pools


def estimate_readout(scores, pools, proxy, confident, exact, sample):
    matrices = {FULL: scores, **{name: scores.masked_fill(~pool, -torch.inf) for name, pool in pools.items()}}
    recall, selected = {}, {}
    for name, matrix in matrices.items():
        ranked = matrix.topk(32, dim=1).indices
        ranks = ranked == proxy[:, None]
        recall[name] = {str(k): float(ranks[confident, :k].any(1).float().mean()) if confident.any() else None for k in (1, 8, 32)}
        selected[name] = exact.evaluate(ranked[sample, :1])[:, 0]
    return recall, selected


def neural_summary(by_arm):
    return {name: {"vs_unihan_budget": summarize(values, by_arm[UNI]), "vs_full_unihan": summarize(values, by_arm[FULL])}
            for name, values in by_arm.items()}


def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU notebook with official training input only.")
    torch.set_num_threads(4)
    device = torch.device("cpu")
    started = time.monotonic()
    rng = np.random.default_rng(SEED + 1)
    root = locate_data()
    images = load_images(root, device)
    if images.shape != (2748, 64, 64) or images.dtype != torch.uint8:
        raise ValueError("Expected the full official labeled training raster gallery.")
    (fit, validation, held), overlap = independent_split(len(images))
    labeled, references = images[fit], images[held]
    raw, codepoints, font = fit_bank(FONT, labeled, SEED)
    rasters = raw.to(torch.uint8) * 255
    del raw
    atlas = FontAtlas(rasters, torch.cat([image_descriptors(batch) for batch in rasters.split(512)]))
    standards, usage, frequency, unihan = standard_repertoires(codepoints)
    grades, release = hsk_levels(codepoints)
    threshold, audit = calibrate(atlas, codepoints, labeled, rng, SEED)
    fit_match = shape_matches(atlas, labeled)
    fit_confident = fit_match.confident(*threshold)
    removed = torch.zeros(len(codepoints), dtype=torch.bool)
    removed[fit_match.closest[fit_confident]] = True
    seen, match_stats = seen_glyph_evidence(atlas, labeled)
    lexical, _ = _usage_log_enrichment(usage, frequency, seen)
    unseen = torch.log1p(-0.94 * seen)
    prior = 0.55 * lexical + unseen
    members = {name: level > 0 for name, level in grades.items()}
    random_members = matched_membership(usage, frequency, members[NEW], rng)
    learned, fit_report = fit_membership(usage, frequency, grades, standards["gb2312_level1"], seen)
    priorities = {UNI: prior, NEW: prior + 8 * members[NEW], OLD: prior + 8 * members[OLD],
                  CTRL: prior + 8 * random_members, FITTED: learned + unseen}
    if int((~removed).sum()) < max(BUDGETS):
        raise ValueError("Every candidate policy needs enough allowed columns for all budgets.")
    hsk_residual = int((members[NEW] & ~removed).sum())
    all_budgets = tuple(sorted(set(BUDGETS + (hsk_residual,))))
    pools = limited_pools(priorities, ~removed, rng, all_budgets)
    report = {"seed": SEED, "mouse": "mice2", "fit_glyphs": len(fit), "held_inventory_glyphs": len(held),
              "held_gallery_indices": held.tolist(), "prior_held_overlap": overlap,
              "training_raster_sha256": tensor_hash(images), "codepoint_order_sha256": tensor_hash(codepoints),
              "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "font": font, "unicode": unihan,
              "hsk": release, "alignment_public_font_audit": audit, "fit_alignment": match_stats,
              "learned_prior": fit_report, "hard_excluded_fit_glyphs": int(removed.sum()),
              "hsk_2025_unseen_atlas_columns": hsk_residual,
              "formal_queries_loaded": False, "match_budgets": all_budgets, "neural_budget": NEURAL_BUDGET,
              "protocol": "Two fixed independent public recognition lists and categorical training occupancy are "
                          "compared to Unihan on new whole glyphs. Every shortlist uses the same confidently observed "
                          "fit exclusions and the same budget. No codepoint ordering, sample ordering or test files "
                          "participate. Image-to-codepoint matches are audited proxies. The neural comparison uses "
                          "the same Unihan plus contour/OCR score to choose from each fixed support.", "recordings": {}}

    for name, matching in (("fit", fit_match), ("validation", shape_matches(atlas, images[validation])),
                           ("held", shape_matches(atlas, references))):
        confident = matching.confident(*threshold)
        raw_members = {**members, "tgh_level1": standards["tgh_level1"], "gb2312_level1": standards["gb2312_level1"]}
        details = {}
        for arm, mask in raw_members.items():
            measure = raw_coverage(matching, confident, mask)
            details[arm] = {key: {"rate": float(value.mean()) if len(value) else None, "glyphs": len(value)}
                            for key, value in measure.items()}
        budgets = {}
        if name != "fit":
            for budget, candidates in pools.items():
                rates = {arm: raw_coverage(matching, confident, mask) for arm, mask in candidates.items()}
                budgets[str(budget)] = {arm: paired_coverage(scores, rates[UNI]) for arm, scores in rates.items()}
        report[f"{name}_coverage"] = {"raw_membership": details, "matched_budget": budgets,
                                       "trusted_proxy_glyph_count": int(confident.sum())}
        if name == "held":
            held_match, held_confident = matching, confident
    publish(report, OUTPUT, started, "coverage_complete")

    groups = grouped_blocks(root, load_catalog(root), "mice2", device, len(images))
    if tuple(groups) != SESSIONS or any(group.shape[:2] != (len(images), 4) for group in groups.values()):
        raise ValueError("Expected four measured trials in every labeled recording.")
    views = response_views(groups, fit, held, rng)
    del groups
    ocr = fit_ocr_logits(labeled, atlas.images, views)
    fitted_contours = image_descriptors((labeled >= 128).to(torch.uint8) * 255)
    sample = torch.from_numpy(np.sort(rng.choice(len(held), size=EXACT, replace=False)))
    report["ocf_gallery_indices"] = held[sample].tolist()
    exact = PairedOCF(official_scorer(root), atlas.images, (references[sample] >= 128).numpy(), device, 8)
    totals, observations = None, {}
    for day, (ridge, _, query) in views.items():
        contour = standardized(F.normalize(query @ (ridge @ fitted_contours)) @ atlas.contours.T)
        scores = standardized(0.5 * (contour + ocr[day])) + prior
        recalls, selected = estimate_readout(scores, pools[NEURAL_BUDGET], held_match.closest, held_confident, exact, sample)
        for name, result in selected.items():
            observations.setdefault(name, []).append(result)
        report["recordings"][day] = {"ocf": neural_summary({name: [values] for name, values in selected.items()}), "recall": recalls}
        totals = scores.clone() if totals is None else totals.add_(scores)
        publish(report, OUTPUT, started, f"{day}_scored")
    report["same_day_four_repeat"] = neural_summary(observations)
    recalls, selected = estimate_readout(totals, pools[NEURAL_BUDGET], held_match.closest, held_confident, exact, sample)
    report["pooled_16_known_correspondence"] = {"ocf": neural_summary({name: [values] for name, values in selected.items()}), "recall": recalls}
    report["unique_exact_pairs"] = len(exact.values)
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
