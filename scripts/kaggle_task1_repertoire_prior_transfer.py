"""Measure a public repertoire bonus on top of the existing Unihan usage prior."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_prior import _usage_log_enrichment, seen_glyph_evidence, unihan_usage
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import build_font_atlas, image_descriptors, standardized
from kaggle_task1_homoglyph_transfer import SHARE, measurements, read_release
from kaggle_task1_mice2_textverifier_recheck import fresh_cohort
from kaggle_task1_microns_hanzi_recheck import fit_ocr_logits
from kaggle_task1_native_hanzi import response_views
from kaggle_task1_native_hanzi_open_atlas import visual_proxy
from kaggle_v8_metric_validation import summarize


SEED = 223681
BASE, PUBLISHED, CONTROL, NO_PRIOR = "unihan", "unihan_plus_published", "unihan_plus_matched", "no_usage_prior"
OUTPUT = Path("/kaggle/working/task1_repertoire_prior_transfer.json")


def publish(report, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


def matched_membership(usage, frequency, membership, rng):
    strata = (usage * 4 + frequency).numpy()
    matched = torch.zeros_like(membership)
    for stratum in np.unique(strata):
        positions = np.flatnonzero(strata == stratum)
        count = int(membership[positions].long().sum())
        matched[rng.choice(positions, size=count, replace=False)] = True
    return matched


def report_scores(results):
    summary = {name: summarize(values, results[BASE]) for name, values in results.items()}
    summary["published_vs_matched"] = summarize(results[PUBLISHED], results[CONTROL])
    return summary


@torch.inference_mode()
def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    root = locate_data()
    images = load_images(root, device)
    eligible, earlier, _ = fresh_cohort(len(images))
    held = torch.from_numpy(np.sort(rng.choice(eligible.numpy(), size=320, replace=False)))
    fit = torch.from_numpy(np.setdiff1d(np.arange(len(images)), held.numpy()))
    styles = rng.choice(fit.numpy(), 128, replace=False)
    atlas, font, codepoints = build_font_atlas(images[styles], device, return_codepoints=True)
    dictionary, provenance = read_release()
    if provenance["sha256"] != "c2506df8760024cce7121b12a2822572db55744bdade8e7451abe8332a4e5a55":
        raise ValueError("The released repertoire changed since the first comparison.")
    published = torch.tensor([chr(point) in dictionary for point in codepoints.tolist()])
    del dictionary
    usage, frequency, unihan = unihan_usage(codepoints)
    matched = matched_membership(usage, frequency, published, rng)
    support = torch.where(published)[0]
    labeled = images[fit]
    seen, calibration = seen_glyph_evidence(atlas, labeled)
    lexical, _ = _usage_log_enrichment(usage, frequency, seen)
    unseen = torch.log1p(-0.94 * seen)
    prior = 0.55 * lexical + unseen
    sample = torch.from_numpy(np.sort(rng.choice(len(held), size=128, replace=False)))
    references = images[held]
    report = {"mouse": "mice2", "seed": SEED, "fit_glyphs": len(fit), "held_glyphs": len(held),
              "overlap_with_first_transfer": int(torch.isin(held, earlier).sum()),
              "scored_gallery_indices": held[sample].tolist(),
              "atlas_size": len(atlas.images), "covered_candidates": int(published.sum()),
              "control_covered_candidates": int(matched.sum()),
              "shared_candidates": int((published & matched).sum()), "bonus": SHARE,
              "font": font, "unicode": unihan, "release": provenance, "seen_fit": calibration,
              "protocol": "Estimate existing Unihan usage and fit-glyph exclusion on the full fitting cohort. "
                          "Compare equal additive bonuses for author-published membership and an independently "
                          "seeded repertoire stratified on Unihan usage and frequency, under a common fixed "
                          "contour/OCR readout. All 27,550 candidates remain eligible in each arm. "
                          "Same-day means use four observed repeats. Summed scores across days use known held "
                          "glyph correspondence as a 16-repeat diagnostic. Only training spikes are queried.",
              "recordings": {}}
    publish(report, started, "repertoires_ready")
    groups = grouped_blocks(root, load_catalog(root), "mice2", device, len(images))
    views = response_views(groups, fit, held, rng)
    del groups
    ocr = fit_ocr_logits(labeled, atlas.images, views)
    contour = image_descriptors((labeled >= 128).to(torch.uint8) * 255)
    proxy, confident, _ = visual_proxy(atlas, references)
    report["trusted_glyphs"] = int(confident.sum())
    report["trusted_glyphs_covered"] = {"published": int((confident & published[proxy]).sum()),
                                        "matched": int((confident & matched[proxy]).sum())}
    metric = PairedOCF(official_scorer(root), atlas.images, (references[sample] >= 128).numpy(), device, 8)
    day_values, totals = {}, {}
    for day, (ridge, _, query) in views.items():
        shape = standardized(F.normalize(query @ (ridge @ contour)) @ atlas.contours.T)
        base = standardized(0.5 * (shape + ocr[day]))
        incumbent = base + prior
        scores = {BASE: incumbent, PUBLISHED: incumbent + SHARE * published, CONTROL: incumbent + SHARE * matched,
                  NO_PRIOR: base + unseen}
        recall, exact = measurements(scores, sample, metric, proxy, confident, support)
        for arm, values in scores.items():
            if arm in totals:
                totals[arm].add_(values)
            else:
                totals[arm] = values
            day_values.setdefault(arm, []).append(exact[arm])
        report["recordings"][day] = {"ocf": report_scores({name: [vals] for name, vals in exact.items()}),
                                      "recall": recall}
        publish(report, started, f"{day}_scored")
    recall, exact = measurements(totals, sample, metric, proxy, confident, support)
    report["same_day_four_repeat"] = report_scores(day_values)
    report["pooled_16_known_correspondence"] = {"ocf": report_scores({name: [v] for name, v in exact.items()}),
                                               "recall": recall}
    report["unique_exact_pairs"] = len(metric.values)
    publish(report, started, "complete")


if __name__ == "__main__":
    main()
