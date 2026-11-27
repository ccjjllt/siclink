"""Measure a fixed spatial TextVerifier blend against the held mice2 Task 1 v20 policy."""

import argparse
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standardized
from kaggle_task1_bipartite_validation import image_proxy
from kaggle_task1_crossfit_candidate import fit_crossfit_policy
from kaggle_task1_hanzi_latent_likelihood import SEED as LIKELIHOOD_SEED, previous_cohorts, split as likelihood_split
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED
from kaggle_task1_microns_hanzi_recheck import (
    ATLAS_SIZE, CONTOUR_OCR_WEIGHT, FIXED, FULL, NOMINATE, V14, V16,
    calibration_key, evaluate, fit_ocr_logits, with_hanzi,
)
from kaggle_task1_microns_repeat_recheck import accumulate
from kaggle_task1_native_hanzi import OCR_CHECKSUM, OCR_REVISION, response_views
from kaggle_task1_textverifier import (
    CHECKSUM, OUTER, REVISION, SPATIAL, cohorts, coordinates, publish, textverifier, verifier_features,
)
from kaggle_task1_v11_repeat_weight_transfer import (
    POISSON_AMPLIFICATION, normalized_sources, pooled_scores, weighted_fusion,
)
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, rendered_bank, summarize


SEED = 223397
MOUSE = "mice2"
OUTPUT = Path("/kaggle/working/task1_mice2_textverifier_recheck.json")
ARMS = {V14: "v14_reference", V16: "v20", FULL: "v20_textverifier", FIXED: "fixed_pool", NOMINATE: "nomination_only"}


def fresh_cohort(size):
    previous = previous_cohorts(size)
    mice1 = likelihood_split(size, np.random.default_rng(LIKELIHOOD_SEED), previous)[1]
    mice2 = likelihood_split(size, np.random.default_rng(223071), {**previous, "mice1_ocr_likelihood": mice1})[1]
    _, pilot, full, _ = cohorts(size)
    protected = {**previous, "textverifier_pilot": pilot, "textverifier_full": full}
    likelihoods = {"mice1_ocr_likelihood": mice1, "mice2_ocr_likelihood": mice2}
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(tuple(protected.values()))))
    clean = np.setdiff1d(eligible, np.union1d(mice1, mice2))
    rng = np.random.default_rng(SEED)
    picked = rng.choice(clean, min(OUTER, len(clean)), replace=False)
    # Use the oldest calibration-only cohort first when there are too few entirely fresh glyphs.
    for prior in likelihoods.values():
        available = np.setdiff1d(np.intersect1d(eligible, prior), picked)
        count = min(OUTER - len(picked), len(available))
        picked = np.concatenate((picked, rng.choice(available, count, replace=False)))
    if len(picked) != OUTER:
        raise ValueError("Fewer than 320 holdout glyphs outside the protected OCR and TextVerifier cohorts.")
    held = np.sort(picked)
    fit = np.setdiff1d(np.arange(size), held)
    overlap = {name: int(np.intersect1d(held, previous_rows).size)
               for name, previous_rows in {**protected, **likelihoods}.items()}
    return torch.from_numpy(fit), torch.from_numpy(held), {"clean_eligible": len(clean), "overlap_by_cohort": overlap}


def tensor_hash(values):
    return sha256(memoryview(values.detach().cpu().contiguous().numpy()).cast("B")).hexdigest()


@torch.inference_mode()
def spatial_logits(fit_images, atlas, views, device, checkpoint):
    vision, projector = textverifier(checkpoint, device)

    def extract(images):
        return {SPATIAL: verifier_features(images, vision, projector)[SPATIAL]}

    fitted, banks = coordinates(fit_images, {"atlas": atlas.cpu()}, extract, batch_size=32)
    design, bank = fitted[SPATIAL], banks["atlas"][SPATIAL]
    return {day: standardized(F.normalize(query.to(device) @ (ridge.to(device) @ design)) @ bank)
            for day, (ridge, _, query) in views.items()}


def evaluate_condition(exact, base, amplified, model, sources, ocr, spatial, ranker, proxy, trusted, inverse):
    incumbent = with_hanzi(model, amplified, sources, ocr)
    blend = 0.5 * (standardized(ocr) + standardized(spatial))
    proposed = with_hanzi(model, amplified, sources, blend)
    values, details = evaluate(exact, base, incumbent, proposed, ranker, proxy, trusted, inverse)
    details["recall"] = {ARMS[arm]: result for arm, result in details["recall"].items()}
    details["changed_choices_vs_v20"] = {ARMS[arm]: result for arm, result in details.pop("selected_changes_vs_v16").items()}
    details["mean_new_candidates_vs_v20"] = details.pop("hanzi_mean_novel_candidates_vs_v16")
    details["rows_with_new_candidates_vs_v20"] = details.pop("hanzi_rows_with_new_candidates_vs_v16")
    return values, details


def paired(values):
    report = {ARMS[arm]: summarize(scores, values[V16]) for arm, scores in values.items()}
    report["full_vs_fixed_pool"] = summarize(values[FULL], values[FIXED])
    report["full_vs_nomination_only"] = summarize(values[FULL], values[NOMINATE])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Extracted competition directory containing catalog.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--calibration-cache", type=Path, help="Save and resume the exact four fit-only v20 folds")
    parser.add_argument("--checkpoint", type=Path, help="Pinned TextVerifier checkpoint or an input directory containing it")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle or Colab CUDA GPU for this held evaluation.")
    started = time.monotonic()
    torch.set_num_threads(4)
    cpu, device = torch.device("cpu"), torch.device("cuda:0")
    root = args.data_root.resolve(strict=True) if args.data_root else locate_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scorer = official_scorer(root)
    images = load_images(root, cpu)
    if images.shape != (2748, 64, 64) or images.dtype != torch.uint8:
        raise ValueError("Expected all 2,748 labeled 64×64 uint8 training rasters.")
    fit, held, independence = fresh_cohort(len(images))
    groups = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images))
    if tuple(groups) != SESSIONS or any(counts.shape[:2] != (len(images), 4) for counts in groups.values()):
        raise ValueError("Expected four complete labeled mice2 recordings with four repeats per glyph.")

    labeled = images[fit]
    references = images[held]
    cache_key, cache_dir = None, None
    if args.calibration_cache is not None:
        cache_key = calibration_key(fit, held, labeled, groups, scorer, device, mouse=MOUSE)
        cache_dir = args.calibration_cache.resolve() / cache_key
    report = {"mouse": MOUSE, "outer_seed": SEED, "v14_seed": MODEL_SEED, "gpu": torch.cuda.get_device_name(device),
              "torch_version": torch.__version__, "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "training_raster_sha256": tensor_hash(images), "fit_glyph_count": len(fit), "held_glyph_count": len(held),
              "fit_gallery_indices_sha256": tensor_hash(fit), "held_gallery_indices": held.tolist(),
              "held_gallery_indices_sha256": tensor_hash(held), "held_independence": independence,
              "formal_queries_loaded": False, "anchor_ocf": ANCHOR, "nominations": NOMINATIONS,
              "poisson_multiplier": POISSON_AMPLIFICATION, "v20_ocr_share": CONTOUR_OCR_WEIGHT,
              "ocr": {"revision": OCR_REVISION, "sha256": OCR_CHECKSUM},
              "textverifier": {"revision": REVISION, "sha256": CHECKSUM, "spatial_share_within_ocr": 0.5},
              "primary": "Same recording, four measured repeats per held glyph",
              "secondary": "Known correspondence across four recordings, 16 measured repeats per held glyph",
              "protocol": "Compare held v20 decisions and one fixed equal standardized OCR/spatial blend on the same "
                          "v14-matched candidate counts, fourfold ranker, and output renderer. The four-fold model, "
                          "noise fit, outer model and both image readouts exclude all held glyphs. The held targets "
                          "supply only visual proxy and official OCF endpoints.", "sessions": {}}
    if cache_dir is not None:
        report["calibration_cache"] = {"key": cache_key, "directory": str(cache_dir)}
    publish(report, args.output, started, "fourfold_started")
    training = {day: counts[fit].to(device) for day, counts in groups.items()}
    ranker, folds, noise, codepoints = fit_crossfit_policy(training, labeled.to(device), scorer,
                                                        cache_dir=cache_dir, cache_key=cache_key)
    report["v20_calibration"] = {"folds": folds, "ranker": ranker.diagnostics, "noise": "literal calibration fold zero"}
    publish(report, args.output, started, "outer_model_started")
    model = fit_v8(training, labeled.to(device), MODEL_SEED)
    del training
    if len(codepoints) != ATLAS_SIZE or not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("The original v20 atlas order must match every calibration fold.")
    font, unihan = model.report["atlas"]["sha256"], model.report["prior"]["unicode"]["sha256"]
    for fold in folds.values():
        source = fold["atlas_and_font_style"]
        if (source["atlas"]["sha256"], source["prior"]["unicode"]["sha256"]) != (font, unihan):
            raise ValueError("All calibration folds must use the same font and Unihan assets as the outer fit.")
    rendered = rendered_bank(model)
    report["v20_atlas"] = {**model.report, "size": len(codepoints), "codepoint_order_sha256": tensor_hash(codepoints),
                            "feature_bank_sha256": tensor_hash(model.atlas.images), "rendered_bank_sha256": tensor_hash(rendered)}
    publish(report, args.output, started, "ocr_started")
    views = response_views(groups, fit, held, np.random.default_rng(SEED + 1))
    ocr = fit_ocr_logits(labeled, model.atlas.images, views)
    publish(report, args.output, started, "textverifier_started")
    spatial = spatial_logits(labeled, model.atlas.images, views, device, args.checkpoint)
    del views, labeled, images
    publish(report, args.output, started, "textverifier_ready")

    proxy, trusted = image_proxy(model, references.to(device))
    report["trusted_proxy_glyphs"] = int(trusted.sum())
    exact = PairedOCF(scorer, rendered, (references >= 128).numpy(), device, 12)
    rng = np.random.default_rng(SEED + 2)
    totals, measured = {}, {}
    for day, counts in groups.items():
        order = torch.as_tensor(rng.permutation(len(held)), device=device)
        inverse = order.argsort()
        base = atlas_scores(model, noise[day], counts[held].to(device)[order], day)
        sources = normalized_sources(base)
        amplified = {**base, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}
        values, details = evaluate_condition(exact, base, amplified, model, sources, ocr[day][order], spatial[day][order],
                                             ranker, proxy[order], trusted[order], inverse)
        for arm, score in values.items():
            measured.setdefault(arm, []).append(score)
        accumulate(totals, sources, inverse)
        report["sessions"][day] = {**details, "paired_exact_ocf": paired({arm: [score] for arm, score in values.items()})}
        report["same_day_four_repeats"] = paired(measured)
        publish(report, args.output, started, f"{day}_scored")

    sources = {name: matrix.div_(len(SESSIONS)) for name, matrix in totals.items()}
    base, amplified = pooled_scores(model, sources)
    values, details = evaluate_condition(exact, base, amplified, model, sources, torch.stack(tuple(ocr.values())).mean(0),
                                         torch.stack(tuple(spatial.values())).mean(0), ranker, proxy, trusted,
                                         torch.arange(len(held), device=device))
    report["pooled_16_known_correspondence"] = {**details, "paired_exact_ocf": paired({arm: [score] for arm, score in values.items()})}
    report["unique_exact_pairs_scored"] = len(exact.values)
    publish(report, args.output, started, "complete")


if __name__ == "__main__":
    main()
