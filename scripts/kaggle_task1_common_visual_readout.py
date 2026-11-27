"""Decode held glyphs through a shared eigenspace learned across recordings."""

from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_prior import fit_atlas_prior
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_multiview_alignment import fit_multiview
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_gallery_validation import build_font_atlas, standardized
from kaggle_task1_mice2_textverifier_recheck import fresh_cohort
from kaggle_task1_native_hanzi import feature_extractors, fit_response_ridge
from kaggle_task1_native_hanzi_open_atlas import image_coordinates, visual_proxy
from kaggle_task1_repertoire_prior_transfer import SEED as REPERTOIRE_SEED
from kaggle_task1_textverifier import publish
from kaggle_v8_metric_validation import summarize


SEED = 223849
VIEW, DIMENSION, SHARE = "evoked-multiscale", 96, 0.30
SCORED = 96
MOUSE, PEER = "mice2", "mice1"
STOCK, WITHIN, ACROSS, SHUFFLED = "per_day_ridge", "within_mouse", "joint_mice", "shuffled_peer"
OUTPUT = Path("/kaggle/working/task1_common_visual_readout.json")


def held_partition(size, rng):
    eligible, reference, _ = fresh_cohort(size)
    repertoire = np.random.default_rng(REPERTOIRE_SEED).choice(eligible.numpy(), 320, replace=False)
    clean = np.setdiff1d(eligible.numpy(), repertoire)
    held = np.sort(rng.choice(clean, 320, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    report = {"prior_direct_holdout_overlap": int(np.intersect1d(held, reference.numpy()).size),
              "prior_repertoire_holdout_overlap": int(np.intersect1d(held, repertoire).size)}
    return torch.from_numpy(fit), torch.from_numpy(held), report


def fitted_visual_scores(training, queries, coordinates):
    ridge, center, scale = fit_response_ridge(training)
    coefficients = {name: ridge @ target for name, (target, _) in coordinates.items()}
    return {key: decision((values - center) / scale, coefficients, coordinates) for key, values in queries.items()}


def decision(observation, coefficients, coordinates):
    sources = []
    for name, (_, atlas) in coordinates.items():
        projection = F.normalize(observation @ coefficients[name])
        sources.append(standardized(projection @ atlas))
    return standardized(torch.stack(sources).mean(0))


def cohort_shared(training, held, coordinates, permutation):
    blocks = tuple(held)
    target = {key: training[key] for key in blocks}
    control = {key: (counts[permutation] if key[0] == PEER else counts) for key, counts in training.items()}
    repeated = {feature: (labels.repeat(len(blocks), 1), atlas)
                for feature, (labels, atlas) in coordinates.items()}
    reports, predictions = {}, {}
    for name, sources in ((WITHIN, target), (ACROSS, training), (SHUFFLED, control)):
        model = fit_multiview(sources, view=VIEW, dimension=DIMENSION)
        participants = torch.cat([model.encode(key, target[key]) for key in blocks])
        queries = {key: model.encode(key, held[key]) for key in blocks}
        predictions[name] = fitted_visual_scores(participants, queries, repeated)
        reports[name] = {
            "views": len(sources),
            "dimensions": len(model.spectrum),
            "smallest_gcca_eigenvalue": float(model.spectrum.min()),
            "largest_gcca_eigenvalue": float(model.spectrum.max()),
            "cross_mouse_fit_agreement": (float(model.cross_mouse_agreement.mean())
                                           if model.cross_mouse_agreement is not None else None),
        }
    return predictions, reports


def measure(scores, rows, metric, proxy, trusted):
    exact, recall = {}, {}
    for name, values in scores.items():
        top = values.topk(32, dim=1).indices
        found = top == proxy[:, None]
        recall[name] = {str(k): float(found[trusted, :k].any(1).float().mean())
                        if bool(trusted.any()) else None for k in (1, 16, 32)}
        exact[name] = metric.evaluate(top[rows, :1])[:, 0]
    return exact, recall


def paired(scores):
    report = {name: summarize(values, scores[STOCK]) for name, values in scores.items()}
    report["joint_vs_within"] = summarize(scores[ACROSS], scores[WITHIN])
    report["joint_vs_shuffled"] = summarize(scores[ACROSS], scores[SHUFFLED])
    return report


@torch.inference_mode()
def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    root = locate_data()
    images, catalog = load_images(root, device), load_catalog(root)
    fit, held, overlap = held_partition(len(images), rng)
    training, queries = {}, {}
    for mouse in (MOUSE, PEER):
        for day, counts in grouped_blocks(root, catalog, mouse, device, len(images)).items():
            key = (mouse, day)
            training[key] = counts[fit]
            if mouse == MOUSE:
                queries[key] = counts[held]
    report = {"mouse": MOUSE, "peer_training_mouse": PEER, "seed": SEED, "view": VIEW, "blend": SHARE,
              "fit_glyphs": len(fit), "held_glyphs": len(held), "cohort_overlap": overlap, "recordings": {},
              "protocol": "Compare separate raw per-day image readouts with a fixed blend of a shared 96-dimensional "
                          "GCCA readout. Calibrate shared views with mice2 only, both mice using paired training "
                          "glyphs, or both mice after applying one common permutation to the four peer days. "
                          "All shared visual heads learn from the same four mice2 recordings and visual features. "
                          "Every query uses only its own mice2 recording. Known correspondence is used only in "
                          "the diagnostic of four independent recording days pooled to 16 spikes trials."}
    publish(report, OUTPUT, started, "inputs_ready")

    picked = rng.choice(fit.numpy(), size=128, replace=False)
    atlas, font, codepoints = build_font_atlas(images[picked], device, return_codepoints=True)
    prior = fit_atlas_prior(atlas, codepoints, images[fit])
    features = feature_extractors()
    features.pop("random_mobilenet")
    coordinates = image_coordinates(images[fit], atlas.images, features)
    permutation = torch.from_numpy(rng.permutation(len(fit)))
    shared, model_reports = cohort_shared(training, queries, coordinates, permutation)
    report.update(atlas=font, unihan=prior.diagnostics, candidates=len(codepoints), gcca=model_reports)
    publish(report, OUTPUT, started, "scores_ready")

    selection = torch.from_numpy(np.sort(rng.choice(len(held), SCORED, replace=False)))
    proxy, trusted, _ = visual_proxy(atlas, images[held])
    metric = PairedOCF(official_scorer(root), atlas.images, (images[held[selection]] >= 128).numpy(), device, 8)
    report.update(scored_glyph_indices=held[selection].tolist(), trusted_proxy_glyphs=int(trusted.sum()))
    all_exact, sums = {}, {}
    for key, query in queries.items():
        fit_view, moments = standardize(feature_view(training[key], VIEW))
        test_view, _ = standardize(feature_view(query, VIEW), moments)
        raw = fitted_visual_scores(fit_view.mean(1), {key: test_view.mean(1)}, coordinates)[key]
        fused = {STOCK: raw, **{name: (1 - SHARE) * raw + SHARE * values.pop(key) for name, values in shared.items()}}
        scores = {name: prior.variants(values)["usage_unseen"] for name, values in fused.items()}
        exact, recall = measure(scores, selection, metric, proxy, trusted)
        for name, values in scores.items():
            if name in sums:
                sums[name].add_(values)
            else:
                sums[name] = values
            all_exact.setdefault(name, []).append(exact[name])
        report["recordings"][key[1]] = {"ocf": paired({name: [values] for name, values in exact.items()}),
                                         "proxy_recall": recall}
        publish(report, OUTPUT, started, f"{key[1]}_scored")
    exact, recall = measure(sums, selection, metric, proxy, trusted)
    report["same_day_four_repeat"] = paired(all_exact)
    report["pooled_16_known_correspondence"] = {"ocf": paired({name: [values] for name, values in exact.items()}),
                                               "proxy_recall": recall}
    report["unique_exact_pairs"] = len(metric.values)
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
