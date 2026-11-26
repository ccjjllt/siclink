"""Test whether another mouse's independent recording improves Task 2 assignments."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_mbr import fit_temperature, glyph_posterior
from glyph_multiview_alignment import fit_multiview, pair_similarity
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_hierarchical_gallery_validation import validated_logits
from kaggle_task2_metric_validation import SESSIONS
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_v8_metric_validation import summarize


MICE = ("mice1", "mice2")
LATENT_PARTS = 48
PEER_IDENTITY_TEMPERATURE = 0.34628259916890003
PEER_WEIGHT = 0.25
BALANCE_STEPS = 24


def logit_scale(matrix):
    centered = matrix - matrix.mean(dim=1, keepdim=True)
    return centered / centered.square().mean().sqrt().clamp_min(1e-6)


def alignment(sources):
    dates = {mouse: torch.cat(tuple(days.values()), dim=1) for mouse, days in sources.items()}
    return fit_multiview({(mouse, "pooled"): counts for mouse, counts in dates.items()},
                         view="early-late", dimension=LATENT_PARTS, ridge=0.55)


@torch.no_grad()
def crossmouse_scores(model, queries):
    embedding = {}
    for mouse, counts in queries.items():
        values = model.encode((mouse, "pooled"), counts)
        embedding[mouse] = F.normalize(values - values.mean(dim=0))
    return logit_scale(pair_similarity(embedding["mice2"], embedding["mice1"]))


def calibrate_on_sources(sources, count, seed):
    rng = np.random.default_rng(seed)
    calibration = []
    for session in sources["mice1"]:
        paired_other_days = {mouse: {day: counts for day, counts in blocks.items() if day != session}
                             for mouse, blocks in sources.items()}
        model = alignment(paired_other_days)
        order = {mouse: torch.as_tensor(rng.permutation(count)) for mouse in MICE}
        queries = {mouse: sources[mouse][session][order[mouse]] for mouse in MICE}
        score = crossmouse_scores(model, queries)
        pairing = order["mice1"].argsort()[order["mice2"]]
        calibration.append((score, pairing))
    return fit_temperature(calibration)


def task2_scores(mouse, sources, query, images, args):
    validated = validated_logits(mouse, sources, query, images, images.device, args)
    gallery = fit_hierarchical_gallery(sources, view="total")
    empirical = gallery.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    return blend_hierarchical_logits(validated, empirical)


def row_diagnostics(scores, truth):
    leaders = scores.topk(32, dim=1).indices == truth[:, None]
    return {"top1": float(leaders[:, 0].float().mean()), "top5": float(leaders[:, :5].any(dim=1).float().mean()),
            "top32": float(leaders.any(dim=1).float().mean())}


def evaluate(held_day, recordings, images, masks, scorer, samples, args):
    started = time.monotonic()
    count = len(images)
    sources = {mouse: {day: counts for day, counts in rows.items() if day != held_day}
               for mouse, rows in recordings.items()}
    model = alignment(sources)
    temperature = calibrate_on_sources(sources, count, args.split_seed)
    rng = np.random.default_rng(args.split_seed + SESSIONS.index(held_day))
    order = {mouse: torch.as_tensor(rng.permutation(count)) for mouse in MICE}
    queries = {mouse: recordings[mouse][held_day][order[mouse]] for mouse in MICE}
    estimates = {mouse: task2_scores(mouse, sources[mouse], queries[mouse], images, args) for mouse in MICE}
    cross = crossmouse_scores(model, queries)
    link = glyph_posterior(cross, temperature["temperature"], balance=True, iterations=BALANCE_STEPS)
    peer = glyph_posterior(logit_scale(estimates["mice1"]), PEER_IDENTITY_TEMPERATURE, balance=True,
                           iterations=BALANCE_STEPS)
    random_indices = torch.as_tensor(rng.permutation(count))
    baseline = estimates["mice2"]
    scaled = logit_scale(baseline)
    measurements = {"v9": baseline,
                    "aligned_peer": scaled + PEER_WEIGHT * logit_scale((link @ peer).clamp_min(1e-12).log()),
                    "shuffled_peer": scaled + PEER_WEIGHT * logit_scale((link @ peer[random_indices]).clamp_min(1e-12).log())}
    choices = {name: assign_glyphs(scores) for name, scores in measurements.items()}

    # This separately labeled diagnostic estimates the headroom if the
    # cross-mouse match were known. It never affects the other actions.
    true_peer_rows = order["mice1"].argsort()[order["mice2"]]
    oracle = scaled + PEER_WEIGHT * logit_scale(peer[true_peer_rows].clamp_min(1e-12).log())
    choices["labeled_peer_diagnostic"] = assign_glyphs(oracle)
    query_rows = order["mice2"].argsort()[samples]
    selected = torch.stack([values[query_rows] for values in choices.values()])
    exact, unique = unique_pair_ocf(selected, samples.expand_as(selected), masks, scorer, images.device, 384)
    by_name = dict(zip(choices, exact.cpu().numpy()))
    report = {
        "session": held_day,
        "peer_correspondence": {"raw": row_diagnostics(cross, true_peer_rows),
                                "balanced": row_diagnostics(link, true_peer_rows), "temperature_fit": temperature},
        "mice1_identity": row_diagnostics(estimates["mice1"], order["mice1"]),
        "mice2_identity": {name: row_diagnostics(scores, order["mice2"]) for name, scores in measurements.items()},
        "actions": {name: {**summarize([by_name[name]], [by_name["v9"]]),
                            "identity_top1": float((values == order["mice2"]).float().mean()),
                            "changed_from_v9": int((values != choices["v9"]).sum()),
                            "unique_glyphs": int(values.unique().numel())}
                    for name, values in choices.items()},
        "unique_exact_ocf_pairs": unique, "seconds": time.monotonic() - started,
    }
    return report, by_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-days", nargs="+", choices=SESSIONS, default=("session1", "session3"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--split-seed", type=int, default=17691)
    parser.add_argument("--ocf-images", type=int, default=160)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--retrieval-epochs", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_peer_mouse.json"))
    args = parser.parse_args()
    if min(args.ocf_images, args.encoder_epochs, args.retrieval_epochs) < 2:
        parser.error("Configure at least two paired images and sufficient training epochs.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run with Kaggle competition input.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, torch.device("cpu"))
    masks = images >= 128
    catalog = load_catalog(root)
    scorer = official_scorer(root)
    recordings = {mouse: grouped_blocks(root, catalog, mouse, images.device, len(images)) for mouse in MICE}
    sample = torch.as_tensor(np.random.default_rng(args.split_seed).permutation(len(images))[:args.ocf_images])
    report = {"model_seed": args.seed, "split_seed": args.split_seed, "query_mouse": "mice2", "peer_mouse": "mice1",
              "ocf_rows_per_recording": len(sample), "peer_weight": PEER_WEIGHT,
              "peer_identity_temperature": PEER_IDENTITY_TEMPERATURE,
              "protocol": "Both animals' same-named recording blocks are held out independently. Permute their "
                          "queries separately. Fit the cross-mouse coordinates only from other days using "
                          "documented training glyph correspondences. Propagate the independent peer's own "
                          "glyph scores via spike-only soft correspondence, and compare with its row permutation. "
                          "Keep one-to-one decisions. The labeled diagnostic alone replaces the learned "
                          "correspondence with the held training glyph correspondence. No official test input.", "days": {}}
    metrics = {}
    for day in dict.fromkeys(args.held_days):
        day_report, values = evaluate(day, recordings, images, masks, scorer, sample, args)
        report["days"][day] = day_report
        for name, observations in values.items():
            metrics.setdefault(name, []).append(observations)
        report["summary"] = {name: summarize(observations, metrics["v9"]) for name, observations in metrics.items()}
        report["aligned_vs_shuffled"] = summarize(metrics["aligned_peer"], metrics["shuffled_peer"])
        report["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(day_report), flush=True)


if __name__ == "__main__":
    main()
