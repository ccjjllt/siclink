"""Test background-calibrated shared atlas support on whole held training glyphs."""

import argparse
import json
from math import log
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import pack_atlas_candidates
from glyph_atlas_prior import _aligned_overlap
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_gallery_validation import image_descriptors
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_task1_half_proposal_validation import ANCHOR_OCF, MODEL_SEED, twos
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank, summarize


EARLIER_OUTER_SEEDS = (961, 2209, 3931, 4931, 5147)
BACKGROUND_GLYPHS = 384
SUPPORT_SIZE = 320
SUPPORT_BONUS_OCF = 2.0
TAIL_PROBABILITY = 0.025
NULL_EXCEEDANCES_PER_COLUMN = 0.20
MAX_BLOCK_EVIDENCE = 6.0
VISUAL_NEIGHBORS = 24
VISUAL_QUALITY = 0.70
VISUAL_MARGIN = 0.03


def half_scores(model, likelihood, counts, session, rates, seed):
    """Get the same neural score used for v8 before its public atlas prior."""
    halves = twos(counts, seed)
    prior = 0.55 * model.prior.lexical_log + model.prior.unseen_log
    return tuple(atlas_scores(model, likelihood, half, session, rates=rates)["v8"] - prior for half in halves)


def standardized_joint(background, observed, column_specific):
    """Fit half-score scales on background rows separate from the joint null tail."""
    bg_joint, observed_joint = None, None
    for fit, queries in zip(background, observed):
        location_rows, tail_rows = fit.chunk(2, dim=0)
        source = location_rows if column_specific else location_rows.flatten()
        lower, median, upper = torch.quantile(source, source.new_tensor((0.25, 0.50, 0.75)), dim=0)
        inverse_scale = 1.349 / (upper - lower).clamp_min(0.135)
        normalized_tail = (tail_rows - median) * inverse_scale
        normalized_query = (queries - median) * inverse_scale
        bg_joint = normalized_tail if bg_joint is None else torch.minimum(bg_joint, normalized_tail)
        observed_joint = normalized_query if observed_joint is None else torch.minimum(observed_joint, normalized_query)
    return bg_joint, observed_joint


def block_evidence(background, observed, column_specific):
    """Require both held halves to exceed an independent, recording-specific null."""
    bg_joint, joint = standardized_joint(background, observed, column_specific)
    source = bg_joint if column_specific else bg_joint.flatten()
    lo, upper = torch.quantile(source, source.new_tensor((0.90, 1 - TAIL_PROBABILITY)), dim=0)
    tail_scale = ((upper - lo) / log(0.10 / TAIL_PROBABILITY)).clamp_min(0.15)
    threshold = upper + tail_scale * log(len(joint) * TAIL_PROBABILITY / NULL_EXCEEDANCES_PER_COLUMN)
    # The maximum can be supplied by any anonymous row, at most once per column and recording.
    delta = ((joint.amax(dim=0) - threshold) / tail_scale).clamp(0, MAX_BLOCK_EVIDENCE)
    report = {"columns_with_evidence": int((delta > 0).sum()), "maximum_evidence": float(delta.max()),
              "background_tail_threshold_median": float(threshold.median()),
              "background_null_exceedance_scale": NULL_EXCEEDANCES_PER_COLUMN}
    return delta, report


@torch.no_grad()
def recording_data(model, ranker, likelihood, background, held, session, seed):
    with torch.autocast(device_type="cuda"):
        rates = model.encoder.expected_counts(model.atlas.images, session, batch_size=192).float()
    full = atlas_scores(model, likelihood, held, session, rates=rates)
    candidates = pack_atlas_candidates(full, NOMINATIONS, baseline="v8")
    del full
    objective = ranker.rank(candidates, anchor_ocf=ANCHOR_OCF)
    calibration = half_scores(model, likelihood, background, session, rates, seed)
    evidence = half_scores(model, likelihood, held, session, rates, seed + 1)
    delta, stats = {}, {}
    for name, column_specific in (("calibrated", True), ("pooled_null", False)):
        delta[name], stats[name] = block_evidence(calibration, evidence, column_specific)
    return candidates, objective, delta, stats


def strata_for_shuffle(model, rendered):
    ink = torch.cat([(part >= 128).float().mean(dim=(1, 2)) for part in rendered.split(1024)])
    def quartiles(values):
        return torch.bucketize(values.contiguous(), torch.quantile(values, values.new_tensor((0.25, 0.5, 0.75))))

    return (quartiles(model.prior.lexical_log) * 4 + quartiles(model.prior.unseen_log)) * 4 + quartiles(ink)


def shuffle_columns(scores, strata, rng):
    shuffled = torch.empty_like(scores)
    for stratum in strata.unique():
        columns = (strata == stratum).nonzero()[:, 0]
        permuted = torch.as_tensor(rng.permutation(len(columns)), device=columns.device)
        shuffled[columns] = scores[columns[permuted]]
    return shuffled


def peer_support(block, votes, lexical, strata, rng):
    """Build a fixed-size support estimate using only the other recordings."""
    names = [session for session in votes if session != block]
    tie_prior = 1e-3 * lexical + torch.as_tensor(rng.normal(0, 1e-6, len(lexical)), device=lexical.device,
                                               dtype=lexical.dtype)
    scores = {key: torch.stack([votes[peer][key] for peer in names]).sum(0) + tie_prior
              for key in ("calibrated", "pooled_null")}
    scores["shuffled"] = shuffle_columns(scores["calibrated"], strata, rng)
    scores["public_prior"] = tie_prior
    rankings, supports = {}, {}
    for name, score in scores.items():
        order = score.argsort(descending=True, stable=True)
        rankings[name] = order.argsort()
        selected = torch.zeros_like(score, dtype=torch.bool)
        selected[order[:SUPPORT_SIZE]] = True
        supports[name] = selected
    return names, supports, rankings


def choose(candidates, objective, supports):
    choices = {"v11": candidates.indices.gather(1, objective.argmax(1, keepdim=True))[:, 0]}
    matched_edges = {}
    for name, columns in supports.items():
        matches = columns[candidates.indices] & candidates.valid
        updated = objective + SUPPORT_BONUS_OCF * matches
        choices[name] = candidates.indices.gather(1, updated.argmax(1, keepdim=True))[:, 0]
        matched_edges[name] = {"fixed_list_columns": int(columns.sum()), "supported_candidate_edges": int(matches.sum()),
                               "queries_with_supported_edges": int(matches.any(1).sum()),
                               "changed_full_winners": int((choices[name] != choices["v11"]).sum())}
    return choices, matched_edges


@torch.no_grad()
def visual_proxy(rendered, reference):
    """Map held images to public shapes strictly after all spike-based decisions."""
    contours = torch.cat([image_descriptors(batch) for batch in rendered.split(1024)])
    columns, similarity, consensus = [], [], []
    for masks in reference.split(32):
        cosine, indices = (image_descriptors(masks.to(torch.uint8) * 255) @ contours.T).topk(VISUAL_NEIGHBORS, dim=1)
        overlap = _aligned_overlap(masks[:, None], rendered[indices] >= 128)
        scores, order = (0.65 * cosine + 0.35 * overlap).sort(dim=1, descending=True)
        columns.append(indices.gather(1, order))
        similarity.append(scores)
        consensus.append((order[:, 0] == 0) & (order[:, 0] == overlap.argmax(1)))
    columns, similarity, consensus = torch.cat(columns), torch.cat(similarity), torch.cat(consensus)
    agree = (consensus & (similarity[:, 0] >= VISUAL_QUALITY)
             & ((similarity[:, 0] - similarity[:, 1]) >= VISUAL_MARGIN))
    repeated = torch.bincount(columns[:, 0], minlength=len(rendered)) > 1
    confidence = agree & ~repeated[columns[:, 0]]
    return columns, similarity, confidence


def support_summary(selected, rank, proxy):
    columns, similarity, confidence = proxy
    nearest = columns[:, 0]
    near = similarity >= similarity[:, :1] - 0.035
    support_rank = rank[nearest] + 1
    full_count = int(selected.sum())
    available = float(selected.float().mean())
    matched = int((selected[nearest] & confidence).sum())
    nearest_recall = float(selected[nearest].float().mean())
    high_confidence = int(confidence.sum())
    return {"nearest_shape_recall_at_320": nearest_recall,
            "nearest_shape_enrichment_over_uniform": nearest_recall / available,
            "near_shape_recall_at_320": float((selected[columns] & near).any(1).float().mean()),
            "nearest_shape_median_rank": float(support_rank.float().median()),
            "nearest_shape_mean_reciprocal_rank": float(support_rank.float().reciprocal().mean()),
            "high_confidence_reference_count": high_confidence, "high_confidence_support_count": matched,
            "high_confidence_recall_at_320": matched / high_confidence if high_confidence else None,
            "high_confidence_support_precision_proxy": matched / full_count}


def comparable_ocf(observed):
    result = {name: summarize(values, observed["v11"]) for name, values in observed.items()}
    result["calibrated_vs_pooled_null"] = summarize(observed["calibrated"], observed["pooled_null"])
    result["calibrated_vs_shuffled"] = summarize(observed["calibrated"], observed["shuffled"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=8081)
    parser.add_argument("--ocf-images", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_shared_support_validation.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_images <= SUPPORT_SIZE:
        parser.error("Sample between two and 320 held images for paired OCF.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on a private Kaggle GPU worker with the competition input and Internet enabled.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fitting, held = outer_split(len(images), args.seed, EARLIER_OUTER_SEEDS, device)
    rng = np.random.default_rng(args.seed + 1)
    bg_order = torch.as_tensor(rng.permutation(len(fitting)), device=device)
    background, fit = fitting[bg_order[:BACKGROUND_GLYPHS]], fitting[bg_order[BACKGROUND_GLYPHS:]].sort().values
    sample = torch.as_tensor(rng.choice(len(held), args.ocf_images, replace=False), device=device)
    recording = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if tuple(recording) != SESSIONS or any(values.shape[1] != 4 for values in recording.values()):
        raise ValueError("Expected four mice1 source recordings with four trials per labeled glyph.")
    scorer = official_scorer(root)
    training = {name: values[fit] for name, values in recording.items()}
    split = split_calibration(len(fit), 96, 384, device, MODEL_SEED)
    ranker, likelihood, nested, codepoints = nested_ranker(training, images[fit], split, scorer, MODEL_SEED)
    model = fit_v8(training, images[fit], MODEL_SEED)
    del training
    if not torch.equal(model.codepoints.cpu(), codepoints):
        raise ValueError("Nested and outer public candidate columns must match exactly.")
    rendered = rendered_bank(model)
    output = {"mouse": "mice1", "outer_seed": args.seed, "model_seed": MODEL_SEED, "fit_glyphs": len(fit),
              "background_glyphs": len(background), "held_glyphs": len(held), "repeats_per_glyph": 4,
              "older_excluded_outer_seeds": EARLIER_OUTER_SEEDS, "paired_per_recording": len(sample),
              "atlas_columns": len(rendered), "support_columns_per_arm": SUPPORT_SIZE, "support_bonus_ocf": SUPPORT_BONUS_OCF,
              "v11_anchor_ocf": ANCHOR_OCF, "v11_nomination": NOMINATIONS, "ranker": ranker.diagnostics,
              "nested": nested, "sessions": {},
              "protocol": "Fit glyphs, background glyphs and outer glyphs are disjoint. The independent halves of "
                          "each background query estimate location and scale on 192 groups and the joint tail on "
                          "another 192 groups, per recording. One recording's maximum joint evidence per atlas column "
                          "is bounded. Only the other three recordings contribute to its estimated support. Compare "
                          "column-specific background with an atlas-pooled null, score permutations in matching prior/ink "
                          "strata, and public prior, at exactly 320 columns and equal bonus on identical v11 row edges. "
                          "All recordings have independently shuffled anonymous query rows. No query correspondence, "
                          "sample ID, known evaluation label or official test block enters decisions. Held reference "
                          "images are used after decisions for visual support proxies and exact OCF; official "
                          "training images have no known Unicode identity."}

    order, blocks, votes, background_report = {}, {}, {}, {}
    for index, name in enumerate(SESSIONS):
        order[name] = torch.as_tensor(rng.permutation(len(held)), device=device)
        candidates, objective, votes[name], background_report[name] = recording_data(
            model, ranker, likelihood[name], recording[name][background], recording[name][held[order[name]]],
            name, args.seed + 1000 + 10 * index)
        blocks[name] = candidates, objective
        print(json.dumps({"recording_evidence_ready": name, "background": background_report[name]}), flush=True)

    strata = strata_for_shuffle(model, rendered)
    lexical = 0.55 * model.prior.lexical_log + model.prior.unseen_log
    decisions = {}
    for name, (candidates, objective) in blocks.items():
        peers, supports, ranks = peer_support(name, votes, lexical, strata, rng)
        choices, changes = choose(candidates, objective, supports)
        decisions[name] = choices, changes, peers, supports, ranks

    # Nothing above this point uses the held images.
    references = images[held] >= 128
    proxy = visual_proxy(rendered, references)
    output["visual_proxy"] = {"neighbors": VISUAL_NEIGHBORS, "min_similarity": VISUAL_QUALITY,
                             "min_margin": VISUAL_MARGIN, "high_confidence_count": int(proxy[2].sum())}
    exact = PairedOCF(scorer, rendered, references[sample].cpu().numpy(), device, 12)
    all_observed = {}
    for name, (choices, changes, peers, supports, ranks) in decisions.items():
        rows = order[name].argsort()[sample]
        observed = {arm: [exact.evaluate(indices[rows, None])[:, 0]] for arm, indices in choices.items()}
        for arm, scores in observed.items():
            all_observed.setdefault(arm, []).extend(scores)
        report = {"source_recordings": peers, "ocf": comparable_ocf(observed), "equal_budget_edges": changes,
                  "recording_evidence": background_report[name], "v11_candidate_edges": int(blocks[name][0].valid.sum()),
                  "support": {arm: support_summary(columns, ranks[arm], proxy) for arm, columns in supports.items()}}
        output["sessions"][name] = report
        output["summary"] = comparable_ocf(all_observed)
        output["unique_exact_pairs"] = len(exact.values)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"held_recording": name, **report}), flush=True)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
