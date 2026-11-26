"""Test full-atlas cross-day pairing on separate matching and selection repeats."""

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_font_style import fit_aligned_font_style
from glyph_mbr import contour_signatures
from glyph_multiview_alignment import fit_multiview, pair_similarity
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import (FontAtlas, align_training_glyphs, build_font_atlas, image_descriptors,
                                       standard_error, standardized)
from kaggle_task1_microns_repeat_recheck import fresh_cohort
from kaggle_task1_peer_transport import choose_split
from kaggle_task1_repertoire_coverage import shape_matches


MOUSE = "mice1"
SESSIONS = ("session1", "session3")
SEED = 95233
HELD = 320
OCF_SAMPLE = 80
NEIGHBORS = 24
DEPTH = 16
TEMPERATURE = 1.0
ATLAS_WEIGHT = 0.50
OUTPUT = Path("/kaggle/working/task1_joint_atlas_pairing.json")


def holdout(size, device):
    previous = np.union1d(choose_split(size, device)[1].numpy(), fresh_cohort(size, device)[1].numpy())
    eligible = np.setdiff1d(np.arange(size), previous)
    held = np.sort(np.random.default_rng(SEED).choice(eligible, size=HELD, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    return torch.from_numpy(fit), torch.from_numpy(held), len(previous)


def split_trials(groups, held, rng):
    shuffled, order = {}, {}
    for session, counts in groups.items():
        order[session] = torch.from_numpy(rng.permutation(len(held)))
        rows = counts[held[order[session]]]
        repeats = torch.from_numpy(rng.random((len(rows), 4)).argsort(axis=1))
        shuffled[session] = rows.gather(1, repeats[..., None].expand_as(rows))
    return shuffled, order


def assignment(score):
    _, columns = linear_sum_assignment(-score.numpy())
    return torch.from_numpy(columns)


def full_atlas_agreement(left, right, rows, columns):
    left, right = left / TEMPERATURE, right / TEMPERATURE
    left_normalizer, right_normalizer = left.logsumexp(dim=1), right.logsumexp(dim=1)
    log_columns = np.log(left.shape[1])
    agreement = []
    for row, column in zip(rows.split(32), columns.split(32)):
        common = (left[row] + right[column]).logsumexp(dim=1)
        agreement.append(common - left_normalizer[row] - right_normalizer[column] + log_columns)
    return torch.cat(agreement)


def zscore(values):
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-5)


def make_links(spike, matching, rng):
    rows = torch.arange(len(spike))
    spike_links = assignment(spike)
    eligible = torch.zeros_like(spike, dtype=torch.bool)
    eligible.scatter_(1, spike.topk(NEIGHBORS, dim=1).indices, True)
    eligible.scatter_(0, spike.topk(NEIGHBORS, dim=0).indices, True)
    eligible[rows, spike_links] = True
    left, right = eligible.nonzero(as_tuple=True)
    spike_edges = zscore(spike[left, right])
    atlas_edges = zscore(full_atlas_agreement(*matching, left, right))
    shuffled = torch.empty_like(atlas_edges)
    for bin_edges in spike_edges.argsort().tensor_split(8):
        shuffled[bin_edges] = atlas_edges[bin_edges[torch.from_numpy(rng.permutation(len(bin_edges)))]]

    combined, null = torch.full_like(spike, -torch.inf), torch.full_like(spike, -torch.inf)
    combined[left, right] = spike_edges + ATLAS_WEIGHT * atlas_edges
    null[left, right] = spike_edges + ATLAS_WEIGHT * shuffled
    joint = assignment(combined)

    # A unique optimum should map back to the same links after independent relabeling.
    row_order, col_order = (torch.from_numpy(rng.permutation(len(rows))) for _ in range(2))
    permuted = col_order[assignment(combined[row_order][:, col_order])][row_order.argsort()]
    permuted_gap = float((combined[rows, joint] - combined[rows, permuted]).sum().abs())
    if permuted_gap > 1e-4:
        raise RuntimeError("Relabeling rows or columns changed the matching objective.")
    links = {"spike_only": spike_links, "atlas_conditioned": joint, "matched_shuffled": assignment(null)}
    diagnostics = {"eligible_edges": int(eligible.sum()), "relabeling_objective_gap": permuted_gap,
                   "relabeling_changed_links": int((permuted != joint).sum())}
    return links, eligible, diagnostics


def top16(scores, prior):
    return prior.variants(scores)["usage_unseen"].topk(DEPTH, dim=1).indices


def summarize(choices, solo, full_repeat, matches, score, samples):
    confident = matches.confident(0.55, 0.01)
    known_column = matches.closest
    both_solo = torch.stack(solo)
    results, exact = {}, {}
    for arm, variants in choices.items():
        outputs = torch.stack(variants)
        selected = outputs[..., 0]
        novel = (outputs[..., None] != both_solo[:, None, :, None]).all(dim=-1)
        novel_full = (selected[..., None] != full_repeat[None, None]).all(dim=-1)
        hit = outputs == known_column[None, None, :, None]
        exact[arm] = np.stack([score.evaluate(rows[samples, :1])[:, 0]
                               for rows in outputs.flatten(0, 1)]).reshape(2, 2, -1)
        results[arm] = {
            "exact_ocf": float(exact[arm].mean()),
            "exact_ocf_by_matching_half_and_day": exact[arm].mean(axis=-1).tolist(),
            "top16_proxy_recall": float(hit[:, :, confident].any(dim=-1).float().mean()) if confident.any() else None,
            "top1_outside_both_selection_half_top16_rate": float(novel[..., 0].float().mean()),
            "top1_outside_both_single_day_four_repeat_top16_rate": float(novel_full.float().mean()),
            "avg_top16_proposals_outside_both_selection_half_top16": float(novel.sum(dim=-1).float().mean()),
            "top16_proxy_newly_recovered_outside_both_selection_half_top16_rate":
                float((hit & novel)[:, :, confident].any(dim=-1).float().mean()) if confident.any() else None,
        }
    contrasts = (("atlas_conditioned", "spike_only"), ("atlas_conditioned", "matched_shuffled"),
                 ("atlas_conditioned", "no_link"), ("atlas_conditioned", "no_link_own4"),
                 ("known_true_grouping", "no_link_own4"))
    deltas = {}
    for candidate, control in contrasts:
        paired = (exact[candidate] - exact[control]).mean(axis=(0, 1))
        deltas[f"{candidate}_vs_{control}"] = {"mean": float(paired.mean()), "se": standard_error(paired)}
    return {"arms": results, "paired_exact_ocf_deltas": deltas,
            "confident_held_render_proxies": int(confident.sum()), "unique_exact_ocf_pairs": len(score.values)}


def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("This experiment runs in Kaggle.")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED + 1)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    fit, held, excluded = holdout(len(images), device)
    catalog = load_catalog(root)
    catalog["blocks"] = [entry for entry in catalog["blocks"]
                         if entry["task"] == "train" and entry["session"] in SESSIONS]
    groups = grouped_blocks(root, catalog, MOUSE, device, len(images))
    if tuple(groups) != SESSIONS or any(counts.shape[1] != 4 for counts in groups.values()):
        raise ValueError("Expected both labeled sessions with four repeats per fitting and held glyph.")
    training = {day: rows[fit] for day, rows in groups.items()}
    queries, orders = split_trials(groups, held, rng)
    inverse = tuple(orders[day].argsort() for day in SESSIONS)
    del groups

    print(f"Fit public atlas, alignment and contour readout on {len(fit)} disjoint glyphs", flush=True)
    train_images = images[fit]
    style_subset = rng.choice(len(fit), 128, replace=False)
    atlas, provenance, codepoints = build_font_atlas(train_images[style_subset], device, return_codepoints=True)
    aligned, fit_agreement = align_training_glyphs(atlas, train_images)
    prior = fit_atlas_prior(atlas, codepoints, train_images)
    style, style_fit = fit_aligned_font_style(aligned, train_images, seed=SEED)
    rendered = torch.cat([(style.apply(rows[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                          for rows in atlas.images.split(1024)])
    contour = fit_contour_retrieval(training, *contour_signatures(aligned), task="task2", seed=SEED)
    signature, energy = contour_signatures(atlas.images)
    contour = replace(contour, signature=signature, energy=energy)
    alignment = fit_multiview({(MOUSE, day): counts for day, counts in training.items()})
    del training, aligned

    print(f"Score both halves and all four repeats over {len(rendered):,} public atlas columns", flush=True)
    unary, embeddings, own4 = [], [], []
    for day, source in queries.items():
        halves = source.split(2, dim=1)
        unary.append(tuple(standardized(contour_retrieval_scores(contour, half, day)["contour_ridge"])
                           for half in halves))
        embeddings.append(tuple(alignment.encode((MOUSE, day), half) for half in halves))
        own4.append(top16(contour_retrieval_scores(contour, source, day)["contour_ridge"], prior))
    del queries, contour, signature, energy, alignment, atlas
    own4 = torch.stack(tuple(rows[restore] for rows, restore in zip(own4, inverse)))

    true_links = inverse[1][orders[SESSIONS[0]]]
    left_rows = torch.arange(HELD)
    choices, solo, links_by_half = {}, [], []
    for matching_half in range(2):
        matching = tuple(day[matching_half] for day in unary)
        selection = tuple(day[1 - matching_half] for day in unary)
        spike = pair_similarity(*(day[matching_half] for day in embeddings))
        links, eligible, diagnostics = make_links(spike, matching, rng)
        diagnostics["true_link_graph_recall"] = float(eligible[left_rows, true_links].float().mean())
        links["known_true_grouping"] = true_links
        solo_top = torch.stack(tuple(top16(scores, prior)[restore] for scores, restore in zip(selection, inverse)))
        solo.append(solo_top.transpose(0, 1).reshape(HELD, -1))
        choices.setdefault("no_link", []).append(solo_top)
        choices.setdefault("no_link_own4", []).append(own4)
        link_stats = {}
        for name, columns in links.items():
            combined = top16((selection[0] + selection[1][columns]) / 2, prior)
            choices.setdefault(name, []).append(torch.stack((combined[inverse[0]],
                                                             combined[columns.argsort()[inverse[1]]])))
            left_peaks, right_peaks = matching[0].argmax(dim=1), matching[1][columns].argmax(dim=1)
            same_peak = left_peaks == right_peaks
            correct = columns == true_links
            link_stats[name] = {"accuracy": float(correct.float().mean()),
                                "shared_matching_half_top1_fraction": float(same_peak.float().mean()),
                                "accuracy_with_shared_matching_half_top1": float(correct[same_peak].float().mean())
                                if same_peak.any() else None,
                                "accuracy_without_shared_matching_half_top1": float(correct[~same_peak].float().mean())
                                if (~same_peak).any() else None}
        diagnostics.update(matching_half=matching_half, arms=link_stats)
        links_by_half.append(diagnostics)
        print(json.dumps(diagnostics), flush=True)
    del unary

    # The selection process above never reads held rasters or visual proxy columns.
    print("Calculate held render-proxy recall and exact paired official OCF", flush=True)
    render_atlas = FontAtlas(rendered, torch.cat([image_descriptors(rows) for rows in rendered.split(1024)]))
    matches = shape_matches(render_atlas, images[held])
    sample = torch.from_numpy(rng.choice(HELD, OCF_SAMPLE, replace=False))
    score = PairedOCF(official_scorer(root), rendered, (images[held[sample]] >= 128).numpy(), device, 12)
    full_repeat = own4.transpose(0, 1).reshape(HELD, -1)
    results = summarize(choices, solo, full_repeat, matches, score, sample)
    result = {
        "seed": SEED, "mouse": MOUSE, "sessions": SESSIONS, "fit_glyphs": len(fit), "held_glyphs": len(held),
        "excluded_previous_peer_and_microns_cohort_glyphs": excluded, "ocf_glyphs": OCF_SAMPLE,
        "atlas_columns": len(rendered), "atlas": provenance, "style_fit": style_fit,
        "fit_alignment_agreement": fit_agreement, "prior": prior.diagnostics, "atlas_weight": ATLAS_WEIGHT,
        "atlas_temperature": TEMPERATURE, "link_stats_by_matching_half": links_by_half, **results,
        "protocol": "Two independently shuffled labeled recording days. Match using two held repeats per day, "
                    "select with the complementary two and swap halves. True pairing appears only in the "
                    "oracle and diagnostics. All methods use a single fit-only lexical/unseen prior at output. "
                    "Atlas edge agreement covers every column under a uniform reference distribution. The "
                    "sham permutes agreement within eight equal-count spike-score bins on the same edges. "
                    "Neural and atlas evidence from the matching half are correlated, and agreement scores "
                    "can reward the same high-scoring render on both days. Render proxy recall measures "
                    "agreement with a high-confidence visual neighbor, since held character codes are unavailable.",
        "seconds": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arms": results["arms"], "paired": results["paired_exact_ocf_deltas"],
                      "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
