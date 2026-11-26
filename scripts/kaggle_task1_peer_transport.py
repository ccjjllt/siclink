"""Test cross-recording spike evidence transport with independent repeat halves."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import pack_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_dual_alignment import standardized_pair
from glyph_mbr import fit_temperature, glyph_posterior
from glyph_multiview_alignment import fit_multiview, pair_similarity
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_half_proposal_validation import twos
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE
from kaggle_task1_poisson_bridge import fresh_split
from kaggle_task1_raster_generation import CONTEXT_GLYPHS, FIT_GLYPHS, Raster, load_variable_font, rasterize
from kaggle_task1_v13_repeat_weight_recheck import independent_outer
from kaggle_task1_v11_repeat_weight_transfer import normalized_sources
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank, summarize


MOUSE = "mice1"
SEED = 27367
HELD = 320
LINK_CALIBRATION = 256
MAX_PARTNERS = 12
PEER_WEIGHT = 3.0
ARMS = ("v14", "peer", "rate_stratified_shuffle", "known_correspondence")
OUTPUT = Path("/kaggle/working/task1_peer_transport.json")
RASTER_SEED = 32507
RASTER_PREVIOUS_SCORE_GLYPHS = 160
RASTER_FONT_SHA = "d13ed01ec8aa45d6178999b648e96fb92150683e9f8e2a581f2acf208dcbe44b"
RASTER_RECIPE = Raster(1, "NATIVE", 400, 50, 1.012682910597594, 0.08227016031742096, -2.6884519026829645)


def choose_split(size, device):
    cpu = torch.device("cpu")
    _, bridge_outer, _ = fresh_split(size, cpu)
    _, previous_outer, _ = independent_outer(size, cpu)
    raster_order = np.random.default_rng(RASTER_SEED).permutation(size)
    raster_used = raster_order[:CONTEXT_GLYPHS + FIT_GLYPHS + RASTER_PREVIOUS_SCORE_GLYPHS]
    omitted = np.unique(np.concatenate((bridge_outer.numpy(), previous_outer.numpy(), raster_used)))
    eligible = np.setdiff1d(np.arange(size), omitted)
    rng = np.random.default_rng(SEED)
    held = np.sort(rng.choice(eligible, size=HELD, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    calibration = rng.choice(fit, LINK_CALIBRATION, replace=False)
    inner = np.setdiff1d(fit, calibration)
    return tuple(torch.as_tensor(rows, device=device) for rows in (fit, held, inner, calibration))


def permute_within_rate(counts, rng):
    rate_order = counts.sum(dim=(1, 2)).argsort()
    donors = torch.empty_like(rate_order)
    for rows in rate_order.tensor_split(8):
        rows = rows[torch.as_tensor(rng.permutation(len(rows)), device=rows.device)]
        donors[rows] = rows.roll(1)
    return donors


@torch.no_grad()
def pair_logits(left, right):
    return standardized_pair(pair_similarity(left, right))


def link_model(recordings, inner, calibration):
    alignment = fit_multiview({(MOUSE, day): counts[inner] for day, counts in recordings.items()})
    embeddings = {day: tuple(alignment.encode((MOUSE, day), part) for part in twos(counts[calibration], SEED + index))
                  for index, (day, counts) in enumerate(recordings.items())}
    calibration_truth = torch.arange(len(calibration), device=calibration.device)
    temperatures = {}
    for target in SESSIONS:
        for peer in SESSIONS:
            if target == peer:
                continue
            pairs = [(pair_logits(a, b), calibration_truth) for a, b in zip(embeddings[target], embeddings[peer])]
            temperatures[target, peer] = fit_temperature(pairs)
    return alignment, temperatures


@torch.no_grad()
def observed_score(model, likelihood, counts, day, alignment, rng, seed):
    embed, evidence, shuffle = [], [], []
    with torch.autocast(device_type="cuda"):
        rates = model.encoder.expected_counts(model.atlas.images, day, batch_size=192).float()
    all_scores = atlas_scores(model, likelihood, counts, day, rates=rates)
    candidates = pack_atlas_candidates(all_scores, NOMINATIONS, baseline="v8")
    del all_scores
    for half in twos(counts, seed):
        embed.append(alignment.encode((MOUSE, day), half))
        raw = atlas_scores(model, likelihood, half, day, rates=rates)
        normalized = normalized_sources(raw)["poisson_mix"]
        # Remove atlas columns that attract matching scores for virtually any
        # query in the recording. No shape or availability prior is transported.
        normalized -= normalized.mean(dim=0)
        evidence.append(normalized.clamp(-4, 4).half())
        shuffle.append(permute_within_rate(half, rng))
    return candidates, embed, evidence, shuffle


@torch.no_grad()
def peer_additions(target, peer, observations, orders, temperatures):
    graph, target_embed, _, _ = observations[target]
    _, peer_embed, peer_score, donors = observations[peer]
    indices = graph.indices
    matched_truth = orders[peer].argsort()[orders[target]]
    result = {name: indices.new_zeros(indices.shape, dtype=torch.float32) for name in ARMS[1:]}
    summaries = []
    temperature = temperatures[target, peer]["temperature"]
    for match_half, scoring_half in ((0, 1), (1, 0)):
        prob = glyph_posterior(pair_logits(target_embed[match_half], peer_embed[match_half]), temperature, balance=True)
        weights, partners = prob.topk(MAX_PARTNERS, dim=1)
        field = peer_score[scoring_half]
        for name, links in (("peer", partners), ("rate_stratified_shuffle", donors[scoring_half][partners])):
            selected = field[links[:, :, None], indices[:, None, :]].float()
            result[name] += 0.5 * (weights[:, :, None] * selected).sum(dim=1)
        result["known_correspondence"] += 0.5 * field[matched_truth[:, None], indices].float()
        summaries.append({"true_top1": float((partners[:, 0] == matched_truth).float().mean()),
                          "true_top12": float((partners == matched_truth[:, None]).any(dim=1).float().mean()),
                          "true_mass": float(prob.gather(1, matched_truth[:, None]).mean()),
                          "mean_mass_in_top12": float(weights.sum(dim=1).mean())})
    return result, summaries


def fixed_choice_rendering(model, atlas, observed, ranker, reference, scorer, device):
    stock_choices = {day: ranker.select(observed[day][0], anchor_ocf=ANCHOR) for day in SESSIONS}
    selected = torch.cat(tuple(stock_choices.values())).unique(sorted=True).cpu()
    codes = model.codepoints.cpu()[selected]
    content, supported, source = load_variable_font()
    if source["sha256"] != RASTER_FONT_SHA or any(code not in supported for code in codes.tolist()):
        raise ValueError("The fixed raster experiment needs the verified variable font and all selected glyph columns.")
    alternate = rasterize(content, codes, RASTER_RECIPE).to(device)
    old = atlas[selected.to(device)]
    changed_pixels = (alternate >= 128).flatten(1) != (old >= 128).flatten(1)
    alternate_exact = PairedOCF(scorer, alternate, reference, device, 12)
    return stock_choices, selected.to(device), alternate_exact, {
        "pinned_font": source, "recipe": asdict(RASTER_RECIPE), "unique_fixed_decoder_choices": len(selected),
        "unique_renders_with_different_pixels": int(changed_pixels.any(1).sum()),
        "mean_changed_binary_pixels": float(changed_pixels.sum(1).float().mean()),
    }


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run using Kaggle T4 and the official labeled competition data.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held, inner, calibration = choose_split(len(images), device)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    alignment, temperatures = link_model(recordings, inner, calibration)
    fit_counts = {day: counts[fit] for day, counts in recordings.items()}
    query = {day: counts[held] for day, counts in recordings.items()}
    del recordings
    scorer = official_scorer(root)
    split = split_calibration(len(fit), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    _, noise, _, codes = nested_ranker(fit_counts, images[fit], split, scorer, MODEL_SEED)
    ranker, folds = fit_crossfit_head(fit_counts, images[fit], scorer, codes)
    model = fit_v8(fit_counts, images[fit], MODEL_SEED)
    del fit_counts
    if not torch.equal(model.codepoints.cpu(), codes):
        raise ValueError("Expected identical calibrated and inference atlas columns.")
    atlas = rendered_bank(model)
    reference = (images[held] >= 128).cpu().numpy()
    exact = PairedOCF(scorer, atlas, reference, device, 12)
    rng = np.random.default_rng(SEED + 100)
    orders = {day: torch.as_tensor(rng.permutation(len(held)), device=device) for day in SESSIONS}
    observed = {day: observed_score(model, noise[day], query[day][orders[day]], day, alignment, rng, SEED + 20 + index)
                for index, day in enumerate(SESSIONS)}
    stock, selected, alternate_exact, raster_report = fixed_choice_rendering(
        model, atlas, observed, ranker, reference, scorer, device)
    del model, query
    report = {"mouse": MOUSE, "whole_held_glyphs": len(held), "calibration_glyphs": len(calibration),
              "match_fit_glyphs": len(inner), "seed": SEED, "peer_weight": PEER_WEIGHT,
              "held_disjoint_from_weight_recheck_and_poisson_bridge": True,
              "held_disjoint_from_raster_context_fit_and_previous_scoring": True,
              "top_partners": MAX_PARTNERS, "calibration_folds": folds,
              "temperature": {f"{a}->{b}": values for (a, b), values in temperatures.items()},
              "protocol": "Freeze v14 candidates and OCF ranker. Anonymous target/peer half-one spikes define "
                          "soft row correspondences. Peer half-two supplies fresh Poisson evidence on the same "
                          "public candidate columns. Swap halves and average. Rate-stratified permutation keeps "
                          "the correspondence weights fixed and changes only evidence donors. Known held glyph "
                          "correspondence is used only for retrospective diagnostics and the labeled ceiling.",
              "fixed_choice_raster_protocol": "Apply the locked grayscale-derived raster program to the exact same "
                                              "codepoints selected by the frozen v14 spike decoder. Keep scoring "
                                              "characters outside all previous raster fit and evaluation sets.",
              "fixed_choice_raster": raster_report, "sessions": {}}
    measured = {name: [] for name in ARMS}
    native_scores = []
    for day in SESSIONS:
        graph = observed[day][0]
        baseline = ranker.rank(graph, anchor_ocf=ANCHOR)
        parts, link_report = {}, {}
        for peer in SESSIONS:
            if peer != day:
                parts[peer], link_report[peer] = peer_additions(day, peer, observed, orders, temperatures)
        choices = {"v14": stock[day]}
        differences = {}
        for name in ARMS[1:]:
            offset = PEER_WEIGHT * torch.stack([arm[name] for arm in parts.values()]).mean(dim=0).clamp(-1, 1)
            winner = (baseline + offset).argmax(1, keepdim=True)
            choices[name] = graph.indices.gather(1, winner)[:, 0]
            differences[name] = float(offset[graph.valid].abs().mean())
        ordered = torch.stack([choices[name][orders[day].argsort()] for name in ARMS], dim=1)
        values = exact.evaluate(ordered)
        alternate_rows = torch.searchsorted(selected, ordered[:, :1].contiguous())
        native = alternate_exact.evaluate(alternate_rows)[:, 0]
        native_scores.append(native)
        for index, name in enumerate(ARMS):
            measured[name].append(values[:, index])
        report["sessions"][day] = {"links": link_report, "mean_absolute_offset": differences,
                                    "actual_changes": {name: int((choice != choices["v14"]).sum())
                                                       for name, choice in choices.items()},
                                    "ocf": {name: summarize([values[:, index]], [values[:, 0]])
                                            for index, name in enumerate(ARMS)},
                                    "fixed_choice_native_raster": summarize([native], [values[:, 0]])}
        report["combined_ocf"] = {name: summarize(rows, measured["v14"]) for name, rows in measured.items()}
        report["peer_minus_shuffle"] = summarize(measured["peer"], measured["rate_stratified_shuffle"])
        report["fixed_choice_raster"]["combined_ocf"] = summarize(native_scores, measured["v14"])
        report["unique_ocf_pairs"] = len(exact.values)
        report["seconds"] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": day, **report["sessions"][day]}), flush=True)


if __name__ == "__main__":
    main()
