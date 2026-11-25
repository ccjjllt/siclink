"""Test a fixed morphology/Unihan prior on paired mice1 Task 1 OCF."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_prior import _usage_log_enrichment, seen_glyph_evidence, unihan_usage
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import fit_glyph_encoder
from glyph_font_oracle import FONT_SOURCES, render_font
from glyph_font_style import fit_aligned_font_style
from glyph_mbr import contour_signatures
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_exact_rerank_validation import fit_direct
from kaggle_gallery_validation import FontAtlas, align_training_glyphs, font_scores, standard_error, standardized
from kaggle_lexical_support_validation import fit_atlas, fit_density, morphology


def candidate_budget(scores, budget, tie_order):
    """Keep all strict winners and draw uniformly from any boundary tie."""
    budget = min(len(scores), budget)
    cutoff = scores.topk(budget).values.min()
    kept = scores > cutoff
    strict = int(kept.sum())
    boundary = tie_order[scores[tie_order] == cutoff]
    chosen = budget - strict
    kept[boundary[:chosen]] = True
    return kept, {"budget": budget, "strict": strict, "boundary": len(boundary), "boundary_kept": chosen}


def paired_summary(measurements, budgets):
    base = np.stack(measurements["v8"])
    result = {}
    for name, rows in measurements.items():
        values = np.stack(rows)
        paired = (values - base).mean(axis=0)
        result[name] = {"ocf": float(values.mean()), "ocf_n": values.size,
                        "paired_delta_vs_v8": float(paired.mean()), "paired_glyph_se": standard_error(paired)}
    for number in budgets:
        current = np.stack(measurements[f"current_hard_{number}"])
        name = f"morphology_hard_{number}"
        difference = (np.stack(measurements[name]) - current).mean(axis=0)
        result[name].update(paired_delta_vs_matched_budget=float(difference.mean()),
                            paired_glyph_se_vs_matched_budget=standard_error(difference))
    return result


@torch.no_grad()
def score_spikes(atlas, encoder, contour, decoder, stats, counts, session):
    features, _ = standardize(feature_view(counts, "evoked-total"), stats[session])
    direct = decoder.predict_variants(session, features.mean(dim=1))
    with torch.autocast(device_type="cuda"):
        rates = encoder.expected_counts(atlas.images, session, batch_size=192).float()
    anchor = standardized(font_scores(atlas, rates, counts, direct)["font_fused"])
    directional = contour_retrieval_scores(contour, counts, session)["contour_ridge"]
    return standardized(0.60 * anchor + 0.40 * standardized(directional))


def run_split(record, raw_atlas, images, grouped, fit, held, tiers, pinlu, device, score, args, save):
    seed = record["seed"]
    stage = time.monotonic()
    fit_images_cpu = images[fit]
    atlas, alignment = fit_atlas(raw_atlas, fit_images_cpu, seed)
    seen, match = seen_glyph_evidence(atlas, fit_images_cpu)
    current, _ = _usage_log_enrichment(tiers, pinlu, seen)
    usage = torch.cat((F.one_hot(tiers, 4), F.one_hot(pinlu, 4)), dim=1).float()
    morphology_density = fit_density(torch.cat((morphology(atlas.images), usage), dim=1), seen)
    unseen = torch.log1p(-0.94 * seen)
    order = torch.as_tensor(np.random.default_rng(seed + 5000).permutation(len(atlas.images)))
    budgets, budget_report = {}, {}
    for name, values in (("current", current), ("morphology", morphology_density)):
        for budget in args.budgets:
            key = f"{name}_hard_{budget}"
            included, budget_report[key] = candidate_budget(0.55 * values + unseen, budget, order)
            budgets[key] = included.to(device)

    public = FontAtlas(atlas.images.to(device), atlas.contours.to(device))
    del atlas, usage
    fit_images = fit_images_cpu.to(device)
    glyphs = np.random.default_rng(seed + 1).choice(held, args.ocf_images, replace=False)
    fitted_rows = torch.as_tensor(fit, device=device)
    query_rows = torch.as_tensor(glyphs, device=device)
    training = {name: counts[fitted_rows] for name, counts in grouped.items()}
    aligned, agreement = align_training_glyphs(public, fit_images)
    style, style_fit = fit_aligned_font_style(aligned, fit_images, seed=seed)
    learned = fit_contour_retrieval(training, *contour_signatures(aligned), task="task2", seed=seed)
    signature, energy = contour_signatures(public.images)
    contour = replace(learned, signature=signature, energy=energy)
    del learned
    encoder = fit_glyph_encoder(training, aligned, epochs=18, seed=seed)
    decoder, stats = fit_direct(training, fit_images, seed)
    del training
    rendered = torch.cat([(style.apply(part[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                          for part in public.images.split(1024)])
    scorer = PairedOCF(score, rendered, (images[glyphs] >= 128).numpy(), device, args.ocf_batch_size)
    current, morphology_density, unseen = current.to(device), morphology_density.to(device), unseen.to(device)
    measurements = {}
    record.update(fit_glyphs=len(fit), held_glyphs=len(held), scored_glyphs=len(glyphs),
                  atlas_alignment=alignment, training_match=match, training_contour_agreement=agreement,
                  style_fit=style_fit, budgets=budget_report, sessions={})
    for session in args.sessions:
        neural = score_spikes(public, encoder, contour, decoder, stats, grouped[session][query_rows], session)
        v8 = neural + 0.55 * current + unseen
        new_prior = neural + 0.55 * morphology_density + unseen
        chosen = {"v8": v8.argmax(dim=1), "morphology_continuous": new_prior.argmax(dim=1)}
        # Every hard shortlist uses the same full-atlas v8 scores, with no
        # score calibration changes induced by the smaller search space.
        chosen.update({key: v8.masked_fill(~include, -torch.inf).argmax(dim=1)
                       for key, include in budgets.items()})
        values = {name: scorer.evaluate(index[:, None])[:, 0] for name, index in chosen.items()}
        for name, scores in values.items():
            measurements.setdefault(name, []).append(scores)
        record["sessions"][session] = paired_summary({name: [scores] for name, scores in values.items()}, args.budgets)
        record["summary"] = paired_summary(measurements, args.budgets)
        record["unique_exact_pairs"] = len(scorer.values)
        record["seconds"] = time.monotonic() - stage
        save()
        print(json.dumps({"seed": seed, "session": session, "metrics": record["sessions"][session]}), flush=True)
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=(613, 1283))
    parser.add_argument("--held-glyphs", type=int, default=320)
    parser.add_argument("--ocf-images", type=int, default=96, help="Paired exact OCF glyphs per session and fold")
    parser.add_argument("--budgets", nargs="+", type=int, default=(1000, 3000))
    parser.add_argument("--sessions", nargs="+", choices=("session1", "session2", "session3", "session4"),
                        default=("session1", "session2", "session3", "session4"))
    parser.add_argument("--ocf-batch-size", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/lexical_ocf_validation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run in a Kaggle GPU Notebook with public assets available.")
    if not 2 <= args.ocf_images <= args.held_glyphs or min(*args.budgets, args.ocf_batch_size) < 1:
        parser.error("Choose valid OCF sample counts, batch size and shortlist budgets.")

    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, torch.device("cpu"))
    if len(args.seeds) * args.held_glyphs > len(images) or len(images) - args.held_glyphs < 128:
        parser.error("The disjoint holdouts must leave enough fit glyphs for atlas alignment.")
    raw_atlas, provenance, codepoints = render_font(*FONT_SOURCES[0], return_codepoints=True)
    tiers, pinlu, unicode_source = unihan_usage(codepoints)
    grouped = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    score = official_scorer(root)
    output = {"mouse": "mice1", "sessions": args.sessions, "budgets": args.budgets,
              "font": provenance, "unihan": unicode_source, "split_protocol": "Disjoint held glyphs; all fits outer-train only.",
              "v8_weights": [0.60, 0.40], "lexical_weight": 0.55, "results": []}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    previous = np.zeros(len(images), dtype=bool)
    for seed in args.seeds:
        order = np.random.default_rng(seed).permutation(len(images))
        held = np.sort(order[~previous[order]][:args.held_glyphs])
        previous[held] = True
        fit = np.setdiff1d(np.arange(len(images)), held)
        record = {"seed": seed}
        output["results"].append(record)
        print(f"seed {seed}: training mice1 with {len(fit)} outer fit glyphs", flush=True)
        run_split(record, raw_atlas, images, grouped, fit, held, tiers, pinlu, device, score, args, save)
    save()
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
