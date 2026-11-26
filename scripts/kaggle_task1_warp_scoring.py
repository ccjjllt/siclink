"""Rerank fixed Task 1 candidates using cross-validated equal-ink warp evidence."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import fit_glyph_encoder
from glyph_equal_ink_warp import conditional_likelihood, equal_ink_render, score_raster
from glyph_local_warp import EPOCHS, KNOTS, MAX_CENTROID_SHIFT, MAX_SHIFT_PIXELS, invert_local, warp_statistics
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_task1_half_proposal_validation import ANCHOR_OCF, MODEL_SEED, twos
from kaggle_task1_inverse_warp import INVERSE_ENCODER_EPOCHS, shuffled_responses
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank, shortlist, summarize


PRIOR_OUTER_SEEDS = (961, 2209, 3931, 4931, 5147, 8081, 12457)
EDGE_BUDGET = 8
WEIGHT = 1.5
CAP = 2.5


def crossfit(encoder, templates, counts, session, seed):
    """Fit each candidate to one half and measure improvement on the other."""
    rows, depth = templates.shape[:2]
    images = templates.flatten(0, 1)
    halves = twos(counts, seed)
    gains = {arm: {measure: [] for measure in ("conditional", "poisson")} for arm in ("real", "shuffled")}
    geometry = []
    for iteration, (fitting, held) in enumerate(zip(halves, halves[::-1])):
        independent = held.repeat_interleave(depth, dim=0)
        baseline = score_raster(encoder, images, independent, session)
        shuffled = shuffled_responses(templates[:, 0], fitting, seed + 1 + iteration)
        fold_geometry = {}
        for name, trials in (("real", fitting), ("shuffled", shuffled)):
            result = invert_local(encoder, images, trials.repeat_interleave(depth, dim=0), session,
                                  score=conditional_likelihood, render=equal_ink_render)
            warped = score_raster(encoder, result.images, independent, session)
            for measure, score in warped.items():
                gains[name][measure].append((score - baseline[measure]).reshape(rows, depth))
            report = warp_statistics(images, result)
            if report["max_absolute_ink_fraction_change"] != 0:
                raise RuntimeError("A rendered deformation did not preserve the original ink exactly.")
            fold_geometry[name] = report
        geometry.append(fold_geometry)
    return {name: {measure: torch.stack(values).mean(0) for measure, values in scores.items()}
            for name, scores in gains.items()}, geometry


@torch.no_grad()
def fixed_candidates(model, ranker, likelihood, rendered, counts, session):
    candidates, _ = shortlist(model, likelihood, counts, session)
    scores = ranker.rank(candidates, anchor_ocf=ANCHOR_OCF)
    positions = scores.argsort(dim=1, descending=True, stable=True)[:, :EDGE_BUDGET]
    if not bool(candidates.valid.gather(1, positions).all()):
        raise ValueError("Every fixed v11 row must include eight valid contenders.")
    overlays = rendered[candidates.indices.gather(1, positions)] >= 128
    return candidates, scores, positions, overlays


@torch.no_grad()
def choose(candidates, scores, positions, gains):
    original = candidates.indices.gather(1, scores.argmax(1, keepdim=True))[:, 0]
    chosen, diagnostics = {"v11": original}, {}
    for name, metrics in gains.items():
        correction = WEIGHT * metrics["conditional"]
        amended = scores.scatter_add(1, positions, correction.clamp(-CAP, CAP))
        image = candidates.indices.gather(1, amended.argmax(1, keepdim=True))[:, 0]
        chosen[name] = image
        diagnostics[name] = {"changed_winners": int((image != original).sum()),
                             "fraction_clipped_scored_edges": float((correction.abs() > CAP).float().mean()),
                             "mean_abs_adjustment_ocf": float(correction.clamp(-CAP, CAP).abs().mean()),
                             "mean_max_minus_min_adjustment_ocf": float((correction.clamp(-CAP, CAP).amax(1)
                                                                        - correction.clamp(-CAP, CAP).amin(1)).mean())}
    return chosen, diagnostics


def likelihood_report(values):
    output = {}
    for measure in ("conditional", "poisson"):
        a, b = (np.stack([part[name][measure] for part in values]) for name in ("real", "shuffled"))
        entries = {}
        for name, margins in (("real_vs_zero", a), ("shuffled_vs_zero", b), ("real_vs_shuffled", a - b)):
            summaries = {}
            for subset, scores in (("all_scored_edges", margins.mean(axis=2)), ("v11_winner_edge", margins[..., 0])):
                by_glyph = scores.mean(axis=0)
                summaries[subset] = {"nats_per_trial": float(by_glyph.mean()),
                                     "paired_glyph_se": float(by_glyph.std(ddof=1) / len(by_glyph)**0.5)}
            entries[name] = summaries
        output[measure] = entries
    return output


def metrics(observed):
    result = {name: summarize(values, observed["v11"]) for name, values in observed.items()}
    result["real_vs_shuffled"] = summarize(observed["real"], observed["shuffled"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=16633)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_warp_scoring.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_images <= 320:
        parser.error("Sample between two and 320 held characters per recording.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on a private Kaggle T4 worker with the official training input and public font enabled.")
    started = time.monotonic()
    device = torch.device("cuda:0")
    torch.set_num_threads(4)

    root = locate_data()
    images = load_images(root, device)
    fit, held = outer_split(len(images), args.seed, PRIOR_OUTER_SEEDS, device)
    rng = np.random.default_rng(args.seed + 1)
    selection = held[torch.as_tensor(rng.choice(len(held), args.ocf_images, replace=False), device=device)]
    recording = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if tuple(recording) != SESSIONS or any(values.shape[1] != 4 for values in recording.values()):
        raise ValueError("Require four official training recordings with four measured repeats per character.")
    training = {name: values[fit] for name, values in recording.items()}
    scorer = official_scorer(root)
    split = split_calibration(len(fit), 96, 384, device, MODEL_SEED)
    ranker, likelihoods, nested, codepoints = nested_ranker(training, images[fit], split, scorer, MODEL_SEED)
    model = fit_v8(training, images[fit], MODEL_SEED)
    if not torch.equal(model.codepoints.cpu(), codepoints):
        raise ValueError("Nested and outer public atlas columns must match exactly.")
    encoder = fit_glyph_encoder(training, images[fit], epochs=INVERSE_ENCODER_EPOCHS, seed=MODEL_SEED + 499)
    encoder.eval().requires_grad_(False)
    del training
    rendered = rendered_bank(model)
    exact = PairedOCF(scorer, rendered, (images[selection] >= 128).cpu().numpy(), device, 12)
    output = {"outer_seed": args.seed, "fit_glyphs": len(fit), "held_glyphs": len(held),
              "earlier_outer_seeds": PRIOR_OUTER_SEEDS, "paired_per_recording": len(selection), "repeats_per_glyph": 4,
              "v11_anchor_ocf": ANCHOR_OCF, "v11_nominations": NOMINATIONS, "warp_scored_edges_per_query": EDGE_BUDGET,
              "ocf_points_per_nats_per_trial": WEIGHT, "ocf_offset_cap": CAP, "model_seed": MODEL_SEED,
              "encoder_seed": MODEL_SEED + 499, "encoder_epochs": INVERSE_ENCODER_EPOCHS, "ranker": ranker.diagnostics,
              "nested": nested, "boundary_fixed_lattice": KNOTS, "adam_steps": EPOCHS,
              "max_displacement_px": MAX_SHIFT_PIXELS, "max_centroid_shift_px": MAX_CENTROID_SHIFT,
              "protocol": "Use the existing v11 ranker and full four-repeat candidate edges. The same eight best v11 "
                          "edges are eligible for every scoring arm. A fixed two/two repeat split fits each warp on "
                          "one half, measures its gain over zero warp on the other, and repeats with the halves swapped. "
                          "Candidate nomination has used the full four repeats, while its fitted warps have not seen "
                          "their scoring repeats. Both arms compare 50–250 ms unit allocation conditional on the "
                          "population total in each 25 ms bin and exactly match each raster's foreground count to "
                          "the original. The shuffled arm learns from other fitting queries balanced by activity "
                          "and initial template ink and scores against the same independent counts. All OCF outputs "
                          "are the unwarped original v11 binary renderings; only their rankings can differ. "
                          "No official test data, held images or sample IDs enter ranking.", "sessions": {}}

    all_observations, all_gains = {}, []
    for index, name in enumerate(SESSIONS):
        session_started = time.monotonic()
        counts = recording[name][selection]
        candidates, objective, positions, templates = fixed_candidates(model, ranker, likelihoods[name], rendered, counts, name)
        print(f"{name}: scoring {templates.shape[0] * templates.shape[1]} fixed candidate edges", flush=True)
        gains, geometry = crossfit(encoder, templates, counts, name, args.seed + 1000 + index * 10)
        choices, diagnostics = choose(candidates, objective, positions, gains)
        observed = {arm: exact.evaluate(indices[:, None])[:, 0] for arm, indices in choices.items()}
        for arm, results in observed.items():
            all_observations.setdefault(arm, []).append(results)
        snapshot = {arm: {measure: values.cpu().numpy() for measure, values in margins.items()}
                    for arm, margins in gains.items()}
        all_gains.append(snapshot)
        report = {"ocf": metrics({arm: [values] for arm, values in observed.items()}),
                  "likelihood": likelihood_report([snapshot]), "rank_changes": diagnostics, "warp_geometry": geometry,
                  "all_v11_edges": int(candidates.valid.sum()), "scored_edges": int(positions.numel()),
                  "seconds": time.monotonic() - session_started}
        output["sessions"][name] = report
        output["summary"] = {"ocf": metrics(all_observations), "likelihood": likelihood_report(all_gains)}
        output["unique_exact_image_pairs"] = len(exact.values)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": name, "ocf": report["ocf"], "rank_changes": diagnostics}), flush=True)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
