"""Test repeat-wise retinal-shift evidence in the actual mice2 v11 atlas decisions."""

import argparse
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import (SHORTLIST as LEGACY_NOMINATIONS, fit_calibrated_ranker,
                                           shortlist as legacy_candidates, split_calibration)
from kaggle_gallery_validation import align_training_glyphs
from kaggle_retinal_shift_validation import (CENTER, fit_spatial_prior, gamma_adjustment, shifted_rates,
                                            shuffled_shift_rates)
from kaggle_v8_metric_validation import (NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank,
                                         shortlist as contour_candidates, summarize)


SHIFT_PIXELS = 3
SHIFTS = tuple(product((-SHIFT_PIXELS, 0, SHIFT_PIXELS), repeat=2))
SETTINGS = ((0.25, 2.0), (0.50, 4.0))
ROOT_HELD_SEED = 2651
ROOT_HELD_SIZE = 320


def outer_indices(n_glyphs, seed, n_score, device):
    previous = np.random.default_rng(ROOT_HELD_SEED).permutation(n_glyphs)[-ROOT_HELD_SIZE:]
    avoid = np.zeros(n_glyphs, dtype=bool)
    avoid[previous] = True
    order = np.random.default_rng(seed).permutation(n_glyphs)
    held = np.sort(order[~avoid[order]][:ROOT_HELD_SIZE])
    fitted = np.setdiff1d(np.arange(n_glyphs), held)
    sample = np.random.default_rng(seed + 1).choice(held, n_score, replace=False)
    return tuple(torch.as_tensor(rows, device=device) for rows in (fitted, held, sample))


@torch.no_grad()
def repeat_log_likelihood(counts, expected, gain):
    """Score per-repeat, per-candidate shift likelihoods under the same Gamma gain."""
    counts = counts.unflatten(-1, (-1, 10)).double()
    expected = expected.unflatten(-1, (-1, 10)).double()
    scores = counts.new_zeros((*counts.shape[:2], *expected.shape[1:3]))
    for (start, stop), alpha in zip(gain.windows, gain.alpha.tolist()):
        y = counts[..., start:stop].flatten(-2)
        rate = expected[..., start:stop].flatten(-2).clamp_min(1e-5)
        totals = y.sum(dim=-1)[..., None, None]
        exposure = rate.sum(dim=-1)[:, None]
        scores += torch.einsum("qrf,qksf->qrks", y, rate.log())
        scores += gamma_adjustment(totals, exposure, alpha)
        scores -= torch.lgamma(y + 1).sum(-1)[..., None, None]
    return scores


def calibrate_displacement(training, aligned, split, seed):
    _, noise, inner = split
    fitted = {session: counts[inner] for session, counts in training.items()}
    encoder = fit_glyph_encoder(fitted, aligned[inner], epochs=18, seed=seed)
    observations = []
    for session, counts in fitted.items():
        gain = fit_gain_model({session: counts}, "timed", temporal=True)
        rates = shifted_rates(encoder, aligned[noise], session, SHIFTS, SHIFT_PIXELS)
        matched = [repeat_log_likelihood(query, expected[:, None], gain).squeeze(2)
                   for query, expected in zip(training[session][noise].split(32), rates.split(32))]
        observations.append(torch.cat(matched))
    prior, calibration = fit_spatial_prior(observations, SHIFTS)
    calibration.update(noise_glyphs=len(noise), forward_fit_glyphs=len(inner), probabilities=prior.exp().tolist())
    return prior, calibration


def glyph_donors(size, seed, device):
    random = torch.as_tensor(np.random.default_rng(seed).permutation(size), device=device)
    donor = torch.empty_like(random)
    donor[random] = random.roll(1)
    return donor


def shift_corrections(model, counts, candidates, session, gain, prior, donor):
    used, inverse = torch.unique(candidates.indices, return_inverse=True)
    available, positions = torch.unique(torch.cat((used, donor[used])), return_inverse=True)
    rates = shifted_rates(model.encoder, model.atlas.images[available], session, SHIFTS, SHIFT_PIXELS)
    original, unrelated = rates[positions[:len(used)]], rates[positions[len(used):]]
    swap = (torch.arange(2 * len(used), device=used.device) + len(used)).remainder(2 * len(used))
    shuffled, norm_error = shuffled_shift_rates(torch.cat((original, unrelated)), swap)
    shuffled = shuffled[:len(used)]
    average = (original * prior.exp()[None, :, None]).sum(dim=1, keepdim=True)

    blocks = {name: [] for name in ("shift", "shuffled", "moment")}
    for query, rows in zip(counts.split(12), inverse.split(12)):
        true_likelihood = repeat_log_likelihood(query, original[rows], gain)
        null_likelihood = repeat_log_likelihood(query, shuffled[rows], gain)
        moment_likelihood = repeat_log_likelihood(query, average[rows], gain)
        zero = true_likelihood[..., CENTER]
        values = {"shift": (true_likelihood + prior).logsumexp(-1),
                  "shuffled": (null_likelihood + prior).logsumexp(-1), "moment": moment_likelihood[..., 0]}
        for name, value in values.items():
            blocks[name].append((value - zero).mean(dim=1).float())

    centered, report = {}, {}
    support = candidates.valid.sum(dim=1, keepdim=True)
    for name, parts in blocks.items():
        values = torch.cat(parts)
        average = values.masked_fill(~candidates.valid, 0).sum(dim=1, keepdim=True) / support
        centered[name] = (values - average).masked_fill(~candidates.valid, 0)
        report[name] = {"uncentered_mean_nats_per_repeat": float(values[candidates.valid].mean()),
                        "centered_sd_nats_per_repeat": float(centered[name][candidates.valid].std())}
    report["max_shuffled_norm_error"] = norm_error
    report["atlas_candidates_evaluated"] = len(used)
    return centered, report


@torch.no_grad()
def choose(ranker, candidates, anchor, corrections):
    logits = ranker.rank(candidates, anchor_ocf=anchor)
    methods = {"v11": logits}
    caps = {}
    for name, delta in corrections.items():
        settings = SETTINGS if name != "moment" else SETTINGS[-1:]
        for weight, cap in settings:
            key = f"{name}_w{weight:g}"
            offset = weight * delta
            methods[key] = logits + offset.clamp(-cap, cap)
            caps[key] = float((offset[candidates.valid].abs() > cap).float().mean())
    return {name: candidates.indices.gather(1, values.argmax(dim=1, keepdim=True))[:, 0]
            for name, values in methods.items()}, caps


def metrics(measurements):
    report = {name: summarize(values, measurements["v11"]) for name, values in measurements.items()}
    for weight, _ in SETTINGS:
        name = f"shift_w{weight:g}"
        match = summarize(measurements[name], measurements[f"shuffled_w{weight:g}"])
        report[name].update(paired_delta_vs_shuffle=match["paired_delta"], paired_se_vs_shuffle=match["paired_se"])
    moment = summarize(measurements["shift_w0.5"], measurements["moment_w0.5"])
    report["shift_w0.5"].update(paired_delta_vs_moment=moment["paired_delta"], paired_se_vs_moment=moment["paired_se"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-seed", type=int, default=3541)
    parser.add_argument("--model-seed", type=int, default=211)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_atlas_shift_validation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the open-atlas experiment on Kaggle GPU with public font access.")
    if not 2 <= args.ocf_images <= ROOT_HELD_SIZE:
        parser.error("Choose a paired sample between two and 320 outer glyphs.")

    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    device = torch.device("cuda:0")
    images = load_images(root, device)
    raw = grouped_blocks(root, load_catalog(root), "mice2", device, len(images))
    fit, held, scored = outer_indices(len(images), args.outer_seed, args.ocf_images, device)
    training = {session: counts[fit] for session, counts in raw.items()}
    queries = {session: counts[scored] for session, counts in raw.items()}
    del raw
    if any(counts.shape[1] != 4 for counts in queries.values()):
        raise ValueError("The unseen-glyph proxy requires four independent labeled training repeats.")
    fit_images = images[fit]
    split = split_calibration(len(fit), 96, 384, device, args.model_seed)
    scorer = official_scorer(root)
    output = {"mouse": "mice2", "outer_seed": args.outer_seed, "model_seed": args.model_seed,
              "excluded_previous_tail_seed": ROOT_HELD_SEED, "excluded_previous_holdout": ROOT_HELD_SIZE,
              "fit_glyphs": len(fit), "held_glyphs": len(held), "scored_glyphs_per_session": len(scored),
              "repeats": 4, "official_task1_repeats": 20, "shifts_xy_pixels": SHIFTS,
              "score_weights_ocf_points_per_nat_per_repeat_and_caps": SETTINGS,
              "nominations": {"session1": NOMINATIONS, "session2_to_4": LEGACY_NOMINATIONS},
              "anchors": {"session1": 4.0, "session2_to_4": 2.0}, "sessions": {},
              "protocol": "Disjoint outer unseen glyphs exclude the previous closed-gallery holdout. Both v11 policy "
                          "rankers and atlases are fitted using only outer-fit glyphs. Fit the shift prior on separate "
                          "noise-calibration glyphs after training another encoder on inner prediction glyphs. Keep "
                          "the v11 proposals and rendering, adding capped centered marginal-minus-zero likelihood "
                          "per repeat. Controls preserve shift magnitude or use the shift-averaged rate. No official "
                          "test responses or IDs enter the experiment."}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("Calibrate exact v11 session1 contour-union policy on four fit-only recordings", flush=True)
    modern, modern_likelihood, nested, codepoints = nested_ranker(training, fit_images, split, scorer, args.model_seed)
    print("Refit final v11 atlas and calibrate exact legacy policy for sessions 2–4", flush=True)
    model = fit_v8(training, fit_images, args.model_seed)
    if not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("Both atlas calibrations need identical public codepoint order.")
    rendered = rendered_bank(model)
    aligned, _ = align_training_glyphs(model.atlas, fit_images)
    legacy, legacy_likelihood, legacy_report = fit_calibrated_ranker(
        model.atlas, model.codepoints, fit_images, aligned, training, split, rendered, scorer, SESSIONS, args.model_seed
    )
    output["v11_fit"] = {"full_atlas": model.report, "nested_contour": nested,
                         "contour_ranker": modern.diagnostics, "legacy_ranker": legacy_report}
    save()
    print("Calibrate and freeze per-repeat translation evidence", flush=True)
    prior, report = calibrate_displacement(training, aligned, split, args.model_seed)
    output["shift_calibration"] = report
    save()
    print(json.dumps({"shift_calibration": report}), flush=True)

    accumulator = {}
    exact = PairedOCF(scorer, rendered, (images[scored] >= 128).cpu().numpy(), device, 12)
    donors = glyph_donors(len(model.atlas.images), args.outer_seed + 2, device)
    for session in SESSIONS:
        observed = queries[session]
        if session == "session1":
            shortlisted, _ = contour_candidates(model, modern_likelihood[session], observed, session)
            ranker, anchor = modern, 4.0
        else:
            shortlisted = legacy_candidates(model.atlas, model.prior, model.encoder, model.decoder, model.normalization,
                                            legacy_likelihood[session], observed, session)
            ranker, anchor = legacy, 2.0
        gain = fit_gain_model({session: training[session]}, "timed", temporal=True)
        correction, diagnostics = shift_corrections(model, observed, shortlisted, session, gain, prior, donors)
        choices, caps = choose(ranker, shortlisted, anchor, correction)
        measured = {name: exact.evaluate(index[:, None])[:, 0] for name, index in choices.items()}
        for name, values in measured.items():
            accumulator.setdefault(name, []).append(values)
        block = metrics({name: [values] for name, values in measured.items()})
        for name, values in block.items():
            values["changed_vs_v11"] = int((choices[name] != choices["v11"]).sum())
        output["sessions"][session] = {"metrics": block, "anchor": anchor, "shift_diagnostics": diagnostics,
                                       "cap_fraction": caps, "mean_candidates": float(shortlisted.valid.float().sum(1).mean())}
        output["summary"] = metrics(accumulator)
        output["unique_exact_pairs"] = len(exact.values)
        save()
        print(json.dumps({"session": session, **output["sessions"][session]}), flush=True)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
