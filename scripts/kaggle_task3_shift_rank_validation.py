"""Test frozen retinal-shift likelihood corrections in an unchanged Task 3 shortlist."""

import argparse
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder, make_count_templates
from glyph_gain_likelihood import fit_gain_model
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_posterior_utility_validation import summarize
from kaggle_retinal_shift_validation import (CENTER, fit_spatial_prior, gamma_adjustment, shifted_rates,
                                            shuffled_shift_rates)
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf


DEPTH = 16
WEIGHTS = (0.5, 1.0)


def expected_at_shifts(empirical, forward):
    """Anchor every shifted response on the same empirical shrinkage template."""
    log_rates = forward.double().log()
    relative = log_rates - log_rates[:, CENTER:CENTER + 1]
    return empirical[:, None].double() * relative.exp()


@torch.no_grad()
def shift_scores(observed, expected, gain, *, batch_size=24):
    """Evaluate one shared translation and analytical Gamma gain for each candidate."""
    outputs = []
    for counts, rates in zip(observed.split(batch_size), expected.split(batch_size)):
        counts = counts.double().unflatten(-1, (-1, 10))
        rates = rates.unflatten(-1, (-1, 10))
        score = counts.new_zeros(rates.shape[:3])
        for (start, stop), alpha in zip(gain.windows, gain.alpha.tolist()):
            y = counts[..., start:stop].flatten(-2)
            mu = rates[..., start:stop].flatten(-2).clamp_min(1e-5)
            count = y.sum(dim=-1)[:, None, None]
            exposure = mu.sum(dim=-1)
            score += torch.einsum("qf,qksf->qks", y, mu.log())
            score += gamma_adjustment(count, exposure, alpha)
            score -= torch.lgamma(y + 1).sum(dim=-1)[:, None, None]
        outputs.append(score)
    return torch.cat(outputs)


def smooth_bank(templates, counts, session):
    baseline = templates.for_recording(session, counts)["shrinkage"]
    return (0.98 * baseline + 0.02 * baseline.mean(dim=0)).clamp_min(1e-5)


def calibrate(inner_fit, inner_query, images, cal_ids, shifts, args):
    encoder = fit_glyph_encoder(inner_fit, images, epochs=args.encoder_epochs, seed=args.seed)
    fit_subset = {session: counts[cal_ids] for session, counts in inner_fit.items()}
    small_images = images[cal_ids]
    templates = make_count_templates(encoder, small_images, fit_subset)
    likelihoods = []
    for session, counts in fit_subset.items():
        expected = smooth_bank(templates, counts, session)
        forward = shifted_rates(encoder, small_images, session, shifts, args.shift_pixels)
        shifted = expected_at_shifts(expected, forward)
        gain = fit_gain_model({session: inner_fit[session]}, "timed", temporal=True)
        likelihoods.append(shift_scores(inner_query[session][cal_ids, 0], shifted[:, None], gain).squeeze(1))
    return fit_spatial_prior(likelihoods, shifts)


@torch.no_grad()
def shortlist(encoder, contour, images, query, session):
    rates = encoder.expected_counts(images, session)
    poisson = count_template_logits(query, rates, "timed")
    direction = contour_retrieval_scores(contour, query, session)["contour_ridge"]
    v11 = blend_contour_poisson(direction, poisson)
    indices = v11.argsort(dim=1, descending=True, stable=True)[:, :DEPTH]
    scores = v11.gather(1, indices)
    scale = poisson.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    return scores, indices, scale


def candidate_shift_bank(encoder, images, indices, empirical, session, shifts, shuffle, margin):
    used, inverse = torch.unique(indices, return_inverse=True)
    donors = shuffle[used]
    available, positions = torch.unique(torch.cat((used, donors)), return_inverse=True)
    bank = shifted_rates(encoder, images[available], session, shifts, margin)
    target, donor = bank[positions[:len(used)]], bank[positions[len(used):]]
    order = (torch.arange(2 * len(used), device=used.device) + len(used)).remainder(2 * len(used))
    permuted, error = shuffled_shift_rates(torch.cat((target, donor)), order)
    base = empirical[used]
    shared = expected_at_shifts(base, target)
    null = expected_at_shifts(base, permuted[:len(used)])
    return shared, null, inverse, error


def rank_variants(scores, indices, scale, corrections):
    choices = {"v11": indices}
    for name, difference in corrections.items():
        for weight in WEIGHTS:
            calibrated = scores + weight * difference / scale
            order = calibrated.argsort(dim=1, descending=True, stable=True)
            choices[f"{name}_w{weight:g}"] = indices.gather(1, order)
    return choices


def identity_rows(ranked, targets):
    output = {}
    for name, indices in ranked.items():
        hits = (indices == targets[:, None]).float()
        ranks = torch.arange(1, hits.shape[1] + 1, device=hits.device)
        output[name] = torch.stack((hits[:, 0], hits[:, :5].sum(1), hits[:, :DEPTH].sum(1),
                                    (hits / ranks).sum(1)), dim=0).cpu().numpy()
    return output


def identity_summary(values, base):
    values, base = np.asarray(values), np.asarray(base)
    mean = values.mean(axis=(0, 2))
    difference = (values - base).mean(axis=0)
    return {name: {"value": float(mean[index]), "delta_vs_v11": float(difference[index].mean()),
                   "paired_glyph_se": float(difference[index].std(ddof=1) / np.sqrt(values.shape[-1]))}
            for index, name in enumerate(("top1", "top5", "top16_proposal", "mrr_at16"))}


def reports(ocf, identity):
    summary = {name: {"ocf": summarize(ocf[name], ocf["v11"]),
                      "identity": identity_summary(values, identity["v11"])}
               for name, values in identity.items()}
    for weight in WEIGHTS:
        shared, shuffled = f"shared_w{weight:g}", f"shuffled_w{weight:g}"
        control = summarize(ocf[shared], ocf[shuffled])
        summary[shared]["ocf"].update(delta_vs_shuffled=control["delta_vs_v11"],
                                      paired_glyph_se_vs_shuffled=control["paired_glyph_se"])
    return summary


def evaluate(encoder, contour, images, outer_fit, outer_query, held_ids, prior, shifts, scorer, args, output, save):
    device = images.device
    masks = images >= 128
    templates = make_count_templates(encoder, images, outer_fit)
    order = torch.as_tensor(np.random.default_rng(args.glyph_seed + 1).permutation(len(images)), device=device)
    shuffle = torch.empty_like(order)
    shuffle[order] = order.roll(1)
    ocf, identities = {}, {}
    for session, counts in outer_query.items():
        queries = counts[held_ids]
        scores, indices, scale = shortlist(encoder, contour, images, queries, session)
        empirical = smooth_bank(templates, outer_fit[session], session)
        gain = fit_gain_model({session: outer_fit[session]}, "timed", temporal=True)
        shared, null, inverse, norm_error = candidate_shift_bank(encoder, images, indices, empirical, session,
                                                                  shifts, shuffle, args.shift_pixels)
        corrections = {}
        for name, rates in (("shared", shared), ("shuffled", null)):
            blocks = [shift_scores(batch, rates[rows], gain) for batch, rows in
                      zip(queries[:, 0].split(24), inverse.split(24))]
            evaluated = torch.cat(blocks)
            corrections[name] = ((evaluated + prior).logsumexp(-1) - evaluated[..., CENTER]).float()
        ranked = rank_variants(scores, indices, scale, corrections)
        predicted = torch.stack([query[:, 0] for query in ranked.values()])
        exact, pairs = unique_pair_ocf(predicted, held_ids.expand_as(predicted), masks, scorer, device, 384)
        identity = identity_rows(ranked, held_ids)
        measured = dict(zip(ranked, exact.cpu().numpy()))
        for name in ranked:
            ocf.setdefault(name, []).append(measured[name])
            identities.setdefault(name, []).append(identity[name])
        report = reports({name: [value] for name, value in measured.items()},
                         {name: [value] for name, value in identity.items()})
        for name, result in report.items():
            result["changed_fraction"] = float((ranked[name][:, 0] != ranked["v11"][:, 0]).float().mean())
        shifted = {name: {"mean_correction": float(values.mean()), "sd_correction": float(values.std()),
                          "mean_within_shortlist_sd": float(values.std(dim=1).mean())}
                   for name, values in corrections.items()}
        output["sessions"][session] = {"metrics": report, "exact_pairs": pairs, "shift_corrections": shifted,
                                       "max_shuffled_log_rate_norm_error": norm_error}
        output["summary"] = reports(ocf, identities)
        save()
        print(json.dumps({"session": session, **output["sessions"][session]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, default=1201)
    parser.add_argument("--glyph-seed", type=int, default=1801)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--calibration-glyphs", type=int, default=320)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--shift-pixels", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_shift_rank_validation.json"))
    args = parser.parse_args()
    if min(args.calibration_glyphs, args.ocf_images) < 2 or min(args.shift_pixels, args.encoder_epochs) < 1:
        parser.error("Choose positive shift and epoch counts and at least two glyphs per sampled fold.")
    if not Path("/kaggle/input").is_dir() or (args.device == "cuda:0" and not torch.cuda.is_available()):
        raise RuntimeError("Use a Kaggle Notebook with the requested device.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, args.device)
    raw = grouped_blocks(root, load_catalog(root), "mice1", images.device, len(images))
    inner_fit, inner_query, outer_fit, outer_query = nested_repeats(raw, args.repeat_seed)
    del raw
    selection = torch.as_tensor(np.random.default_rng(args.glyph_seed).permutation(len(images)), device=images.device)
    stop = args.calibration_glyphs + args.ocf_images
    if stop > len(selection):
        parser.error("Disjoint calibration and scoring glyphs must fit in the labeled gallery.")
    cal_ids, held_ids = selection[:args.calibration_glyphs], selection[args.calibration_glyphs:stop]
    shifts = tuple(product((-args.shift_pixels, 0, args.shift_pixels), repeat=2))
    output = {"mouse": "mice1", "device": args.device, "fit_seed": args.seed, "repeat_seed": args.repeat_seed,
              "glyph_seed": args.glyph_seed, "gallery_glyphs": len(images), "calibration_glyphs": len(cal_ids),
              "ocf_glyphs_per_session": len(held_ids), "proposal_depth": DEPTH, "weights": WEIGHTS,
              "shifts_xy_pixels": shifts, "sessions": {},
              "protocol": "Hold one trial per glyph and session before fitting. Calibrate the spatial prior from an "
                          "inner third trial on disjoint scoring glyphs. Shift log-rate ratios are applied to "
                          "measurement-plus-forward shrinkage means fitted on the remaining repeats. Add the "
                          "empirical-centered shift log likelihood difference to the frozen v11 top-16, using the "
                          "full-gallery Poisson scale. The shuffled control keeps each candidate's zero rate and "
                          "forward shift log-ratio norm. Uses only competition training data."}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("Fit the two-repeat inner encoder and calibrate shifts before scoring any outer query", flush=True)
    prior, summary = calibrate(inner_fit, inner_query, images, cal_ids, shifts, args)
    output["prior"] = {**summary, "shift_probabilities": prior.exp().tolist()}
    save()
    del inner_fit, inner_query
    print(json.dumps({"prior": output["prior"]}), flush=True)
    encoder = fit_glyph_encoder(outer_fit, images, epochs=args.encoder_epochs, seed=args.seed)
    contour = fit_contour_retrieval(outer_fit, *contour_signatures(images), task="task3", seed=args.seed)
    evaluate(encoder, contour, images, outer_fit, outer_query, held_ids, prior, shifts, official_scorer(root),
             args, output, save)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
