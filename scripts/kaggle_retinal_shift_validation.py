"""Test whether a shared latent image translation predicts independent V1 units."""

import argparse
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model


PRIOR_WIDTHS = (0.0, 0.75, 1.5, 2.25, 3.0, 4.5)
CENTER = 4


def translate(images, dx, dy, margin):
    """Move the rendered stimulus by integer pixels with black fill."""
    padded = F.pad(images, (margin, margin, margin, margin))
    y, x = margin - dy, margin - dx
    return padded[:, y:y + images.shape[-2], x:x + images.shape[-1]]


@torch.no_grad()
def shifted_rates(encoder, images, session, shifts, margin):
    fit_baseline = encoder.training_rates[encoder.session_index[session]]
    rates = torch.stack([encoder.expected_counts(translate(images, dx, dy, margin), session, batch_size=192)
                         for dx, dy in shifts], dim=1)
    return (0.98 * rates + 0.02 * fit_baseline).clamp_min(1e-5)


def shuffled_shift_rates(rates, permutation):
    """Preserve each image's center rate and per-shift log-rate change norm."""
    baseline = rates[:, CENTER:CENTER + 1].double().log()
    delta = rates.double().log() - baseline
    norms = delta.norm(dim=-1, keepdim=True)
    donor = delta[permutation]
    donor_norm = donor.norm(dim=-1, keepdim=True)
    default = torch.full_like(donor, donor.shape[-1] ** -0.5)
    direction = torch.where(donor_norm > 0, donor / donor_norm.clamp_min(1e-12), default)
    changed = direction * norms
    recovered = (baseline + changed).exp()
    error = (changed.norm(dim=-1, keepdim=True) - norms).abs().amax()
    return recovered, float(error)


def unit_halves(training, seed):
    """Match unit partitions by fitted evoked activity without using held glyphs."""
    activity = torch.stack([counts.unflatten(-1, (-1, 10))[..., 2:].sum(dim=-1).mean(dim=(0, 1))
                            for counts in training.values()]).mean(0)
    order = activity.argsort().cpu().numpy()
    rng = np.random.default_rng(seed)
    paired = order[:len(order) // 2 * 2].reshape(-1, 2)
    orientation = rng.integers(0, 2, size=len(paired))
    rows = np.arange(len(paired))
    left, right = paired[rows, orientation], paired[rows, 1 - orientation]
    if len(order) % 2:
        left = np.append(left, order[-1])
    return tuple(torch.as_tensor(selection, device=activity.device) for selection in (left, right))


def side_statistics(counts, rates, units, start, stop):
    observed = counts[:, :, units, start:stop].flatten(-2).double()
    expected = rates[:, :, units, start:stop].flatten(-2).double()
    evidence = torch.einsum("grf,gsf->grs", observed, expected.log())
    evidence -= torch.lgamma(observed + 1).sum(dim=-1, keepdim=True)
    return evidence, observed.sum(dim=-1), expected.sum(dim=-1)


def gamma_adjustment(totals, exposure, alpha):
    """Integrate one mean-one Gamma gain and retain the normalization constant."""
    if alpha <= 1e-8:
        return -exposure
    shape = totals.new_tensor(1 / alpha)
    gamma_normalizer = torch.lgamma(totals + shape) - torch.lgamma(shape) - totals * shape.log()
    return gamma_normalizer - (totals + shape) * (alpha * exposure).log1p()


def likelihoods(counts, rates, halves, gain):
    """Return A, B, and A/B log probabilities for all paired shift states."""
    counts = counts.unflatten(-1, (-1, 10))
    rates = rates.unflatten(-1, (-1, 10))
    shape = (*counts.shape[:2], rates.shape[1])
    left = counts.new_zeros(shape, dtype=torch.float64)
    right = torch.zeros_like(left)
    joint = counts.new_zeros((*shape, rates.shape[1]), dtype=torch.float64)
    for (start, stop), alpha in zip(gain.windows, gain.alpha.tolist()):
        a, count_a, mass_a = side_statistics(counts, rates, halves[0], start, stop)
        b, count_b, mass_b = side_statistics(counts, rates, halves[1], start, stop)
        left += a + gamma_adjustment(count_a[..., None], mass_a[:, None, :], alpha)
        right += b + gamma_adjustment(count_b[..., None], mass_b[:, None, :], alpha)
        joint += a[..., :, None] + b[..., None, :] + gamma_adjustment(
            (count_a + count_b)[..., None, None], mass_a[:, None, :, None] + mass_b[:, None, None, :], alpha)
    return left, right, joint


def spatial_prior(shifts, width, like):
    square_radius = like.new_tensor([dx ** 2 + dy ** 2 for dx, dy in shifts])
    if width == 0:
        return torch.where(square_radius == 0, 0, -torch.inf)
    return (-square_radius / (2 * width ** 2)).log_softmax(dim=0)


def fit_spatial_prior(calibration, shifts):
    combined = torch.cat(calibration, dim=0)
    scores = {width: float((combined + spatial_prior(shifts, width, combined)).logsumexp(dim=-1).mean())
              for width in PRIOR_WIDTHS}
    winner = max(scores, key=scores.get)
    width = max((candidate for candidate in scores if candidate > 0), key=scores.get)
    return spatial_prior(shifts, width, combined), {"width_pixels": width, "best_width_including_zero": winner,
                                                    "calibration_joint_log_likelihood": scores}


def conditional_likelihood(joint, left, right):
    return torch.stack((joint - left, joint - right), dim=-1)


def predictive_parts(left, right, joint, prior):
    """Contrast shared or independent translations with the same shared gain."""
    left_mix = (left + prior).logsumexp(dim=-1)
    right_mix = (right + prior).logsumexp(dim=-1)
    shared = (joint.diagonal(dim1=-2, dim2=-1) + prior).logsumexp(dim=-1)
    independent = (joint + prior[:, None] + prior).flatten(-2).logsumexp(dim=-1)
    zero = conditional_likelihood(joint[..., CENTER, CENTER], left[..., CENTER], right[..., CENTER])
    return {"zero_shift": zero,
            "shared_shift": conditional_likelihood(shared, left_mix, right_mix),
            "independent_shift": conditional_likelihood(independent, left_mix, right_mix)}


def paired_summary(values, baseline):
    # Average repeats and both unit-half directions before taking a glyph-cluster SE.
    differences = (values - baseline).mean(dim=(0, 2, 3))
    return {"mean_conditional_log_likelihood": float(values.mean()),
            "delta_nats_per_target_half": float(differences.mean()),
            "paired_glyph_se": float(differences.std(unbiased=True) / len(differences) ** 0.5),
            "glyph_clusters": len(differences), "conditional_directions": values.numel(),
            "directional_delta": (values - baseline).mean(dim=(0, 1, 2)).tolist()}


def compare(blocks):
    stacked = {name: torch.stack(values) for name, values in blocks.items()}
    summary = {name: paired_summary(values, stacked["zero_shift"]) for name, values in stacked.items()}
    for control in ("independent_shift", "shuffled_shift"):
        summary[f"shared_vs_{control}"] = paired_summary(stacked["shared_shift"], stacked[control])
    return summary


def run_probe(encoder, images, training, calibration, held, args, output, save):
    shifts = tuple(product((-args.shift_pixels, 0, args.shift_pixels), repeat=2))
    halves = unit_halves(training, args.seed + 1)
    gains = {session: fit_gain_model({session: counts}, "timed", temporal=True)
             for session, counts in training.items()}
    output["unit_halves"] = [len(units) for units in halves]
    output["shifts_xy_pixels"] = shifts
    output["population_gain_variance"] = {session: gain.alpha.tolist() for session, gain in gains.items()}
    calibration_likelihood = []
    for session, counts in calibration.items():
        means = shifted_rates(encoder, images[0], session, shifts, args.shift_pixels)
        _, _, joint = likelihoods(counts, means, halves, gains[session])
        calibration_likelihood.append(joint.diagonal(dim1=-2, dim2=-1).contiguous())
    prior, prior_metrics = fit_spatial_prior(calibration_likelihood, shifts)
    output["prior"] = {**prior_metrics, "shift_probabilities": prior.exp().tolist()}
    print(json.dumps({"prior": output["prior"]}), flush=True)
    save()

    order = np.random.default_rng(args.seed + 2).permutation(len(images[1]))
    shuffle = torch.empty(len(order), dtype=torch.long, device=images[1].device)
    shuffle[torch.as_tensor(order, device=shuffle.device)] = torch.as_tensor(np.roll(order, 1), device=shuffle.device)
    records = {}
    for session, counts in held.items():
        means = shifted_rates(encoder, images[1], session, shifts, args.shift_pixels)
        values = predictive_parts(*likelihoods(counts, means, halves, gains[session]), prior)
        shuffled, error = shuffled_shift_rates(means, shuffle)
        shuffled_values = predictive_parts(*likelihoods(counts, shuffled, halves, gains[session]), prior)
        values["shuffled_shift"] = shuffled_values["shared_shift"]
        output["sessions"][session] = {"metrics": compare({name: [item] for name, item in values.items()}),
                                       "max_shuffled_log_rate_norm_error": error}
        for name, item in values.items():
            records.setdefault(name, []).append(item)
        output["summary"] = compare(records)
        save()
        print(json.dumps({"session": session, **output["sessions"][session]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1171)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--calibration-glyphs", type=int, default=320)
    parser.add_argument("--held-glyphs", type=int, default=320)
    parser.add_argument("--shift-pixels", type=int, default=3)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/retinal_shift_validation.json"))
    args = parser.parse_args()
    if min(args.calibration_glyphs, args.held_glyphs) < 2 or min(args.shift_pixels, args.encoder_epochs) < 1:
        parser.error("Choose at least two glyphs per split and positive shift and epoch settings.")
    if not Path("/kaggle/input").is_dir() or (args.device == "cuda:0" and not torch.cuda.is_available()):
        raise RuntimeError("Use Kaggle GPU or pass --device cpu for a Kaggle CPU notebook.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, args.device)
    raw = grouped_blocks(root, load_catalog(root), "mice1", images.device, len(images))
    n_fit = len(images) - args.calibration_glyphs - args.held_glyphs
    if n_fit < 128 or any(counts.shape[1] != 4 for counts in raw.values()):
        raise ValueError("Require four labeled repeats and at least 128 disjoint encoder-fit glyphs.")
    order = torch.as_tensor(np.random.default_rng(args.seed).permutation(len(images)), device=images.device)
    fit_ids, cal_ids, held_ids = order.split((n_fit, args.calibration_glyphs, args.held_glyphs))
    training = {session: counts[fit_ids] for session, counts in raw.items()}
    calibration = {session: counts[cal_ids] for session, counts in raw.items()}
    held = {session: counts[held_ids] for session, counts in raw.items()}
    del raw
    output = {"mouse": "mice1", "seed": args.seed, "device": args.device, "encoder_epochs": args.encoder_epochs,
              "fit_glyphs": n_fit, "calibration_glyphs": len(cal_ids), "held_glyphs": len(held_ids), "repeats": 4,
              "direction_order": ["right_given_left", "left_given_right"],
              "protocol": "Disjoint fit, prior-calibration and outer glyphs. Only official labeled training "
                          "observations enter this mechanistic comparison. The known held image is used only to "
                          "condition the held likelihood. All models integrate the same population gain. "
                          "A positive-width prior is frozen on calibration to compare with fixed zero shift. "
                          "Held observations do not fit the prior or encoder. This measures spike likelihood.",
              "sessions": {}}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"Fit encoder on {n_fit} mice1 training glyphs", flush=True)
    encoder = fit_glyph_encoder(training, images[fit_ids], epochs=args.encoder_epochs, seed=args.seed)
    run_probe(encoder, (images[cal_ids], images[held_ids]), training, calibration, held, args, output, save)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
