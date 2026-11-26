"""Test the independent value of spikes in a public-glyph residual DDPM."""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from glyph_atlas_prior import fit_atlas_prior
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import FontAtlas, align_training_glyphs, build_font_atlas, image_descriptors, standardized
from kaggle_task1_official_ddpm import (EMA_DECAY, GUIDANCE, condition, counts50, fit_input_scale, held_split,
                                       official_model, paired_stats, shuffled_rows)
from kaggle_task1_repertoire_coverage import shape_matches


MOUSE = "mice1"
MARGIN = 3
MAX_INK_FRACTION = 0.12
DROP_PROBABILITY = 0.20
ANCHOR_MODES = ("spike_selected", "public_match", "public_near_shape")
ARMS = ("original_public", "shape_only", "shuffled_spikes", "true_spikes")


@torch.no_grad()
def freeze_atlas(images, fit, seed):
    training = images[fit]
    device = images.device
    style_rows = torch.as_tensor(np.random.default_rng(seed + 2).choice(len(fit), 128, replace=False), device=device)
    atlas, provenance, codes = build_font_atlas(training[style_rows], device, return_codepoints=True)
    aligned, agreement = align_training_glyphs(atlas, training)
    return atlas, aligned, codes, provenance, agreement


def styled_bank(atlas, aligned, target, seed):
    style, report = fit_aligned_font_style(aligned, target, seed=seed)
    with torch.no_grad():
        renders = torch.cat([style.apply(rows[:, None].float() / 255)[:, 0] >= 0.5
                             for rows in atlas.images.split(1024)])
    return renders, report


@torch.inference_mode()
def visual_anchors(rendered, sampled_images):
    bitmaps = rendered.to(torch.uint8) * 255
    descriptors = torch.cat([image_descriptors(batch).cpu() for batch in bitmaps.split(512)])
    matches = shape_matches(FontAtlas(bitmaps.cpu(), descriptors), sampled_images.cpu())
    high = matches.confident(0.55, 0.01)
    if int(high.sum()) < 256:
        raise ValueError("The fixed matching rule needs at least 256 fitting anchors.")
    return (torch.where(high)[0].to(rendered.device), matches.closest.to(rendered.device),
            matches.candidates[:, 1:8].to(rendered.device),
            {"high_confidence_fit_matches": int(high.sum()),
             "mean_accepted_match_quality": float(matches.scores[high, 0].mean()), "near_shape_ranks": [2, 8]})


def fit_ranker(atlas, codes, full_signature, full_energy, source, images, aligned, seed):
    prior = fit_atlas_prior(atlas, codes, images)
    contour = fit_contour_retrieval(source, *contour_signatures(aligned), task="task2", seed=seed)
    contour = replace(contour, signature=full_signature, energy=full_energy)
    neural = fit_glyph_encoder(source, aligned, epochs=18, seed=seed)
    return contour, neural, prior


@torch.inference_mode()
def rank_anchors(atlas, model, blocks, rows):
    contour, encoder, prior = model
    selected = []
    for session, trials in blocks.items():
        rates = encoder.expected_counts(atlas.images, session, batch_size=192)
        choices = []
        for batch in rows.split(128):
            query = trials[batch]
            direction = standardized(contour_retrieval_scores(contour, query, session)["contour_ridge"])
            rate = standardized(count_template_logits(query, rates, "rate"))
            timed = standardized(count_template_logits(query, rates, "timed"))
            likelihood = standardized(0.5 * (rate + timed))
            scores = prior.variants(0.65 * direction + 0.35 * likelihood)["usage_unseen"]
            choices.append(scores.argmax(dim=1))
        selected.append(torch.cat(choices))
    return torch.stack(selected)


def edge_support(anchor):
    image = anchor.float()
    band = []
    for radius in range(1, MARGIN + 1):
        padded = F.pad(image, (radius,) * 4)
        width = 2 * radius + 1
        outside = F.max_pool2d(padded, width, stride=1)
        inside = -F.max_pool2d(-padded, width, stride=1)
        band.append(outside > inside)
    # Lower distances occur in more rings, giving these pixels priority.
    return band[-1], sum(ring.float() for ring in band)


def bounded_flips(scores, potential, anchor):
    scores = scores.flatten(1).masked_fill(~potential.flatten(1), -torch.inf)
    cap = (MAX_INK_FRACTION * anchor.flatten(1).sum(dim=1)).long()
    available = potential.flatten(1).sum(dim=1)
    taken = torch.minimum(available, cap)
    priority = scores.argsort(dim=1, descending=True)
    mask = torch.arange(scores.shape[-1], device=anchor.device)[None] < taken[:, None]
    selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, priority, mask)
    return selected.reshape_as(anchor), available, taken


def train_batch(trials, bank, images, high, visual, near, inner, spike_anchors, gain, batch_size, generator):
    device = trials.device
    day = torch.randint(len(gain), (batch_size,), generator=generator, device=device)
    mode = torch.randint(4, (batch_size,), generator=generator, device=device)
    choose_inner = mode < 2
    inner_row = torch.randint(len(inner), (batch_size,), generator=generator, device=device)
    visual_row = high[torch.randint(len(high), (batch_size,), generator=generator, device=device)]
    glyph = torch.where(choose_inner, inner[inner_row], visual_row)
    near_rank = torch.randint(near.shape[1], (batch_size,), generator=generator, device=device)
    seen = torch.where(mode == 2, visual[glyph], near[glyph, near_rank])
    selected = torch.where(choose_inner, spike_anchors[day, inner_row], seen)
    anchor = bank[selected, None]
    truth = images[glyph, None]
    edge, proximity = edge_support(anchor)
    priority = proximity + 0.1 * torch.rand(proximity.shape, device=device, generator=generator)
    use, possible, taken = bounded_flips(priority, (truth != anchor) & edge, anchor)
    residual = (truth.float() - anchor.float()) * use
    features = condition(trials[day, glyph], day, gain)
    report = torch.stack((torch.ones_like(taken), possible, taken, (truth != anchor).flatten(1).sum(dim=1))).T.float()
    compact = torch.where(choose_inner, 0, mode - 1)
    return anchor, edge, residual, features, compact, report


def residual_model(official, inputs, device):
    model = official.build_model(inputs)
    old = model.input
    extra = nn.Conv2d(2, old.out_channels, old.kernel_size, padding=old.padding)
    with torch.no_grad():
        extra.weight[:, :1].copy_(old.weight)
        extra.weight[:, 1:].zero_()
        extra.bias.copy_(old.bias)
    model.input = extra
    return model.to(device, memory_format=torch.channels_last)


def fit_residual(official, trials, bank, images, high, visual, near, inner, spike_anchors, gain, args, result):
    device = trials.device
    torch.manual_seed(args.model_seed)
    model = residual_model(official, trials.shape[-1] + len(gain), device).train()
    ema = deepcopy(model).eval().requires_grad_(False)
    scheduler = official.diffusion_scheduler()
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler("cuda", init_scale=2048)
    sampling = torch.Generator(device=device).manual_seed(args.model_seed + 31)
    noise = torch.Generator(device=device).manual_seed(args.model_seed + 47)
    started = time.monotonic()
    successes = attempts = window = 0
    exposure = torch.zeros((3, 4), device=device)
    loss_sum = torch.zeros((), device=device)
    records = []
    while successes < args.updates:
        if attempts >= args.updates + 100:
            raise RuntimeError("Too many FP16 overflows to complete the specified update budget.")
        anchor, edge, residual, features, modes, details = train_batch(
            trials, bank, images, high, visual, near, inner, spike_anchors, gain, args.batch_size, sampling)
        size = len(anchor)
        t = torch.randint(official.DIFFUSION_STEPS, (size,), device=device, generator=noise)
        random_noise = torch.randn(residual.shape, device=device, generator=noise) * edge
        noisy = scheduler.add_noise(residual, random_noise, t)
        target = scheduler.get_velocity(residual, random_noise, t)
        dropped = torch.rand(size, device=device, generator=noise) < DROP_PROBABILITY
        inputs = torch.cat((noisy, 2 * anchor.float() - 1), dim=1).contiguous(memory_format=torch.channels_last)
        alpha = scheduler.alphas_cumprod[t]
        snr = alpha / (1 - alpha)
        snr_weight = snr.clamp(max=5) / (snr + 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(inputs, t, features, dropped)
        pixel_weight = (0.15 + 0.85 * edge.float())
        errors = ((prediction.float() - target).square() * pixel_weight).flatten(1).sum(1)
        loss = (snr_weight * errors / pixel_weight.flatten(1).sum(1)).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        attempts += 1
        if bool(torch.isfinite(torch.stack((norm, loss.detach()))).all()):
            scaler.step(optimizer)
            with torch.no_grad():
                for average, param in zip(ema.parameters(), model.parameters()):
                    average.lerp_(param, 1 - EMA_DECAY)
                exposure.index_add_(0, modes, details)
                loss_sum += loss.detach()
            successes += 1
            window += 1
        scaler.update()
        if window and (successes % args.log_every == 0 or successes == args.updates):
            event = {"successful_updates": successes, "attempted_updates": attempts, "min_snr_v_loss": float(loss_sum / window),
                     "grad_scale": scaler.get_scale(), "training_seconds": time.monotonic() - started}
            records.append(event)
            result["training_history"] = records
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(event), flush=True)
            window = 0
            loss_sum.zero_()
    details = {}
    for name, (samples, possible, taken, full) in zip(ANCHOR_MODES, exposure.tolist()):
        details[name] = {"examples": int(samples), "mean_local_residual_pixels_available": possible / samples if samples else None,
                         "mean_training_edits": taken / samples if samples else None,
                         "mean_anchor_target_hamming_pixels": full / samples if samples else None}
    return ema, {"successful_updates": successes, "skipped_steps": attempts - successes, "anchor_training_exposure": details,
                 "parameters": sum(value.numel() for value in model.parameters()), "seconds": time.monotonic() - started}


@torch.inference_mode()
def sample_residual(model, official, anchor, spikes, seed, *, shape_only=False):
    device = spikes.device
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler = official.diffusion_scheduler()
    scheduler.set_timesteps(official.DIFFUSION_STEPS, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    anchor = anchor[:, None]
    edge, _ = edge_support(anchor)
    spatial = 2 * anchor.float() - 1
    state = torch.randn(anchor.shape, device=device, generator=generator) * edge
    n = len(anchor)
    double = torch.cat((spikes, spikes))
    dropped = torch.ones(2 * n, device=device, dtype=torch.bool) if shape_only else torch.arange(2 * n, device=device) >= n
    for step in scheduler.timesteps.tolist():
        time_rows = torch.full((2 * n,), step, device=device, dtype=torch.long)
        field = torch.cat((state, spatial), dim=1)
        field = torch.cat((field, field)).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            velocity = model(field, time_rows, double, dropped)
        conditional, null = velocity.float().chunk(2)
        guided = null + GUIDANCE * (conditional - null)
        state = scheduler.step(guided, step, state, generator=generator).prev_sample * edge
    probability = (anchor.float() + state).clamp(0, 1)
    potential = ((probability >= 0.5) != anchor) & edge
    edited, before, actual = bounded_flips((probability - 0.5).abs(), potential, anchor)
    output = (anchor ^ edited)[:, 0]
    return output.cpu(), before.cpu(), actual.cpu()


def summarize(exact):
    base = exact[ARMS[0]]
    summary = {name: paired_stats(values, base) for name, values in exact.items()}
    summary["true_vs_shape_only"] = paired_stats(exact[ARMS[-1]], exact[ARMS[1]])
    summary["true_vs_shuffled"] = paired_stats(exact[ARMS[-1]], exact[ARMS[2]])
    return summary


def assess(model, official, held, rendered, first, experiment_started, gain, reference, scorer, args, output):
    device = rendered.device
    values = {}
    for day, (name, responses) in enumerate(held.items()):
        started = time.monotonic()
        anchor = rendered[first[day]]
        day_id = torch.full((len(anchor),), day, device=device, dtype=torch.long)
        true = condition(responses, day_id, gain)
        donors = shuffled_rows(len(anchor), device, args.seed + 5009 * (day + 1))
        conditions = {ARMS[1]: true, ARMS[2]: true[donors], ARMS[3]: true}
        prediction, diagnostics = {ARMS[0]: anchor.cpu()}, {}
        for arm, spike in conditions.items():
            parts, eligible, edits = [], [], []
            for index, rows in enumerate(torch.arange(len(anchor), device=device).split(args.sample_batch_size)):
                seed = args.seed + 1201 * (day + 1) + 31 * index
                bitmap, before, actual = sample_residual(model, official, anchor[rows], spike[rows], seed,
                                                        shape_only=arm == ARMS[1])
                parts.append(bitmap)
                eligible.append(before)
                edits.append(actual)
            prediction[arm] = torch.cat(parts)
            eligible, edits = torch.cat(eligible), torch.cat(edits)
            diagnostics[arm] = {"unchanged_glyphs": int((edits == 0).sum()), "over_budget_before_clip": int((eligible > edits).sum()),
                                "mean_proposed_edits": float(eligible.float().mean()), "mean_accepted_edits": float(edits.float().mean()),
                                "empty_output_glyphs": int((~prediction[arm].flatten(1).any(dim=1)).sum())}
        generated = time.monotonic()
        exact = {arm: scorer(bits.numpy(), reference, device="cuda:0", batch_size=64) for arm, bits in prediction.items()}
        fold = {"session": name, "metrics": summarize({arm: [scores] for arm, scores in exact.items()}),
                "rendering": diagnostics, "sampling_seconds": generated - started, "ocf_seconds": time.monotonic() - generated}
        for arm, measured in exact.items():
            values.setdefault(arm, []).append(measured)
        output["folds"].append(fold)
        output["summary"] = summarize(values)
        output["seconds"] = time.monotonic() - experiment_started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(fold), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17761)
    parser.add_argument("--model-seed", type=int, default=211)
    parser.add_argument("--inner-anchors", type=int, default=640)
    parser.add_argument("--updates", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--sample-batch-size", type=int, default=24)
    parser.add_argument("--ocf-glyphs", type=int, default=96)
    parser.add_argument("--log-every", type=int, default=320)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_anchored_ddpm.json"))
    args = parser.parse_args()
    if (not 2 <= args.ocf_glyphs <= 320 or args.inner_anchors < 96
            or min(args.updates, args.batch_size, args.sample_batch_size, args.log_every) < 1):
        parser.error("Use a positive matched budget and a sufficient disjoint inner-anchor cohort.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Use one Kaggle T4 with the official training input and public font access.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    official, provenance = official_model(root)
    scorer = official_scorer(root)
    images = load_images(root, device)
    fit, held = held_split(len(images), args.seed, device, earlier_seeds=(961, 2209, 17291))
    if len(fit) - args.inner_anchors < 128:
        raise ValueError("Keep enough independent glyphs in the inner spike-ranking model.")
    sample = held[:args.ocf_glyphs]
    rng = torch.Generator(device=device).manual_seed(args.seed + 37)
    order = torch.randperm(len(fit), device=device, generator=rng)
    inner, core = order[:args.inner_anchors], order[args.inner_anchors:]
    groups = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if len(groups) != 4 or any(counts.shape[1] != 4 for counts in groups.values()):
        raise ValueError("All source queries must use four measured within-day trials.")
    atlas, aligned, codes, font, agreement = freeze_atlas(images, fit, args.model_seed)
    bank, style = styled_bank(atlas, aligned, images[fit], args.model_seed)
    high, visual, near, matches = visual_anchors(bank, images[fit])
    signature, energy = contour_signatures(atlas.images)
    rank_started = time.monotonic()
    inner_model = fit_ranker(atlas, codes, signature, energy, {name: counts[fit[core]] for name, counts in groups.items()},
                            images[fit[core]], aligned[core], args.model_seed)
    train_anchors = rank_anchors(atlas, inner_model, groups, fit[inner])
    del inner_model
    middle = time.monotonic()
    outer_model = fit_ranker(atlas, codes, signature, energy, {name: counts[fit] for name, counts in groups.items()},
                            images[fit], aligned, args.model_seed)
    first = rank_anchors(atlas, outer_model, groups, sample)
    final_rank_time = time.monotonic()
    del atlas, signature, energy, outer_model, aligned
    fit_trials = torch.stack([counts50(counts[fit]) for counts in groups.values()])
    gain = fit_input_scale(fit_trials)
    trials = fit_trials.mean(dim=2)
    held_counts = {name: counts50(values[sample]).mean(dim=1) for name, values in groups.items()}
    del groups, fit_trials
    output = {"seed": args.seed, "model_seed": args.model_seed, "mouse": MOUSE, "gpu": torch.cuda.get_device_name(device),
              "model": provenance, "font": font, "font_alignment": agreement, "font_style": style, "fitting_matches": matches,
              "fit_glyphs": len(fit), "outer_whole_glyphs": len(held), "ocf_glyphs_per_session": len(sample),
              "inner_spike_anchor_glyphs": len(inner), "inner_ranker_fit_glyphs": len(core),
              "ranker": {"policy": "Full public atlas, 65% directional ridge and 35% dual Poisson; fixed Unihan and rendering.",
                         "inner_anchor_is_paired_visual_match_fraction": float((train_anchors == visual[inner]).float().mean()),
                         "fit_and_rank_inner_seconds": middle - rank_started, "fit_and_rank_outer_seconds": final_rank_time - middle},
              "settings": {"training_updates": args.updates, "batch_size": args.batch_size, "ema": EMA_DECAY, "spike_dropout": DROP_PROBABILITY,
                           "cfg": GUIDANCE, "signed_residual_mask_radius_chebyshev": MARGIN, "maximum_ink_edit_fraction": MAX_INK_FRACTION,
                           "anchor_source_weights": dict(zip(ANCHOR_MODES, (0.5, 0.25, 0.25)))},
              "protocol": "The entire outer glyph cohort is excluded from every fit. Training same-day four-repeat inputs pair with high-confidence "
                          "Noto images, shape neighbors, and spike-only choices from a separate inner glyph-excluded ranker. The same binary atlas, "
                          "edge band and edit cap apply to training targets and inference outputs. The official cosine velocity denoiser receives "
                          "the unchanging anchor as a second input channel. Residual diffusion starts with Gaussian noise on the editable support; "
                          "outside support is zero throughout training and sampling. The outer readout freezes one common first-choice glyph. True, "
                          "same-day deranged and absent spikes use the same EMA, 64-step sampler, and stochastic noise. No reference-selected "
                          "timesteps, draws or candidates. Exact held OCF is paired by glyph. No official test records.",
              "preparation_seconds": time.monotonic() - started, "folds": []}
    print(json.dumps({"anchored_protocol": output["settings"], "ranker": output["ranker"],
                      "preparation_seconds": output["preparation_seconds"]}), flush=True)
    model, trained = fit_residual(official, trials, bank, images[fit] >= 128, high, visual, near, inner,
                                  train_anchors, gain, args, output)
    output["training"] = trained
    del trials
    reference = (images[sample] >= 128).cpu().numpy()
    assess(model, official, held_counts, bank, first, started, gain, reference, scorer, args, output)
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
