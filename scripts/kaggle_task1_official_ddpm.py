"""Pair the official Task 1 denoiser with single or mixed-repeat conditioning.

Both training arms share the same initialization, anchor trials, timestep,
Gaussian noise, conditional-drop mask and successful update budget. Whole
glyphs are withheld from every model. One predetermined, noise-matched draw
per glyph is scored against exact held OCF and a frozen public-atlas decoder.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_prior import fit_atlas_prior
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, standard_error, standardized


MOUSE = "mice1"
DIFFUSERS_VERSION = "0.35.2"
EMA_DECAY = 0.999
GUIDANCE = 1.5
DROP_PROBABILITY = 0.10
BINARY_CUTOFF = 0.50
ARM_NAMES = ("single_training", "mixed_training")
MIX_COUNTS = (1, 2, 4)
ATLAS_NAME = "public_atlas"


def official_model(root):
    bundled = root.parent / "participant_package_en" / "participant_package_en"
    source = bundled / "example" / "model.py"
    if not source.is_file():
        raise FileNotFoundError(f"Official SpikeDenoiser not found at {source}")
    try:
        installed = version("diffusers")
    except PackageNotFoundError:
        installed = None
    if installed != DIFFUSERS_VERSION:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", f"diffusers=={DIFFUSERS_VERSION}"], check=True)
    sys.path.insert(0, str(bundled))
    spec = importlib.util.spec_from_file_location("siclink_official_ddpm_example", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.DIFFUSION_STEPS != 64 or tuple(module.MODEL_IMAGE_SIZE) != (64, 64):
        raise RuntimeError("The official example no longer uses the predeclared native 64-step configuration.")
    return module, {"model_source_sha256": sha256(source.read_bytes()).hexdigest(),
                    "diffusers_version": version("diffusers"), "official_diffusion_steps": module.DIFFUSION_STEPS}


def held_split(size, seed, device, *, earlier_seeds=(961, 2209)):
    # Reconstruct previous 320-glyph cohorts in their original split order.
    used = np.zeros(size, dtype=bool)
    for earlier in earlier_seeds:
        order = np.random.default_rng(earlier).permutation(size)
        used[order[~used[order]][:320]] = True
    order = np.random.default_rng(seed).permutation(size)
    held = order[~used[order]][:320]
    fit = np.setdiff1d(np.arange(size), held)
    if len(held) != 320:
        raise ValueError("Need a full, glyph-disjoint 320-row held comparison.")
    return torch.as_tensor(fit, device=device), torch.as_tensor(held, device=device)


def counts50(counts):
    return counts.unflatten(-1, (-1, 5, 2)).sum(-1).flatten(-2)


def fit_input_scale(training):
    day_means = training.mean(dim=(1, 2))
    reference = day_means.mean(dim=0)
    return ((reference + 0.25) / (day_means + 0.25)).clamp(0.50, 2.0)


def condition(counts, day, gain):
    scaled = counts * gain[day]
    return torch.cat((scaled, F.one_hot(day, num_classes=len(gain)).to(scaled)), dim=1)


def matched_inputs(training, order, repeat_generator):
    _, glyphs, _, features = training.shape
    day = order // (4 * glyphs)
    glyph = (order // 4) % glyphs
    anchor = order % 4
    rows = training[day, glyph]
    other = torch.rand((len(order), 3), generator=repeat_generator, device=training.device).argsort(dim=1) + 1
    order = torch.cat((anchor[:, None], (anchor[:, None] + other) % 4), dim=1)
    arranged = rows.gather(1, order[:, :, None].expand(-1, -1, features))
    draw = torch.randint(3, (len(rows),), generator=repeat_generator, device=training.device)
    # The index 0/1/2 maps to exactly 1/2/4 repeats.
    repeats = 2 ** draw
    retained = torch.arange(4, device=training.device)[None] < repeats[:, None]
    pooled = (arranged * retained[:, :, None]).sum(dim=1) / repeats[:, None]
    return day, glyph, arranged[:, 0], pooled, draw


def source_targets(scheduler, target, generator):
    device = target.device
    timestep = torch.randint(scheduler.config.num_train_timesteps, (len(target),), device=device, generator=generator)
    noise = torch.randn(target.shape, dtype=target.dtype, device=device, generator=generator)
    noisy = scheduler.add_noise(target, noise, timestep).contiguous(memory_format=torch.channels_last)
    velocity = scheduler.get_velocity(target, noise, timestep)
    drop = torch.rand(len(target), device=device, generator=generator) < DROP_PROBABILITY
    alpha = scheduler.alphas_cumprod[timestep]
    snr = alpha / (1 - alpha)
    weight = snr.clamp(max=5) / (snr + 1)
    return timestep, noisy, velocity, drop, weight


def train(official, training, image_bank, gain, args, report):
    device = training.device
    torch.manual_seed(args.model_seed)
    first = official.build_model(training.shape[-1] + len(gain)).to(device, memory_format=torch.channels_last)
    models = (first, deepcopy(first))
    averages = [deepcopy(model).eval().requires_grad_(False) for model in models]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, fused=True) for model in models]
    scaler = torch.amp.GradScaler("cuda", init_scale=2048)
    scheduler = official.diffusion_scheduler()
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=device)
    shuffled, repeats, noise = (torch.Generator(device=device).manual_seed(args.model_seed + offset)
                                for offset in (29, 43, 61))
    anchors = 4 * training.shape[1] * len(gain)
    usable = (anchors // args.batch_size) * args.batch_size
    if not usable:
        raise ValueError("The fixed training batch exceeds the source trial bank.")
    batches = iter(())
    history, start = [], time.monotonic()
    successes, attempts, windows = 0, 0, 0
    accumulated = torch.zeros(2, device=device)
    chosen_repeats = torch.zeros(3, dtype=torch.long, device=device)
    for model in models:
        model.train()
    while successes < args.updates:
        if attempts >= args.updates + 100:
            raise RuntimeError("FP16 overflow prevented the same fixed number of successful training updates.")
        batch = next(batches, None)
        if batch is None:
            batches = iter(torch.randperm(anchors, device=device, generator=shuffled)[:usable].split(args.batch_size))
            batch = next(batches)
        day, glyph, once, mixed, draw = matched_inputs(training, batch, repeats)
        chosen_repeats += torch.bincount(draw, minlength=3)
        clean = image_bank[glyph, None].float().div_(127.5).sub_(1)
        timestep, noisy, velocity, drop, weight = source_targets(scheduler, clean, noise)
        norms, readings = [], []
        for model, optimizer, counts in zip(models, optimizers, (once, mixed)):
            optimizer.zero_grad(set_to_none=True)
            spike = condition(counts, day, gain)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(noisy, timestep, spike, drop)
            error = (prediction.float() - velocity).square().flatten(1).mean(dim=1)
            loss = (weight * error).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            readings.append(loss.detach())
        # Preserve the matched update schedule even if either branch overflows.
        attempts += 1
        loss_values = torch.stack(readings)
        if bool(torch.isfinite(torch.cat((torch.stack(norms), loss_values))).all()):
            for optimizer in optimizers:
                scaler.step(optimizer)
            with torch.no_grad():
                for ema, model in zip(averages, models):
                    for average, parameter in zip(ema.parameters(), model.parameters()):
                        average.lerp_(parameter, 1 - EMA_DECAY)
            successes += 1
            windows += 1
            accumulated += loss_values
        scaler.update()
        if windows and (successes % args.log_every == 0 or successes == args.updates):
            event = {"paired_successful_updates": successes, "paired_attempts": attempts,
                     "train_min_snr_v_loss": dict(zip(ARM_NAMES, (accumulated / windows).tolist())),
                     "grad_scale": scaler.get_scale(), "training_seconds": time.monotonic() - start}
            history.append(event)
            report["training_history"] = history
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(event), flush=True)
            accumulated.zero_()
            windows = 0
    final = {"successful_steps_each": successes, "paired_skipped_steps": attempts - successes,
             "anchor_examples_per_arm": successes * args.batch_size,
             "mixed_repeat_count_in_all_attempts": dict(zip(MIX_COUNTS, chosen_repeats.tolist())),
             "trainable_parameters_per_arm": sum(parameter.numel() for parameter in first.parameters()),
             "seconds": time.monotonic() - start}
    if args.save_ema:
        path = args.output.with_name("official_ddpm_repeat_ema.pt")
        torch.save({"inputs": training.shape[-1] + len(gain), "day_gain": gain.cpu(), "updates": successes,
                    **{name: model.state_dict() for name, model in zip(ARM_NAMES, averages)}}, path)
        final["ema_checkpoint"] = path.name
    return dict(zip(ARM_NAMES, averages)), final


@torch.inference_mode()
def draw_images(model, official, counts, seed):
    """Replicate the official stochastic DDPM sampler with FP16 convolutions."""
    device = counts.device
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler = official.diffusion_scheduler()
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=device)
    n = len(counts)
    image = torch.randn((n, 1, *official.MODEL_IMAGE_SIZE), device=device, generator=generator)
    doubled = torch.cat((counts, counts))
    dropped = torch.arange(2 * n, device=device) >= n
    for timestep in scheduler.timesteps.tolist():
        time_rows = torch.full((2 * n,), timestep, device=device, dtype=torch.long)
        input_image = torch.cat((image, image)).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            velocity = model(input_image, time_rows, doubled, dropped)
        conditional, unconditional = velocity.float().chunk(2)
        guided = unconditional + GUIDANCE * (conditional - unconditional)
        image = scheduler.step(guided, timestep, image, generator=generator).prev_sample
    return (image[:, 0] + 1).mul(127.5).clamp(0, 255).round().to(torch.uint8).cpu()


def paired_stats(values, base):
    values = np.stack(values)
    delta = (values - np.stack(base)).mean(axis=0)
    return {"ocf": float(values.mean()), "paired_delta": float(delta.mean()), "paired_glyph_se": standard_error(delta),
            "pairs": int(values.size), "glyphs": len(delta)}


def summarize(rows):
    four = "_4_repeats"
    one = "_1_repeats"
    return {"primary_binary_mixed_vs_single_4": paired_stats(rows[ARM_NAMES[1] + four + "_binary"],
                                                               rows[ARM_NAMES[0] + four + "_binary"]),
            "secondary_gray_mixed_vs_single_4": paired_stats(rows[ARM_NAMES[1] + four + "_gray"],
                                                               rows[ARM_NAMES[0] + four + "_gray"]),
            "secondary_binary_mixed_vs_single_1": paired_stats(rows[ARM_NAMES[1] + one + "_binary"],
                                                                rows[ARM_NAMES[0] + one + "_binary"]),
            "secondary_gray_mixed_vs_single_1": paired_stats(rows[ARM_NAMES[1] + one + "_gray"],
                                                                rows[ARM_NAMES[0] + one + "_gray"]),
            "against_matched_public_atlas": {name: paired_stats(values, rows[ATLAS_NAME + ("_1_repeats" if "_1_repeats" in name else four)])
                                             for name, values in rows.items()}}


def paired_open_decoder(groups, images, fit, held_queries, args):
    device = images.device
    training = {name: counts[fit] for name, counts in groups.items()}
    bank = images[fit]
    rng = np.random.default_rng(args.model_seed + 2)
    selected = torch.as_tensor(rng.choice(len(fit), 128, replace=False), device=device)
    atlas, font, codes = build_font_atlas(bank[selected], device, return_codepoints=True)
    aligned, agreement = align_training_glyphs(atlas, bank)
    style, style_report = fit_aligned_font_style(aligned, bank, seed=args.model_seed)
    prior = fit_atlas_prior(atlas, codes, bank)
    contour = fit_contour_retrieval(training, *contour_signatures(aligned), task="task2", seed=args.model_seed)
    signature, energy = contour_signatures(atlas.images)
    contour = replace(contour, signature=signature, energy=energy)
    encoder = fit_glyph_encoder(training, aligned, epochs=18, seed=args.model_seed)
    choices = {}
    with torch.inference_mode():
        for session, values in held_queries.items():
            rates = encoder.expected_counts(atlas.images, session, batch_size=192)
            choices[session] = {}
            for repeats in (1, 4):
                query = values[:, :repeats]
                direction = standardized(contour_retrieval_scores(contour, query, session)["contour_ridge"])
                total = standardized(count_template_logits(query, rates, "rate"))
                timed = standardized(count_template_logits(query, rates, "timed"))
                neural = standardized(0.5 * (total + timed))
                score = prior.variants(0.65 * direction + 0.35 * neural)["usage_unseen"]
                choice = score.argmax(dim=1)
                rendered = style.apply(atlas.images[choice, None].float() / 255)[:, 0] >= BINARY_CUTOFF
                choices[session][f"{ATLAS_NAME}_{repeats}_repeats"] = rendered.cpu()
    return choices, {"policy": "65% directional contour ridge, 35% rate and timed Poisson, fixed Unihan prior.",
                     "atlas": font, "font_alignment": agreement, "style_fit": style_report,
                     "stimulus_support": "entire fixed public Noto atlas", "sharp_ocf": True}


def shuffled_rows(number, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    order = torch.randperm(number, generator=generator, device=device)
    permutation = torch.empty_like(order)
    permutation[order] = order.roll(1)
    return permutation


def exact(prediction, reference, scorer, *, grayscale):
    pixels = prediction.float().div(255).numpy() if grayscale else (prediction >= 128).numpy()
    return scorer(pixels, reference, device="cuda:0", batch_size=48)


def score_session(models, official, day, session, queries, gain, matched, reference, scorer, args):
    started = time.monotonic()
    device = queries.device
    values = counts50(queries)
    days = torch.full((len(queries),), day, device=device, dtype=torch.long)
    predictions = {}
    prefix = args.shuffle_glyphs
    # Keep the shuffled subset as one whole chunk to share exactly its draw stream.
    chunks = [slice(0, prefix)] + [slice(index, min(index + args.sample_batch_size, len(queries)))
                                   for index in range(prefix, len(queries), args.sample_batch_size)]
    for repeats in (1, 4):
        features = condition(values[:, :repeats].mean(dim=1), days, gain)
        for name, model in models.items():
            labels = []
            for number, rows in enumerate(chunks):
                seed = args.seed + 1009 * (day + 1) + 31 * number
                labels.append(draw_images(model, official, features[rows], seed))
                if repeats == 4 and name == ARM_NAMES[1] and number == 0:
                    deranged = shuffled_rows(prefix, device, seed + 997)
                    null = draw_images(model, official, features[rows][deranged], seed)
            predictions[f"{name}_{repeats}_repeats"] = torch.cat(labels)
    generated = time.monotonic()
    shuffled_binary = exact(null, reference[:prefix], scorer, grayscale=False)
    shuffled_gray = exact(null, reference[:prefix], scorer, grayscale=True)
    result = {name: scorer(bits.numpy(), reference, device="cuda:0", batch_size=48) for name, bits in matched.items()}
    for name, encoded in predictions.items():
        result[name + "_binary"] = exact(encoded, reference, scorer, grayscale=False)
        result[name + "_gray"] = exact(encoded, reference, scorer, grayscale=True)
    control = {"binary_real": result[ARM_NAMES[1] + "_4_repeats_binary"][:prefix], "binary_shuffled": shuffled_binary,
               "gray_real": result[ARM_NAMES[1] + "_4_repeats_gray"][:prefix], "gray_shuffled": shuffled_gray}
    diagnostics = {"binary_foreground_fraction": {name: float((bits >= 128).float().mean())
                                                    for name, bits in predictions.items()},
                   "binary_empty_outputs": {name: int((~(bits >= 128).flatten(1).any(dim=1)).sum())
                                            for name, bits in predictions.items()},
                   "sampling_seconds": generated - started, "exact_ocf_seconds": time.monotonic() - generated}
    print(json.dumps({"session": session, "single_4_binary": float(result[ARM_NAMES[0] + "_4_repeats_binary"].mean()),
                      "mixed_4_binary": float(result[ARM_NAMES[1] + "_4_repeats_binary"].mean()),
                      "open_atlas_4": float(result[ATLAS_NAME + "_4_repeats"].mean())}), flush=True)
    return result, control, diagnostics


def summarize_shuffle(controls):
    return {mode: {**paired_stats(controls[mode + "_real"], controls[mode + "_shuffled"]),
                   "shuffled_ocf": float(np.stack(controls[mode + "_shuffled"]).mean())}
            for mode in ("binary", "gray")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17291)
    parser.add_argument("--model-seed", type=int, default=211)
    parser.add_argument("--updates", type=int, default=3200)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--sample-batch-size", type=int, default=24)
    parser.add_argument("--ocf-glyphs", type=int, default=96)
    parser.add_argument("--shuffle-glyphs", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=160)
    parser.add_argument("--save-ema", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_official_ddpm.json"))
    args = parser.parse_args()
    if not 2 <= args.shuffle_glyphs <= args.ocf_glyphs <= 320 or min(args.updates, args.batch_size, args.sample_batch_size, args.log_every) < 1:
        parser.error("Use a positive matched budget and a shuffle subset within the full score cohort.")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Use one Kaggle T4 and the mounted official training files.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    official, provenance = official_model(root)
    scorer = official_scorer(root)
    images = load_images(root, device)
    fit, held = held_split(len(images), args.seed, device)
    scored = held[:args.ocf_glyphs]
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if len(recordings) != 4 or any(group.shape[1] != 4 for group in recordings.values()):
        raise ValueError("Expected the four labeled recording days and four repeats per glyph in every day.")
    training = torch.stack([counts50(values[fit]) for values in recordings.values()])
    day_gain = fit_input_scale(training)
    generator = torch.Generator(device=device).manual_seed(args.seed + 57)
    queries = {}
    for name, counts in recordings.items():
        values = counts[scored]
        order = torch.rand(values.shape[:2], generator=generator, device=device).argsort(dim=1)
        queries[name] = values.gather(1, order[:, :, None].expand_as(values))

    report = {"seed": args.seed, "model_seed": args.model_seed, "mouse": MOUSE, "gpu": torch.cuda.get_device_name(device),
              "fit_glyphs": len(fit), "whole_held_glyphs": len(held), "scored_glyphs_per_day": len(scored),
              "condition_shuffle_glyphs_per_day": args.shuffle_glyphs, "settings": {"updates": args.updates,
               "batch_size": args.batch_size, "cfg": GUIDANCE, "binary_cutoff": BINARY_CUTOFF, "ema_decay": EMA_DECAY,
               "condition_dropout": DROP_PROBABILITY, "training_model": provenance, "precision": "FP16 with GradScaler",
               "count_bins_ms": 50, "count_input_gain_range": [float(day_gain.min()), float(day_gain.max())], "session_one_hot": True},
              "protocol": "All candidate arms share their training images, fold, starting trial, noise and weights initialization. "
                          "The mixed arm randomly averages an anchored 1, 2 or 4 trials from the same glyph and day, without replacement. "
                          "We skip both updates on an overflow and update both EMA models only after successful steps. The official "
                          "cosine 64-step velocity scheduler and its Min-SNR loss and DDPM sampler are unchanged except FP16 autocast. "
                          "Fixed seed streams are reset across arms, repeat counts and the deranged-conditioning control. "
                          "Each arm submits one precommitted draw to each metric, never choosing against the held reference. "
                          "Grayscale uses 8-bit quantization, binary uses a fixed 0.5 threshold. The primary paired OCF comparison "
                          "is binary mixed vs single training at four observed repeats per recorded day. A separate, frozen whole "
                          "atlas decoder receives exactly those repeats. No official test records.", "folds": []}
    print(json.dumps({"gpu": report["gpu"], "fit_glyphs": len(fit), "settings": report["settings"]}), flush=True)
    report["preparation_seconds"] = time.monotonic() - started
    models, training_report = train(official, training, images[fit], day_gain, args, report)
    report["training"] = training_report
    del training
    atlas_started = time.monotonic()
    atlas_choices, atlas_report = paired_open_decoder(recordings, images, fit, queries, args)
    report["atlas_comparator"] = {**atlas_report, "seconds": time.monotonic() - atlas_started}
    print(json.dumps({"matched_atlas_seconds": report["atlas_comparator"]["seconds"]}), flush=True)
    del recordings
    refs = (images[scored] >= 128).cpu().numpy()
    seen, controls = {}, {}
    for day, (session, counts) in enumerate(queries.items()):
        measured, shuffled, details = score_session(models, official, day, session, counts, day_gain,
                                                     atlas_choices[session], refs, scorer, args)
        for name, score in measured.items():
            seen.setdefault(name, []).append(score)
        for name, score in shuffled.items():
            controls.setdefault(name, []).append(score)
        report["folds"].append({"session": session, "ocf": summarize({name: [score] for name, score in measured.items()}),
                                "condition_shuffle": summarize_shuffle({name: [score] for name, score in shuffled.items()}),
                                **details})
        report["summary"] = summarize(seen)
        report["condition_shuffle_summary"] = summarize_shuffle(controls)
        report["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "condition_shuffle": report["condition_shuffle_summary"],
                      "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
