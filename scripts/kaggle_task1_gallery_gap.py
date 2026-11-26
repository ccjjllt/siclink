"""Measure the separate rendering and distractor changes in Task 1 forward retrieval."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import fit_glyph_encoder, poisson_logits
from glyph_font_style import fit_aligned_font_style
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, image_descriptors
from kaggle_task1_repeat_ceiling_validation import fit_inverse_temperature, held_queries
from kaggle_v8_metric_validation import summarize


COUNTS_PER_DAY = (1, 2, 4)
DISTRACTOR_COUNTS = (1024, 4096)


def split_glyphs(count, seed):
    previous = np.concatenate([np.random.default_rng(211).permutation(count)[:320],
                               np.random.default_rng(2651).permutation(count)[-320:]])
    order = np.random.default_rng(seed).permutation(count)
    held = order[~np.isin(order, previous)][:320]
    available = order[~np.isin(order, held)]
    return tuple(torch.as_tensor(rows) for rows in (available[192:], available[:192], held))


@torch.no_grad()
def controlled_atlas(atlas, style, references):
    styled = []
    for batch in atlas.images.split(512):
        alpha = style.apply(batch[:, None].float() / 255)[:, 0]
        styled.append((255 * alpha.clamp(0, 1)).round().to(torch.uint8))
    rendered = torch.cat(styled)
    signatures = torch.cat([image_descriptors((batch >= 128).to(torch.uint8) * 255)
                            for batch in rendered.split(512)])
    refs = image_descriptors((references >= 128).to(torch.uint8) * 255)
    similarity = refs @ signatures.T
    rows, columns = linear_sum_assignment(similarity.numpy(), maximize=True)
    if not np.array_equal(rows, np.arange(len(references))):
        raise RuntimeError("Each reference must have a distinct matched public rendering")
    matched = torch.as_tensor(columns)
    match_quality = float(similarity[torch.arange(len(matched)), matched].mean())
    similarity[:, matched] = -torch.inf
    # Cycle through the reference rows before increasing the visual-neighbor depth.
    ranked = similarity.topk(96, dim=1).indices.T.reshape(-1).tolist()
    distractors = torch.as_tensor(list(dict.fromkeys(ranked))[:max(DISTRACTOR_COUNTS)])
    if len(distractors) != max(DISTRACTOR_COUNTS):
        raise RuntimeError("The controlled comparison needs the full matched set of distractors")
    return rendered, matched, distractors, match_quality


def evoked(values):
    return values.unflatten(-1, (-1, 10))[..., 2:].flatten(-2)


@torch.no_grad()
def balanced_scores(encoder, training, groups, calibration, known_images, rendered, queries):
    exact, public, scales = {}, {}, {}
    calibration_images, held_images = known_images
    for session, (query, restore) in queries.items():
        background = evoked(training[session].mean(dim=(0, 1)))
        cal_rates = 0.98 * evoked(encoder.expected_counts(calibration_images, session)) + 0.02 * background
        cal_spikes = evoked(groups[session][calibration])
        observations = cal_spikes.reshape(-1, 1, cal_spikes.shape[-1])
        trial_logits = poisson_logits(observations, cal_rates).reshape(len(cal_spikes), 4, len(cal_rates))
        scale, report = fit_inverse_temperature(trial_logits)
        scales[session] = report

        closed = 0.98 * evoked(encoder.expected_counts(held_images, session)) + 0.02 * background
        opened = 0.98 * evoked(encoder.expected_counts(rendered, session, batch_size=192)) + 0.02 * background
        for count in COUNTS_PER_DAY:
            spikes = evoked(query[:, :count])
            add_exact = scale * count * poisson_logits(spikes, closed)[restore]
            add_public = scale * count * poisson_logits(spikes, opened)[restore]
            if count not in exact:
                exact[count], public[count] = add_exact, add_public
            else:
                exact[count].add_(add_exact)
                public[count].add_(add_public)
        print(f"{session}: fitted day scale and evaluated both stimulus renderings", flush=True)
    return exact, public, scales


def choices_and_identity(logits, correct_columns, columns):
    best = logits.argmax(1)
    expected = logits.gather(1, correct_columns[:, None])
    ranks = 1 + (logits > expected).sum(dim=1)
    identity = {"top1": float((best == correct_columns).float().mean()),
                "top5": float((ranks <= 5).float().mean()), "top32": float((ranks <= 32).float().mean()),
                "median_rank": float(ranks.float().median()), "gallery_size": logits.shape[1]}
    return columns[best], identity


def measure(repeats, matrix_exact, matrix_public, matched, distractors, sample, metric, render_upper):
    count = len(matched)
    rows = torch.arange(count)
    public_columns = count + torch.arange(matrix_public.shape[1])
    pictures, identities = {}, {}
    for extra in (0, *DISTRACTOR_COUNTS):
        added = distractors[:extra]
        exact_matrix = torch.cat((matrix_exact, matrix_public[:, added]), dim=1)
        approx_matrix = matrix_public[:, torch.cat((matched, added))]
        for family, values, indices in (("exact", exact_matrix, torch.cat((rows, count + added))),
                                        ("rendered", approx_matrix, count + torch.cat((matched, added)))):
            name = f"{family}_{count + extra}"
            pictures[name], identities[name] = choices_and_identity(values, rows, indices)
    pictures["all_public"], identities["all_public"] = choices_and_identity(matrix_public, matched, public_columns)
    pictures["exact_choice_rendered_320"] = count + matched[pictures["exact_320"]]
    exact_ocf = {name: metric.evaluate(indices[sample, None])[:, 0] for name, indices in pictures.items()}
    paired = {name: summarize([values], [exact_ocf["exact_320"]]) for name, values in exact_ocf.items()}
    return {"repeat_count": repeats, "identity_or_matched_font_proxy": identities, "paired_ocf_vs_exact_320": paired,
            "matched_rendering_ocf_if_identity_known": float(render_upper.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7529)
    parser.add_argument("--model-seed", type=int, default=211)
    parser.add_argument("--ocf-glyphs", type=int, default=96)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_gallery_gap.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_glyphs <= 320:
        parser.error("Choose between two and 320 paired reference glyphs.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle Notebook with official training data.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    fit, calibration, held = split_glyphs(len(images), args.seed)
    train = images[fit]
    test = images[held]
    style_sample = np.random.default_rng(args.model_seed + 2).choice(len(train), 128, replace=False)
    atlas, font = build_font_atlas(train[style_sample], device)
    aligned_fit, _ = align_training_glyphs(atlas, train)
    style, style_info = fit_aligned_font_style(aligned_fit, train, seed=args.model_seed)
    rendered, matched, distractors, match_quality = controlled_atlas(atlas, style, test)
    del atlas

    groups = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    training = {session: counts[fit] for session, counts in groups.items()}
    encoder = fit_glyph_encoder(training, train, epochs=18, seed=args.model_seed)
    generator = torch.Generator().manual_seed(args.seed + 1)
    queries = held_queries(groups, held, generator)
    scores_exact, scores_public, calibration_report = balanced_scores(encoder, training, groups, calibration,
                                                                     (images[calibration], test), rendered, queries)
    sample = torch.as_tensor(np.random.default_rng(args.seed + 2).choice(len(held), args.ocf_glyphs, replace=False))
    bank = torch.cat((test, rendered))
    metric = PairedOCF(official_scorer(root), bank, (test[sample] >= 128).numpy(), device, 12)
    upper = metric.evaluate((len(held) + matched[sample])[:, None])[:, 0]
    output = {"seed": args.seed, "model_seed": args.model_seed, "mouse": "mice1", "device": "cpu",
              "fit_glyphs": len(fit), "calibration_glyphs": len(calibration), "held_glyphs": len(held),
              "paired_ocf_glyphs": len(sample), "font": font, "font_style_fit": style_info,
              "matched_font_similarity": match_quality, "calibration": calibration_report,
              "protocol": "Train one encoder on actual fit stimuli and use the same Poisson likelihood for all banks. "
                          "Align known candidate representations once, and add exactly the same shape-near distractors. "
                          "Held images define pairings and distractors only for controlled diagnostics. Full public ranking "
                          "uses only the entire fixed atlas and spike likelihood. Four days are co-indexed by known "
                          "held training glyph here; no actual competition test responses or labels are read.",
              "steps": {}}
    for count in COUNTS_PER_DAY:
        repeats = 4 * count
        output["steps"][repeats] = measure(repeats, scores_exact[count], scores_public[count], matched, distractors, sample,
                                           metric, upper)
        output["seconds"] = time.monotonic() - started
        output["exact_cache_size"] = len(metric.values)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output["steps"][repeats]), flush=True)


if __name__ == "__main__":
    main()
