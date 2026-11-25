"""Fit the validated SiClink models to all labels and package Kaggle predictions."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from glyph_data import BIN_MS, as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_encoding import count_template_logits, fit_glyph_encoder, make_count_templates
from glyph_reconstruction import fit_glyph_reconstructor
from glyph_retrieval import assign_glyphs, combine_logits, fit_retrieval, retrieval_score_components
from kaggle_baseline import write_submission
from kaggle_diagnostics import feature_view, standardize


RETRIEVAL_DECODERS = ("balanced", "template", "reliable")
POISSON_DECODERS = ("poisson", "poisson-balanced")
CONTOUR_DECODERS = ("contour", "contour-poisson")
SEEN_DECODERS = (*RETRIEVAL_DECODERS, *POISSON_DECODERS, *CONTOUR_DECODERS)


def fit_mouse(mouse, device, root, catalog, args, atlas, aligned_images=None, atlas_prior=None, font_style=None):
    images = load_images(root, device)
    masks = (images >= 128).to(torch.uint8) * 255
    training = grouped_blocks(root, catalog, mouse, device, len(images))
    task2_decoder = args.task2_decoder
    blend_hierarchical = task2_decoder == "validated-hierarchical"
    if task2_decoder in ("validated", "validated-hierarchical"):
        task2_decoder = "poisson-balanced" if mouse == "mice1" else "balanced"
    task3_decoder = args.task3_decoder
    if task3_decoder == "contour-validated":
        task3_decoder = "contour-poisson" if mouse == "mice1" else "contour"
    chosen = {"task2": task2_decoder, "task3": task3_decoder}

    unseen_prototypes = {}
    stats = {}
    for session, counts in training.items():
        standardized, stats[session] = standardize(feature_view(counts, args.unseen_view))
        unseen_prototypes[session] = standardized.mean(1)
    unseen = fit_glyph_reconstructor(unseen_prototypes, images, seed=args.seed,
                                   refine_epochs=args.refine_epochs, contour_epochs=args.contour_epochs)
    del unseen_prototypes
    print(f"{mouse} on {device}: unseen decoder fitted", flush=True)

    retrieval = None
    if any(name in RETRIEVAL_DECODERS for name in chosen.values()):
        retrieval = fit_retrieval(as_trial_blocks(training), len(images), device=device,
                                  epochs=args.retrieval_epochs, images=images, seed=args.seed)
    hierarchical = None
    if blend_hierarchical:
        from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery

        hierarchical = fit_hierarchical_gallery(training)
    contours = {}
    if any(name in CONTOUR_DECODERS for name in chosen.values()):
        from glyph_contour_retrieval import fit_contour_retrieval
        from glyph_mbr import contour_signatures

        signature, energy = contour_signatures(images)
        contours = {task: fit_contour_retrieval(training, signature, energy, task=task, seed=args.seed,
                                                n_bins=catalog["window_ms"] // BIN_MS)
                    for task, method in chosen.items() if method in CONTOUR_DECODERS}
    poisson = None
    uses_poisson = any(name in POISSON_DECODERS or name == "contour-poisson" for name in chosen.values())
    if uses_poisson or atlas is not None:
        encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
    if uses_poisson:
        poisson = make_count_templates(encoder, images, training)
    if atlas is not None:
        font_encoder = encoder
        if args.align_font_training:
            font_encoder = fit_glyph_encoder(training, aligned_images.to(device), epochs=args.encoder_epochs,
                                             seed=args.seed)
    del training
    print(f"{mouse}: {json.dumps(chosen)}", flush=True)

    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == mouse and entry["split"] == "test"),
                     key=lambda entry: (entry["task"], entry["session"]))
    results = []
    for entry in entries:
        task = entry["task"]
        block = read_trials(root, catalog, entry, device)
        session = block.session
        if task == "task1":
            counts, _ = standardize(feature_view(block.counts, args.unseen_view), stats[session])
            if atlas is None:
                glyphs = unseen.predict(session, counts.mean(1), variant=args.unseen_variant)
            else:
                from kaggle_gallery_validation import font_scores

                decoded = unseen.predict_variants(session, counts.mean(1))
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    rates = font_encoder.expected_counts(atlas.images, session, batch_size=256)
                scores = font_scores(atlas, rates.float(), block.counts, decoded)["font_fused"]
                if atlas_prior is not None:
                    scores = atlas_prior.variants(scores)["usage_unseen"]
                selected = atlas.images[scores.argmax(1)]
                if font_style is not None:
                    selected = font_style.apply(selected[:, None].float() / 255)[:, 0] * 255
                glyphs = (selected >= 127.5).to(torch.uint8) * 255
        else:
            method = chosen[task]
            if method in POISSON_DECODERS:
                estimates = poisson.for_recording(session, block.counts)
                if method == "poisson-balanced":
                    scores = count_template_logits(block.counts, estimates["shrinkage"], "rate")
                else:
                    scores = count_template_logits(block.counts, estimates["forward"], "timed")
            elif method in CONTOUR_DECODERS:
                from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores

                scores = contour_retrieval_scores(contours[task], block.counts, session)["contour_ridge"]
                if method == "contour-poisson":
                    rates = poisson.for_recording(session, block.counts)["forward"]
                    likelihood = count_template_logits(block.counts, rates, "timed")
                    scores = blend_contour_poisson(scores, likelihood)
            else:
                components = retrieval_score_components(retrieval, block.counts, session, task=task)
                scores = components["reliable"] if method == "reliable" else combine_logits(components, task)
            if task == "task2" and hierarchical is not None:
                repeat_scores = hierarchical.logits(block.counts, query_repeat_mix=0.5)
                scores = blend_hierarchical_logits(scores, repeat_scores["empirical_bayes_repeat_noise"])
            # Candidate scores come only from spikes. The official count of one
            # sample per seen character in a complete session sets the prior.
            labels = assign_glyphs(scores) if method in ("balanced", "poisson-balanced") else scores.argmax(1)
            glyphs = masks[labels]
        results.append((task, block.ids, glyphs.detach().cpu().numpy()))
        print(f"{mouse}/{session}/{task}: {len(block.ids):,} predictions", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--unseen-view", choices=("evoked-multiscale", "evoked-total", "early-late"),
                        default="evoked-total")
    parser.add_argument("--unseen-variant", choices=("linear", "kernel_ink", "refined"), default="refined")
    parser.add_argument("--unseen-decoder", choices=("direct", "font-fused"), default="direct")
    parser.add_argument("--align-font-training", action="store_true",
                        help="fit the public-atlas encoder on matched labeled training glyphs")
    parser.add_argument("--font-usage-prior", action="store_true",
                        help="use public character-usage tiers and soft training-glyph exclusion")
    parser.add_argument("--calibrate-font-output", action="store_true",
                        help="apply a single fit-only geometric correction to selected public glyphs")
    parser.add_argument("--refine-epochs", type=int, default=12)
    parser.add_argument("--contour-epochs", type=int, default=3)
    parser.add_argument("--task2-decoder", choices=("validated", "validated-hierarchical", *SEEN_DECODERS),
                        default="validated")
    parser.add_argument("--task3-decoder", choices=("contour-validated", *SEEN_DECODERS), default="poisson")
    parser.add_argument("--retrieval-epochs", type=int, default=12)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    args = parser.parse_args()
    if (args.align_font_training or args.font_usage_prior or args.calibrate_font_output) and args.unseen_decoder != "font-fused":
        parser.error("Font calibration and the public usage prior require --unseen-decoder font-fused")

    start = time.monotonic()
    torch.set_num_threads(4)
    root = locate_data()
    catalog = load_catalog(root)
    devices = [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())] or [torch.device("cpu")]
    print(f"Candidate devices: {devices}", flush=True)
    atlas, provenance, aligned_images, atlas_prior, font_style = None, None, None, None, None
    if args.unseen_decoder == "font-fused":
        if args.refine_epochs < 1:
            parser.error("The font reranker needs a fitted residual decoder")
        import numpy as np
        from kaggle_gallery_validation import align_training_glyphs, build_font_atlas

        glyphs = load_images(root, devices[0])
        style = np.random.default_rng(args.seed + 2).choice(len(glyphs), 128, replace=False)
        atlas, provenance, *codepoints = build_font_atlas(glyphs[torch.as_tensor(style, device=devices[0])], devices[0],
                                                         return_codepoints=args.font_usage_prior)
        if args.align_font_training or args.calibrate_font_output:
            aligned_images, agreement = align_training_glyphs(atlas, glyphs)
            provenance["training_alignment_agreement"] = agreement
        if args.font_usage_prior:
            from glyph_atlas_prior import fit_atlas_prior

            atlas_prior = fit_atlas_prior(atlas, codepoints[0], glyphs)
            provenance["usage_prior"] = atlas_prior.diagnostics
        if args.calibrate_font_output:
            from glyph_font_style import fit_aligned_font_style

            font_style, summary = fit_aligned_font_style(aligned_images, glyphs, seed=args.seed)
            provenance["output_style"] = {"parameters": asdict(font_style), "fit": summary}
        print(json.dumps(provenance), flush=True)
    predictions = []
    # CUDA lazily initializes its linear-algebra backend on the first fit.
    for index, mouse in enumerate(("mice1", "mice2")):
        device = devices[index % len(devices)]
        mouse_atlas = atlas
        if atlas is not None and atlas.images.device != device:
            mouse_atlas = type(atlas)(atlas.images.to(device), atlas.contours.to(device))
        predictions.extend(fit_mouse(mouse, device, root, catalog, args, mouse_atlas,
                                     aligned_images, atlas_prior, font_style))
    working_dir = Path("/kaggle/working")
    write_submission(working_dir / "kaggle_submission.zip", predictions)
    (working_dir / "candidate.json").write_text(json.dumps({"options": vars(args),
                                                            "devices": [str(device) for device in devices],
                                                            "public_atlas": provenance,
                                                            "seconds": time.monotonic() - start}, indent=2) + "\n")


if __name__ == "__main__":
    main()
