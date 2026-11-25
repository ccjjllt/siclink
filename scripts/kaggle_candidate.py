"""Fit the validated SiClink models to all labels and package Kaggle predictions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import torch

from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_encoding import count_template_logits, fit_glyph_encoder, make_count_templates
from glyph_reconstruction import fit_glyph_reconstructor
from glyph_retrieval import assign_glyphs, combine_logits, fit_retrieval, retrieval_score_components
from kaggle_baseline import write_submission
from kaggle_diagnostics import feature_view, standardize


RETRIEVAL_DECODERS = ("balanced", "template", "reliable")
POISSON_DECODERS = ("poisson", "poisson-balanced")
SEEN_DECODERS = (*RETRIEVAL_DECODERS, *POISSON_DECODERS)


def fit_mouse(mouse, device, root, catalog, args):
    images = load_images(root, device)
    masks = (images >= 128).to(torch.uint8) * 255
    training = grouped_blocks(root, catalog, mouse, device, len(images))
    task2_decoder = args.task2_decoder
    if task2_decoder == "validated":
        task2_decoder = "poisson-balanced" if mouse == "mice1" else "balanced"
    chosen = {"task2": task2_decoder, "task3": args.task3_decoder}

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
    poisson = None
    if any(name in POISSON_DECODERS for name in chosen.values()):
        encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
        poisson = make_count_templates(encoder, images, training)
        del encoder
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
            glyphs = unseen.predict(session, counts.mean(1), variant=args.unseen_variant)
        else:
            method = chosen[task]
            if method in POISSON_DECODERS:
                estimates = poisson.for_recording(session, block.counts)
                if method == "poisson-balanced":
                    scores = count_template_logits(block.counts, estimates["shrinkage"], "rate")
                else:
                    scores = count_template_logits(block.counts, estimates["forward"], "timed")
            else:
                components = retrieval_score_components(retrieval, block.counts, session, task=task)
                scores = components["reliable"] if method == "reliable" else combine_logits(components, task)
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
    parser.add_argument("--refine-epochs", type=int, default=12)
    parser.add_argument("--contour-epochs", type=int, default=3)
    parser.add_argument("--task2-decoder", choices=("validated", *SEEN_DECODERS), default="validated")
    parser.add_argument("--task3-decoder", choices=SEEN_DECODERS, default="poisson")
    parser.add_argument("--retrieval-epochs", type=int, default=12)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    args = parser.parse_args()

    start = time.monotonic()
    torch.set_num_threads(2)
    root = locate_data()
    catalog = load_catalog(root)
    devices = [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())] or [torch.device("cpu")]
    print(f"Candidate devices: {devices}", flush=True)
    with ThreadPoolExecutor(max_workers=min(2, len(devices))) as executor:
        jobs = [executor.submit(fit_mouse, mouse, devices[index % len(devices)], root, catalog, args)
                for index, mouse in enumerate(("mice1", "mice2"))]
        predictions = [group for job in jobs for group in job.result()]
    working_dir = Path("/kaggle/working")
    write_submission(working_dir / "kaggle_submission.zip", predictions)
    (working_dir / "candidate.json").write_text(json.dumps({"options": vars(args),
                                                            "devices": [str(device) for device in devices],
                                                            "seconds": time.monotonic() - start}, indent=2) + "\n")


if __name__ == "__main__":
    main()
