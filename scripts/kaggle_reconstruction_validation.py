"""Compare unseen-glyph decoders on labeled character holdouts in Kaggle."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_reconstruction import fit_glyph_reconstructor
from kaggle_diagnostics import feature_view, official_scorer, standardize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine-epochs", type=int, default=6)
    parser.add_argument("--contour-epochs", type=int, default=0)
    parser.add_argument("--flow-epochs", type=int, default=0, help="also fit the conditional flow on each selected view")
    parser.add_argument("--flow-samples", type=int, default=4, help="draws per held-out glyph")
    parser.add_argument("--ocf-per-session", type=int, default=48)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--views", nargs="+", default=["original", "evoked-multiscale"])
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/reconstruction.json"))
    args = parser.parse_args()
    if args.flow_epochs:
        from glyph_flow import fit_glyph_flow

    start = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    score = official_scorer(root)
    train, held = split_glyphs(len(images), 320, args.seed, device)
    rng = np.random.default_rng(args.seed + 1)
    queries = rng.choice(len(held), args.ocf_per_session, replace=False)
    query_glyphs = held[torch.as_tensor(queries, device=device)]
    references = images[query_glyphs].cpu().numpy() >= 128
    results = []
    print(f"Task 1 on {device}: {len(held)} unseen glyphs, OCF on {len(queries)} per block", flush=True)

    for mouse in ("mice1", "mice2"):
        counts_by_session = grouped_blocks(root, catalog, mouse, device, len(images))
        for view in args.views:
            train_prototypes, validation_prototypes = {}, {}
            flow_training, flow_queries = {}, {}
            for session, counts in counts_by_session.items():
                fit_responses, stats = standardize(feature_view(counts[train], view))
                train_prototypes[session] = fit_responses.mean(1)
                held_responses, _ = standardize(feature_view(counts[query_glyphs], view), stats)
                validation_prototypes[session] = held_responses.mean(1)
                if args.flow_epochs:
                    flow_training[session] = fit_responses
                    flow_queries[session] = held_responses
            epochs = args.refine_epochs if view != "original" else 0
            decoder = fit_glyph_reconstructor(train_prototypes, images[train], seed=args.seed,
                                              refine_epochs=epochs, contour_epochs=min(epochs, args.contour_epochs))
            if args.flow_epochs:
                flow = fit_glyph_flow(flow_training, images[train], epochs=args.flow_epochs,
                                      seed=args.seed, verbose=True)
                del flow_training
            for session in ("session1", "session4"):
                predictions = decoder.predict_variants(session, validation_prototypes[session])
                if args.flow_epochs:
                    predictions.update(flow.predict_variants(session, flow_queries[session],
                                                              n_samples=args.flow_samples, seed=args.seed))
                paired = {}
                for variant, values in predictions.items():
                    predicted = values.detach().cpu().numpy().astype(np.float32) / 255
                    paired[variant] = score(predicted, references, device=str(device), batch_size=4)
                for variant, scores in paired.items():
                    result = {"mouse": mouse, "session": session, "view": view, "variant": variant,
                              "ocf": float(scores.mean()), "ocf_n": len(scores)}
                    if variant.startswith("flow_") and "refined" in paired:
                        delta = scores - paired["refined"]
                        result["ocf_delta_vs_refined"] = float(delta.mean())
                    results.append(result)
                    print(json.dumps(result), flush=True)
                args.output.write_text(json.dumps({"seed": args.seed, "device": str(device),
                                                   "refine_epochs": args.refine_epochs,
                                                   "contour_epochs": args.contour_epochs,
                                                   "flow_epochs": args.flow_epochs,
                                                   "flow_samples": args.flow_samples if args.flow_epochs else 0,
                                                   "seconds": time.monotonic() - start,
                                                   "results": results}, indent=2) + "\n")
            del decoder
            if args.flow_epochs:
                del flow, flow_queries
        del counts_by_session


if __name__ == "__main__":
    main()
