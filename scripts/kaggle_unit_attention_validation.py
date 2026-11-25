"""Compare unit attention and the Task 3 v6 policy on matched held repeats."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder, make_count_templates
from glyph_mbr import contour_signatures
from glyph_unit_attention import attention_glyph_scores, fit_glyph_target, fit_unit_attention, prepare_units
from kaggle_diagnostics import official_scorer
from kaggle_encoding_validation import seen_folds


def evaluate(models, moments, target, contour, poisson, queries, images, score, device, args):
    labels = torch.arange(len(images), device=device)
    masks = (images >= 128).cpu().numpy()
    random = np.random.default_rng(args.seed)
    n_per_session = min(len(images), args.ocf_images // len(queries))
    references, predictions, top1, top5 = [], {}, {}, {}
    for session, observed in queries.items():
        selection = random.choice(len(images), n_per_session, replace=False)
        selected = torch.as_tensor(selection, device=device)
        references.append(masks[selection])
        v6 = contour_retrieval_scores(contour, observed, session)["contour_ridge"]
        if poisson is not None:
            rates = poisson.for_recording(session, observed)["forward"]
            v6 = blend_contour_poisson(v6, count_template_logits(observed, rates, "timed"))
        logits = {name: attention_glyph_scores(model, observed, moments[session], target)
                  for name, model in models.items()}
        gallery_shape = (v6.shape[1],)
        blend = 0.70 * F.layer_norm(v6, gallery_shape, eps=1e-12)
        blend += 0.30 * F.layer_norm(logits["unit_identity"], gallery_shape, eps=1e-12)
        for name, values in {"v6": v6, **logits, "v6_identity_blend": blend}.items():
            top = values.topk(5, dim=1).indices
            hits = top == labels[:, None]
            top1[name] = top1.get(name, 0) + int(hits[:, 0].sum())
            top5[name] = top5.get(name, 0) + int(hits.any(dim=1).sum())
            predictions.setdefault(name, []).append(masks[top[selected, 0].cpu().numpy()])

    reference = np.concatenate(references)
    scored = {name: np.asarray(score(np.concatenate(blocks), reference, device=str(device), batch_size=4))
              for name, blocks in predictions.items()}
    n = len(queries) * len(images)
    metrics = []
    for name, values in scored.items():
        delta = values - scored["v6"]
        by_session = delta.reshape(len(queries), n_per_session).mean(axis=1)
        row = {"method": name, "ocf": float(values.mean()), "ocf_n": len(values),
               "paired_delta_vs_v6": float(delta.mean()),
               "paired_se_vs_v6": float(delta.std(ddof=1) / len(delta)**0.5),
               "paired_delta_by_session": dict(zip(queries, by_session.tolist())),
               "top1": top1[name] / n, "top5": top5[name] / n, "top_n": n}
        if name == "unit_identity" and "anonymous" in scored:
            identity_delta = values - scored["anonymous"]
            row.update(paired_delta_vs_anonymous=float(identity_delta.mean()),
                       paired_se_vs_anonymous=float(identity_delta.std(ddof=1) / len(delta)**0.5))
        metrics.append(row)
    return metrics


def run_mouse(mouse, device, root, catalog, score, args):
    started = time.monotonic()
    images = load_images(root, device)
    raw = grouped_blocks(root, catalog, mouse, device, len(images))
    repeat_seed = args.seed if args.repeat_seed is None else args.repeat_seed
    training, queries = seen_folds(raw, repeat_seed, ("task3",))["task3"]
    del raw
    print(f"{mouse} on {device}: fitting organized unit tokens", flush=True)
    signature, energy = contour_signatures(images)
    target = fit_glyph_target(signature, energy, seed=args.seed)
    normalized, moments, footprint = prepare_units(training, images, functional_footprints=args.functional_footprints)
    variants = {"unit_identity": (True, None)}
    if not args.skip_no_id:
        variants["anonymous"] = (False, None)
    if footprint is not None:
        variants["functional_identity"] = (True, footprint)
    models, fit_losses = {}, {}
    for name, (identity, functional) in variants.items():
        models[name], fit_losses[name] = fit_unit_attention(normalized, target, identity=identity,
                                                          footprint=functional, epochs=args.epochs,
                                                          seed=args.seed, batch_size=args.batch_size)
    del normalized

    contour = fit_contour_retrieval(training, signature, energy, task="task3", seed=args.seed)
    poisson = None
    if mouse == "mice1":
        encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
        poisson = make_count_templates(encoder, images, training)
    metrics = evaluate(models, moments, target, contour, poisson, queries, images, score, device, args)
    for row in metrics:
        print(json.dumps({"mouse": mouse, **row}), flush=True)
    return {"mouse": mouse, "device": str(device), "sessions": list(training), "fit_losses": fit_losses,
            "v6_policy": "contour_poisson" if poisson is not None else "contour_ridge",
            "metrics": metrics, "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ocf-images", type=int, default=384)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--functional-footprints", action="store_true")
    parser.add_argument("--skip-no-id", action="store_true", help="Omit the matched anonymous-unit ablation")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/unit_attention_validation.json"))
    args = parser.parse_args()
    if args.ocf_images < 4 or min(args.epochs, args.encoder_epochs, args.batch_size) < 1:
        parser.error("Use at least four paired OCF images and positive training settings")

    devices = [] if args.device == "cpu" else [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if args.device == "cuda" and not devices:
        parser.error("The requested CUDA device is unavailable")
    devices = devices or [torch.device("cpu")]
    torch.set_num_threads(4 if devices[0].type == "cpu" else 2)
    if len(args.mice) > 1:
        for device in devices:
            if device.type == "cuda":
                identity = torch.eye(2, device=device)
                torch.linalg.cholesky(identity)
                torch.linalg.qr(identity)
                torch.linalg.svd(identity)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    score = official_scorer(root)
    output = {"seed": args.seed, "repeat_seed": args.seed if args.repeat_seed is None else args.repeat_seed,
              "epochs": args.epochs, "encoder_epochs": args.encoder_epochs, "ocf_images": args.ocf_images,
              "functional_footprints": args.functional_footprints, "batch_size": args.batch_size,
              "score_weights": {"v6": 0.70, "unit_identity": 0.30}, "results": []}
    with ThreadPoolExecutor(max_workers=min(len(args.mice), len(devices))) as pool:
        jobs = [pool.submit(run_mouse, mouse, devices[index % len(devices)], root, catalog, score, args)
                for index, mouse in enumerate(args.mice)]
        for job in as_completed(jobs):
            output["results"].append(job.result())
            output["results"].sort(key=lambda result: result["mouse"])
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
