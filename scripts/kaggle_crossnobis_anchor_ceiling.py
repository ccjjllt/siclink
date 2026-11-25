"""Measure Task 2 geometry with known held-fold anchors as a nonproduction ceiling."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import fit_hierarchical_gallery
from kaggle_crossnobis_validation import (blend_shortlist, crossnobis_profiles, profile_cosines, projected_views,
                                         select_anchors, source_geometry)


def ranks(logits, truth):
    top = logits.topk(32, dim=1).indices
    correct = top == truth[:, None]
    return {f"top{k}": float(correct[:, :k].any(1).float().mean()) for k in (1, 5, 16, 32)}


def anchor_probe(unary, source, a, b, rows, glyphs, truth):
    """Use the identical non-anchor rows to score baseline and both anchor identities."""
    if len(rows) < 4:
        return {"count": len(rows)}
    kept = torch.ones(len(truth), dtype=torch.bool)
    kept[rows] = False
    target_profiles = crossnobis_profiles(a, b, a[rows], b[rows])
    similarity = profile_cosines(target_profiles, source_geometry(source, glyphs))
    blended = blend_shortlist(unary, similarity, rows, shortlist=32, weight=0.15)
    return {"count": len(rows), "anchor_precision": float((truth[rows] == glyphs).float().mean()),
            "scored_rows": int(kept.sum()), "baseline": ranks(unary[kept], truth[kept]),
            "geometry_only": ranks(similarity[kept], truth[kept]),
            "blended": ranks(blended[kept], truth[kept])}


def summarize(folds):
    output = {}
    for name in next(iter(folds.values())):
        rows = [fold[name] for fold in folds.values() if "baseline" in fold[name]]
        if not rows:
            continue
        output[name] = {"folds": len(rows), "mean_anchor_count": float(np.mean([r["count"] for r in rows])),
                        "mean_anchor_precision": float(np.mean([r["anchor_precision"] for r in rows]))}
        for method in ("baseline", "geometry_only", "blended"):
            output[name][method] = {key: float(np.mean([r[method][key] for r in rows])) for key in rows[0][method]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/crossnobis_anchor_ceiling.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    rng = np.random.default_rng(args.seed)
    root = locate_data()
    catalog = load_catalog(root)
    size = len(load_images(root, "cpu"))
    output = {"diagnostic_only": True, "seed": args.seed,
              "protocol": "Labels identify oracle anchors in held training folds only. Anchor rows are excluded "
                          "from all ranking metrics. No official test inputs or predictions are used.", "folds": {}}
    for mouse in ("mice1", "mice2"):
        recordings = grouped_blocks(root, catalog, mouse, "cpu", size)
        for session, held in recordings.items():
            training = {name: values for name, values in recordings.items() if name != session}
            truth = torch.as_tensor(rng.permutation(size))
            block = held[truth]
            order = torch.as_tensor(rng.random((len(block), block.shape[1])).argsort(1))
            query = block.gather(1, order[..., None].expand_as(block))
            gallery = fit_hierarchical_gallery(training)
            unary = gallery.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
            source, a, b = projected_views(training, query, gallery, dimension=32)
            rows, glyphs = select_anchors(gallery, query, count=128)
            results = {"predicted_anchors": anchor_probe(unary, source, a, b, rows, glyphs, truth),
                       "oracle_matched_rows": anchor_probe(unary, source, a, b, rows, truth[rows], truth)}
            random_rows = torch.as_tensor(rng.permutation(size)[:96])
            for count in (16, 48, 96):
                chosen = random_rows[:count]
                results[f"oracle_random_{count}"] = anchor_probe(unary, source, a, b, chosen, truth[chosen], truth)
            output["folds"][f"{mouse}/{session}"] = results
            output["summary"] = summarize(output["folds"])
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"mouse": mouse, "session": session, **results}), flush=True)
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
