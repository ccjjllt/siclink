"""Update mice2 Task 3 using contour retrieval and temporal population gain."""

import argparse
import json
from pathlib import Path
import time

import torch

from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_encoding import fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model, gain_template_logits
from glyph_mbr import contour_signatures
from kaggle_baseline import TASK_SIZES
from kaggle_merge_candidate_outputs import encode_grayscale_png, mounted_submission, replace_submission_rows, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle GPU for the forward encoding and single-trial inference.")
    torch.set_num_threads(4)
    started = time.monotonic()
    base = mounted_submission(args.base_sha)
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cuda:0")
    images = load_images(root, device)
    masks = (images >= 128).to(torch.uint8) * 255
    training = grouped_blocks(root, catalog, "mice2", device, len(images))
    contour = fit_contour_retrieval(training, *contour_signatures(images), task="task3", seed=args.seed)
    encoder = fit_glyph_encoder(training, images, epochs=18, seed=args.seed)
    gain = fit_gain_model(training, "timed", temporal=True)

    entries = sorted((block for block in catalog["blocks"]
                      if block["mouse"] == "mice2" and block["split"] == "test" and block["task"] == "task3"),
                     key=lambda block: block["session"])
    predictions = {}
    for entry in entries:
        block = read_trials(root, catalog, entry, device)
        rates = encoder.expected_counts(images, block.session)
        shape = contour_retrieval_scores(contour, block.counts, block.session)["contour_ridge"]
        likelihood = gain_template_logits(block.counts, rates, gain)
        labels = blend_contour_poisson(shape, likelihood).argmax(dim=1)
        for sample_id, pixels in zip(block.ids, masks[labels].cpu().numpy()):
            if sample_id in predictions or not sample_id.startswith("test_task3_pic"):
                raise ValueError("Expected unique Task 3 IDs in the mice2 test recordings.")
            predictions[sample_id] = encode_grayscale_png(pixels)
        print(f"mice2/{block.session}: {len(block.ids)} Task 3 predictions", flush=True)
    if len(predictions) != TASK_SIZES["task3"] // 2:
        raise ValueError("Expected the full set of mice2 Task 3 replacements.")

    output = Path("/kaggle/working/kaggle_submission.zip")
    rows, changed = replace_submission_rows(base, output, predictions)
    manifest = {"base_sha256": args.base_sha, "submission_sha256": sha256(output), "rows": rows,
                "target": "mice2 task3", "selected_rows": len(predictions), "changed_predictions": changed,
                "sessions": [entry["session"] for entry in entries], "seed": args.seed,
                "contour_weight": 0.5, "temporal_gain_weight": 0.5, "gain_per_window": gain.alpha.tolist(),
                "gain_windows_ms": [[25 * start, 25 * end] for start, end in gain.windows],
                "seconds": time.monotonic() - started}
    output.with_name("source_candidates.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
