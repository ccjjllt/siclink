"""Package the validated mice1 Task 3 top-seven density decision over v11."""

import argparse
import base64
import csv
import io
from itertools import zip_longest
import json
from pathlib import Path
import time
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_mbr import contour_signatures
from kaggle_baseline import TASK_SIZES
from kaggle_merge_candidate_outputs import encode_grayscale_png, mounted_submission, replace_submission_rows, sha256
from kaggle_task3_density_top7_recheck import top7_predictions
from kaggle_task3_part_discovery import SESSIONS
from kaggle_task3_repeat_posterior import CLASSIFIER_WEIGHT, fit_models, score_sources


V11_SHA = "20e4dd1f80f63f5635f0bc5cf16369500bb4d15b2d2d3ea9971a8925b07f21e8"
MOUSE, TASK = "mice1", "task3"
SETTINGS = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128)


def open_png(payload):
    glyph = Image.open(io.BytesIO(base64.b64decode(payload, validate=True)))
    if glyph.format != "PNG" or glyph.size != (64, 64) or glyph.mode != "L":
        glyph.close()
        raise ValueError("Expected each prediction to be a 64×64 grayscale PNG.")
    return glyph


def stored_predictions(path, selected_ids):
    with ZipFile(path) as archive, io.TextIOWrapper(archive.open("submission.csv"), encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream)
        if next(rows) != ["sample_id", "prediction"]:
            raise ValueError("Expected the official v11 CSV header.")
        predictions = {sample_id: value for sample_id, value in rows if sample_id in selected_ids}
    if set(predictions) != selected_ids:
        raise ValueError("The v11 package does not contain every targeted Task 3 sample.")
    return predictions


def verify_archive(original, amended, changes):
    """Check every row's identity, unchanged bytes, and decoded image properties."""
    expected_ids = (f"test_{task}_pic{index}" for task, size in TASK_SIZES.items() for index in range(size))
    actual = set()
    with ZipFile(original) as before, ZipFile(amended) as after:
        if after.namelist() != ["submission.csv"] or after.testzip() is not None:
            raise ValueError("The amended submission ZIP has unexpected entries or corruption.")
        with before.open("submission.csv") as raw, after.open("submission.csv") as updated:
            head = raw.readline()
            if head.rstrip(b"\r\n") != b"sample_id,prediction" or updated.readline() != head:
                raise ValueError("The official CSV header must be preserved byte for byte.")
            for expected, first, second in zip_longest(expected_ids, raw, updated):
                if expected is None or first is None or second is None:
                    raise ValueError("The submitted CSV row count has changed.")
                original_id, _ = first.rstrip(b"\r\n").split(b",", 1)
                updated_id, payload = second.rstrip(b"\r\n").split(b",", 1)
                if original_id != updated_id or updated_id != expected.encode("ascii"):
                    raise ValueError("The package must keep the complete official ID ordering.")
                if expected in changes:
                    if first == second or payload != changes[expected].encode("ascii"):
                        raise ValueError("A targeted replacement is missing or does not match the new image.")
                    actual.add(expected)
                elif first != second:
                    raise ValueError("An unchanged prediction was modified.")
                with open_png(payload) as glyph:
                    glyph.verify()
    if actual != set(changes):
        raise ValueError("Expected all and only the listed replacements.")
    return len(actual)


def predict(models, entries, root, catalog, images, masks):
    device = images.device
    target_ids, blocks = set(), {}
    for entry in entries:
        block = read_trials(root, catalog, entry, device)
        if block.counts.shape[:2] != (len(masks), 1) or len(block.ids) != len(masks):
            raise ValueError("Each target recording must contain one full gallery of single-repeat queries.")
        v11, _, density = score_sources(models, block.counts, images, block.session, MOUSE, SETTINGS.dimensions)
        original, adjusted = top7_predictions(v11, density)
        ids = tuple(block.ids)
        if len(set(ids)) != len(ids) or target_ids.intersection(ids):
            raise ValueError("Expected unique sample IDs throughout mice1 Task 3.")
        if any(not sample_id.startswith("test_task3_pic") for sample_id in ids):
            raise ValueError("Every target ID must belong to Task 3.")
        target_ids.update(ids)
        blocks[block.session] = (ids, original.cpu().tolist(), adjusted.cpu().tolist())
    if len(target_ids) != TASK_SIZES[TASK] // 2:
        raise ValueError("The candidate must cover the four complete mice1 Task 3 recordings.")
    return blocks, target_ids


def changes_from_predictions(blocks, stored, rendered):
    sessions, replacements, changed = {}, {}, {}
    for session, (ids, recomputed, selected) in blocks.items():
        summary = {"queries": len(ids), "v11_recreated_pixels_match": 0, "v11_recreated_png_bytes_match": 0,
                   "same_recreated_label": 0, "changed_recreated_label": 0, "changed_png_rows": 0,
                   "changed_png_rows_with_recreation_mismatch": 0}
        changed[session] = []
        for sample_id, anchor, choice in zip(ids, recomputed, selected):
            original = stored[sample_id]
            with open_png(original) as image:
                original_pixels = np.asarray(image)
                recreated = np.array_equal(original_pixels, rendered[anchor])
                summary["v11_recreated_pixels_match"] += int(recreated)
                if recreated:
                    summary["v11_recreated_png_bytes_match"] += int(original == encode_grayscale_png(rendered[anchor]))
                if choice == anchor:
                    summary["same_recreated_label"] += 1
                    continue
                summary["changed_recreated_label"] += 1
                if np.array_equal(original_pixels, rendered[choice]):
                    continue
            replacements[sample_id] = encode_grayscale_png(rendered[choice])
            changed[session].append(sample_id)
            summary["changed_png_rows"] += 1
            summary["changed_png_rows_with_recreation_mismatch"] += int(not recreated)
        sessions[session] = summary
        print(json.dumps({"session": session, **summary}), flush=True)
    return replacements, changed, sessions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle GPU for the primary candidate recreation.")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    torch.set_num_threads(4)
    started = time.monotonic()
    base = mounted_submission(V11_SHA)
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    masks = images >= 128
    signature, energy = contour_signatures(images)
    training = grouped_blocks(root, catalog, MOUSE, device, len(images))
    entries = sorted((row for row in catalog["blocks"] if row["mouse"] == MOUSE and row["split"] == "test"
                      and row["task"] == TASK), key=lambda row: row["session"])
    if (tuple(training) != SESSIONS or tuple(row["session"] for row in entries) != SESSIONS
            or any(values.shape[:2] != (len(images), 4) for values in training.values())):
        raise ValueError("Expected four matched mice1 sessions with four labeled repetitions per glyph.")
    models, fit = fit_models(training, images, signature, energy, MOUSE, SETTINGS)
    del training
    blocks, target_ids = predict(models, entries, root, catalog, images, masks)
    stored = stored_predictions(base, target_ids)
    rendered = (masks.to(torch.uint8) * 255).cpu().numpy()
    patch, changed, sessions = changes_from_predictions(blocks, stored, rendered)

    working = Path("/kaggle/working")
    candidate = working / "kaggle_submission.zip"
    staged = candidate.with_suffix(".tmp")
    try:
        rows, updated = replace_submission_rows(base, staged, patch)
        if updated != len(patch) or verify_archive(base, staged, patch) != updated:
            raise ValueError("The full verification must match the number of intended changes.")
        staged.replace(candidate)
    finally:
        staged.unlink(missing_ok=True)

    changed_path = working / "changed_task3_mice1_ids.json"
    changed_path.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
    manifest = {"base_sha256": V11_SHA, "submission_sha256": sha256(candidate), "target": "mice1 task3",
                "rows": rows, "target_rows": len(target_ids), "changed_predictions": updated,
                "other_rows_verified_unchanged": rows - len(target_ids), "all_unchanged_verified": rows - updated,
                "changed_ids_file": changed_path.name, "changed_ids_sha256": sha256(changed_path),
                "recreated_matching_v11_pixels": sum(detail["v11_recreated_pixels_match"] for detail in sessions.values()),
                "recreated_matching_v11_bytes": sum(detail["v11_recreated_png_bytes_match"] for detail in sessions.values()),
                "sessions": sessions, "fit": fit, "fit_parameters": vars(SETTINGS), "training_repeats": 4,
                "classifier_weight": CLASSIFIER_WEIGHT, "v11_shortlist": 7, "device": str(device),
                "sha_verified_input": True, "row_order_verified": True, "png_verified": "64x64 grayscale with CRC check",
                "seconds": time.monotonic() - started}
    (working / "source_candidates.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
