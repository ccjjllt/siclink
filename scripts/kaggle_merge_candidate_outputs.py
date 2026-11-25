"""Package a task update from two already generated Kaggle candidates."""

import argparse
import base64
import csv
import hashlib
import io
from itertools import zip_longest
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

from kaggle_baseline import TASK_SIZES


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mounted_submission(digest):
    packages = Path("/kaggle/input").rglob("kaggle_submission.zip")
    path = next((path for path in packages if sha256(path) == digest), None)
    if path is None:
        raise FileNotFoundError("Mount the exact base Kaggle submission as a notebook input.")
    return path


def encode_grayscale_png(pixels):
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def replace_submission_rows(base, output, predictions):
    """Copy original prediction strings except for the supplied sample IDs."""
    row_count = changed = 0
    pending = predictions.copy()
    with ZipFile(base) as original, ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        with (io.TextIOWrapper(original.open("submission.csv"), encoding="utf-8", newline="") as raw,
              io.TextIOWrapper(archive.open("submission.csv", "w"), encoding="utf-8", newline="") as stream):
            rows = csv.reader(raw)
            writer = csv.writer(stream)
            if next(rows) != ["sample_id", "prediction"]:
                raise ValueError("Expected the official Kaggle submission header.")
            writer.writerow(("sample_id", "prediction"))
            for row_count, (sample_id, prediction) in enumerate(rows, 1):
                update = pending.pop(sample_id, prediction)
                changed += update != prediction
                writer.writerow((sample_id, update))
    if pending or row_count != sum(TASK_SIZES.values()):
        raise ValueError("Every replacement must appear in the complete base package.")
    return row_count, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--update-sha", required=True)
    parser.add_argument("--task", choices=tuple(TASK_SIZES), required=True)
    args = parser.parse_args()
    candidates = {sha256(path): path for path in Path("/kaggle/input").rglob("kaggle_submission.zip")}
    if args.base_sha not in candidates or args.update_sha not in candidates:
        raise RuntimeError("The mounted notebook output hashes must match both requested candidates.")
    working = Path("/kaggle/working")
    output = working / "kaggle_submission.zip"
    row_ids = (f"test_{task}_pic{index}" for task, count in TASK_SIZES.items() for index in range(count))
    with ZipFile(candidates[args.base_sha]) as base, ZipFile(candidates[args.update_sha]) as update, \
            ZipFile(output, "w", compression=ZIP_DEFLATED) as dest:
        streams = [io.TextIOWrapper(archive.open("submission.csv", mode), encoding="utf-8", newline="")
                   for archive, mode in ((base, "r"), (update, "r"), (dest, "w"))]
        with streams[0], streams[1], streams[2]:
            header = ["sample_id", "prediction"]
            originals, updates = csv.reader(streams[0]), csv.reader(streams[1])
            if next(originals) != header or next(updates) != header:
                raise ValueError("Each input must have the official CSV header.")
            writer = csv.writer(streams[2])
            writer.writerow(header)
            for expected, original, amended in zip_longest(row_ids, originals, updates):
                if expected is None or not original or not amended or expected != original[0] or expected != amended[0]:
                    raise ValueError("Each input must contain the full set of official IDs in order.")
                writer.writerow(amended if expected.startswith(f"test_{args.task}_pic") else original)
    manifest = {"submission_sha256": sha256(output), "rows": sum(TASK_SIZES.values()),
                "base_sha256": args.base_sha, "update_sha256": args.update_sha, "updated_task": args.task}
    (working / "source_candidates.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
