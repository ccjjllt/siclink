"""Package the held mice2 Task 3 independent utility policy over verified v17."""

from hashlib import sha256 as hash_bytes
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

from glyph_contour_retrieval import fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_encoding import fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model
from glyph_mbr import contour_signatures, glyph_posterior
from kaggle_baseline import TASK_SIZES
from kaggle_merge_candidate_outputs import mounted_submission, replace_submission_rows, sha256
from kaggle_posterior_utility_validation import v11_logits
from kaggle_task3_density_candidate import stored_predictions, verify_archive
from kaggle_task3_independent_utility_candidate import changed_images
from kaggle_task3_part_discovery import SESSIONS
from kaggle_task3_rankband_ocf_validation import NeuralRows, stage
from kaggle_task3_relational_consensus import ACTUAL, INDEPENDENT, UTILITY_WEIGHT, choose_images


BASE_SHA = "61dc0a81e8ff1f3f5c323f516818947c24886f7a6708ad71b84b9ee38aa52ec5"
BASE_SOURCE = "https://www.kaggle.com/code/ccjjllt/siclink-candidate-v17-task3-utility/versions/1"
SUPPORT = "https://www.kaggle.com/code/ccjjllt/siclink-task3-mice2-full-balance-replication/versions/2"
REPORT_NAME = "task3_mice2_full_balance_replication.json"
REPORT_SHA = "a41fbde8e19b1109e7c118dab383e93dcb3c53c37bbde6e40cb54df9df689b41"
TABLE_NAME = "task3_mice2_full_balance_utility.npy"
TABLE_SHA = "c24896c88a11d6408e699ff05769df101035659203de4a339db7e47fa86c9f1f"
GALLERY_SHA = "7132bae68dc9e854bb785d3bba60d24661f64b647115a7be3441f9dc6e2a498a"
MOUSE, TASK = "mice2", "task3"
GALLERY_SIZE = 2748
SEED, ENCODER_EPOCHS, SHORTLIST = 211, 18, 7
INPUT, WORKING = Path("/kaggle/input"), Path("/kaggle/working")


def validated_utility(gallery, device, input_root=INPUT):
    report_path = next((path for path in input_root.rglob(REPORT_NAME) if sha256(path) == REPORT_SHA), None)
    if report_path is None:
        raise FileNotFoundError("Mount the exact finished mice2 holdout replication report.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = report["visual_fit"]["utility_table"]
    if (report["stage"] != "complete" or report["protocol"]["mouse"] != MOUSE
            or report["protocol"]["formal_test_accessed"] or report["protocol"]["same_candidate_budget"] != SHORTLIST
            or report["protocol"]["fixed_exact_utility_weight"] != UTILITY_WEIGHT
            or metadata["file"] != TABLE_NAME or metadata["sha256"] != TABLE_SHA
            or metadata["ordered_gallery_sha256"] != GALLERY_SHA or set(report["sessions"]) != set(SESSIONS)):
        raise ValueError("Expected the completed matching seven-action independent utility validation.")
    source = report_path.with_name(TABLE_NAME)
    if sha256(source) != TABLE_SHA:
        raise ValueError("The mounted utility table must match the verified holdout output.")
    matrix = np.load(source, allow_pickle=False)
    glyphs = gallery.contiguous().numpy()
    if (matrix.shape != (GALLERY_SIZE, GALLERY_SIZE) or matrix.dtype != np.float32 or not np.isfinite(matrix).all()
            or glyphs.shape != (GALLERY_SIZE, 64, 64) or glyphs.dtype != np.uint8
            or hash_bytes(glyphs.tobytes()).hexdigest() != GALLERY_SHA):
        raise ValueError("Utility rows and columns must align with the exact ordered official raw glyph gallery.")

    temperatures = {}
    for day in SESSIONS:
        values = report["sessions"][day]
        peers = [peer for peer in SESSIONS if peer != day]
        fit = values["temperature"]
        if (values["source_days_for_temperature"] != peers or fit["n"] != 576
                or not isfinite(fit["temperature"]) or fit["temperature"] <= 0):
            raise ValueError(f"Temperature for {day} must come from the verified three distinct source recordings.")
        temperatures[day] = {**fit, "source_days": peers}
    return torch.from_numpy(matrix).to(device), temperatures, report["summary_ocf"][INDEPENDENT]


@torch.no_grad()
def predict(model, entries, root, catalog, gallery, utility, temperatures, started):
    assignments, scope = {}, set()
    for entry in entries:
        block = read_trials(root, catalog, entry, gallery.device, load_labels=False)
        day, ids = block.session, tuple(block.ids)
        if (block.counts.shape[:2] != (GALLERY_SIZE, 1) or len(ids) != GALLERY_SIZE or len(set(ids)) != GALLERY_SIZE
                or scope.intersection(ids) or any(not key.startswith("test_task3_pic") for key in ids)):
            raise ValueError("Expected distinct official mice2 Task 3 groups of one measured trial each.")
        scope.update(ids)
        # Reproduce the v14 gain updater's full-recording score shape. The held recipe only reranks its seven leaders.
        logits = v11_logits(model, block.counts, day, gallery)
        anchor = logits.argmax(dim=1, keepdim=True)
        following = logits.clone().scatter_(1, anchor, -torch.inf).topk(SHORTLIST - 1, dim=1).indices
        panel = NeuralRows(logits, logits, torch.cat((anchor, following), dim=1))
        temperature = temperatures[day]["temperature"]
        choice = choose_images(panel, {INDEPENDENT: glyph_posterior(logits, temperature, balance=False)}, utility)
        assignments[day] = (ids, choice[ACTUAL].tolist(), choice[INDEPENDENT].tolist())
        stage("formal_spikes_scored", started, session=day, queries=len(ids), temperature=temperature,
              changed_atlas_choices=int((choice[ACTUAL] != choice[INDEPENDENT]).sum()))
    if len(scope) != TASK_SIZES[TASK] // 2:
        raise ValueError("The patch scope must contain all four complete mice2 Task 3 recordings.")
    return assignments, scope


def main():
    if not INPUT.is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Use a private Kaggle T4 with the verified v17 and source utility outputs mounted.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    base = mounted_submission(BASE_SHA)
    root = locate_data()
    catalog = load_catalog(root)
    gallery_cpu = load_images(root, torch.device("cpu"))
    utility, temperatures, held = validated_utility(gallery_cpu, device)
    gallery = gallery_cpu.to(device)
    del gallery_cpu
    stage("sources_verified", started, report_sha256=REPORT_SHA, utility_sha256=TABLE_SHA, gallery_sha256=GALLERY_SHA)
    training = grouped_blocks(root, catalog, MOUSE, device, GALLERY_SIZE)
    entries = sorted((entry for entry in catalog["blocks"] if entry["mouse"] == MOUSE
                      and entry["split"] == "test" and entry["task"] == TASK), key=lambda entry: entry["session"])
    if (tuple(training) != SESSIONS or tuple(entry["session"] for entry in entries) != SESSIONS
            or any(block.shape[:2] != (GALLERY_SIZE, 4) for block in training.values())):
        raise ValueError("Expected the complete labeled gallery and all four mice2 recordings.")
    contour = fit_contour_retrieval(training, *contour_signatures(gallery), task=TASK, seed=SEED)
    encoder = fit_glyph_encoder(training, gallery, epochs=ENCODER_EPOCHS, seed=SEED)
    gain = fit_gain_model(training, "timed", temporal=True)
    del training
    stage("v14_model_ready", started, gain_per_window=gain.alpha.tolist())

    assignments, scope = predict((encoder, contour, gain), entries, root, catalog, gallery, utility, temperatures, started)
    rendered = ((gallery >= 128).to(torch.uint8) * 255).cpu().numpy()
    patches, changed_ids, sessions = changed_images(assignments, stored_predictions(base, scope), rendered)

    with TemporaryDirectory(prefix=".mice2-utility-", dir=WORKING) as folder:
        temporary = Path(folder)
        archive = temporary / "kaggle_submission.zip"
        rows, count = replace_submission_rows(base, archive, patches)
        if (rows != 30_200 or count != len(patches) or not count or patches.keys() - scope
                or verify_archive(base, archive, patches) != count):
            raise ValueError("The candidate must retain every unchanged CSV row and contain only protected mice2 changes.")
        changes = temporary / "changed_task3_mice2_ids.json"
        changes.write_text(json.dumps(changed_ids, indent=2) + "\n", encoding="utf-8")
        manifest = {"stage": "complete", "candidate": "v17b-mice2-utility", "target": "mice2 task3",
                    "base_sha256": BASE_SHA, "base_source": BASE_SOURCE, "submission_sha256": sha256(archive),
                    "validation_notebook": SUPPORT, "validation_report_sha256": REPORT_SHA,
                    "exact_utility_table_sha256": TABLE_SHA, "ordered_raw_gallery_sha256": GALLERY_SHA,
                    "validation_held_independent_utility": held, "rows": rows, "target_rows": len(scope),
                    "changed_predictions": count, "untargeted_rows_byte_exact": rows - len(scope),
                    "unchanged_rows_byte_exact": rows - count, "changed_ids_file": changes.name,
                    "changed_ids_sha256": sha256(changes), "sessions": sessions,
                    "recreated_v14_pixel_exact": sum(day["v14_recreated_pixel_exact"] for day in sessions.values()),
                    "source_fit_repeats": 4, "formal_query_repeats": 1, "formal_labels_loaded": False,
                    "same_v14_candidate_budget": SHORTLIST, "utility_weight": UTILITY_WEIGHT,
                    "posterior": "rowwise_temperature_softmax", "temperature_by_session": temperatures,
                    "seed": SEED, "encoder_epochs": ENCODER_EPOCHS, "contour_weight": 0.5, "temporal_gain_weight": 0.5,
                    "gain_per_window": gain.alpha.tolist(), "gain_windows_ms": [[25 * a, 25 * b] for a, b in gain.windows],
                    "only_pixel_exact_recreated_v14_rows_changed": True,
                    "candidate_archive_verified": "exact CSV order and untouched bytes, ZIP CRC, 64x64 grayscale PNG",
                    "seconds": time.monotonic() - started}
        report = temporary / "source_candidates.json"
        report.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for path in (archive, changes, report):
            path.replace(WORKING / path.name)
    stage("complete", started, archive_sha256=manifest["submission_sha256"], changed_predictions=count,
          reproduced_pixels=manifest["recreated_v14_pixel_exact"])


if __name__ == "__main__":
    main()
