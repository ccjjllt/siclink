"""Package the fixed mice2 Task 2 CDF policy in replay-protected permutation cycles."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import time

import scipy
import torch

from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data, read_trials
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, combine_logits, fit_retrieval, retrieval_score_components
from kaggle_baseline import TASK_SIZES
from kaggle_merge_candidate_outputs import replace_submission_rows, sha256
from kaggle_task2_peer_mouse import logit_scale
from kaggle_task2_rank_marginals import ADAPTED, BLEND, EPOCHS, MODEL_SEED, MOUSE, marginal_logits
from kaggle_task2_rank_marginals_recheck import PROVENANCE, fit_marginal_templates
from kaggle_task3_density_candidate import open_png, stored_predictions, verify_archive
from kaggle_v8_metric_validation import SESSIONS


BASE_SHA = PROVENANCE["v20_sha256"]
HELD_REPORT_SHA = "163fc0a63c28d71ddd9a6e6d6e39740b62b1bb5d6c574479beb80b4c4c66bbd4"
BASE_FILENAMES = ("v20-base.zip.bin", "kaggle_submission.zip")
TASK, DAY = "task2", "session5"
WORKING = Path("/kaggle/working")
REPORT = "source_candidates.json"
ARCHIVED_CLASSIFIER_LOSS = {1: 7.0394, 5: 2.6052, 10: 0.3807, 12: 0.2500}


def verified_base(path):
    if path is not None:
        source = path.resolve(strict=True)
        if sha256(source) != BASE_SHA:
            raise ValueError("The supplied ZIP must match immutable v20.")
        return source
    inputs = Path("/kaggle/input")
    for filename in BASE_FILENAMES:
        for source in inputs.rglob(filename):
            if source.is_file() and sha256(source) == BASE_SHA:
                return source
    raise FileNotFoundError("Mount the exact verified private v20 submission ZIP.")


def publish(report, output, stage, started):
    report.update(stage=stage, seconds=time.monotonic() - started)
    pending = output / ".source_candidates.tmp"
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(output / REPORT)
    print(json.dumps({"stage": stage, "seconds": round(report["seconds"], 3)}), flush=True)


def png_pixels(ids, stored):
    pixels = []
    for sample in ids:
        with open_png(stored[sample]) as image:
            pixels.append(image.tobytes())
    return pixels


def recover_base(gallery, pixels, stock):
    """Find a bijection to the exact official masks, preferring replayed labels in equivalent groups."""
    if Counter(gallery) != Counter(pixels):
        raise ValueError("The archived mice2 block must contain the exact official gallery pixel multiset.")
    glyph_groups, row_groups = defaultdict(list), defaultdict(list)
    for glyph, mask in enumerate(gallery):
        glyph_groups[mask].append(glyph)
    for row, mask in enumerate(pixels):
        row_groups[mask].append(row)
    replay = [gallery[glyph] == mask for glyph, mask in zip(stock, pixels, strict=True)]
    recovered, ambiguous = [-1] * len(pixels), [False] * len(pixels)
    for mask, glyphs in glyph_groups.items():
        remaining, unresolved = set(glyphs), []
        for row in row_groups[mask]:
            ambiguous[row] = len(glyphs) > 1
            if replay[row]:
                recovered[row] = stock[row]
                remaining.remove(stock[row])
            else:
                unresolved.append(row)
        for row, glyph in zip(unresolved, sorted(remaining), strict=True):
            recovered[row] = glyph
    if sorted(recovered) != list(range(len(gallery))):
        raise ValueError("The archived rows must identify one of every official gallery glyph.")
    return recovered, replay, ambiguous


def protect_cycles(base, proposed, replay, ambiguous):
    if sorted(proposed) != list(range(len(base))):
        raise ValueError("The full candidate assignment must use each gallery class once.")
    owner = {glyph: row for row, glyph in enumerate(base)}
    next_row = [owner[glyph] for glyph in proposed]
    visited = set()
    accepted, rejected_replay, rejected_ambiguous, all_lengths = [], Counter(), Counter(), Counter()
    final = base.copy()
    full_replay = all(replay)
    for start in range(len(base)):
        cycle = []
        row = start
        while row not in visited:
            visited.add(row)
            cycle.append(row)
            row = next_row[row]
        if len(cycle) <= 1:
            continue
        if row != start:
            raise ValueError("The full proposal must decompose into disjoint complete permutation cycles.")
        all_lengths[len(cycle)] += 1
        if not all(replay[row] for row in cycle):
            rejected_replay[len(cycle)] += 1
        elif not full_replay and any(ambiguous[row] for row in cycle):
            rejected_ambiguous[len(cycle)] += 1
        else:
            accepted.append(cycle)
            for row in cycle:
                final[row] = proposed[row]
    if sorted(final) != list(range(len(base))) or (full_replay and final != proposed):
        raise ValueError("Apply only complete cycles and retain the full proposed assignment when stock replay is exact.")
    return final, accepted, {"proposed_length_counts": dict(all_lengths),
                             "accepted_length_counts": dict(Counter(map(len, accepted))),
                             "blocked_by_replay_length_counts": dict(rejected_replay),
                             "blocked_by_ambiguity_length_counts": dict(rejected_ambiguous),
                             "proposed_nontrivial_cycles": sum(all_lengths.values()), "accepted_cycles": len(accepted),
                             "proposed_cycle_rows": sum(length * count for length, count in all_lengths.items()),
                             "accepted_cycle_rows": sum(map(len, accepted))}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Device for replaying the original v9 hybrid")
    parser.add_argument("--data-root", type=Path, help="Extracted competition directory containing catalog.json")
    parser.add_argument("--base", type=Path, help="Exact verified v20 submission ZIP")
    parser.add_argument("--output-dir", type=Path, default=WORKING)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    cpu, device = torch.device("cpu"), torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("Choose CPU or an available remote CUDA device for the v9 replay.")
    base = verified_base(args.base)
    output = args.output_dir.resolve()
    archive = output / "kaggle_submission.zip"
    if archive.resolve() == base.resolve():
        raise ValueError("Write outputs separately from the immutable base.")
    output.mkdir(parents=True, exist_ok=True)
    root = args.data_root.resolve(strict=True) if args.data_root else locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    training = grouped_blocks(root, catalog, MOUSE, device, len(images))
    entries = [entry for entry in catalog["blocks"] if entry["split"] == "test" and entry["mouse"] == MOUSE and entry["task"] == TASK]
    size = TASK_SIZES[TASK] // 2
    if (len(images) != size or tuple(training) != SESSIONS or any(part.shape[:2] != (size, 4) for part in training.values())
            or len(entries) != 1 or entries[0]["session"] != DAY):
        raise ValueError("Expected four complete labeled days and the one complete mice2 Task 2 session5.")
    query = read_trials(root, catalog, entries[0], device, load_labels=False)
    ids = tuple(map(str, query.ids))
    scope = set(ids)
    if query.counts.shape[:2] != (size, 4) or len(scope) != size or any(not sample.startswith("test_task2_pic") for sample in ids):
        raise ValueError("Expected a unique ID per four-repeat standard Task 2 query.")
    stored = stored_predictions(base, scope)
    bank = ((images >= 128).to(torch.uint8) * 255).cpu().numpy()
    gallery = [mask.tobytes() for mask in bank]
    pixels = png_pixels(ids, stored)
    if Counter(gallery) != Counter(pixels):
        raise ValueError("Target mice2 v20 rows must contain the exact official gallery pixel multiset.")
    report = {"candidate": "v21-mice2-task2-target-cdf", "base_sha256": BASE_SHA,
              "baseline_archive_provenance": PROVENANCE, "held_validation_report_sha256": HELD_REPORT_SHA,
              "target": "mice2 task2", "target_session": DAY, "target_rows": size, "input_labeled_days": SESSIONS,
              "source_fit_repeats": 4, "formal_query_repeats": 4, "formal_labels_loaded": False,
              "model_seed": MODEL_SEED, "v9_epochs": EPOCHS, "target_empirical_cdf_weight": BLEND,
              "v9_replay_device": str(device), "cdf_and_blend_device": str(cpu),
              "archived_v5_classifier_loss_at_epoch": ARCHIVED_CLASSIFIER_LOSS,
              "replay_runtime": {"torch_version": str(torch.__version__), "scipy_version": scipy.__version__,
                                 "cuda_version": torch.version.cuda, "float32_matmul_precision": torch.get_float32_matmul_precision(),
                                 "cuda_device_count": torch.cuda.device_count()},
              "input_gallery_pixels_verified": True, "stock_replay_guard": "exact PNG pixels on every changed permutation cycle"}
    if device.type == "cuda":
        report["replay_runtime"].update(cuda_device_name=torch.cuda.get_device_name(device),
                                        cuda_capability=torch.cuda.get_device_capability(device),
                                        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    publish(report, output, "inputs_verified", started)

    torch.manual_seed(MODEL_SEED)
    with torch.enable_grad():
        retrieval = fit_retrieval(as_trial_blocks(training), len(images), images=images, device=device,
                                  seed=MODEL_SEED, epochs=EPOCHS)
    hierarchy = fit_hierarchical_gallery(training, view="total")
    publish(report, output, "four_day_fit_ready", started)
    validated = combine_logits(retrieval_score_components(retrieval, query.counts, DAY, task=TASK), TASK)
    empirical = hierarchy.logits(query.counts, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    hybrid = blend_hierarchical_logits(validated, empirical)
    stock = assign_glyphs(hybrid).tolist()
    hybrid = hybrid.cpu()
    del validated, empirical, retrieval, hierarchy
    if sorted(stock) != list(range(size)):
        raise ValueError("The stock assignment must use each gallery class once.")
    original, replay, ambiguous = recover_base(gallery, pixels, stock)
    report.update(raw_v9_pixel_replay_count=sum(replay), raw_v9_pixel_replay_rate=sum(replay) / size)
    publish(report, output, "stock_replay_measured", started)

    templates = fit_marginal_templates({day: counts.cpu() for day, counts in training.items()})
    del training
    target_cdf = marginal_logits(query.counts.cpu(), templates)[ADAPTED]
    proposed = assign_glyphs(logit_scale(hybrid).lerp(target_cdf, BLEND)).tolist()
    final, accepted, cycle_report = protect_cycles(original, proposed, replay, ambiguous)
    changed_indices = [row for row, glyph in enumerate(final) if gallery[glyph] != pixels[row]]
    if any(not replay[row] for row in changed_indices):
        raise ValueError("Every changed PNG must replay the archived v20 stock pixels.")
    owner = {glyph: row for row, glyph in enumerate(original)}
    updates = {ids[row]: stored[ids[owner[final[row]]]] for row in changed_indices}
    report.update(ambiguous_base_pixel_rows=sum(ambiguous), observed_gallery_distinct_masks=len(set(gallery)),
                  raw_cdf_vs_v9_label_changes=sum(a != b for a, b in zip(proposed, stock, strict=True)),
                  raw_cdf_vs_v20_pixel_changes=sum(gallery[glyph] != mask for glyph, mask in zip(proposed, pixels, strict=True)),
                  cycles=cycle_report, candidate_gallery_classes=len(set(final)))
    publish(report, output, "protected_cycles_selected", started)

    with TemporaryDirectory(prefix=".mice2-task2-cdf-", dir=output) as folder:
        stage = Path(folder)
        staged_archive = stage / archive.name
        if updates:
            rows, changed = replace_submission_rows(base, staged_archive, updates)
        else:
            copyfile(base, staged_archive)
            rows, changed = sum(TASK_SIZES.values()), 0
        if (rows != 30_200 or changed != len(changed_indices) or updates.keys() - scope
                or verify_archive(base, staged_archive, updates) != changed):
            raise ValueError("Expected exactly the listed mice2 Task 2 changes and byte-identical preservation of all other rows.")
        written = png_pixels(ids, stored_predictions(staged_archive, scope))
        if written != [gallery[glyph] for glyph in final] or Counter(gallery) != Counter(written):
            raise ValueError("The staged archive must still represent the one-of-each official gallery.")
        ids_path = stage / "changed_task2_mice2_ids.json"
        changes = {"session": DAY, "changed": [ids[row] for row in changed_indices],
                   "accepted_cycles": [[ids[row] for row in cycle] for cycle in accepted]}
        ids_path.write_text(json.dumps(changes, indent=2) + "\n", encoding="utf-8")
        report.update(stage="complete", seconds=time.monotonic() - started, submission_sha256=sha256(staged_archive),
                      rows=rows, changed_predictions=changed, unchanged_rows_verified_byte_exact=rows - changed,
                      outside_scope_verified_byte_exact=rows - size, changed_ids_file=ids_path.name,
                      changed_ids_sha256=sha256(ids_path), official_gallery_reverified=True,
                      archive_verification="official IDs and ordering, exact untouched CSV bytes, ZIP CRC, 64x64 grayscale PNG")
        manifest = stage / REPORT
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for file in (staged_archive, ids_path, manifest):
            file.replace(output / file.name)
    print(json.dumps({"stage": "complete", "seconds": report["seconds"], "submission_sha256": report["submission_sha256"],
                      "changed_predictions": changed, "raw_v9_pixel_replay": sum(replay), "accepted_cycles": len(accepted)}), flush=True)


if __name__ == "__main__":
    main()
