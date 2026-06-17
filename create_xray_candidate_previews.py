import argparse
import csv
import hashlib
import io
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import SimpleITK as sitk
from PIL import Image

from create_visual_qc_previews import (
    IMAGE_SUFFIXES,
    extract_virtual_files,
    image_to_uint8,
    parent_virtual_path,
    split_virtual_archive_path,
)


LATERAL_RE = re.compile(r"\b(lat|lateral|ll|rl|yan|sagittal|profile|obl|oblique)\b", re.IGNORECASE)
FRONTAL_RE = re.compile(r"\b(ap|pa|frontal|coronal)\b", re.IGNORECASE)
FAST_SUFFIXES = {".dcm", ".png", ".jpg", ".jpeg"}
FALLBACK_SUFFIXES = {".nrrd", ".nhdr"}


class VirtualFileReader:
    def __init__(self):
        self.zip_cache = {}
        self.nested_bytes_cache = {}

    def close(self):
        for zf in self.zip_cache.values():
            try:
                zf.close()
            except Exception:
                pass
        self.zip_cache.clear()
        self.nested_bytes_cache.clear()

    def zip_for_path(self, archive):
        key = str(Path(archive).resolve())
        if key not in self.zip_cache:
            self.zip_cache[key] = zipfile.ZipFile(key)
        return self.zip_cache[key]

    def read(self, vpath):
        archive, inner = split_virtual_archive_path(vpath)
        if archive is None:
            return Path(vpath).read_bytes()
        if archive.suffix.lower() == ".7z":
            raise RuntimeError("skip_7z_direct_candidate")

        zf = self.zip_for_path(archive)
        parts = list(PurePosixPath(inner).parts)
        prefix = []
        for idx, part in enumerate(parts):
            prefix.append(part)
            if part.lower().endswith(".zip"):
                nested_name = "/".join(prefix)
                cache_key = (str(Path(archive).resolve()), nested_name)
                if cache_key not in self.nested_bytes_cache:
                    self.nested_bytes_cache[cache_key] = zf.read(nested_name)
                nested_bytes = self.nested_bytes_cache[cache_key]
                rest = "/".join(parts[idx + 1 :])
                with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested_zf:
                    return nested_zf.read(rest)
        return zf.read(inner)


def safe_token(value, fallback="view"):
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = value.strip("_")
    return value[:32] or fallback


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def label_map(final_labels_csv):
    rows = list(csv.DictReader(open(final_labels_csv, encoding="utf-8")))
    return {row["tc"]: row["label"] for row in rows}


def candidate_key(record):
    suffix = record["suffix"]
    vpath = record["virtual_path"]
    if suffix == ".nhdr":
        return "nhdr", parent_virtual_path(vpath)
    return suffix.lstrip(".") or "file", vpath


def build_candidates(records, include_nrrd=False):
    all_grouped = defaultdict(list)
    for row in records:
        if row.get("modality") != "xray":
            continue
        if not row.get("tc"):
            continue
        suffix = row.get("suffix")
        if suffix not in IMAGE_SUFFIXES or suffix == ".seg.nrrd":
            continue
        kind, key = candidate_key(row)
        all_grouped[(row["tc"], kind, key)].append(row)

    by_tc = defaultdict(list)
    for key, rows in all_grouped.items():
        by_tc[key[0]].append((key, rows))

    selected = {}
    for tc, items in by_tc.items():
        fast = [(key, rows) for key, rows in items if rows[0]["suffix"] in FAST_SUFFIXES]
        fallback = [(key, rows) for key, rows in items if rows[0]["suffix"] in FALLBACK_SUFFIXES]
        chosen = fast
        if include_nrrd or not chosen:
            chosen = fast + fallback
        for key, rows in chosen:
            selected[key] = rows
    return selected


def read_metadata(reader):
    meta = {}
    tags = {
        "0008|0060": "modality",
        "0008|1030": "study_description",
        "0008|103e": "series_description",
        "0018|0015": "body_part",
        "0018|1030": "protocol_name",
        "0018|5101": "view_position",
        "0020|0060": "laterality",
        "0020|0062": "image_laterality",
    }
    for tag, name in tags.items():
        if reader.HasMetaDataKey(tag):
            meta[name] = reader.GetMetaData(tag).strip()
        else:
            meta[name] = ""
    return meta


def squeeze_2d(arr):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(f"skip_timeseries_or_volume_shape_{tuple(arr.shape)}")
    return arr


def read_sitk_2d(path, is_dicom=False):
    metadata = {}
    if is_dicom:
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        metadata = read_metadata(reader)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(path))

    arr = sitk.GetArrayFromImage(image)
    if image.GetNumberOfComponentsPerPixel() > 1:
        arr = np.asarray(arr)[..., 0]
    arr = squeeze_2d(arr)
    return image_to_uint8(arr), metadata


def read_candidate(kind, records, reader):
    record = records[0]
    suffix = record["suffix"]
    archive, _inner = split_virtual_archive_path(record["virtual_path"])
    if archive and archive.suffix.lower() == ".7z":
        raise RuntimeError("skip_7z_direct_candidate")

    with tempfile.TemporaryDirectory() as tmp:
        if kind == "nhdr":
            parent = parent_virtual_path(record["virtual_path"])
            siblings = [r for r in records if parent_virtual_path(r["virtual_path"]) == parent]
            paths = extract_virtual_files(siblings, tmp, suffix_filter={".nhdr", ".raw.gz"})
            nhdrs = [p for p in paths if p.suffix.lower() == ".nhdr"]
            if not nhdrs:
                raise RuntimeError("no_nhdr_extracted")
            arr, metadata = read_sitk_2d(nhdrs[0], is_dicom=False)
            return arr, metadata

        raw = reader.read(record["virtual_path"])
        ext = suffix if suffix.startswith(".") else ".bin"
        path = Path(tmp) / f"candidate{ext}"
        path.write_bytes(raw)

        if suffix in {".png", ".jpg", ".jpeg"}:
            image = Image.open(path).convert("L")
            return image_to_uint8(np.asarray(image)), {}

        return read_sitk_2d(path, is_dicom=(suffix == ".dcm"))


def view_category(metadata, virtual_path):
    text = " ".join(
        [
            metadata.get("view_position", ""),
            metadata.get("series_description", ""),
            metadata.get("protocol_name", ""),
            metadata.get("study_description", ""),
            virtual_path,
        ]
    )
    if LATERAL_RE.search(text):
        return "lateral"
    if FRONTAL_RE.search(text):
        return "frontal"
    return "unknown"


def pixel_hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def save_png(arr, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).convert("L").save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Create per-TC X-ray candidate PNGs for manual selection.")
    parser.add_argument("--stage_dir", default=r"C:\np_stage\xray_anchor_with_healthy_20260614_v1")
    parser.add_argument("--out_name", default="xray_review_candidates")
    parser.add_argument("--keep_lateral", action="store_true")
    parser.add_argument("--include_nrrd", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir)
    out_dir = stage_dir / "visual_qc" / args.out_name
    records = list(csv.DictReader(open(stage_dir / "all_tc_source_records.csv", encoding="utf-8")))
    labels = label_map(stage_dir / "final_dataset_audit" / "final_xray_anchor_labels.csv")
    candidates = build_candidates(records, include_nrrd=args.include_nrrd)

    items = sorted(candidates.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
    if args.limit:
        items = items[: args.limit]

    rows = []
    per_tc_written = defaultdict(int)
    per_tc_seen_hash = defaultdict(dict)
    status_counts = Counter()
    reader = VirtualFileReader()

    for (tc, kind, _key), candidate_records in items:
        label = labels.get(tc, "unknown")
        source = candidate_records[0]["virtual_path"]
        metadata = {}
        out_path = ""
        duplicate_of = ""
        status = "ok"
        reason = ""
        h = ""
        shape = ""
        category = "unknown"

        try:
            arr, metadata = read_candidate(kind, candidate_records, reader)
            shape = f"{arr.shape[0]}x{arr.shape[1]}"
            category = view_category(metadata, source)

            if category == "lateral" and not args.keep_lateral:
                status = "excluded_lateral"
                reason = "metadata_or_path_indicates_lateral"
            else:
                h = pixel_hash(arr)
                if h in per_tc_seen_hash[tc]:
                    status = "duplicate"
                    duplicate_of = per_tc_seen_hash[tc][h]
                    reason = "same_pixel_hash_within_tc"
                else:
                    per_tc_written[tc] += 1
                    idx = per_tc_written[tc]
                    view_token = safe_token(metadata.get("view_position") or category)
                    filename = f"candidate_{idx:03d}_{kind}_{view_token}.png"
                    out_path_obj = out_dir / tc / filename
                    save_png(arr, out_path_obj)
                    out_path = str(out_path_obj)
                    per_tc_seen_hash[tc][h] = filename
        except Exception as exc:
            status = "failed"
            reason = str(exc)

        status_counts[status] += 1
        rows.append(
            {
                "tc": tc,
                "label": label,
                "status": status,
                "reason": reason,
                "output_path": out_path,
                "duplicate_of": duplicate_of,
                "kind": kind,
                "view_category": category,
                "view_position": metadata.get("view_position", ""),
                "series_description": metadata.get("series_description", ""),
                "protocol_name": metadata.get("protocol_name", ""),
                "study_description": metadata.get("study_description", ""),
                "body_part": metadata.get("body_part", ""),
                "pixel_hash": h,
                "pixel_shape": shape,
                "record_count": len(candidate_records),
                "first_virtual_path": source,
            }
        )

    reader.close()

    write_csv(out_dir / "xray_review_candidate_manifest.csv", rows)
    summary = [
        {"metric": "candidate_groups", "value": len(rows)},
        {"metric": "tc_with_candidates", "value": len({row["tc"] for row in rows})},
        {"metric": "written_png", "value": sum(1 for row in rows if row["status"] == "ok")},
        {"metric": "excluded_lateral", "value": status_counts["excluded_lateral"]},
        {"metric": "duplicates", "value": status_counts["duplicate"]},
        {"metric": "failed", "value": status_counts["failed"]},
        {"metric": "output_dir", "value": str(out_dir)},
    ]
    write_csv(out_dir / "xray_review_candidate_summary.csv", summary)

    print("X-ray candidate export complete")
    for row in summary:
        print(f"{row['metric']}: {row['value']}")


if __name__ == "__main__":
    main()
