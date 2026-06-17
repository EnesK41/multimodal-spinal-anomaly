import argparse
import csv
import hashlib
import json
import re
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath


DEFAULT_KEEP_SUFFIXES = {
    ".dcm",
    ".nrrd",
    ".seg.nrrd",
    ".nii",
    ".nii.gz",
    ".png",
    ".jpg",
    ".jpeg",
    ".nhdr",
    ".raw.gz",
}

SKIP_SUFFIXES = {".mrml", ".lnk"}
TC_RE = re.compile(r"(?<!\d)(\d{11})(?!\d)")
GENERIC_PARTS = {
    "",
    ".",
    "__macosx",
    "mrg_segmente_edilenler",
    "segmente_edilenler",
    "segmente_rontgen",
    "en_yeni_eklenen_bt_segmentasyon",
    "mrg segmente edilenler-20260613t084706z-3-001",
    "mrg_segmente_edilenler-20260613t084706z-3-001",
    "segmente edilenler-20260613t085953z-3-006",
    "segmente_edilenler-20260613t085953z-3-006",
    "segmente rontgen-20260613t093516z-3-004",
    "segmente röntgen-20260613t093516z-3-004",
    "segmente_rontgen-20260613t093516z-3-004",
}


def normalize_text(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("ı", "i")
    return value


def safe_name(value, max_len=48):
    value = normalize_text(value)
    value = re.sub(r"[^a-z0-9._-]+", "_", value).strip("._-")
    return (value or "item")[:max_len]


def suffix_key(path):
    name = PurePosixPath(str(path).replace("\\", "/")).name.lower()
    if name.endswith(".seg.nrrd"):
        return ".seg.nrrd"
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".raw.gz"):
        return ".raw.gz"
    return Path(name).suffix.lower() or "<no_ext>"


def role_from_suffix(path):
    suffix = suffix_key(path)
    name = PurePosixPath(str(path).replace("\\", "/")).name.lower()
    if suffix == ".dcm":
        return "dicom"
    if suffix == ".seg.nrrd" or "segmentation" in name:
        return "segmentation"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "preview_or_image"
    if suffix in {".nrrd", ".nii", ".nii.gz", ".nhdr", ".raw.gz"}:
        return "volume_or_image"
    return "other"


def modality_from_name(name):
    norm = normalize_text(name)
    if "rontgen" in norm or "xray" in norm or "x-ray" in norm:
        return "xray"
    if "mrg" in norm or re.search(r"\bmr\b", norm):
        return "mr"
    if "segmente edilenler" in norm or "bt" in norm or "ct" in norm:
        return "ct"
    return "unknown"


def path_parts(path):
    return [p for p in PurePosixPath(str(path).replace("\\", "/")).parts if p not in {"/", ""}]


def patient_token_from_path(source_name, inner_path, nested_name=None):
    candidates = []
    if nested_name:
        nested_parts = path_parts(nested_name)
        if nested_parts and nested_parts[-1].lower().endswith(".zip"):
            nested_parts[-1] = nested_parts[-1][:-4]
        candidates.extend(nested_parts)
    candidates.extend(path_parts(inner_path))

    direct_match = TC_RE.search("/".join(candidates))
    if direct_match:
        return direct_match.group(1)

    source_norm = safe_name(Path(source_name).stem, 96)
    cleaned_candidates = []
    for part in candidates:
        part_norm = safe_name(part, 96)
        if part_norm in GENERIC_PARTS:
            continue
        if part_norm == source_norm:
            continue
        if part_norm.startswith("volume_") or part_norm.startswith("segmentation"):
            continue
        if suffix_key(part_norm) != "<no_ext>":
            continue
        cleaned_candidates.append(part_norm)

    for part_norm in cleaned_candidates:
        match = TC_RE.search(part_norm)
        if match:
            return match.group(1)

    for part_norm in cleaned_candidates:
        return part_norm

    joined = "/".join(candidates) or str(inner_path)
    return "unknown_" + hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()[:10]


def case_id_for_token(token, id_mode="case", first_digits=True):
    if id_mode == "tc" and TC_RE.fullmatch(token):
        return token
    digest = hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()[:8]
    digits = re.sub(r"\D", "", token)
    if first_digits and len(digits) >= 4:
        return f"case_{digits[:4]}_{digest}"
    return f"case_{digest}"


def iter_source_zips(source):
    for item in sorted(Path(source).glob("*.zip")):
        if item.is_file():
            yield item


def keep_member(name, keep_mrml=False):
    suffix = suffix_key(name)
    if suffix == ".zip":
        return True
    if suffix in SKIP_SUFFIXES and not (keep_mrml and suffix == ".mrml"):
        return False
    return suffix in DEFAULT_KEEP_SUFFIXES or (keep_mrml and suffix == ".mrml")


def short_filename(original_name, counters):
    suffix = suffix_key(original_name)
    original_base = PurePosixPath(str(original_name).replace("\\", "/")).name
    if suffix in {".nhdr", ".raw.gz"}:
        return original_base
    role = role_from_suffix(original_name)
    counters[(role, suffix)] += 1
    idx = counters[(role, suffix)]
    if suffix == ".seg.nrrd":
        return f"seg_{idx:04d}.seg.nrrd"
    if suffix == ".nii.gz":
        return f"nii_{idx:04d}.nii.gz"
    if suffix == ".raw.gz":
        return f"raw_{idx:04d}.raw.gz"
    if suffix == ".nhdr":
        return f"nhdr_{idx:04d}.nhdr"
    if suffix == "<no_ext>":
        return f"file_{idx:04d}"
    prefix = {
        "dicom": "dcm",
        "segmentation": "seg",
        "preview_or_image": "img",
        "volume_or_image": "vol",
    }.get(role, "file")
    return f"{prefix}_{idx:04d}{suffix}"


def bundle_key_for(inner_name, nested_name=None):
    inner_parent = str(PurePosixPath(str(inner_name).replace("\\", "/")).parent)
    if inner_parent == ".":
        inner_parent = ""
    key = f"{nested_name or ''}|{inner_parent}"
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:8]


def write_stream_to_path(stream, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def copy_zip_member_to_temp(zf, info, temp_dir):
    temp_path = Path(temp_dir) / (hashlib.sha1(info.filename.encode("utf-8")).hexdigest()[:12] + ".zip")
    with zf.open(info) as nested_stream:
        write_stream_to_path(nested_stream, temp_path)
    return temp_path


def process_member(
    *,
    source_zip,
    modality,
    inner_name,
    size,
    nested_name,
    mode,
    stage,
    public_rows,
    private_rows,
    counters_by_case,
    case_tokens,
    id_mode,
    stream_factory=None,
):
    suffix = suffix_key(inner_name)
    if suffix == ".zip":
        return

    token = patient_token_from_path(source_zip.name, inner_name, nested_name)
    case_id = case_id_for_token(token, id_mode=id_mode)
    case_tokens[case_id].add(token)

    role = role_from_suffix(inner_name)
    dest_rel = ""
    if mode == "extract":
        bundle = bundle_key_for(inner_name, nested_name)
        dest_dir = stage / "raw" / modality / case_id / bundle
        dest_name = short_filename(inner_name, counters_by_case[(modality, case_id, bundle)])
        dest = dest_dir / dest_name
        with stream_factory() as stream:
            write_stream_to_path(stream, dest)
        dest_rel = str(dest.relative_to(stage))

    public_rows.append(
        {
            "case_id": case_id,
            "modality": modality,
            "role": role,
            "suffix": suffix,
            "size_mb": round(size / (1024 * 1024), 3),
            "dest_rel": dest_rel,
        }
    )
    private_rows.append(
        {
            "case_id": case_id,
            "modality": modality,
            "source_zip": source_zip.name,
            "nested_zip": nested_name or "",
            "original_path": inner_name,
            "patient_token": token,
            "role": role,
            "suffix": suffix,
            "size_bytes": size,
            "dest_rel": dest_rel,
        }
    )


def process_zip(source_zip, mode, stage, include_mr, keep_mrml, id_mode, public_rows, private_rows, case_tokens):
    modality = modality_from_name(source_zip.name)
    if modality == "mr" and not include_mr:
        return {"zip": source_zip.name, "modality": modality, "status": "skipped_mr"}

    counters_by_case = defaultdict(Counter)
    summary = Counter()
    with tempfile.TemporaryDirectory(dir=stage) as temp_dir, zipfile.ZipFile(source_zip) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            suffix = suffix_key(info.filename)
            summary[suffix] += 1
            if suffix == ".zip":
                nested_path = copy_zip_member_to_temp(zf, info, temp_dir)
                with zipfile.ZipFile(nested_path) as nested_zf:
                    for nested_info in nested_zf.infolist():
                        if nested_info.is_dir():
                            continue
                        nested_suffix = suffix_key(nested_info.filename)
                        summary[nested_suffix] += 1
                        if not keep_member(nested_info.filename, keep_mrml=keep_mrml):
                            continue
                        process_member(
                            source_zip=source_zip,
                            modality=modality,
                            inner_name=nested_info.filename,
                            size=nested_info.file_size,
                            nested_name=info.filename,
                            mode=mode,
                            stage=stage,
                            public_rows=public_rows,
                            private_rows=private_rows,
                            counters_by_case=counters_by_case,
                            case_tokens=case_tokens,
                            id_mode=id_mode,
                            stream_factory=lambda nzf=nested_zf, ni=nested_info: nzf.open(ni),
                        )
                continue

            if not keep_member(info.filename, keep_mrml=keep_mrml):
                continue
            process_member(
                source_zip=source_zip,
                modality=modality,
                inner_name=info.filename,
                size=info.file_size,
                nested_name=None,
                mode=mode,
                stage=stage,
                public_rows=public_rows,
                private_rows=private_rows,
                counters_by_case=counters_by_case,
                case_tokens=case_tokens,
                id_mode=id_mode,
                stream_factory=lambda zf=zf, info=info: zf.open(info),
            )

    return {
        "zip": source_zip.name,
        "modality": modality,
        "status": "ok",
        "extension_counts": dict(summary),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_case_summary(public_rows, case_tokens):
    grouped = defaultdict(Counter)
    sizes = defaultdict(float)
    for row in public_rows:
        key = (row["case_id"], row["modality"])
        grouped[key][row["role"]] += 1
        grouped[key][row["suffix"]] += 1
        sizes[key] += float(row["size_mb"])

    rows = []
    for (case_id, modality), counts in sorted(grouped.items()):
        warnings = []
        if modality == "ct" and counts["dicom"] and counts["dicom"] < 200:
            warnings.append("low_ct_dicom_count")
        if counts["segmentation"] == 0:
            warnings.append("no_segmentation_file")
        if modality == "xray" and not (counts["preview_or_image"] or counts["volume_or_image"] or counts["dicom"]):
            warnings.append("no_xray_image_candidate")
        rows.append(
            {
                "case_id": case_id,
                "modality": modality,
                "token_variants": len(case_tokens.get(case_id, [])),
                "total_size_mb": round(sizes[(case_id, modality)], 3),
                "dicom": counts["dicom"],
                "segmentation": counts["segmentation"],
                "image_preview": counts["preview_or_image"],
                "volume_or_image": counts["volume_or_image"],
                "dcm": counts[".dcm"],
                "nrrd": counts[".nrrd"],
                "seg_nrrd": counts[".seg.nrrd"],
                "png_jpg": counts[".png"] + counts[".jpg"] + counts[".jpeg"],
                "nhdr_rawgz": counts[".nhdr"] + counts[".raw.gz"],
                "warnings": ";".join(warnings),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Controlled short-path inventory/extraction for new patient zip packages."
    )
    parser.add_argument("--source", default="data/newpatient", help="Folder containing the original zip files.")
    parser.add_argument("--stage", default=r"C:\np_stage", help="Short staging root for outputs.")
    parser.add_argument("--mode", choices=["inventory", "extract"], default="inventory")
    parser.add_argument(
        "--id_mode",
        choices=["case", "tc"],
        default="case",
        help="Folder naming mode. 'case' anonymizes-ish with hash, 'tc' uses the 11-digit TC when present.",
    )
    parser.add_argument("--include_mr", action="store_true", help="Also process MR zip. Default skips MR.")
    parser.add_argument("--keep_mrml", action="store_true", help="Keep .mrml files. Default skips them.")
    parser.add_argument(
        "--run_name",
        default=None,
        help="Output subfolder name. Default: timestamped run under the stage folder.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    stage = Path(args.stage) / run_name
    stage.mkdir(parents=True, exist_ok=True)

    public_rows = []
    private_rows = []
    case_tokens = defaultdict(set)
    zip_summaries = []

    for source_zip in iter_source_zips(source):
        zip_summaries.append(
            process_zip(
                source_zip=source_zip,
                mode=args.mode,
                stage=stage,
                include_mr=args.include_mr,
                keep_mrml=args.keep_mrml,
                id_mode=args.id_mode,
                public_rows=public_rows,
                private_rows=private_rows,
                case_tokens=case_tokens,
            )
        )

    case_summary = build_case_summary(public_rows, case_tokens)
    inventory_dir = stage / "inventory"
    write_csv(
        inventory_dir / "public_file_inventory.csv",
        public_rows,
        ["case_id", "modality", "role", "suffix", "size_mb", "dest_rel"],
    )
    write_csv(
        inventory_dir / "private_mapping.csv",
        private_rows,
        [
            "case_id",
            "modality",
            "source_zip",
            "nested_zip",
            "original_path",
            "patient_token",
            "role",
            "suffix",
            "size_bytes",
            "dest_rel",
        ],
    )
    write_csv(
        inventory_dir / "case_summary.csv",
        case_summary,
        [
            "case_id",
            "modality",
            "token_variants",
            "total_size_mb",
            "dicom",
            "segmentation",
            "image_preview",
            "volume_or_image",
            "dcm",
            "nrrd",
            "seg_nrrd",
            "png_jpg",
            "nhdr_rawgz",
            "warnings",
        ],
    )
    with open(inventory_dir / "zip_summary.json", "w", encoding="utf-8") as f:
        json.dump(zip_summaries, f, indent=2, ensure_ascii=False)

    modality_counts = Counter(row["modality"] for row in public_rows)
    print(f"Mode: {args.mode}")
    print(f"Stage: {stage}")
    print(f"Files kept in inventory: {len(public_rows)}")
    print(f"Cases/modalities: {len(case_summary)}")
    print(f"Modality file counts: {dict(modality_counts)}")
    print(f"Inventory written to: {inventory_dir}")


if __name__ == "__main__":
    main()
