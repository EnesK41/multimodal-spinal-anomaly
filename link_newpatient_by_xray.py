import argparse
import csv
import io
import json
import re
import subprocess
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath


LEADING_TC_RE = re.compile(r"^(\d{11})(?:\D|$)")


def normalize_text(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value.lower().replace("ı", "i")


def modality_from_path(path):
    norm = normalize_text(path)
    if "rontgen" in norm or "xray" in norm or "x-ray" in norm:
        return "xray"
    if "mrg" in norm or re.search(r"\bmr\b", norm):
        return "mr"
    if "segmente edilenler" in norm or "bt" in norm or "ct" in norm:
        return "ct"
    return "unknown"


def suffix_key(path):
    name = PurePosixPath(str(path).replace("\\", "/")).name.lower()
    if name.endswith(".seg.nrrd"):
        return ".seg.nrrd"
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".raw.gz"):
        return ".raw.gz"
    return Path(name).suffix.lower() or "<no_ext>"


def strip_package_suffix(name):
    base = PurePosixPath(str(name).replace("\\", "/")).name
    lower = base.lower()
    for suffix in [".seg.nrrd", ".nii.gz", ".raw.gz", ".zip"]:
        if lower.endswith(suffix):
            return base[: -len(suffix)]
    suffix = Path(base).suffix
    if suffix:
        return base[: -len(suffix)]
    return base


def tc_prefix_from_component(component):
    base = strip_package_suffix(component).strip()
    match = LEADING_TC_RE.match(base)
    if match:
        return match.group(1)
    return ""


def parts_from_virtual_path(path):
    return [p for p in PurePosixPath(str(path).replace("\\", "/")).parts if p not in {"", "/"}]


def first_tc_in_parts(parts):
    for idx, part in enumerate(parts):
        tc = tc_prefix_from_component(part)
        if tc:
            return tc, idx, part
    return "", -1, ""


def scan_virtual_path(records, *, source_group, modality, virtual_path, item_type, size_bytes=0, suffix=""):
    parts = parts_from_virtual_path(virtual_path)
    tc, tc_depth, tc_component = first_tc_in_parts(parts)
    records.append(
        {
            "source_group": source_group,
            "modality": modality or modality_from_path(virtual_path),
            "tc": tc,
            "tc_depth": tc_depth,
            "tc_component": tc_component,
            "item_type": item_type,
            "suffix": suffix or suffix_key(virtual_path),
            "size_mb": round(size_bytes / (1024 * 1024), 4),
            "virtual_path": virtual_path,
        }
    )


def scan_zip_bytes(data, context, source_group, records, depth=0):
    if depth > 8:
        return
    modality = modality_from_path(context)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            seen_dirs = set()
            for info in zf.infolist():
                if info.is_dir():
                    continue
                member_path = f"{context}/{info.filename}"
                member_parts = parts_from_virtual_path(member_path)
                accum = []
                for part in member_parts[:-1]:
                    accum.append(part)
                    dir_path = "/".join(accum)
                    if dir_path not in seen_dirs:
                        seen_dirs.add(dir_path)
                        scan_virtual_path(
                            records,
                            source_group=source_group,
                            modality=modality,
                            virtual_path=dir_path,
                            item_type="zip_dir",
                        )

                suffix = suffix_key(info.filename)
                if suffix == ".zip":
                    scan_virtual_path(
                        records,
                        source_group=source_group,
                        modality=modality,
                        virtual_path=member_path,
                        item_type="zip_package",
                        size_bytes=info.file_size,
                        suffix=suffix,
                    )
                    try:
                        nested_data = zf.open(info).read()
                        scan_zip_bytes(nested_data, member_path, source_group, records, depth + 1)
                    except Exception:
                        continue
                else:
                    scan_virtual_path(
                        records,
                        source_group=source_group,
                        modality=modality,
                        virtual_path=member_path,
                        item_type="zip_file",
                        size_bytes=info.file_size,
                        suffix=suffix,
                    )
    except Exception:
        return


def scan_zip_file(path, source_group, records):
    scan_virtual_path(
        records,
        source_group=source_group,
        modality=modality_from_path(str(path)),
        virtual_path=str(path),
        item_type="fs_zip_package",
        size_bytes=path.stat().st_size,
        suffix=".zip",
    )
    scan_zip_bytes(path.read_bytes(), str(path), source_group, records)


def scan_7z_file(path, source_group, records):
    scan_virtual_path(
        records,
        source_group=source_group,
        modality=modality_from_path(str(path)),
        virtual_path=str(path),
        item_type="fs_7z_package",
        size_bytes=path.stat().st_size,
        suffix=".7z",
    )
    proc = subprocess.run(
        ["tar", "-tf", str(path.resolve())],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return
    for line in proc.stdout.splitlines():
        virtual_path = f"{path}/{line}"
        scan_virtual_path(
            records,
            source_group=source_group,
            modality=modality_from_path(str(path)),
            virtual_path=virtual_path,
            item_type="archive_file",
            suffix=suffix_key(line),
        )


def scan_open_dir(path, records):
    scan_virtual_path(
        records,
        source_group="open_folder",
        modality=modality_from_path(str(path)),
        virtual_path=str(path),
        item_type="fs_dir",
    )
    for item in path.rglob("*"):
        modality = modality_from_path(str(item))
        if item.is_dir():
            scan_virtual_path(
                records,
                source_group="open_folder",
                modality=modality,
                virtual_path=str(item),
                item_type="fs_dir",
            )
            continue
        suffix = suffix_key(item.name)
        if suffix == ".zip":
            scan_zip_file(item, "open_folder_nested_zip", records)
        elif suffix == ".7z":
            scan_7z_file(item, "open_folder_nested_7z", records)
        else:
            scan_virtual_path(
                records,
                source_group="open_folder",
                modality=modality,
                virtual_path=str(item),
                item_type="fs_file",
                size_bytes=item.stat().st_size,
                suffix=suffix,
            )


def choose_representative(records):
    if not records:
        return ""
    priority = {
        "fs_dir": 0,
        "fs_zip_package": 1,
        "zip_package": 2,
        "zip_dir": 3,
        "fs_file": 4,
        "zip_file": 5,
    }
    sorted_records = sorted(
        records,
        key=lambda row: (
            priority.get(row["item_type"], 9),
            int(row["tc_depth"]) if str(row["tc_depth"]).isdigit() else 99,
            len(row["virtual_path"]),
            row["virtual_path"],
        ),
    )
    return sorted_records[0]["virtual_path"]


def build_anchor_rows(records):
    valid = [row for row in records if row["tc"]]
    by_tc_modality = defaultdict(lambda: defaultdict(list))
    package_counts = defaultdict(lambda: defaultdict(Counter))
    suffix_counts = defaultdict(lambda: defaultdict(Counter))

    for row in valid:
        by_tc_modality[row["tc"]][row["modality"]].append(row)
        if row["item_type"] in {"fs_dir", "fs_zip_package", "zip_package", "zip_dir"}:
            package_counts[row["tc"]][row["modality"]][row["item_type"]] += 1
        suffix_counts[row["tc"]][row["modality"]][row["suffix"]] += 1

    xray_tcs = sorted(tc for tc, mods in by_tc_modality.items() if mods.get("xray"))
    rows = []
    for tc in xray_tcs:
        mods = by_tc_modality[tc]
        rows.append(
            {
                "tc": tc,
                "has_xray": bool(mods.get("xray")),
                "has_mr": bool(mods.get("mr")),
                "has_ct": bool(mods.get("ct")),
                "xray_items": len(mods.get("xray", [])),
                "mr_items": len(mods.get("mr", [])),
                "ct_items": len(mods.get("ct", [])),
                "xray_packages": sum(package_counts[tc]["xray"].values()),
                "mr_packages": sum(package_counts[tc]["mr"].values()),
                "ct_packages": sum(package_counts[tc]["ct"].values()),
                "xray_suffixes": json.dumps(suffix_counts[tc]["xray"].most_common(8)),
                "mr_suffixes": json.dumps(suffix_counts[tc]["mr"].most_common(8)),
                "ct_suffixes": json.dumps(suffix_counts[tc]["ct"].most_common(8)),
                "xray_representative": choose_representative(mods.get("xray", [])),
                "mr_representative": choose_representative(mods.get("mr", [])),
                "ct_representative": choose_representative(mods.get("ct", [])),
            }
        )
    return rows, by_tc_modality


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build an X-ray anchored TC index across newpatient sources.")
    parser.add_argument("--source", default="data/newpatient")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out or (r"C:\np_stage" + "\\" + datetime.now().strftime("xray_anchor_%Y%m%d_%H%M%S")))
    records = []

    for item in sorted(source.iterdir()):
        if item.is_file() and suffix_key(item.name) == ".zip":
            scan_zip_file(item, "top_zip", records)
        elif item.is_file() and suffix_key(item.name) == ".7z":
            scan_7z_file(item, "top_7z", records)
        elif item.is_dir():
            scan_open_dir(item, records)

    anchor_rows, by_tc_modality = build_anchor_rows(records)
    all_modality_tcs = {
        modality: sorted({tc for tc, mods in by_tc_modality.items() if mods.get(modality)})
        for modality in ["xray", "mr", "ct", "unknown"]
    }
    summary = {
        "xray_anchor_tc_count": len(anchor_rows),
        "xray_with_mr": sum(1 for row in anchor_rows if row["has_mr"]),
        "xray_with_ct": sum(1 for row in anchor_rows if row["has_ct"]),
        "xray_with_mr_and_ct": sum(1 for row in anchor_rows if row["has_mr"] and row["has_ct"]),
        "all_xray_tc": len(all_modality_tcs["xray"]),
        "all_mr_tc": len(all_modality_tcs["mr"]),
        "all_ct_tc": len(all_modality_tcs["ct"]),
    }

    write_csv(
        out / "all_tc_source_records.csv",
        records,
        [
            "source_group",
            "modality",
            "tc",
            "tc_depth",
            "tc_component",
            "item_type",
            "suffix",
            "size_mb",
            "virtual_path",
        ],
    )
    write_csv(
        out / "xray_anchor_links.csv",
        anchor_rows,
        [
            "tc",
            "has_xray",
            "has_mr",
            "has_ct",
            "xray_items",
            "mr_items",
            "ct_items",
            "xray_packages",
            "mr_packages",
            "ct_packages",
            "xray_suffixes",
            "mr_suffixes",
            "ct_suffixes",
            "xray_representative",
            "mr_representative",
            "ct_representative",
        ],
    )
    with open(out / "xray_anchor_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Output: {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
