import argparse
import csv
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath


TC_RE = re.compile(r"(?<!\d)(\d{11})(?!\d)")


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


def tcs_in(*parts):
    text = "/".join(str(part) for part in parts if part)
    return sorted(set(TC_RE.findall(text)))


def scan_zip_bytes(data, context, source_group, rows, depth=0):
    if depth > 8:
        rows.append(
            {
                "source_group": source_group,
                "modality": modality_from_path(context),
                "suffix": "<max_depth>",
                "size_bytes": len(data),
                "virtual_path": context,
                "tc_values": ";".join(tcs_in(context)),
            }
        )
        return
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                member_context = f"{context}/{info.filename}"
                suffix = suffix_key(info.filename)
                if suffix == ".zip":
                    try:
                        nested_data = zf.open(info).read()
                        scan_zip_bytes(nested_data, member_context, source_group, rows, depth + 1)
                    except Exception as exc:
                        rows.append(
                            {
                                "source_group": source_group,
                                "modality": modality_from_path(member_context),
                                "suffix": "<bad_zip>",
                                "size_bytes": info.file_size,
                                "virtual_path": member_context,
                                "tc_values": ";".join(tcs_in(member_context)),
                                "error": str(exc).splitlines()[0],
                            }
                        )
                else:
                    rows.append(
                        {
                            "source_group": source_group,
                            "modality": modality_from_path(member_context),
                            "suffix": suffix,
                            "size_bytes": info.file_size,
                            "virtual_path": member_context,
                            "tc_values": ";".join(tcs_in(member_context)),
                        }
                    )
    except Exception as exc:
        rows.append(
            {
                "source_group": source_group,
                "modality": modality_from_path(context),
                "suffix": "<bad_zip>",
                "size_bytes": len(data),
                "virtual_path": context,
                "tc_values": ";".join(tcs_in(context)),
                "error": str(exc).splitlines()[0],
            }
        )


def scan_zip_file(path, source_group, rows):
    scan_zip_bytes(path.read_bytes(), str(path), source_group, rows)


def scan_open_dir(path, rows):
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        suffix = suffix_key(item.name)
        if suffix == ".zip":
            scan_zip_file(item, "open_folder_nested_zip", rows)
        else:
            rows.append(
                {
                    "source_group": "open_folder_file",
                    "modality": modality_from_path(str(item)),
                    "suffix": suffix,
                    "size_bytes": item.stat().st_size,
                    "virtual_path": str(item),
                    "tc_values": ";".join(tcs_in(str(item))),
                }
            )


def summarize(rows):
    source_summary = defaultdict(lambda: defaultdict(set))
    file_counts = defaultdict(Counter)
    no_tc = defaultdict(Counter)
    suffix_counts = defaultdict(Counter)

    for row in rows:
        source_group = row["source_group"]
        modality = row["modality"]
        file_counts[source_group][modality] += 1
        suffix_counts[(source_group, modality)][row["suffix"]] += 1
        tcs = [tc for tc in row.get("tc_values", "").split(";") if tc]
        if not tcs:
            no_tc[source_group][modality] += 1
        for tc in tcs:
            source_summary[source_group][modality].add(tc)

    union_by_modality = defaultdict(set)
    for source_group, modalities in source_summary.items():
        for modality, values in modalities.items():
            union_by_modality[modality].update(values)

    summary_rows = []
    for source_group in sorted(file_counts):
        for modality in sorted(file_counts[source_group]):
            summary_rows.append(
                {
                    "source_group": source_group,
                    "modality": modality,
                    "unique_tc": len(source_summary[source_group][modality]),
                    "rows": file_counts[source_group][modality],
                    "no_tc_rows": no_tc[source_group][modality],
                    "top_suffixes": json.dumps(suffix_counts[(source_group, modality)].most_common(8)),
                }
            )
    for modality in sorted(union_by_modality):
        summary_rows.append(
            {
                "source_group": "combined_union",
                "modality": modality,
                "unique_tc": len(union_by_modality[modality]),
                "rows": "",
                "no_tc_rows": "",
                "top_suffixes": "",
            }
        )

    overlap = {
        "ct_xray": len(union_by_modality["ct"] & union_by_modality["xray"]),
        "mr_ct": len(union_by_modality["mr"] & union_by_modality["ct"]),
        "mr_xray": len(union_by_modality["mr"] & union_by_modality["xray"]),
        "all_three": len(union_by_modality["mr"] & union_by_modality["ct"] & union_by_modality["xray"]),
    }
    return summary_rows, overlap


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Audit newpatient data across top zips and open folders.")
    parser.add_argument("--source", default="data/newpatient")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out or (r"C:\np_stage" + "\\" + datetime.now().strftime("source_audit_%Y%m%d_%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in sorted(source.iterdir()):
        if item.is_file() and suffix_key(item.name) == ".zip":
            scan_zip_file(item, "top_zip", rows)
        elif item.is_dir():
            scan_open_dir(item, rows)

    for row in rows:
        row.setdefault("error", "")
        row["size_mb"] = round(int(row["size_bytes"]) / (1024 * 1024), 4)

    summary_rows, overlap = summarize(rows)

    write_csv(
        out / "source_file_audit.csv",
        rows,
        ["source_group", "modality", "suffix", "size_bytes", "size_mb", "tc_values", "virtual_path", "error"],
    )
    write_csv(
        out / "source_summary.csv",
        summary_rows,
        ["source_group", "modality", "unique_tc", "rows", "no_tc_rows", "top_suffixes"],
    )
    with open(out / "overlap_summary.json", "w", encoding="utf-8") as f:
        json.dump(overlap, f, indent=2)

    print(f"Output: {out}")
    for row in summary_rows:
        if row["source_group"] == "combined_union":
            print(f"{row['modality']}: combined unique TC = {row['unique_tc']}")
    print(f"Overlap: {overlap}")


if __name__ == "__main__":
    main()
