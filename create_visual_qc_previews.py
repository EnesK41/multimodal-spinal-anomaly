import argparse
import csv
import io
import math
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".nrrd", ".nhdr", ".dcm"}


def norm_path(value):
    return str(value).replace("\\", "/")


def suffix_key(path):
    name = PurePosixPath(norm_path(path)).name.lower()
    if name.endswith(".seg.nrrd"):
        return ".seg.nrrd"
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".raw.gz"):
        return ".raw.gz"
    return Path(name).suffix.lower() or "<no_ext>"


def parent_virtual_path(path):
    return str(PurePosixPath(norm_path(path)).parent)


def split_virtual_archive_path(vpath):
    parts = list(PurePosixPath(norm_path(vpath)).parts)
    for idx, part in enumerate(parts):
        lower = part.lower()
        if lower.endswith(".zip") or lower.endswith(".7z"):
            archive = Path(*parts[: idx + 1])
            inner = "/".join(parts[idx + 1 :])
            return archive, inner
    return None, norm_path(vpath)


def read_from_zip_bytes(zip_bytes, inner):
    parts = list(PurePosixPath(inner).parts)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for idx, part in enumerate(parts):
            if part.lower().endswith(".zip"):
                nested_name = "/".join(parts[: idx + 1])
                nested_bytes = zf.read(nested_name)
                return read_from_zip_bytes(nested_bytes, "/".join(parts[idx + 1 :]))
        return zf.read(inner)


def read_virtual_file(vpath):
    archive, inner = split_virtual_archive_path(vpath)
    if archive is None:
        return Path(vpath).read_bytes()
    if archive.suffix.lower() == ".zip":
        return read_from_zip_bytes(archive.read_bytes(), inner)
    raise ValueError(f"Direct file read from {archive.suffix} is not supported for {vpath}")


def extract_virtual_files(records, temp_dir, suffix_filter=None, max_files=None):
    out_paths = []
    for idx, record in enumerate(records):
        if max_files is not None and len(out_paths) >= max_files:
            break
        vpath = record["virtual_path"]
        suffix = record["suffix"]
        if suffix_filter and suffix not in suffix_filter:
            continue
        archive, inner = split_virtual_archive_path(vpath)
        ext = suffix if suffix.startswith(".") else ""
        dest = Path(temp_dir) / f"file_{len(out_paths):04d}{ext}"
        if archive and archive.suffix.lower() == ".7z":
            # 7z extraction is handled at group level; skip individual reads here.
            continue
        try:
            dest.write_bytes(read_virtual_file(vpath))
            out_paths.append(dest)
        except Exception:
            continue
    return out_paths


def extract_7z_group(archive_path, inner_parent, temp_dir):
    # Extract the whole group folder. bsdtar accepts 7z archives on this machine.
    subprocess.run(
        ["tar", "-xf", str(archive_path.resolve()), "-C", str(temp_dir), inner_parent],
        check=False,
        capture_output=True,
        text=True,
    )
    extracted_root = Path(temp_dir) / inner_parent
    if extracted_root.exists():
        return sorted(p for p in extracted_root.rglob("*") if p.is_file())
    return []


def image_to_uint8(arr):
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[..., 0]
    if arr.ndim == 3:
        # For X-ray volumes/multiframe images, keep projection so orientation problems remain visible.
        arr = np.max(arr, axis=0)
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim != 2:
        arr = np.reshape(arr, arr.shape[-2:])
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros((256, 256), dtype=np.uint8)
    values = arr[finite]
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def read_image_file(path):
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)
    if image.GetNumberOfComponentsPerPixel() > 1:
        arr = np.asarray(arr)
        arr = arr[..., 0]
    return image_to_uint8(arr)


def read_dicom_group(paths):
    folder = Path(paths[0]).parent
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder)) or []
    image = None
    if series_ids:
        ranked = []
        for sid in series_ids:
            names = reader.GetGDCMSeriesFileNames(str(folder), sid)
            ranked.append((len(names), names))
        for _, names in sorted(ranked, reverse=True):
            try:
                reader.SetFileNames(names)
                image = reader.Execute()
                break
            except Exception:
                image = None
    if image is None:
        best = None
        best_score = -1
        for path in paths:
            try:
                candidate = sitk.ReadImage(str(path))
                size = candidate.GetSize()
                score = int(np.prod(size[:2])) * max(1, size[2] if len(size) > 2 else 1)
                if score > best_score:
                    best = candidate
                    best_score = score
            except Exception:
                continue
        if best is None:
            raise RuntimeError("No readable DICOM image in group")
        image = best
    arr = sitk.GetArrayFromImage(image)
    if image.GetNumberOfComponentsPerPixel() > 1:
        arr = np.asarray(arr)[..., 0]
    return image_to_uint8(arr), image


def make_ct_drr(image, hu_threshold=300, clip_max=1800, axis=1, rotate_k=1):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    arr[arr < hu_threshold] = 0
    arr = np.clip(arr, hu_threshold, clip_max)
    proj = np.max(arr, axis=axis)
    if rotate_k:
        proj = np.rot90(proj, k=rotate_k)
    return image_to_uint8(proj)


def save_preview(arr, out_path, label_text="", size=(320, 320)):
    img = Image.fromarray(arr).convert("L")
    img.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size[0], size[1] + 34), "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img.convert("RGB"), (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, size[1] + 6), label_text[:52], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_contact_sheets(image_rows, out_dir, prefix, cols=5, rows_per_page=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    page_size = cols * rows_per_page
    pages = []
    for page_idx in range(math.ceil(len(image_rows) / page_size)):
        chunk = image_rows[page_idx * page_size : (page_idx + 1) * page_size]
        thumbs = []
        for item in chunk:
            img = Image.open(item["path"]).convert("RGB")
            thumbs.append(img)
        if not thumbs:
            continue
        w, h = thumbs[0].size
        sheet = Image.new("RGB", (cols * w, rows_per_page * h), "white")
        for i, img in enumerate(thumbs):
            sheet.paste(img, ((i % cols) * w, (i // cols) * h))
        out_path = out_dir / f"{prefix}_page_{page_idx + 1:02d}.png"
        sheet.save(out_path)
        pages.append(out_path)
    return pages


def choose_xray_records(tc, records_by_tc_mod):
    records = [
        r
        for r in records_by_tc_mod[(tc, "xray")]
        if r["suffix"] in IMAGE_SUFFIXES and r["suffix"] != ".seg.nrrd"
    ]
    if not records:
        return []
    priority = {".png": 0, ".jpg": 0, ".jpeg": 0, ".nrrd": 1, ".nhdr": 2, ".dcm": 3}
    groups = defaultdict(list)
    for r in records:
        key = parent_virtual_path(r["virtual_path"]) if r["suffix"] in {".dcm", ".nhdr"} else r["virtual_path"]
        groups[key].append(r)
    ranked = sorted(
        groups.values(),
        key=lambda g: (min(priority.get(r["suffix"], 9) for r in g), -len(g), g[0]["virtual_path"]),
    )
    return ranked[0]


def choose_ct_group(tc, records_by_tc_mod, min_slices):
    records = records_by_tc_mod[(tc, "ct")]
    dcm_groups = defaultdict(list)
    nrrds = []
    for r in records:
        if r["suffix"] == ".dcm":
            dcm_groups[parent_virtual_path(r["virtual_path"])].append(r)
        elif r["suffix"] == ".nrrd":
            nrrds.append(r)
    large_dcm = sorted([g for g in dcm_groups.values() if len(g) >= min_slices], key=len, reverse=True)
    if large_dcm:
        return "dicom", large_dcm[0]
    # Fallback to NRRDs; dimensions are checked after extraction/read.
    if nrrds:
        return "nrrd", sorted(nrrds, key=lambda r: r["virtual_path"])[0:1]
    return "", []


def render_xray(tc, label, records, out_path):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            suffixes = {r["suffix"] for r in records}
            if ".dcm" in suffixes:
                paths = extract_virtual_files(records, tmp, suffix_filter={".dcm"})
                if not paths:
                    return "failed_extract_dicom"
                arr, _ = read_dicom_group(paths)
            elif ".nhdr" in suffixes:
                # Need nhdr + raw.gz from same parent.
                parent = parent_virtual_path(records[0]["virtual_path"])
                sibling_records = [r for r in records if parent_virtual_path(r["virtual_path"]) == parent]
                paths = extract_virtual_files(sibling_records, tmp, suffix_filter={".nhdr", ".raw.gz"})
                nhdrs = [p for p in paths if p.suffix.lower() == ".nhdr"]
                if not nhdrs:
                    return "failed_extract_nhdr"
                arr = read_image_file(nhdrs[0])
            else:
                paths = extract_virtual_files(records, tmp, max_files=1)
                if not paths:
                    return "failed_extract_image"
                arr = read_image_file(paths[0])
            save_preview(arr, out_path, f"{tc} | {label}")
            return "ok"
        except Exception as exc:
            return f"error:{type(exc).__name__}"


def render_ct(tc, group_type, records, out_path, min_slices):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            if group_type == "dicom":
                archive, inner = split_virtual_archive_path(records[0]["virtual_path"])
                parent = parent_virtual_path(records[0]["virtual_path"])
                if archive and archive.suffix.lower() == ".7z":
                    inner_parent = parent.split(norm_path(str(archive)) + "/", 1)[-1]
                    paths = extract_7z_group(archive, inner_parent, tmp)
                    paths = [p for p in paths if p.suffix.lower() == ".dcm"]
                else:
                    paths = extract_virtual_files(records, tmp, suffix_filter={".dcm"})
                if len(paths) < min_slices:
                    return f"too_few_extracted_slices:{len(paths)}", 0
                _, image = read_dicom_group(paths)
            elif group_type == "nrrd":
                paths = extract_virtual_files(records, tmp, max_files=1)
                if not paths:
                    return "failed_extract_nrrd", 0
                image = sitk.ReadImage(str(paths[0]))
                if max(image.GetSize()) < min_slices:
                    return f"too_small_nrrd:{image.GetSize()}", 0
            else:
                return "no_ct_group", 0
            size = image.GetSize()
            if max(size) < min_slices:
                return f"too_few_slices:{size}", 0
            arr = make_ct_drr(image)
            save_preview(arr, out_path, f"{tc} | CT {size}")
            return "ok", max(size)
        except Exception as exc:
            return f"error:{type(exc).__name__}", 0


def main():
    parser = argparse.ArgumentParser(description="Create visual QC previews for new patient X-ray and CT data.")
    parser.add_argument("--audit_dir", required=True, help="Folder containing all_tc_source_records.csv and final_dataset_audit.")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--min_ct_slices", type=int, default=200)
    parser.add_argument("--mode", choices=["all", "xray", "ct", "sheets"], default="all")
    parser.add_argument("--max_ct", type=int, default=0, help="Optional limit for CT DRR generation.")
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    final_dir = audit_dir / "final_dataset_audit"
    out_dir = Path(args.out_dir or (audit_dir / "visual_qc"))
    out_dir.mkdir(parents=True, exist_ok=True)

    records = list(csv.DictReader(open(audit_dir / "all_tc_source_records.csv", encoding="utf-8")))
    labels = list(csv.DictReader(open(final_dir / "final_xray_anchor_labels.csv", encoding="utf-8")))
    records_by_tc_mod = defaultdict(list)
    for r in records:
        if r.get("tc"):
            records_by_tc_mod[(r["tc"], r["modality"])].append(r)

    xray_rows = []
    if args.mode in {"all", "xray"}:
        for row in labels:
            tc = row["tc"]
            label = row["label"]
            selected = choose_xray_records(tc, records_by_tc_mod)
            subdir = "healthy" if label == "healthy" else "anomaly"
            out_path = out_dir / "xray_previews" / subdir / f"{tc}.png"
            if out_path.exists():
                status = "ok"
            else:
                status = render_xray(tc, label, selected, out_path) if selected else "no_xray_image_record"
            xray_rows.append(
                {
                    "tc": tc,
                    "label": label,
                    "status": status,
                    "preview_path": str(out_path) if out_path.exists() else "",
                    "selected_records": len(selected),
                    "selected_suffixes": ";".join(sorted(Counter(r["suffix"] for r in selected))),
                    "selected_first_path": selected[0]["virtual_path"] if selected else "",
                }
            )

    ct_tcs = sorted({r["tc"] for r in records if r.get("tc") and r.get("modality") == "ct"})
    if args.max_ct:
        existing = {p.stem for p in (out_dir / "ct_drr_previews").glob("*.png")}
        ct_tcs = [tc for tc in ct_tcs if tc not in existing][: args.max_ct]
    ct_rows = []
    if args.mode in {"all", "ct"}:
        for tc in ct_tcs:
            group_type, selected = choose_ct_group(tc, records_by_tc_mod, args.min_ct_slices)
            out_path = out_dir / "ct_drr_previews" / f"{tc}.png"
            if out_path.exists():
                status, slice_metric = "ok", 0
            elif selected:
                status, slice_metric = render_ct(tc, group_type, selected, out_path, args.min_ct_slices)
            else:
                status, slice_metric = "no_ct_candidate", 0
            ct_rows.append(
                {
                    "tc": tc,
                    "status": status,
                    "group_type": group_type,
                    "slice_metric": slice_metric,
                    "preview_path": str(out_path) if out_path.exists() else "",
                    "selected_records": len(selected),
                    "selected_first_path": selected[0]["virtual_path"] if selected else "",
                }
            )

    def write_csv(path, rows):
        fields = list(rows[0].keys()) if rows else []
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    if xray_rows:
        write_csv(out_dir / "xray_preview_summary.csv", xray_rows)
    if ct_rows:
        write_csv(out_dir / "ct_drr_summary_latest_batch.csv", ct_rows)

    # Sheet mode can work from existing PNGs even if summaries were not written.
    ok_xray = []
    for row in labels:
        subdir = "healthy" if row["label"] == "healthy" else "anomaly"
        path = out_dir / "xray_previews" / subdir / f"{row['tc']}.png"
        if path.exists():
            ok_xray.append({"label": row["label"], "preview_path": str(path)})
    ok_ct = [{"preview_path": str(p)} for p in sorted((out_dir / "ct_drr_previews").glob("*.png"))]
    if args.mode in {"all", "sheets", "xray", "ct"}:
        make_contact_sheets(
            [{"path": r["preview_path"]} for r in ok_xray if r["label"] == "healthy"],
            out_dir / "contact_sheets",
            "xray_healthy",
        )
        make_contact_sheets(
            [{"path": r["preview_path"]} for r in ok_xray if r["label"] != "healthy"],
            out_dir / "contact_sheets",
            "xray_anomaly",
        )
        make_contact_sheets(
            [{"path": r["preview_path"]} for r in ok_ct],
            out_dir / "contact_sheets",
            "ct_drr",
        )

    summary = {
        "xray_png_existing": len(ok_xray),
        "ct_drr_png_existing": len(ok_ct),
        "xray_batch_total": len(xray_rows),
        "xray_batch_ok": sum(r["status"] == "ok" for r in xray_rows),
        "xray_batch_failed": sum(r["status"] != "ok" for r in xray_rows),
        "ct_batch_total": len(ct_rows),
        "ct_batch_ok": sum(r["status"] == "ok" for r in ct_rows),
        "ct_batch_failed_or_skipped": sum(r["status"] != "ok" for r in ct_rows),
        "out_dir": str(out_dir),
    }
    (out_dir / "visual_qc_summary.txt").write_text("\n".join(f"{k}: {v}" for k, v in summary.items()), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
