import base64
import glob
import io
import json
import mimetypes
import os
import re
import sys
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from models.SmallXrayEncoder import SmallXrayEncoder
from models.XrayEncoder import XrayEncoder
from models.classifier import XrayClassifier

try:
    import nibabel as nib
except Exception:
    nib = None


IMG_SIZE = (1024, 512)
ROI_WIDTH_RATIO = 0.45
DEFAULT_CHECKPOINTS = [
    PROJECT_ROOT
    / "checkpoints"
    / "png_roi045_no_extra_roi_seed42_resnet34_pretrained_mask_scrubbed_fold_04_best.pth",
    PROJECT_ROOT / "checkpoints" / "png_cv5_seed42_resnet34_pretrained_mask_scrubbed_roi045_fold_04_best.pth",
    PROJECT_ROOT / "checkpoints" / "png_full_demo_holdout_seed42_best.pth",
    PROJECT_ROOT
    / "checkpoints"
    / "cv5_resnet34_pretrained_mask_scrubbed_roi045_fold_01_best_xray_classifier_roi_045_balacc.pth",
]


class ModelRuntime:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False
        self.error = None
        self.checkpoint_path = None
        self.checkpoint_name = None
        self.threshold = 0.5
        self.val_patients = []
        self.xray_enc = None
        self.classifier = None

    def load(self):
        if self.loaded or self.error:
            return

        env_checkpoint = os.environ.get("DEMO_CHECKPOINT")
        candidates = [Path(env_checkpoint)] if env_checkpoint else DEFAULT_CHECKPOINTS
        self.checkpoint_path = ""
        for candidate in candidates:
            matches = sorted(glob.glob(str(candidate)))
            if matches:
                self.checkpoint_path = matches[0]
                break
        if not self.checkpoint_path:
            self.error = "No demo checkpoint found."
            return

        self.checkpoint_name = os.path.basename(self.checkpoint_path)

        try:
            ckpt = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            encoder_type = ckpt.get("encoder_type", "resnet34")
            if encoder_type == "small_cnn":
                self.xray_enc = SmallXrayEncoder(embedding_dim=512).to(self.device)
            else:
                self.xray_enc = XrayEncoder(embedding_dim=512, pretrained=False).to(self.device)

            self.classifier = XrayClassifier(embedding_dim=512, dropout=0.5).to(self.device)
            self.xray_enc.load_state_dict(ckpt["xray_enc"])
            self.classifier.load_state_dict(ckpt["classifier"])
            self.xray_enc.eval()
            self.classifier.eval()
            self.threshold = float(ckpt.get("best_threshold", 0.5))
            self.val_patients = list(ckpt.get("val_patients", []))
            self.loaded = True
        except Exception as exc:
            self.error = str(exc)

    def health(self):
        self.load()
        return {
            "model_loaded": self.loaded,
            "error": self.error,
            "device": str(self.device),
            "checkpoint_name": self.checkpoint_name,
            "threshold": self.threshold,
            "held_out_patients": self.val_patients,
        }


runtime = ModelRuntime()


def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val > min_val:
        tensor = (tensor - min_val) / (max_val - min_val)
    return tensor


def resize_with_padding(tensor, size=IMG_SIZE):
    _, h, w = tensor.shape
    target_h, target_w = size
    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))


def body_center_roi_box(xray, roi_width_ratio=ROI_WIDTH_RATIO):
    _, height, width = xray.shape
    img = xray[0]
    threshold = max(0.03, float(img.mean()) * 0.5)
    foreground = img > threshold
    ys, xs = torch.where(foreground)

    if len(ys) == 0:
        return 0, height, 0, width

    y1 = int(ys.min().item())
    y2 = int(ys.max().item())
    x1_body = int(xs.min().item())
    x2_body = int(xs.max().item())
    body_w = x2_body - x1_body + 1
    cx = (x1_body + x2_body) // 2
    roi_w = int(body_w * roi_width_ratio)
    x1 = max(0, cx - roi_w // 2)
    x2 = min(width, cx + roi_w // 2)
    margin_y = int(height * 0.02)
    y1 = max(0, y1 - margin_y)
    y2 = min(height, y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return 0, height, 0, width
    return y1, y2, x1, x2


def load_image_from_bytes(filename, raw_bytes):
    lower = filename.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        if nib is None:
            raise RuntimeError("nibabel is required for NIfTI upload support.")
        suffix = ".nii.gz" if lower.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            img_obj = nib.as_closest_canonical(nib.load(tmp_path))
            img = np.squeeze(img_obj.get_fdata())
        finally:
            os.remove(tmp_path)
        if img.ndim > 2:
            img = img[:, :, img.shape[2] // 2]
        return img.astype(np.float32)

    image = Image.open(io.BytesIO(raw_bytes))
    image = ImageOps.exif_transpose(image).convert("L")
    image = crop_qc_label_strip(image)
    return np.asarray(image).astype(np.float32)


def crop_qc_label_strip(image):
    arr = np.asarray(image).astype(np.float32) / 255.0
    height = arr.shape[0]
    band_h = max(12, int(height * 0.08))
    bottom = arr[-band_h:, :]
    above = arr[max(0, height - band_h * 3) : height - band_h, :]

    bottom_bright = float((bottom > 0.85).mean())
    bottom_mean = float(bottom.mean())
    above_mean = float(above.mean()) if above.size else 0.0

    if bottom_bright > 0.60 and bottom_mean > above_mean + 0.25:
        return image.crop((0, 0, image.width, height - band_h))
    return image


def array_to_display_tensor(array):
    tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
    return normalize_tensor(tensor)


def preprocess(filename, raw_bytes):
    array = load_image_from_bytes(filename, raw_bytes)
    display = array_to_display_tensor(array)
    tensor = normalize_tensor(display.clone())
    full = resize_with_padding(tensor)
    y1, y2, x1, x2 = body_center_roi_box(full)
    roi = resize_with_padding(full[:, y1:y2, x1:x2])
    return display, full, roi, (y1, y2, x1, x2)


def tensor_to_png_data_url(tensor):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="L")
    out = io.BytesIO()
    image.save(out, format="PNG")
    encoded = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def overlay_to_png_data_url(xray, cam):
    base = xray.detach().cpu().numpy()[0]
    heat = cam.detach().cpu().numpy()[0]
    base = np.clip(base, 0.0, 1.0)
    heat = np.clip(heat, 0.0, 1.0)

    rgb = np.stack([base, base, base], axis=-1)
    color = np.zeros_like(rgb)
    color[..., 0] = heat
    color[..., 1] = np.clip(1.4 * heat - 0.35, 0.0, 1.0)
    color[..., 2] = np.clip(0.7 - heat, 0.0, 1.0)
    alpha = (heat * 0.55)[..., None]
    overlay = rgb * (1 - alpha) + color * alpha
    overlay = (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)

    out = io.BytesIO()
    Image.fromarray(overlay, mode="RGB").save(out, format="PNG")
    encoded = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def compute_gradcam(xray):
    if not hasattr(runtime.xray_enc, "resnet"):
        return torch.zeros_like(xray)

    activations = {}
    gradients = {}

    def forward_hook(_module, _module_input, module_output):
        activations["value"] = module_output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = runtime.xray_enc.resnet.layer4.register_forward_hook(forward_hook)
    backward_handle = runtime.xray_enc.resnet.layer4.register_full_backward_hook(backward_hook)

    runtime.xray_enc.zero_grad(set_to_none=True)
    runtime.classifier.zero_grad(set_to_none=True)

    embedding, _ = runtime.xray_enc(xray)
    logits = runtime.classifier(embedding)
    score = logits.view(-1)[0]
    score.backward()

    acts = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=xray.shape[-2:], mode="bilinear", align_corners=False)
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)

    forward_handle.remove()
    backward_handle.remove()
    return cam.detach()


def confidence_label(probability, threshold):
    distance = abs(probability - threshold)
    if distance >= 0.35:
        return "High"
    if distance >= 0.18:
        return "Moderate"
    return "Low / borderline"


def predict_from_bytes(filename, raw_bytes):
    runtime.load()
    if not runtime.loaded:
        raise RuntimeError(runtime.error or "Model could not be loaded.")

    display, full, roi, box = preprocess(filename, raw_bytes)
    xray = roi.unsqueeze(0).to(runtime.device)

    with torch.no_grad():
        embedding, _ = runtime.xray_enc(xray)
        logits = runtime.classifier(embedding)
        probability = float(torch.sigmoid(logits.view(-1)[0]).item())

    cam = compute_gradcam(xray)[0].cpu()
    prediction = int(probability >= runtime.threshold)
    y1, y2, x1, x2 = box

    return {
        "model_status": "ready",
        "checkpoint_name": runtime.checkpoint_name,
        "probability": probability,
        "probability_text": f"{probability * 100:.1f}%",
        "threshold": runtime.threshold,
        "prediction": prediction,
        "confidence": confidence_label(probability, runtime.threshold),
        "roi_box": f"x {x1}-{x2}, y {y1}-{y2}",
        "original_image": tensor_to_png_data_url(display),
        "model_input_image": tensor_to_png_data_url(full),
        "roi_image": tensor_to_png_data_url(roi),
        "heatmap_image": overlay_to_png_data_url(roi, cam),
    }


def decode_data_url(data_url):
    if "," not in data_url:
        raise ValueError("Invalid data URL.")
    return base64.b64decode(data_url.split(",", 1)[1])


def find_sample_xray(case_name="anomaly"):
    if case_name == "healthy":
        patterns = [
            PROJECT_ROOT / "data" / "png_patients_full" / "patient_032" / "patient_032_xray.png",
            PROJECT_ROOT / "data" / "png_patients_full" / "patient_042" / "patient_042_xray.png",
            PROJECT_ROOT / "data" / "png_patients_full" / "patient_038" / "patient_038_xray.png",
        ]
    else:
        patterns = [
            PROJECT_ROOT / "data" / "png_patients_full" / "patient_002" / "patient_002_xray.png",
            PROJECT_ROOT / "data" / "png_patients_full" / "patient_004" / "patient_004_xray.png",
            PROJECT_ROOT / "data" / "png_patients_full" / "patient_028" / "patient_028_xray.png",
        ]

    patterns.append(PROJECT_ROOT / "data" / "png_patients_full" / "*" / "*_xray.png")
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if matches:
            return matches[0]
    return None


def sample_base_patient_id(sample_path):
    match = re.search(r"patient_\d+", str(sample_path))
    return match.group(0) if match else ""


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/health":
            self.send_json(runtime.health())
            return
        if path == "/api/sample":
            query = parse_qs(parsed.query)
            case_name = query.get("case", ["anomaly"])[0]
            sample = find_sample_xray(case_name)
            if not sample:
                self.send_json({"error": "No sample X-ray found."}, status=404)
                return
            try:
                with open(sample, "rb") as f:
                    payload = predict_from_bytes(os.path.basename(sample), f.read())
                base_pid = sample_base_patient_id(sample)
                held_out = base_pid in runtime.val_patients
                payload["sample_path"] = os.path.relpath(sample, PROJECT_ROOT)
                payload["sample_case"] = case_name
                payload["sample_base_patient_id"] = base_pid
                payload["held_out_from_training"] = held_out
                if held_out:
                    payload["sample_note"] = (
                        f"{base_pid} was held out from this fold model's training set."
                    )
                self.send_json(payload)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
            return
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/predict":
            self.send_json({"error": "Not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            filename = payload.get("filename") or "uploaded_xray.png"
            raw_bytes = decode_data_url(payload["data_url"])
            result = predict_from_bytes(filename, raw_bytes)
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def guess_type(self, path):
        if path.endswith(".js"):
            return "application/javascript"
        if path.endswith(".css"):
            return "text/css"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


def main():
    port = int(os.environ.get("DEMO_PORT", "8008"))
    server = ThreadingHTTPServer(("127.0.0.1", port), DemoHandler)
    print(f"Demo running at http://127.0.0.1:{port}")
    print("Open this URL in the browser. Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
