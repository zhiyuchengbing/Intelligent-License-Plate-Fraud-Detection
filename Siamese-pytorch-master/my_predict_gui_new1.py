import os
import sys
import threading
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
from PIL import Image
from flask import Flask, jsonify, request
from ultralytics import YOLO

from siamese import Siamese

parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_chuli.cropper import VehicleCropper


app = Flask(__name__)

_INIT_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_INITIALIZED = False

_CROPPER: Optional[VehicleCropper] = None
_HEAD_MODEL: Optional[Siamese] = None
_TAIL_MODEL: Optional[Siamese] = None
_HEADTAIL_MODEL: Optional[YOLO] = None


def _get_allowed_base_dirs() -> Tuple[str, ...]:
    raw = os.environ.get("ALLOWED_BASE_DIRS", "").strip()
    if not raw:
        return tuple()
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    return tuple(os.path.abspath(p) for p in parts)


def _is_path_allowed(path: str) -> bool:
    allowed = _get_allowed_base_dirs()
    if not allowed:
        return True
    try:
        abs_path = os.path.abspath(path)
        for base in allowed:
            if os.path.commonpath([abs_path, base]) == base:
                return True
        return False
    except Exception:
        return False


def _validate_image_path(p: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(p, str) or not p.strip():
        return False, "path must be a non-empty string"
    abs_path = os.path.abspath(p)
    if not os.path.isabs(abs_path):
        return False, "path must be absolute"
    if not _is_path_allowed(abs_path):
        return False, "path not allowed"
    if not os.path.exists(abs_path):
        return False, "file not found"
    if not os.path.isfile(abs_path):
        return False, "path is not a file"
    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return False, "unsupported file extension"
    return True, abs_path


def _init_models() -> None:
    global _INITIALIZED, _CROPPER, _HEAD_MODEL, _TAIL_MODEL, _HEADTAIL_MODEL
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return

        head_model_path = os.environ.get(
            "HEAD_MODEL_PATH",
            r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\logs\head\1211\best_epoch_weights.pth",
        )
        tail_model_path = os.environ.get(
            "TAIL_MODEL_PATH",
            r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\logs\weibu\1211\best_epoch_weights.pth",
        )
        headtail_model_path = os.environ.get(
            "HEADTAIL_MODEL_PATH",
            r"D:\data2\runs\detect\train\weights\best.pt",
        )

        _CROPPER = VehicleCropper()
        _HEAD_MODEL = Siamese(model_path=head_model_path)
        _TAIL_MODEL = Siamese(model_path=tail_model_path)
        _HEADTAIL_MODEL = YOLO(headtail_model_path)

        _INITIALIZED = True


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _crop_part_from_vehicle_pil(vehicle_image: Image.Image, cls_id: int) -> Image.Image:
    try:
        if vehicle_image is None:
            return vehicle_image
        if _HEADTAIL_MODEL is None:
            return vehicle_image

        bgr = _pil_to_bgr(vehicle_image)
        results = _HEADTAIL_MODEL(bgr, conf=0.25, verbose=False)
        if not results:
            return vehicle_image
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return vehicle_image

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        best_idx = None
        best_score = -1.0
        for i, (c, s) in enumerate(zip(classes, scores)):
            if int(c) != int(cls_id):
                continue
            if float(s) > best_score:
                best_score = float(s)
                best_idx = i

        if best_idx is None:
            return vehicle_image

        x1, y1, x2, y2 = boxes[int(best_idx)]
        h, w = bgr.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return vehicle_image

        crop = bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return vehicle_image
        return _bgr_to_pil(crop)
    except Exception:
        return vehicle_image


def _compute_head_tail_probs(path1: str, path2: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    try:
        _init_models()
        if _CROPPER is None or _HEAD_MODEL is None or _TAIL_MODEL is None:
            return None, None, "models not initialized"

        img1 = Image.open(path1)
        img2 = Image.open(path2)

        img1 = _CROPPER.process_pil(img1)
        img2 = _CROPPER.process_pil(img2)

        head1 = _crop_part_from_vehicle_pil(img1, cls_id=0)
        head2 = _crop_part_from_vehicle_pil(img2, cls_id=0)
        tail1 = _crop_part_from_vehicle_pil(img1, cls_id=1)
        tail2 = _crop_part_from_vehicle_pil(img2, cls_id=1)

        with _INFER_LOCK:
            head_prob = _HEAD_MODEL.detect_image(head1, head2)
            tail_prob = _TAIL_MODEL.detect_image(tail1, tail2)

        if hasattr(head_prob, "item"):
            head_prob = head_prob.item()
        if hasattr(tail_prob, "item"):
            tail_prob = tail_prob.item()

        return float(head_prob), float(tail_prob), None
    except Exception as e:
        return None, None, str(e)


def _classify_case(head_prob: Optional[float], tail_prob: Optional[float]) -> str:
    if head_prob is None or tail_prob is None:
        return "abnormal"

    head_low_th = float(os.environ.get("HEAD_LOW_TH", "0.8"))
    head_same_th = float(os.environ.get("HEAD_SAME_TH", "0.3"))
    tail_low_th = float(os.environ.get("TAIL_LOW_TH", "0.3"))

    if head_prob < head_low_th:
        return "fake_plate"
    if head_prob > head_same_th and tail_prob <= tail_low_th:
        return "change_trailer"
    return "normal"


@app.get("/")
def index() -> Any:
    return jsonify({"endpoints": {"health": "/health", "predict": "/predict"}})


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict() -> Any:
    payload = request.get_json(silent=True) or {}
    ok1, p1 = _validate_image_path(payload.get("path1"))
    ok2, p2 = _validate_image_path(payload.get("path2"))
    if not ok1:
        return jsonify({"ok": False, "error": f"path1 invalid: {p1}"}), 400
    if not ok2:
        return jsonify({"ok": False, "error": f"path2 invalid: {p2}"}), 400

    head_prob, tail_prob, err = _compute_head_tail_probs(p1, p2)
    case_type = _classify_case(head_prob, tail_prob)

    resp: Dict[str, Any] = {
        "ok": case_type != "abnormal",
        "case_type": case_type,
        "head_prob": head_prob,
        "tail_prob": tail_prob,
    }
    if err:
        resp["error"] = err
    return jsonify(resp)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    app.run(host=host, port=port, threaded=True)
