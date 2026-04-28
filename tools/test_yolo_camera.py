#!/usr/bin/env python3
"""
test_yolo_camera.py
────────────────────────────────────────────────────────────────────────────
Standalone YOLO detection test with webcam preview.

Layered false-positive filtering:
  1. Class grouping   (cake/cake 2 → CAKE, donut/donut 1 → DONUT)
  2. Per-group confidence filter  (CAKE >= 0.55, DONUT >= 0.65 by default)
  3. Per-group bbox area filter  (CAKE >= 1500px², DONUT >= 1800px² by default)
  4. Global area / aspect filters
  5. Group-level NMS (within same class group)
  6. Cross-group duplicate suppression
  7. Temporal stability (CANDIDATE → TARGET LOCKED)

Usage:
  /home/andra/envs/robot_yolo_env/bin/python tools/test_yolo_camera.py --show --device auto --roi 100,250,620,470 --cake-conf 0.55 --donut-conf 0.65 --stable-frames 5 --center-tolerance 25 --save-debug
  /home/andra/envs/robot_yolo_env/bin/python tools/test_yolo_camera.py --raw-preview --show

Arguments:
  --model             YOLO .pt weights  (default: assets/best.pt)
  --cam               webcam device index  (default: 2, external UVC camera)
  --conf              global confidence fallback  (default: 0.5)
  --cake-conf         CAKE confidence threshold  (default: 0.55)
  --donut-conf        DONUT confidence threshold  (default: 0.65)
  --imgsz             inference image size px  (default: 640)
  --show              display annotated preview window
  --save-debug        save debug_yolo_raw.jpg and debug_yolo_annotated.jpg
  --raw-preview       raw camera frame only — no YOLO
  --roi               x1,y1,x2,y2 crop for YOLO inference
  --device            YOLO device: 'auto'|'cpu'|'0'  (default: auto)
  --max-det           max YOLO detections per frame  (default: 5)
  --min-area          global min bbox area px²  (default: 800)
  --max-area          global max bbox area px²  (default: 50000)
  --cake-min-area     CAKE min bbox area px²  (default: 1500)
  --donut-min-area    DONUT min bbox area px²  (default: 1800)
  --aspect-min        min bbox aspect ratio (default: 0.4)
  --aspect-max        max bbox aspect ratio (default: 2.5)
  --group-nms-iou     IoU threshold for group-level NMS  (default: 0.35)
  --cross-nms-iou     IoU threshold for cross-group duplicate suppression (default: 0.30)
  --stable-frames     consecutive frames for TARGET LOCKED  (default: 5)
  --center-tolerance  pixel tolerance for stable center  (default: 25)
  --show-rejected     show rejected detections in red

Warning:
  Testing with objects displayed on a phone screen, held in hand, or near
  cluttered backgrounds can create false detections. Final validation must use
  physical objects on the checkerboard workspace.
"""

import argparse
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

# ── YOLO (lazy import — do not fail --help) ──────────────────────────────────
YOLO = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except ImportError:
    pass

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL          = "assets/best.pt"
DEFAULT_CAM           = 2
DEFAULT_CONF          = 0.5
DEFAULT_CAKE_CONF     = 0.55
DEFAULT_DONUT_CONF    = 0.65
DEFAULT_IMGSZ         = 640
DEFAULT_DEVICE        = "auto"
DEFAULT_MAXDET        = 5
DEFAULT_MINAREA       = 800
DEFAULT_MAXAREA       = 50000
DEFAULT_CAKE_MINAREA  = 1500
DEFAULT_DONUT_MINAREA = 1800
DEFAULT_ASPECT_MIN    = 0.4
DEFAULT_ASPECT_MAX    = 2.5
DEFAULT_NMS_IOU       = 0.35
DEFAULT_CROSS_NMS_IOU = 0.30
DEFAULT_STABLE_FRAMES = 5
DEFAULT_CENTER_TOL   = 25

# ── Fixed window ──────────────────────────────────────────────────────────────
WINDOW_NAME = "Robot Vision YOLO Debug"
WINDOW_W, WINDOW_H = 960, 720

# ── Class grouping ─────────────────────────────────────────────────────────────
CLASS_GROUPS = {
    "cake":    "CAKE",
    "cake 2":  "CAKE",
    "donut":   "DONUT",
    "donut 1": "DONUT",
}

# ── Colour palette ────────────────────────────────────────────────────────────────
# Class colours — used for BOTH the bbox border AND the label pill background
CLASS_COLORS = {
    "CAKE":    ( 40, 140, 230),   # orange  (BGR)
    "DONUT":   ( 50, 200,  80),   # green   (BGR)
    "UNKNOWN":  (  0,   0, 255),   # red
}

# Status colours — used ONLY for the status word at the end of the label text
STATUS_COLORS = {
    "LOCKED":    (220, 220,  60),   # yellow-green / lime
    "CANDIDATE": ( 80, 180, 255),   # sky blue
    "REJECTED":  (  0,   0, 255),   # red
}


# ═══════════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_device(requested: str) -> str:
    if requested == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
        except ImportError:
            pass
        return "cpu"
    return requested if requested in ("cpu", "0") else "cpu"


def print_environment(device: str) -> None:
    print(f"\n{'='*50}")
    print(f"  Environment")
    print(f"{'='*50}")
    try:
        import torch
        print(f"  torch version  : {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA available: True")
            for i in range(torch.cuda.device_count()):
                print(f"    [{i}] {torch.cuda.get_device_name(i)}")
        else:
            print(f"  CUDA available: False")
        print(f"  YOLO device   : '{device}'")
    except ImportError:
        print(f"  torch         : not installed")
        print(f"  YOLO device   : '{device}'")
    print(f"{'='*50}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Model helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str, device: str):
    global YOLO
    if YOLO is None:
        print("[ERROR] ultralytics not installed.")
        sys.exit(1)
    print(f"[INFO] Loading model: {model_path}  device={device}")
    model = YOLO(model_path)
    print(f"[INFO] Model loaded. {len(model.names)} class(es):")
    for idx, name in sorted(model.names.items()):
        grp = CLASS_GROUPS.get(name, "???")
        marker = " ← tracked" if name in CLASS_GROUPS else ""
        print(f"         [{idx:>3}] {name!r:15s} → {grp}{marker}")
    return model


def group_class(raw_name: str) -> str:
    return CLASS_GROUPS.get(raw_name, "UNKNOWN")


def parse_roi(s: str):
    try:
        parts = [int(p.strip()) for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError
        x1, y1, x2, y2 = parts
        if not (x2 > x1 and y2 > y1):
            raise ValueError("x2 must > x1, y2 must > y1")
        return x1, y1, x2, y2
    except Exception:
        raise ValueError(
            f"Invalid ROI '{s}'. Expected 'x1,y1,x2,y2'  "
            "(e.g., '100,250,620,470')."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Filtering pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def box_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def box_aspect(bbox):
    h = bbox[3] - bbox[1]
    return (bbox[2] - bbox[0]) / h if h > 0 else 0.0


def compute_iou(a, b):
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    return inter / (box_area(a) + box_area(b) - inter) if inter > 0 else 0.0


def group_level_nms(dets: list, iou_thr: float) -> list:
    """NMS per class group, keep highest-conf box of overlapping pairs."""
    groups = defaultdict(list)
    for d in dets:
        groups[d["group"]].append(d)
    result = []
    for gname, gd in groups.items():
        gd = sorted(gd, key=lambda d: d["conf"], reverse=True)
        supp = set()
        for i, a in enumerate(gd):
            if i in supp:
                continue
            for j, b in enumerate(gd[i + 1:], start=i + 1):
                if j not in supp and compute_iou(a["bbox"], b["bbox"]) > iou_thr:
                    supp.add(j)
        for i, d in enumerate(gd):
            if i not in supp:
                result.append(d)
    return result


def cross_group_nms(dets: list, iou_thr: float) -> tuple[list, list]:
    """Suppress overlapping accepted boxes across different class groups."""
    kept = []
    rejected = []
    for d in sorted(dets, key=lambda item: item["conf"], reverse=True):
        duplicate = any(
            d["group"] != k["group"] and compute_iou(d["bbox"], k["bbox"]) >= iou_thr
            for k in kept
        )
        if duplicate:
            rejected.append({**d, "reject_reason": "cross_nms"})
        else:
            kept.append(d)
    return kept, rejected


def filter_detections(raw_dets: list,
                      conf_thr: float,
                      cake_conf: float, donut_conf: float,
                      min_area: int, max_area: int,
                      cake_min_area: int, donut_min_area: int,
                      aspect_min: float, aspect_max: float) -> list:
    """
    Per-group filtering.  Each detection gets a 'reject_reason' (None = accepted).

    Filters applied in order:
      1. unknown class         → bad_class
      2. below group conf      → low_conf
      3. below group min area  → bad_area
      4. outside global area   → bad_area
      5. bad aspect ratio      → bad_aspect
    """
    result = []
    for d in raw_dets:
        grp   = d["group"]
        conf  = d["conf"]
        area  = box_area(d["bbox"])
        reason = None

        if grp == "UNKNOWN":
            reason = "bad_class"
        elif grp == "CAKE" and conf < cake_conf:
            reason = "low_conf"
        elif grp == "DONUT" and conf < donut_conf:
            reason = "low_conf"
        elif conf < conf_thr:          # global fallback (should rarely trigger)
            reason = "low_conf"
        elif grp == "CAKE" and area < cake_min_area:
            reason = "bad_area"
        elif grp == "DONUT" and area < donut_min_area:
            reason = "bad_area"
        elif area < min_area or area > max_area:
            reason = "bad_area"
        else:
            ar = box_aspect(d["bbox"])
            if ar < aspect_min or ar > aspect_max:
                reason = "bad_aspect"

        result.append({**d, "reject_reason": reason})
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal stability tracker
# ═══════════════════════════════════════════════════════════════════════════════

class StabilityTracker:
    """
    Tracks detections per class group across frames.
    LOCKED: same group confirmed for >= stable_frames consecutive frames,
            and center within center_tol pixels of median.
    CANDIDATE: not yet stable.
    """
    def __init__(self, stable_frames: int = 3, center_tol: int = 45):
        self.stable_frames = stable_frames
        self.center_tol   = center_tol
        self.history: dict[str, list] = defaultdict(list)

    def update(self, accepted: list) -> list:
        # current centres
        current = {d["group"]: (d["u"], d["v"]) for d in accepted}
        # drop stale groups
        for g in list(self.history):
            if g not in current:
                del self.history[g]
        result = []
        for d in accepted:
            g  = d["group"]
            ct = (d["u"], d["v"])
            self.history[g].append(ct)
            self.history[g] = self.history[g][-self.stable_frames:]
            h = self.history[g]
            if len(h) < self.stable_frames:
                d["status"] = "CANDIDATE"
            else:
                mid = self.stable_frames // 2
                mu  = sorted(u for u, v in h)[mid]
                mv  = sorted(v for u, v in h)[mid]
                dist = ((ct[0] - mu) ** 2 + (ct[1] - mv) ** 2) ** 0.5
                d["status"] = "LOCKED" if dist <= self.center_tol else "CANDIDATE"
            result.append(d)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# YOLO runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_yolo(model, frame: np.ndarray, roi: tuple | None,
             conf_thr: float, imgsz: int, max_det: int, device: str) -> list:
    if roi:
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return []
        res = model(crop, conf=conf_thr, imgsz=imgsz, verbose=False,
                    device=device, max_det=max_det)[0]
    else:
        res = model(frame, conf=conf_thr, imgsz=imgsz, verbose=False,
                    device=device, max_det=max_det)[0]
    dets = []
    for box in res.boxes:
        rn  = res.names[int(box.cls)]
        c1x, c1y, c2x, c2y = map(int, box.xyxy[0])
        if roi:
            fx1, fy1 = c1x + x1, c1y + y1
            fx2, fy2 = c2x + x1, c2y + y1
        else:
            fx1, fy1, fx2, fy2 = c1x, c1y, c2x, c2y
        dets.append({
            "raw":   rn,
            "group": group_class(rn),
            "conf":  float(box.conf),
            "u":     int((fx1 + fx2) / 2),
            "v":     int((fy1 + fy2) / 2),
            "bbox":  (fx1, fy1, fx2, fy2),
        })
    return dets


# ═══════════════════════════════════════════════════════════════════════════════
# Overlay renderer
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_overlay(frame: np.ndarray,
                 accepted: list, rejected: list,
                 fps: float, n_raw: int,
                 show_rejected: bool) -> np.ndarray:
    """
    Draw on a copy of the camera frame.

    Per-detection:
      - Bbox border  → CLASS colour (CAKE=orange, DONUT=green)
      - Label pill    → CLASS colour
      - Label text     → white, ends with status word in STATUS colour
      - Centre dot     → CLASS colour (filled) + white ring

    HUD bar: FPS | RAW count | ACC count | REJ count
    Legend (bottom-right): colour swatches for class + status colours
    """
    ann = frame.copy()
    H, W = frame.shape[:2]

    # ── HUD bar ────────────────────────────────────────────────────
    bar_h = 32
    cv2.rectangle(ann, (0, 0), (W, bar_h), (12, 12, 12), -1)
    cv2.putText(ann,
                f"FPS:{fps:4.1f}  RAW:{n_raw}  ACC:{len(accepted)}  REJ:{len(rejected)}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

    # ── Per-detection overlays ─────────────────────────────────────
    def _draw(det: dict, rejected: bool):
        x1, y1, x2, y2 = det["bbox"]
        u, v = det["u"], det["v"]
        grp   = det["group"]
        conf  = det["conf"]
        raw   = det["raw"]
        reason = det.get("reject_reason")
        status = det.get("status", "")

        cls_col = CLASS_COLORS.get(grp, CLASS_COLORS["UNKNOWN"])

        if rejected:
            label = f"REJECTED: {reason or '?'}"
            status_col = STATUS_COLORS["REJECTED"]
        else:
            label = f"{grp} | {raw} | {conf:.2f} | u={u} v={v} | {status}"
            status_col = STATUS_COLORS.get(status, STATUS_COLORS["CANDIDATE"])

        # ── Bbox border — class colour ────────────────────────────
        cv2.rectangle(ann, (x1, y1), (x2, y2), cls_col, 2)

        # ── Label pill — class colour background ──────────────────
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
        bg_y1 = max(0, y1 - lh - 9)
        cv2.rectangle(ann, (x1, bg_y1), (x1 + lw + 8, y1), cls_col, -1)

        # ── Label text — white body ───────────────────────────────
        cv2.putText(ann, label, (x1 + 4, y1 - 3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1)

        # ── Status word — coloured overlay on the status text ──────
        # Draw a small coloured rect behind the status word
        if not rejected:
            # find the status word position within the label
            # label ends with " | {status}"
            stat_prefix = f" | {status}"
            prefix_w = cv2.getTextSize(
                label[:len(label) - len(stat_prefix) + 2],
                cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)[0][0]
            sw, sh = cv2.getTextSize(status,
                                    cv2.FONT_HERSHEY_DUPLEX, 0.48, 2)[0]
            sx = x1 + 4 + prefix_w
            sy1 = bg_y1 + 1
            sy2 = y1 - 2
            # draw status coloured stripe
            cv2.rectangle(ann, (sx - 2, sy1), (sx + sw + 2, sy2), status_col, -1)
            # redraw full label text in white
            cv2.putText(ann, label, (x1 + 4, y1 - 3),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1)

        # ── Centre dot — class colour ───────────────────────────────
        cv2.circle(ann, (u, v), 5, cls_col, -1)
        cv2.circle(ann, (u, v), 5, (255, 255, 255), 1)

    for d in accepted:
        _draw(d, rejected=False)
    if show_rejected:
        for d in rejected:
            _draw(d, rejected=True)

    # ── NO TARGET ─────────────────────────────────────────────────
    if not accepted:
        (lw, lh), _ = cv2.getTextSize("NO TARGET",
                                      cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)
        tx = (W - lw) // 2
        ty = (H + lh) // 2
        cv2.putText(ann, "NO TARGET", (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 3)

    # ── Legend bottom-right ──────────────────────────────────────
    legend = [
        ("CAKE",   CLASS_COLORS["CAKE"]),
        ("DONUT",  CLASS_COLORS["DONUT"]),
        ("LOCKED",  STATUS_COLORS["LOCKED"]),
        ("CANDIDATE", STATUS_COLORS["CANDIDATE"]),
    ]
    if show_rejected:
        legend.append(("REJECTED", STATUS_COLORS["REJECTED"]))
    lw_max = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
                  for l, _ in legend)
    for i, (lbl, col) in enumerate(reversed(legend)):
        lx = W - lw_max - 24
        ly = H - 14 - i * (22)
        cv2.rectangle(ann, (lx - 4, ly - 14), (lx + lw_max + 4, ly + 4), col, -1)
        cv2.putText(ann, lbl, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    return ann


# ═══════════════════════════════════════════════════════════════════════════════
# Camera loops
# ═══════════════════════════════════════════════════════════════════════════════

def run_raw_preview(cam: int) -> None:
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
    print(f"[INFO] Camera {cam} opened ({W}×{H})  RAW PREVIEW  |  Press 'q'")
    fps_s = 0.0
    t_prev = time.time()
    fc = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Cannot read frame")
            time.sleep(0.5)
            continue
        fc += 1
        t = time.time()
        fps_s = 0.9 * fps_s + 0.1 / max(t - t_prev, 1e-6)
        t_prev = t
        ann = frame.copy()
        cv2.rectangle(ann, (0, 0), (W, 28), (10, 10, 10), -1)
        cv2.putText(ann, f"RAW PREVIEW  FPS:{fps_s:.1f}", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, ann)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] {fc} frames processed.")


def run_detection(model, cam: int,
                 conf_thr: float,
                 cake_conf: float, donut_conf: float,
                 cake_min_area: int, donut_min_area: int,
                 imgsz: int, roi: tuple | None,
                 save_debug: bool,
                 max_det: int,
                 min_area: int, max_area: int,
                 aspect_min: float, aspect_max: float,
                 nms_iou: float,
                 cross_nms_iou: float,
                 stable_frames: int, center_tol: int,
                 show_rejected: bool,
                 device: str) -> None:
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
    tracker = StabilityTracker(stable_frames=stable_frames,
                              center_tol=center_tol)

    print(f"[INFO] Camera {cam} opened ({W}×{H})")
    print(f"[INFO] Filters:")
    print(f"  conf_thr={conf_thr}  "
          f"cake_conf>={cake_conf}  donut_conf>={donut_conf}")
    print(f"  area=[{min_area},{max_area}]  "
          f"cake_min={cake_min_area}  donut_min={donut_min_area}")
    print(f"  aspect=[{aspect_min},{aspect_max}]  "
          f"nms_iou={nms_iou}  cross_nms_iou={cross_nms_iou}")
    print(f"  stable_frames={stable_frames}  center_tol={center_tol}px")
    if roi:
        print(f"  ROI: {roi}")
    else:
        print(f"  ROI: none (full-frame)")
        print("  WARNING: Full-frame detection is for debug only and may produce false positives. Use checkerboard ROI.")
    print(f"  show_rejected={show_rejected}")
    print()
    print("  FRAME  STAT   GROUP   RAW              CONF       u       v  AREA  REASON")
    print("  ------ ------ ------- --------------- -------- ------- ------- -----  ------")
    print("[INFO] Press 'q' to quit\n")

    fps_s = 0.0
    t_prev = time.time()
    fc = 0
    saved = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Cannot read frame")
            time.sleep(0.5)
            continue

        t = time.time()
        fc += 1

        # Step 1 — YOLO
        raw = run_yolo(model, frame, roi, conf_thr, imgsz, max_det, device)

        # Step 2 — per-group filters
        filt = filter_detections(
            raw,
            conf_thr=conf_thr,
            cake_conf=cake_conf, donut_conf=donut_conf,
            min_area=min_area, max_area=max_area,
            cake_min_area=cake_min_area, donut_min_area=donut_min_area,
            aspect_min=aspect_min, aspect_max=aspect_max,
        )
        acc = [d for d in filt if d["reject_reason"] is None]
        rej = [d for d in filt if d["reject_reason"] is not None]

        # Step 3 — group-level NMS
        acc = group_level_nms(acc, nms_iou)

        # Step 4 — cross-group duplicate suppression
        acc, cross_rej = cross_group_nms(acc, cross_nms_iou)
        rej.extend(cross_rej)

        # Step 5 — temporal stability
        acc = tracker.update(acc)

        fps_s = 0.9 * fps_s + 0.1 / max(t - t_prev, 1e-6)
        t_prev = t

        # ── Console log ────────────────────────────────────────
        for d in acc:
            area = box_area(d["bbox"])
            print(
                f"  {fc:06d}  {d['status']:<6}  "
                f"{d['group']:<7} {d['raw']:<15} "
                f"conf={d['conf']:.3f}  "
                f"u={d['u']:>5}  v={d['v']:>5}  "
                f"area={area:>5}"
            )
        for d in rej:
            area = box_area(d["bbox"])
            print(
                f"  {fc:06d}  REJ     {d['group']:<7} {d['raw']:<15} "
                f"conf={d['conf']:.3f}  "
                f"u={d['u']:>5}  v={d['v']:>5}  "
                f"area={area:>5}  {d['reject_reason']}"
            )
        if not raw:
            print(f"  {fc:06d}  (no YOLO detections)")

        # ── Debug image save on first frame ────────────────────
        if save_debug and not saved:
            cv2.imwrite("debug_yolo_raw.jpg", frame)
            ann_s = _draw_overlay(frame, acc, rej, fps_s, len(raw), show_rejected)
            cv2.imwrite("debug_yolo_annotated.jpg", ann_s)
            print(f"[INFO] Saved debug_yolo_raw.jpg and debug_yolo_annotated.jpg")
            saved = True

        # ── Show ──────────────────────────────────────────────
        ann = _draw_overlay(frame, acc, rej, fps_s, len(raw), show_rejected)
        cv2.imshow(WINDOW_NAME, ann)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Session: {fc} frames.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    epilog = (
        "examples:\n"
        "  /home/andra/envs/robot_yolo_env/bin/python tools/test_yolo_camera.py "
        "--show --device auto --roi 100,250,620,470 "
        "--cake-conf 0.55 --donut-conf 0.65 --stable-frames 5 --center-tolerance 25\n"
        "  /home/andra/envs/robot_yolo_env/bin/python tools/test_yolo_camera.py --raw-preview --show\n"
        "\nwarning:\n"
        "  Phone screens, objects held in hand, and cluttered backgrounds can create false detections.\n"
        "  Final validation must use physical objects on the checkerboard workspace.\n"
    )
    p = argparse.ArgumentParser(
        description="Standalone YOLO detection with layered false-positive filtering.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=epilog,
    )
    p.add_argument("--model")
    p.add_argument("--cam",            type=int)
    p.add_argument("--conf",           type=float)
    p.add_argument("--cake-conf",      type=float)
    p.add_argument("--donut-conf",     type=float)
    p.add_argument("--imgsz",          type=int)
    p.add_argument("--show",           action="store_true")
    p.add_argument("--save-debug",    action="store_true")
    p.add_argument("--raw-preview",    action="store_true")
    p.add_argument("--roi",           type=str)
    p.add_argument("--device",         default=DEFAULT_DEVICE, choices=["auto","cpu","0"])
    p.add_argument("--max-det",        type=int)
    p.add_argument("--min-area",      type=int)
    p.add_argument("--max-area",      type=int)
    p.add_argument("--cake-min-area", type=int)
    p.add_argument("--donut-min-area", type=int)
    p.add_argument("--aspect-min",     type=float)
    p.add_argument("--aspect-max",    type=float)
    p.add_argument("--group-nms-iou", type=float)
    p.add_argument("--cross-nms-iou", type=float)
    p.add_argument("--stable-frames",  type=int)
    p.add_argument("--center-tolerance", type=int)
    p.add_argument("--show-rejected", action="store_true")

    args = p.parse_args()

    # Fill defaults
    args.model          = args.model          or DEFAULT_MODEL
    args.cam            = args.cam            or DEFAULT_CAM
    args.conf           = args.conf           or DEFAULT_CONF
    args.cake_conf      = args.cake_conf      or DEFAULT_CAKE_CONF
    args.donut_conf     = args.donut_conf     or DEFAULT_DONUT_CONF
    args.imgsz          = args.imgsz          or DEFAULT_IMGSZ
    args.max_det        = args.max_det        or DEFAULT_MAXDET
    args.min_area       = args.min_area       or DEFAULT_MINAREA
    args.max_area       = args.max_area       or DEFAULT_MAXAREA
    args.cake_min_area  = args.cake_min_area  or DEFAULT_CAKE_MINAREA
    args.donut_min_area  = args.donut_min_area or DEFAULT_DONUT_MINAREA
    args.aspect_min     = args.aspect_min     or DEFAULT_ASPECT_MIN
    args.aspect_max     = args.aspect_max     or DEFAULT_ASPECT_MAX
    args.group_nms_iou  = args.group_nms_iou  or DEFAULT_NMS_IOU
    args.cross_nms_iou  = args.cross_nms_iou  or DEFAULT_CROSS_NMS_IOU
    args.stable_frames  = args.stable_frames  or DEFAULT_STABLE_FRAMES
    args.center_tolerance = args.center_tolerance or DEFAULT_CENTER_TOL

    args.conf = max(0.0, min(1.0, args.conf))

    device = resolve_device(args.device)
    print_environment(device)

    print(f"[INFO] Active thresholds:")
    print(f"  cake_conf >= {args.cake_conf}  donut_conf >= {args.donut_conf}  "
          f"(global fallback conf >= {args.conf})")
    print(f"  cake_min_area >= {args.cake_min_area}  "
          f"donut_min_area >= {args.donut_min_area}  "
          f"(global area [{args.min_area},{args.max_area}])")
    print(f"  aspect in [{args.aspect_min}, {args.aspect_max}]  "
          f"nms_iou = {args.group_nms_iou}  "
          f"cross_nms_iou = {args.cross_nms_iou}")
    print(f"  stable_frames = {args.stable_frames}  "
          f"center_tolerance = {args.center_tolerance}px")
    print(f"  show_rejected = {args.show_rejected}")
    print()

    if args.roi:
        try:
            args.roi = parse_roi(args.roi)
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    else:
        print("WARNING: Full-frame detection is for debug only and may produce false positives. Use checkerboard ROI.")

    if args.raw_preview:
        run_raw_preview(args.cam)
        return

    model = load_model(args.model, device)
    run_detection(
        model, args.cam,
        conf_thr=args.conf,
        cake_conf=args.cake_conf, donut_conf=args.donut_conf,
        cake_min_area=args.cake_min_area, donut_min_area=args.donut_min_area,
        imgsz=args.imgsz, roi=args.roi,
        save_debug=args.save_debug,
        max_det=args.max_det,
        min_area=args.min_area, max_area=args.max_area,
        aspect_min=args.aspect_min, aspect_max=args.aspect_max,
        nms_iou=args.group_nms_iou,
        cross_nms_iou=args.cross_nms_iou,
        stable_frames=args.stable_frames,
        center_tol=args.center_tolerance,
        show_rejected=args.show_rejected,
        device=device,
    )


if __name__ == "__main__":
    main()
