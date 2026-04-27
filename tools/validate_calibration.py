#!/usr/bin/env python3
"""
validate_calibration.py
────────────────────────────────────────────────────────────────────────────
Load a camera calibration .npz file (OpenCV format) and verify its contents.
Can also display a live webcam feed comparing original vs. undistorted frames.

Usage:
  python tools/validate_calibration.py
  python tools/validate_calibration.py --calib assets/kacamata_kamera.npz
  python tools/validate_calibration.py --show --cam 0 --width 640 --height 480
  python tools/validate_calibration.py --raw-only --show
  python tools/validate_calibration.py --save-debug

Arguments:
  --calib       path to .npz calibration file  (default: assets/kacamata_kamera.npz)
  --cam         webcam device index             (default: 0)
  --width       requested frame width          (default: 640)
  --height      requested frame height          (default: 480)
  --show        open webcam and show preview
  --raw-only    show original frame only (no undistorted side-by-side)
  --save-debug  save debug_original.jpg and debug_undistorted.jpg to current dir
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

DEFAULT_CALIB = "assets/kacamata_kamera.npz"
DEFAULT_CAM   = 0
DEFAULT_W     = 640
DEFAULT_H     = 480

EXPECTED_NPZ_KEYS = {"mtx", "dist"}


# ═══════════════════════════════════════════════════════════════════════════
# Calibration loading & validation
# ═══════════════════════════════════════════════════════════════════════════

def load_calibration(path: str) -> dict:
    """
    Load and validate a camera calibration .npz file.

    Expected contents:
      mtx  : 3×3 camera intrinsic matrix  (float64)
      dist : distortion coefficients vector (float64, ≥4 elements)

    Returns dict with keys 'mtx' and 'dist'.
    Raises FileNotFoundError, ValueError on issues.
    """
    if not path.endswith(".npz"):
        print(f"[WARN] File does not end with .npz: '{path}'")

    data = np.load(path)

    missing = EXPECTED_NPZ_KEYS - set(data.keys())
    if missing:
        raise ValueError(
            f"Calibration file is missing expected keys: {missing}\n"
            f"  Found keys: {list(data.keys())}"
        )

    mtx  = data["mtx"]
    dist = data["dist"]

    # Validate mtx shape
    if mtx.shape != (3, 3):
        raise ValueError(
            f"Camera matrix 'mtx' has wrong shape {mtx.shape}, expected (3, 3)"
        )

    # Validate dist shape (must be at least 4 coefficients)
    if dist.size < 4:
        raise ValueError(
            f"Distortion vector 'dist' has too few elements ({dist.size}), expected ≥4"
        )

    return {"mtx": mtx, "dist": dist, "path": path}


def print_calibration(calib: dict) -> None:
    """Pretty-print calibration parameters to the terminal."""
    mtx  = calib["mtx"]
    dist = calib["dist"]

    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]

    print(f"\n{'='*60}")
    print(f"  Camera Calibration Report")
    print(f"  File: {calib['path']}")
    print(f"{'='*60}")
    print(f"\n  Camera Intrinsic Matrix (mtx)  [3×3]")
    print(f"    fx (focal x)     = {fx:12.4f}")
    print(f"    fy (focal y)     = {fy:12.4f}")
    print(f"    cx (principal x) = {cx:12.4f}")
    print(f"    cy (principal y) = {cy:12.4f}")
    print(f"\n  Principal point: ({cx:.2f}, {cy:.2f})")
    print(f"\n  Distortion Coefficients (dist)  [{dist.size} values]")
    dist_names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
                  "s1", "s2", "s3", "s4"]
    for i, val in enumerate(dist.flat):
        name = dist_names[i] if i < len(dist_names) else f"??[{i}]"
        print(f"    {name:>4} = {val:+.6f}")
    print(f"{'='*60}\n")


def check_principal_point(mtx: np.ndarray, frame_h: int, frame_w: int) -> None:
    """
    Check if the principal point (cx, cy) is inside the image frame.
    Warn if it looks unusual — especially if outside frame bounds.
    """
    cx, cy = mtx[0, 2], mtx[1, 2]
    center_x, center_y = frame_w / 2, frame_h / 2

    dist_from_center = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
    # Warn if > 10% of the smaller frame dimension from centre, or outside frame
    warn_threshold = min(frame_w, frame_h) * 0.10

    print(f"\n[CALIB] Principal point check:")
    print(f"        Frame centre : ({center_x:.1f}, {center_y:.1f})")
    print(f"        Principal pt: ({cx:.2f}, {cy:.2f})")
    print(f"        Distance    : {dist_from_center:.2f} px")
    print(f"        Warn threshold: {warn_threshold:.1f} px")

    if cx < 0 or cx > frame_w or cy < 0 or cy > frame_h:
        print(f"[WARN]  Principal point is OUTSIDE the image frame!")
        print(f"        cy={cy:.2f} is outside frame height range [0, {frame_h}]")
        print(f"        This may indicate a calibration issue or wide-angle lens.")
    elif dist_from_center > warn_threshold:
        print(f"[WARN]  Principal point is far from frame centre (> 10% of frame).")
        print(f"        This may indicate a calibration issue or wide-angle lens.")
    else:
        print(f"[OK]    Principal point is within expected range.")


# ═══════════════════════════════════════════════════════════════════════════
# Undistortion helpers
# ═══════════════════════════════════════════════════════════════════════════

def undistort_frame(frame: np.ndarray, mtx: np.ndarray,
                    dist: np.ndarray) -> np.ndarray:
    """Return an undistorted (rectified) copy of the input frame."""
    h, w = frame.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


def _annotate_frame(frame: np.ndarray, label: str) -> np.ndarray:
    """Add a bold label bar at the top of a frame copy."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Dark overlay bar at top
    bar_h = max(30, int(h * 0.06))
    cv2.rectangle(annotated, (0, 0), (w, bar_h), (20, 20, 20), -1)

    # Label text centred in bar
    font_scale = max(0.5, bar_h / 45.0)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    text_x = max(8, (w - lw) // 2)
    text_y = int((bar_h + lh) / 2) - 2
    cv2.putText(annotated, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    # Thin border around the frame
    cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (80, 80, 80), 1)
    return annotated


def _resize_same_height(frames: list, target_h: int) -> list:
    """
    Resize each frame in `frames` to target_h (preserving aspect ratio).
    All frames will end up with the same height, different widths.
    Returns list of resized numpy arrays.
    """
    resized = []
    for f in frames:
        if f.shape[0] != target_h:
            scale = target_h / f.shape[0]
            new_w = int(f.shape[1] * scale)
            f = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(f)
    return resized


# ═══════════════════════════════════════════════════════════════════════════
# Preview loop
# ═══════════════════════════════════════════════════════════════════════════

def run_preview(calib: dict, camera_id: int,
                frame_w: int, frame_h: int,
                raw_only: bool = False,
                save_debug: bool = False) -> None:
    """
    Open webcam and display preview.

    If raw_only=False: show original + undistorted side-by-side,
                       both resized to the same height before stacking.
    If raw_only=True:  show original frame only.

    If save_debug=True: save debug_original.jpg and debug_undistorted.jpg
                        after the first successful frame.
    Press 'q' to quit.
    """
    mtx, dist = calib["mtx"], calib["dist"]

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera {camera_id} opened ({actual_w}×{actual_h})")
    print(f"[INFO] Press 'q' to quit")

    fps_smooth = 0.0
    t_prev = time.time()
    debug_saved = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Cannot read frame, retrying…")
            time.sleep(0.5)
            continue

        undist = undistort_frame(frame, mtx, dist)

        # FPS
        t_now = time.time()
        fps_smooth = 0.9 * fps_smooth + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # Save debug images on first frame
        if save_debug and not debug_saved:
            cv2.imwrite("debug_original.jpg", frame)
            cv2.imwrite("debug_undistorted.jpg", undist)
            print(f"[INFO] Saved debug_original.jpg and debug_undistorted.jpg")
            debug_saved = True

        # Annotate with labels
        if raw_only:
            labeled = _annotate_frame(frame, "ORIGINAL")
            # FPS in corner
            cv2.putText(labeled, f"FPS: {fps_smooth:.1f}", (8, labeled.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            window_label = "validate_calibration.py — ORIGINAL (raw-only)"
            cv2.imshow(window_label, labeled)
        else:
            labeled_orig   = _annotate_frame(frame,  "ORIGINAL")
            labeled_undist = _annotate_frame(undist, "UNDISTORTED")

            # Resize both to same height before stacking
            target_h = labeled_orig.shape[0]   # use ORIGINAL height
            r_orig, r_undist = _resize_same_height([labeled_orig, labeled_undist], target_h)

            # Stack horizontally
            combined = np.hstack([r_orig, r_undist])

            # FPS bar at the very bottom
            bar_y = combined.shape[0] - 24
            cv2.rectangle(combined, (0, bar_y), (combined.shape[1], combined.shape[0]),
                           (20, 20, 20), -1)
            cv2.putText(combined, f"FPS: {fps_smooth:.1f}", (8, combined.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            window_label = "validate_calibration.py — ORIGINAL | UNDISTORTED"
            cv2.imshow(window_label, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Preview closed.")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate camera calibration .npz and optionally preview undistortion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python tools/validate_calibration.py\n"
            "  python tools/validate_calibration.py --calib assets/kacamata_kamera.npz\n"
            "  python tools/validate_calibration.py --show --cam 0 --width 640 --height 480\n"
            "  python tools/validate_calibration.py --raw-only --show\n"
            "  python tools/validate_calibration.py --save-debug\n"
        ),
    )
    parser.add_argument(
        "--calib", default=DEFAULT_CALIB,
        help=f"Path to .npz calibration file (default: {DEFAULT_CALIB})"
    )
    parser.add_argument(
        "--cam", type=int, default=DEFAULT_CAM,
        help=f"Webcam device index (default: {DEFAULT_CAM})"
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_W,
        help=f"Requested frame width (default: {DEFAULT_W})"
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_H,
        help=f"Requested frame height (default: {DEFAULT_H})"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open webcam and show preview"
    )
    parser.add_argument(
        "--raw-only", action="store_true",
        help="Show original frame only (no undistorted side-by-side)"
    )
    parser.add_argument(
        "--save-debug", action="store_true",
        help="Save debug_original.jpg and debug_undistorted.jpg to current directory"
    )

    args = parser.parse_args()

    # Load calibration
    try:
        calib = load_calibration(args.calib)
    except FileNotFoundError:
        print(f"[ERROR] Calibration file not found: {args.calib}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"[OK] Calibration loaded from: {args.calib}")
    print_calibration(calib)

    # Principal-point sanity check
    check_principal_point(calib["mtx"], args.height, args.width)

    # Optional preview
    if args.show or args.raw_only:
        if args.raw_only:
            print(f"\n[INFO] Starting raw-only preview (press 'q' to quit)…")
        else:
            print(f"\n[INFO] Starting side-by-side preview (press 'q' to quit)…")
        run_preview(calib, args.cam, args.width, args.height,
                    raw_only=args.raw_only, save_debug=args.save_debug)
    elif args.save_debug:
        # --save-debug implies --show behaviour to capture a frame
        print(f"\n[INFO] Capturing frame for debug images…")
        run_preview(calib, args.cam, args.width, args.height,
                    raw_only=False, save_debug=True)


if __name__ == "__main__":
    main()