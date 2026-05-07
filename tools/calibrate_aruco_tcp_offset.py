#!/usr/bin/env python3
"""
calibrate_aruco_tcp_offset.py

Estimate marker0-to-TCP board-coordinate offset from repeated observations at a
known board target. Camera-only diagnostic; no motion or serial access.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

import cv2
import numpy as np

from aruco_utils import (
    DEFAULT_ARUCO_CONFIG,
    DEFAULT_CAMERA_ID,
    DEFAULT_HEIGHT,
    DEFAULT_HOMOGRAPHY,
    DEFAULT_WIDTH,
    WINDOW_H,
    WINDOW_W,
    add_board_coordinates,
    build_detector,
    detect_markers,
    draw_marker_overlay,
    find_record,
    get_aruco_dict_name,
    get_predefined_dictionary,
    load_aruco_config,
    marker_records,
    require_exact,
    resolve_existing_path,
    save_yaml,
    utc_now_iso,
)


WINDOW_NAME = "ArUco TCP Offset Calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate ArUco gripper marker center to TCP offset in board cm.",
    )
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_ID)
    parser.add_argument("--aruco-id", type=int, default=0)
    parser.add_argument("--tcp-board-x-cm", type=float, required=True)
    parser.add_argument("--tcp-board-y-cm", type=float, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--homography", default=DEFAULT_HOMOGRAPHY)
    parser.add_argument("--aruco-config", default=DEFAULT_ARUCO_CONFIG)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--dictionary")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser.parse_args()


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def main() -> int:
    args = parse_args()
    print("[SAFETY] Camera/ArUco diagnostic only. No robot motion.")
    print("[SAFETY] Offset is valid for the current gripper marker mounting and approximate wrist orientation.")
    print("[SAFETY] If CH4 wrist rotate changes significantly, recalibrate this offset.")

    aruco_root = load_aruco_config(args.aruco_config)
    dict_name = get_aruco_dict_name(aruco_root, args.dictionary)
    dictionary = get_predefined_dictionary(dict_name)
    detector = build_detector(dictionary)
    H = np.load(resolve_existing_path(args.homography, args.aruco_config)).astype(np.float64)
    if H.shape != (3, 3):
        print(f"[BOARD][ERROR] homography must be 3x3, got {H.shape}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[VISION][ERROR] Cannot open camera index {args.cam}", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    samples = []
    last_overlay = None
    try:
        while len(samples) < args.samples:
            ok, frame = cap.read()
            if not ok:
                print("[VISION][WARN] Failed to read frame")
                time.sleep(0.1)
                continue
            corners, ids, _rejected, _processed = detect_markers(frame, dictionary, detector)
            records = add_board_coordinates(marker_records(corners, ids, aruco_root), H)
            record = find_record(records, args.aruco_id)
            marker_xy = None
            if record is not None and "board_x_cm" in record:
                marker_xy = (float(record["board_x_cm"]), float(record["board_y_cm"]))
                offset_x = args.tcp_board_x_cm - marker_xy[0]
                offset_y = args.tcp_board_y_cm - marker_xy[1]
                samples.append({"marker_x_cm": marker_xy[0], "marker_y_cm": marker_xy[1], "offset_x_cm": offset_x, "offset_y_cm": offset_y})
                print(
                    f"[ARUCO] sample {len(samples)}/{args.samples} marker=({marker_xy[0]:.2f},{marker_xy[1]:.2f}) cm "
                    f"offset=({offset_x:.2f},{offset_y:.2f}) cm"
                )
                time.sleep(0.15)

            last_overlay = draw_marker_overlay(
                frame,
                records,
                show_board_coords=True,
                target_board_xy=(args.tcp_board_x_cm, args.tcp_board_y_cm),
                marker_board_xy=marker_xy,
            )
            if args.show:
                cv2.imshow(WINDOW_NAME, last_overlay)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Stopped TCP offset calibration.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    if not samples:
        print(f"[VISION][ERROR] Marker ID {args.aruco_id} was not observed.", file=sys.stderr)
        return 2

    mean_x, std_x = mean_std([s["offset_x_cm"] for s in samples])
    mean_y, std_y = mean_std([s["offset_y_cm"] for s in samples])
    print(f"[TCP] mean offset x={mean_x:.3f} cm std={std_x:.3f} cm")
    print(f"[TCP] mean offset y={mean_y:.3f} cm std={std_y:.3f} cm")

    if args.no_save:
        print("[CONFIG] --no-save set. Not writing aruco_config.yaml.")
        return 0

    if not require_exact("SAVE", "Type SAVE to store marker_to_tcp_offset_cm in aruco_config.yaml."):
        print("Save cancelled.")
        return 0

    cfg = dict(aruco_root)
    aruco_cfg = cfg.setdefault("aruco", {})
    gripper = aruco_cfg.setdefault("gripper_marker", {})
    gripper["id"] = int(args.aruco_id)
    gripper["marker_to_tcp_offset_cm"] = {
        "x": round(mean_x, 4),
        "y": round(mean_y, 4),
        "z": gripper.get("marker_to_tcp_offset_cm", {}).get("z"),
    }
    gripper["calibration_status"] = "measured_board_space_center_samples"
    gripper["calibration_note"] = (
        "Marker center to TCP center offset in board cm for current mounting and approximate wrist orientation."
    )
    aruco_cfg["updated_at"] = utc_now_iso()
    save_yaml(args.aruco_config, cfg)
    print(f"[CONFIG] saved marker_to_tcp_offset_cm to {args.aruco_config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
