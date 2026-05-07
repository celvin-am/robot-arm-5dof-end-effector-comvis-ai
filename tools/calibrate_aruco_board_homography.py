#!/usr/bin/env python3
"""
calibrate_aruco_board_homography.py

Compute an alternative ArUco-derived board homography from board marker IDs
1-4. This tool does not overwrite the main homography unless explicitly told to.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from aruco_utils import (
    DEFAULT_ARUCO_CONFIG,
    DEFAULT_ARUCO_HOMOGRAPHY_NPY,
    DEFAULT_ARUCO_HOMOGRAPHY_YAML,
    DEFAULT_CAMERA_ID,
    DEFAULT_HEIGHT,
    DEFAULT_HOMOGRAPHY,
    DEFAULT_WIDTH,
    WINDOW_H,
    WINDOW_W,
    add_board_coordinates,
    build_detector,
    compute_homography_from_markers,
    detect_markers,
    draw_marker_overlay,
    get_aruco_dict_name,
    get_predefined_dictionary,
    homography_reprojection_stats,
    load_aruco_config,
    marker_records,
    require_exact,
    resolve_existing_path,
    resolve_output_path,
    save_yaml,
    utc_now_iso,
)


WINDOW_NAME = "ArUco Homography Calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a separate ArUco-derived board homography from markers 1-4.",
    )
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_ID)
    parser.add_argument("--aruco-config", default=DEFAULT_ARUCO_CONFIG)
    parser.add_argument("--output-npy", default=DEFAULT_ARUCO_HOMOGRAPHY_NPY)
    parser.add_argument("--output-yaml", default=DEFAULT_ARUCO_HOMOGRAPHY_YAML)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--dictionary")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--overwrite-main-homography", action="store_true")
    parser.add_argument("--main-homography", default=DEFAULT_HOMOGRAPHY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("[SAFETY] Camera/ArUco diagnostic only. No robot motion.")

    aruco_root = load_aruco_config(args.aruco_config)
    dict_name = get_aruco_dict_name(aruco_root, args.dictionary)
    dictionary = get_predefined_dictionary(dict_name)
    detector = build_detector(dictionary)
    print(f"[CONFIG] ArUco dictionary: {dict_name}")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[VISION][ERROR] Cannot open camera index {args.cam}", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    H = None
    stats = None
    overlay = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[VISION][WARN] Failed to read frame")
                time.sleep(0.1)
                continue
            corners, ids, _rejected, _processed = detect_markers(frame, dictionary, detector)
            records = marker_records(corners, ids, aruco_root)
            overlay = draw_marker_overlay(frame, records)
            if args.show:
                cv2.imshow(WINDOW_NAME, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            try:
                H, used = compute_homography_from_markers(records, aruco_root)
                mapped = add_board_coordinates(records, H)
                overlay = draw_marker_overlay(frame, mapped, show_board_coords=True)
                stats = homography_reprojection_stats(H, used)
                print("[BOARD] ArUco homography candidate ready.")
                for item in stats["errors"]:
                    print(
                        f"  id={item['id']} reprojection dx={item['dx_cm']:.3f} cm "
                        f"dy={item['dy_cm']:.3f} cm dist={item['dist_cm']:.3f} cm"
                    )
                print(
                    f"[BOARD] mean reprojection error={stats['mean_dist_cm']:.3f} cm "
                    f"max={stats['max_dist_cm']:.3f} cm worst_id={stats['worst_marker_id']}"
                )
                if args.show:
                    cv2.imshow(WINDOW_NAME, overlay)
                    cv2.waitKey(1)
                break
            except RuntimeError:
                pass
    except KeyboardInterrupt:
        print("\n[INFO] Stopped ArUco homography calibration.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    if H is None or stats is None:
        print("[VISION][ERROR] Board markers 1-4 were not all detected together.", file=sys.stderr)
        return 2

    if args.no_save:
        print("[CONFIG] --no-save set. Not writing files.")
        return 0

    if not require_exact("SAVE", "Type SAVE to write ArUco homography outputs."):
        print("Save cancelled.")
        return 0

    output_npy = resolve_output_path(args.output_npy)
    output_yaml = resolve_output_path(args.output_yaml)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, H)
    save_yaml(
        str(output_yaml),
        {
            "aruco_homography": {
                "status": "diagnostic_alternative_not_main_default",
                "dictionary": dict_name,
                "source": "board_marker_centers_ids_1_2_3_4",
                "updated_at": utc_now_iso(),
                "reprojection_error_cm": {
                    "mean": float(stats["mean_dist_cm"]),
                    "max": float(stats["max_dist_cm"]),
                    "worst_marker_id": int(stats["worst_marker_id"]),
                },
                "npy_file": str(output_npy),
                "warning": "Alternative ArUco-derived homography only. Do not replace main checkerboard homography silently.",
            }
        },
    )
    print(f"[CONFIG] saved ArUco homography npy: {output_npy}")
    print(f"[CONFIG] saved ArUco homography yaml: {output_yaml}")

    if args.overwrite_main_homography:
        main_path = resolve_output_path(args.main_homography)
        main_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(main_path, H)
        print(f"[CONFIG] overwrote main homography: {main_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
