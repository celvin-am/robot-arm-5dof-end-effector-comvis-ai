#!/usr/bin/env python3
"""
test_aruco_detection.py

Phase A/B/C camera-only ArUco diagnostic:
- detect board and gripper markers
- optionally map marker centers into board cm with an existing homography
- report board marker error and gripper marker board/TCP estimate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from aruco_utils import (
    DEFAULT_ARUCO_CONFIG,
    DEFAULT_CAMERA_ID,
    DEFAULT_DURATION_SEC,
    DEFAULT_HEIGHT,
    DEFAULT_HOMOGRAPHY,
    DEFAULT_WIDTH,
    WINDOW_H,
    WINDOW_W,
    add_board_coordinates,
    aruco_target_point_from_offset,
    build_detector,
    compute_board_error_stats,
    detect_markers,
    draw_marker_overlay,
    draw_rejected_overlay,
    filter_rejected_candidate_records,
    get_aruco_dict_name,
    get_predefined_dictionary,
    load_aruco_config,
    marker_records,
    rejected_candidate_records,
    resolve_existing_path,
    save_candidate_crops,
)
import numpy as np


WINDOW_NAME = "ArUco Detection Debug"
DEFAULT_DEBUG_DIR = "debug_aruco_candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect ArUco board/gripper markers and report board coordinates.",
    )
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_ID)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--aruco-config", default=DEFAULT_ARUCO_CONFIG)
    parser.add_argument("--homography", default=DEFAULT_HOMOGRAPHY)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--dictionary")
    parser.add_argument("--detect-once", action="store_true")
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--show-rejected", action="store_true")
    parser.add_argument("--save-rejected", action="store_true")
    parser.add_argument("--debug-crops", action="store_true")
    parser.add_argument("--print-rejected-details", action="store_true")
    parser.add_argument("--show-all-rejected", action="store_true")
    parser.add_argument("--min-candidate-width-px", type=float, default=20.0)
    parser.add_argument("--min-candidate-height-px", type=float, default=20.0)
    parser.add_argument("--min-candidate-area-px", type=float, default=500.0)
    parser.add_argument("--max-rejected-print", type=int, default=30)
    parser.add_argument("--adaptive-block-size", type=int, default=31)
    parser.add_argument("--adaptive-c", type=int, default=7)
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--detector-profile", choices=["default", "relaxed", "aggressive"], default="default")
    parser.add_argument("--preprocess", choices=["none", "gray", "clahe", "adaptive", "sharpen"], default="none")
    parser.add_argument("--try-dictionaries", action="store_true")
    parser.add_argument("--aruco-roi")
    parser.add_argument("--roi-name")
    parser.add_argument("--save-raw-frame", action="store_true")
    parser.add_argument("--camera-debug", action="store_true")
    return parser.parse_args()


def load_optional_homography(path_value: str | None, aruco_config_path: str) -> tuple[np.ndarray | None, str | None]:
    if not path_value:
        return None, None
    try:
        path = resolve_existing_path(path_value, aruco_config_path)
        H = np.load(path)
        if H.shape != (3, 3):
            raise ValueError(f"homography must be 3x3, got {H.shape}")
        return H.astype(np.float64), str(path)
    except Exception as exc:
        print(f"[BOARD][WARN] Homography unavailable: {exc}")
        return None, None


def print_marker_report(records: list[dict], aruco_root: dict, H: np.ndarray | None) -> None:
    print(f"[VISION] accepted marker count: {len(records)}")
    if not records:
        return

    for record in records:
        print(f"[ARUCO] id={record['id']} role={record.get('role') or 'unknown'}")
        print(f"  pixel center: u={record['pixel_u']:.1f} v={record['pixel_v']:.1f}")
        print(
            f"  size: w={record['approx_width_px']:.1f}px "
            f"h={record['approx_height_px']:.1f}px "
            f"area={record['approx_area_px']:.1f}px^2"
        )
        print(f"  pixel corners: {record['pixel_corners']}")
        if H is not None and "board_x_cm" in record:
            print(f"  board: x={record['board_x_cm']:.2f} cm y={record['board_y_cm']:.2f} cm")
            if record.get("expected_board_x_cm") is not None:
                dx = float(record["board_x_cm"]) - float(record["expected_board_x_cm"])
                dy = float(record["board_y_cm"]) - float(record["expected_board_y_cm"])
                print(
                    "  expected board: "
                    f"x={record['expected_board_x_cm']:.2f} cm y={record['expected_board_y_cm']:.2f} cm"
                )
                print(f"  board error: dx={dx:.2f} cm dy={dy:.2f} cm")

    stats = compute_board_error_stats(records)
    if stats is not None:
        print(
            f"[BOARD] mean board-marker error={stats['mean_dist_cm']:.3f} cm "
            f"max={stats['max_dist_cm']:.3f} cm worst_id={stats['worst_marker_id']}"
        )

    gripper_cfg = aruco_root.get("aruco", {}).get("gripper_marker", {})
    gripper_id = int(gripper_cfg.get("id", 0))
    gripper_record = next((r for r in records if int(r["id"]) == gripper_id), None)
    if gripper_record is None:
        return
    if H is None or "board_x_cm" not in gripper_record:
        print("[ARUCO][WARN] Gripper marker detected, but board coordinate is unavailable without homography.")
        return
    marker_xy = (float(gripper_record["board_x_cm"]), float(gripper_record["board_y_cm"]))
    print(f"[ARUCO] marker{gripper_id} board=({marker_xy[0]:.2f},{marker_xy[1]:.2f}) cm")
    tcp_xy = aruco_target_point_from_offset(marker_xy, aruco_root)
    if tcp_xy is None:
        print("[TCP][WARN] marker_to_tcp_offset_cm is not calibrated yet.")
    else:
        print(f"[TCP] estimated board=({tcp_xy[0]:.2f},{tcp_xy[1]:.2f}) cm")


def parse_roi(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def print_rejected_report(
    total_rejected: int,
    filtered_records: list[dict],
    print_details: bool = False,
    max_rejected_print: int = 30,
) -> None:
    print(f"[VISION] rejected candidates total={total_rejected}, after_filter={len(filtered_records)}")
    if not print_details:
        return
    for record in filtered_records[:max(0, int(max_rejected_print))]:
        print(
            f"[ARUCO_REJECTED] idx={record['index']} "
            f"u={record['pixel_u']:.1f} v={record['pixel_v']:.1f} "
            f"w={record['approx_width_px']:.1f} h={record['approx_height_px']:.1f} "
            f"area={(record['approx_width_px'] * record['approx_height_px']):.1f}"
        )
        print(f"  pixel corners: {record['pixel_corners']}")
    if len(filtered_records) > max(0, int(max_rejected_print)):
        print(f"[ARUCO_REJECTED] ... truncated {len(filtered_records) - int(max_rejected_print)} more candidates")


def trial_dictionaries(frame: np.ndarray, dict_names: list[str], profile: str, preprocess: str, roi, args: argparse.Namespace) -> None:
    print("[ARUCO] dictionary trial mode:")
    for dict_name in dict_names:
        dictionary = get_predefined_dictionary(dict_name)
        detector = build_detector(dictionary, profile=profile)
        corners, ids, rejected, _processed = detect_markers(
            frame,
            dictionary,
            detector=detector,
            profile=profile,
            preprocess=preprocess,
            roi=roi,
            adaptive_block_size=args.adaptive_block_size,
            adaptive_c=args.adaptive_c,
        )
        detected_ids = [] if ids is None else [int(v) for v in ids.flatten().tolist()]
        rejected_records = rejected_candidate_records(rejected)
        filtered_rejected = filter_rejected_candidate_records(
            rejected_records,
            args.min_candidate_width_px,
            args.min_candidate_height_px,
            args.min_candidate_area_px,
            show_all=args.show_all_rejected,
        )
        print(
            f"  {dict_name}: ids={detected_ids} "
            f"rejected_total={len(rejected_records)} rejected_filtered={len(filtered_rejected)}"
        )


def threshold_sweep(frame: np.ndarray, dictionary, profile: str, roi, args: argparse.Namespace) -> None:
    print("[ARUCO] adaptive threshold sweep:")
    for block_size in (11, 15, 21, 31, 41, 51):
        for adaptive_c in (3, 5, 7, 9):
            detector = build_detector(dictionary, profile=profile)
            corners, ids, rejected, _processed = detect_markers(
                frame,
                dictionary,
                detector=detector,
                profile=profile,
                preprocess="adaptive",
                roi=roi,
                adaptive_block_size=block_size,
                adaptive_c=adaptive_c,
            )
            detected_ids = [] if ids is None else [int(v) for v in ids.flatten().tolist()]
            rejected_records = rejected_candidate_records(rejected)
            filtered_rejected = filter_rejected_candidate_records(
                rejected_records,
                args.min_candidate_width_px,
                args.min_candidate_height_px,
                args.min_candidate_area_px,
                show_all=args.show_all_rejected,
            )
            print(
                f"  block={block_size:02d} C={adaptive_c}: "
                f"ids={detected_ids} rejected_total={len(rejected_records)} rejected_filtered={len(filtered_rejected)}"
            )


def main() -> int:
    args = parse_args()
    print("[SAFETY] Camera/ArUco diagnostic only. No robot motion.")
    try:
        roi = parse_roi(args.aruco_roi)
    except ValueError as exc:
        print(f"[CONFIG][ERROR] {exc}", file=sys.stderr)
        return 2

    aruco_root = load_aruco_config(args.aruco_config)
    dict_name = get_aruco_dict_name(aruco_root, args.dictionary)
    dictionary = get_predefined_dictionary(dict_name)
    detector = build_detector(dictionary, profile=args.detector_profile)
    H, H_path = load_optional_homography(args.homography, args.aruco_config)
    print(f"[CONFIG] ArUco dictionary: {dict_name}")
    print(f"[CONFIG] detector_profile: {args.detector_profile}")
    print(f"[CONFIG] preprocess: {args.preprocess}")
    if args.detector_profile == "aggressive":
        print("[ARUCO][WARN] aggressive detector profile may increase false positives.")
    if H_path:
        print(f"[BOARD] homography: {H_path}")
    if roi is not None:
        print(f"[ROI] Using ROI {roi[0]},{roi[1]},{roi[2]},{roi[3]}")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[VISION][ERROR] Cannot open camera index {args.cam}", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    deadline = time.time() + args.duration_sec
    debug_frame = None
    raw_frame = None
    ran_threshold_sweep = False
    ran_dictionary_trial = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[VISION][WARN] Failed to read frame")
                time.sleep(0.1)
                if time.time() >= deadline:
                    break
                continue
            raw_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args.camera_debug:
                print(
                    f"[CAMERA] frame={frame.shape[1]}x{frame.shape[0]} "
                    f"brightness_mean={float(np.mean(gray)):.2f} "
                    f"brightness_std={float(np.std(gray)):.2f}"
                )
            if args.try_dictionaries and not ran_dictionary_trial:
                trial_dictionaries(
                    frame,
                    ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_5X5_50", "DICT_5X5_100"],
                    args.detector_profile,
                    args.preprocess,
                    roi,
                    args,
                )
                ran_dictionary_trial = True
            if args.threshold_sweep and not ran_threshold_sweep:
                threshold_sweep(frame, dictionary, args.detector_profile, roi, args)
                ran_threshold_sweep = True
            corners, ids, rejected, processed_frame = detect_markers(
                frame,
                dictionary,
                detector=detector,
                profile=args.detector_profile,
                preprocess=args.preprocess,
                roi=roi,
                adaptive_block_size=args.adaptive_block_size,
                adaptive_c=args.adaptive_c,
            )
            records = add_board_coordinates(marker_records(corners, ids, aruco_root), H)
            rejected_records = rejected_candidate_records(rejected)
            filtered_rejected_records = filter_rejected_candidate_records(
                rejected_records,
                args.min_candidate_width_px,
                args.min_candidate_height_px,
                args.min_candidate_area_px,
                show_all=args.show_all_rejected,
            )
            print_marker_report(records, aruco_root, H)
            print_rejected_report(
                len(rejected_records),
                filtered_rejected_records,
                print_details=args.print_rejected_details,
                max_rejected_print=args.max_rejected_print,
            )
            overlay = draw_marker_overlay(frame, records, show_board_coords=H is not None)
            if args.show_rejected and filtered_rejected_records:
                overlay = draw_rejected_overlay(overlay, filtered_rejected_records)
            if roi is not None:
                x1, y1, x2, y2 = roi
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 1)
            debug_frame = overlay
            if args.show:
                cv2.imshow(WINDOW_NAME, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            if args.detect_once:
                break
            if time.time() >= deadline:
                break
            if records:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped ArUco detection.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    if args.save_debug and debug_frame is not None:
        out_path = Path("debug_aruco_detection.jpg")
        cv2.imwrite(str(out_path), debug_frame)
        print(f"[VISION] saved debug overlay: {out_path.resolve()}")
    if args.save_raw_frame and raw_frame is not None:
        out_path = Path("debug_aruco_raw.jpg")
        cv2.imwrite(str(out_path), raw_frame)
        print(f"[VISION] saved raw frame: {out_path.resolve()}")
    if (args.debug_crops or args.save_rejected) and raw_frame is not None:
        corners, ids, rejected, _processed = detect_markers(
            raw_frame,
            dictionary,
            detector=detector,
            profile=args.detector_profile,
            preprocess=args.preprocess,
            roi=roi,
            adaptive_block_size=args.adaptive_block_size,
            adaptive_c=args.adaptive_c,
        )
        accepted_records = marker_records(corners, ids, aruco_root)
        rejected_records = filter_rejected_candidate_records(
            rejected_candidate_records(rejected),
            args.min_candidate_width_px,
            args.min_candidate_height_px,
            args.min_candidate_area_px,
            show_all=args.show_all_rejected,
        )
        prefix_suffix = args.roi_name or "full"
        if args.debug_crops and accepted_records:
            saved = save_candidate_crops(raw_frame, accepted_records, DEFAULT_DEBUG_DIR, f"accepted_{prefix_suffix}")
            print(f"[VISION] saved accepted crops: {len(saved)}")
        if (args.debug_crops or args.save_rejected) and rejected_records:
            saved = save_candidate_crops(raw_frame, rejected_records, DEFAULT_DEBUG_DIR, f"rejected_{prefix_suffix}")
            print(f"[VISION] saved rejected crops: {len(saved)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
