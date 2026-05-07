#!/usr/bin/env python3
"""
test_aruco_gripper_target_error.py

Compare a desired board target against the observed ArUco gripper marker / TCP
estimate. Camera-only diagnostic; no motion or serial.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np

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
    detect_markers,
    draw_marker_overlay,
    find_record,
    get_aruco_dict_name,
    get_predefined_dictionary,
    load_aruco_config,
    marker_records,
    resolve_existing_path,
)
from test_yolo_board_mapping import (
    DEFAULT_BOARD_CONFIG,
    DEFAULT_MODEL,
    annotate_board_coordinates,
    load_board_and_homography,
)
from test_yolo_camera import (
    DEFAULT_ASPECT_MAX,
    DEFAULT_ASPECT_MIN,
    DEFAULT_CAKE_CONF,
    DEFAULT_CAKE_MINAREA,
    DEFAULT_CONF,
    DEFAULT_CROSS_NMS_IOU,
    DEFAULT_DEVICE,
    DEFAULT_DONUT_CONF,
    DEFAULT_DONUT_MINAREA,
    DEFAULT_IMGSZ,
    DEFAULT_MAXAREA,
    DEFAULT_MAXDET,
    DEFAULT_MINAREA,
    DEFAULT_NMS_IOU,
    DEFAULT_STABLE_FRAMES,
    DEFAULT_CENTER_TOL,
    StabilityTracker,
    cross_group_nms,
    filter_detections,
    group_level_nms,
    load_model,
    parse_roi,
    print_environment,
    resolve_device,
    run_yolo,
)
from yolo_ik_sequence_utils import normalize_group


WINDOW_NAME = "ArUco Gripper vs Target Error"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare target board coordinate vs observed ArUco gripper/TCP coordinate.",
    )
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_ID)
    parser.add_argument("--target-board-x-cm", type=float)
    parser.add_argument("--target-board-y-cm", type=float)
    parser.add_argument("--target-from-yolo", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-group", default="any")
    parser.add_argument("--homography", default=DEFAULT_HOMOGRAPHY)
    parser.add_argument("--aruco-config", default=DEFAULT_ARUCO_CONFIG)
    parser.add_argument("--board-config", default=DEFAULT_BOARD_CONFIG)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--dictionary")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["auto", "cpu", "0"])
    parser.add_argument("--roi")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--cake-conf", type=float, default=DEFAULT_CAKE_CONF)
    parser.add_argument("--donut-conf", type=float, default=DEFAULT_DONUT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAXDET)
    parser.add_argument("--min-area", type=int, default=DEFAULT_MINAREA)
    parser.add_argument("--max-area", type=int, default=DEFAULT_MAXAREA)
    parser.add_argument("--cake-min-area", type=int, default=DEFAULT_CAKE_MINAREA)
    parser.add_argument("--donut-min-area", type=int, default=DEFAULT_DONUT_MINAREA)
    parser.add_argument("--aspect-min", type=float, default=DEFAULT_ASPECT_MIN)
    parser.add_argument("--aspect-max", type=float, default=DEFAULT_ASPECT_MAX)
    parser.add_argument("--group-nms-iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--cross-nms-iou", type=float, default=DEFAULT_CROSS_NMS_IOU)
    parser.add_argument("--stable-frames", type=int, default=DEFAULT_STABLE_FRAMES)
    parser.add_argument("--center-tolerance", type=int, default=DEFAULT_CENTER_TOL)
    return parser.parse_args()


def choose_locked_target(detections: list[dict], target_group: str) -> dict | None:
    locked = []
    for det in detections:
        if det.get("status") != "LOCKED":
            continue
        if target_group != "ANY" and det.get("group") != target_group:
            continue
        locked.append(det)
    if not locked:
        return None
    locked.sort(key=lambda det: -float(det.get("conf", 0.0)))
    return locked[0]


def detect_yolo_target(frame, args, model, H, width_cm, height_cm, device, tracker):
    raw = run_yolo(model, frame, args.roi, args.conf, args.imgsz, args.max_det, device)
    filt = filter_detections(
        raw,
        conf_thr=args.conf,
        cake_conf=args.cake_conf,
        donut_conf=args.donut_conf,
        min_area=args.min_area,
        max_area=args.max_area,
        cake_min_area=args.cake_min_area,
        donut_min_area=args.donut_min_area,
        aspect_min=args.aspect_min,
        aspect_max=args.aspect_max,
    )
    accepted = [d for d in filt if d["reject_reason"] is None]
    accepted = group_level_nms(accepted, args.group_nms_iou)
    accepted, _cross_rejected = cross_group_nms(accepted, args.cross_nms_iou)
    accepted = tracker.update(accepted)
    mapped, _outside = annotate_board_coordinates(accepted, H, width_cm, height_cm)
    return choose_locked_target(mapped, args.target_group)


def main() -> int:
    args = parse_args()
    print("[SAFETY] Camera/ArUco diagnostic only. No robot motion.")
    if not args.target_from_yolo and (args.target_board_x_cm is None or args.target_board_y_cm is None):
        print("[TARGET][ERROR] Provide --target-board-x-cm/--target-board-y-cm or use --target-from-yolo.", file=sys.stderr)
        return 2

    aruco_root = load_aruco_config(args.aruco_config)
    dict_name = get_aruco_dict_name(aruco_root, args.dictionary)
    dictionary = get_predefined_dictionary(dict_name)
    detector = build_detector(dictionary)
    _, H, H_path, width_cm, height_cm = load_board_and_homography(args.board_config, args.homography)
    print(f"[CONFIG] ArUco dictionary: {dict_name}")
    print(f"[BOARD] homography: {H_path}")

    model = None
    tracker = None
    device = None
    if args.target_from_yolo:
        try:
            args.target_group = normalize_group(args.target_group)
        except ValueError as exc:
            print(f"[TARGET][ERROR] {exc}", file=sys.stderr)
            return 2
        if args.roi:
            try:
                args.roi = parse_roi(args.roi)
            except ValueError as exc:
                print(f"[CONFIG][ERROR] {exc}", file=sys.stderr)
                return 2
        device = resolve_device(args.device)
        print_environment(device)
        model = load_model(args.model, device)
        tracker = StabilityTracker(stable_frames=args.stable_frames, center_tol=args.center_tolerance)

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
    try:
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            corners, ids, _rej, _processed = detect_markers(frame, dictionary, detector)
            records = add_board_coordinates(marker_records(corners, ids, aruco_root), H)
            gripper_id = int(aruco_root.get("aruco", {}).get("gripper_marker", {}).get("id", 0))
            gripper = find_record(records, gripper_id)
            target_xy = None
            target_label = None
            if args.target_from_yolo:
                assert model is not None and tracker is not None and device is not None
                target = detect_yolo_target(frame, args, model, H, width_cm, height_cm, device, tracker)
                if target is not None:
                    target_xy = (float(target["board_x_cm"]), float(target["board_y_cm"]))
                    target_label = f"{target['group']} {target['conf']:.2f}"
            else:
                target_xy = (float(args.target_board_x_cm), float(args.target_board_y_cm))
                target_label = "manual"

            marker_xy = None
            tcp_xy = None
            if gripper is not None and "board_x_cm" in gripper:
                marker_xy = (float(gripper["board_x_cm"]), float(gripper["board_y_cm"]))
                tcp_xy = aruco_target_point_from_offset(marker_xy, aruco_root) or marker_xy

            error_text = None
            if target_xy is not None and marker_xy is not None:
                tcp_for_error = tcp_xy or marker_xy
                dx = tcp_for_error[0] - target_xy[0]
                dy = tcp_for_error[1] - target_xy[1]
                dist = math.hypot(dx, dy)
                print(f"[TARGET] board=({target_xy[0]:.2f},{target_xy[1]:.2f}) source={target_label}")
                print(f"[ARUCO] marker0 board=({marker_xy[0]:.2f},{marker_xy[1]:.2f})")
                if aruco_target_point_from_offset(marker_xy, aruco_root) is None:
                    print("[TCP][WARN] marker_to_tcp_offset_cm not calibrated; using marker center only.")
                else:
                    print(f"[TCP] estimated board=({tcp_xy[0]:.2f},{tcp_xy[1]:.2f})")
                print(f"[ERROR] dx={dx:.2f} cm dy={dy:.2f} cm dist={dist:.2f} cm")
                error_text = f"err dx={dx:.2f} dy={dy:.2f} dist={dist:.2f} cm"
                overlay = draw_marker_overlay(
                    frame,
                    records,
                    show_board_coords=True,
                    target_board_xy=target_xy,
                    marker_board_xy=marker_xy,
                    tcp_board_xy=tcp_xy,
                    error_text=error_text,
                )
                if args.show:
                    cv2.imshow(WINDOW_NAME, overlay)
                    cv2.waitKey(1)
                return 0

            overlay = draw_marker_overlay(
                frame,
                records,
                show_board_coords=True,
                target_board_xy=target_xy,
                marker_board_xy=marker_xy,
                tcp_board_xy=tcp_xy,
                error_text=error_text,
            )
            if args.show:
                cv2.imshow(WINDOW_NAME, overlay)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Stopped gripper-target diagnostic.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    print("[VISION][ERROR] Timed out before getting both a target and gripper marker observation.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
