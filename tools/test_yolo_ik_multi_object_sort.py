#!/usr/bin/env python3
"""
test_yolo_ik_multi_object_sort.py

Guarded multi-object class sorter test. This is not final autonomy: it rescans
after each pick attempt and stops on any error.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import cv2

from test_esp32_manual_move import open_serial, send_command
from test_ik_to_move_safe import DEFAULT_SERIAL_CONFIG, require_line
from test_yolo_board_mapping import (
    DEFAULT_BOARD_CONFIG,
    DEFAULT_HEIGHT,
    DEFAULT_MODEL,
    DEFAULT_WIDTH,
    WINDOW_H,
    WINDOW_NAME,
    WINDOW_W,
    annotate_board_coordinates,
    draw_overlay,
    load_board_and_homography,
)
from test_yolo_camera import (
    DEFAULT_ASPECT_MAX,
    DEFAULT_ASPECT_MIN,
    DEFAULT_CAKE_CONF,
    DEFAULT_CAKE_MINAREA,
    DEFAULT_CAM,
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
from yolo_ik_sequence_utils import (
    DEFAULT_ARUCO_CONFIG,
    DEFAULT_SEMI_AUTO_CONFIG,
    DEFAULT_TAUGHT_POSE_CONFIG,
    DEFAULT_WORKSPACE_CORRECTION_MAP,
    attempted_too_close,
    build_home_step,
    build_pick_sequence_for_target,
    default_workspace_profile,
    evaluate_aruco_board_error,
    format_sequence_preview,
    get_board_limits,
    get_group_offsets,
    load_optional_ik_calibration,
    load_semi_auto_pick_place_config,
    load_workspace_correction_map,
    normalize_group,
    response_has_esp32_reset,
    resolve_runtime_defaults,
    select_workspace_profile,
    validate_board_target,
    validate_pick_region,
)
from test_ik_dry_run import (
    DEFAULT_KINEMATICS_CONFIG,
    DEFAULT_POSE_CONFIG,
    DEFAULT_SERVO_CONFIG,
    DEFAULT_TRANSFORM_CONFIG,
    board_to_robot,
    load_required,
)


DEFAULT_TIMEOUT_SEC = 20.0
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded multi-object CAKE/DONUT sorter test.",
    )
    parser.add_argument("--cam", type=int)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--port")
    parser.add_argument("--yes-i-understand-hardware-risk", action="store_true")
    parser.add_argument("--confirm-each", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--target-group", default="any")
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--config", default=DEFAULT_SEMI_AUTO_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["auto", "cpu", "0"])
    parser.add_argument("--roi", type=str)
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
    parser.add_argument("--show-rejected", action="store_true")
    parser.add_argument("--allow-out-of-board-target", action="store_true")
    parser.add_argument("--allow-outside-validated-region", action="store_true")
    parser.add_argument("--servo-limit-margin-deg", type=float, default=5.0)
    parser.add_argument("--use-workspace-correction-map", action="store_true")
    parser.add_argument("--workspace-correction-map-config", default=DEFAULT_WORKSPACE_CORRECTION_MAP)
    parser.add_argument("--allow-untested-region", action="store_true")
    parser.add_argument("--attempted-suppression-cm", type=float, default=2.0)
    parser.add_argument("--duplicate-distance-cm", type=float, default=2.0)
    parser.add_argument("--print-object-list", action="store_true")
    parser.add_argument("--selection-strategy", choices=["confidence", "nearest_center"], default="confidence")
    parser.add_argument("--check-aruco-board", action="store_true")
    parser.add_argument("--aruco-config", default=DEFAULT_ARUCO_CONFIG)
    parser.add_argument("--aruco-board-error-threshold-cm", type=float, default=2.0)
    parser.add_argument("--aruco-detector-profile", choices=["default", "relaxed", "aggressive"], default="relaxed")
    parser.add_argument("--aruco-preprocess", choices=["none", "gray", "clahe", "adaptive", "sharpen"], default="clahe")
    parser.add_argument("--tcp-offset-mode")
    parser.add_argument("--solution", choices=["elbow_up", "elbow_down", "best"], default="elbow_up")
    parser.add_argument("--use-ik-servo-calibration", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-z-mode-correction", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ik-servo-calibration-config", default="ros2_ws/src/robot_arm_5dof/config/ik_servo_calibration.yaml")
    parser.add_argument("--board-config", default=DEFAULT_BOARD_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--taught-pose-config", default=DEFAULT_TAUGHT_POSE_CONFIG)
    parser.add_argument("--serial-config", default=DEFAULT_SERIAL_CONFIG)
    return parser.parse_args()


def require_exact(label: str, prompt: str) -> bool:
    print(prompt)
    try:
        typed = input("> ").strip()
    except EOFError:
        return False
    return typed == label


def build_object_list(
    detections: list[dict[str, Any]],
    target_group: str,
    board_width_cm: float,
    board_height_cm: float,
    transform_cfg: dict[str, Any],
    semi_auto_cfg: dict[str, Any],
    workspace_cfg: dict[str, Any] | None,
    use_workspace_correction_map: bool,
    allow_untested_region: bool,
    allow_outside_validated_region: bool,
    attempted_targets: list[dict[str, float]],
    attempted_suppression_cm: float,
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    pose_cfg: dict[str, Any],
    taught_cfg: dict[str, Any],
    ik_servo_cal_cfg: dict[str, Any] | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    object_id = 1
    for det in detections:
        if det.get("status") != "LOCKED":
            continue
        if target_group != "ANY" and det.get("group") != target_group:
            continue
        board_x = float(det["board_x_cm"])
        board_y = float(det["board_y_cm"])
        workspace_profile = None
        workspace_error = ""
        if use_workspace_correction_map and workspace_cfg is not None:
            try:
                workspace_profile = select_workspace_profile(
                    workspace_cfg,
                    board_x,
                    board_y,
                    allow_untested_region=allow_untested_region,
                )
                offset_x_cm = float(workspace_profile["grasp_offset_x_cm"])
                offset_y_cm = float(workspace_profile["grasp_offset_y_cm"])
            except ValueError as exc:
                workspace_error = str(exc)
                workspace_profile = None
                offset_x_cm, offset_y_cm = get_group_offsets(semi_auto_cfg, det["group"])
        else:
            offset_x_cm, offset_y_cm = get_group_offsets(semi_auto_cfg, det["group"])
        adjusted_board_x_cm = board_x + offset_x_cm
        adjusted_board_y_cm = board_y + offset_y_cm
        ok, _message = validate_board_target(
            adjusted_board_x_cm,
            adjusted_board_y_cm,
            board_width_cm,
            board_height_cm,
        )
        robot_x, robot_y = board_to_robot(adjusted_board_x_cm, adjusted_board_y_cm, transform_cfg)
        status = "pending"
        status_reason = ""
        region_ok, region_message = validate_pick_region(
            adjusted_board_x_cm,
            adjusted_board_y_cm,
            semi_auto_cfg,
        )
        if workspace_error:
            status = "workspace_rejected"
            status_reason = workspace_error
        elif not ok:
            status = "outside_board"
            status_reason = _message
        elif not region_ok and not allow_outside_validated_region:
            status = "outside_validated_pick_region"
            status_reason = region_message
        elif attempted_too_close(
            adjusted_board_x_cm,
            adjusted_board_y_cm,
            attempted_targets,
            min_distance_cm=attempted_suppression_cm,
        ):
            status = "too_close_to_attempted"
        else:
            try:
                build_pick_sequence_for_target(
                    det,
                    semi_auto_cfg,
                    kin_cfg,
                    servo_cfg,
                    pose_cfg,
                    taught_cfg,
                    transform_cfg,
                    ik_servo_cal_cfg,
                    args,
                    workspace_profile=workspace_profile,
                )
            except ValueError as exc:
                text = str(exc)
                if "near servo limit:" in text:
                    status = "near_servo_limit"
                    status_reason = text
                elif "servo limit failure:" in text:
                    status = "servo_limit_failure"
                    status_reason = text
                else:
                    status = "ik_rejected"
                    status_reason = text
        objects.append(
            {
                "id": object_id,
                "raw_class": det["raw"],
                "group": det["group"],
                "confidence": float(det["conf"]),
                "pixel_u": int(det["u"]),
                "pixel_v": int(det["v"]),
                "board_x_cm": board_x,
                "board_y_cm": board_y,
                "adjusted_board_x_cm": adjusted_board_x_cm,
                "adjusted_board_y_cm": adjusted_board_y_cm,
                "robot_x_m": robot_x,
                "robot_y_m": robot_y,
                "status": status,
                "workspace_region": workspace_profile["region_name"] if workspace_profile is not None else "global_default",
                "workspace_region_status": workspace_profile["status"] if workspace_profile is not None else "global_default",
                "workspace_error": workspace_error,
                "status_reason": status_reason,
                "workspace_profile": workspace_profile or default_workspace_profile(semi_auto_cfg, det["group"]),
                "_det": det,
            }
        )
        object_id += 1
    return objects


def dedupe_locked_detections(
    detections: list[dict[str, Any]],
    target_group: str,
    duplicate_distance_cm: float,
) -> tuple[list[dict[str, Any]], int]:
    locked = []
    for det in detections:
        if det.get("status") != "LOCKED":
            continue
        if target_group != "ANY" and det.get("group") != target_group:
            continue
        locked.append(det)
    locked.sort(key=lambda det: -float(det.get("conf", 0.0)))
    kept: list[dict[str, Any]] = []
    suppressed = 0
    for det in locked:
        board_x = float(det["board_x_cm"])
        board_y = float(det["board_y_cm"])
        duplicate = False
        for prev in kept:
            if det.get("group") != prev.get("group"):
                continue
            dx = board_x - float(prev["board_x_cm"])
            dy = board_y - float(prev["board_y_cm"])
            if (dx * dx + dy * dy) ** 0.5 <= duplicate_distance_cm:
                duplicate = True
                break
        if duplicate:
            suppressed += 1
            continue
        kept.append(det)
    return kept, suppressed


def choose_target(
    objects: list[dict[str, Any]],
    strategy: str,
    board_width_cm: float,
    board_height_cm: float,
) -> dict[str, Any] | None:
    pending = [obj for obj in objects if obj["status"] == "pending"]
    if not pending:
        return None
    if strategy == "nearest_center":
        center_x = board_width_cm / 2.0
        center_y = board_height_cm / 2.0
        pending.sort(
            key=lambda obj: (
                (float(obj["adjusted_board_x_cm"]) - center_x) ** 2
                + (float(obj["adjusted_board_y_cm"]) - center_y) ** 2,
                -float(obj["confidence"]),
            )
        )
    else:
        pending.sort(key=lambda obj: (-float(obj["confidence"]), obj["id"]))
    return pending[0]


def print_object_list(objects: list[dict[str, Any]]) -> None:
    print("[STATE] object list:")
    if not objects:
        print("  (empty)")
        return
    for obj in objects:
        print(
            f"  id={obj['id']} group={obj['group']} raw={obj['raw_class']} "
            f"conf={obj['confidence']:.3f} "
            f"u={obj['pixel_u']} v={obj['pixel_v']} "
            f"board=({obj['board_x_cm']:.2f},{obj['board_y_cm']:.2f}) "
            f"adjusted=({obj['adjusted_board_x_cm']:.2f},{obj['adjusted_board_y_cm']:.2f}) "
            f"robot=({obj['robot_x_m']:.4f},{obj['robot_y_m']:.4f}) "
            f"region={obj['workspace_region']} "
            f"status={obj['status']}"
        )
        if obj.get("workspace_error"):
            print(f"      [WORKSPACE] {obj['workspace_error']}")
        if obj.get("status_reason"):
            print(f"      [SAFETY] {obj['status_reason']}")


def scan_locked_objects(
    args: argparse.Namespace,
    model,
    H,
    width_cm: float,
    height_cm: float,
    cap,
    tracker: StabilityTracker,
    device: str,
    transform_cfg: dict[str, Any],
    semi_auto_cfg: dict[str, Any],
    workspace_cfg: dict[str, Any] | None,
    attempted_targets: list[dict[str, float]],
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    pose_cfg: dict[str, Any],
    taught_cfg: dict[str, Any],
    ik_servo_cal_cfg: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    deadline = time.time() + args.timeout_sec
    last_objects: list[dict[str, Any]] = []
    best_objects: list[dict[str, Any]] = []
    best_stats = {
        "raw_detections_count": 0,
        "locked_object_count": 0,
        "candidate_count": 0,
        "attempted_suppression_count": 0,
        "duplicate_suppression_count": 0,
    }
    settle_deadline: float | None = None
    last_aruco_warning_at = 0.0
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        if args.check_aruco_board:
            try:
                board_stats = evaluate_aruco_board_error(
                    frame,
                    H,
                    args.aruco_config,
                    detector_profile=args.aruco_detector_profile,
                    preprocess=args.aruco_preprocess,
                )
            except Exception as exc:
                if time.time() - last_aruco_warning_at > 2.0:
                    print(f"[SAFETY] ArUco board diagnostic unavailable: {exc}")
                    last_aruco_warning_at = time.time()
            else:
                if (
                    board_stats is not None
                    and float(board_stats.get("max_dist_cm", 0.0)) > args.aruco_board_error_threshold_cm
                    and time.time() - last_aruco_warning_at > 2.0
                ):
                    print(
                        "[SAFETY] ArUco board marker error exceeded threshold: "
                        f"max={board_stats['max_dist_cm']:.2f} cm "
                        f"(threshold={args.aruco_board_error_threshold_cm:.2f} cm)"
                    )
                    last_aruco_warning_at = time.time()

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
        rejected = [d for d in filt if d["reject_reason"] is not None]
        accepted = group_level_nms(accepted, args.group_nms_iou)
        accepted, cross_rejected = cross_group_nms(accepted, args.cross_nms_iou)
        rejected.extend(cross_rejected)
        accepted = tracker.update(accepted)
        mapped_acc, outside_rej = annotate_board_coordinates(accepted, H, width_cm, height_cm)
        rejected.extend(outside_rej)
        overlay = draw_overlay(frame, mapped_acc, rejected, 0.0, len(raw), args.show_rejected)
        candidate_count = sum(1 for det in mapped_acc if det.get("status") == "CANDIDATE")
        locked_detections, duplicate_suppression_count = dedupe_locked_detections(
            mapped_acc,
            args.target_group,
            args.duplicate_distance_cm,
        )
        last_objects = build_object_list(
            locked_detections,
            args.target_group,
            width_cm,
            height_cm,
            transform_cfg,
            semi_auto_cfg,
            workspace_cfg,
            args.use_workspace_correction_map,
            args.allow_untested_region,
            args.allow_outside_validated_region,
            attempted_targets,
            args.attempted_suppression_cm,
            kin_cfg,
            servo_cfg,
            pose_cfg,
            taught_cfg,
            ik_servo_cal_cfg,
            args,
        )
        attempted_suppression_count = sum(1 for obj in last_objects if obj["status"] == "too_close_to_attempted")
        outside_validated_region_count = sum(1 for obj in last_objects if obj["status"] == "outside_validated_pick_region")
        near_servo_limit_count = sum(1 for obj in last_objects if obj["status"] == "near_servo_limit")
        pending_count = sum(1 for obj in last_objects if obj["status"] == "pending")
        current_stats = {
            "raw_detections_count": len(raw),
            "locked_object_count": len(locked_detections),
            "candidate_count": candidate_count,
            "attempted_suppression_count": attempted_suppression_count,
            "duplicate_suppression_count": duplicate_suppression_count,
            "outside_validated_region_count": outside_validated_region_count,
            "near_servo_limit_count": near_servo_limit_count,
            "pending_count": pending_count,
        }
        if (
            len(last_objects) > len(best_objects)
            or (len(last_objects) == len(best_objects) and pending_count > best_stats.get("pending_count", 0))
        ):
            best_objects = [dict(obj) for obj in last_objects]
            best_stats = dict(current_stats)
        if locked_detections and settle_deadline is None:
            settle_deadline = min(deadline, time.time() + 1.0)

        if args.show:
            cv2.imshow(WINDOW_NAME, overlay)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                raise KeyboardInterrupt
        if settle_deadline is not None and time.time() >= settle_deadline:
            break

    return best_objects if best_objects else last_objects, best_stats


def send_steps(ser, steps: list[dict[str, Any]]) -> None:
    for step in steps:
        if step["type"] == "HOME":
            lines = send_command(ser, "HOME")
            print("COMMAND HOME")
            for line in lines:
                print(f"  {line}")
            if response_has_esp32_reset(lines):
                raise RuntimeError("ESP32 reset detected during motion")
            require_line(lines, "ACK HOME", step["name"])
            require_line(lines, "DONE HOME", step["name"])
            continue
        lines = send_command(ser, step["command"])
        print(f"COMMAND {step['command']}")
        for line in lines:
            print(f"  {line}")
        if response_has_esp32_reset(lines):
            raise RuntimeError("ESP32 reset detected during motion")
        require_line(lines, "ACK MOVE_SAFE", step["name"])
        require_line(lines, "DONE MOVE_SAFE", step["name"])


def main() -> int:
    args = parse_args()

    if args.dry_run and args.send:
        print("[SAFETY][ERROR] Use either --dry-run or --send, not both", file=sys.stderr)
        return 2
    if args.send and args.port is None:
        print("[SERIAL][ERROR] --port is required when --send is used", file=sys.stderr)
        return 2
    if args.send and not args.yes_i_understand_hardware_risk:
        print("[SAFETY][ERROR] Live multi-object motion requires --yes-i-understand-hardware-risk.", file=sys.stderr)
        return 2
    if args.confirm_each and args.auto:
        print("[SAFETY][ERROR] Use either --confirm-each or --auto, not both.", file=sys.stderr)
        return 2

    semi_auto_cfg = load_semi_auto_pick_place_config(args.config)
    resolve_runtime_defaults(args, semi_auto_cfg)
    try:
        args.target_group = normalize_group(args.target_group)
    except ValueError as exc:
        print(f"[CONFIG][ERROR] {exc}", file=sys.stderr)
        return 2

    if args.use_ik_servo_calibration is None:
        args.use_ik_servo_calibration = bool(semi_auto_cfg.get("ik", {}).get("use_ik_servo_calibration", True))
    if args.use_z_mode_correction is None:
        args.use_z_mode_correction = bool(semi_auto_cfg.get("ik", {}).get("use_z_mode_correction", True))

    max_objects_default = int(semi_auto_cfg.get("multi_object", {}).get("max_objects_default", 4))
    if args.max_objects is None:
        if args.auto:
            print("[SAFETY][ERROR] --auto requires explicit --max-objects.", file=sys.stderr)
            return 2
        args.max_objects = max_objects_default
    if args.auto and not require_exact("AUTO_START", "Type AUTO_START to begin guarded multi-object motion."):
        print("AUTO_START cancelled.")
        return 0

    kin_cfg = load_required(args.kinematics_config, "kinematics config")
    servo_cfg = load_required(args.servo_config, "servo config")
    pose_cfg = load_required(args.pose_config, "pose config")
    transform_cfg = load_required(args.transform_config, "transform config")
    board_cfg = load_required(args.board_config, "board config")
    taught_cfg = load_required(args.taught_pose_config, "taught pose config")
    serial_cfg = load_required(args.serial_config, "serial config")
    _ = serial_cfg
    ik_servo_cal_cfg = load_optional_ik_calibration(args)
    workspace_cfg = load_workspace_correction_map(args.workspace_correction_map_config) if args.use_workspace_correction_map else None

    supported_modes = kin_cfg.get("ik", {}).get("tcp_offset_modes_supported", ["none", "planar", "vertical_down", "mixed"])
    if args.tcp_offset_mode not in supported_modes:
        print(f"[CONFIG][ERROR] Unsupported tcp_offset_mode {args.tcp_offset_mode!r}. Supported: {supported_modes}", file=sys.stderr)
        return 2

    if args.roi:
        try:
            args.roi = parse_roi(args.roi)
        except ValueError as exc:
            print(f"[CONFIG][ERROR] {exc}", file=sys.stderr)
            return 2

    _, H, H_path, width_cm, height_cm = load_board_and_homography(args.board_config, None)
    board_width_cfg, board_height_cfg = get_board_limits(board_cfg)
    print("[SAFETY] Multi-object motion is not final validated autonomy.")
    print("[SAFETY] Keep hand near emergency stop / power cutoff.")
    print("[SAFETY] Board and bowls must be fixed.")
    print("[SAFETY] Robot must start from HOME_SAFE.")
    print(f"[CONFIG] semi_auto config: {args.config}")
    print(f"[BOARD] homography: {H_path}")
    print(f"[BOARD] limits from config: x=[0,{board_width_cfg:.2f}] y=[0,{board_height_cfg:.2f}] cm")
    print(f"[IK] tcp_offset_mode: {args.tcp_offset_mode}")
    print(f"[IK] use_ik_servo_calibration: {args.use_ik_servo_calibration}")
    print(f"[IK] use_z_mode_correction: {args.use_z_mode_correction}")
    print(f"[SAFETY] servo_limit_margin_deg: {args.servo_limit_margin_deg:.1f}")
    print(f"[WORKSPACE] use_workspace_correction_map: {args.use_workspace_correction_map}")
    print(f"[STATE] max_objects: {args.max_objects}")
    print(f"[TARGET] selection_strategy: {args.selection_strategy}")
    print(f"[TARGET] attempted_suppression_cm: {args.attempted_suppression_cm:.2f}")
    print(f"[TARGET] duplicate_distance_cm: {args.duplicate_distance_cm:.2f}")
    if args.check_aruco_board:
        print(f"[SAFETY] ArUco board diagnostic enabled (threshold={args.aruco_board_error_threshold_cm:.2f} cm)")

    device = resolve_device(args.device)
    print_environment(device)
    model = load_model(args.model, device)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[VISION][ERROR] Cannot open camera index {args.cam}", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    tracker = StabilityTracker(stable_frames=args.stable_frames, center_tol=args.center_tolerance)
    attempted_targets: list[dict[str, float]] = []
    processed = 0
    ser = None

    try:
        if args.send:
            ser = open_serial(args.port, args.baud, args.timeout)

        while processed < args.max_objects:
            print(f"\n[STATE] rescanning workspace (processed={processed}/{args.max_objects}) ...")
            objects, scan_stats = scan_locked_objects(
                args,
                model,
                H,
                width_cm,
                height_cm,
                cap,
                tracker,
                device,
                transform_cfg,
                semi_auto_cfg,
                workspace_cfg,
                attempted_targets,
                kin_cfg,
                servo_cfg,
                pose_cfg,
                taught_cfg,
                ik_servo_cal_cfg,
            )
            print(f"[VISION] raw detections count: {scan_stats.get('raw_detections_count', 0)}")
            print(f"[VISION] locked object count: {scan_stats.get('locked_object_count', 0)}")
            print(f"[VISION] candidate count: {scan_stats.get('candidate_count', 0)}")
            print(f"[TARGET] attempted suppression count: {scan_stats.get('attempted_suppression_count', 0)}")
            print(f"[TARGET] duplicate suppression count: {scan_stats.get('duplicate_suppression_count', 0)}")
            print(f"[TARGET] outside validated region count: {scan_stats.get('outside_validated_region_count', 0)}")
            print(f"[TARGET] near servo limit count: {scan_stats.get('near_servo_limit_count', 0)}")
            pending_count = sum(1 for obj in objects if obj["status"] == "pending")
            print(f"[TARGET] pending object count: {pending_count}")
            if args.print_object_list:
                print_object_list(objects)
            if not objects:
                print("[STATE] No locked valid objects found in scan. Stopping.")
                break

            scan_consumed = False
            while processed < args.max_objects:
                target_obj = choose_target(objects, args.selection_strategy, board_width_cfg, board_height_cfg)
                if target_obj is None:
                    if not args.send:
                        print("[STATE] No more pending objects in current scan.")
                    break
                print(f"[TARGET] selected object id: {target_obj['id']}")

                target = target_obj["_det"]
                try:
                    sequence, sequence_meta = build_pick_sequence_for_target(
                        target,
                        semi_auto_cfg,
                        kin_cfg,
                        servo_cfg,
                        pose_cfg,
                        taught_cfg,
                        transform_cfg,
                        ik_servo_cal_cfg,
                        args,
                        workspace_profile=target_obj.get("workspace_profile"),
                    )
                except ValueError as exc:
                    text = str(exc)
                    if "near servo limit:" in text:
                        print(f"[SAFETY] rejected near servo limit: {text.split('near servo limit:', 1)[1].strip()}", file=sys.stderr)
                    else:
                        print(f"[IK][ERROR] {text}", file=sys.stderr)
                    return 2

                ok, message = validate_board_target(
                    sequence_meta["adjusted_board_x_cm"],
                    sequence_meta["adjusted_board_y_cm"],
                    board_width_cfg,
                    board_height_cfg,
                )
                if not ok and not args.allow_out_of_board_target:
                    print(f"[SAFETY][ERROR] {message}", file=sys.stderr)
                    return 2
                if not ok:
                    print(f"[SAFETY] {message} (override enabled)")

                region_ok, region_message = validate_pick_region(
                    sequence_meta["adjusted_board_x_cm"],
                    sequence_meta["adjusted_board_y_cm"],
                    semi_auto_cfg,
                )
                if not region_ok and not args.allow_outside_validated_region:
                    print("[SAFETY] rejected outside validated pick region", file=sys.stderr)
                    print(f"[SAFETY][ERROR] {region_message}", file=sys.stderr)
                    return 2
                if not region_ok:
                    print("[SAFETY] Target outside validated pick region. Operator override enabled.")

                steps = [build_home_step(), *sequence]
                print(
                    f"\n[TARGET] id={target_obj['id']} group={target_obj['group']} raw={target_obj['raw_class']} "
                    f"conf={target_obj['confidence']:.3f}"
                )
                print(
                    f"[TARGET_OFFSET] detected board=({target_obj['board_x_cm']:.2f}, {target_obj['board_y_cm']:.2f})"
                )
                print(
                    f"[TARGET_OFFSET] adjusted board=({sequence_meta['adjusted_board_x_cm']:.2f}, "
                    f"{sequence_meta['adjusted_board_y_cm']:.2f})"
                )
                if args.use_workspace_correction_map:
                    print(
                        f"[WORKSPACE] selected region={sequence_meta['workspace_region']} "
                        f"status={sequence_meta['workspace_region_status']}"
                    )
                    print(
                        f"[WORKSPACE] offset=({sequence_meta['grasp_offset_x_cm']:.2f}, "
                        f"{sequence_meta['grasp_offset_y_cm']:.2f}) cm"
                    )
                    print(
                        "[WORKSPACE] z policy="
                        f"safe_hover={sequence_meta['workspace_safe_hover_z_m']:.3f} "
                        f"pre_pick={sequence_meta['workspace_pre_pick_z_m']:.3f} "
                        f"lift={sequence_meta['workspace_lift_z_m']:.3f} m"
                    )
                print("[SEQUENCE] Full sequence preview:")
                print(format_sequence_preview(steps))

                if args.send:
                    if args.confirm_each:
                        if not require_exact("TARGET", f"Type TARGET to accept object id={target_obj['id']}"):
                            print("Target confirmation cancelled.")
                            scan_consumed = True
                            break
                    elif not args.auto:
                        if not require_exact("START", f"Type START to send object id={target_obj['id']} sequence."):
                            print("Sequence cancelled.")
                            scan_consumed = True
                            break

                    assert ser is not None
                    send_steps(ser, steps)

                attempted_targets.append(
                    {
                        "adjusted_board_x_cm": sequence_meta["adjusted_board_x_cm"],
                        "adjusted_board_y_cm": sequence_meta["adjusted_board_y_cm"],
                    }
                )
                target_obj["status"] = "attempted"
                processed += 1
                scan_consumed = True
                if args.send:
                    break

            if not scan_consumed and pending_count == 0:
                print("[STATE] No valid pending target remains. Stopping.")
                break

        return 0
    except KeyboardInterrupt:
        print("\n[SAFETY] Interrupted by user.")
        if ser is not None:
            try:
                lines = send_command(ser, "STOP")
                for line in lines:
                    print(f"  {line}")
            except Exception:
                pass
        return 2
    except RuntimeError as exc:
        if str(exc) == "ESP32 reset detected during motion":
            print("[SERIAL][ERROR] ESP32 reset detected during motion", file=sys.stderr)
        else:
            print(f"[SERIAL][ERROR] {exc}", file=sys.stderr)
        if ser is not None:
            try:
                lines = send_command(ser, "STOP")
                for line in lines:
                    print(f"  {line}")
            except Exception:
                pass
        return 2
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if ser is not None:
            ser.close()


if __name__ == "__main__":
    sys.exit(main())
