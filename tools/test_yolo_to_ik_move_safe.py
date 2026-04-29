#!/usr/bin/env python3
"""
test_yolo_to_ik_move_safe.py

Phase 9 guarded YOLO target to IK MOVE_SAFE hover test.

This tool detects one accepted board target with the existing YOLO mapping
pipeline, converts it to robot coordinates, computes IK, validates limits, and
only sends MOVE_SAFE after explicit confirmation.
"""

import argparse
import sys
import time
from typing import Any

import cv2

from test_esp32_manual_move import open_serial, require_confirmation, send_command
from test_ik_dry_run import (
    DEFAULT_KINEMATICS_CONFIG,
    DEFAULT_POSE_CONFIG,
    DEFAULT_SERVO_CONFIG,
    DEFAULT_TRANSFORM_CONFIG,
    board_to_robot,
    candidate_servos,
    load_required,
    solve_planar_ik,
    validate_servos,
)
from test_ik_to_move_safe import (
    DEFAULT_SERIAL_CONFIG,
    build_move_command,
    require_line,
    rounded_move_safe_angles,
    select_candidate,
    validate_move_safe_angles,
)
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


DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 8.0
DEFAULT_Z = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 9 guarded YOLO target to IK MOVE_SAFE hover test.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cam", type=int, default=DEFAULT_CAM)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--z", type=float, default=DEFAULT_Z)
    parser.add_argument("--target-group", choices=["CAKE", "DONUT", "ANY"], default="ANY")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["auto", "cpu", "0"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--home-first", action="store_true")
    parser.add_argument("--return-home", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--roi", type=str)
    parser.add_argument("--board-config", default=DEFAULT_BOARD_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--serial-config", default=DEFAULT_SERIAL_CONFIG)
    parser.add_argument("--tcp-offset-mode", default="none")
    parser.add_argument("--solution", choices=["elbow_up", "elbow_down", "best"], default="elbow_up")
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
    return parser.parse_args()


def choose_target(detections: list[dict[str, Any]], target_group: str) -> dict[str, Any] | None:
    filtered = []
    for det in detections:
        if target_group != "ANY" and det.get("group") != target_group:
            continue
        filtered.append(det)
    if not filtered:
        return None

    locked = [det for det in filtered if det.get("status") == "LOCKED"]
    pool = locked if locked else filtered
    pool.sort(key=lambda det: (0 if det.get("status") == "LOCKED" else 1, -float(det.get("conf", 0.0))))
    return pool[0]


def print_target_report(target: dict[str, Any], robot_x: float, robot_y: float) -> None:
    print("Detected target:")
    print(f"  group: {target['group']}")
    print(f"  raw_class: {target['raw']}")
    print(f"  confidence: {target['conf']:.3f}")
    print(f"  pixel center: u={target['u']} v={target['v']}")
    print(f"  board: x={target['board_x_cm']:.2f} cm y={target['board_y_cm']:.2f} cm")
    print(f"  robot: x={robot_x:.4f} m y={robot_y:.4f} m")


def print_ik_report(
    args: argparse.Namespace,
    target: dict[str, Any],
    robot_x: float,
    robot_y: float,
    candidate: dict[str, Any],
    servos: dict[str, float],
    move_angles: list[int],
    failures: list[str],
) -> None:
    print("\nIK hover result:")
    print(f"  z: {args.z:.4f} m")
    print(f"  selected solution: {candidate['name']}")
    print(f"  tcp_offset_mode: {args.tcp_offset_mode}")
    print(f"  mode: {'HARDWARE SEND' if args.send and not args.dry_run else 'DRY RUN'}")
    print("  computed servo angles CH1-CH5:")
    for ch in ("ch1", "ch2", "ch3", "ch4", "ch5"):
        print(f"    {ch}: {servos[ch]:.2f}")
    print("  final MOVE_SAFE angles CH1-CH6:")
    print("    " + " ".join(f"CH{index + 1}={value}" for index, value in enumerate(move_angles)))
    print(f"  servo limit: {'PASS' if not failures else 'FAIL'}")
    if failures:
        for failure in failures:
            print(f"    {failure}")
    print("  intended command:")
    print(f"    {build_move_command(move_angles)}")


def main() -> int:
    args = parse_args()
    args.conf = max(0.0, min(1.0, args.conf))

    if args.dry_run and args.send:
        print("[ERROR] Use either --dry-run or --send, not both", file=sys.stderr)
        return 2
    if args.send and args.port is None:
        print("[ERROR] --port is required when --send is used", file=sys.stderr)
        return 2

    kin_cfg = load_required(args.kinematics_config, "kinematics config")
    servo_cfg = load_required(args.servo_config, "servo config")
    pose_cfg = load_required(args.pose_config, "pose config")
    transform_cfg = load_required(args.transform_config, "transform config")
    serial_cfg = load_required(args.serial_config, "serial config")
    _ = serial_cfg

    _, H, H_path, width_cm, height_cm = load_board_and_homography(args.board_config, None)
    print(f"[BOARD] Using homography: {H_path}")

    supported_modes = kin_cfg.get("ik", {}).get("tcp_offset_modes_supported", ["none", "planar", "vertical_down", "mixed"])
    if args.tcp_offset_mode not in supported_modes:
        print(f"[ERROR] Unsupported tcp offset mode {args.tcp_offset_mode!r}. Supported: {supported_modes}", file=sys.stderr)
        return 2

    if args.roi:
        try:
            args.roi = parse_roi(args.roi)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2

    device = resolve_device(args.device)
    print_environment(device)
    model = load_model(args.model, device)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.cam}", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    tracker = StabilityTracker(stable_frames=args.stable_frames, center_tol=args.center_tolerance)
    selected_target: dict[str, Any] | None = None
    selected_overlay = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Cannot read frame")
                time.sleep(0.2)
                continue

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

            selected_overlay = draw_overlay(frame, mapped_acc, rejected, 0.0, len(raw), args.show_rejected)
            selected_target = choose_target(mapped_acc, args.target_group)
            if selected_target is not None:
                break

            if args.show:
                cv2.imshow(WINDOW_NAME, selected_overlay)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("[INFO] Stopped without selecting a target.")
                    return 0
    except KeyboardInterrupt:
        print("\n[INFO] Stopped without selecting a target.")
        return 0
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    assert selected_target is not None
    robot_x, robot_y = board_to_robot(
        float(selected_target["board_x_cm"]),
        float(selected_target["board_y_cm"]),
        transform_cfg,
    )
    print_target_report(selected_target, robot_x, robot_y)

    ik_result = solve_planar_ik(robot_x, robot_y, args.z, kin_cfg, args.tcp_offset_mode, 0.0, 1.0)
    if "unreachable_reason" in ik_result:
        print("[ERROR] IK unreachable:", ik_result["unreachable_reason"], file=sys.stderr)
        return 2

    try:
        candidate = select_candidate(ik_result, args.solution)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    servos = candidate_servos(candidate, kin_cfg)
    limit_ok, limit_lines = validate_servos(servos, servo_cfg)

    home = pose_cfg.get("poses", {}).get("HOME_SAFE", {})
    home_gripper = home.get("ch6")
    if home_gripper is None:
        print("[ERROR] HOME_SAFE ch6 is required in pose_config.yaml", file=sys.stderr)
        return 2

    move_angles = rounded_move_safe_angles(servos, int(home_gripper))
    rounded_failures = validate_move_safe_angles(move_angles, servo_cfg)
    all_failures = []
    if not limit_ok:
        all_failures.extend(line[4:] if line.startswith("BAD ") else line for line in limit_lines if line.startswith("BAD "))
    all_failures.extend(rounded_failures)

    print_ik_report(args, selected_target, robot_x, robot_y, candidate, servos, move_angles, all_failures)

    if all_failures:
        print("\n[ERROR] MOVE_SAFE send blocked because servo limits did not pass.", file=sys.stderr)
        return 2

    move_command = build_move_command(move_angles)
    if args.dry_run or not args.send:
        return 0

    assert args.port is not None
    try:
        ser = open_serial(args.port, args.baud, args.timeout)
    except Exception as exc:  # pragma: no cover - hardware-dependent path
        print(f"[ERROR] Failed to open serial port {args.port}: {exc}", file=sys.stderr)
        return 2

    try:
        if args.home_first:
            if not require_confirmation("HOME", "Type HOME to send HOME."):
                print("HOME command cancelled.")
                return 0
            lines = send_command(ser, "HOME")
            print("COMMAND HOME")
            for line in lines:
                print(f"  {line}")
            require_line(lines, "ACK HOME", "HOME")
            require_line(lines, "DONE HOME", "HOME")

        if not require_confirmation("MOVE", "Type MOVE to send YOLO IK MOVE_SAFE."):
            print("MOVE_SAFE command cancelled.")
            return 0
        lines = send_command(ser, move_command)
        print(f"COMMAND {move_command}")
        for line in lines:
            print(f"  {line}")
        require_line(lines, "ACK MOVE_SAFE", "MOVE_SAFE")
        require_line(lines, "DONE MOVE_SAFE", "MOVE_SAFE")

        if args.return_home:
            if not require_confirmation("HOME", "Type HOME to return HOME_SAFE."):
                print("Return HOME command cancelled.")
                return 0
            lines = send_command(ser, "HOME")
            print("COMMAND HOME")
            for line in lines:
                print(f"  {line}")
            require_line(lines, "ACK HOME", "HOME return")
            require_line(lines, "DONE HOME", "HOME return")
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
