#!/usr/bin/env python3
"""
test_single_object_pick_place.py

Phase 10 guarded single-object YOLO pick-place test.

This tool detects one stable inside-board object, computes a fixed pick/place
sequence using existing IK dry-run logic, and only sends commands after
explicit confirmation. It does not implement autonomous sorting loops, ROS2,
GUI, or configuration writes.
"""

import argparse
import sys
import time
from typing import Any

import cv2

from test_esp32_manual_move import open_serial, send_command
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


DEFAULT_SERIAL_TIMEOUT = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 10 guarded single-object YOLO pick-place test.",
    )
    parser.add_argument("--port", help="Serial port path, required unless --dry-run")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--cam", type=int, default=DEFAULT_CAM)
    parser.add_argument("--z-hover", type=float, default=0.12)
    parser.add_argument("--z-pick", type=float, default=0.05)
    parser.add_argument("--z-lift", type=float, default=0.12)
    parser.add_argument("--z-place-hover", type=float, default=0.12)
    parser.add_argument("--z-place", type=float, default=0.08)
    parser.add_argument("--target-group", choices=["CAKE", "DONUT", "ANY"], default="ANY")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--confirm-each-step", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_SERIAL_TIMEOUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["auto", "cpu", "0"])
    parser.add_argument("--roi", type=str)
    parser.add_argument("--board-config", default=DEFAULT_BOARD_CONFIG)
    parser.add_argument("--pick-place-config", default="ros2_ws/src/robot_arm_5dof/config/pick_place_config.yaml")
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


def require_exact(label: str, prompt: str) -> bool:
    print(prompt)
    try:
        typed = input("> ").strip()
    except EOFError:
        return False
    return typed == label


def require_enter(prompt: str) -> bool:
    print(prompt)
    try:
        typed = input()
    except EOFError:
        return False
    return typed == ""


def choose_locked_target(detections: list[dict[str, Any]], target_group: str) -> dict[str, Any] | None:
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


def same_target(det_a: dict[str, Any], det_b: dict[str, Any], board_tol_cm: float = 2.0) -> bool:
    if det_a.get("group") != det_b.get("group"):
        return False
    dx = abs(float(det_a["board_x_cm"]) - float(det_b["board_x_cm"]))
    dy = abs(float(det_a["board_y_cm"]) - float(det_b["board_y_cm"]))
    return dx <= board_tol_cm and dy <= board_tol_cm


def detect_target(args: argparse.Namespace, model, H, width_cm: float, height_cm: float, cap, tracker: StabilityTracker) -> tuple[dict[str, Any], Any]:
    deadline = time.time() + args.timeout
    last_overlay = None
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        raw = run_yolo(model, frame, args.roi, args.conf, args.imgsz, args.max_det, resolve_device(args.device))
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
        last_overlay = draw_overlay(frame, mapped_acc, rejected, 0.0, len(raw), args.show_rejected)

        target = choose_locked_target(mapped_acc, args.target_group)
        if target is not None:
            return target, last_overlay

        if args.show:
            cv2.imshow(WINDOW_NAME, last_overlay)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                raise KeyboardInterrupt

    raise TimeoutError("No stable inside-board target found before timeout")


def resolve_drop_zone(target_group: str, pick_place_cfg: dict[str, Any]) -> tuple[str, float, float]:
    key = "cake" if target_group == "CAKE" else "donut"
    zone = pick_place_cfg.get("drop_zones", {}).get(key, {})
    bx = zone.get("board_x_cm")
    by = zone.get("board_y_cm")
    label = zone.get("label", key.upper())
    if bx is None or by is None:
        raise ValueError(f"Drop zone {label} is not calibrated in pick_place_config.yaml")
    return str(label), float(bx), float(by)


def compute_move_step(
    step_name: str,
    board_x_cm: float,
    board_y_cm: float,
    z_m: float,
    gripper_angle: int,
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    transform_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    robot_x, robot_y = board_to_robot(board_x_cm, board_y_cm, transform_cfg)
    ik_result = solve_planar_ik(robot_x, robot_y, z_m, kin_cfg, args.tcp_offset_mode, 0.0, 1.0)
    if "unreachable_reason" in ik_result:
        raise ValueError(f"{step_name}: {ik_result['unreachable_reason']}")

    candidate = select_candidate(ik_result, args.solution)
    servos = candidate_servos(candidate, kin_cfg)
    limit_ok, limit_lines = validate_servos(servos, servo_cfg)
    move_angles = rounded_move_safe_angles(servos, gripper_angle)
    rounded_failures = validate_move_safe_angles(move_angles, servo_cfg)
    failures = []
    if not limit_ok:
        failures.extend(line[4:] if line.startswith("BAD ") else line for line in limit_lines if line.startswith("BAD "))
    failures.extend(rounded_failures)
    if failures:
        raise ValueError(f"{step_name}: servo limit failure: {'; '.join(failures)}")

    return {
        "name": step_name,
        "board_x_cm": board_x_cm,
        "board_y_cm": board_y_cm,
        "robot_x_m": robot_x,
        "robot_y_m": robot_y,
        "z_m": z_m,
        "solution": candidate["name"],
        "move_angles": move_angles,
        "command": build_move_command(move_angles),
    }


def print_detection_report(target: dict[str, Any], drop_zone_label: str, drop_board_x: float, drop_board_y: float) -> None:
    print("Detected object:")
    print(f"  group: {target['group']}")
    print(f"  raw_class: {target['raw']}")
    print(f"  confidence: {target['conf']:.3f}")
    print(f"  pixel center: u={target['u']} v={target['v']}")
    print(f"  board target: x={target['board_x_cm']:.2f} cm y={target['board_y_cm']:.2f} cm")
    print(f"  destination: {drop_zone_label} at x={drop_board_x:.2f} cm y={drop_board_y:.2f} cm")


def print_sequence(steps: list[dict[str, Any]]) -> None:
    print("\nGenerated sequence:")
    for idx, step in enumerate(steps, start=1):
        if step["type"] == "HOME":
            print(f"  {idx:02d}. {step['name']}: HOME")
        else:
            print(
                f"  {idx:02d}. {step['name']}: board=({step['board_x_cm']:.2f},{step['board_y_cm']:.2f}) "
                f"robot=({step['robot_x_m']:.4f},{step['robot_y_m']:.4f}) z={step['z_m']:.2f} "
                f"{step['command']}"
            )


def send_sequence(ser, steps: list[dict[str, Any]], confirm_each_step: bool) -> None:
    for step in steps:
        if step["type"] == "HOME":
            lines = send_command(ser, "HOME")
            print("COMMAND HOME")
            for line in lines:
                print(f"  {line}")
            require_line(lines, "ACK HOME", step["name"])
            require_line(lines, "DONE HOME", step["name"])
            continue

        if confirm_each_step:
            if not require_enter(f"Press ENTER to send {step['name']} ({step['command']})"):
                raise RuntimeError(f"{step['name']} cancelled before send")
        lines = send_command(ser, step["command"])
        print(f"COMMAND {step['command']}")
        for line in lines:
            print(f"  {line}")
        require_line(lines, "ACK MOVE_SAFE", step["name"])
        require_line(lines, "DONE MOVE_SAFE", step["name"])


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
    pick_place_cfg = load_required(args.pick_place_config, "pick/place config")
    serial_cfg = load_required(args.serial_config, "serial config")
    _ = serial_cfg

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

    home = pose_cfg.get("poses", {}).get("HOME_SAFE", {})
    home_gripper = int(home.get("ch6", 45))
    gripper_cfg = servo_cfg.get("servos", {}).get("ch6", {})
    gripper_open = int(gripper_cfg.get("open_angle_deg", 50))
    gripper_close = int(gripper_cfg.get("close_angle_deg", 15))

    _, H, H_path, width_cm, height_cm = load_board_and_homography(args.board_config, None)
    print(f"[BOARD] Using homography: {H_path}")

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
    try:
        target, overlay = detect_target(args, model, H, width_cm, height_cm, cap, tracker)
        if args.show and overlay is not None:
            cv2.imshow(WINDOW_NAME, overlay)
            cv2.waitKey(50)
    except TimeoutError as exc:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        print("\n[INFO] Stopped before target selection.")
        return 0

    if args.send:
        # Re-check target just before start so we do not move on stale coordinates.
        try:
            recheck_tracker = StabilityTracker(stable_frames=args.stable_frames, center_tol=args.center_tolerance)
            recheck, _ = detect_target(args, model, H, width_cm, height_cm, cap, recheck_tracker)
            if not same_target(target, recheck):
                cap.release()
                if args.show:
                    cv2.destroyAllWindows()
                print("[ERROR] Target changed or was lost before start. Aborting.", file=sys.stderr)
                return 2
        except Exception:
            cap.release()
            if args.show:
                cv2.destroyAllWindows()
            print("[ERROR] Target lost before start. Aborting.", file=sys.stderr)
            return 2

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    if not (0.0 <= float(target["board_x_cm"]) <= width_cm and 0.0 <= float(target["board_y_cm"]) <= height_cm):
        print("[ERROR] Target is outside board limits. Aborting.", file=sys.stderr)
        return 2

    try:
        drop_label, drop_board_x, drop_board_y = resolve_drop_zone(str(target["group"]), pick_place_cfg)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print_detection_report(target, drop_label, drop_board_x, drop_board_y)

    try:
        steps = [
            {"type": "HOME", "name": "HOME start"},
            {"type": "MOVE_SAFE", **compute_move_step("Move hover object", float(target["board_x_cm"]), float(target["board_y_cm"]), args.z_hover, gripper_open, kin_cfg, servo_cfg, transform_cfg, args)},
            {"type": "MOVE_SAFE", **compute_move_step("Move pick object", float(target["board_x_cm"]), float(target["board_y_cm"]), args.z_pick, gripper_open, kin_cfg, servo_cfg, transform_cfg, args)},
            {"type": "MOVE_SAFE", **compute_move_step("Gripper close", float(target["board_x_cm"]), float(target["board_y_cm"]), args.z_pick, gripper_close, kin_cfg, servo_cfg, transform_cfg, args)},
            {"type": "MOVE_SAFE", **compute_move_step("Move lift object", float(target["board_x_cm"]), float(target["board_y_cm"]), args.z_lift, gripper_close, kin_cfg, servo_cfg, transform_cfg, args)},
            {"type": "MOVE_SAFE", **compute_move_step("Move bowl hover", drop_board_x, drop_board_y, args.z_place_hover, gripper_close, kin_cfg, servo_cfg, transform_cfg, args)},
            {"type": "MOVE_SAFE", **compute_move_step("Move bowl place", drop_board_x, drop_board_y, args.z_place, gripper_close, kin_cfg, servo_cfg, transform_cfg, args)},
            {"type": "MOVE_SAFE", **compute_move_step("Gripper open", drop_board_x, drop_board_y, args.z_place, gripper_open, kin_cfg, servo_cfg, transform_cfg, args)},
            {"type": "HOME", "name": "HOME end"},
        ]
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print_sequence(steps)

    if not args.send:
        print("\nMode: DRY RUN")
        return 0

    if not require_exact("START", "Type START to begin single-object pick-place."):
        print("Pick-place sequence cancelled.")
        return 0

    try:
        ser = open_serial(args.port, args.baud, args.timeout)
    except Exception as exc:  # pragma: no cover - hardware-dependent path
        print(f"[ERROR] Failed to open serial port {args.port}: {exc}", file=sys.stderr)
        return 2

    try:
        send_sequence(ser, steps, args.confirm_each_step)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
