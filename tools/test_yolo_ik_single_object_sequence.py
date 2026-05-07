#!/usr/bin/env python3
"""
test_yolo_ik_single_object_sequence.py

Guarded semi-auto single-object YOLO -> board -> robot -> IK sequence test.

This tool is dry-run first. It detects one locked object on the checkerboard,
builds a guarded single-object pick-place sequence, and only sends live serial
commands after explicit confirmation.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import cv2

from ik_servo_calibration_utils import (
    DEFAULT_IK_SERVO_CALIBRATION_CONFIG,
    apply_ik_servo_calibration,
    apply_z_mode_correction,
    load_ik_servo_calibration,
)
from test_esp32_manual_move import open_serial, send_command
from test_ik_dry_run import (
    DEFAULT_KINEMATICS_CONFIG,
    DEFAULT_POSE_CONFIG,
    DEFAULT_SERVO_CONFIG,
    DEFAULT_TRANSFORM_CONFIG,
    board_to_robot,
    candidate_servos_with_debug,
    load_required,
    solve_planar_ik,
    validate_servos,
)
from test_ik_to_move_safe import (
    DEFAULT_SERIAL_CONFIG,
    build_move_command,
    clamp_move_safe_angles,
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


DEFAULT_TAUGHT_POSE_CONFIG = "ros2_ws/src/robot_arm_5dof/config/taught_pick_place_poses.yaml"
DEFAULT_PICK_PLACE_CONFIG = "ros2_ws/src/robot_arm_5dof/config/pick_place_config.yaml"
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 8.0
DEFAULT_TIMEOUT_SEC = 20.0
DEFAULT_Z_SAFE_HOVER = 0.055
DEFAULT_Z_PRE_PICK = 0.015
DEFAULT_Z_LIFT = 0.125


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded semi-auto YOLO-to-IK single-object pick-place sequence test.",
    )
    parser.add_argument("--cam", type=int, default=DEFAULT_CAM)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--yes-i-understand-hardware-risk", action="store_true")
    parser.add_argument("--confirm-target", action="store_true")
    parser.add_argument("--grasp-board-x-offset-cm", type=float, default=0.0)
    parser.add_argument("--grasp-board-y-offset-cm", type=float, default=0.0)
    parser.add_argument("--target-group", default="any", help="CAKE, DONUT, or any")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--stable-frames", type=int, default=DEFAULT_STABLE_FRAMES)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["auto", "cpu", "0"])
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--roi", type=str)
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
    parser.add_argument("--center-tolerance", type=int, default=DEFAULT_CENTER_TOL)
    parser.add_argument("--show-rejected", action="store_true")
    parser.add_argument("--allow-out-of-board-target", action="store_true")
    parser.add_argument("--home-first", action="store_true", default=True)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--tcp-offset-mode", default="none")
    parser.add_argument("--solution", choices=["elbow_up", "elbow_down", "best"], default="elbow_up")
    parser.add_argument("--use-ik-servo-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-z-mode-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ik-servo-calibration-config", default=DEFAULT_IK_SERVO_CALIBRATION_CONFIG)
    parser.add_argument("--board-config", default=DEFAULT_BOARD_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--taught-pose-config", default=DEFAULT_TAUGHT_POSE_CONFIG)
    parser.add_argument("--pick-place-config", default=DEFAULT_PICK_PLACE_CONFIG)
    parser.add_argument("--serial-config", default=DEFAULT_SERIAL_CONFIG)
    return parser.parse_args()


def require_exact(label: str, prompt: str) -> bool:
    print(prompt)
    try:
        typed = input("> ").strip()
    except EOFError:
        return False
    return typed == label


def normalize_target_group(value: str) -> str:
    upper = str(value).strip().upper()
    if upper in {"ANY", "CAKE", "DONUT"}:
        return upper
    raise ValueError("--target-group must be one of: ANY, CAKE, DONUT")


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


def load_named_pose(name: str, taught_cfg: dict[str, Any], pose_cfg: dict[str, Any]) -> dict[str, Any]:
    taught_pose = taught_cfg.get("poses", {}).get(name)
    if isinstance(taught_pose, dict):
        source = f"taught:{name}"
        pose = taught_pose
    else:
        base_pose = pose_cfg.get("poses", {}).get(name)
        if not isinstance(base_pose, dict):
            raise ValueError(f"Required pose {name} not found in taught or base pose config")
        source = f"pose_config:{name}"
        pose = base_pose

    result = {}
    for channel in ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6"):
        value = pose.get(channel)
        if value is None:
            raise ValueError(f"Required pose {name} from {source} has null {channel}")
        result[channel] = int(round(float(value)))
    result["_source"] = source
    return result


def clamp_pose_angles(move_angles: list[int], servo_cfg: dict[str, Any]) -> tuple[list[int], list[str], list[str]]:
    clamped, clamp_notes = clamp_move_safe_angles(move_angles, servo_cfg)
    failures = validate_move_safe_angles(clamped, servo_cfg)
    return clamped, clamp_notes, failures


def gripper_open_close_angles(servo_cfg: dict[str, Any]) -> tuple[int, int]:
    gripper_cfg = servo_cfg.get("gripper_calibration", {})
    ch6_cfg = servo_cfg.get("servos", {}).get("ch6", {})
    open_deg = int(round(float(gripper_cfg.get("open_deg", ch6_cfg.get("open_angle_deg", 50)))))
    close_deg = int(round(float(gripper_cfg.get("close_full_deg", ch6_cfg.get("close_angle_deg", 20)))))
    return open_deg, close_deg


def compute_ik_step(
    step_name: str,
    board_x_cm: float,
    board_y_cm: float,
    z_m: float,
    z_mode: str,
    gripper_angle: int,
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    pose_cfg: dict[str, Any],
    transform_cfg: dict[str, Any],
    ik_servo_cal_cfg: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    robot_x_m, robot_y_m = board_to_robot(board_x_cm, board_y_cm, transform_cfg)
    ik_result = solve_planar_ik(robot_x_m, robot_y_m, z_m, kin_cfg, args.tcp_offset_mode, 0.0, 1.0)
    if "unreachable_reason" in ik_result:
        raise ValueError(f"{step_name}: {ik_result['unreachable_reason']}")

    candidate = select_candidate(ik_result, args.solution)
    servos, ch1_debug = candidate_servos_with_debug(candidate, kin_cfg, servo_cfg)
    calibration_logs: list[str] = []
    z_mode_logs: list[str] = []
    if ik_servo_cal_cfg is not None:
        if args.use_ik_servo_calibration:
            servos, calibration_logs = apply_ik_servo_calibration(servos, ik_servo_cal_cfg)
        if args.use_z_mode_correction:
            servos, z_mode_logs = apply_z_mode_correction(servos, ik_servo_cal_cfg, z_mode)

    limit_ok, limit_lines = validate_servos(servos, servo_cfg)
    move_angles = rounded_move_safe_angles(servos, gripper_angle)
    move_angles, clamp_notes = clamp_move_safe_angles(move_angles, servo_cfg)
    rounded_failures = validate_move_safe_angles(move_angles, servo_cfg)
    failures = []
    if not limit_ok:
        failures.extend(line[4:] if line.startswith("BAD ") else line for line in limit_lines if line.startswith("BAD "))
    failures.extend(rounded_failures)
    if failures:
        raise ValueError(f"{step_name}: servo limit failure: {'; '.join(failures)}")

    return {
        "type": "MOVE_SAFE",
        "name": step_name,
        "source": "ik",
        "board_x_cm": board_x_cm,
        "board_y_cm": board_y_cm,
        "robot_x_m": robot_x_m,
        "robot_y_m": robot_y_m,
        "z_m": z_m,
        "z_mode": z_mode,
        "solution": candidate["name"],
        "servos": servos,
        "ch1_debug": ch1_debug,
        "calibration_logs": calibration_logs,
        "z_mode_logs": z_mode_logs,
        "move_angles": move_angles,
        "clamp_notes": clamp_notes,
        "command": build_move_command(move_angles),
    }


def build_pose_step(step_name: str, pose_name: str, pose: dict[str, Any], servo_cfg: dict[str, Any]) -> dict[str, Any]:
    move_angles = [pose[channel] for channel in ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6")]
    move_angles, clamp_notes, failures = clamp_pose_angles(move_angles, servo_cfg)
    if failures:
        raise ValueError(f"{step_name}: taught pose {pose_name} servo failure: {'; '.join(failures)}")
    return {
        "type": "MOVE_SAFE",
        "name": step_name,
        "source": f"pose:{pose_name}",
        "move_angles": move_angles,
        "clamp_notes": clamp_notes,
        "command": build_move_command(move_angles),
    }


def build_gripper_variant_step(step_name: str, source_step: dict[str, Any], ch6_value: int, servo_cfg: dict[str, Any]) -> dict[str, Any]:
    move_angles = list(source_step["move_angles"])
    move_angles[5] = int(ch6_value)
    move_angles, clamp_notes, failures = clamp_pose_angles(move_angles, servo_cfg)
    if failures:
        raise ValueError(f"{step_name}: gripper variant servo failure: {'; '.join(failures)}")
    return {
        "type": "MOVE_SAFE",
        "name": step_name,
        "source": f"{source_step['name']}+gripper",
        "move_angles": move_angles,
        "clamp_notes": clamp_notes,
        "command": build_move_command(move_angles),
    }


def build_home_step() -> dict[str, Any]:
    return {"type": "HOME", "name": "HOME_SAFE"}


def print_target_report(target: dict[str, Any], adjusted_board_x: float, adjusted_board_y: float, robot_x: float, robot_y: float) -> None:
    print("[TARGET] selected object:")
    print(f"  raw_class: {target['raw']}")
    print(f"  group: {target['group']}")
    print(f"  confidence: {target['conf']:.3f}")
    print(f"  pixel u/v: {target['u']} / {target['v']}")
    print(f"  board_x_cm / board_y_cm: {target['board_x_cm']:.2f} / {target['board_y_cm']:.2f}")
    print(f"[TARGET_OFFSET] detected board=({target['board_x_cm']:.2f}, {target['board_y_cm']:.2f}) cm")
    print(f"[TARGET_OFFSET] adjusted board=({adjusted_board_x:.2f}, {adjusted_board_y:.2f}) cm")
    print(f"[TARGET] robot target=({robot_x:.4f}, {robot_y:.4f}) m")


def print_sequence(steps: list[dict[str, Any]], args: argparse.Namespace) -> None:
    print("\n============================================================")
    print("[IK] SEMI-AUTO SINGLE OBJECT SEQUENCE")
    print("============================================================")
    print(f"[IK] tcp_offset_mode: {args.tcp_offset_mode}")
    print(f"[IK] use_ik_servo_calibration: {args.use_ik_servo_calibration}")
    print(f"[IK] use_z_mode_correction: {args.use_z_mode_correction}")
    print(f"[SAFETY] mode: {'HARDWARE SEND' if args.send and not args.dry_run else 'DRY RUN'}")
    print("[SAFETY] Semi-auto single-object test only. Not autonomous validation.")
    print("Sequence:")
    for index, step in enumerate(steps, start=1):
        if step["type"] == "HOME":
            print(f"  {index:02d}. {step['name']}: HOME")
            continue
        print(f"  {index:02d}. {step['name']}: {step['command']} ({step['source']})")
        if "board_x_cm" in step:
            print(
                f"      board=({step['board_x_cm']:.2f},{step['board_y_cm']:.2f}) "
                f"robot=({step['robot_x_m']:.4f},{step['robot_y_m']:.4f}) "
                f"z={step['z_m']:.3f} z_mode={step['z_mode']}"
            )
        for note in step.get("clamp_notes", []):
            print(f"      [SAFETY] {note}")


def detect_target(
    args: argparse.Namespace,
    model,
    H,
    width_cm: float,
    height_cm: float,
    cap,
    tracker: StabilityTracker,
    device: str,
) -> dict[str, Any]:
    deadline = time.time() + args.timeout_sec
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
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
        overlay = draw_overlay(frame, mapped_acc, rejected, 0.0, len(raw), args.show_rejected)

        target = choose_locked_target(mapped_acc, args.target_group)
        if target is not None:
            return target

        if args.show:
            cv2.imshow(WINDOW_NAME, overlay)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                raise KeyboardInterrupt

    raise TimeoutError("No locked inside-board target found before timeout")


def send_sequence(ser, steps: list[dict[str, Any]]) -> None:
    for step in steps:
        if step["type"] == "HOME":
            lines = send_command(ser, "HOME")
            print("COMMAND HOME")
            for line in lines:
                print(f"  {line}")
            require_line(lines, "ACK HOME", step["name"])
            require_line(lines, "DONE HOME", step["name"])
            continue

        lines = send_command(ser, step["command"])
        print(f"COMMAND {step['command']}")
        for line in lines:
            print(f"  {line}")
        require_line(lines, "ACK MOVE_SAFE", step["name"])
        require_line(lines, "DONE MOVE_SAFE", step["name"])


def main() -> int:
    args = parse_args()
    args.conf = max(0.0, min(1.0, args.conf))
    try:
        args.target_group = normalize_target_group(args.target_group)
    except ValueError as exc:
        print(f"[CONFIG][ERROR] {exc}", file=sys.stderr)
        return 2

    if args.dry_run and args.send:
        print("[SAFETY][ERROR] Use either --dry-run or --send, not both", file=sys.stderr)
        return 2
    if args.send and args.port is None:
        print("[SERIAL][ERROR] --port is required when --send is used", file=sys.stderr)
        return 2
    if args.send and not args.yes_i_understand_hardware_risk:
        print("[SAFETY][ERROR] Live semi-auto motion requires --yes-i-understand-hardware-risk.", file=sys.stderr)
        return 2

    kin_cfg = load_required(args.kinematics_config, "kinematics config")
    servo_cfg = load_required(args.servo_config, "servo config")
    pose_cfg = load_required(args.pose_config, "pose config")
    transform_cfg = load_required(args.transform_config, "transform config")
    serial_cfg = load_required(args.serial_config, "serial config")
    taught_cfg = load_required(args.taught_pose_config, "taught pose config")
    pick_place_cfg = load_required(args.pick_place_config, "pick/place config")
    ik_servo_cal_cfg = None
    if args.use_ik_servo_calibration or args.use_z_mode_correction:
        ik_servo_cal_cfg = load_ik_servo_calibration(args.ik_servo_calibration_config)
    _ = serial_cfg
    _ = pick_place_cfg

    supported_modes = kin_cfg.get("ik", {}).get("tcp_offset_modes_supported", ["none", "planar", "vertical_down", "mixed"])
    if args.tcp_offset_mode not in supported_modes:
        print(f"[CONFIG][ERROR] Unsupported tcp offset mode {args.tcp_offset_mode!r}. Supported: {supported_modes}", file=sys.stderr)
        return 2

    if args.roi:
        try:
            args.roi = parse_roi(args.roi)
        except ValueError as exc:
            print(f"[CONFIG][ERROR] {exc}", file=sys.stderr)
            return 2

    _, H, H_path, width_cm, height_cm = load_board_and_homography(args.board_config, None)
    print(f"[BOARD] Using homography: {H_path}")
    print(f"[BOARD] Board limits: x=[0,{width_cm:.2f}] cm y=[0,{height_cm:.2f}] cm")

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

    try:
        target = detect_target(args, model, H, width_cm, height_cm, cap, tracker, device)
    except TimeoutError as exc:
        print(f"[VISION][ERROR] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[INFO] Stopped before selecting a target.")
        return 0
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    adjusted_board_x = float(target["board_x_cm"]) + args.grasp_board_x_offset_cm
    adjusted_board_y = float(target["board_y_cm"]) + args.grasp_board_y_offset_cm
    if not (0.0 <= adjusted_board_x <= width_cm and 0.0 <= adjusted_board_y <= height_cm):
        if not args.allow_out_of_board_target:
            print(
                f"[SAFETY][ERROR] Adjusted board target ({adjusted_board_x:.2f}, {adjusted_board_y:.2f}) cm "
                f"is outside the board. Use --allow-out-of-board-target to override.",
                file=sys.stderr,
            )
            return 2
        print("[SAFETY] Adjusted board target is outside board limits, but override is enabled.")

    robot_x, robot_y = board_to_robot(adjusted_board_x, adjusted_board_y, transform_cfg)
    print_target_report(target, adjusted_board_x, adjusted_board_y, robot_x, robot_y)

    try:
        clear_pose = load_named_pose("CLEAR_TEST", taught_cfg, pose_cfg)
        home_pose = load_named_pose("HOME_SAFE", taught_cfg, pose_cfg)
        if target["group"] == "CAKE":
            drop_hover_pose = load_named_pose("CAKE_DROP_HOVER", taught_cfg, pose_cfg)
            drop_place_pose = load_named_pose("CAKE_DROP_PLACE", taught_cfg, pose_cfg)
        else:
            drop_hover_pose = load_named_pose("DONUT_DROP_HOVER", taught_cfg, pose_cfg)
            drop_place_pose = load_named_pose("DONUT_DROP_PLACE", taught_cfg, pose_cfg)
    except ValueError as exc:
        print(f"[CONFIG][ERROR] {exc}", file=sys.stderr)
        return 2

    open_deg, close_deg = gripper_open_close_angles(servo_cfg)

    try:
        safe_hover = compute_ik_step(
            "IK_SAFE_HOVER",
            adjusted_board_x,
            adjusted_board_y,
            DEFAULT_Z_SAFE_HOVER,
            "safe_hover",
            open_deg,
            kin_cfg,
            servo_cfg,
            pose_cfg,
            transform_cfg,
            ik_servo_cal_cfg,
            args,
        )
        pre_pick = compute_ik_step(
            "IK_PRE_PICK",
            adjusted_board_x,
            adjusted_board_y,
            DEFAULT_Z_PRE_PICK,
            "pre_pick",
            open_deg,
            kin_cfg,
            servo_cfg,
            pose_cfg,
            transform_cfg,
            ik_servo_cal_cfg,
            args,
        )
        close_gripper = build_gripper_variant_step("CLOSE_GRIPPER", pre_pick, close_deg, servo_cfg)
        lift = compute_ik_step(
            "IK_LIFT",
            adjusted_board_x,
            adjusted_board_y,
            DEFAULT_Z_LIFT,
            "lift",
            close_deg,
            kin_cfg,
            servo_cfg,
            pose_cfg,
            transform_cfg,
            ik_servo_cal_cfg,
            args,
        )
        clear_step = build_pose_step("CLEAR", "CLEAR_TEST", clear_pose, servo_cfg)
        drop_hover = build_pose_step(
            "DROP_HOVER",
            "CAKE_DROP_HOVER" if target["group"] == "CAKE" else "DONUT_DROP_HOVER",
            drop_hover_pose,
            servo_cfg,
        )
        drop_place = build_pose_step(
            "DROP_PLACE",
            "CAKE_DROP_PLACE" if target["group"] == "CAKE" else "DONUT_DROP_PLACE",
            drop_place_pose,
            servo_cfg,
        )
        open_gripper = build_gripper_variant_step("OPEN_GRIPPER", drop_place, open_deg, servo_cfg)
        home_move = build_pose_step("HOME_SAFE_MOVE", "HOME_SAFE", home_pose, servo_cfg)
    except ValueError as exc:
        print(f"[IK][ERROR] {exc}", file=sys.stderr)
        return 2

    steps: list[dict[str, Any]] = []
    if args.home_first:
        steps.append(build_home_step())
    steps.extend(
        [
            safe_hover,
            pre_pick,
            close_gripper,
            lift,
            clear_step,
            drop_hover,
            drop_place,
            open_gripper,
            home_move,
        ]
    )

    print_sequence(steps, args)

    if not args.send or args.dry_run:
        return 0

    print("\n[SAFETY] Live semi-auto single-object motion requested.")
    print("[SAFETY] Home, hover, pre-pick, and drop path remain human-confirmed only.")
    if args.confirm_target and not require_exact("TARGET", "Type TARGET to accept the selected object."):
        print("Target confirmation cancelled.")
        return 0
    if not require_exact("START", "Type START to send the full sequence."):
        print("Sequence cancelled.")
        return 0

    try:
        ser = open_serial(args.port, args.baud, args.timeout)
    except Exception as exc:  # pragma: no cover
        print(f"[SERIAL][ERROR] Failed to open serial port {args.port}: {exc}", file=sys.stderr)
        return 2

    try:
        send_sequence(ser, steps)
    except KeyboardInterrupt:
        print("\n[SAFETY] Interrupted. Sending STOP.")
        try:
            lines = send_command(ser, "STOP")
            for line in lines:
                print(f"  {line}")
        except Exception:
            pass
        return 2
    except RuntimeError as exc:
        print(f"[SERIAL][ERROR] {exc}", file=sys.stderr)
        try:
            lines = send_command(ser, "STOP")
            for line in lines:
                print(f"  {line}")
        except Exception:
            pass
        return 2
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
