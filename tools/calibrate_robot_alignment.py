#!/usr/bin/env python3
"""
calibrate_robot_alignment.py

Phase 11 robot alignment correction calibration using manual TCP click.

This tool computes IK for safe hover targets, optionally moves the robot after
explicit confirmation, lets the user click the observed TCP/gripper position in
the camera image, and saves a provisional translation or affine correction.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

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
    clamp_move_safe_angles,
    require_line,
    rounded_move_safe_angles,
    select_candidate,
    validate_move_safe_angles,
)
from test_yolo_board_mapping import DEFAULT_BOARD_CONFIG, DEFAULT_HEIGHT, DEFAULT_WIDTH, load_board_and_homography, pixel_to_board


DEFAULT_OUTPUT = "ros2_ws/src/robot_arm_5dof/config/robot_alignment_correction.yaml"
DEFAULT_BAUD = 115200
DEFAULT_CAM = 2
DEFAULT_TIMEOUT = 10.0
DEFAULT_Z = 0.12
DEFAULT_SOLUTION = "elbow_up"
DEFAULT_TCP_OFFSET_MODE = "none"
WINDOW_NAME = "Phase 11 Robot Alignment Calibration"

TARGET_SETS = {
    "basic": [
        ("left_mid", 7.0, 9.0),
        ("center", 13.5, 9.0),
        ("right_mid", 20.0, 9.0),
    ],
    "bowls": [
        ("cake_bowl", 7.0, 6.0),
        ("donut_bowl", 20.0, 6.0),
    ],
    "all": [
        ("left_mid", 7.0, 9.0),
        ("center", 13.5, 9.0),
        ("right_mid", 20.0, 9.0),
        ("cake_bowl", 7.0, 6.0),
        ("donut_bowl", 20.0, 6.0),
    ],
}

SINGLE_TARGETS = {
    "left_mid": ("left_mid", 7.0, 9.0),
    "center": ("center", 13.5, 9.0),
    "right_mid": ("right_mid", 20.0, 9.0),
    "cake_bowl": ("cake_bowl", 7.0, 6.0),
    "donut_bowl": ("donut_bowl", 20.0, 6.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 11 robot alignment calibration by manual TCP click.",
    )
    parser.add_argument("--port")
    parser.add_argument("--cam", type=int, default=DEFAULT_CAM)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--z", type=float, default=DEFAULT_Z)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--yes-i-understand-hardware-risk", action="store_true")
    parser.add_argument("--target-set", choices=["basic", "bowls", "all"], default="basic")
    parser.add_argument("--single-target", choices=list(SINGLE_TARGETS.keys()))
    parser.add_argument("--skip-on-no-click", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--board-config", default=DEFAULT_BOARD_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--serial-config", default=DEFAULT_SERIAL_CONFIG)
    return parser.parse_args()


def compute_hover_step(
    target_name: str,
    board_x_cm: float,
    board_y_cm: float,
    z_m: float,
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    pose_cfg: dict[str, Any],
    transform_cfg: dict[str, Any],
) -> dict[str, Any]:
    robot_x_m, robot_y_m = board_to_robot(board_x_cm, board_y_cm, transform_cfg)
    ik_result = solve_planar_ik(robot_x_m, robot_y_m, z_m, kin_cfg, DEFAULT_TCP_OFFSET_MODE, 0.0, 1.0)
    if "unreachable_reason" in ik_result:
        raise ValueError(f"{target_name}: {ik_result['unreachable_reason']}")

    candidate = select_candidate(ik_result, DEFAULT_SOLUTION)
    servos = candidate_servos(candidate, kin_cfg)
    limit_ok, limit_lines = validate_servos(servos, servo_cfg)
    home = pose_cfg.get("poses", {}).get("HOME_SAFE", {})
    home_gripper = home.get("ch6")
    if home_gripper is None:
        raise ValueError("HOME_SAFE ch6 is required in pose_config.yaml")
    move_angles = rounded_move_safe_angles(servos, int(home_gripper))
    move_angles, clamp_notes = clamp_move_safe_angles(move_angles, servo_cfg)
    rounded_failures = validate_move_safe_angles(move_angles, servo_cfg)
    failures = []
    if not limit_ok:
        failures.extend(line[4:] if line.startswith("BAD ") else line for line in limit_lines if line.startswith("BAD "))
    failures.extend(rounded_failures)
    if failures:
        raise ValueError(f"{target_name}: servo limit failure: {'; '.join(failures)}")

    return {
        "target_name": target_name,
        "board_x_cm": board_x_cm,
        "board_y_cm": board_y_cm,
        "robot_x_m": robot_x_m,
        "robot_y_m": robot_y_m,
        "z_m": z_m,
        "solution": candidate["name"],
        "move_angles": move_angles,
        "clamp_notes": clamp_notes,
        "command": build_move_command(move_angles),
    }


class ClickCollector:
    def __init__(self) -> None:
        self.point: tuple[int, int] | None = None

    def callback(self, event, x, y, flags, userdata) -> None:
        del flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)


def capture_manual_click(
    cap,
    H: np.ndarray,
    target_name: str,
    show: bool,
    timeout: float,
    board_width_cm: float,
    board_height_cm: float,
) -> tuple[tuple[int, int], tuple[float, float]]:
    collector = ClickCollector()
    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, collector.callback)

    deadline = time.time() + timeout
    last_frame = None
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        last_frame = frame.copy()
        cv2.putText(frame, f"Click actual TCP center for {target_name}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(
            frame,
            "Only click the actual TCP/gripper center. If gripper is not visible, reject/skip this sample.",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            2,
        )
        if collector.point is not None:
            cv2.circle(frame, collector.point, 6, (0, 255, 255), -1)
            cv2.circle(frame, collector.point, 6, (255, 255, 255), 1)
        if show:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
        if collector.point is not None:
            board_x_cm, board_y_cm = pixel_to_board(H, float(collector.point[0]), float(collector.point[1]))
            if not (0.0 <= board_x_cm <= board_width_cm and 0.0 <= board_y_cm <= board_height_cm):
                print(
                    f"[WARN] Rejected click for {target_name}: board coordinate "
                    f"({board_x_cm:.2f}, {board_y_cm:.2f}) cm is outside [0,{board_width_cm:g}] x [0,{board_height_cm:g}]"
                )
                collector.point = None
                continue
            return collector.point, (board_x_cm, board_y_cm)

    raise TimeoutError(f"No manual click received for {target_name} before timeout")


def confirm_sample_acceptance(target_name: str, board_point: tuple[float, float], error_x_cm: float, error_y_cm: float) -> bool:
    print(
        f"[REVIEW] {target_name}: actual=({board_point[0]:.2f}, {board_point[1]:.2f}) cm "
        f"error=({error_x_cm:.2f}, {error_y_cm:.2f}) cm"
    )
    print("Type ACCEPT to keep this sample, or REJECT to retry/skip.")
    try:
        typed = input("> ").strip().upper()
    except EOFError:
        return False
    return typed == "ACCEPT"


def fit_translation(samples: list[dict[str, Any]]) -> dict[str, Any]:
    dx = float(np.mean([sample["error_x_cm"] for sample in samples]))
    dy = float(np.mean([sample["error_y_cm"] for sample in samples]))
    return {
        "method": "translation",
        "translation_dx_cm": dx,
        "translation_dy_cm": dy,
        "affine_2x3": None,
    }


def fit_affine(samples: list[dict[str, Any]]) -> dict[str, Any]:
    src = np.array([[sample["target_board_x_cm"], sample["target_board_y_cm"]] for sample in samples], dtype=np.float64)
    dst = np.array(
        [
            [
                sample["target_board_x_cm"] + sample["error_x_cm"],
                sample["target_board_y_cm"] + sample["error_y_cm"],
            ]
            for sample in samples
        ],
        dtype=np.float64,
    )
    A = []
    b = []
    for (x, y), (xp, yp) in zip(src, dst):
        A.append([x, y, 1.0, 0.0, 0.0, 0.0])
        A.append([0.0, 0.0, 0.0, x, y, 1.0])
        b.append(xp)
        b.append(yp)
    params, _, _, _ = np.linalg.lstsq(np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64), rcond=None)
    affine = np.array([[params[0], params[1], params[2]], [params[3], params[4], params[5]]], dtype=np.float64)
    mean_dx = float(np.mean([sample["error_x_cm"] for sample in samples]))
    mean_dy = float(np.mean([sample["error_y_cm"] for sample in samples]))
    return {
        "method": "affine",
        "translation_dx_cm": mean_dx,
        "translation_dy_cm": mean_dy,
        "affine_2x3": affine.tolist(),
    }


def save_yaml(path: str, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def confirm_overwrite(path: str) -> bool:
    out = Path(path)
    if not out.exists():
        return True
    print(f"Existing correction file found: {path}")
    print("Type OVERWRITE to replace it, or anything else to cancel save.")
    try:
        typed = input("> ").strip()
    except EOFError:
        return False
    return typed == "OVERWRITE"


def main() -> int:
    args = parse_args()

    if args.dry_run and args.send:
        print("[SAFETY][ERROR] Use either --dry-run or --send, not both", file=sys.stderr)
        return 2
    if args.send and args.port is None:
        print("[SERIAL][ERROR] --port is required when --send is used", file=sys.stderr)
        return 2
    if args.send and not args.yes_i_understand_hardware_risk:
        print(
            "[SAFETY][ERROR] Live calibration motion requires --yes-i-understand-hardware-risk.",
            file=sys.stderr,
        )
        return 2

    _, H, H_path, board_width_cm, board_height_cm = load_board_and_homography(args.board_config, None)
    print(f"[BOARD] Using homography: {H_path}")

    transform_cfg = load_required(args.transform_config, "transform config")
    kin_cfg = load_required(args.kinematics_config, "kinematics config")
    servo_cfg = load_required(args.servo_config, "servo config")
    pose_cfg = load_required(args.pose_config, "pose config")
    serial_cfg = load_required(args.serial_config, "serial config")
    _ = serial_cfg

    if args.single_target:
        targets = [SINGLE_TARGETS[args.single_target]]
    else:
        targets = TARGET_SETS[args.target_set]
    try:
        steps = [
            compute_hover_step(name, board_x, board_y, args.z, kin_cfg, servo_cfg, pose_cfg, transform_cfg)
            for name, board_x, board_y in targets
        ]
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print("Calibration targets:")
    for step in steps:
        print(
            f"  {step['target_name']}: board=({step['board_x_cm']:.2f},{step['board_y_cm']:.2f}) "
            f"robot=({step['robot_x_m']:.4f},{step['robot_y_m']:.4f}) z={step['z_m']:.2f} {step['command']}"
        )
        for note in step.get("clamp_notes", []):
            print(f"    [SAFETY] {note}")
    print("Only click the actual TCP/gripper center. If gripper is not visible, reject/skip this sample.")

    if args.dry_run or not args.send:
        print("\nMode: DRY RUN")
        return 0

    print("\n[SAFETY] Live alignment calibration requested.")
    print("[SAFETY] Calibration motion proceeds one target at a time only.")

    try:
        ser = open_serial(args.port, args.baud, args.timeout)
    except Exception as exc:  # pragma: no cover - hardware-dependent path
        print(f"[SERIAL][ERROR] Failed to open serial port {args.port}: {exc}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        ser.close()
        print(f"[ERROR] Cannot open camera index {args.cam}", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    samples = []
    try:
        for step in steps:
            print(f"[SAFETY] Command preview for {step['target_name']}: {step['command']}")
            if not require_confirmation("MOVE", f"Type MOVE to move to calibration target {step['target_name']}."):
                print("Calibration cancelled.")
                return 0
            lines = send_command(ser, step["command"])
            print(f"COMMAND {step['command']}")
            for line in lines:
                print(f"  {line}")
            require_line(lines, "ACK MOVE_SAFE", step["target_name"])
            require_line(lines, "DONE MOVE_SAFE", step["target_name"])

            while True:
                try:
                    pixel_point, board_point = capture_manual_click(
                        cap,
                        H,
                        step["target_name"],
                        args.show,
                        args.timeout,
                        board_width_cm,
                        board_height_cm,
                    )
                except TimeoutError:
                    if args.skip_on_no_click:
                        print(f"[INFO] No valid click for {step['target_name']}; skipping sample.")
                        break
                    raise

                error_x_cm = float(step["board_x_cm"] - board_point[0])
                error_y_cm = float(step["board_y_cm"] - board_point[1])
                accepted = confirm_sample_acceptance(step["target_name"], board_point, error_x_cm, error_y_cm)
                if accepted:
                    sample = {
                        "target_name": step["target_name"],
                        "target_board_x_cm": float(step["board_x_cm"]),
                        "target_board_y_cm": float(step["board_y_cm"]),
                        "actual_tcp_board_x_cm": float(board_point[0]),
                        "actual_tcp_board_y_cm": float(board_point[1]),
                        "error_x_cm": error_x_cm,
                        "error_y_cm": error_y_cm,
                        "clicked_pixel_u": int(pixel_point[0]),
                        "clicked_pixel_v": int(pixel_point[1]),
                    }
                    samples.append(sample)
                    print(
                        f"[SAMPLE] {step['target_name']} actual=({board_point[0]:.2f},{board_point[1]:.2f}) "
                        f"error=({error_x_cm:.2f},{error_y_cm:.2f}) cm"
                    )
                    break

                if args.skip_on_no_click:
                    print(f"[INFO] Sample rejected for {step['target_name']}; skipping target.")
                    break
                print(f"[INFO] Sample rejected for {step['target_name']}; retrying click.")
    except (KeyboardInterrupt, TimeoutError) as exc:
        print(f"\n[INFO] Calibration stopped: {exc}")
        return 0
    except RuntimeError as exc:
        print(f"[SERIAL][ERROR] {exc}", file=sys.stderr)
        return 2
    finally:
        cap.release()
        ser.close()
        if args.show:
            cv2.destroyAllWindows()

    if not samples:
        print("[WARN] No valid samples collected; correction file was not written.")
        return 0

    if len(samples) >= 3:
        correction = fit_affine(samples)
    else:
        correction = fit_translation(samples)

    mean_error_before_cm = float(
        np.mean([np.hypot(sample["error_x_cm"], sample["error_y_cm"]) for sample in samples])
    )
    payload = {
        "robot_alignment_correction": {
            "status": "provisional_manual_click",
            "method": correction["method"],
            "created_at": datetime.utcnow().isoformat() + "Z",
            "samples": samples,
            "mean_error_before_cm": mean_error_before_cm,
            "correction": {
                "translation_dx_cm": correction["translation_dx_cm"],
                "translation_dy_cm": correction["translation_dy_cm"],
                "affine_2x3": correction["affine_2x3"],
            },
            "note": "correction is valid only while camera, board, and robot base do not move",
        }
    }
    if not confirm_overwrite(args.output):
        print("[INFO] Save cancelled; correction file was not written.")
        return 0
    save_yaml(args.output, payload)
    print(f"[INFO] Saved correction to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
