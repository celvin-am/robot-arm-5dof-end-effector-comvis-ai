#!/usr/bin/env python3
"""
test_corrected_ik_to_move_safe.py

Phase 11 corrected board-target to IK MOVE_SAFE test.

This tool applies a saved robot alignment correction before converting the
board target to robot coordinates, computing IK, and optionally sending a
guarded MOVE_SAFE command.
"""

import argparse
import sys
from typing import Any

import numpy as np

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


DEFAULT_CORRECTION_CONFIG = "ros2_ws/src/robot_arm_5dof/config/robot_alignment_correction.yaml"
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 8.0
DEFAULT_BOARD_X = 7.0
DEFAULT_BOARD_Y = 9.0
DEFAULT_Z = 0.12
DEFAULT_SOLUTION = "elbow_up"
DEFAULT_TCP_OFFSET_MODE = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 11 corrected board target to IK MOVE_SAFE tester.",
    )
    parser.add_argument("--port")
    parser.add_argument("--board-x", type=float, default=DEFAULT_BOARD_X)
    parser.add_argument("--board-y", type=float, default=DEFAULT_BOARD_Y)
    parser.add_argument("--z", type=float, default=DEFAULT_Z)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--home-first", action="store_true")
    parser.add_argument("--return-home", action="store_true")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--correction-config", default=DEFAULT_CORRECTION_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--serial-config", default=DEFAULT_SERIAL_CONFIG)
    return parser.parse_args()


def apply_correction(board_x_cm: float, board_y_cm: float, correction_cfg: dict[str, Any]) -> tuple[float, float, str]:
    root = correction_cfg.get("robot_alignment_correction", {})
    method = str(root.get("method", "identity"))
    correction = root.get("correction", {})

    if method in {"identity", "", "null"}:
        return board_x_cm, board_y_cm, "identity"

    if method == "translation":
        dx = float(correction.get("translation_dx_cm", 0.0))
        dy = float(correction.get("translation_dy_cm", 0.0))
        return board_x_cm + dx, board_y_cm + dy, "translation"

    if method == "affine":
        affine = correction.get("affine_2x3")
        if affine is None:
            raise ValueError("Correction method is affine but affine_2x3 is missing")
        mat = np.asarray(affine, dtype=np.float64)
        if mat.shape != (2, 3):
            raise ValueError(f"affine_2x3 must be shape (2, 3), got {mat.shape}")
        vec = np.array([board_x_cm, board_y_cm, 1.0], dtype=np.float64)
        out = mat @ vec
        return float(out[0]), float(out[1]), "affine"

    raise ValueError(f"Unsupported correction method: {method}")


def print_report(
    board_x_cm: float,
    board_y_cm: float,
    corrected_x_cm: float,
    corrected_y_cm: float,
    robot_x_m: float,
    robot_y_m: float,
    candidate: dict[str, Any],
    servos: dict[str, float],
    move_angles: list[int],
    failures: list[str],
    method: str,
    send_mode: bool,
) -> None:
    print("============================================================")
    print("PHASE 11 CORRECTED IK TO MOVE_SAFE TEST")
    print("============================================================")
    print(f"original board target: x={board_x_cm:.2f} cm y={board_y_cm:.2f} cm")
    print(f"corrected board target: x={corrected_x_cm:.2f} cm y={corrected_y_cm:.2f} cm")
    print(f"correction method: {method}")
    print(f"robot target: x={robot_x_m:.4f} m y={robot_y_m:.4f} m")
    print(f"selected solution: {candidate['name']}")
    print(f"mode: {'HARDWARE SEND' if send_mode else 'DRY RUN'}")
    print("computed servo angles CH1-CH5:")
    for ch in ("ch1", "ch2", "ch3", "ch4", "ch5"):
        print(f"  {ch}: {servos[ch]:.2f}")
    print("final MOVE_SAFE angles CH1-CH6:")
    print("  " + " ".join(f"CH{index + 1}={value}" for index, value in enumerate(move_angles)))
    print(f"servo limit: {'PASS' if not failures else 'FAIL'}")
    if failures:
        for failure in failures:
            print(f"  {failure}")
    print("intended command:")
    print(f"  {build_move_command(move_angles)}")


def main() -> int:
    args = parse_args()

    if args.dry_run and args.send:
        print("[ERROR] Use either --dry-run or --send, not both", file=sys.stderr)
        return 2
    if args.send and args.port is None:
        print("[ERROR] --port is required when --send is used", file=sys.stderr)
        return 2

    transform_cfg = load_required(args.transform_config, "transform config")
    kin_cfg = load_required(args.kinematics_config, "kinematics config")
    servo_cfg = load_required(args.servo_config, "servo config")
    pose_cfg = load_required(args.pose_config, "pose config")
    serial_cfg = load_required(args.serial_config, "serial config")
    correction_cfg = load_required(args.correction_config, "correction config")
    _ = serial_cfg

    supported_modes = kin_cfg.get("ik", {}).get("tcp_offset_modes_supported", ["none", "planar", "vertical_down", "mixed"])
    if DEFAULT_TCP_OFFSET_MODE not in supported_modes:
        print(f"[ERROR] Default tcp offset mode {DEFAULT_TCP_OFFSET_MODE!r} not supported", file=sys.stderr)
        return 2

    try:
        corrected_x_cm, corrected_y_cm, method = apply_correction(args.board_x, args.board_y, correction_cfg)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    robot_x_m, robot_y_m = board_to_robot(corrected_x_cm, corrected_y_cm, transform_cfg)
    ik_result = solve_planar_ik(robot_x_m, robot_y_m, args.z, kin_cfg, DEFAULT_TCP_OFFSET_MODE, 0.0, 1.0)
    if "unreachable_reason" in ik_result:
        print(f"[ERROR] IK unreachable: {ik_result['unreachable_reason']}", file=sys.stderr)
        return 2

    candidate = select_candidate(ik_result, DEFAULT_SOLUTION)
    servos = candidate_servos(candidate, kin_cfg)
    limit_ok, limit_lines = validate_servos(servos, servo_cfg)

    home = pose_cfg.get("poses", {}).get("HOME_SAFE", {})
    home_gripper = home.get("ch6")
    if home_gripper is None:
        print("[ERROR] HOME_SAFE ch6 is required in pose_config.yaml", file=sys.stderr)
        return 2

    move_angles = rounded_move_safe_angles(servos, int(home_gripper))
    rounded_failures = validate_move_safe_angles(move_angles, servo_cfg)
    failures = []
    if not limit_ok:
        failures.extend(line[4:] if line.startswith("BAD ") else line for line in limit_lines if line.startswith("BAD "))
    failures.extend(rounded_failures)

    print_report(
        args.board_x,
        args.board_y,
        corrected_x_cm,
        corrected_y_cm,
        robot_x_m,
        robot_y_m,
        candidate,
        servos,
        move_angles,
        failures,
        method,
        args.send and not args.dry_run,
    )

    if failures:
        print("[ERROR] MOVE_SAFE send blocked because servo limits did not pass.", file=sys.stderr)
        return 2

    move_command = build_move_command(move_angles)
    if args.dry_run or not args.send:
        return 0

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

        if not require_confirmation("MOVE", "Type MOVE to send corrected IK MOVE_SAFE."):
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
