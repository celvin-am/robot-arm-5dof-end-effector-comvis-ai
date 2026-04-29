#!/usr/bin/env python3
"""
test_ik_to_move_safe.py

Phase 8 guarded IK-to-MOVE_SAFE hardware test.

This tool computes IK from a known safe target, validates servo limits, and
only sends MOVE_SAFE to the ESP32 after explicit confirmation. It never uses
camera, YOLO, ROS2, or autonomous loops.
"""

import argparse
import sys
from typing import Any

from test_esp32_manual_move import open_serial, print_response, require_confirmation, send_command
from test_ik_dry_run import (
    DEFAULT_KINEMATICS_CONFIG,
    DEFAULT_POSE_CONFIG,
    DEFAULT_SERVO_CONFIG,
    DEFAULT_TRANSFORM_CONFIG,
    board_to_robot,
    candidate_servos,
    channel_limits,
    load_required,
    solve_planar_ik,
    validate_servos,
)


DEFAULT_SERIAL_CONFIG = "ros2_ws/src/robot_arm_5dof/config/serial_config.yaml"
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 8 guarded IK-to-MOVE_SAFE tester. No YOLO or autonomous motion.",
    )
    parser.add_argument("--port", help="Serial port path, required unless --dry-run")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--board-x", type=float, default=7.0)
    parser.add_argument("--board-y", type=float, default=9.0)
    parser.add_argument("--z", type=float, default=0.12)
    parser.add_argument("--robot-x", type=float)
    parser.add_argument("--robot-y", type=float)
    parser.add_argument("--from-board", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tcp-offset-mode", default="none")
    parser.add_argument("--solution", choices=["elbow_up", "elbow_down", "best"], default="elbow_up")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--home-first", action="store_true")
    parser.add_argument("--return-home", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
    parser.add_argument("--serial-config", default=DEFAULT_SERIAL_CONFIG)
    return parser.parse_args()


def resolve_target(args: argparse.Namespace, transform_cfg: dict[str, Any]) -> tuple[float | None, float | None, float, float]:
    if args.robot_x is not None or args.robot_y is not None:
        if args.robot_x is None or args.robot_y is None:
            raise ValueError("--robot-x and --robot-y must both be provided for direct robot-frame targets")
        return None, None, float(args.robot_x), float(args.robot_y)

    if not args.from_board:
        raise ValueError("Board target disabled but no direct robot target was provided")

    robot_x, robot_y = board_to_robot(args.board_x, args.board_y, transform_cfg)
    return float(args.board_x), float(args.board_y), robot_x, robot_y


def select_candidate(ik_result: dict[str, Any], solution: str) -> dict[str, Any]:
    if "unreachable_reason" in ik_result:
        raise ValueError(ik_result["unreachable_reason"])

    candidates = {candidate["name"]: candidate for candidate in ik_result["candidates"]}
    if solution in candidates:
        return candidates[solution]

    if solution == "best":
        return ik_result["candidates"][0]

    raise ValueError(f"Requested IK solution {solution!r} was not available")


def rounded_move_safe_angles(servos: dict[str, float], gripper_home: int) -> list[int]:
    return [
        int(round(servos["ch1"])),
        int(round(servos["ch2"])),
        int(round(servos["ch3"])),
        int(round(servos["ch4"])),
        int(round(servos["ch5"])),
        int(gripper_home),
    ]


def validate_move_safe_angles(move_angles: list[int], servo_cfg: dict[str, Any]) -> list[str]:
    failures = []
    for index, channel in enumerate(("ch1", "ch2", "ch3", "ch4", "ch5", "ch6")):
        lo, hi = channel_limits(servo_cfg, channel)
        value = move_angles[index]
        if not lo <= value <= hi:
            failures.append(f"{channel}={value} not in [{lo:.1f}, {hi:.1f}]")
    return failures


def build_move_command(move_angles: list[int]) -> str:
    return "MOVE_SAFE " + " ".join(str(value) for value in move_angles)


def print_report(
    args: argparse.Namespace,
    board_target: tuple[float | None, float | None],
    robot_target: tuple[float, float],
    candidate: dict[str, Any],
    servos: dict[str, float],
    move_angles: list[int],
    servo_failures: list[str],
) -> None:
    board_x, board_y = board_target
    robot_x, robot_y = robot_target

    print("============================================================")
    print("PHASE 8 IK TO MOVE_SAFE TEST")
    print("============================================================")
    if board_x is not None and board_y is not None:
        print(f"board target: x={board_x:.2f} cm, y={board_y:.2f} cm")
    else:
        print("board target: direct robot-frame input")
    print(f"robot target: x={robot_x:.4f} m, y={robot_y:.4f} m")
    print(f"z: {args.z:.4f} m")
    print(f"selected solution: {candidate['name']}")
    print(f"tcp_offset_mode: {args.tcp_offset_mode}")
    print(f"mode: {'HARDWARE SEND' if args.send and not args.dry_run else 'DRY RUN'}")

    print("\nComputed servo angles CH1-CH5:")
    for channel in ("ch1", "ch2", "ch3", "ch4", "ch5"):
        print(f"  {channel}: {servos[channel]:.2f}")

    print("\nFinal MOVE_SAFE angles CH1-CH6:")
    print("  " + " ".join(f"CH{index + 1}={value}" for index, value in enumerate(move_angles)))

    if servo_failures:
        print("\nServo limit: FAIL")
        for failure in servo_failures:
            print(f"  {failure}")
    else:
        print("\nServo limit: PASS")

    print("\nIntended command:")
    print(f"  {build_move_command(move_angles)}")


def require_line(lines: list[str], expected: str, label: str) -> None:
    if expected not in lines:
        raise RuntimeError(f"Expected {expected!r} in {label} response, got: {lines}")


def main() -> int:
    args = parse_args()

    if args.dry_run and args.send:
        print("[ERROR] Use either --dry-run or --send, not both", file=sys.stderr)
        return 2

    if not args.dry_run and args.port is None:
        print("[ERROR] --port is required unless --dry-run is used", file=sys.stderr)
        return 2

    kin_cfg = load_required(args.kinematics_config, "kinematics config")
    servo_cfg = load_required(args.servo_config, "servo config")
    pose_cfg = load_required(args.pose_config, "pose config")
    transform_cfg = load_required(args.transform_config, "transform config")
    serial_cfg = load_required(args.serial_config, "serial config")

    _ = serial_cfg

    try:
        board_x, board_y, robot_x, robot_y = resolve_target(args, transform_cfg)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    supported_modes = kin_cfg.get("ik", {}).get("tcp_offset_modes_supported", ["none", "planar", "vertical_down", "mixed"])
    if args.tcp_offset_mode not in supported_modes:
        print(f"[ERROR] Unsupported tcp offset mode {args.tcp_offset_mode!r}. Supported: {supported_modes}", file=sys.stderr)
        return 2

    ik_result = solve_planar_ik(robot_x, robot_y, args.z, kin_cfg, args.tcp_offset_mode, 0.0, 1.0)
    if "unreachable_reason" in ik_result:
        print("============================================================")
        print("PHASE 8 IK TO MOVE_SAFE TEST")
        print("============================================================")
        print(f"Reachable: NO")
        print(f"Reason: {ik_result['unreachable_reason']}")
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

    print_report(args, (board_x, board_y), (robot_x, robot_y), candidate, servos, move_angles, all_failures)

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
            print_response("HOME", lines)
            require_line(lines, "ACK HOME", "HOME")
            require_line(lines, "DONE HOME", "HOME")

        if not require_confirmation("MOVE", "Type MOVE to send IK MOVE_SAFE."):
            print("MOVE_SAFE command cancelled.")
            return 0
        lines = send_command(ser, move_command)
        print_response(move_command, lines)
        require_line(lines, "ACK MOVE_SAFE", "MOVE_SAFE")
        require_line(lines, "DONE MOVE_SAFE", "MOVE_SAFE")

        if args.return_home:
            if not require_confirmation("HOME", "Type HOME to return HOME_SAFE."):
                print("Return HOME command cancelled.")
                return 0
            lines = send_command(ser, "HOME")
            print_response("HOME", lines)
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
