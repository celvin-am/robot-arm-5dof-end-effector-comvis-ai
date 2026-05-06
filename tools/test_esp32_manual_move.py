#!/usr/bin/env python3
"""
test_esp32_manual_move.py

Phase 7 guarded manual servo movement tester for ESP32 serial firmware.

This tool supports STATUS, HOME, and MOVE_SAFE only. It never uses camera,
YOLO, IK target motion, ROS2, or autonomous loops.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 5.0
DEFAULT_SERVO_CONFIG = "ros2_ws/src/robot_arm_5dof/config/servo_config.yaml"
DEFAULT_SERIAL_CONFIG = "ros2_ws/src/robot_arm_5dof/config/serial_config.yaml"
FALLBACK_MOVE_SAFE_LIMITS = ((40, 140), (40, 140), (40, 140), (40, 140), (40, 140), (10, 60))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7 guarded manual ESP32 servo tester. No IK or autonomous motion.",
    )
    parser.add_argument("--port", required=True, help="Serial port path")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--home", action="store_true")
    parser.add_argument(
        "--move-safe",
        nargs=6,
        metavar=("CH1", "CH2", "CH3", "CH4", "CH5", "CH6"),
        type=int,
        help="Send guarded MOVE_SAFE ch1 ch2 ch3 ch4 ch5 ch6",
    )
    parser.add_argument("--yes-i-understand-hardware-risk", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print intended command without opening the port")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--serial-config", default=DEFAULT_SERIAL_CONFIG)
    return parser.parse_args()


def load_required_yaml(path: str, label: str) -> dict[str, Any]:
    if not Path(path).exists():
        raise RuntimeError(f"Missing {label}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} must contain a YAML mapping: {path}")
    return data


def load_move_safe_limits(servo_cfg: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    servos = servo_cfg.get("servos", {})
    limits = []
    for index in range(1, 7):
        channel = f"ch{index}"
        block = servos.get(channel, {})
        lo = block.get("min_angle_deg")
        hi = block.get("max_angle_deg")
        if lo is None or hi is None:
            lo, hi = FALLBACK_MOVE_SAFE_LIMITS[index - 1]
        limits.append((int(lo), int(hi)))
    return tuple(limits)


def load_serial_module():
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required for serial access. Install it in /home/andra/envs/robot_yolo_env."
        ) from exc
    return serial


def require_confirmation(label: str, prompt: str) -> bool:
    print(prompt)
    try:
        typed = input("> ").strip()
    except EOFError:
        return False
    return typed == label


def build_actions(args: argparse.Namespace, move_safe_limits: tuple[tuple[int, int], ...]) -> list[str]:
    actions: list[str] = []
    if args.status:
        actions.append("STATUS")
    if args.home:
        actions.append("HOME")
    if args.move_safe is not None:
        clamped = clamp_move_safe_values(args.move_safe, move_safe_limits)
        actions.append("MOVE_SAFE " + " ".join(str(value) for value in clamped))
    if not actions:
        raise ValueError("Select at least one action: --status, --home, or --move-safe")
    return actions


def clamp_move_safe_values(values: Iterable[int], move_safe_limits: tuple[tuple[int, int], ...]) -> list[int]:
    clamped = []
    for index, (value, limits) in enumerate(zip(values, move_safe_limits), start=1):
        lo, hi = limits
        bounded = max(lo, min(hi, int(value)))
        if bounded != value:
            print(f"[SAFETY] CH{index} clamped from {value} to {bounded} within [{lo}, {hi}]")
        clamped.append(bounded)
    return clamped


def print_dry_run(port: str, baud: int, actions: Iterable[str]) -> None:
    for command in actions:
        print(f"DRY RUN: port={port} baud={baud} command={command}")


def open_serial(port: str, baud: int, timeout: float):
    serial = load_serial_module()
    return serial.Serial(port=port, baudrate=baud, timeout=timeout, write_timeout=timeout)


def send_command(ser, command: str) -> list[str]:
    ser.reset_input_buffer()
    ser.write((command + "\n").encode("ascii"))
    ser.flush()

    lines: list[str] = []
    deadline = time.time() + float(ser.timeout or DEFAULT_TIMEOUT)
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        lines.append(line)

        if command == "STATUS" and line.startswith("STATUS "):
            break
        if command == "HOME" and line == "DONE HOME":
            break
        if command.startswith("MOVE_SAFE") and line == "DONE MOVE_SAFE":
            break
        if line.startswith("ERR "):
            break
    return lines


def print_response(command: str, lines: list[str]) -> None:
    print(f"COMMAND {command}")
    if not lines:
        print("  no response")
        return
    for line in lines:
        print(f"  {line}")


def main() -> int:
    args = parse_args()

    try:
        servo_cfg = load_required_yaml(args.servo_config, "servo config")
        _ = load_required_yaml(args.serial_config, "serial config")
        move_safe_limits = load_move_safe_limits(servo_cfg)
        actions = build_actions(args, move_safe_limits)
    except (RuntimeError, ValueError) as exc:
        print(f"[SAFETY][ERROR] {exc}", file=sys.stderr)
        return 2

    live_motion_requested = not args.dry_run and (args.home or args.move_safe is not None)
    if live_motion_requested and not args.yes_i_understand_hardware_risk:
        print(
            "[SAFETY][ERROR] Live HOME or MOVE_SAFE commands require --yes-i-understand-hardware-risk.",
            file=sys.stderr,
        )
        return 2

    if args.home:
        print("[SAFETY] HOME_SAFE is validated for idle/manual testing only.")
        print("[SAFETY] Command preview: HOME")
    if args.move_safe is not None:
        move_command = next((action for action in actions if action.startswith("MOVE_SAFE ")), None)
        if move_command is not None:
            print("[SAFETY] Final MOVE_SAFE command preview:")
            print(f"  {move_command}")

    if args.home and not args.dry_run:
        print("[SAFETY] Live hardware HOME requested.")
        if not require_confirmation("HOME", "Type HOME to send HOME."):
            print("HOME command cancelled.")
            return 0

    if args.move_safe is not None and not args.dry_run:
        print("[SAFETY] Live hardware MOVE_SAFE requested.")
        if not require_confirmation("MOVE", "Type MOVE to send MOVE_SAFE."):
            print("MOVE_SAFE command cancelled.")
            return 0

    if args.dry_run:
        print_dry_run(args.port, args.baud, actions)
        return 0

    try:
        ser = open_serial(args.port, args.baud, args.timeout)
    except Exception as exc:  # pragma: no cover - hardware-dependent path
        print(f"[SERIAL][ERROR] Failed to open serial port {args.port}: {exc}", file=sys.stderr)
        return 2

    try:
        for command in actions:
            lines = send_command(ser, command)
            print_response(command, lines)
            if lines and lines[-1].startswith("ERR "):
                return 2
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
