#!/usr/bin/env python3
"""
test_esp32_serial_home.py

Phase 6 serial testing helper for ESP32 HOME_SAFE and diagnostics only.

This tool never uses camera, IK, YOLO, ROS2, or autonomous loops. It only
supports guarded diagnostics plus an explicitly confirmed HOME command.
"""

import argparse
import glob
import sys
import time
from typing import Iterable


DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6 ESP32 serial HOME_SAFE tester. No IK or autonomous motion.",
    )
    parser.add_argument("--port", help="Serial port path, required unless --list-ports")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--ping", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--limits", action="store_true")
    parser.add_argument("--home", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--dry-run", action="store_true", help="Print intended command without opening the port")
    return parser.parse_args()


def selected_actions(args: argparse.Namespace) -> list[str]:
    actions = []
    if args.ping:
        actions.append("PING")
    if args.status:
        actions.append("STATUS")
    if args.limits:
        actions.append("LIMITS")
    if args.home:
        actions.append("HOME")
    return actions


def load_serial_modules():
    try:
        import serial  # type: ignore
        from serial.tools import list_ports  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required for serial access. Install it in /home/andra/envs/robot_yolo_env."
        ) from exc
    return serial, list_ports


def list_ports_action() -> int:
    try:
        _, list_ports = load_serial_modules()
        ports = [(port.device, port.description or "unknown", port.hwid or "unknown") for port in list_ports.comports()]
    except RuntimeError:
        ports = [(device, "pyserial unavailable", "fallback-scan") for device in sorted(_fallback_port_scan())]

    if not ports:
        print("No serial ports found.")
        return 0

    for device, desc, hwid in ports:
        print(f"{device} | {desc} | {hwid}")
    return 0


def _fallback_port_scan() -> list[str]:
    patterns = ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyS*", "/dev/cu.*")
    found: set[str] = set()
    for pattern in patterns:
        found.update(glob.glob(pattern))
    return sorted(found)


def require_home_confirmation() -> bool:
    print("Type HOME to send HOME command.")
    try:
        typed = input("> ").strip()
    except EOFError:
        return False
    return typed == "HOME"


def print_dry_run(port: str, baud: int, actions: Iterable[str]) -> None:
    for command in actions:
        print(f"DRY RUN: port={port} baud={baud} command={command}")


def open_serial(port: str, baud: int, timeout: float):
    serial, _ = load_serial_modules()
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
        if command == "PING" and line == "PONG":
            break
        if command == "HOME" and line == "DONE HOME":
            break
        if line.startswith("ERR ") or line.startswith("DONE ") or line.startswith("STATUS ") or line.startswith("LIMITS ") or line.startswith("HELP "):
            if command != "HOME":
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
    actions = selected_actions(args)

    if args.list_ports:
        if actions:
            print("[ERROR] --list-ports cannot be combined with command actions", file=sys.stderr)
            return 2
        return list_ports_action()

    if not actions:
        print("[ERROR] Select at least one action: --ping, --status, --limits, or --home", file=sys.stderr)
        return 2

    if args.port is None:
        print("[ERROR] --port is required unless --list-ports is used", file=sys.stderr)
        return 2

    if args.home and not args.dry_run:
        if not require_home_confirmation():
            print("HOME command cancelled.")
            return 0

    if args.dry_run:
        print_dry_run(args.port, args.baud, actions)
        return 0

    try:
        ser = open_serial(args.port, args.baud, args.timeout)
    except Exception as exc:  # pragma: no cover - hardware-dependent path
        print(f"[ERROR] Failed to open serial port {args.port}: {exc}", file=sys.stderr)
        return 2

    try:
        for command in actions:
            lines = send_command(ser, command)
            print_response(command, lines)
            if command == "PING" and "PONG" not in lines:
                print("[ERROR] Expected PONG response", file=sys.stderr)
                return 2
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
