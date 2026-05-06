#!/usr/bin/env python3
"""
teach_servo_poses.py

Phase 12 manual teach-pose helper.

This tool uses guarded serial MOVE_SAFE and HOME commands so a human can jog
servo angles in small steps and save practical working poses without relying
on the current IK model.
"""

import argparse
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from test_esp32_manual_move import open_serial, require_confirmation, send_command


DEFAULT_BAUD = 115200
DEFAULT_STEP = 2
DEFAULT_OUTPUT = "ros2_ws/src/robot_arm_5dof/config/taught_pick_place_poses.yaml"
DEFAULT_TIMEOUT = 5.0
DEFAULT_POSE_CONFIG = "ros2_ws/src/robot_arm_5dof/config/pose_config.yaml"
DEFAULT_SERIAL_CONFIG = "ros2_ws/src/robot_arm_5dof/config/serial_config.yaml"
DEFAULT_SERVO_CONFIG = "ros2_ws/src/robot_arm_5dof/config/servo_config.yaml"

CHANNELS = ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6")
POSE_ORDER = (
    "OBJECT_HOVER",
    "OBJECT_PICK",
    "OBJECT_LIFT",
    "CAKE_BOWL_HOVER",
    "CAKE_BOWL_PLACE",
    "DONUT_BOWL_HOVER",
    "DONUT_BOWL_PLACE",
    "HOME_SAFE",
)
POSE_DESCRIPTIONS = {
    "OBJECT_HOVER": "Manual taught hover pose above object before pick.",
    "OBJECT_PICK": "Manual taught object contact pose.",
    "OBJECT_LIFT": "Manual taught lift pose after grasp.",
    "CAKE_BOWL_HOVER": "Manual taught hover pose above CAKE bowl.",
    "CAKE_BOWL_PLACE": "Manual taught place pose inside CAKE bowl.",
    "DONUT_BOWL_HOVER": "Manual taught hover pose above DONUT bowl.",
    "DONUT_BOWL_PLACE": "Manual taught place pose inside DONUT bowl.",
    "HOME_SAFE": "Safe startup/shutdown pose, gripper open.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 12 manual teach-pose tool using guarded MOVE_SAFE. No IK, YOLO, or autonomous motion.",
    )
    parser.add_argument("--port", required=True, help="Serial port path")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Do not open serial; print intended commands only")
    parser.add_argument("--yes-i-understand-hardware-risk", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--serial-config", default=DEFAULT_SERIAL_CONFIG)
    return parser.parse_args()


def load_required_yaml(path: str, label: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing {label}: {path}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in {label}: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} must contain a YAML mapping: {path}")
    return data


def read_home_angles(pose_cfg: dict[str, Any]) -> dict[str, int]:
    home = pose_cfg.get("poses", {}).get("HOME_SAFE", {})
    values: dict[str, int] = {}
    for channel in CHANNELS:
        value = home.get(channel)
        if value is None:
            raise RuntimeError(f"pose_config HOME_SAFE missing {channel}")
        values[channel] = int(value)
    return values


def read_limits(servo_cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    servos = servo_cfg.get("servos", {})
    limits: dict[str, tuple[int, int]] = {}
    for channel in CHANNELS:
        block = servos.get(channel, {})
        lo = block.get("min_angle_deg")
        hi = block.get("max_angle_deg")
        if lo is None or hi is None:
            raise RuntimeError(f"servo_config missing limits for {channel}")
        limits[channel] = (int(lo), int(hi))
    return limits


def make_pose_entry(name: str, source_angles: dict[str, int] | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"description": POSE_DESCRIPTIONS[name]}
    for channel in CHANNELS:
        entry[channel] = None if source_angles is None else int(source_angles[channel])
    return entry


def build_default_output(home_angles: dict[str, int], step_deg: int) -> dict[str, Any]:
    poses = {name: make_pose_entry(name, home_angles if name == "HOME_SAFE" else None) for name in POSE_ORDER}
    return {
        "metadata": {
            "status": "phase12_manual_teach_template",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": None,
            "step_deg": int(step_deg),
            "note": "Manual taught MOVE_SAFE poses for practical pick-place testing. No IK in this file.",
        },
        "poses": poses,
    }


def load_or_init_output(path: str, home_angles: dict[str, int], step_deg: int) -> dict[str, Any]:
    out_path = Path(path)
    if out_path.exists():
        data = load_required_yaml(path, "taught pose output")
        if "poses" not in data or not isinstance(data["poses"], dict):
            raise RuntimeError(f"Existing taught pose file is missing poses mapping: {path}")
        return data
    return build_default_output(home_angles, step_deg)


def save_output(path: str, payload: dict[str, Any], step_deg: int) -> None:
    payload.setdefault("metadata", {})
    payload["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    payload["metadata"]["step_deg"] = int(step_deg)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def clamp_angle(channel: str, value: int, limits: dict[str, tuple[int, int]]) -> int:
    lo, hi = limits[channel]
    return max(lo, min(hi, value))


def build_move_safe_command(angles: dict[str, int]) -> str:
    return "MOVE_SAFE " + " ".join(str(angles[channel]) for channel in CHANNELS)


def format_angles(angles: dict[str, int]) -> str:
    return " ".join(f"{channel.upper()}={angles[channel]}" for channel in CHANNELS)


def print_session_state(current_angles: dict[str, int], staged_angles: dict[str, int], step_deg: int) -> None:
    print("------------------------------------------------------------")
    print(f"Current robot pose: {format_angles(current_angles)}")
    print(f"Staged pose:        {format_angles(staged_angles)}")
    print(f"Step size: {step_deg} deg")
    print("Commands: chN + | chN - | set chN ANGLE | send | home | save NAME | poses | reset | step N | show | help | quit")


def print_help() -> None:
    print("Manual teach commands:")
    print("  ch1 +            increase ch1 by current step size")
    print("  ch1 -            decrease ch1 by current step size")
    print("  set ch3 120      set one channel directly")
    print("  send             confirm and send staged MOVE_SAFE")
    print("  home             confirm and send HOME, then reset staged pose")
    print("  save OBJECT_HOVER")
    print("                   save current robot pose to the output YAML")
    print("  poses            list saveable pose names")
    print("  reset            discard staged edits and copy current pose")
    print("  step 5           change jog step size")
    print("  show             print current and staged angles")
    print("  help             show this help")
    print("  quit             exit")


def print_pose_names() -> None:
    print("Saveable pose names:")
    for name in POSE_ORDER:
        print(f"  {name}")


def send_with_guard(
    command: str,
    label: str,
    prompt: str,
    port: str,
    baud: int,
    timeout: float,
    dry_run: bool,
    ser,
) -> list[str]:
    print(f"[SAFETY] Command preview: {command}")
    if not require_confirmation(label, prompt):
        print(f"{command} cancelled.")
        return []
    if dry_run:
        print(f"DRY RUN: port={port} baud={baud} command={command}")
        return []
    lines = send_command(ser, command)
    print(f"COMMAND {command}")
    if not lines:
        print("  no response")
        return lines
    for line in lines:
        print(f"  {line}")
    return lines


def require_success(lines: list[str], expected_done: str) -> None:
    if not lines:
        return
    if any(line.startswith("ERR ") for line in lines):
        raise RuntimeError("; ".join(line for line in lines if line.startswith("ERR ")))
    if expected_done not in lines:
        raise RuntimeError(f"Expected {expected_done}, got: {lines}")


def interactive_loop(
    args: argparse.Namespace,
    current_angles: dict[str, int],
    staged_angles: dict[str, int],
    limits: dict[str, tuple[int, int]],
    output_data: dict[str, Any],
    ser,
) -> int:
    step_deg = int(args.step)
    print_session_state(current_angles, staged_angles, step_deg)

    while True:
        try:
            raw = input("teach> ").strip()
        except EOFError:
            print("\n[INFO] End of input. Exiting teach session.")
            return 0

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in {"quit", "exit"}:
            print("[INFO] Teach session finished.")
            return 0

        if cmd == "help":
            print_help()
            continue

        if cmd == "show":
            print_session_state(current_angles, staged_angles, step_deg)
            continue

        if cmd == "poses":
            print_pose_names()
            continue

        if cmd == "reset":
            staged_angles.clear()
            staged_angles.update(current_angles)
            print("[INFO] Staged pose reset to current robot pose.")
            print_session_state(current_angles, staged_angles, step_deg)
            continue

        if cmd == "step":
            if len(parts) != 2:
                print("[ERROR] Usage: step N")
                continue
            try:
                new_step = int(parts[1])
            except ValueError:
                print("[ERROR] Step must be an integer.")
                continue
            if new_step <= 0:
                print("[ERROR] Step must be positive.")
                continue
            step_deg = new_step
            print(f"[INFO] Step size set to {step_deg} deg.")
            continue

        if cmd == "home":
            lines = send_with_guard(
                "HOME",
                "HOME",
                "Type HOME to send HOME.",
                args.port,
                args.baud,
                args.timeout,
                args.dry_run,
                ser,
            )
            try:
                require_success(lines, "DONE HOME")
            except RuntimeError as exc:
                print(f"[ERROR] {exc}")
                continue
            current_angles.clear()
            current_angles.update(read_home_angles(load_required_yaml(args.pose_config, "pose config")))
            staged_angles.clear()
            staged_angles.update(current_angles)
            print("[INFO] Session reset to HOME_SAFE.")
            print_session_state(current_angles, staged_angles, step_deg)
            continue

        if cmd == "send":
            command = build_move_safe_command(staged_angles)
            lines = send_with_guard(
                command,
                "MOVE",
                "Type MOVE to send MOVE_SAFE.",
                args.port,
                args.baud,
                args.timeout,
                args.dry_run,
                ser,
            )
            try:
                require_success(lines, "DONE MOVE_SAFE")
            except RuntimeError as exc:
                print(f"[ERROR] {exc}")
                continue
            current_angles.clear()
            current_angles.update(staged_angles)
            print("[INFO] Current robot pose updated from staged MOVE_SAFE.")
            print_session_state(current_angles, staged_angles, step_deg)
            continue

        if cmd == "save":
            if len(parts) != 2:
                print("[ERROR] Usage: save POSE_NAME")
                continue
            pose_name = parts[1].upper()
            if pose_name not in POSE_ORDER:
                print(f"[ERROR] Unknown pose name: {pose_name}")
                print_pose_names()
                continue
            output_data.setdefault("poses", {})
            output_data["poses"][pose_name] = make_pose_entry(pose_name, current_angles)
            save_output(args.output, output_data, step_deg)
            print(f"[INFO] Saved {pose_name} to {args.output}")
            continue

        if cmd == "set":
            if len(parts) != 3:
                print("[ERROR] Usage: set chN ANGLE")
                continue
            channel = parts[1].lower()
            if channel not in CHANNELS:
                print(f"[ERROR] Unknown channel: {channel}")
                continue
            try:
                angle = int(parts[2])
            except ValueError:
                print("[ERROR] Angle must be an integer.")
                continue
            clamped = clamp_angle(channel, angle, limits)
            if clamped != angle:
                lo, hi = limits[channel]
                print(f"[WARN] {channel} clamped from {angle} to {clamped} within [{lo}, {hi}]")
            staged_angles[channel] = clamped
            print(f"[INFO] Staged {channel} = {clamped}")
            continue

        if len(parts) == 2 and parts[0].lower() in CHANNELS and parts[1] in {"+", "-"}:
            channel = parts[0].lower()
            delta = step_deg if parts[1] == "+" else -step_deg
            before = staged_angles[channel]
            after = clamp_angle(channel, before + delta, limits)
            staged_angles[channel] = after
            if after != before + delta:
                lo, hi = limits[channel]
                print(f"[WARN] {channel} clamped to {after} within [{lo}, {hi}]")
            else:
                print(f"[INFO] Staged {channel}: {before} -> {after}")
            continue

        print("[ERROR] Unknown command.")
        print_help()


def main() -> int:
    args = parse_args()
    if args.step <= 0:
        print("[ERROR] --step must be positive", file=sys.stderr)
        return 2
    if not args.dry_run and not args.yes_i_understand_hardware_risk:
        print(
            "[SAFETY][ERROR] Live teach mode requires --yes-i-understand-hardware-risk.",
            file=sys.stderr,
        )
        return 2

    try:
        servo_cfg = load_required_yaml(args.servo_config, "servo config")
        pose_cfg = load_required_yaml(args.pose_config, "pose config")
        _ = load_required_yaml(args.serial_config, "serial config")
        home_angles = read_home_angles(pose_cfg)
        limits = read_limits(servo_cfg)
        output_data = load_or_init_output(args.output, home_angles, args.step)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    current_angles = deepcopy(home_angles)
    staged_angles = deepcopy(home_angles)

    if args.dry_run:
        print(f"[INFO] DRY RUN mode. Session starts from HOME_SAFE: {format_angles(current_angles)}")
        print(f"[INFO] Intended startup command: HOME on port={args.port} baud={args.baud}")
        return interactive_loop(args, current_angles, staged_angles, limits, output_data, ser=None)

    print("[SAFETY] Live teach session requested.")
    print("[SAFETY] HOME_SAFE is only validated for idle/manual testing, not full path validation.")

    try:
        ser = open_serial(args.port, args.baud, args.timeout)
    except Exception as exc:  # pragma: no cover - hardware-dependent path
        print(f"[SERIAL][ERROR] Failed to open serial port {args.port}: {exc}", file=sys.stderr)
        return 2

    try:
        lines = send_with_guard(
            "HOME",
            "HOME",
            "Type HOME to move robot to HOME_SAFE and begin teach session.",
            args.port,
            args.baud,
            args.timeout,
            False,
            ser,
        )
        require_success(lines, "DONE HOME")
        return interactive_loop(args, current_angles, staged_angles, limits, output_data, ser)
    except RuntimeError as exc:
        print(f"[SERIAL][ERROR] {exc}", file=sys.stderr)
        return 2
    finally:
        ser.close()


if __name__ == "__main__":
    sys.exit(main())
