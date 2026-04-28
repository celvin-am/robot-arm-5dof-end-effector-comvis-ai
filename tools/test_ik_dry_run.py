#!/usr/bin/env python3
"""
test_ik_dry_run.py

Phase 5 dry-run IK and pose planning helper.

This script performs math-only IK candidate generation. It never imports ROS2,
never opens serial, never sends servo commands, and never moves hardware.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_KINEMATICS_CONFIG = "ros2_ws/src/robot_arm_5dof/config/robot_kinematics.yaml"
DEFAULT_SERVO_CONFIG = "ros2_ws/src/robot_arm_5dof/config/servo_config.yaml"
DEFAULT_POSE_CONFIG = "ros2_ws/src/robot_arm_5dof/config/pose_config.yaml"
DEFAULT_TRANSFORM_CONFIG = "ros2_ws/src/robot_arm_5dof/config/robot_board_transform.yaml"
DEFAULT_PICK_PLACE_CONFIG = "ros2_ws/src/robot_arm_5dof/config/pick_place_config.yaml"


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_required(path: str, label: str) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return load_yaml(path)


def board_to_robot(board_x_cm: float, board_y_cm: float, transform_cfg: dict[str, Any]) -> tuple[float, float]:
    cfg = transform_cfg["board_to_robot"]
    base_x = float(cfg["robot_base_x_board_cm"])
    base_y = float(cfg["robot_base_y_board_cm"])
    yaw = math.radians(float(cfg.get("robot_yaw_offset_deg", 0.0)))
    dx = board_x_cm - base_x
    dy = board_y_cm - base_y
    robot_x_cm = math.cos(yaw) * dx - math.sin(yaw) * dy
    robot_y_cm = math.sin(yaw) * dx + math.cos(yaw) * dy
    return robot_x_cm / 100.0, robot_y_cm / 100.0


def mode_z_height(mode: str, cli_z: float, pick_place_cfg: dict[str, Any]) -> float:
    if mode == "custom":
        return cli_z

    key_map = {
        "hover": "z_hover",
        "pick": "z_pick",
        "lift": "z_lift",
        "place": "z_place",
    }
    key = key_map[mode]
    value = pick_place_cfg.get("heights_m", {}).get(key)
    if value is None:
        raise ValueError(
            f"--mode {mode} requires pick_place_config heights_m.{key}, "
            "but it is null. Perform physical Z calibration before using this mode."
        )
    return float(value)


def servo_value(joint_deg: float, model: dict[str, Any]) -> float:
    return (
        float(model["zero_reference_deg"])
        + float(model.get("direction", 1)) * joint_deg
        + float(model.get("offset_deg", 0))
    )


def channel_limits(servo_cfg: dict[str, Any], channel: str) -> tuple[float, float]:
    entry = servo_cfg.get("servos", {}).get(channel, {})
    lo = entry.get("min_angle_deg")
    hi = entry.get("max_angle_deg")
    if lo is None or hi is None:
        return 0.0, 180.0
    return float(lo), float(hi)


def limit_status(angle: float, limits: tuple[float, float]) -> tuple[bool, str]:
    lo, hi = limits
    ok = lo <= angle <= hi
    return ok, f"{angle:.2f} deg in [{lo:.1f}, {hi:.1f}]"


def solve_planar_ik(robot_x: float, robot_y: float, z: float, kin_cfg: dict[str, Any]) -> dict[str, Any]:
    links = kin_cfg["links"]
    shoulder_z = float(links["board_to_shoulder_height_m"])
    l1 = float(links["shoulder_to_elbow_m"])
    l2 = float(links["elbow_to_wrist_rotate_m"])
    tcp_offset = float(links["effective_wrist_to_tcp_m"])

    heading_deg = math.degrees(math.atan2(robot_y, robot_x))
    r_tcp = math.hypot(robot_x, robot_y)
    z_rel = z - shoulder_z
    r_wrist = r_tcp - tcp_offset
    z_wrist = z_rel
    reach = math.hypot(r_wrist, z_wrist)

    result = {
        "base_heading_deg": heading_deg,
        "base_joint_deg": heading_deg - 90.0,
        "r_tcp_m": r_tcp,
        "z_rel_m": z_rel,
        "r_wrist_m": r_wrist,
        "z_wrist_m": z_wrist,
        "reach_m": reach,
        "l1_m": l1,
        "l2_m": l2,
        "tcp_offset_m": tcp_offset,
        "candidates": [],
    }

    if r_wrist < 0:
        result["unreachable_reason"] = (
            f"target horizontal reach {r_tcp:.4f} m is smaller than effective TCP offset {tcp_offset:.4f} m"
        )
        return result

    max_reach = l1 + l2
    min_reach = abs(l1 - l2)
    if reach > max_reach or reach < min_reach:
        result["unreachable_reason"] = (
            f"wrist reach {reach:.4f} m outside planar range [{min_reach:.4f}, {max_reach:.4f}] m"
        )
        return result

    cos_elbow = (reach * reach - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))

    for name, sign in (("elbow_down", 1.0), ("elbow_up", -1.0)):
        elbow_rad = sign * math.acos(cos_elbow)
        shoulder_rad = math.atan2(z_wrist, r_wrist) - math.atan2(
            l2 * math.sin(elbow_rad),
            l1 + l2 * math.cos(elbow_rad),
        )
        shoulder_deg = math.degrees(shoulder_rad)
        elbow_deg = math.degrees(elbow_rad)
        wrist_pitch_deg = -(shoulder_deg + elbow_deg)
        result["candidates"].append(
            {
                "name": name,
                "j1_base_yaw_deg": result["base_joint_deg"],
                "j2_shoulder_pitch_deg": shoulder_deg,
                "j3_elbow_pitch_deg": elbow_deg,
                "j4_wrist_yaw_deg": 0.0,
                "j5_wrist_pitch_deg": wrist_pitch_deg,
            }
        )

    return result


def candidate_servos(candidate: dict[str, float], kin_cfg: dict[str, Any]) -> dict[str, float]:
    model = kin_cfg["servo_model"]
    return {
        "ch1": servo_value(candidate["j1_base_yaw_deg"], model["ch1_base_yaw"]),
        "ch2": servo_value(candidate["j2_shoulder_pitch_deg"], model["ch2_shoulder_pitch"]),
        "ch3": servo_value(candidate["j3_elbow_pitch_deg"], model["ch3_elbow_pitch"]),
        "ch4": servo_value(candidate["j4_wrist_yaw_deg"], model["ch4_wrist_yaw"]),
        "ch5": servo_value(candidate["j5_wrist_pitch_deg"], model["ch5_wrist_pitch"]),
    }


def validate_servos(servos: dict[str, float], servo_cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    ok_all = True
    lines = []
    for ch in ("ch1", "ch2", "ch3", "ch4", "ch5"):
        ok, msg = limit_status(servos[ch], channel_limits(servo_cfg, ch))
        ok_all = ok_all and ok
        prefix = "OK " if ok else "BAD"
        lines.append(f"{prefix} {ch}: {msg}")
    return ok_all, lines


def print_home_pose(pose_cfg: dict[str, Any]) -> None:
    home = pose_cfg.get("poses", {}).get("HOME_SAFE", {})
    print("\nHOME_SAFE (unchanged reference):")
    for ch in ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6"):
        print(f"  {ch}: {home.get(ch)}")


def print_report(
    args: argparse.Namespace,
    target: tuple[float, float, float],
    ik: dict[str, Any],
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    pose_cfg: dict[str, Any],
) -> None:
    robot_x, robot_y, z = target
    links = kin_cfg["links"]
    gripper = servo_cfg.get("servos", {}).get("ch6", {})

    print("\n============================================================")
    print("DRY RUN ONLY - no serial, no ESP32, no ROS2, no hardware motion")
    print("============================================================")
    if args.target_name:
        print(f"target_name: {args.target_name}")
    print(f"mode: {args.mode}")
    print(f"input target robot frame: x={robot_x:.4f} m, y={robot_y:.4f} m, z={z:.4f} m")
    if args.from_board:
        print(f"source board coordinate: x={args.board_x:.2f} cm, y={args.board_y:.2f} cm")

    print("\nLink lengths:")
    for key in (
        "board_to_shoulder_height_m",
        "shoulder_to_elbow_m",
        "elbow_to_wrist_rotate_m",
        "wrist_rotate_to_wrist_pitch_m",
        "wrist_pitch_to_tcp_m",
        "effective_wrist_to_tcp_m",
    ):
        print(f"  {key}: {links[key]:.4f} m")

    print("\nReach geometry:")
    print(f"  base heading atan2(y,x): {ik['base_heading_deg']:.2f} deg")
    print(f"  provisional base joint: {ik['base_joint_deg']:.2f} deg")
    print(f"  horizontal TCP r: {ik['r_tcp_m']:.4f} m")
    print(f"  z relative to shoulder: {ik['z_rel_m']:.4f} m")
    print(f"  provisional wrist target r: {ik['r_wrist_m']:.4f} m")
    print(f"  provisional wrist target z: {ik['z_wrist_m']:.4f} m")
    print(f"  planar wrist reach: {ik['reach_m']:.4f} m")

    print("\nCH6 gripper reference (not part of IK):")
    print(f"  open_angle_deg: {gripper.get('open_angle_deg')}")
    print(f"  close_angle_deg: {gripper.get('close_angle_deg')}")

    if args.print_home:
        print_home_pose(pose_cfg)

    if "unreachable_reason" in ik:
        print("\nREACHABLE: NO")
        print(f"reason: {ik['unreachable_reason']}")
        print("No servo candidate generated.")
        return

    print("\nREACHABLE: YES")
    for candidate in ik["candidates"]:
        print(f"\nCandidate: {candidate['name']}")
        print("  Raw joint angles:")
        for key in (
            "j1_base_yaw_deg",
            "j2_shoulder_pitch_deg",
            "j3_elbow_pitch_deg",
            "j4_wrist_yaw_deg",
            "j5_wrist_pitch_deg",
        ):
            print(f"    {key}: {candidate[key]:.2f} deg")
        servos = candidate_servos(candidate, kin_cfg)
        ok, limit_lines = validate_servos(servos, servo_cfg)
        print("  Candidate servo angles CH1-CH5:")
        for ch in ("ch1", "ch2", "ch3", "ch4", "ch5"):
            print(f"    {ch}: {servos[ch]:.2f} deg")
        print(f"  Servo limit check: {'PASS' if ok else 'FAIL'}")
        for line in limit_lines:
            print(f"    {line}")

    print("\nNotes:")
    print("  CH4 wrist yaw remains neutral/home for this first dry run.")
    print("  CH5 wrist pitch compensation is provisional only; sign/offset must be validated.")
    print("  No command was sent to hardware.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 dry-run IK candidate calculator. No robot motion.",
    )
    parser.add_argument("--robot-x", type=float, help="Robot-frame x target in meters")
    parser.add_argument("--robot-y", type=float, help="Robot-frame y target in meters")
    parser.add_argument("--z", type=float, required=True, help="Target z height in meters")
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
    parser.add_argument("--board-x", type=float)
    parser.add_argument("--board-y", type=float)
    parser.add_argument("--from-board", action="store_true")
    parser.add_argument("--target-name")
    parser.add_argument("--mode", choices=["hover", "pick", "lift", "place", "custom"], default="custom")
    parser.add_argument("--print-home", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    kin_cfg = load_required(args.kinematics_config, "kinematics config")
    servo_cfg = load_required(args.servo_config, "servo config")
    pose_cfg = load_required(args.pose_config, "pose config")
    pick_place_cfg = load_required(DEFAULT_PICK_PLACE_CONFIG, "pick/place config")

    if args.from_board:
        if args.board_x is None or args.board_y is None:
            print("[ERROR] --from-board requires --board-x and --board-y", file=sys.stderr)
            return 2
        transform_cfg = load_required(args.transform_config, "transform config")
        robot_x, robot_y = board_to_robot(args.board_x, args.board_y, transform_cfg)
    else:
        if args.robot_x is None or args.robot_y is None:
            print("[ERROR] --robot-x and --robot-y are required unless --from-board is used", file=sys.stderr)
            return 2
        robot_x, robot_y = args.robot_x, args.robot_y

    try:
        z = mode_z_height(args.mode, args.z, pick_place_cfg)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.check_only:
        print("[INFO] Configs loaded. Performing dry-run math only.")

    ik = solve_planar_ik(robot_x, robot_y, z, kin_cfg)
    print_report(args, (robot_x, robot_y, z), ik, kin_cfg, servo_cfg, pose_cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
