#!/usr/bin/env python3
"""
test_board_to_robot_transform.py

Phase 4 standalone board-to-robot transform test.

Converts board coordinates in centimeters to robot-frame coordinates in meters.
This tool does not access camera, ESP32/serial, servos, IK, or robot motion.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = "ros2_ws/src/robot_arm_5dof/config/robot_board_transform.yaml"

GRID_POINTS = [
    ("top_left", 0.0, 0.0),
    ("top_right", 27.0, 0.0),
    ("bottom_right", 27.0, 18.0),
    ("bottom_left", 0.0, 18.0),
    ("board_center", 13.5, 9.0),
    ("CAKE_BOWL", 7.0, 6.0),
    ("DONUT_BOWL", 20.0, 6.0),
]


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_transform_config(path: str) -> dict[str, Any]:
    data = load_yaml(path)
    cfg = data.get("board_to_robot", {})
    required = [
        "robot_base_x_board_cm",
        "robot_base_y_board_cm",
        "robot_yaw_offset_deg",
    ]
    missing = [key for key in required if cfg.get(key) is None]
    if missing:
        raise ValueError(f"Missing required transform values in {path}: {missing}")
    return cfg


def transform_board_to_robot(
    board_x_cm: float,
    board_y_cm: float,
    cfg: dict[str, Any],
) -> dict[str, float]:
    base_x = float(cfg["robot_base_x_board_cm"])
    base_y = float(cfg["robot_base_y_board_cm"])
    yaw_deg = float(cfg.get("robot_yaw_offset_deg", 0.0))

    dx_cm = board_x_cm - base_x
    dy_cm = board_y_cm - base_y

    theta = math.radians(yaw_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    robot_x_cm = cos_t * dx_cm - sin_t * dy_cm
    robot_y_cm = sin_t * dx_cm + cos_t * dy_cm

    distance_cm = math.hypot(dx_cm, dy_cm)
    yaw_from_base_deg = math.degrees(math.atan2(dy_cm, dx_cm))

    return {
        "board_x_cm": board_x_cm,
        "board_y_cm": board_y_cm,
        "base_x_cm": base_x,
        "base_y_cm": base_y,
        "dx_cm": dx_cm,
        "dy_cm": dy_cm,
        "yaw_offset_deg": yaw_deg,
        "robot_x_m": robot_x_cm / 100.0,
        "robot_y_m": robot_y_cm / 100.0,
        "distance_cm": distance_cm,
        "yaw_from_base_deg": yaw_from_base_deg,
    }


def print_transform(label: str, result: dict[str, float]) -> None:
    print(f"\n[{label}]")
    print(f"  input board coordinate : ({result['board_x_cm']:.2f}, {result['board_y_cm']:.2f}) cm")
    print(f"  robot base on board    : ({result['base_x_cm']:.2f}, {result['base_y_cm']:.2f}) cm")
    print(f"  dx / dy from base      : dx={result['dx_cm']:.2f} cm, dy={result['dy_cm']:.2f} cm")
    print(f"  yaw offset applied     : {result['yaw_offset_deg']:.2f} deg")
    print(f"  robot coordinate       : x={result['robot_x_m']:.4f} m, y={result['robot_y_m']:.4f} m")
    print(f"  distance from base     : {result['distance_cm']:.2f} cm")
    print(f"  yaw angle from base    : {result['yaw_from_base_deg']:.2f} deg")


def run_grid_test(cfg: dict[str, Any]) -> None:
    print("\n=== Grid Test ===")
    for label, x_cm, y_cm in GRID_POINTS:
        result = transform_board_to_robot(x_cm, y_cm, cfg)
        print(
            f"{label:>13s}: board=({x_cm:5.1f},{y_cm:5.1f}) cm "
            f"dx={result['dx_cm']:7.2f} dy={result['dy_cm']:7.2f} cm "
            f"robot=({result['robot_x_m']: .4f},{result['robot_y_m']: .4f}) m "
            f"dist={result['distance_cm']:6.2f} cm "
            f"yaw={result['yaw_from_base_deg']:7.2f} deg"
        )


def run_interactive(cfg: dict[str, Any]) -> None:
    print("\nInteractive mode. Enter board coordinates as: x y")
    print("Press Ctrl-D or enter a blank line to quit.")
    while True:
        try:
            line = input("board x y cm> ").strip()
        except EOFError:
            print()
            break
        if not line:
            break
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            print("[WARN] Expected two numbers: x y")
            continue
        try:
            x_cm, y_cm = float(parts[0]), float(parts[1])
        except ValueError:
            print("[WARN] Could not parse numbers.")
            continue
        print_transform("interactive", transform_board_to_robot(x_cm, y_cm, cfg))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 board-cm to robot-meter transform test. No robot motion.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--board-x", type=float, required=True)
    parser.add_argument("--board-y", type=float, required=True)
    parser.add_argument("--grid-test", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_transform_config(args.config)
    print(f"[INFO] Loaded transform config: {Path(args.config)}")
    print("[SAFETY] Math-only transform test. No camera, no serial, no IK, no robot motion.")
    print(f"[INFO] status: {cfg.get('status', 'unknown')}")
    if cfg.get("note"):
        print(f"[INFO] note: {cfg['note']}")
    print(f"[INFO] robot_forward_direction_on_board: {cfg.get('robot_forward_direction_on_board', 'unspecified')}")
    print(f"[INFO] units: {cfg.get('units_input', 'cm')} -> {cfg.get('units_output', 'meter')}")

    result = transform_board_to_robot(args.board_x, args.board_y, cfg)
    print_transform("requested", result)

    if args.grid_test:
        run_grid_test(cfg)
    if args.interactive:
        run_interactive(cfg)


if __name__ == "__main__":
    main()
