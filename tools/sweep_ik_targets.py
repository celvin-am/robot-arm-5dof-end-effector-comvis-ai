#!/usr/bin/env python3
"""
sweep_ik_targets.py

Phase 5E dry-run reachability and servo-limit sweep for fixed board targets.

This tool reuses the existing IK dry-run math and reports which target/Z
combinations pass configured servo limits. It never imports ROS2 or serial
modules, never opens hardware, and never sends commands to the robot.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

from test_ik_dry_run import (
    DEFAULT_KINEMATICS_CONFIG,
    DEFAULT_PICK_PLACE_CONFIG,
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


DEFAULT_Z_VALUES = "0.05,0.08,0.10,0.12,0.15"
DEFAULT_TCP_MODE = "none"
DEFAULT_SOLUTION_PREFERENCE = "elbow_up"
DEFAULT_MAX_PRINT = 100

BOARD_TARGETS = (
    ("center", 13.5, 9.0),
    ("left_mid", 7.0, 9.0),
    ("right_mid", 20.0, 9.0),
    ("cake_bowl", 7.0, 6.0),
    ("donut_bowl", 20.0, 6.0),
    ("lower_center", 13.5, 15.0),
    ("upper_center", 13.5, 3.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5E dry-run IK reachability and servo-limit sweep. No robot motion.",
    )
    parser.add_argument("--z-values", default=DEFAULT_Z_VALUES, help="Comma-separated Z values in meters")
    parser.add_argument("--tcp-offset-mode", default=DEFAULT_TCP_MODE)
    parser.add_argument(
        "--solution-preference",
        choices=["elbow_up", "elbow_down", "best"],
        default=DEFAULT_SOLUTION_PREFERENCE,
    )
    parser.add_argument("--output-csv", help="Optional CSV report output path")
    parser.add_argument("--max-print", type=int, default=DEFAULT_MAX_PRINT)
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
    parser.add_argument("--pick-place-config", default=DEFAULT_PICK_PLACE_CONFIG)
    return parser.parse_args()


def parse_z_values(text: str) -> list[float]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("No valid Z values were provided")
    return values


def build_targets(transform_cfg: dict[str, Any]) -> list[dict[str, float | str]]:
    targets = []
    for name, board_x, board_y in BOARD_TARGETS:
        robot_x, robot_y = board_to_robot(board_x, board_y, transform_cfg)
        targets.append(
            {
                "target_name": name,
                "board_x_cm": board_x,
                "board_y_cm": board_y,
                "robot_x_m": robot_x,
                "robot_y_m": robot_y,
            }
        )
    return targets


def limit_margin(angle: float, limits: tuple[float, float]) -> float:
    lo, hi = limits
    return min(angle - lo, hi - angle)


def candidate_analysis(candidate: dict[str, Any], kin_cfg: dict[str, Any], servo_cfg: dict[str, Any]) -> dict[str, Any]:
    servos = candidate_servos(candidate, kin_cfg)
    ok, limit_lines = validate_servos(servos, servo_cfg)
    failed_channels = [line[4:] for line in limit_lines if line.startswith("BAD ")]
    margins = {ch: limit_margin(servos[ch], channel_limits(servo_cfg, ch)) for ch in ("ch1", "ch2", "ch3", "ch4", "ch5")}
    min_margin = min(margins.values())
    violation_sum = sum(abs(value) for value in margins.values() if value < 0.0)
    return {
        "candidate_name": candidate["name"],
        "raw_candidate": candidate,
        "servos": servos,
        "ok": ok,
        "failed_channels": failed_channels,
        "limit_lines": limit_lines,
        "margins": margins,
        "min_margin_deg": min_margin,
        "violation_sum_deg": violation_sum,
    }


def select_candidate(
    ik_result: dict[str, Any],
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    preference: str,
) -> dict[str, Any] | None:
    if "unreachable_reason" in ik_result:
        return None

    analyses = [candidate_analysis(candidate, kin_cfg, servo_cfg) for candidate in ik_result["candidates"]]
    by_name = {item["candidate_name"]: item for item in analyses}

    if preference in by_name:
        return by_name[preference]

    analyses.sort(
        key=lambda item: (
            0 if item["ok"] else 1,
            -item["min_margin_deg"] if item["ok"] else item["violation_sum_deg"],
            item["candidate_name"],
        )
    )
    return analyses[0]


def evaluate_case(
    target: dict[str, float | str],
    z_value: float,
    tcp_offset_mode: str,
    preference: str,
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
) -> dict[str, Any]:
    ik_result = solve_planar_ik(
        float(target["robot_x_m"]),
        float(target["robot_y_m"]),
        z_value,
        kin_cfg,
        tcp_offset_mode,
        0.0,
        1.0,
    )
    selected = select_candidate(ik_result, kin_cfg, servo_cfg, preference)

    if "unreachable_reason" in ik_result:
        return {
            **target,
            "z_m": z_value,
            "reachable": False,
            "status": "UNREACHABLE",
            "selected_solution": None,
            "failed_channels": [ik_result["unreachable_reason"]],
            "servos": None,
            "min_margin_deg": None,
            "ik": ik_result,
        }

    assert selected is not None
    return {
        **target,
        "z_m": z_value,
        "reachable": True,
        "status": "PASS" if selected["ok"] else "FAIL",
        "selected_solution": selected["candidate_name"],
        "failed_channels": selected["failed_channels"],
        "servos": selected["servos"],
        "min_margin_deg": selected["min_margin_deg"],
        "ik": ik_result,
        "selected": selected,
    }


def format_servos(servos: dict[str, float] | None) -> str:
    if servos is None:
        return "n/a"
    return ", ".join(f"{ch}={servos[ch]:.2f}" for ch in ("ch1", "ch2", "ch3", "ch4", "ch5"))


def print_case(case: dict[str, Any]) -> None:
    print(
        f"{case['target_name']}: board=({case['board_x_cm']:.1f},{case['board_y_cm']:.1f}) cm "
        f"robot=({case['robot_x_m']:.4f},{case['robot_y_m']:.4f}) m "
        f"z={case['z_m']:.2f} reachable={'YES' if case['reachable'] else 'NO'} "
        f"status={case['status']} solution={case['selected_solution'] or 'n/a'}"
    )
    print(f"  CH1-CH5: {format_servos(case['servos'])}")
    print("  failed_channels: " + ("none" if not case["failed_channels"] else "; ".join(case["failed_channels"])))


def print_summary(cases: list[dict[str, Any]], preference: str, tcp_offset_mode: str) -> None:
    total = len(cases)
    passes = [case for case in cases if case["status"] == "PASS"]
    fails = [case for case in cases if case["status"] == "FAIL"]
    unreachable = [case for case in cases if case["status"] == "UNREACHABLE"]

    safest = sorted(
        passes,
        key=lambda case: (-float(case["min_margin_deg"]), case["z_m"], case["target_name"]),
    )

    print("\nSummary:")
    print(f"  total cases: {total}")
    print(f"  PASS count: {len(passes)}")
    print(f"  FAIL count: {len(fails)}")
    print(f"  unreachable count: {len(unreachable)}")
    print(f"  tcp_offset_mode: {tcp_offset_mode}")
    print(f"  solution_preference: {preference}")

    print("\nSafest target/z combinations:")
    if not safest:
        print("  none")
    else:
        for case in safest[:5]:
            print(
                f"  {case['target_name']} z={case['z_m']:.2f} "
                f"solution={case['selected_solution']} min_limit_margin_deg={case['min_margin_deg']:.2f} "
                f"servos=[{format_servos(case['servos'])}]"
            )

    print("\nRecommended first manual hardware test target (future hardware phase only):")
    if safest:
        first = safest[0]
        print(
            f"  {first['target_name']} at z={first['z_m']:.2f} with {first['selected_solution']} "
            f"(min_limit_margin_deg={first['min_margin_deg']:.2f})"
        )
    else:
        print("  none")


def write_csv(path: str, cases: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_name",
                "board_x_cm",
                "board_y_cm",
                "robot_x_m",
                "robot_y_m",
                "z_m",
                "reachable",
                "status",
                "selected_solution",
                "min_margin_deg",
                "failed_channels",
                "ch1",
                "ch2",
                "ch3",
                "ch4",
                "ch5",
            ],
        )
        writer.writeheader()
        for case in cases:
            servos = case["servos"] or {}
            writer.writerow(
                {
                    "target_name": case["target_name"],
                    "board_x_cm": case["board_x_cm"],
                    "board_y_cm": case["board_y_cm"],
                    "robot_x_m": case["robot_x_m"],
                    "robot_y_m": case["robot_y_m"],
                    "z_m": case["z_m"],
                    "reachable": case["reachable"],
                    "status": case["status"],
                    "selected_solution": case["selected_solution"],
                    "min_margin_deg": case["min_margin_deg"],
                    "failed_channels": " | ".join(case["failed_channels"]),
                    "ch1": servos.get("ch1"),
                    "ch2": servos.get("ch2"),
                    "ch3": servos.get("ch3"),
                    "ch4": servos.get("ch4"),
                    "ch5": servos.get("ch5"),
                }
            )


def main() -> int:
    args = parse_args()

    try:
        z_values = parse_z_values(args.z_values)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    kin_cfg = load_required(args.kinematics_config, "kinematics config")
    servo_cfg = load_required(args.servo_config, "servo config")
    pose_cfg = load_required(args.pose_config, "pose config")
    transform_cfg = load_required(args.transform_config, "transform config")
    pick_place_cfg = load_required(args.pick_place_config, "pick/place config")

    _ = pose_cfg
    _ = pick_place_cfg

    targets = build_targets(transform_cfg)
    supported_modes = kin_cfg.get("ik", {}).get("tcp_offset_modes_supported", ["none", "planar", "vertical_down", "mixed"])
    if args.tcp_offset_mode not in supported_modes:
        print(f"[ERROR] Unsupported tcp offset mode {args.tcp_offset_mode!r}. Supported: {supported_modes}", file=sys.stderr)
        return 2

    print("============================================================")
    print("PHASE 5E IK TARGET SWEEP - DRY RUN ONLY")
    print("No serial, no ESP32, no ROS2, no hardware motion.")
    print("============================================================")

    cases = []
    for target in targets:
        for z_value in z_values:
            cases.append(
                evaluate_case(
                    target,
                    z_value,
                    args.tcp_offset_mode,
                    args.solution_preference,
                    kin_cfg,
                    servo_cfg,
                )
            )

    for case in cases[: args.max_print]:
        print_case(case)

    print_summary(cases, args.solution_preference, args.tcp_offset_mode)

    if args.output_csv:
        write_csv(args.output_csv, cases)
        print(f"\nCSV written: {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
