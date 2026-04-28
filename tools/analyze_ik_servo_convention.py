#!/usr/bin/env python3
"""
analyze_ik_servo_convention.py

Phase 5C math-only sweep for servo direction and offset conventions.

This tool reuses the existing dry-run IK math, evaluates candidate servo
conventions for CH2/CH3/CH5, and ranks them by how many test scenarios pass
configured servo limits. It never imports ROS2 or serial modules, never opens
hardware, and never sends commands to the robot.
"""

import argparse
import copy
import itertools
import sys
from typing import Any

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
)


DEFAULT_Z_VALUES = "0.05,0.08,0.10,0.12,0.15"
TCP_MODES = ("none", "vertical_down")
TARGETS = (
    ("center", 13.5, 9.0),
    ("left", 7.0, 9.0),
    ("right", 20.0, 9.0),
    ("cake_bowl", 7.0, 6.0),
    ("donut_bowl", 20.0, 6.0),
)
SWEEP_VALUES = (1, -1)
OFFSET_VALUES = (-90, 0, 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5C math-only sweep for IK servo direction/offset conventions.",
    )
    parser.add_argument("--z-values", default=DEFAULT_Z_VALUES, help="Comma-separated Z values in meters")
    parser.add_argument("--max-results", type=int, default=20, help="Maximum ranked convention rows to print")
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
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


def format_signed_int(value: int) -> str:
    return f"+{value}" if value >= 0 else str(value)


def format_servo_angles(servos: dict[str, float]) -> str:
    return ", ".join(f"{ch}={servos[ch]:.2f}" for ch in ("ch1", "ch2", "ch3", "ch4", "ch5"))


def violation_amount(angle: float, limits: tuple[float, float]) -> float:
    lo, hi = limits
    if angle < lo:
        return lo - angle
    if angle > hi:
        return angle - hi
    return 0.0


def failed_channels(servos: dict[str, float], servo_cfg: dict[str, Any]) -> tuple[list[str], float]:
    failures = []
    total_violation = 0.0
    for ch in ("ch1", "ch2", "ch3", "ch4", "ch5"):
        limits = channel_limits(servo_cfg, ch)
        violation = violation_amount(servos[ch], limits)
        if violation > 0.0:
            lo, hi = limits
            failures.append(f"{ch}={servos[ch]:.2f} not in [{lo:.1f}, {hi:.1f}]")
            total_violation += violation
    return failures, total_violation


def build_targets(transform_cfg: dict[str, Any]) -> list[dict[str, float | str]]:
    resolved = []
    for name, board_x, board_y in TARGETS:
        robot_x, robot_y = board_to_robot(board_x, board_y, transform_cfg)
        resolved.append(
            {
                "name": name,
                "board_x_cm": board_x,
                "board_y_cm": board_y,
                "robot_x_m": robot_x,
                "robot_y_m": robot_y,
            }
        )
    return resolved


def build_convention(base_kin_cfg: dict[str, Any], ch2_dir: int, ch2_offset: int, ch3_dir: int, ch3_offset: int, ch5_dir: int, ch5_offset: int) -> dict[str, Any]:
    kin_cfg = copy.deepcopy(base_kin_cfg)
    servo_model = kin_cfg["servo_model"]
    servo_model["ch2_shoulder_pitch"]["direction"] = ch2_dir
    servo_model["ch2_shoulder_pitch"]["offset_deg"] = ch2_offset
    servo_model["ch3_elbow_pitch"]["direction"] = ch3_dir
    servo_model["ch3_elbow_pitch"]["offset_deg"] = ch3_offset
    servo_model["ch5_wrist_pitch"]["direction"] = ch5_dir
    servo_model["ch5_wrist_pitch"]["offset_deg"] = ch5_offset
    return kin_cfg


def evaluate_best_candidate(
    ik_result: dict[str, Any],
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
) -> dict[str, Any]:
    if "unreachable_reason" in ik_result:
        return {
            "ok": False,
            "candidate_name": None,
            "servos": None,
            "failed_channels": [ik_result["unreachable_reason"]],
            "violation_sum_deg": 1000.0,
            "failed_count": 5,
        }

    evaluated = []
    for candidate in ik_result["candidates"]:
        servos = candidate_servos(candidate, kin_cfg)
        failures, violation_sum = failed_channels(servos, servo_cfg)
        evaluated.append(
            {
                "ok": not failures,
                "candidate_name": candidate["name"],
                "candidate": candidate,
                "servos": servos,
                "failed_channels": failures,
                "violation_sum_deg": violation_sum,
                "failed_count": len(failures),
            }
        )

    evaluated.sort(
        key=lambda item: (
            0 if item["ok"] else 1,
            item["failed_count"],
            item["violation_sum_deg"],
            item["candidate_name"],
        )
    )
    return evaluated[0]


def evaluate_convention(
    convention: dict[str, int],
    base_kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    targets: list[dict[str, float | str]],
    z_values: list[float],
) -> dict[str, Any]:
    kin_cfg = build_convention(
        base_kin_cfg,
        convention["ch2_dir"],
        convention["ch2_offset"],
        convention["ch3_dir"],
        convention["ch3_offset"],
        convention["ch5_dir"],
        convention["ch5_offset"],
    )

    outcomes = []
    pass_count = 0
    elbow_usage = {"elbow_down": 0, "elbow_up": 0}
    total_failed_channels = 0
    total_violation = 0.0
    unreachable_count = 0

    for target, z_value, tcp_mode in itertools.product(targets, z_values, TCP_MODES):
        ik_result = solve_planar_ik(
            float(target["robot_x_m"]),
            float(target["robot_y_m"]),
            z_value,
            kin_cfg,
            tcp_mode,
            0.0,
            1.0,
        )
        best = evaluate_best_candidate(ik_result, kin_cfg, servo_cfg)
        outcome = {
            "target_name": target["name"],
            "board_x_cm": target["board_x_cm"],
            "board_y_cm": target["board_y_cm"],
            "robot_x_m": target["robot_x_m"],
            "robot_y_m": target["robot_y_m"],
            "z_m": z_value,
            "tcp_mode": tcp_mode,
            "best": best,
            "ik": ik_result,
        }
        outcomes.append(outcome)

        if best["ok"]:
            pass_count += 1
            elbow_usage[best["candidate_name"]] += 1
        else:
            total_failed_channels += best["failed_count"]
            total_violation += best["violation_sum_deg"]
            if best["candidate_name"] is None:
                unreachable_count += 1

    failing_outcomes = [item for item in outcomes if not item["best"]["ok"]]
    passing_outcomes = [item for item in outcomes if item["best"]["ok"]]
    closest_fail = min(
        failing_outcomes,
        key=lambda item: (item["best"]["failed_count"], item["best"]["violation_sum_deg"], item["target_name"]),
        default=None,
    )
    sample_pass = passing_outcomes[0] if passing_outcomes else None

    return {
        "convention": convention,
        "pass_count": pass_count,
        "total_cases": len(outcomes),
        "elbow_usage": elbow_usage,
        "total_failed_channels": total_failed_channels,
        "total_violation_deg": total_violation,
        "unreachable_count": unreachable_count,
        "sample_pass": sample_pass,
        "closest_fail": closest_fail,
        "outcomes": outcomes,
    }


def convention_label(convention: dict[str, int]) -> str:
    return (
        f"CH2(dir={format_signed_int(convention['ch2_dir'])}, off={format_signed_int(convention['ch2_offset'])}) "
        f"CH3(dir={format_signed_int(convention['ch3_dir'])}, off={format_signed_int(convention['ch3_offset'])}) "
        f"CH5(dir={format_signed_int(convention['ch5_dir'])}, off={format_signed_int(convention['ch5_offset'])})"
    )


def print_outcome(prefix: str, outcome: dict[str, Any] | None) -> None:
    if outcome is None:
        print(f"  {prefix}: none")
        return

    best = outcome["best"]
    elbow = best["candidate_name"] or "unreachable"
    print(
        f"  {prefix}: target={outcome['target_name']} z={outcome['z_m']:.2f} "
        f"tcp={outcome['tcp_mode']} elbow={elbow}"
    )
    if best["servos"] is not None:
        print(f"    servos: {format_servo_angles(best['servos'])}")
    print(
        "    failed: "
        + ("none" if not best["failed_channels"] else "; ".join(best["failed_channels"]))
    )


def print_suggested_patch(best_summary: dict[str, Any]) -> None:
    convention = best_summary["convention"]
    print("\nSuggested config patch text only (not applied):")
    print("robot_kinematics.yaml")
    print("servo_model:")
    print("  ch2_shoulder_pitch:")
    print(f"    direction: {convention['ch2_dir']}")
    print(f"    offset_deg: {convention['ch2_offset']}")
    print("  ch3_elbow_pitch:")
    print(f"    direction: {convention['ch3_dir']}")
    print(f"    offset_deg: {convention['ch3_offset']}")
    print("  ch5_wrist_pitch:")
    print(f"    direction: {convention['ch5_dir']}")
    print(f"    offset_deg: {convention['ch5_offset']}")
    print("\nFuture servo_config.yaml sync to test later, not applied in this phase:")
    print("servos:")
    print("  ch2:")
    print(f"    direction: {convention['ch2_dir']}")
    print(f"    offset_deg: {convention['ch2_offset']}")
    print("  ch3:")
    print(f"    direction: {convention['ch3_dir']}")
    print(f"    offset_deg: {convention['ch3_offset']}")
    print("  ch5:")
    print(f"    direction: {convention['ch5_dir']}")
    print(f"    offset_deg: {convention['ch5_offset']}")


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

    home = pose_cfg.get("poses", {}).get("HOME_SAFE", {})
    targets = build_targets(transform_cfg)

    print("============================================================")
    print("PHASE 5C IK SERVO CONVENTION SWEEP - MATH ONLY")
    print("No serial, no ESP32, no ROS2, no hardware motion.")
    print("============================================================")
    print(
        f"HOME_SAFE reference preserved: ch1={home.get('ch1')} ch2={home.get('ch2')} "
        f"ch3={home.get('ch3')} ch4={home.get('ch4')} ch5={home.get('ch5')} ch6={home.get('ch6')}"
    )
    print(f"Targets tested: {', '.join(str(target['name']) for target in targets)}")
    print(f"Z values tested: {', '.join(f'{value:.2f}' for value in z_values)}")
    print(f"TCP modes tested: {', '.join(TCP_MODES)}")
    print("Sweeping CH2/CH3/CH5 direction in {+1,-1} and offset in {-90,0,+90}.")
    print("CH1 and CH4 mapping remain fixed from current robot_kinematics.yaml.")

    summaries = []
    for ch2_dir, ch2_offset, ch3_dir, ch3_offset, ch5_dir, ch5_offset in itertools.product(
        SWEEP_VALUES,
        OFFSET_VALUES,
        SWEEP_VALUES,
        OFFSET_VALUES,
        SWEEP_VALUES,
        OFFSET_VALUES,
    ):
        convention = {
            "ch2_dir": ch2_dir,
            "ch2_offset": ch2_offset,
            "ch3_dir": ch3_dir,
            "ch3_offset": ch3_offset,
            "ch5_dir": ch5_dir,
            "ch5_offset": ch5_offset,
        }
        summaries.append(evaluate_convention(convention, kin_cfg, servo_cfg, targets, z_values))

    summaries.sort(
        key=lambda item: (
            -item["pass_count"],
            item["unreachable_count"],
            item["total_failed_channels"],
            item["total_violation_deg"],
            convention_label(item["convention"]),
        )
    )

    print(f"\nRanked results (top {min(args.max_results, len(summaries))} conventions):")
    for index, summary in enumerate(summaries[: args.max_results], start=1):
        print(
            f"\n[{index:02d}] PASS {summary['pass_count']}/{summary['total_cases']} "
            f"| unreachable={summary['unreachable_count']} "
            f"| failed_channels={summary['total_failed_channels']} "
            f"| violation_sum_deg={summary['total_violation_deg']:.2f}"
        )
        print(f"  convention: {convention_label(summary['convention'])}")
        print(
            f"  elbow wins: elbow_down={summary['elbow_usage']['elbow_down']} "
            f"elbow_up={summary['elbow_usage']['elbow_up']}"
        )
        print_outcome("sample_pass", summary["sample_pass"])
        print_outcome("closest_fail", summary["closest_fail"])

    best = summaries[0]
    print("\nBest candidate convention summary:")
    print(f"  convention: {convention_label(best['convention'])}")
    print(f"  pass_count: {best['pass_count']}/{best['total_cases']}")
    print(f"  unreachable_count: {best['unreachable_count']}")
    print(f"  total_failed_channels: {best['total_failed_channels']}")
    print(f"  total_violation_deg: {best['total_violation_deg']:.2f}")
    print(
        f"  elbow usage: elbow_down={best['elbow_usage']['elbow_down']} "
        f"elbow_up={best['elbow_usage']['elbow_up']}"
    )
    print_suggested_patch(best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
