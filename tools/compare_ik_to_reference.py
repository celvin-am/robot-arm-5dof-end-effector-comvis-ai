#!/usr/bin/env python3
"""
compare_ik_to_reference.py

Dry-run diagnostic tool for comparing current IK output against human-taught
IK reference samples. This tool never opens serial, never moves hardware, and
never uses camera or YOLO.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Any

from ik_servo_calibration_utils import (
    ACTIVE_IK_CHANNELS,
    DEFAULT_IK_SERVO_CALIBRATION_CONFIG,
    apply_ik_servo_calibration,
    apply_z_mode_correction,
    build_ik_servo_calibration_document,
    fit_affine_channels_from_samples,
    fit_z_mode_corrections_from_samples,
    load_ik_servo_calibration,
    load_yaml_if_exists,
    write_ik_servo_calibration,
)
from test_ik_dry_run import (
    DEFAULT_KINEMATICS_CONFIG,
    DEFAULT_POSE_CONFIG,
    DEFAULT_SERVO_CONFIG,
    DEFAULT_TRANSFORM_CONFIG,
    board_to_robot,
    candidate_servos_with_debug,
    load_required,
    solve_planar_ik,
    validate_servos,
)
from test_ik_to_move_safe import (
    build_move_command,
    clamp_move_safe_angles,
    rounded_move_safe_angles,
    select_candidate,
    validate_move_safe_angles,
)


DEFAULT_REFERENCE_CONFIG = "ros2_ws/src/robot_arm_5dof/config/ik_reference_samples.yaml"
ALL_CHANNELS = ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6")
DEFAULT_SOLUTION = "elbow_up"
PREFERRED_MODE_ORDER = ("none", "planar", "vertical_down", "mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run diagnostic tool for comparing current IK output to human-taught IK reference samples.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample", help="Reference sample name, e.g. IK_REF_PRE_PICK_CENTER")
    selection.add_argument("--all", action="store_true", help="Compare all samples in the reference config")
    parser.add_argument("--tcp-offset-mode", default="none")
    parser.add_argument("--reference-config", default=DEFAULT_REFERENCE_CONFIG)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--fit-affine", action="store_true")
    parser.add_argument("--write-calibration", action="store_true")
    parser.add_argument("--use-ik-servo-calibration", action="store_true")
    parser.add_argument("--ik-servo-calibration-config", default=DEFAULT_IK_SERVO_CALIBRATION_CONFIG)
    parser.add_argument("--compare-tcp-offset-modes", action="store_true")
    parser.add_argument("--use-z-mode-correction", action="store_true")
    parser.add_argument("--fit-z-mode-corrections", action="store_true")
    parser.add_argument("--write-z-mode-corrections", action="store_true")
    return parser.parse_args()


def comparison_grade(max_abs_error: float) -> str:
    if max_abs_error <= 5.0:
        return "PASS"
    if max_abs_error <= 15.0:
        return "WARN"
    return "FAIL"


def load_reference_samples(path: str) -> dict[str, Any]:
    root = load_required(path, "IK reference config")
    samples = root.get("samples", {})
    if not isinstance(samples, dict):
        raise RuntimeError(f"[CONFIG][ERROR] samples mapping missing in {path}")
    return samples


def select_samples(args: argparse.Namespace, all_samples: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if args.all:
        return list(all_samples.items())

    assert args.sample is not None
    sample = all_samples.get(args.sample)
    if not isinstance(sample, dict):
        raise RuntimeError(f"[CONFIG][ERROR] Sample {args.sample!r} not found in reference config")
    return [(args.sample, sample)]


def evaluate_sample(
    sample_name: str,
    sample: dict[str, Any],
    args: argparse.Namespace,
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    pose_cfg: dict[str, Any],
    transform_cfg: dict[str, Any],
    ik_servo_cal_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    board = sample.get("board", {})
    robot = sample.get("robot", {})
    servo = sample.get("servo", {})
    gripper_state = sample.get("gripper_state", {})

    board_x = float(board["x_cm"])
    board_y = float(board["y_cm"])
    z_m = float(robot["z_m"])
    z_mode = str(sample.get("z_mode", "custom"))
    gripper_state_value = str(gripper_state.get("value", "unknown"))
    reference_servos = {channel: int(servo[channel]) for channel in ALL_CHANNELS}

    robot_x, robot_y = board_to_robot(board_x, board_y, transform_cfg)
    ik_result = solve_planar_ik(robot_x, robot_y, z_m, kin_cfg, args.tcp_offset_mode, 0.0, 1.0)

    if "unreachable_reason" in ik_result:
        return {
            "sample_name": sample_name,
            "board_x": board_x,
            "board_y": board_y,
            "robot_x": robot_x,
            "robot_y": robot_y,
            "z_m": z_m,
            "z_mode": z_mode,
            "gripper_state": gripper_state_value,
            "reference_servos": reference_servos,
            "ik_unreachable_reason": ik_result["unreachable_reason"],
        }

    candidate = select_candidate(ik_result, DEFAULT_SOLUTION)
    raw_ik_servos, ch1_debug = candidate_servos_with_debug(candidate, kin_cfg, servo_cfg)
    calibration_logs: list[str] = []
    z_mode_logs: list[str] = []
    ik_servos = dict(raw_ik_servos)
    if ik_servo_cal_cfg is not None:
        ik_servos, calibration_logs = apply_ik_servo_calibration(ik_servos, ik_servo_cal_cfg)
    if args.use_z_mode_correction and ik_servo_cal_cfg is not None:
        ik_servos, z_mode_logs = apply_z_mode_correction(ik_servos, ik_servo_cal_cfg, z_mode)
    limit_ok, limit_lines = validate_servos(ik_servos, servo_cfg)

    home = pose_cfg.get("poses", {}).get("HOME_SAFE", {})
    home_gripper = home.get("ch6")
    if home_gripper is None:
        raise RuntimeError("[CONFIG][ERROR] HOME_SAFE ch6 is required in pose_config.yaml")

    move_angles = rounded_move_safe_angles(ik_servos, int(home_gripper))
    move_angles, clamp_notes = clamp_move_safe_angles(move_angles, servo_cfg)
    rounded_failures = validate_move_safe_angles(move_angles, servo_cfg)

    ik_move_safe = {channel: move_angles[index] for index, channel in enumerate(ALL_CHANNELS)}
    errors = {channel: ik_move_safe[channel] - reference_servos[channel] for channel in ALL_CHANNELS}
    active_abs_errors = {channel: abs(errors[channel]) for channel in ACTIVE_IK_CHANNELS}
    mean_abs_error = statistics.mean(active_abs_errors.values())
    max_abs_error = max(active_abs_errors.values())
    worst_channel = max(active_abs_errors, key=active_abs_errors.get)

    failures = []
    if not limit_ok:
        failures.extend(line[4:] if line.startswith("BAD ") else line for line in limit_lines if line.startswith("BAD "))
    failures.extend(rounded_failures)

    return {
        "sample_name": sample_name,
        "board_x": board_x,
        "board_y": board_y,
        "robot_x": robot_x,
        "robot_y": robot_y,
        "z_m": z_m,
        "z_mode": z_mode,
        "gripper_state": gripper_state_value,
        "reference_servos": reference_servos,
        "candidate_name": candidate["name"],
        "raw_ik_servos_float": raw_ik_servos,
        "ik_servos_float": ik_servos,
        "ik_move_safe": ik_move_safe,
        "errors": errors,
        "active_abs_errors": active_abs_errors,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "worst_channel": worst_channel,
        "grade": comparison_grade(max_abs_error),
        "clamp_notes": clamp_notes,
        "failures": failures,
        "ch1_debug": ch1_debug,
        "calibration_logs": calibration_logs,
        "z_mode_logs": z_mode_logs,
    }


def print_result(result: dict[str, Any], calibrated: bool, use_z_mode_correction: bool) -> None:
    print("============================================================")
    print(f"[IK_REF] sample: {result['sample_name']}")
    print("============================================================")
    print(f"target board: x={result['board_x']:.2f} cm, y={result['board_y']:.2f} cm")
    print(f"target robot: x={result['robot_x']:.4f} m, y={result['robot_y']:.4f} m, z={result['z_m']:.4f} m")
    print(f"z_mode: {result['z_mode']}")
    print(f"gripper_state: {result['gripper_state']}")

    if "ik_unreachable_reason" in result:
        print("[IK_REF][ERROR] IK unreachable")
        print(f"reason: {result['ik_unreachable_reason']}")
        print("reference MOVE_SAFE:", build_move_command([result["reference_servos"][ch] for ch in ALL_CHANNELS]))
        print()
        return

    print(f"selected IK solution: {result['candidate_name']}")
    print(f"[IK] base_angle_relative_deg = {result['ch1_debug']['base_angle_relative_deg']:.2f}")
    print(f"[IK] ch1_front_servo_deg = {result['ch1_debug']['ch1_front_servo_deg']:.2f}")
    print(f"[IK] ch1_servo_deg = {result['ch1_debug']['ch1_servo_deg']:.2f}")
    if calibrated and result["calibration_logs"]:
        print("[IK_CAL] Applied empirical IK servo calibration:")
        for line in result["calibration_logs"]:
            print(f"  {line}")
    if use_z_mode_correction and result["z_mode_logs"]:
        print("[IK_ZMODE] Applied z_mode correction:")
        for line in result["z_mode_logs"]:
            print(f"  {line}")

    print("reference MOVE_SAFE:", build_move_command([result["reference_servos"][ch] for ch in ALL_CHANNELS]))
    print("ik MOVE_SAFE:       ", build_move_command([result["ik_move_safe"][ch] for ch in ALL_CHANNELS]))

    print("error CH1..CH6:")
    for channel in ALL_CHANNELS:
        role = ""
        if channel in ACTIVE_IK_CHANNELS:
            role = "active_ik"
        elif channel == "ch4":
            role = "fixed_wrist_rotate"
        elif channel == "ch6":
            role = "gripper_action"
        print(
            f"  {channel.upper()}: ik={result['ik_move_safe'][channel]:>3} "
            f"ref={result['reference_servos'][channel]:>3} "
            f"err={result['errors'][channel]:>+4}  {role}"
        )

    print("active IK error summary:")
    print(f"  mean_abs_error(CH1/CH2/CH3/CH5): {result['mean_abs_error']:.2f} deg")
    print(f"  max_abs_error: {result['max_abs_error']:.2f} deg")
    print(f"  worst_channel: {result['worst_channel'].upper()}")
    print(f"  grade: {result['grade']}")

    if result["failures"]:
        print("[SAFETY] Servo limit issues:")
        for failure in result["failures"]:
            print(f"  {failure}")

    if result["clamp_notes"]:
        print("[SAFETY] Clamp notes:")
        for note in result["clamp_notes"]:
            print(f"  {note}")

    print("[IK_REF] Notes:")
    print("  CH4 is shown as fixed wrist rotate.")
    print("  CH6 is shown as gripper action only and is not part of the active IK summary.")
    print()


def print_affine_fit(fitted_channels: dict[str, dict[str, float]]) -> None:
    print("============================================================")
    print("[IK_CAL] Proposed affine IK servo calibration fit")
    print("============================================================")
    for channel in ACTIVE_IK_CHANNELS:
        entry = fitted_channels[channel]
        print(
            f"{channel.upper()}: calibrated = {entry['scale']:.9f} * raw + {entry['offset_deg']:.9f}"
        )
    print()


def print_z_mode_fit(z_mode_corrections: dict[str, Any]) -> None:
    print("============================================================")
    print("[IK_ZMODE] Proposed z_mode servo corrections")
    print("============================================================")
    modes = z_mode_corrections.get("modes", {})
    for z_mode, entry in modes.items():
        print(f"{z_mode}:")
        for channel in ACTIVE_IK_CHANNELS:
            key = f"{channel}_offset_deg"
            print(f"  {key}: {float(entry.get(key, 0.0)):+.6f}")
    print()


def ordered_supported_modes(supported_modes: list[str]) -> list[str]:
    ordered = [mode for mode in PREFERRED_MODE_ORDER if mode in supported_modes]
    ordered.extend(mode for mode in supported_modes if mode not in ordered)
    return ordered


def build_mode_args(args: argparse.Namespace, tcp_offset_mode: str) -> argparse.Namespace:
    clone = argparse.Namespace(**vars(args))
    clone.tcp_offset_mode = tcp_offset_mode
    return clone


def print_mode_comparison_table(mode_results: dict[str, list[dict[str, Any]]]) -> None:
    print("============================================================")
    print("[IK_REF] TCP offset mode comparison")
    print("============================================================")
    for mode, rows in mode_results.items():
        print(f"mode: {mode}")
        for row in rows:
            if "ik_unreachable_reason" in row:
                print(
                    f"  {row['sample_name']:<26} "
                    f"grade=UNREACHABLE  reason={row['ik_unreachable_reason']}"
                )
                continue
            print(
                f"  {row['sample_name']:<26} "
                f"ik={build_move_command([row['ik_move_safe'][ch] for ch in ALL_CHANNELS])}  "
                f"ref={build_move_command([row['reference_servos'][ch] for ch in ALL_CHANNELS])}  "
                f"mean_abs={row['mean_abs_error']:.2f}  "
                f"max_abs={row['max_abs_error']:.2f}  "
                f"worst={row['worst_channel'].upper()}  "
                f"grade={row['grade']}"
            )
        print()


def summarize_mode_results(mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    reachable = [row for row in rows if "ik_unreachable_reason" not in row]
    unreachable_count = len(rows) - len(reachable)
    grade_counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for row in reachable:
        grade_counts[row["grade"]] += 1

    if reachable:
        avg_mean_abs = statistics.mean(row["mean_abs_error"] for row in reachable)
        avg_max_abs = statistics.mean(row["max_abs_error"] for row in reachable)
    else:
        avg_mean_abs = None
        avg_max_abs = None

    return {
        "mode": mode,
        "reachable_count": len(reachable),
        "unreachable_count": unreachable_count,
        "avg_mean_abs_error": avg_mean_abs,
        "avg_max_abs_error": avg_max_abs,
        "pass_count": grade_counts["PASS"],
        "warn_count": grade_counts["WARN"],
        "fail_count": grade_counts["FAIL"],
    }


def print_mode_ranking(summaries: list[dict[str, Any]]) -> None:
    print("============================================================")
    print("[IK_REF] TCP offset mode ranking")
    print("============================================================")
    for summary in summaries:
        if summary["avg_mean_abs_error"] is None:
            avg_mean = "n/a"
            avg_max = "n/a"
        else:
            avg_mean = f"{summary['avg_mean_abs_error']:.2f}"
            avg_max = f"{summary['avg_max_abs_error']:.2f}"
        print(
            f"{summary['mode']:<14} "
            f"avg_mean_abs={avg_mean:<6} "
            f"avg_max_abs={avg_max:<6} "
            f"PASS={summary['pass_count']} WARN={summary['warn_count']} FAIL={summary['fail_count']} "
            f"UNREACHABLE={summary['unreachable_count']}"
        )
    print()


def recommended_mode_summary(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [summary for summary in summaries if summary["avg_mean_abs_error"] is not None]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda summary: (
            summary["avg_mean_abs_error"],
            summary["avg_max_abs_error"],
            summary["fail_count"],
            summary["unreachable_count"],
        ),
    )


def main() -> int:
    args = parse_args()
    if args.write_calibration and not args.fit_affine:
        print("[IK_CAL][ERROR] --write-calibration requires --fit-affine", file=sys.stderr)
        return 2
    if args.write_z_mode_corrections and not args.fit_z_mode_corrections:
        print("[IK_ZMODE][ERROR] --write-z-mode-corrections requires --fit-z-mode-corrections", file=sys.stderr)
        return 2
    if args.fit_z_mode_corrections and not args.use_ik_servo_calibration:
        print("[IK_ZMODE][ERROR] --fit-z-mode-corrections requires --use-ik-servo-calibration", file=sys.stderr)
        return 2

    kin_cfg = load_required(args.kinematics_config, "kinematics config")
    servo_cfg = load_required(args.servo_config, "servo config")
    pose_cfg = load_required(args.pose_config, "pose config")
    transform_cfg = load_required(args.transform_config, "transform config")
    samples = load_reference_samples(args.reference_config)
    ik_servo_cal_cfg = None
    if args.use_ik_servo_calibration:
        ik_servo_cal_cfg = load_ik_servo_calibration(args.ik_servo_calibration_config)

    supported_modes = kin_cfg.get("ik", {}).get("tcp_offset_modes_supported", ["none", "planar", "vertical_down", "mixed"])
    if args.tcp_offset_mode not in supported_modes:
        print(
            f"[CONFIG][ERROR] Unsupported tcp offset mode {args.tcp_offset_mode!r}. Supported: {supported_modes}",
            file=sys.stderr,
        )
        return 2

    try:
        selected = select_samples(args, samples)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print("[SAFETY] Dry-run only. No serial, no camera, no YOLO, no robot motion.")
    print(f"[CONFIG] reference_config: {args.reference_config}")
    print(f"[CONFIG] tcp_offset_mode: {args.tcp_offset_mode}")
    if args.use_ik_servo_calibration:
        print(f"[CONFIG] ik_servo_calibration_config: {args.ik_servo_calibration_config}")
    if args.use_z_mode_correction:
        print("[CONFIG] z_mode correction: ENABLED (explicit CLI opt-in)")
    print("[SAFETY] Diagnostic only. Do not live-test pre_pick until hover accuracy is acceptable.")
    print()

    if args.compare_tcp_offset_modes:
        modes = ordered_supported_modes(supported_modes)
        print(f"[CONFIG] supported tcp_offset_modes: {', '.join(modes)}")
        print()
        overall_exit = 0
        mode_results: dict[str, list[dict[str, Any]]] = {}
        for mode in modes:
            mode_args = build_mode_args(args, mode)
            rows: list[dict[str, Any]] = []
            for sample_name, sample in selected:
                try:
                    result = evaluate_sample(
                        sample_name,
                        sample,
                        mode_args,
                        kin_cfg,
                        servo_cfg,
                        pose_cfg,
                        transform_cfg,
                        ik_servo_cal_cfg=ik_servo_cal_cfg,
                    )
                except Exception as exc:
                    result = {
                        "sample_name": sample_name,
                        "ik_unreachable_reason": f"ERROR: {exc}",
                    }
                rows.append(result)
                if "ik_unreachable_reason" in result or result.get("grade") == "FAIL":
                    overall_exit = 2
            mode_results[mode] = rows

        print_mode_comparison_table(mode_results)
        summaries = [summarize_mode_results(mode, rows) for mode, rows in mode_results.items()]
        summaries.sort(
            key=lambda summary: (
                float("inf") if summary["avg_mean_abs_error"] is None else summary["avg_mean_abs_error"],
                float("inf") if summary["avg_max_abs_error"] is None else summary["avg_max_abs_error"],
                summary["fail_count"],
                summary["unreachable_count"],
            )
        )
        print_mode_ranking(summaries)
        best = recommended_mode_summary(summaries)
        if best is None:
            print("[IK_REF] recommended next dry-run mode: none (no reachable candidate modes found)")
        else:
            print(
                "[IK_REF] recommended next dry-run mode: "
                f"{best['mode']} "
                f"(avg_mean_abs={best['avg_mean_abs_error']:.2f}, "
                f"avg_max_abs={best['avg_max_abs_error']:.2f})"
            )
        return overall_exit

    overall_exit = 0
    all_results: list[dict[str, Any]] = []
    for sample_name, sample in selected:
        try:
            result = evaluate_sample(
                sample_name,
                sample,
                args,
                kin_cfg,
                servo_cfg,
                pose_cfg,
                transform_cfg,
                ik_servo_cal_cfg=ik_servo_cal_cfg,
            )
        except Exception as exc:
            print("============================================================")
            print(f"[IK_REF] sample: {sample_name}")
            print("============================================================")
            print(f"[IK_REF][ERROR] {exc}")
            print()
            overall_exit = 2
            continue

        all_results.append(result)
        print_result(
            result,
            calibrated=args.use_ik_servo_calibration,
            use_z_mode_correction=args.use_z_mode_correction,
        )
        if "ik_unreachable_reason" in result or result.get("grade") == "FAIL":
            overall_exit = 2

    if args.fit_affine:
        fit_rows = [row for row in all_results if "ik_move_safe" in row]
        if not fit_rows:
            print("[IK_CAL][ERROR] No successful IK rows were available to fit affine calibration.", file=sys.stderr)
            return 2
        fitted_channels = fit_affine_channels_from_samples(fit_rows)
        print_affine_fit(fitted_channels)

        if args.write_calibration:
            existing = load_yaml_if_exists(args.ik_servo_calibration_config)
            existing_cfg = existing.get("ik_servo_calibration", {}) if isinstance(existing, dict) else {}
            document = build_ik_servo_calibration_document(
                fitted_channels,
                [row["sample_name"] for row in fit_rows],
                z_mode_corrections=existing_cfg.get("z_mode_corrections"),
            )
            write_ik_servo_calibration(args.ik_servo_calibration_config, document)
            print(f"[IK_CAL] Wrote calibration file: {args.ik_servo_calibration_config}")

    if args.fit_z_mode_corrections:
        fit_rows = [row for row in all_results if "ik_move_safe" in row]
        if not fit_rows:
            print("[IK_ZMODE][ERROR] No successful IK rows were available to fit z_mode corrections.", file=sys.stderr)
            return 2
        z_mode_corrections = fit_z_mode_corrections_from_samples(fit_rows)
        print_z_mode_fit(z_mode_corrections)
        if args.write_z_mode_corrections:
            existing = load_yaml_if_exists(args.ik_servo_calibration_config)
            existing_cfg = existing.get("ik_servo_calibration", {}) if isinstance(existing, dict) else {}
            fitted_channels = existing_cfg.get("channels")
            if not isinstance(fitted_channels, dict):
                print(
                    "[IK_ZMODE][ERROR] Cannot write z_mode corrections because affine channel calibration is missing in the config.",
                    file=sys.stderr,
                )
                return 2
            document = {
                "ik_servo_calibration": dict(existing_cfg),
            }
            document["ik_servo_calibration"]["updated_at"] = existing_cfg.get("updated_at") or document["ik_servo_calibration"].get("updated_at")
            document["ik_servo_calibration"]["z_mode_corrections"] = z_mode_corrections
            write_ik_servo_calibration(args.ik_servo_calibration_config, document)
            print(f"[IK_ZMODE] Wrote z_mode corrections to: {args.ik_servo_calibration_config}")

    return overall_exit


if __name__ == "__main__":
    sys.exit(main())
