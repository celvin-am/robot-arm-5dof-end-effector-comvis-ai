#!/usr/bin/env python3
"""
test_ik_workspace_grid.py

Guarded workspace-grid dry-run and single-point live checker for IK
safe_hover/pre_pick/lift validation. This tool never closes the gripper by
default and only sends one point and one z_mode at a time in live mode.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from test_esp32_manual_move import open_serial, send_command
from test_ik_dry_run import (
    DEFAULT_KINEMATICS_CONFIG,
    DEFAULT_SERVO_CONFIG,
    DEFAULT_TRANSFORM_CONFIG,
    load_required,
)
from test_ik_to_move_safe import DEFAULT_SERIAL_CONFIG, require_line
from yolo_ik_sequence_utils import (
    DEFAULT_SEMI_AUTO_CONFIG,
    DEFAULT_WORKSPACE_CORRECTION_MAP,
    build_ik_step,
    format_sequence_preview,
    get_workspace_grid_points,
    load_optional_ik_calibration,
    load_semi_auto_pick_place_config,
    load_workspace_correction_map,
    resolve_runtime_defaults,
)


DEFAULT_BAUD = 115200
Z_MODES = ("safe_hover", "pre_pick", "lift")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded workspace-grid IK checker. Dry-run by default.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--yes-i-understand-hardware-risk", action="store_true")
    parser.add_argument("--confirm-each", action="store_true")
    parser.add_argument("--point", default="all", help="Grid point name or 'all'")
    parser.add_argument("--z-mode", choices=[*Z_MODES, "all"], default="all")
    parser.add_argument("--mark-pass", action="store_true")
    parser.add_argument("--mark-fail", action="store_true")
    parser.add_argument("--config", default=DEFAULT_SEMI_AUTO_CONFIG)
    parser.add_argument("--workspace-correction-map-config", default=DEFAULT_WORKSPACE_CORRECTION_MAP)
    parser.add_argument("--tcp-offset-mode")
    parser.add_argument("--solution", choices=["elbow_up", "elbow_down", "best"], default="elbow_up")
    parser.add_argument("--use-ik-servo-calibration", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-z-mode-correction", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ik-servo-calibration-config", default="ros2_ws/src/robot_arm_5dof/config/ik_servo_calibration.yaml")
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
    parser.add_argument("--serial-config", default=DEFAULT_SERIAL_CONFIG)
    return parser.parse_args()


def load_yaml_root(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"[CONFIG][ERROR] YAML root must be a mapping: {path}")
    return data


def save_yaml_root(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def resolve_open_gripper_deg(semi_auto_cfg: dict[str, Any], servo_cfg: dict[str, Any]) -> int:
    gripper_cfg = semi_auto_cfg.get("gripper", {})
    for key in ("donut", "cake"):
        entry = gripper_cfg.get(key, {})
        if isinstance(entry, dict) and entry.get("open_angle_deg") is not None:
            return int(round(float(entry["open_angle_deg"])))
    calibration = servo_cfg.get("gripper_calibration", {})
    if calibration.get("open_deg") is not None:
        return int(round(float(calibration["open_deg"])))
    ch6 = servo_cfg.get("servos", {}).get("ch6", {})
    return int(round(float(ch6.get("open_angle_deg", 50))))


def resolve_region_profile(workspace_cfg: dict[str, Any], point_name: str) -> dict[str, Any]:
    defaults = workspace_cfg.get("defaults", {})
    default_grasp = defaults.get("grasp_offset", {})
    default_z = defaults.get("z_heights_m", {})
    regions = workspace_cfg.get("regions", {})
    region = regions.get(point_name)
    if not isinstance(region, dict):
        raise ValueError(f"[WORKSPACE][ERROR] Region {point_name!r} not found in workspace correction map.")
    return {
        "region_name": str(point_name),
        "status": str(region.get("status", "untested")),
        "distance_cm": 0.0,
        "validated": str(region.get("status", "untested")).strip().lower() == "live_validated",
        "grasp_offset_x_cm": float(region.get("board_x_offset_cm", default_grasp.get("board_x_offset_cm", 0.0))),
        "grasp_offset_y_cm": float(region.get("board_y_offset_cm", default_grasp.get("board_y_offset_cm", 0.0))),
        "safe_hover_z_m": float(region.get("safe_hover_z_m", default_z.get("safe_hover", 0.055))),
        "pre_pick_z_m": float(region.get("pre_pick_z_m", default_z.get("pre_pick", 0.015))),
        "lift_z_m": float(region.get("lift_z_m", default_z.get("lift", 0.125))),
    }


def build_steps_for_point(
    point_name: str,
    board_x_cm: float,
    board_y_cm: float,
    profile: dict[str, Any],
    semi_auto_cfg: dict[str, Any],
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    transform_cfg: dict[str, Any],
    ik_servo_cal_cfg: dict[str, Any] | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    open_deg = resolve_open_gripper_deg(semi_auto_cfg, servo_cfg)
    z_lookup = {
        "safe_hover": float(profile["safe_hover_z_m"]),
        "pre_pick": float(profile["pre_pick_z_m"]),
        "lift": float(profile["lift_z_m"]),
    }
    selected_modes = Z_MODES if args.z_mode == "all" else (args.z_mode,)
    steps = []
    for z_mode in selected_modes:
        step_name = f"{point_name.upper()}_{z_mode.upper()}"
        try:
            steps.append(
                build_ik_step(
                    step_name,
                    board_x_cm,
                    board_y_cm,
                    z_lookup[z_mode],
                    z_mode,
                    open_deg,
                    kin_cfg,
                    servo_cfg,
                    transform_cfg,
                    ik_servo_cal_cfg,
                    args,
                )
            )
        except ValueError as exc:
            steps.append(
                {
                    "type": "ERROR",
                    "name": step_name,
                    "source": "ik",
                    "board_x_cm": board_x_cm,
                    "board_y_cm": board_y_cm,
                    "z_m": z_lookup[z_mode],
                    "z_mode": z_mode,
                    "error": str(exc),
                }
            )
    return steps


def update_workspace_result(config_path: str, point_name: str, z_mode: str, result: str) -> None:
    root = load_yaml_root(config_path)
    workspace = root.get("workspace_correction_map", {})
    if not isinstance(workspace, dict):
        raise RuntimeError("[CONFIG][ERROR] workspace_correction_map mapping missing")
    region = workspace.get("regions", {}).get(point_name)
    if not isinstance(region, dict):
        raise RuntimeError(f"[CONFIG][ERROR] Region {point_name!r} not found in workspace correction map.")
    manual_results = region.setdefault("manual_results", {})
    manual_results[z_mode] = {
        "result": result,
        "updated_at": utc_now_iso(),
        "source": "tools/test_ik_workspace_grid.py",
    }
    pass_count = sum(1 for mode in Z_MODES if manual_results.get(mode, {}).get("result") == "pass")
    fail_count = sum(1 for mode in Z_MODES if manual_results.get(mode, {}).get("result") == "fail")
    if pass_count == len(Z_MODES):
        region["status"] = "live_validated"
    elif fail_count > 0:
        region["status"] = "needs_correction"
    elif pass_count > 0:
        region["status"] = "partially_validated"
    save_yaml_root(config_path, root)


def print_region_header(point_name: str, board_x_cm: float, board_y_cm: float, profile: dict[str, Any]) -> None:
    print(
        f"[WORKSPACE] point={point_name} board=({board_x_cm:.2f}, {board_y_cm:.2f}) "
        f"status={profile['status']}"
    )
    print(
        f"[WORKSPACE] offset=({profile['grasp_offset_x_cm']:.2f}, {profile['grasp_offset_y_cm']:.2f}) cm"
    )
    print(
        "[WORKSPACE] z policy="
        f"safe_hover={profile['safe_hover_z_m']:.3f} "
        f"pre_pick={profile['pre_pick_z_m']:.3f} "
        f"lift={profile['lift_z_m']:.3f} m"
    )


def send_single_step(ser, step: dict[str, Any]) -> None:
    lines = send_command(ser, step["command"])
    print(f"COMMAND {step['command']}")
    for line in lines:
        print(f"  {line}")
    require_line(lines, "ACK MOVE_SAFE", step["name"])
    require_line(lines, "DONE MOVE_SAFE", step["name"])


def main() -> int:
    args = parse_args()

    if args.dry_run and args.send:
        print("[SAFETY][ERROR] Use either --dry-run or --send, not both.", file=sys.stderr)
        return 2
    if args.mark_pass and args.mark_fail:
        print("[CONFIG][ERROR] Use either --mark-pass or --mark-fail, not both.", file=sys.stderr)
        return 2
    if args.send and args.port is None:
        print("[SERIAL][ERROR] --port is required when --send is used.", file=sys.stderr)
        return 2
    if args.send and not args.yes_i_understand_hardware_risk:
        print("[SAFETY][ERROR] Live workspace testing requires --yes-i-understand-hardware-risk.", file=sys.stderr)
        return 2
    if args.send and not args.confirm_each:
        print("[SAFETY][ERROR] Live workspace testing requires --confirm-each.", file=sys.stderr)
        return 2
    if args.send and (args.point == "all" or args.z_mode == "all"):
        print("[SAFETY][ERROR] Live mode only supports one point and one z_mode at a time.", file=sys.stderr)
        return 2
    if (args.mark_pass or args.mark_fail) and (args.point == "all" or args.z_mode == "all"):
        print("[CONFIG][ERROR] Marking pass/fail requires a single --point and single --z-mode.", file=sys.stderr)
        return 2

    semi_auto_cfg = load_semi_auto_pick_place_config(args.config)
    resolve_runtime_defaults(args, semi_auto_cfg)
    if args.use_ik_servo_calibration is None:
        args.use_ik_servo_calibration = bool(semi_auto_cfg.get("ik", {}).get("use_ik_servo_calibration", True))
    if args.use_z_mode_correction is None:
        args.use_z_mode_correction = bool(semi_auto_cfg.get("ik", {}).get("use_z_mode_correction", True))

    kin_cfg = load_required(args.kinematics_config, "kinematics config")
    servo_cfg = load_required(args.servo_config, "servo config")
    transform_cfg = load_required(args.transform_config, "transform config")
    _ = load_required(args.serial_config, "serial config")
    workspace_cfg = load_workspace_correction_map(args.workspace_correction_map_config)
    ik_servo_cal_cfg = load_optional_ik_calibration(args)

    supported_modes = kin_cfg.get("ik", {}).get("tcp_offset_modes_supported", ["none", "planar", "vertical_down", "mixed"])
    if args.tcp_offset_mode not in supported_modes:
        print(f"[CONFIG][ERROR] Unsupported tcp_offset_mode {args.tcp_offset_mode!r}. Supported: {supported_modes}", file=sys.stderr)
        return 2

    grid_points = get_workspace_grid_points(workspace_cfg)
    selected_point_names = list(grid_points.keys()) if args.point == "all" else [args.point]
    missing = [name for name in selected_point_names if name not in grid_points]
    if missing:
        print(f"[CONFIG][ERROR] Unknown workspace point(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    print(f"[CONFIG] workspace correction map: {args.workspace_correction_map_config}")
    print(f"[IK] tcp_offset_mode: {args.tcp_offset_mode}")
    print(f"[IK] use_ik_servo_calibration: {args.use_ik_servo_calibration}")
    print(f"[IK] use_z_mode_correction: {args.use_z_mode_correction}")

    all_steps: list[dict[str, Any]] = []
    selected_profile: dict[str, Any] | None = None
    selected_point_name: str | None = None
    for point_name in selected_point_names:
        board_x_cm, board_y_cm = grid_points[point_name]
        profile = resolve_region_profile(workspace_cfg, point_name)
        print_region_header(point_name, board_x_cm, board_y_cm, profile)
        steps = build_steps_for_point(
            point_name,
            board_x_cm,
            board_y_cm,
            profile,
            semi_auto_cfg,
            kin_cfg,
            servo_cfg,
            transform_cfg,
            ik_servo_cal_cfg,
            args,
        )
        print(format_sequence_preview(steps))
        print("")
        all_steps.extend(steps)
        selected_profile = profile
        selected_point_name = point_name

    if args.dry_run or not args.send:
        if selected_point_name is not None and args.z_mode != "all" and (args.mark_pass or args.mark_fail):
            update_workspace_result(
                args.workspace_correction_map_config,
                selected_point_name,
                args.z_mode,
                "pass" if args.mark_pass else "fail",
            )
            print(
                f"[WORKSPACE] Saved manual result for {selected_point_name}/{args.z_mode}: "
                f"{'pass' if args.mark_pass else 'fail'}"
            )
        return 0

    assert selected_point_name is not None
    assert selected_profile is not None
    step = all_steps[0]
    if step["type"] != "MOVE_SAFE":
        print(f"[IK][ERROR] Selected point/z_mode is not sendable: {step['error']}", file=sys.stderr)
        return 2
    print(f"[SERIAL] Final command preview: {step['command']}")
    print("[SAFETY] Live hardware MOVE_SAFE requested.")
    if not args.confirm_each:
        return 2
    if input(f"Type START to send {selected_point_name}/{args.z_mode}: \n> ").strip() != "START":
        print("START cancelled.")
        return 0

    try:
        ser = open_serial(args.port, args.baud, args.timeout)
    except Exception as exc:  # pragma: no cover - hardware-dependent path
        print(f"[SERIAL][ERROR] Failed to open serial port {args.port}: {exc}", file=sys.stderr)
        return 2

    try:
        send_single_step(ser, step)
    finally:
        ser.close()

    if args.mark_pass or args.mark_fail:
        update_workspace_result(
            args.workspace_correction_map_config,
            selected_point_name,
            args.z_mode,
            "pass" if args.mark_pass else "fail",
        )
        print(
            f"[WORKSPACE] Saved manual result for {selected_point_name}/{args.z_mode}: "
            f"{'pass' if args.mark_pass else 'fail'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
