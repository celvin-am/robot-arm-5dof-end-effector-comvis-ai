#!/usr/bin/env python3
"""
yolo_ik_sequence_utils.py

Shared helpers for guarded YOLO -> board -> robot -> IK semi-auto sequence
tools. This module does not open serial or camera by itself.
"""

from __future__ import annotations

import math
from typing import Any

from ik_servo_calibration_utils import (
    apply_ik_servo_calibration,
    apply_z_mode_correction,
    load_ik_servo_calibration,
)
from test_ik_dry_run import (
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


DEFAULT_SEMI_AUTO_CONFIG = "ros2_ws/src/robot_arm_5dof/config/semi_auto_pick_place_config.yaml"
DEFAULT_TAUGHT_POSE_CONFIG = "ros2_ws/src/robot_arm_5dof/config/taught_pick_place_poses.yaml"
DEFAULT_WORKSPACE_CORRECTION_MAP = "ros2_ws/src/robot_arm_5dof/config/workspace_correction_map.yaml"
DEFAULT_ARUCO_CONFIG = "ros2_ws/src/robot_arm_5dof/config/aruco_config.yaml"
GROUPS = ("CAKE", "DONUT")
DEFAULT_WORKSPACE_GRID = {
    "top_left": (7.0, 5.0),
    "top_center": (14.25, 5.0),
    "top_right": (21.0, 5.0),
    "mid_left": (7.0, 9.0),
    "mid_center": (14.25, 9.0),
    "mid_right": (21.0, 9.0),
    "low_left": (7.0, 13.0),
    "low_center": (14.25, 13.0),
    "low_right": (21.0, 13.0),
}


def normalize_group(value: str) -> str:
    upper = str(value).strip().upper()
    if upper in {"ANY", "CAKE", "DONUT"}:
        return upper
    raise ValueError("target group must be ANY, CAKE, or DONUT")


def raw_class_to_group(raw_name: str) -> str | None:
    text = str(raw_name).strip().lower()
    if text in {"cake", "cake 2"}:
        return "CAKE"
    if text in {"donut", "donut 1"}:
        return "DONUT"
    return None


def load_semi_auto_pick_place_config(path: str) -> dict[str, Any]:
    root = load_required(path, "semi-auto pick-place config")
    cfg = root.get("semi_auto_pick_place", {})
    if not isinstance(cfg, dict):
        raise RuntimeError("[CONFIG][ERROR] semi_auto_pick_place mapping missing")
    return cfg


def load_workspace_correction_map(path: str) -> dict[str, Any]:
    root = load_required(path, "workspace correction map config")
    cfg = root.get("workspace_correction_map", {})
    if not isinstance(cfg, dict):
        raise RuntimeError("[CONFIG][ERROR] workspace_correction_map mapping missing")
    return cfg


def get_board_limits(board_cfg: dict[str, Any]) -> tuple[float, float]:
    board = board_cfg.get("board", {})
    return float(board.get("width_cm", 28.5)), float(board.get("height_cm", 18.0))


def get_workspace_grid_points(workspace_cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    grid_cfg = workspace_cfg.get("grid_points", {})
    if not isinstance(grid_cfg, dict) or not grid_cfg:
        return dict(DEFAULT_WORKSPACE_GRID)
    points: dict[str, tuple[float, float]] = {}
    for name, entry in grid_cfg.items():
        if not isinstance(entry, dict):
            continue
        points[str(name)] = (float(entry["board_x_cm"]), float(entry["board_y_cm"]))
    return points or dict(DEFAULT_WORKSPACE_GRID)


def validate_board_target(
    board_x_cm: float,
    board_y_cm: float,
    board_width_cm: float,
    board_height_cm: float,
) -> tuple[bool, str]:
    ok = 0.0 <= board_x_cm <= board_width_cm and 0.0 <= board_y_cm <= board_height_cm
    if ok:
        return True, ""
    return False, (
        f"board target ({board_x_cm:.2f}, {board_y_cm:.2f}) cm outside "
        f"[0,{board_width_cm:.2f}] x [0,{board_height_cm:.2f}] cm"
    )


def get_validated_pick_region(semi_auto_cfg: dict[str, Any]) -> dict[str, Any]:
    region = semi_auto_cfg.get("validated_pick_region", {})
    if not isinstance(region, dict) or not region:
        return {
            "board_x_min_cm": 7.0,
            "board_x_max_cm": 21.0,
            "board_y_min_cm": 5.0,
            "board_y_max_cm": 10.5,
            "status": "center_and_mid_region_validated_only",
            "note": "Lower board row is not live validated for pre-pick. Reject by default.",
        }
    return region


def validate_pick_region(
    adjusted_board_x_cm: float,
    adjusted_board_y_cm: float,
    semi_auto_cfg: dict[str, Any],
) -> tuple[bool, str]:
    region = get_validated_pick_region(semi_auto_cfg)
    x_min = float(region["board_x_min_cm"])
    x_max = float(region["board_x_max_cm"])
    y_min = float(region["board_y_min_cm"])
    y_max = float(region["board_y_max_cm"])
    ok = x_min <= adjusted_board_x_cm <= x_max and y_min <= adjusted_board_y_cm <= y_max
    if ok:
        return True, ""
    return False, (
        f"adjusted target ({adjusted_board_x_cm:.2f}, {adjusted_board_y_cm:.2f}) cm outside validated pick region "
        f"x=[{x_min:.2f},{x_max:.2f}] y=[{y_min:.2f},{y_max:.2f}] cm"
    )


def workspace_region_is_validated(status: str) -> bool:
    normalized = str(status).strip().lower()
    return normalized in {"live_validated", "validated", "user_confirmed", "live_validated_center_area"}


def load_named_pose(name: str, taught_cfg: dict[str, Any], pose_cfg: dict[str, Any]) -> dict[str, int]:
    taught_pose = taught_cfg.get("poses", {}).get(name)
    pose = taught_pose if isinstance(taught_pose, dict) else pose_cfg.get("poses", {}).get(name)
    if not isinstance(pose, dict):
        raise ValueError(f"Required pose {name} not found in taught or base pose config")
    out: dict[str, int] = {}
    for ch in ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6"):
        value = pose.get(ch)
        if value is None:
            raise ValueError(f"Required pose {name} has null {ch}")
        out[ch] = int(round(float(value)))
    return out


def clamp_move_angles(move_angles: list[int], servo_cfg: dict[str, Any]) -> tuple[list[int], list[str], list[str]]:
    clamped, clamp_notes = clamp_move_safe_angles(move_angles, servo_cfg)
    failures = validate_move_safe_angles(clamped, servo_cfg)
    return clamped, clamp_notes, failures


def resolve_runtime_defaults(args: Any, semi_auto_cfg: dict[str, Any]) -> None:
    if getattr(args, "cam", None) is None:
        args.cam = int(semi_auto_cfg.get("camera_id", 2))
    if getattr(args, "tcp_offset_mode", None) in (None, ""):
        args.tcp_offset_mode = str(semi_auto_cfg.get("ik", {}).get("tcp_offset_mode", "none"))


def group_key(group: str) -> str:
    upper = normalize_group(group)
    if upper == "ANY":
        raise ValueError("ANY is not valid where a concrete class group is required")
    return upper.lower()


def get_group_offsets(semi_auto_cfg: dict[str, Any], group: str) -> tuple[float, float]:
    entry = semi_auto_cfg.get("grasp_offsets", {}).get(group_key(group), {})
    if not isinstance(entry, dict):
        raise ValueError(f"Missing grasp offset config for {group}")
    return float(entry["board_x_offset_cm"]), float(entry["board_y_offset_cm"])


def default_workspace_profile(semi_auto_cfg: dict[str, Any], group: str) -> dict[str, Any]:
    safe_hover_z, pre_pick_z, lift_z = get_validated_heights(semi_auto_cfg)
    grasp_dx, grasp_dy = get_group_offsets(semi_auto_cfg, group)
    return {
        "region_name": "global_default",
        "status": "global_default",
        "distance_cm": 0.0,
        "selection_threshold_cm": None,
        "validated": True,
        "grasp_offset_x_cm": grasp_dx,
        "grasp_offset_y_cm": grasp_dy,
        "safe_hover_z_m": safe_hover_z,
        "pre_pick_z_m": pre_pick_z,
        "lift_z_m": lift_z,
    }


def select_workspace_profile(
    workspace_cfg: dict[str, Any],
    board_x_cm: float,
    board_y_cm: float,
    *,
    allow_untested_region: bool = False,
) -> dict[str, Any]:
    defaults = workspace_cfg.get("defaults", {})
    default_grasp = defaults.get("grasp_offset", {})
    default_z = defaults.get("z_heights_m", {})
    regions = workspace_cfg.get("regions", {})
    selection_cfg = workspace_cfg.get("selection", {})
    max_distance_cm = float(selection_cfg.get("nearest_region_max_distance_cm", 5.0))

    candidates: list[dict[str, Any]] = []
    for name, entry in regions.items():
        if not isinstance(entry, dict):
            continue
        region_x = float(entry.get("board_x_cm"))
        region_y = float(entry.get("board_y_cm"))
        distance_cm = math.hypot(board_x_cm - region_x, board_y_cm - region_y)
        status = str(entry.get("status", "untested"))
        candidates.append(
            {
                "region_name": str(name),
                "status": status,
                "validated": workspace_region_is_validated(status),
                "distance_cm": distance_cm,
                "selection_threshold_cm": max_distance_cm,
                "board_x_cm": region_x,
                "board_y_cm": region_y,
                "grasp_offset_x_cm": float(entry.get("board_x_offset_cm", default_grasp.get("board_x_offset_cm", 0.0))),
                "grasp_offset_y_cm": float(entry.get("board_y_offset_cm", default_grasp.get("board_y_offset_cm", 0.0))),
                "safe_hover_z_m": float(entry.get("safe_hover_z_m", default_z.get("safe_hover", 0.055))),
                "pre_pick_z_m": float(entry.get("pre_pick_z_m", default_z.get("pre_pick", 0.015))),
                "lift_z_m": float(entry.get("lift_z_m", default_z.get("lift", 0.125))),
            }
        )

    if not candidates:
        raise ValueError("[WORKSPACE][ERROR] No workspace regions defined in correction map.")

    validated = [item for item in candidates if item["validated"]]
    if validated:
        nearest_validated = min(validated, key=lambda item: item["distance_cm"])
        if nearest_validated["distance_cm"] <= max_distance_cm:
            return nearest_validated
        if not allow_untested_region:
            raise ValueError(
                "[WORKSPACE][ERROR] Nearest validated region "
                f"{nearest_validated['region_name']} is {nearest_validated['distance_cm']:.2f} cm away, "
                f"outside threshold {max_distance_cm:.2f} cm."
            )
    elif not allow_untested_region:
        raise ValueError("[WORKSPACE][ERROR] No validated workspace region is available for this target.")

    nearest_any = min(candidates, key=lambda item: item["distance_cm"])
    nearest_any["validated"] = workspace_region_is_validated(nearest_any["status"])
    return nearest_any


def evaluate_aruco_board_error(
    frame,
    H,
    aruco_config_path: str,
    *,
    detector_profile: str = "relaxed",
    preprocess: str = "clahe",
    adaptive_block_size: int = 31,
    adaptive_c: int = 7,
) -> dict[str, Any] | None:
    from aruco_utils import (
        add_board_coordinates,
        build_detector,
        compute_board_error_stats,
        detect_markers,
        get_aruco_dict_name,
        get_predefined_dictionary,
        load_aruco_config,
        marker_records,
    )

    aruco_root = load_aruco_config(aruco_config_path)
    dict_name = get_aruco_dict_name(aruco_root)
    dictionary = get_predefined_dictionary(dict_name)
    detector = build_detector(dictionary, profile=detector_profile)
    corners, ids, _rejected, _processed = detect_markers(
        frame,
        dictionary,
        detector,
        profile=detector_profile,
        preprocess=preprocess,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
    )
    records = add_board_coordinates(marker_records(corners, ids, aruco_root), H)
    return compute_board_error_stats(records)


def get_group_gripper_angles(semi_auto_cfg: dict[str, Any], servo_cfg: dict[str, Any], group: str) -> tuple[int, int]:
    entry = semi_auto_cfg.get("gripper", {}).get(group_key(group), {})
    ch6_cfg = servo_cfg.get("servos", {}).get("ch6", {})
    open_deg = entry.get("open_angle_deg", servo_cfg.get("gripper_calibration", {}).get("open_deg", ch6_cfg.get("open_angle_deg", 50)))
    close_deg = entry.get("close_angle_deg", servo_cfg.get("gripper_calibration", {}).get("close_full_deg", ch6_cfg.get("close_angle_deg", 20)))
    return int(round(float(open_deg))), int(round(float(close_deg)))


def get_group_drop_pose_names(semi_auto_cfg: dict[str, Any], group: str) -> tuple[str, str]:
    entry = semi_auto_cfg.get("drop_poses", {}).get(group_key(group), {})
    if not isinstance(entry, dict):
        raise ValueError(f"Missing drop pose config for {group}")
    hover = entry.get("hover")
    place = entry.get("place")
    if not hover or not place:
        raise ValueError(f"Drop pose config for {group} must include hover/place")
    return str(hover), str(place)


def get_validated_heights(semi_auto_cfg: dict[str, Any]) -> tuple[float, float, float]:
    z_heights = semi_auto_cfg.get("z_heights_m", {})
    return (
        float(z_heights["safe_hover"]),
        float(z_heights["pre_pick"]),
        float(z_heights["lift"]),
    )


def load_optional_ik_calibration(args: Any) -> dict[str, Any] | None:
    if getattr(args, "use_ik_servo_calibration", False) or getattr(args, "use_z_mode_correction", False):
        return load_ik_servo_calibration(args.ik_servo_calibration_config)
    return None


def near_servo_limit_messages(move_angles: list[int], servo_cfg: dict[str, Any], margin_deg: float) -> list[str]:
    if margin_deg <= 0:
        return []
    failures = []
    active_channels = (("ch1", 0), ("ch2", 1), ("ch3", 2), ("ch5", 4))
    servos_cfg = servo_cfg.get("servos", {})
    for channel, index in active_channels:
        cfg = servos_cfg.get(channel, {})
        lo = float(cfg.get("min_angle_deg"))
        hi = float(cfg.get("max_angle_deg"))
        value = int(move_angles[index])
        if value <= lo + margin_deg:
            failures.append(f"{channel.upper()}={value} min={int(lo)} margin={int(margin_deg)}")
        elif value >= hi - margin_deg:
            failures.append(f"{channel.upper()}={value} max={int(hi)} margin={int(margin_deg)}")
    return failures


def build_ik_step(
    step_name: str,
    board_x_cm: float,
    board_y_cm: float,
    z_m: float,
    z_mode: str,
    gripper_angle: int,
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    transform_cfg: dict[str, Any],
    ik_servo_cal_cfg: dict[str, Any] | None,
    args: Any,
) -> dict[str, Any]:
    robot_x_m, robot_y_m = board_to_robot(board_x_cm, board_y_cm, transform_cfg)
    ik_result = solve_planar_ik(robot_x_m, robot_y_m, z_m, kin_cfg, args.tcp_offset_mode, 0.0, 1.0)
    if "unreachable_reason" in ik_result:
        raise ValueError(f"{step_name}: {ik_result['unreachable_reason']}")

    candidate = select_candidate(ik_result, args.solution)
    servos, ch1_debug = candidate_servos_with_debug(candidate, kin_cfg, servo_cfg)
    calibration_logs: list[str] = []
    z_mode_logs: list[str] = []
    if ik_servo_cal_cfg is not None:
        if getattr(args, "use_ik_servo_calibration", False):
            servos, calibration_logs = apply_ik_servo_calibration(servos, ik_servo_cal_cfg)
        if getattr(args, "use_z_mode_correction", False):
            servos, z_mode_logs = apply_z_mode_correction(servos, ik_servo_cal_cfg, z_mode)

    limit_ok, limit_lines = validate_servos(servos, servo_cfg)
    move_angles = rounded_move_safe_angles(servos, gripper_angle)
    move_angles, clamp_notes = clamp_move_safe_angles(move_angles, servo_cfg)
    rounded_failures = validate_move_safe_angles(move_angles, servo_cfg)
    margin_failures = near_servo_limit_messages(
        move_angles,
        servo_cfg,
        float(getattr(args, "servo_limit_margin_deg", 0.0)),
    )
    failures = []
    if not limit_ok:
        failures.extend(line[4:] if line.startswith("BAD ") else line for line in limit_lines if line.startswith("BAD "))
    failures.extend(rounded_failures)
    if failures:
        raise ValueError(f"{step_name}: servo limit failure: {'; '.join(failures)}")
    if margin_failures:
        raise ValueError(f"{step_name}: near servo limit: {'; '.join(margin_failures)}")

    return {
        "type": "MOVE_SAFE",
        "name": step_name,
        "source": "ik",
        "board_x_cm": board_x_cm,
        "board_y_cm": board_y_cm,
        "robot_x_m": robot_x_m,
        "robot_y_m": robot_y_m,
        "z_m": z_m,
        "z_mode": z_mode,
        "solution": candidate["name"],
        "servos": servos,
        "ch1_debug": ch1_debug,
        "calibration_logs": calibration_logs,
        "z_mode_logs": z_mode_logs,
        "move_angles": move_angles,
        "clamp_notes": clamp_notes,
        "command": build_move_command(move_angles),
    }


def build_pose_step(step_name: str, pose_name: str, pose: dict[str, int], servo_cfg: dict[str, Any]) -> dict[str, Any]:
    move_angles = [pose[ch] for ch in ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6")]
    move_angles, clamp_notes, failures = clamp_move_angles(move_angles, servo_cfg)
    if failures:
        raise ValueError(f"{step_name}: taught pose {pose_name} servo failure: {'; '.join(failures)}")
    return {
        "type": "MOVE_SAFE",
        "name": step_name,
        "source": f"pose:{pose_name}",
        "move_angles": move_angles,
        "clamp_notes": clamp_notes,
        "command": build_move_command(move_angles),
    }


def build_gripper_variant_step(step_name: str, source_step: dict[str, Any], ch6_value: int, servo_cfg: dict[str, Any]) -> dict[str, Any]:
    move_angles = list(source_step["move_angles"])
    move_angles[5] = int(ch6_value)
    move_angles, clamp_notes, failures = clamp_move_angles(move_angles, servo_cfg)
    if failures:
        raise ValueError(f"{step_name}: gripper variant servo failure: {'; '.join(failures)}")
    return {
        "type": "MOVE_SAFE",
        "name": step_name,
        "source": f"{source_step['name']}+gripper",
        "move_angles": move_angles,
        "clamp_notes": clamp_notes,
        "command": build_move_command(move_angles),
    }


def build_home_step() -> dict[str, Any]:
    return {"type": "HOME", "name": "HOME_SAFE"}


def build_class_drop_sequence(
    group: str,
    semi_auto_cfg: dict[str, Any],
    taught_cfg: dict[str, Any],
    pose_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    open_deg: int,
) -> list[dict[str, Any]]:
    hover_name, place_name = get_group_drop_pose_names(semi_auto_cfg, group)
    hover_pose = load_named_pose(hover_name, taught_cfg, pose_cfg)
    place_pose = load_named_pose(place_name, taught_cfg, pose_cfg)
    drop_hover = build_pose_step("DROP_HOVER", hover_name, hover_pose, servo_cfg)
    drop_place = build_pose_step("DROP_PLACE", place_name, place_pose, servo_cfg)
    open_gripper = build_gripper_variant_step("OPEN_GRIPPER", drop_place, open_deg, servo_cfg)
    return [drop_hover, drop_place, open_gripper]


def build_pick_sequence_for_target(
    target: dict[str, Any],
    semi_auto_cfg: dict[str, Any],
    kin_cfg: dict[str, Any],
    servo_cfg: dict[str, Any],
    pose_cfg: dict[str, Any],
    taught_cfg: dict[str, Any],
    transform_cfg: dict[str, Any],
    ik_servo_cal_cfg: dict[str, Any] | None,
    args: Any,
    workspace_profile: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group = normalize_group(target["group"])
    if workspace_profile is None:
        workspace_profile = default_workspace_profile(semi_auto_cfg, group)
    safe_hover_z = float(workspace_profile["safe_hover_z_m"])
    pre_pick_z = float(workspace_profile["pre_pick_z_m"])
    lift_z = float(workspace_profile["lift_z_m"])
    grasp_dx = float(workspace_profile["grasp_offset_x_cm"])
    grasp_dy = float(workspace_profile["grasp_offset_y_cm"])
    open_deg, close_deg = get_group_gripper_angles(semi_auto_cfg, servo_cfg, group)

    adjusted_board_x = float(target["board_x_cm"]) + grasp_dx
    adjusted_board_y = float(target["board_y_cm"]) + grasp_dy

    safe_hover = build_ik_step(
        "IK_SAFE_HOVER",
        adjusted_board_x,
        adjusted_board_y,
        safe_hover_z,
        "safe_hover",
        open_deg,
        kin_cfg,
        servo_cfg,
        transform_cfg,
        ik_servo_cal_cfg,
        args,
    )
    pre_pick = build_ik_step(
        "IK_PRE_PICK",
        adjusted_board_x,
        adjusted_board_y,
        pre_pick_z,
        "pre_pick",
        open_deg,
        kin_cfg,
        servo_cfg,
        transform_cfg,
        ik_servo_cal_cfg,
        args,
    )
    close_gripper = build_gripper_variant_step("CLOSE_GRIPPER", pre_pick, close_deg, servo_cfg)
    lift = build_ik_step(
        "IK_LIFT",
        adjusted_board_x,
        adjusted_board_y,
        lift_z,
        "lift",
        close_deg,
        kin_cfg,
        servo_cfg,
        transform_cfg,
        ik_servo_cal_cfg,
        args,
    )

    clear_pose = load_named_pose("CLEAR_TEST", taught_cfg, pose_cfg)
    home_pose = load_named_pose("HOME_SAFE", taught_cfg, pose_cfg)
    clear_step = build_pose_step("CLEAR", "CLEAR_TEST", clear_pose, servo_cfg)
    drop_steps = build_class_drop_sequence(group, semi_auto_cfg, taught_cfg, pose_cfg, servo_cfg, open_deg)
    home_move = build_pose_step("HOME_SAFE_MOVE", "HOME_SAFE", home_pose, servo_cfg)

    sequence = [safe_hover, pre_pick, close_gripper, lift, clear_step, *drop_steps, home_move]
    meta = {
        "group": group,
        "raw_class": target["raw"],
        "confidence": float(target["conf"]),
        "pixel_u": int(target["u"]),
        "pixel_v": int(target["v"]),
        "board_x_cm": float(target["board_x_cm"]),
        "board_y_cm": float(target["board_y_cm"]),
        "adjusted_board_x_cm": adjusted_board_x,
        "adjusted_board_y_cm": adjusted_board_y,
        "robot_x_m": safe_hover["robot_x_m"],
        "robot_y_m": safe_hover["robot_y_m"],
        "grasp_offset_x_cm": grasp_dx,
        "grasp_offset_y_cm": grasp_dy,
        "workspace_region": workspace_profile["region_name"],
        "workspace_region_status": workspace_profile["status"],
        "workspace_region_distance_cm": float(workspace_profile["distance_cm"]),
        "workspace_validated": bool(workspace_profile["validated"]),
        "workspace_safe_hover_z_m": safe_hover_z,
        "workspace_pre_pick_z_m": pre_pick_z,
        "workspace_lift_z_m": lift_z,
    }
    return sequence, meta


def format_sequence_preview(steps: list[dict[str, Any]]) -> str:
    lines = []
    for index, step in enumerate(steps, start=1):
        if step["type"] == "HOME":
            lines.append(f"  {index:02d}. {step['name']}: HOME")
            continue
        if step["type"] == "ERROR":
            lines.append(f"  {index:02d}. {step['name']}: ERROR ({step['source']})")
            if "board_x_cm" in step:
                lines.append(
                    f"      board=({step['board_x_cm']:.2f},{step['board_y_cm']:.2f}) "
                    f"z={step.get('z_m', 0.0):.3f} z_mode={step.get('z_mode', '?')}"
                )
            lines.append(f"      [IK][ERROR] {step['error']}")
            continue
        lines.append(f"  {index:02d}. {step['name']}: {step['command']} ({step['source']})")
        if "board_x_cm" in step:
            lines.append(
                f"      board=({step['board_x_cm']:.2f},{step['board_y_cm']:.2f}) "
                f"robot=({step['robot_x_m']:.4f},{step['robot_y_m']:.4f}) "
                f"z={step['z_m']:.3f} z_mode={step['z_mode']}"
            )
        for note in step.get("clamp_notes", []):
            lines.append(f"      [SAFETY] {note}")
    return "\n".join(lines)


def response_has_esp32_reset(lines: list[str]) -> bool:
    reset_markers = ("rst:", "boot:", "ets ", "brownout", "guru meditation")
    for line in lines:
        normalized = line.strip().lower()
        if any(marker in normalized for marker in reset_markers):
            return True
    return False


def attempted_too_close(
    board_x_cm: float,
    board_y_cm: float,
    attempted_targets: list[dict[str, float]],
    min_distance_cm: float = 1.5,
) -> bool:
    for prev in attempted_targets:
        dx = board_x_cm - float(prev["adjusted_board_x_cm"])
        dy = board_y_cm - float(prev["adjusted_board_y_cm"])
        if (dx * dx + dy * dy) ** 0.5 <= min_distance_cm:
            return True
    return False
