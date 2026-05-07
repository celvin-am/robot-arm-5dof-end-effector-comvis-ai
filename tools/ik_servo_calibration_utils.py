#!/usr/bin/env python3
"""
ik_servo_calibration_utils.py

Utilities for fitting and applying an empirical post-IK servo calibration
layer. This layer is intentionally separate from the IK math itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_IK_SERVO_CALIBRATION_CONFIG = (
    "ros2_ws/src/robot_arm_5dof/config/ik_servo_calibration.yaml"
)
ACTIVE_IK_CHANNELS = ("ch1", "ch2", "ch3", "ch5")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"[CONFIG][ERROR] Root YAML object must be a mapping: {path}")
    return data


def load_yaml_if_exists(path: str) -> dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}
    return load_yaml(path)


def load_ik_servo_calibration(path: str) -> dict[str, Any]:
    root = load_yaml(path)
    cfg = root.get("ik_servo_calibration", {})
    if not isinstance(cfg, dict):
        raise RuntimeError(f"[CONFIG][ERROR] ik_servo_calibration mapping missing in {path}")
    return cfg


def apply_ik_servo_calibration(
    servos: dict[str, float],
    calibration_cfg: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    calibrated = dict(servos)
    logs: list[str] = []
    channels = calibration_cfg.get("channels", {})
    if not isinstance(channels, dict):
        raise RuntimeError("[CONFIG][ERROR] ik_servo_calibration.channels must be a mapping")

    for channel in ACTIVE_IK_CHANNELS:
        entry = channels.get(channel, {})
        if not isinstance(entry, dict) or not entry.get("enabled", False):
            continue

        raw_value = float(servos[channel])
        scale = float(entry.get("scale", 1.0))
        offset = float(entry.get("offset_deg", 0.0))
        out = scale * raw_value + offset
        calibrated[channel] = out
        logs.append(f"[IK_CAL] {channel}_raw_servo = {raw_value:.2f}")
        logs.append(f"[IK_CAL] {channel}_scale = {scale:.6f}")
        logs.append(f"[IK_CAL] {channel}_offset_deg = {offset:.6f}")
        logs.append(f"[IK_CAL] {channel}_calibrated_servo = {out:.2f}")

    return calibrated, logs


def apply_z_mode_correction(
    servos: dict[str, float],
    calibration_cfg: dict[str, Any],
    z_mode: str,
) -> tuple[dict[str, float], list[str]]:
    corrected = dict(servos)
    logs: list[str] = [f"[IK_ZMODE] z_mode={z_mode}"]
    z_mode_corrections = calibration_cfg.get("z_mode_corrections", {})
    if not isinstance(z_mode_corrections, dict):
        raise RuntimeError("[CONFIG][ERROR] z_mode_corrections must be a mapping")
    modes = z_mode_corrections.get("modes", {})
    if not isinstance(modes, dict):
        raise RuntimeError("[CONFIG][ERROR] z_mode_corrections.modes must be a mapping")
    mode_entry = modes.get(z_mode, {})
    if not isinstance(mode_entry, dict):
        raise RuntimeError(f"[CONFIG][ERROR] z_mode_corrections.modes.{z_mode} must be a mapping")

    for channel in ACTIVE_IK_CHANNELS:
        key = f"{channel}_offset_deg"
        offset = float(mode_entry.get(key, 0.0))
        before = float(servos[channel])
        after = before + offset
        corrected[channel] = after
        logs.append(f"[IK_ZMODE] {key}={offset:.6f}")
        logs.append(f"[IK_ZMODE] {channel}_before={before:.2f}")
        logs.append(f"[IK_ZMODE] {channel}_after={after:.2f}")

    return corrected, logs


def fit_affine_channels_from_samples(
    sample_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    fitted: dict[str, dict[str, float]] = {}
    for channel in ACTIVE_IK_CHANNELS:
        raw_values = []
        ref_values = []
        for row in sample_rows:
            raw_values.append(float(row["ik_move_safe"][channel]))
            ref_values.append(float(row["reference_servos"][channel]))

        x = np.asarray(raw_values, dtype=np.float64)
        y = np.asarray(ref_values, dtype=np.float64)

        if len(x) == 0:
            raise RuntimeError(f"[IK_CAL][ERROR] No fit samples available for {channel}")
        if len(x) == 1:
            scale = 1.0
            offset = float(y[0] - x[0])
        else:
            A = np.vstack([x, np.ones_like(x)]).T
            scale, offset = np.linalg.lstsq(A, y, rcond=None)[0]

        fitted[channel] = {
            "scale": float(scale),
            "offset_deg": float(offset),
        }
    return fitted


def build_ik_servo_calibration_document(
    fitted_channels: dict[str, dict[str, float]],
    sample_names: list[str],
    z_mode_corrections: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc = {
        "ik_servo_calibration": {
            "method": "affine_from_reference_samples",
            "status": "initial_fit_from_center_references",
            "source_samples": sample_names,
            "updated_at": utc_timestamp(),
            "warning": (
                "Initial empirical fit from center references only. "
                "Not validated for whole workspace."
            ),
            "channels": {
                "ch1": {
                    "enabled": True,
                    "scale": float(fitted_channels["ch1"]["scale"]),
                    "offset_deg": float(fitted_channels["ch1"]["offset_deg"]),
                    "note": "CH1 already uses front_servo_deg=80; affine fit should remain close.",
                },
                "ch2": {
                    "enabled": True,
                    "scale": float(fitted_channels["ch2"]["scale"]),
                    "offset_deg": float(fitted_channels["ch2"]["offset_deg"]),
                },
                "ch3": {
                    "enabled": True,
                    "scale": float(fitted_channels["ch3"]["scale"]),
                    "offset_deg": float(fitted_channels["ch3"]["offset_deg"]),
                    "note": "CH3 has physical direction observation; validate carefully before live IK.",
                },
                "ch5": {
                    "enabled": True,
                    "scale": float(fitted_channels["ch5"]["scale"]),
                    "offset_deg": float(fitted_channels["ch5"]["offset_deg"]),
                },
            },
        }
    }
    if z_mode_corrections is not None:
        doc["ik_servo_calibration"]["z_mode_corrections"] = z_mode_corrections
    return doc


def fit_z_mode_corrections_from_samples(
    sample_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in sample_rows:
        z_mode = str(row.get("z_mode", "custom"))
        grouped.setdefault(z_mode, []).append(row)

    modes: dict[str, Any] = {}
    for z_mode, rows in grouped.items():
        mode_entry: dict[str, Any] = {}
        for channel in ACTIVE_IK_CHANNELS:
            residuals = [
                float(row["reference_servos"][channel]) - float(row["ik_move_safe"][channel])
                for row in rows
            ]
            mode_entry[f"{channel}_offset_deg"] = round(float(np.mean(residuals)), 6)
        modes[z_mode] = mode_entry

    return {
        "enabled": False,
        "status": "initial_from_center_reference_errors",
        "warning": (
            "Initial z_mode correction from center references only; "
            "not whole-workspace validated."
        ),
        "modes": modes,
    }


def write_ik_servo_calibration(path: str, document: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=False)
