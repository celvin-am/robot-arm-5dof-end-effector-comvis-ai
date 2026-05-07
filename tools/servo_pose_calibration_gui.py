#!/usr/bin/env python3
"""
servo_pose_calibration_gui.py

Tkinter GUI for guarded manual servo and pose calibration over the stable ESP32
serial protocol. This tool is dry-run by default unless the operator
explicitly enables live hardware risk mode.
"""

from __future__ import annotations

import argparse
import math
import queue
import string
import sys
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import yaml

from test_esp32_manual_move import load_serial_module


DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD_FALLBACK = 115200
DEFAULT_STEP = 2
DEFAULT_SERVO_CONFIG = "ros2_ws/src/robot_arm_5dof/config/servo_config.yaml"
DEFAULT_POSE_CONFIG = "ros2_ws/src/robot_arm_5dof/config/pose_config.yaml"
DEFAULT_KINEMATICS_CONFIG = "ros2_ws/src/robot_arm_5dof/config/robot_kinematics.yaml"
DEFAULT_SERIAL_CONFIG = "ros2_ws/src/robot_arm_5dof/config/serial_config.yaml"
DEFAULT_OUTPUT = "ros2_ws/src/robot_arm_5dof/config/taught_pick_place_poses.yaml"
DEFAULT_BOARD_CONFIG = "ros2_ws/src/robot_arm_5dof/config/board_config.yaml"
DEFAULT_TRANSFORM_CONFIG = "ros2_ws/src/robot_arm_5dof/config/robot_board_transform.yaml"
DEFAULT_IK_REFERENCE_OUTPUT = "ros2_ws/src/robot_arm_5dof/config/ik_reference_samples.yaml"
DEFAULT_DIRECTION_OBSERVATIONS_OUTPUT = "ros2_ws/src/robot_arm_5dof/config/servo_direction_observations.yaml"
SERIAL_TIMEOUT_SEC = 5.0
CONNECT_RESET_WAIT_SEC = 2.0
POST_CONNECT_FLUSH_SEC = 0.1
DEFAULT_FIRMWARE_HOME = {"ch1": 90, "ch2": 130, "ch3": 130, "ch4": 95, "ch5": 60, "ch6": 45}

CHANNELS = ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6")
POSE_NAMES = (
    "HOME_SAFE",
    "READY_ABOVE_BOARD",
    "HOVER_PICK_TEST",
    "PICK_TEST",
    "LIFT_TEST",
    "HOVER_DROP_TEST",
    "PLACE_TEST",
    "CLEAR_TEST",
    "CAKE_DROP_HOVER",
    "CAKE_DROP_PLACE",
    "DONUT_DROP_HOVER",
    "DONUT_DROP_PLACE",
)
CHANNEL_META = {
    "ch1": {"gpio": 13, "label": "CH1", "joint": "base_yaw"},
    "ch2": {"gpio": 14, "label": "CH2", "joint": "shoulder_pitch"},
    "ch3": {"gpio": 27, "label": "CH3", "joint": "elbow_pitch"},
    "ch4": {"gpio": 26, "label": "CH4", "joint": "wrist_rotate"},
    "ch5": {"gpio": 25, "label": "CH5", "joint": "wrist_pitch"},
    "ch6": {"gpio": 33, "label": "CH6", "joint": "gripper"},
}
POSE_DESCRIPTIONS = {
    "HOME_SAFE": "Human-taught HOME_SAFE reference.",
    "READY_ABOVE_BOARD": "Human-taught ready pose above checkerboard.",
    "HOVER_PICK_TEST": "Human-taught hover pose above object before pick.",
    "PICK_TEST": "Human-taught object contact pose.",
    "LIFT_TEST": "Human-taught lift pose after pick.",
    "HOVER_DROP_TEST": "Human-taught hover pose above generic drop zone.",
    "PLACE_TEST": "Human-taught generic place pose.",
    "CLEAR_TEST": "Human-taught clear / retreat pose.",
    "CAKE_DROP_HOVER": "Human-taught CAKE drop hover pose.",
    "CAKE_DROP_PLACE": "Human-taught CAKE drop place pose.",
    "DONUT_DROP_HOVER": "Human-taught DONUT drop hover pose.",
    "DONUT_DROP_PLACE": "Human-taught DONUT drop place pose.",
}
GRIPPER_DEFAULTS = {"open_deg": 50, "close_soft_deg": 35, "close_full_deg": 15}
EXPECTED_SERIAL_PREFIXES = ("PONG", "STATUS ", "LIMITS ", "HELP ", "ACK ", "DONE ", "ERR ")
BOOT_NOISE_PREFIXES = ("READY ", "HELP ")
OBSERVATION_OPTIONS = {
    "ch1": ("clockwise", "counter_clockwise", "unknown"),
    "ch2": ("arm_up", "arm_down", "tcp_up", "tcp_down", "unknown"),
    "ch3": ("elbow_up", "elbow_down", "tcp_up", "tcp_down", "elbow_or_tcp_down", "elbow_or_tcp_more_down", "unknown"),
    "ch4": ("rotate_left", "rotate_right", "unknown"),
    "ch5": ("gripper_tip_up", "gripper_tip_down", "unknown"),
    "ch6": ("open", "close", "unknown"),
}
OBSERVATION_STATUS_OPTIONS = ("unknown", "physically_observed", "needs_retest")


@dataclass
class SerialAction:
    command: str
    expect_motion: bool = False
    update_from_status: bool = False
    apply_angles_on_done: dict[str, int] | None = None
    user_label: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml_file(path: str, label: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise RuntimeError(f"Missing {label}: {path}")
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in {label}: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} must contain a YAML mapping: {path}")
    return data


def save_yaml_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def default_baud_from_serial_config(path: str) -> int:
    try:
        serial_cfg = load_yaml_file(path, "serial config")
    except RuntimeError:
        return DEFAULT_BAUD_FALLBACK
    return int(serial_cfg.get("serial", {}).get("baud_rate", DEFAULT_BAUD_FALLBACK))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tkinter GUI for guarded servo pose calibration using ESP32 HOME/MOVE_SAFE/STOP.",
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=None, help="Defaults to serial_config baud_rate or 115200.")
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--kinematics-config", default=DEFAULT_KINEMATICS_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--board-config", default=DEFAULT_BOARD_CONFIG)
    parser.add_argument("--transform-config", default=DEFAULT_TRANSFORM_CONFIG)
    parser.add_argument("--ik-reference-output", default=DEFAULT_IK_REFERENCE_OUTPUT)
    parser.add_argument("--servo-direction-observations-output", default=DEFAULT_DIRECTION_OBSERVATIONS_OUTPUT)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--yes-i-understand-hardware-risk", action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    parser.add_argument("--serial-config", default=DEFAULT_SERIAL_CONFIG)
    return parser


class ServoPoseCalibrationGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.live_enabled = bool(args.yes_i_understand_hardware_risk)
        self.dry_run_mode = not self.live_enabled
        self.serial = None
        self.worker_thread: threading.Thread | None = None
        self.worker_busy = False
        self.pending_stop = False
        self.stop_sent_for_current_motion = False
        self.current_action: SerialAction | None = None
        self.ui_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        self.output_path = Path(args.output)
        self.ik_reference_output_path = Path(args.ik_reference_output)
        self.direction_observations_output_path = Path(args.servo_direction_observations_output)
        self.servo_cfg = {}
        self.pose_cfg = {}
        self.kin_cfg = {}
        self.serial_cfg = {}
        self.board_cfg = {}
        self.transform_cfg = {}
        self.taught_payload = {}
        self.ik_reference_payload = {}
        self.direction_observations_payload = {}
        self.pose_cache: dict[str, dict[str, int]] = {}
        self.limit_cache: dict[str, tuple[int, int]] = {}
        self.gripper_calibration = deepcopy(GRIPPER_DEFAULTS)
        self.board_width_cm = 28.5
        self.board_height_cm = 18.0
        self.session_log_lines: list[str] = []

        self.channel_vars = {channel: tk.IntVar(value=DEFAULT_FIRMWARE_HOME[channel]) for channel in CHANNELS}
        self.status_vars = {channel: tk.StringVar(value="-") for channel in CHANNELS}
        self.limit_vars = {channel: tk.StringVar(value="") for channel in CHANNELS}
        self.port_var = tk.StringVar(value=args.port)
        self.baud_var = tk.IntVar(value=int(args.baud))
        self.step_var = tk.IntVar(value=max(1, int(args.step)))
        self.connection_var = tk.StringVar(value="DRY RUN" if self.dry_run_mode else "DISCONNECTED")
        self.mode_var = tk.StringVar(value="DRY RUN" if self.dry_run_mode else "LIVE")
        self.active_loaded_pose_var = tk.StringVar(value="(none)")
        self.safety_warning_var = tk.StringVar(value="")
        self.current_command_var = tk.StringVar(value="")
        self.latest_status_var = tk.StringVar(value="No STATUS yet")
        self.custom_pose_name_var = tk.StringVar(value="")
        self.ik_ref_name_var = tk.StringVar(value="IK_REF_HOVER_CENTER")
        self.ik_ref_board_x_var = tk.StringVar(value="13.51")
        self.ik_ref_board_y_var = tk.StringVar(value="8.80")
        self.ik_ref_z_var = tk.StringVar(value="0.12")
        self.ik_ref_z_mode_var = tk.StringVar(value="safe_hover")
        self.ik_ref_object_class_var = tk.StringVar(value="center_test")
        self.ik_ref_comment_var = tk.StringVar(value="")
        self.ik_ref_gripper_state_var = tk.StringVar(value="unknown")
        self.ik_ref_robot_x_var = tk.StringVar(value="-")
        self.ik_ref_robot_y_var = tk.StringVar(value="-")
        self.ik_ref_active_name_var = tk.StringVar(value="(none)")
        self.ik_ref_out_of_board_allowed_var = tk.BooleanVar(value=False)
        self.ik_ref_validation_var = tk.StringVar(value="VALID")
        self.gripper_open_var = tk.IntVar(value=GRIPPER_DEFAULTS["open_deg"])
        self.gripper_soft_var = tk.IntVar(value=GRIPPER_DEFAULTS["close_soft_deg"])
        self.gripper_full_var = tk.IntVar(value=GRIPPER_DEFAULTS["close_full_deg"])
        self.direction_plus_vars = {channel: tk.StringVar(value="unknown") for channel in CHANNELS}
        self.direction_minus_vars = {channel: tk.StringVar(value="unknown") for channel in CHANNELS}
        self.direction_status_vars = {channel: tk.StringVar(value="unknown") for channel in CHANNELS}
        self.direction_note_vars = {channel: tk.StringVar(value="") for channel in CHANNELS}
        self.direction_plus_from_vars = {channel: tk.StringVar(value="") for channel in CHANNELS}
        self.direction_plus_to_vars = {channel: tk.StringVar(value="") for channel in CHANNELS}
        self.direction_minus_from_vars = {channel: tk.StringVar(value="") for channel in CHANNELS}
        self.direction_minus_to_vars = {channel: tk.StringVar(value="") for channel in CHANNELS}

        for variable in (
            self.ik_ref_name_var,
            self.ik_ref_board_x_var,
            self.ik_ref_board_y_var,
            self.ik_ref_z_var,
            self.ik_ref_z_mode_var,
            self.ik_ref_object_class_var,
            self.ik_ref_comment_var,
            self.ik_ref_gripper_state_var,
            self.ik_ref_out_of_board_allowed_var,
        ):
            variable.trace_add("write", lambda *_args: self._refresh_ik_ref_preview())

        self.log_widget: scrolledtext.ScrolledText | None = None
        self.yaml_widget: scrolledtext.ScrolledText | None = None
        self.ik_ref_preview_widget: scrolledtext.ScrolledText | None = None
        self.telemetry_widget: scrolledtext.ScrolledText | None = None
        self.top_view_canvas: tk.Canvas | None = None
        self.side_view_canvas: tk.Canvas | None = None
        self.scale_widgets: dict[str, tk.Scale] = {}
        self.pose_button_frame: ttk.Frame | None = None
        self.pose_notebook: ttk.Notebook | None = None

        self._reload_all_configs(initial=True)
        self._apply_angles_to_ui(self._resolve_home_safe_angles())

        self.root.title("Servo Pose Calibration GUI")
        self.root.geometry("1740x980")
        self.root.minsize(1400, 820)

        self._build_ui()
        self._refresh_servo_limit_labels()
        self._refresh_gripper_widgets()
        self._refresh_yaml_preview()
        self._refresh_ik_ref_preview()
        self._update_command_preview()

        self._log(
            "[CONFIG] "
            + ("Dry-run mode: serial commands will be logged only." if self.dry_run_mode else "Live mode enabled by operator risk flag.")
        )
        self._log("[CONFIG] Firmware HOME is not YAML HOME_SAFE.")
        self._log("[CONFIG] Saving a pose means human_taught_live, validated_for=manual_testing_only, autonomous_validated=false.")
        self._log("[IK_REF] IK reference samples are coordinate-to-servo calibration data, separate from taught pick/place poses.")
        self.root.after(100, self._drain_ui_queue)

    def _reload_all_configs(self, initial: bool = False) -> None:
        self.servo_cfg = load_yaml_file(self.args.servo_config, "servo config")
        self.pose_cfg = load_yaml_file(self.args.pose_config, "pose config")
        self.kin_cfg = load_yaml_file(self.args.kinematics_config, "kinematics config")
        self.serial_cfg = load_yaml_file(self.args.serial_config, "serial config")
        self.board_cfg = load_yaml_file(self.args.board_config, "board config")
        self.transform_cfg = load_yaml_file(self.args.transform_config, "board transform config")
        self.taught_payload = self._load_or_init_output()
        self.ik_reference_payload = self._load_or_init_ik_reference_output()
        self.direction_observations_payload = self._load_or_init_direction_observations_output()
        self.limit_cache = self._read_limits(self.servo_cfg)
        self.pose_cache = self._build_pose_cache()
        self.gripper_calibration = self._read_gripper_calibration(self.servo_cfg)
        self.board_width_cm, self.board_height_cm = self._read_board_limits(self.board_cfg)
        self._load_direction_observations_into_vars()
        self._update_direction_observation_warning()
        if not initial:
            self._refresh_servo_limit_labels()
            self._refresh_gripper_widgets()
            self._refresh_yaml_preview()
            self._refresh_ik_ref_preview()
            self._refresh_telemetry_preview()
            self._refresh_visualization()
            self._update_command_preview()

    def _load_or_init_output(self) -> dict[str, Any]:
        if self.output_path.exists():
            payload = load_yaml_file(str(self.output_path), "taught pose output")
        else:
            payload = {
                "metadata": {
                    "status": "phase13_gui_pose_teach",
                    "created_at": utc_now(),
                    "updated_at": None,
                    "note": "GUI-taught MOVE_SAFE servo poses. Software-only unless a human confirms live motion in the current session.",
                },
                "poses": {},
            }
        payload.setdefault("metadata", {})
        payload.setdefault("poses", {})
        return payload

    def _load_or_init_ik_reference_output(self) -> dict[str, Any]:
        if self.ik_reference_output_path.exists():
            payload = load_yaml_file(str(self.ik_reference_output_path), "IK reference output")
        else:
            payload = {
                "metadata": {
                    "status": "ik_reference_samples",
                    "note": "Human-taught servo poses paired with measured board/robot coordinates for IK calibration.",
                    "updated_at": None,
                },
                "samples": {},
            }
        payload.setdefault("metadata", {})
        payload.setdefault("samples", {})
        return payload

    def _load_or_init_direction_observations_output(self) -> dict[str, Any]:
        if self.direction_observations_output_path.exists():
            payload = load_yaml_file(str(self.direction_observations_output_path), "servo direction observations output")
        else:
            payload = {
                "metadata": {
                    "status": "servo_direction_observations",
                    "updated_at": utc_now(),
                    "note": "Physical observations only. Do not directly treat as IK calibration until verified.",
                },
                "observations": {
                    "ch3": {
                        "function": "elbow_pitch",
                        "plus_motion": {
                            "from_deg": 130,
                            "to_deg": 140,
                            "observed": "elbow_or_tcp_more_down",
                        },
                        "minus_motion": {
                            "from_deg": 130,
                            "to_deg": 100,
                            "observed": "elbow_or_tcp_down",
                        },
                        "status": "physically_observed",
                        "note": "CH3 real motion differs from initial expectation; validate joint-to-servo conversion before live IK.",
                    }
                },
            }
            save_yaml_file(self.direction_observations_output_path, payload)
        payload.setdefault("metadata", {})
        payload.setdefault("observations", {})
        return payload

    def _read_limits(self, servo_cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
        servos = servo_cfg.get("servos", {})
        limits: dict[str, tuple[int, int]] = {}
        for channel in CHANNELS:
            entry = servos.get(channel, {})
            lo = entry.get("min_angle_deg")
            hi = entry.get("max_angle_deg")
            if lo is None or hi is None:
                raise RuntimeError(f"servo_config missing min/max for {channel}")
            limits[channel] = (int(lo), int(hi))
        return limits

    def _read_gripper_calibration(self, servo_cfg: dict[str, Any]) -> dict[str, int]:
        root = servo_cfg.get("gripper_calibration", {})
        servo_ch6 = servo_cfg.get("servos", {}).get("ch6", {})
        return {
            "open_deg": int(root.get("open_deg", servo_ch6.get("open_angle_deg", GRIPPER_DEFAULTS["open_deg"]))),
            "close_soft_deg": int(root.get("close_soft_deg", 35)),
            "close_full_deg": int(root.get("close_full_deg", servo_ch6.get("close_angle_deg", GRIPPER_DEFAULTS["close_full_deg"]))),
        }

    def _read_board_limits(self, board_cfg: dict[str, Any]) -> tuple[float, float]:
        board = board_cfg.get("board", {})
        return float(board.get("width_cm", 28.5)), float(board.get("height_cm", 18.0))

    def _load_direction_observations_into_vars(self) -> None:
        observations = self.direction_observations_payload.get("observations", {})
        for channel in CHANNELS:
            obs = observations.get(channel, {})
            plus = obs.get("plus_motion", {})
            minus = obs.get("minus_motion", {})
            self.direction_plus_vars[channel].set(str(plus.get("observed", "unknown")))
            self.direction_minus_vars[channel].set(str(minus.get("observed", "unknown")))
            self.direction_status_vars[channel].set(str(obs.get("status", "unknown")))
            self.direction_note_vars[channel].set(str(obs.get("note", "")))
            self.direction_plus_from_vars[channel].set("" if plus.get("from_deg") is None else str(plus.get("from_deg")))
            self.direction_plus_to_vars[channel].set("" if plus.get("to_deg") is None else str(plus.get("to_deg")))
            self.direction_minus_from_vars[channel].set("" if minus.get("from_deg") is None else str(minus.get("from_deg")))
            self.direction_minus_to_vars[channel].set("" if minus.get("to_deg") is None else str(minus.get("to_deg")))

    def _update_direction_observation_warning(self) -> None:
        ch3_obs = self.direction_observations_payload.get("observations", {}).get("ch3", {})
        if ch3_obs and str(ch3_obs.get("status", "unknown")) == "physically_observed":
            self.safety_warning_var.set(
                "[SAFETY] CH3 direction observation exists. Do not trust live IK until CH2/CH3/CH5 conversion is calibrated from IK reference samples."
            )
        else:
            self.safety_warning_var.set("")

    def _angles_from_block(self, block: dict[str, Any]) -> dict[str, int] | None:
        if not isinstance(block, dict):
            return None
        values: dict[str, int] = {}
        for channel in CHANNELS:
            value = block.get(channel)
            if value is None:
                return None
            values[channel] = int(value)
        return values

    def _build_pose_cache(self) -> dict[str, dict[str, int]]:
        cache: dict[str, dict[str, int]] = {}
        taught_poses = self.taught_payload.get("poses", {})
        pose_cfg_poses = self.pose_cfg.get("poses", {})
        for pose_name in POSE_NAMES:
            taught = self._angles_from_block(taught_poses.get(pose_name, {}))
            config_pose = self._angles_from_block(pose_cfg_poses.get(pose_name, {}))
            if taught is not None:
                cache[pose_name] = taught
            elif config_pose is not None:
                cache[pose_name] = config_pose
        home_from_cache = cache.get("HOME_SAFE")
        if home_from_cache is None:
            cache["HOME_SAFE"] = deepcopy(DEFAULT_FIRMWARE_HOME)
        return cache

    def _resolve_home_safe_angles(self) -> dict[str, int]:
        return deepcopy(self.pose_cache.get("HOME_SAFE", DEFAULT_FIRMWARE_HOME))

    def _current_angles(self) -> dict[str, int]:
        return {channel: int(self.channel_vars[channel].get()) for channel in CHANNELS}

    def _current_status_angles(self) -> dict[str, int] | None:
        values: dict[str, int] = {}
        found = False
        for channel in CHANNELS:
            raw = self.status_vars[channel].get()
            if raw != "-":
                found = True
                try:
                    values[channel] = int(raw)
                except ValueError:
                    return None
        return values if found else None

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(outer)
        top.pack(fill=tk.X, pady=(0, 8))
        body = ttk.Panedwindow(outer, orient=tk.VERTICAL)
        body.pack(fill=tk.BOTH, expand=True)

        middle = ttk.Panedwindow(body, orient=tk.HORIZONTAL)
        bottom = ttk.Panedwindow(body, orient=tk.HORIZONTAL)
        body.add(middle, weight=7)
        body.add(bottom, weight=4)

        left_top = ttk.Frame(top)
        left_top.pack(side=tk.LEFT, fill=tk.X, expand=True)
        right_top = ttk.Frame(top)
        right_top.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_connection_panel(left_top)
        self._build_status_panel(right_top)
        self._build_servo_panel(middle)
        self._build_pose_panel(middle)
        self._build_bottom_panels(bottom)

    def _build_connection_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Connection")
        frame.pack(fill=tk.X, expand=True)

        ttk.Label(frame, text="Port").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.port_var, width=16).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Baud").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(frame, textvariable=self.baud_var, width=10).grid(row=0, column=3, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Connection").grid(row=0, column=4, sticky="w", padx=4, pady=4)
        ttk.Label(frame, textvariable=self.connection_var).grid(row=0, column=5, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Mode").grid(row=0, column=6, sticky="w", padx=4, pady=4)
        ttk.Label(frame, textvariable=self.mode_var).grid(row=0, column=7, sticky="w", padx=4, pady=4)

        buttons = [
            ("Connect", self.on_connect),
            ("Disconnect", self.on_disconnect),
            ("Flush Serial", self.on_flush_serial),
            ("PING", lambda: self.queue_serial_action(SerialAction("PING", user_label="PING"))),
            ("STATUS", lambda: self.queue_serial_action(SerialAction("STATUS", update_from_status=True, user_label="STATUS"))),
            ("LIMITS", lambda: self.queue_serial_action(SerialAction("LIMITS", user_label="LIMITS"))),
            ("Firmware HOME", self.on_firmware_home),
            ("Go HOME_SAFE", lambda: self.go_pose("HOME_SAFE")),
            ("Reload Poses", self.reload_pose_cache),
            ("STOP", self.on_stop),
        ]
        for idx, (label, cmd) in enumerate(buttons):
            ttk.Button(frame, text=label, command=cmd).grid(row=1, column=idx, sticky="ew", padx=4, pady=6)

    def _build_status_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Current Pose Reporting")
        frame.pack(fill=tk.BOTH, padx=(8, 0))
        ttk.Label(frame, text="Active loaded pose").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(frame, textvariable=self.active_loaded_pose_var).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Latest STATUS").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(frame, textvariable=self.latest_status_var, wraplength=400).grid(row=1, column=1, sticky="w", padx=4, pady=4)
        tk.Label(frame, textvariable=self.safety_warning_var, wraplength=420, justify="left", fg="#b00020").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=4
        )

    def _build_servo_panel(self, parent: ttk.Panedwindow) -> None:
        frame = ttk.LabelFrame(parent, text="Servo Control / Limit Calibration")
        parent.add(frame, weight=5)

        header = ("Channel", "Slider", "Current", "STATUS", "Min/Max", "Jog", "Limit Cal")
        for col, title in enumerate(header):
            ttk.Label(frame, text=title).grid(row=0, column=col, sticky="w", padx=4, pady=4)

        for row, channel in enumerate(CHANNELS, start=1):
            meta = CHANNEL_META[channel]
            lo, hi = self.limit_cache[channel]
            ttk.Label(frame, text=f"{meta['label']} GPIO{meta['gpio']} {meta['joint']}").grid(
                row=row, column=0, sticky="w", padx=4, pady=4
            )

            scale = tk.Scale(
                frame,
                from_=lo,
                to=hi,
                orient=tk.HORIZONTAL,
                showvalue=False,
                length=360,
                variable=self.channel_vars[channel],
                command=lambda _value, ch=channel: self.on_slider_change(ch),
            )
            scale.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
            self.scale_widgets[channel] = scale

            ttk.Label(frame, textvariable=self.channel_vars[channel], width=4).grid(row=row, column=2, sticky="w", padx=4, pady=4)
            ttk.Label(frame, textvariable=self.status_vars[channel], width=4).grid(row=row, column=3, sticky="w", padx=4, pady=4)
            ttk.Label(frame, textvariable=self.limit_vars[channel], width=12).grid(row=row, column=4, sticky="w", padx=4, pady=4)

            jog = ttk.Frame(frame)
            jog.grid(row=row, column=5, sticky="w", padx=4, pady=4)
            ttk.Button(jog, text="-", width=3, command=lambda ch=channel: self.jog_channel(ch, -1)).pack(side=tk.LEFT, padx=(0, 2))
            ttk.Button(jog, text="+", width=3, command=lambda ch=channel: self.jog_channel(ch, 1)).pack(side=tk.LEFT)

            limit_bar = ttk.Frame(frame)
            limit_bar.grid(row=row, column=6, sticky="w", padx=4, pady=4)
            ttk.Button(limit_bar, text="Set Min From Current", command=lambda ch=channel: self.set_limit_from_current(ch, "min")).pack(
                side=tk.LEFT, padx=(0, 2)
            )
            ttk.Button(limit_bar, text="Set Max From Current", command=lambda ch=channel: self.set_limit_from_current(ch, "max")).pack(side=tk.LEFT)

        frame.columnconfigure(1, weight=1)

        bottom_controls = ttk.Frame(frame)
        bottom_controls.grid(row=len(CHANNELS) + 1, column=0, columnspan=7, sticky="ew", padx=4, pady=(8, 4))
        ttk.Button(bottom_controls, text="Save Servo Limits", command=self.save_servo_limits).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(bottom_controls, text="Step (deg)").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Spinbox(bottom_controls, from_=1, to=30, textvariable=self.step_var, width=6, command=self._on_step_changed).pack(side=tk.LEFT, padx=(0, 6))
        for step_value in (1, 2, 5, 10):
            ttk.Button(bottom_controls, text=str(step_value), command=lambda value=step_value: self.set_step(value)).pack(side=tk.LEFT, padx=(0, 4))

        command_bar = ttk.Frame(frame)
        command_bar.grid(row=len(CHANNELS) + 2, column=0, columnspan=7, sticky="ew", padx=4, pady=(4, 0))
        ttk.Label(command_bar, text="Current MOVE_SAFE").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(command_bar, textvariable=self.current_command_var).pack(side=tk.LEFT)

        actions = ttk.LabelFrame(frame, text="Action")
        actions.grid(row=len(CHANNELS) + 3, column=0, columnspan=7, sticky="ew", padx=4, pady=(8, 4))
        ttk.Button(actions, text="Send Current MOVE_SAFE", command=self.on_send_current_pose).grid(
            row=0, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(actions, text="OPEN_GRIPPER", command=lambda: self.perform_gripper_action("open_deg", "OPEN_GRIPPER")).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(actions, text="CLOSE_SOFT", command=lambda: self.perform_gripper_action("close_soft_deg", "CLOSE_SOFT")).grid(
            row=0, column=2, sticky="ew", padx=4, pady=4
        )
        ttk.Button(actions, text="CLOSE_FULL", command=lambda: self.perform_gripper_action("close_full_deg", "CLOSE_FULL")).grid(
            row=0, column=3, sticky="ew", padx=4, pady=4
        )
        ttk.Button(actions, text="STOP", command=self.on_stop).grid(row=0, column=4, sticky="ew", padx=4, pady=4)
        for col in range(5):
            actions.columnconfigure(col, weight=1)

        grip = ttk.LabelFrame(frame, text="Gripper Calibration")
        grip.grid(row=len(CHANNELS) + 4, column=0, columnspan=7, sticky="ew", padx=4, pady=(8, 4))
        ttk.Label(grip, text="OPEN").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(grip, textvariable=self.gripper_open_var).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(grip, text="Set OPEN From Current", command=lambda: self.set_gripper_calibration_from_current("open_deg")).grid(
            row=0, column=2, sticky="ew", padx=4, pady=4
        )
        ttk.Label(grip, text="CLOSE_SOFT").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(grip, textvariable=self.gripper_soft_var).grid(row=1, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(grip, text="Set CLOSE_SOFT From Current", command=lambda: self.set_gripper_calibration_from_current("close_soft_deg")).grid(
            row=1, column=2, sticky="ew", padx=4, pady=4
        )
        ttk.Label(grip, text="CLOSE_FULL").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(grip, textvariable=self.gripper_full_var).grid(row=2, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(grip, text="Set CLOSE_FULL From Current", command=lambda: self.set_gripper_calibration_from_current("close_full_deg")).grid(
            row=2, column=2, sticky="ew", padx=4, pady=4
        )
        ttk.Button(grip, text="Save Gripper Calibration", command=self.save_gripper_calibration).grid(
            row=0, column=3, rowspan=3, sticky="ns", padx=8, pady=4
        )
        for col in range(4):
            grip.columnconfigure(col, weight=1)

    def _build_pose_panel(self, parent: ttk.Panedwindow) -> None:
        frame = ttk.LabelFrame(parent, text="Calibration Workflow")
        parent.add(frame, weight=3)
        self.pose_button_frame = frame
        notebook = ttk.Notebook(frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.pose_notebook = notebook

        taught_tab = ttk.Frame(notebook)
        ik_ref_tab = ttk.Frame(notebook)
        direction_tab = ttk.Frame(notebook)
        reporting_tab = ttk.Frame(notebook)
        notebook.add(taught_tab, text="Taught Poses")
        notebook.add(ik_ref_tab, text="IK Reference Capture")
        notebook.add(direction_tab, text="Servo Direction Observation")
        notebook.add(reporting_tab, text="Reporting / YAML")

        ttk.Label(taught_tab, text="Pose").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(taught_tab, text="Load").grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(taught_tab, text="Go").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Label(taught_tab, text="Save").grid(row=0, column=3, sticky="w", padx=4, pady=4)

        for row, pose_name in enumerate(POSE_NAMES, start=1):
            ttk.Label(taught_tab, text=pose_name).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            ttk.Button(taught_tab, text=f"Load {pose_name}", command=lambda name=pose_name: self.load_pose(name)).grid(
                row=row, column=1, sticky="ew", padx=4, pady=3
            )
            ttk.Button(taught_tab, text=f"Go {pose_name}", command=lambda name=pose_name: self.go_pose(name)).grid(
                row=row, column=2, sticky="ew", padx=4, pady=3
            )
            ttk.Button(taught_tab, text=f"Save {pose_name}", command=lambda name=pose_name: self.save_pose(name)).grid(
                row=row, column=3, sticky="ew", padx=4, pady=3
            )

        custom = ttk.LabelFrame(taught_tab, text="Custom Pose")
        custom.grid(row=len(POSE_NAMES) + 1, column=0, columnspan=4, sticky="ew", padx=4, pady=(8, 4))
        ttk.Entry(custom, textvariable=self.custom_pose_name_var, width=22).pack(side=tk.LEFT, padx=(4, 6), pady=4)
        ttk.Button(custom, text="Save Custom", command=self.save_custom_pose).pack(side=tk.LEFT, padx=(0, 6), pady=4)

        ik_ref = ttk.LabelFrame(ik_ref_tab, text="IK Reference Capture")
        ik_ref.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        ttk.Label(ik_ref, text="Reference Name").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(ik_ref, textvariable=self.ik_ref_name_var, width=24).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(ik_ref, text="Z Mode").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Combobox(
            ik_ref,
            textvariable=self.ik_ref_z_mode_var,
            values=("safe_hover", "pre_pick", "pick", "lift", "custom"),
            state="readonly",
            width=12,
        ).grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(ik_ref, text="board_x_cm").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(ik_ref, textvariable=self.ik_ref_board_x_var, width=12).grid(row=1, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(ik_ref, text="board_y_cm").grid(row=1, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(ik_ref, textvariable=self.ik_ref_board_y_var, width=12).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(ik_ref, text="z_m").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(ik_ref, textvariable=self.ik_ref_z_var, width=12).grid(row=2, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(ik_ref, text="Object/Class Note").grid(row=2, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(ik_ref, textvariable=self.ik_ref_object_class_var, width=18).grid(row=2, column=3, sticky="ew", padx=4, pady=4)

        ttk.Label(ik_ref, text="Gripper State").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(
            ik_ref,
            textvariable=self.ik_ref_gripper_state_var,
            values=("open", "close", "half_open", "unknown"),
            state="readonly",
            width=14,
        ).grid(row=3, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(ik_ref, text="Comment").grid(row=3, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(ik_ref, textvariable=self.ik_ref_comment_var, width=24).grid(
            row=3, column=3, sticky="ew", padx=4, pady=4
        )

        ttk.Label(ik_ref, text="Validation").grid(row=4, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(ik_ref, textvariable=self.ik_ref_validation_var).grid(row=4, column=1, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(
            ik_ref,
            text="Allow out-of-board save",
            variable=self.ik_ref_out_of_board_allowed_var,
            command=self._refresh_ik_ref_preview,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        ttk.Label(ik_ref, text="robot_x_m").grid(row=4, column=2, sticky="w", padx=4, pady=4)
        ttk.Label(ik_ref, textvariable=self.ik_ref_robot_x_var).grid(row=4, column=3, sticky="w", padx=4, pady=4)
        ttk.Label(ik_ref, text="robot_y_m").grid(row=5, column=2, sticky="w", padx=4, pady=4)
        ttk.Label(ik_ref, textvariable=self.ik_ref_robot_y_var).grid(row=5, column=3, sticky="w", padx=4, pady=4)
        ttk.Label(ik_ref, text="Active IK Ref").grid(row=6, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(ik_ref, textvariable=self.ik_ref_active_name_var).grid(row=6, column=1, sticky="w", padx=4, pady=4)

        button_row1 = ttk.Frame(ik_ref)
        button_row1.grid(row=7, column=0, columnspan=4, sticky="ew", padx=4, pady=(6, 2))
        ttk.Button(button_row1, text="Use Last YOLO Board Target", command=self.use_last_yolo_board_target).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_row1, text="Compute Robot XY", command=self.compute_robot_xy).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_row1, text="Save IK Reference From Current Pose", command=self.save_ik_reference_from_current_pose).pack(side=tk.LEFT)

        button_row2 = ttk.Frame(ik_ref)
        button_row2.grid(row=8, column=0, columnspan=4, sticky="ew", padx=4, pady=(2, 4))
        ttk.Button(button_row2, text="Load IK Reference", command=self.load_ik_reference).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_row2, text="Go IK Reference Servo Pose", command=self.go_ik_reference_servo_pose).pack(side=tk.LEFT)

        self.ik_ref_preview_widget = scrolledtext.ScrolledText(ik_ref, wrap=tk.WORD, height=14)
        self.ik_ref_preview_widget.grid(row=9, column=0, columnspan=4, sticky="nsew", padx=4, pady=(4, 4))
        self.ik_ref_preview_widget.configure(state=tk.DISABLED)

        direction_frame = ttk.LabelFrame(direction_tab, text="Servo Direction Observation")
        direction_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        headers = ("Channel", "Function", "+ Motion", "+ From", "+ To", "- Motion", "- From", "- To", "Status", "Note")
        for col, label in enumerate(headers):
            ttk.Label(direction_frame, text=label).grid(row=0, column=col, sticky="w", padx=4, pady=4)
        for row, channel in enumerate(CHANNELS, start=1):
            ttk.Label(direction_frame, text=CHANNEL_META[channel]["label"]).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            ttk.Label(direction_frame, text=CHANNEL_META[channel]["joint"]).grid(row=row, column=1, sticky="w", padx=4, pady=3)
            ttk.Combobox(direction_frame, textvariable=self.direction_plus_vars[channel], values=OBSERVATION_OPTIONS[channel], state="readonly", width=18).grid(
                row=row, column=2, sticky="ew", padx=4, pady=3
            )
            ttk.Entry(direction_frame, textvariable=self.direction_plus_from_vars[channel], width=6).grid(row=row, column=3, sticky="ew", padx=4, pady=3)
            ttk.Entry(direction_frame, textvariable=self.direction_plus_to_vars[channel], width=6).grid(row=row, column=4, sticky="ew", padx=4, pady=3)
            ttk.Combobox(direction_frame, textvariable=self.direction_minus_vars[channel], values=OBSERVATION_OPTIONS[channel], state="readonly", width=18).grid(
                row=row, column=5, sticky="ew", padx=4, pady=3
            )
            ttk.Entry(direction_frame, textvariable=self.direction_minus_from_vars[channel], width=6).grid(row=row, column=6, sticky="ew", padx=4, pady=3)
            ttk.Entry(direction_frame, textvariable=self.direction_minus_to_vars[channel], width=6).grid(row=row, column=7, sticky="ew", padx=4, pady=3)
            ttk.Combobox(direction_frame, textvariable=self.direction_status_vars[channel], values=OBSERVATION_STATUS_OPTIONS, state="readonly", width=18).grid(
                row=row, column=8, sticky="ew", padx=4, pady=3
            )
            ttk.Entry(direction_frame, textvariable=self.direction_note_vars[channel], width=28).grid(row=row, column=9, sticky="ew", padx=4, pady=3)

        direction_buttons = ttk.Frame(direction_frame)
        direction_buttons.grid(row=len(CHANNELS) + 1, column=0, columnspan=10, sticky="ew", padx=4, pady=(8, 4))
        ttk.Button(direction_buttons, text="Save Direction Observations", command=self.save_direction_observations).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(direction_buttons, text="Reload Direction Observations", command=self.reload_direction_observations).pack(side=tk.LEFT)

        report_frame = ttk.Frame(reporting_tab)
        report_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        ttk.Label(report_frame, text="Current Pose Reporting / YAML Preview").pack(anchor="w", padx=4, pady=(0, 4))
        ttk.Label(report_frame, text="YAML preview stays here so the bottom log panel remains visible.").pack(
            anchor="w", padx=4, pady=(0, 6)
        )
        self.yaml_widget = scrolledtext.ScrolledText(report_frame, wrap=tk.NONE, height=24)
        self.yaml_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.yaml_widget.configure(state=tk.DISABLED)

        button_bar = ttk.Frame(report_frame)
        button_bar.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(button_bar, text="Copy YAML to Clipboard", command=self.copy_yaml_to_clipboard).pack(side=tk.LEFT, padx=(0, 6))

        taught_tab.columnconfigure(0, weight=1)
        taught_tab.columnconfigure(1, weight=1)
        taught_tab.columnconfigure(2, weight=1)
        taught_tab.columnconfigure(3, weight=1)
        ik_ref.columnconfigure(1, weight=1)
        ik_ref.columnconfigure(3, weight=1)
        ik_ref.rowconfigure(9, weight=1)

    def _build_bottom_panels(self, parent: ttk.Panedwindow) -> None:
        log_frame = ttk.LabelFrame(parent, text="Log / Report")
        viz_frame = ttk.LabelFrame(parent, text="Robot Pose Visualization / Telemetry")
        parent.add(log_frame, weight=3)
        parent.add(viz_frame, weight=4)

        self.log_widget = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=16)
        self.log_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.log_widget.configure(state=tk.DISABLED)

        log_buttons = ttk.Frame(log_frame)
        log_buttons.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(log_buttons, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(log_buttons, text="Copy Log", command=self.copy_log_to_clipboard).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(log_buttons, text="Export Log", command=self.export_session_log).pack(side=tk.LEFT)

        viz_top = ttk.Frame(viz_frame)
        viz_top.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        viz_bottom = ttk.Frame(viz_frame)
        viz_bottom.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        top_view_frame = ttk.LabelFrame(viz_top, text="Top View")
        side_view_frame = ttk.LabelFrame(viz_top, text="Side View")
        top_view_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        side_view_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self.top_view_canvas = tk.Canvas(top_view_frame, width=360, height=260, background="#ffffff", highlightthickness=1)
        self.top_view_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.side_view_canvas = tk.Canvas(side_view_frame, width=360, height=260, background="#ffffff", highlightthickness=1)
        self.side_view_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        ttk.Label(
            viz_bottom,
            text="Approximate pose preview based on current servo values and configured link lengths.",
            wraplength=700,
        ).pack(anchor="w", padx=4, pady=(0, 4))

        self.telemetry_widget = scrolledtext.ScrolledText(viz_bottom, wrap=tk.WORD, height=12)
        self.telemetry_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.telemetry_widget.configure(state=tk.DISABLED)

    def _refresh_servo_limit_labels(self) -> None:
        for channel in CHANNELS:
            lo, hi = self.limit_cache[channel]
            self.limit_vars[channel].set(f"{lo}..{hi}")
            scale = self.scale_widgets.get(channel)
            if scale is not None:
                scale.configure(from_=lo, to=hi)
                current = int(self.channel_vars[channel].get())
                bounded = max(lo, min(hi, current))
                if bounded != current:
                    self.channel_vars[channel].set(bounded)
        self._update_command_preview()

    def _refresh_gripper_widgets(self) -> None:
        self.gripper_open_var.set(int(self.gripper_calibration["open_deg"]))
        self.gripper_soft_var.set(int(self.gripper_calibration["close_soft_deg"]))
        self.gripper_full_var.set(int(self.gripper_calibration["close_full_deg"]))

    def _refresh_yaml_preview(self) -> None:
        current_pose = {
            "mode": self.mode_var.get(),
            "active_loaded_pose": self.active_loaded_pose_var.get(),
            "latest_status": self._current_status_angles(),
            "current_move_safe": self.current_command_var.get(),
            "staged_pose": self._current_angles(),
            "saved_output": self.taught_payload,
        }
        rendered = yaml.safe_dump(current_pose, sort_keys=False)
        if self.yaml_widget is not None:
            self.yaml_widget.configure(state=tk.NORMAL)
            self.yaml_widget.delete("1.0", tk.END)
            self.yaml_widget.insert("1.0", rendered)
            self.yaml_widget.configure(state=tk.DISABLED)

    def _parse_float_field(self, label: str, value: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric") from exc

    def _current_ik_reference_name(self) -> str:
        raw_name = self.ik_ref_name_var.get().strip().upper()
        return raw_name.replace(" ", "_")

    def _compute_robot_xy(self) -> tuple[float, float]:
        board_x = self._parse_float_field("board_x_cm", self.ik_ref_board_x_var.get().strip())
        board_y = self._parse_float_field("board_y_cm", self.ik_ref_board_y_var.get().strip())
        cfg = self.transform_cfg["board_to_robot"]
        base_x = float(cfg["robot_base_x_board_cm"])
        base_y = float(cfg["robot_base_y_board_cm"])
        yaw_rad = math.radians(float(cfg.get("robot_yaw_offset_deg", 0.0)))
        dx = board_x - base_x
        dy = board_y - base_y
        robot_x_cm = math.cos(yaw_rad) * dx - math.sin(yaw_rad) * dy
        robot_y_cm = math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
        return robot_x_cm / 100.0, robot_y_cm / 100.0

    def _validate_ik_reference_inputs(self) -> tuple[bool, list[str]]:
        errors: list[str] = []
        try:
            board_x = self._parse_float_field("board_x_cm", self.ik_ref_board_x_var.get().strip())
            board_y = self._parse_float_field("board_y_cm", self.ik_ref_board_y_var.get().strip())
            z_m = self._parse_float_field("z_m", self.ik_ref_z_var.get().strip())
        except ValueError as exc:
            return False, [str(exc)]

        if z_m <= 0.0:
            errors.append("z_m must be > 0")

        out_of_board = not (0.0 <= board_x <= self.board_width_cm and 0.0 <= board_y <= self.board_height_cm)
        if out_of_board and not self.ik_ref_out_of_board_allowed_var.get():
            errors.append(
                f"board target ({board_x:.2f}, {board_y:.2f}) cm is outside measured board "
                f"[0,{self.board_width_cm:.1f}] x [0,{self.board_height_cm:.1f}] cm"
            )
        return not errors, errors

    def _current_ik_reference_preview_payload(self) -> dict[str, Any]:
        valid, errors = self._validate_ik_reference_inputs()
        return {
            "name": self._current_ik_reference_name() or "(unnamed)",
            "active_ik_reference": self.ik_ref_active_name_var.get(),
            "validation": {
                "valid": valid,
                "errors": errors,
                "board_limits_cm": {"x_max": self.board_width_cm, "y_max": self.board_height_cm},
                "out_of_board_allowed": bool(self.ik_ref_out_of_board_allowed_var.get()),
            },
            "board": {
                "x_cm": self.ik_ref_board_x_var.get().strip(),
                "y_cm": self.ik_ref_board_y_var.get().strip(),
            },
            "robot": {
                "x_m": self.ik_ref_robot_x_var.get(),
                "y_m": self.ik_ref_robot_y_var.get(),
                "z_m": self.ik_ref_z_var.get().strip(),
            },
            "z_mode": self.ik_ref_z_mode_var.get().strip(),
            "gripper_state": {
                "value": self.ik_ref_gripper_state_var.get().strip(),
            },
            "notes": {
                "object_class": self.ik_ref_object_class_var.get().strip(),
                "comment": self.ik_ref_comment_var.get().strip(),
            },
            "servo": self._current_angles(),
            "mode": self.mode_var.get(),
            "direction_warning": self.safety_warning_var.get(),
        }

    def _refresh_ik_ref_preview(self) -> None:
        if self.ik_ref_preview_widget is None:
            return
        preview_payload = self._current_ik_reference_preview_payload()
        validation = preview_payload["validation"]
        if validation["valid"]:
            self.ik_ref_validation_var.set("VALID")
        else:
            self.ik_ref_validation_var.set("INVALID: " + "; ".join(validation["errors"]))
        rendered = yaml.safe_dump(preview_payload, sort_keys=False)
        self.ik_ref_preview_widget.configure(state=tk.NORMAL)
        self.ik_ref_preview_widget.delete("1.0", tk.END)
        self.ik_ref_preview_widget.insert("1.0", rendered)
        self.ik_ref_preview_widget.configure(state=tk.DISABLED)

    def _estimate_front_servo_deg(self) -> float:
        candidates = []
        for pose_name in ("HOVER_PICK_TEST", "PICK_TEST", "LIFT_TEST", "HOME_SAFE"):
            pose = self.pose_cache.get(pose_name)
            if pose and "ch1" in pose:
                candidates.append(float(pose["ch1"]))
        if candidates:
            return sum(candidates) / len(candidates)
        return 80.0

    def _inverse_joint_deg(self, channel: str, servo_deg: float) -> float:
        model_key_map = {
            "ch1": "ch1_base_yaw",
            "ch2": "ch2_shoulder_pitch",
            "ch3": "ch3_elbow_pitch",
            "ch4": "ch4_wrist_rotate",
            "ch5": "ch5_wrist_pitch",
        }
        if channel == "ch1":
            front = self._estimate_front_servo_deg()
            direction = float(self.kin_cfg.get("servo_model", {}).get("ch1_base_yaw", {}).get("direction", 1))
            offset = float(self.kin_cfg.get("servo_model", {}).get("ch1_base_yaw", {}).get("offset_deg", 0))
            if direction == 0:
                return 0.0
            return (servo_deg - front - offset) / direction
        model = self.kin_cfg.get("servo_model", {}).get(model_key_map[channel], {})
        zero = float(model.get("zero_reference_deg", 90))
        direction = float(model.get("direction", 1))
        offset = float(model.get("offset_deg", 0))
        if direction == 0:
            return 0.0
        return (servo_deg - zero - offset) / direction

    def _approx_pose_geometry(self) -> dict[str, Any]:
        angles = self._current_angles()
        links = self.kin_cfg.get("links", {})
        l1 = float(links.get("shoulder_to_elbow_m", 0.11716))
        l2 = float(links.get("elbow_to_wrist_pitch_m", 0.12683))
        tool = float(links.get("wrist_pitch_to_tcp_direct_m", 0.12))
        shoulder_z = float(links.get("base_to_shoulder_m", 0.03798))

        ch1_rel = self._inverse_joint_deg("ch1", float(angles["ch1"]))
        ch2_rel = self._inverse_joint_deg("ch2", float(angles["ch2"]))
        ch3_rel = self._inverse_joint_deg("ch3", float(angles["ch3"]))
        ch5_rel = self._inverse_joint_deg("ch5", float(angles["ch5"]))

        shoulder_rad = math.radians(ch2_rel)
        elbow_rad = math.radians(ch3_rel)
        wrist_rad = math.radians(ch5_rel)

        elbow_x = l1 * math.cos(shoulder_rad)
        elbow_z = shoulder_z + l1 * math.sin(shoulder_rad)
        wrist_x = elbow_x + l2 * math.cos(shoulder_rad + elbow_rad)
        wrist_z = elbow_z + l2 * math.sin(shoulder_rad + elbow_rad)
        tcp_x = wrist_x + tool * math.cos(shoulder_rad + elbow_rad + wrist_rad)
        tcp_z = wrist_z + tool * math.sin(shoulder_rad + elbow_rad + wrist_rad)

        return {
            "angles": angles,
            "ch1_relative_deg": ch1_rel,
            "shoulder_deg": ch2_rel,
            "elbow_deg": ch3_rel,
            "wrist_deg": ch5_rel,
            "shoulder_point": (0.0, shoulder_z),
            "elbow_point": (elbow_x, elbow_z),
            "wrist_point": (wrist_x, wrist_z),
            "tcp_point": (tcp_x, tcp_z),
            "max_planar_reach_m": max(0.2, l1 + l2 + tool),
            "max_height_m": max(0.25, shoulder_z + l1 + l2 + tool),
        }

    def _refresh_telemetry_preview(self) -> None:
        if self.telemetry_widget is None:
            return
        slider = self._current_angles()
        status = self._current_status_angles()
        delta = {}
        for channel in CHANNELS:
            status_value = None if status is None else status.get(channel)
            delta[channel] = None if status_value is None else slider[channel] - status_value
        pose = self._approx_pose_geometry()
        payload = {
            "active_loaded_pose": self.active_loaded_pose_var.get(),
            "active_ik_reference": self.ik_ref_active_name_var.get(),
            "safety_warning": self.safety_warning_var.get(),
            "mode": self.mode_var.get(),
            "current_move_safe": self.current_command_var.get(),
            "slider_pose": slider,
            "status_pose": status,
            "delta_slider_minus_status": delta,
            "ik_reference_target": {
                "board_x_cm": self.ik_ref_board_x_var.get().strip(),
                "board_y_cm": self.ik_ref_board_y_var.get().strip(),
                "robot_x_m": self.ik_ref_robot_x_var.get(),
                "robot_y_m": self.ik_ref_robot_y_var.get(),
                "z_m": self.ik_ref_z_var.get().strip(),
                "z_mode": self.ik_ref_z_mode_var.get().strip(),
                "gripper_state": self.ik_ref_gripper_state_var.get().strip(),
            },
            "estimated_pose": {
                "base_heading_relative_deg": round(pose["ch1_relative_deg"], 2),
                "estimated_tcp_x_m": round(pose["tcp_point"][0], 4),
                "estimated_tcp_z_m": round(pose["tcp_point"][1], 4),
            },
            "direction_observations": {
                "ch3": self.direction_observations_payload.get("observations", {}).get("ch3", {}),
            },
        }
        rendered = yaml.safe_dump(payload, sort_keys=False)
        self.telemetry_widget.configure(state=tk.NORMAL)
        self.telemetry_widget.delete("1.0", tk.END)
        self.telemetry_widget.insert("1.0", rendered)
        self.telemetry_widget.configure(state=tk.DISABLED)

    def _refresh_visualization(self) -> None:
        if self.top_view_canvas is None or self.side_view_canvas is None:
            return

        pose = self._approx_pose_geometry()
        target_x = None
        target_y = None
        try:
            if self.ik_ref_robot_x_var.get() not in {"", "-"}:
                target_x = float(self.ik_ref_robot_x_var.get())
            if self.ik_ref_robot_y_var.get() not in {"", "-"}:
                target_y = float(self.ik_ref_robot_y_var.get())
        except ValueError:
            target_x = None
            target_y = None

        canvas = self.top_view_canvas
        canvas.delete("all")
        w = max(int(canvas.winfo_width() or 360), 200)
        h = max(int(canvas.winfo_height() or 260), 160)
        ox = 40
        oy = h / 2
        scale = min((w - 80) / 0.36, (h - 40) / 0.30)
        canvas.create_text(8, 8, text="Top view: robot X -> right, robot Y+ -> up", anchor="nw")
        canvas.create_oval(ox - 6, oy - 6, ox + 6, oy + 6, fill="#222222")
        canvas.create_text(ox - 10, oy + 12, text="Base", anchor="ne")

        heading_rad = math.radians(pose["ch1_relative_deg"])
        arm_len = 0.16
        hx = ox + arm_len * scale * math.cos(heading_rad)
        hy = oy - arm_len * scale * math.sin(heading_rad)
        canvas.create_line(ox, oy, hx, hy, width=4, fill="#2c7be5", arrow=tk.LAST)
        canvas.create_text(hx + 4, hy - 4, text=f"CH1 {pose['ch1_relative_deg']:.1f} deg", anchor="sw")

        if target_x is not None and target_y is not None:
            tx = ox + target_x * scale
            ty = oy - target_y * scale
            canvas.create_oval(tx - 5, ty - 5, tx + 5, ty + 5, fill="#d9480f", outline="")
            canvas.create_text(tx + 8, ty - 8, text="IK ref target", anchor="sw", fill="#d9480f")

        canvas = self.side_view_canvas
        canvas.delete("all")
        w = max(int(canvas.winfo_width() or 360), 200)
        h = max(int(canvas.winfo_height() or 260), 160)
        ox = 36
        ground_y = h - 28
        scale = min((w - 60) / pose["max_planar_reach_m"], (h - 40) / pose["max_height_m"])
        canvas.create_text(8, 8, text="Side view: approximate CH2/CH3/CH5 chain", anchor="nw")
        canvas.create_line(0, ground_y, w, ground_y, fill="#888888")

        def pt(point: tuple[float, float]) -> tuple[float, float]:
            return ox + point[0] * scale, ground_y - point[1] * scale

        shoulder = pt(pose["shoulder_point"])
        elbow = pt(pose["elbow_point"])
        wrist = pt(pose["wrist_point"])
        tcp = pt(pose["tcp_point"])

        canvas.create_oval(shoulder[0] - 4, shoulder[1] - 4, shoulder[0] + 4, shoulder[1] + 4, fill="#222222")
        canvas.create_line(*shoulder, *elbow, width=4, fill="#2b8a3e")
        canvas.create_line(*elbow, *wrist, width=4, fill="#f08c00")
        canvas.create_line(*wrist, *tcp, width=4, fill="#7b2cbf")
        canvas.create_oval(tcp[0] - 4, tcp[1] - 4, tcp[0] + 4, tcp[1] + 4, fill="#c92a2a")
        canvas.create_text(tcp[0] + 6, tcp[1] - 6, text=f"TCP ~ ({pose['tcp_point'][0]:.3f}, {pose['tcp_point'][1]:.3f}) m", anchor="sw")

        try:
            target_z = float(self.ik_ref_z_var.get().strip())
        except ValueError:
            target_z = None
        if target_x is not None and target_z is not None:
            tx = ox + target_x * scale
            tz = ground_y - target_z * scale
            canvas.create_line(tx, ground_y, tx, tz, fill="#d9480f", dash=(4, 2))
            canvas.create_oval(tx - 4, tz - 4, tx + 4, tz + 4, fill="#d9480f", outline="")
            canvas.create_text(tx + 6, tz - 6, text="Target Z", anchor="sw", fill="#d9480f")

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        full = f"{timestamp} {message}"
        self.session_log_lines.append(full)
        if self.log_widget is not None:
            self.log_widget.configure(state=tk.NORMAL)
            self.log_widget.insert(tk.END, full + "\n")
            self.log_widget.see(tk.END)
            self.log_widget.configure(state=tk.DISABLED)

    def _set_connection_label(self, value: str) -> None:
        self.connection_var.set(value)
        self.mode_var.set("DRY RUN" if self.dry_run_mode else "LIVE")

    def _clamped_angles_from_dict(self, raw_angles: dict[str, int]) -> tuple[dict[str, int], list[str]]:
        clamped: dict[str, int] = {}
        notes: list[str] = []
        for channel in CHANNELS:
            raw = int(raw_angles[channel])
            lo, hi = self.limit_cache[channel]
            bounded = max(lo, min(hi, raw))
            clamped[channel] = bounded
            if bounded != raw:
                notes.append(f"{channel}: clamped {raw} -> {bounded} within [{lo}, {hi}]")
        return clamped, notes

    def _clamped_angles_from_ui(self) -> tuple[dict[str, int], list[str]]:
        return self._clamped_angles_from_dict(self._current_angles())

    def _apply_angles_to_ui(self, angles: dict[str, int]) -> None:
        for channel in CHANNELS:
            self.channel_vars[channel].set(int(angles[channel]))
        self._update_command_preview()

    def _build_move_safe_command(self, angles: dict[str, int]) -> str:
        return "MOVE_SAFE " + " ".join(str(angles[channel]) for channel in CHANNELS)

    def _serial_timeout(self) -> float:
        return float(self.servo_cfg.get("safety", {}).get("serial_timeout_sec", SERIAL_TIMEOUT_SEC))

    def _confirm_live_action(self, title: str, prompt: str) -> bool:
        if self.dry_run_mode or self.args.no_confirm:
            return True
        return bool(messagebox.askyesno(title, prompt))

    def _save_output_yaml(self) -> None:
        self.taught_payload.setdefault("metadata", {})
        self.taught_payload["metadata"]["updated_at"] = utc_now()
        save_yaml_file(self.output_path, self.taught_payload)

    def _save_ik_reference_yaml(self) -> None:
        self.ik_reference_payload.setdefault("metadata", {})
        self.ik_reference_payload["metadata"]["status"] = "ik_reference_samples"
        self.ik_reference_payload["metadata"]["note"] = (
            "Human-taught servo poses paired with measured board/robot coordinates for IK calibration."
        )
        self.ik_reference_payload["metadata"]["updated_at"] = utc_now()
        save_yaml_file(self.ik_reference_output_path, self.ik_reference_payload)

    def _save_servo_cfg(self) -> None:
        save_yaml_file(Path(self.args.servo_config), self.servo_cfg)

    def _observation_motion_block(
        self,
        from_var: tk.StringVar,
        to_var: tk.StringVar,
        observed_var: tk.StringVar,
    ) -> dict[str, Any]:
        block: dict[str, Any] = {
            "observed": observed_var.get().strip() or "unknown",
        }
        if from_var.get().strip():
            try:
                block["from_deg"] = int(float(from_var.get().strip()))
            except ValueError:
                block["from_deg"] = from_var.get().strip()
        if to_var.get().strip():
            try:
                block["to_deg"] = int(float(to_var.get().strip()))
            except ValueError:
                block["to_deg"] = to_var.get().strip()
        return block

    def save_direction_observations(self) -> None:
        observations: dict[str, Any] = {}
        for channel in CHANNELS:
            observations[channel] = {
                "function": CHANNEL_META[channel]["joint"],
                "plus_motion": self._observation_motion_block(
                    self.direction_plus_from_vars[channel],
                    self.direction_plus_to_vars[channel],
                    self.direction_plus_vars[channel],
                ),
                "minus_motion": self._observation_motion_block(
                    self.direction_minus_from_vars[channel],
                    self.direction_minus_to_vars[channel],
                    self.direction_minus_vars[channel],
                ),
                "status": self.direction_status_vars[channel].get().strip() or "unknown",
                "note": self.direction_note_vars[channel].get().strip(),
            }
        self.direction_observations_payload = {
            "metadata": {
                "status": "servo_direction_observations",
                "updated_at": utc_now(),
                "note": "Physical observations only. Do not directly treat as IK calibration until verified.",
            },
            "observations": observations,
        }
        save_yaml_file(self.direction_observations_output_path, self.direction_observations_payload)
        self._update_direction_observation_warning()
        self._refresh_telemetry_preview()
        self._refresh_ik_ref_preview()
        self._log(f"[CONFIG] Saved servo direction observations to {self.direction_observations_output_path}")

    def reload_direction_observations(self) -> None:
        self.direction_observations_payload = self._load_or_init_direction_observations_output()
        self._load_direction_observations_into_vars()
        self._update_direction_observation_warning()
        self._refresh_telemetry_preview()
        self._refresh_ik_ref_preview()
        self._log("[CONFIG] Servo direction observations reloaded")

    def _sanitize_serial_line(self, raw: bytes) -> tuple[str | None, bool]:
        decoded = raw.decode("utf-8", errors="replace").strip()
        if not decoded:
            return None, False
        sanitized = "".join(ch for ch in decoded if ch in string.printable and ch not in "\r\n")
        if not sanitized:
            return decoded, True
        if "\ufffd" in decoded or any(ord(ch) < 32 and ch not in "\t " for ch in decoded):
            return sanitized, True
        if sanitized.startswith(BOOT_NOISE_PREFIXES):
            return sanitized, True
        if not sanitized.startswith(EXPECTED_SERIAL_PREFIXES):
            return sanitized, True
        return sanitized, False

    def on_connect(self) -> None:
        if self.dry_run_mode:
            self._set_connection_label("DRY RUN")
            self._log("[SERIAL] Dry-run mode active; no serial port will be opened.")
            return
        if self.serial is not None:
            self._log("[SERIAL] Already connected.")
            return
        try:
            serial_mod = load_serial_module()
            self.serial = serial_mod.Serial(
                port=self.port_var.get().strip(),
                baudrate=int(self.baud_var.get()),
                timeout=self._serial_timeout(),
                write_timeout=self._serial_timeout(),
            )
            self._log("[SERIAL] Waiting 2.0 seconds for ESP32 auto-reset.")
            time.sleep(CONNECT_RESET_WAIT_SEC)
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            self.serial.write(b"\n")
            self.serial.flush()
            time.sleep(POST_CONNECT_FLUSH_SEC)
            self.serial.reset_input_buffer()
        except Exception as exc:  # pragma: no cover - serial hardware path
            self._log(f"[SERIAL][ERROR] Failed to connect: {exc}")
            self._set_connection_label("DISCONNECTED")
            if self.serial is not None:
                try:
                    self.serial.close()
                except Exception:
                    pass
                self.serial = None
            return
        self._set_connection_label("CONNECTED")
        self._log(f"[SERIAL] Connected to {self.port_var.get().strip()} @ {int(self.baud_var.get())} baud.")

    def on_disconnect(self) -> None:
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
            self.serial = None
        self._set_connection_label("DRY RUN" if self.dry_run_mode else "DISCONNECTED")
        self._log("[SERIAL] Disconnected.")

    def on_flush_serial(self) -> None:
        if self.dry_run_mode:
            self._log("[SERIAL] DRY RUN serial flush requested.")
            return
        if self.serial is None:
            self._log("[SERIAL][ERROR] Not connected.")
            return
        try:
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
        except Exception as exc:  # pragma: no cover
            self._log(f"[SERIAL][ERROR] Flush failed: {exc}")
            return
        self._log("[SERIAL] Input/output buffers flushed.")

    def queue_serial_action(self, action: SerialAction) -> None:
        if self.worker_busy:
            self._log("[SERIAL][ERROR] Another command is already in progress.")
            return

        if action.expect_motion:
            preview = action.command
            self._log("[SAFETY] HOME_SAFE is validated for idle/manual testing only.")
            self._log(f"[SERIAL] Final command preview: {preview}")
            if not self._confirm_live_action("Confirm Motion", f"Send {preview}?"):
                self._log(f"[SERIAL] {action.user_label or action.command} cancelled by operator.")
                return

        if self.dry_run_mode:
            self._log(f"[SERIAL] DRY RUN command: {action.command}")
            if action.apply_angles_on_done is not None:
                self._apply_angles_to_ui(action.apply_angles_on_done)
            return

        if self.serial is None:
            self._log("[SERIAL][ERROR] Not connected. Use Connect first.")
            return

        self.worker_busy = True
        self.pending_stop = False
        self.stop_sent_for_current_motion = False
        self.current_action = action
        self.worker_thread = threading.Thread(target=self._worker_send_action, args=(action,), daemon=True)
        self.worker_thread.start()

    def _worker_send_action(self, action: SerialAction) -> None:
        assert self.serial is not None
        ser = self.serial
        try:
            self.ui_queue.put(("log", f"[SERIAL] Sending: {action.command}"))
            ser.reset_input_buffer()
            ser.write((action.command + "\n").encode("ascii"))
            ser.flush()

            lines: list[str] = []
            deadline = time.time() + (30.0 if action.expect_motion else max(float(getattr(ser, "timeout", 1.0) or 1.0), 1.0))
            while time.time() < deadline:
                if action.expect_motion and self.pending_stop and not self.stop_sent_for_current_motion:
                    ser.write(b"STOP\n")
                    ser.flush()
                    self.stop_sent_for_current_motion = True
                    self.ui_queue.put(("log", "[SERIAL] STOP sent during active motion."))

                raw = ser.readline()
                if not raw:
                    continue
                line, boot_noise = self._sanitize_serial_line(raw)
                if line is None:
                    continue
                if boot_noise:
                    self.ui_queue.put(("log", f"[SERIAL][BOOT_NOISE] {line}"))
                    continue

                lines.append(line)
                self.ui_queue.put(("log", f"[SERIAL] {line}"))

                if action.command == "PING" and line == "PONG":
                    break
                if action.command == "STATUS" and line.startswith("STATUS "):
                    self.ui_queue.put(("status", self._parse_status_line(line)))
                    break
                if action.command == "LIMITS" and line.startswith("LIMITS "):
                    break
                if action.command == "STOP" and line == "DONE STOP":
                    break
                if action.command == "HOME" and line == "DONE HOME":
                    break
                if action.command.startswith("MOVE_SAFE") and (line == "DONE MOVE_SAFE" or line == "DONE STOP"):
                    break
                if line.startswith("ERR "):
                    break

            if action.apply_angles_on_done is not None and ("DONE MOVE_SAFE" in lines or "DONE HOME" in lines):
                self.ui_queue.put(("apply_angles", deepcopy(action.apply_angles_on_done)))
        except Exception as exc:  # pragma: no cover - serial hardware path
            self.ui_queue.put(("log", f"[SERIAL][ERROR] {exc}"))
        finally:
            self.ui_queue.put(("worker_done", None))

    def _parse_status_line(self, line: str) -> dict[str, int]:
        parts = line.split()
        result: dict[str, int] = {}
        for token in parts:
            if token.startswith("CH") and "=" in token:
                channel_name, raw_value = token.split("=", 1)
                channel = "ch" + channel_name[2:]
                try:
                    result[channel] = int(raw_value)
                except ValueError:
                    continue
        return result

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                kind, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._log(str(payload))
            elif kind == "status":
                status_angles: dict[str, int] = payload
                for channel in CHANNELS:
                    if channel in status_angles:
                        self.status_vars[channel].set(str(status_angles[channel]))
                self.latest_status_var.set(" ".join(f"{channel.upper()}={status_angles.get(channel, '-')}" for channel in CHANNELS))
                self._log("[SERIAL] STATUS parsed into GUI status fields.")
                self._refresh_yaml_preview()
                self._refresh_telemetry_preview()
                self._refresh_visualization()
            elif kind == "apply_angles":
                self._apply_angles_to_ui(payload)
            elif kind == "worker_done":
                self.worker_busy = False
                self.pending_stop = False
                self.stop_sent_for_current_motion = False
                self.current_action = None
        self.root.after(100, self._drain_ui_queue)

    def _update_command_preview(self) -> None:
        move_angles, clamp_notes = self._clamped_angles_from_ui()
        for note in clamp_notes:
            self._log(f"[SAFETY] {note}")
        command = self._build_move_safe_command(move_angles)
        self.current_command_var.set(command)
        self._refresh_yaml_preview()
        self._refresh_ik_ref_preview()
        self._refresh_telemetry_preview()
        self._refresh_visualization()

    def on_slider_change(self, _channel: str) -> None:
        self.active_loaded_pose_var.set("(manual sliders)")
        self._update_command_preview()

    def jog_channel(self, channel: str, direction: int) -> None:
        step = max(1, int(self.step_var.get()))
        current = int(self.channel_vars[channel].get())
        lo, hi = self.limit_cache[channel]
        bounded = max(lo, min(hi, current + (step * direction)))
        self.channel_vars[channel].set(bounded)
        self.active_loaded_pose_var.set("(manual sliders)")
        self._update_command_preview()

    def set_step(self, value: int) -> None:
        self.step_var.set(int(value))
        self._log(f"[CONFIG] Step size set to {int(value)} deg.")

    def _on_step_changed(self) -> None:
        try:
            value = int(self.step_var.get())
        except Exception:
            value = DEFAULT_STEP
        if value <= 0:
            value = DEFAULT_STEP
            self.step_var.set(value)
        self._log(f"[CONFIG] Step size set to {value} deg.")

    def use_gripper_quick(self, key: str) -> None:
        self.channel_vars["ch6"].set(int(self.gripper_calibration[key]))
        self.active_loaded_pose_var.set("(manual sliders)")
        self._update_command_preview()

    def perform_gripper_action(self, key: str, label: str) -> None:
        self.use_gripper_quick(key)
        self._log(f"[POSE] {label} uses configured CH6={self.gripper_calibration[key]}.")
        self.on_send_current_pose()

    def set_gripper_calibration_from_current(self, key: str) -> None:
        current = int(self.channel_vars["ch6"].get())
        self.gripper_calibration[key] = current
        self._refresh_gripper_widgets()
        self._log(f"[CONFIG] Gripper calibration {key} set from current CH6={current}.")

    def save_gripper_calibration(self) -> None:
        self.servo_cfg["gripper_calibration"] = {
            "open_deg": int(self.gripper_calibration["open_deg"]),
            "close_soft_deg": int(self.gripper_calibration["close_soft_deg"]),
            "close_full_deg": int(self.gripper_calibration["close_full_deg"]),
            "source": "gui_taught_live",
        }
        self._save_servo_cfg()
        self._reload_all_configs()
        self._log("[CONFIG] Saved gripper calibration to servo_config.yaml")

    def set_limit_from_current(self, channel: str, which: str) -> None:
        current = int(self.channel_vars[channel].get())
        lo, hi = self.limit_cache[channel]
        if which == "min":
            lo = min(current, hi)
        else:
            hi = max(current, lo)
        self.limit_cache[channel] = (lo, hi)
        self.limit_vars[channel].set(f"{lo}..{hi}")
        scale = self.scale_widgets.get(channel)
        if scale is not None:
            scale.configure(from_=lo, to=hi)
        self._log(f"[CONFIG] {channel} {which}_angle_deg staged from current angle {current}.")
        self._update_command_preview()

    def save_servo_limits(self) -> None:
        servos = self.servo_cfg.setdefault("servos", {})
        for channel in CHANNELS:
            lo, hi = self.limit_cache[channel]
            servos.setdefault(channel, {})
            servos[channel]["min_angle_deg"] = int(lo)
            servos[channel]["max_angle_deg"] = int(hi)
        self._save_servo_cfg()
        self._reload_all_configs()
        self._log("[CONFIG] Saved servo limits to servo_config.yaml")

    def _pose_entry_for_save(self, name: str) -> dict[str, Any]:
        angles, clamp_notes = self._clamped_angles_from_ui()
        for note in clamp_notes:
            self._log(f"[SAFETY] {note}")
        entry: dict[str, Any] = {
            "description": POSE_DESCRIPTIONS.get(name, f"Human-taught pose {name}."),
            "status": "human_taught_live",
            "validated_for": "manual_testing_only",
            "autonomous_validated": False,
            "taught_at": utc_now(),
        }
        for channel in CHANNELS:
            entry[channel] = int(angles[channel])
        return entry

    def save_pose(self, pose_name: str) -> None:
        self.taught_payload.setdefault("poses", {})
        self.taught_payload["poses"][pose_name] = self._pose_entry_for_save(pose_name)
        self._save_output_yaml()
        self._reload_all_configs()
        self.active_loaded_pose_var.set(pose_name)
        self._refresh_yaml_preview()
        self._log(f"[POSE] Saved {pose_name} to {self.output_path}")

    def save_custom_pose(self) -> None:
        raw_name = self.custom_pose_name_var.get().strip().upper()
        if not raw_name:
            self._log("[POSE][ERROR] Enter a custom pose name first.")
            return
        safe_name = raw_name.replace(" ", "_")
        self.taught_payload.setdefault("poses", {})
        self.taught_payload["poses"][safe_name] = self._pose_entry_for_save(safe_name)
        self._save_output_yaml()
        self._reload_all_configs()
        self.active_loaded_pose_var.set(safe_name)
        self._refresh_yaml_preview()
        self._log(f"[POSE] Saved custom pose {safe_name} to {self.output_path}")

    def use_last_yolo_board_target(self) -> None:
        self._log("[IK_REF][WARN] No live YOLO-board target available in this GUI yet.")

    def compute_robot_xy(self) -> None:
        try:
            robot_x_m, robot_y_m = self._compute_robot_xy()
        except (KeyError, ValueError) as exc:
            self._log(f"[IK_REF][ERROR] {exc}")
            return
        self.ik_ref_robot_x_var.set(f"{robot_x_m:.4f}")
        self.ik_ref_robot_y_var.set(f"{robot_y_m:.4f}")
        self._refresh_ik_ref_preview()
        self._log(
            f"[IK_REF] board=({self.ik_ref_board_x_var.get().strip()}, {self.ik_ref_board_y_var.get().strip()}) cm "
            f"-> robot=({robot_x_m:.4f}, {robot_y_m:.4f}) m"
        )

    def save_ik_reference_from_current_pose(self) -> None:
        sample_name = self._current_ik_reference_name()
        if not sample_name:
            self._log("[IK_REF][ERROR] Reference name is required.")
            return

        valid, errors = self._validate_ik_reference_inputs()
        if not valid:
            for error in errors:
                self._log(f"[IK_REF][ERROR] {error}")
            return

        try:
            board_x = self._parse_float_field("board_x_cm", self.ik_ref_board_x_var.get().strip())
            board_y = self._parse_float_field("board_y_cm", self.ik_ref_board_y_var.get().strip())
            z_m = self._parse_float_field("z_m", self.ik_ref_z_var.get().strip())
            robot_x_m, robot_y_m = self._compute_robot_xy()
        except (KeyError, ValueError) as exc:
            self._log(f"[IK_REF][ERROR] {exc}")
            return

        self.ik_ref_robot_x_var.set(f"{robot_x_m:.4f}")
        self.ik_ref_robot_y_var.set(f"{robot_y_m:.4f}")

        self.ik_reference_payload.setdefault("samples", {})
        self.ik_reference_payload["samples"][sample_name] = {
            "status": "human_taught_live",
            "source": "gui_ik_reference_capture",
            "board": {
                "x_cm": round(board_x, 4),
                "y_cm": round(board_y, 4),
            },
            "robot": {
                "x_m": round(robot_x_m, 4),
                "y_m": round(robot_y_m, 4),
                "z_m": round(z_m, 4),
            },
            "z_mode": self.ik_ref_z_mode_var.get().strip(),
            "gripper_state": {
                "value": self.ik_ref_gripper_state_var.get().strip() or "unknown",
            },
            "servo": self._current_angles(),
            "notes": {
                "object_class": self.ik_ref_object_class_var.get().strip() or "custom",
                "comment": self.ik_ref_comment_var.get().strip(),
            },
            "autonomous_validated": False,
        }
        self._save_ik_reference_yaml()
        self.ik_ref_active_name_var.set(sample_name)
        self._refresh_ik_ref_preview()
        self._log(f"[IK_REF] Saved {sample_name} to {self.ik_reference_output_path}")

    def load_ik_reference(self) -> None:
        sample_name = self._current_ik_reference_name()
        if not sample_name:
            self._log("[IK_REF][ERROR] Reference name is required.")
            return
        sample = self.ik_reference_payload.get("samples", {}).get(sample_name)
        if not isinstance(sample, dict):
            self._log(f"[IK_REF][ERROR] Sample {sample_name} not found in {self.ik_reference_output_path}.")
            return

        board = sample.get("board", {})
        robot = sample.get("robot", {})
        servo = sample.get("servo", {})
        notes = sample.get("notes", {})
        gripper_state = sample.get("gripper_state", {})
        self.ik_ref_board_x_var.set(str(board.get("x_cm", "")))
        self.ik_ref_board_y_var.set(str(board.get("y_cm", "")))
        self.ik_ref_z_var.set(str(robot.get("z_m", "")))
        loaded_z_mode = str(sample.get("z_mode", "custom"))
        if loaded_z_mode == "hover":
            loaded_z_mode = "safe_hover"
            self._log("[IK_REF] Legacy z_mode 'hover' mapped to 'safe_hover'.")
        self.ik_ref_z_mode_var.set(loaded_z_mode)
        self.ik_ref_object_class_var.set(str(notes.get("object_class", "")))
        self.ik_ref_comment_var.set(str(notes.get("comment", "")))
        self.ik_ref_gripper_state_var.set(str(gripper_state.get("value", "unknown")))
        if "x_m" in robot:
            self.ik_ref_robot_x_var.set(f"{float(robot['x_m']):.4f}")
        if "y_m" in robot:
            self.ik_ref_robot_y_var.set(f"{float(robot['y_m']):.4f}")
        servo_angles = self._angles_from_block(servo)
        if servo_angles is not None:
            self._apply_angles_to_ui(servo_angles)
        self.ik_ref_active_name_var.set(sample_name)
        self.active_loaded_pose_var.set(f"IK_REF:{sample_name}")
        self._refresh_ik_ref_preview()
        self._log(f"[IK_REF] Loaded {sample_name} into fields and sliders only.")

    def go_ik_reference_servo_pose(self) -> None:
        sample_name = self._current_ik_reference_name()
        if not sample_name:
            self._log("[IK_REF][ERROR] Reference name is required.")
            return
        sample = self.ik_reference_payload.get("samples", {}).get(sample_name)
        if not isinstance(sample, dict):
            self._log(f"[IK_REF][ERROR] Sample {sample_name} not found in {self.ik_reference_output_path}.")
            return
        servo_angles = self._angles_from_block(sample.get("servo", {}))
        if servo_angles is None:
            self._log(f"[IK_REF][ERROR] Sample {sample_name} does not contain a complete servo pose.")
            return
        self.ik_ref_active_name_var.set(sample_name)
        self._move_to_pose_angles(f"IK_REF:{sample_name}", servo_angles)

    def load_pose(self, pose_name: str) -> None:
        angles = self.pose_cache.get(pose_name)
        if angles is None:
            self._log(f"[POSE][ERROR] Pose {pose_name} is not available in taught YAML or pose_config.")
            return
        self._apply_angles_to_ui(angles)
        self.active_loaded_pose_var.set(pose_name)
        self._log(f"[POSE] Loaded {pose_name} into sliders only.")

    def _move_to_pose_angles(self, pose_name: str, angles: dict[str, int]) -> None:
        clamped, clamp_notes = self._clamped_angles_from_dict(angles)
        for note in clamp_notes:
            self._log(f"[SAFETY] {note}")
        command = self._build_move_safe_command(clamped)
        self.active_loaded_pose_var.set(pose_name)
        self.queue_serial_action(
            SerialAction(command, expect_motion=True, apply_angles_on_done=clamped, user_label=f"Go {pose_name}")
        )

    def go_pose(self, pose_name: str) -> None:
        angles = self.pose_cache.get(pose_name)
        if angles is None:
            self._log(f"[POSE][ERROR] Pose {pose_name} is not available in taught YAML or pose_config.")
            return
        self._move_to_pose_angles(pose_name, angles)

    def reload_pose_cache(self) -> None:
        self._reload_all_configs()
        self._log("[CONFIG] Poses reloaded")

    def on_firmware_home(self) -> None:
        self._log("[POSE] Firmware HOME uses ESP32 hardcoded HOME, not YAML HOME_SAFE.")
        self.queue_serial_action(
            SerialAction("HOME", expect_motion=True, apply_angles_on_done=deepcopy(DEFAULT_FIRMWARE_HOME), user_label="Firmware HOME")
        )

    def on_send_current_pose(self) -> None:
        angles, clamp_notes = self._clamped_angles_from_ui()
        for note in clamp_notes:
            self._log(f"[SAFETY] {note}")
        command = self._build_move_safe_command(angles)
        self.queue_serial_action(SerialAction(command, expect_motion=True, apply_angles_on_done=angles, user_label="MOVE_SAFE"))

    def on_stop(self) -> None:
        if self.dry_run_mode:
            self._log("[SERIAL] DRY RUN command: STOP")
            return
        if self.worker_busy and self.current_action and self.current_action.expect_motion:
            self.pending_stop = True
            self._log("[SAFETY] STOP requested. It will be sent immediately by the active serial worker.")
            return
        self.queue_serial_action(SerialAction("STOP", user_label="STOP"))

    def copy_yaml_to_clipboard(self) -> None:
        if self.yaml_widget is None:
            return
        payload = self.yaml_widget.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(payload)
        self._log("[CONFIG] YAML copied to clipboard.")

    def clear_log(self) -> None:
        self.session_log_lines.clear()
        if self.log_widget is not None:
            self.log_widget.configure(state=tk.NORMAL)
            self.log_widget.delete("1.0", tk.END)
            self.log_widget.configure(state=tk.DISABLED)
        self._log("[CONFIG] Log cleared.")

    def copy_log_to_clipboard(self) -> None:
        payload = "\n".join(self.session_log_lines).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(payload)
        self._log("[CONFIG] Log copied to clipboard.")

    def export_session_log(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export Session Log",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        Path(path).write_text("\n".join(self.session_log_lines) + "\n", encoding="utf-8")
        self._log(f"[CONFIG] Session log exported to {path}.")

    def close(self) -> None:
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
            self.serial = None


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.step <= 0:
        parser.error("--step must be positive")
    if args.baud is None:
        args.baud = default_baud_from_serial_config(args.serial_config)
    return args


def main() -> int:
    args = parse_args()
    try:
        root = tk.Tk()
        app = ServoPoseCalibrationGUI(root, args)
    except Exception as exc:
        print(f"[CONFIG][ERROR] {exc}", file=sys.stderr)
        return 2

    def on_close() -> None:
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
