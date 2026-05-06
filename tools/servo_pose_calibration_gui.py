#!/usr/bin/env python3
"""
servo_pose_calibration_gui.py

Tkinter GUI for guarded manual servo and pose calibration over the stable ESP32
serial protocol. This tool is dry-run by default unless the operator
explicitly enables live hardware risk mode.
"""

from __future__ import annotations

import argparse
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
DEFAULT_SERIAL_CONFIG = "ros2_ws/src/robot_arm_5dof/config/serial_config.yaml"
DEFAULT_OUTPUT = "ros2_ws/src/robot_arm_5dof/config/taught_pick_place_poses.yaml"
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
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
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
        self.servo_cfg = {}
        self.pose_cfg = {}
        self.serial_cfg = {}
        self.taught_payload = {}
        self.pose_cache: dict[str, dict[str, int]] = {}
        self.limit_cache: dict[str, tuple[int, int]] = {}
        self.gripper_calibration = deepcopy(GRIPPER_DEFAULTS)
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
        self.current_command_var = tk.StringVar(value="")
        self.latest_status_var = tk.StringVar(value="No STATUS yet")
        self.custom_pose_name_var = tk.StringVar(value="")
        self.gripper_open_var = tk.IntVar(value=GRIPPER_DEFAULTS["open_deg"])
        self.gripper_soft_var = tk.IntVar(value=GRIPPER_DEFAULTS["close_soft_deg"])
        self.gripper_full_var = tk.IntVar(value=GRIPPER_DEFAULTS["close_full_deg"])

        self.log_widget: scrolledtext.ScrolledText | None = None
        self.yaml_widget: scrolledtext.ScrolledText | None = None
        self.scale_widgets: dict[str, tk.Scale] = {}
        self.pose_button_frame: ttk.Frame | None = None

        self._reload_all_configs(initial=True)
        self._apply_angles_to_ui(self._resolve_home_safe_angles())

        self.root.title("Servo Pose Calibration GUI")
        self.root.geometry("1740x980")
        self.root.minsize(1400, 820)

        self._build_ui()
        self._refresh_servo_limit_labels()
        self._refresh_gripper_widgets()
        self._refresh_yaml_preview()
        self._update_command_preview()

        self._log(
            "[CONFIG] "
            + ("Dry-run mode: serial commands will be logged only." if self.dry_run_mode else "Live mode enabled by operator risk flag.")
        )
        self._log("[CONFIG] Firmware HOME is not YAML HOME_SAFE.")
        self._log("[CONFIG] Saving a pose means human_taught_live, validated_for=manual_testing_only, autonomous_validated=false.")
        self.root.after(100, self._drain_ui_queue)

    def _reload_all_configs(self, initial: bool = False) -> None:
        self.servo_cfg = load_yaml_file(self.args.servo_config, "servo config")
        self.pose_cfg = load_yaml_file(self.args.pose_config, "pose config")
        self.serial_cfg = load_yaml_file(self.args.serial_config, "serial config")
        self.taught_payload = self._load_or_init_output()
        self.limit_cache = self._read_limits(self.servo_cfg)
        self.pose_cache = self._build_pose_cache()
        self.gripper_calibration = self._read_gripper_calibration(self.servo_cfg)
        if not initial:
            self._refresh_servo_limit_labels()
            self._refresh_gripper_widgets()
            self._refresh_yaml_preview()
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
        middle = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        middle.pack(fill=tk.BOTH, expand=True)
        bottom = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        bottom.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        left_top = ttk.Frame(top)
        left_top.pack(side=tk.LEFT, fill=tk.X, expand=True)
        right_top = ttk.Frame(top)
        right_top.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_connection_panel(left_top)
        self._build_status_panel(right_top)
        self._build_servo_panel(middle)
        self._build_pose_panel(middle)
        self._build_log_and_yaml_panel(bottom)

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
        ttk.Button(bottom_controls, text="Send Current MOVE_SAFE", command=self.on_send_current_pose).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bottom_controls, text="Save Servo Limits", command=self.save_servo_limits).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(bottom_controls, text="Step (deg)").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Spinbox(bottom_controls, from_=1, to=30, textvariable=self.step_var, width=6, command=self._on_step_changed).pack(side=tk.LEFT, padx=(0, 6))
        for step_value in (1, 2, 5, 10):
            ttk.Button(bottom_controls, text=str(step_value), command=lambda value=step_value: self.set_step(value)).pack(side=tk.LEFT, padx=(0, 4))

        command_bar = ttk.Frame(frame)
        command_bar.grid(row=len(CHANNELS) + 2, column=0, columnspan=7, sticky="ew", padx=4, pady=(4, 0))
        ttk.Label(command_bar, text="Current MOVE_SAFE").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(command_bar, textvariable=self.current_command_var).pack(side=tk.LEFT)

        grip = ttk.LabelFrame(frame, text="Gripper Calibration")
        grip.grid(row=len(CHANNELS) + 3, column=0, columnspan=7, sticky="ew", padx=4, pady=(8, 4))
        ttk.Label(grip, text="OPEN").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(grip, textvariable=self.gripper_open_var).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(grip, text="Set OPEN From Current", command=lambda: self.set_gripper_calibration_from_current("open_deg")).grid(
            row=0, column=2, sticky="ew", padx=4, pady=4
        )
        ttk.Label(grip, text="SOFT CLOSE").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(grip, textvariable=self.gripper_soft_var).grid(row=1, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(grip, text="Set SOFT CLOSE From Current", command=lambda: self.set_gripper_calibration_from_current("close_soft_deg")).grid(
            row=1, column=2, sticky="ew", padx=4, pady=4
        )
        ttk.Label(grip, text="FULL CLOSE").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(grip, textvariable=self.gripper_full_var).grid(row=2, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(grip, text="Set FULL CLOSE From Current", command=lambda: self.set_gripper_calibration_from_current("close_full_deg")).grid(
            row=2, column=2, sticky="ew", padx=4, pady=4
        )
        ttk.Button(grip, text="Save Gripper Calibration", command=self.save_gripper_calibration).grid(
            row=0, column=3, rowspan=3, sticky="ns", padx=8, pady=4
        )
        ttk.Button(grip, text="OPEN CH6", command=lambda: self.use_gripper_quick("open_deg")).grid(row=0, column=4, sticky="ew", padx=4, pady=4)
        ttk.Button(grip, text="SOFT CLOSE CH6", command=lambda: self.use_gripper_quick("close_soft_deg")).grid(
            row=1, column=4, sticky="ew", padx=4, pady=4
        )
        ttk.Button(grip, text="FULL CLOSE CH6", command=lambda: self.use_gripper_quick("close_full_deg")).grid(
            row=2, column=4, sticky="ew", padx=4, pady=4
        )

    def _build_pose_panel(self, parent: ttk.Panedwindow) -> None:
        frame = ttk.LabelFrame(parent, text="Pose Load / Go / Save")
        parent.add(frame, weight=3)
        self.pose_button_frame = frame

        ttk.Label(frame, text="Pose").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Load").grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Go").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Save").grid(row=0, column=3, sticky="w", padx=4, pady=4)

        for row, pose_name in enumerate(POSE_NAMES, start=1):
            ttk.Label(frame, text=pose_name).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            ttk.Button(frame, text=f"Load {pose_name}", command=lambda name=pose_name: self.load_pose(name)).grid(
                row=row, column=1, sticky="ew", padx=4, pady=3
            )
            ttk.Button(frame, text=f"Go {pose_name}", command=lambda name=pose_name: self.go_pose(name)).grid(
                row=row, column=2, sticky="ew", padx=4, pady=3
            )
            ttk.Button(frame, text=f"Save {pose_name}", command=lambda name=pose_name: self.save_pose(name)).grid(
                row=row, column=3, sticky="ew", padx=4, pady=3
            )

        custom = ttk.LabelFrame(frame, text="Custom Pose")
        custom.grid(row=len(POSE_NAMES) + 1, column=0, columnspan=4, sticky="ew", padx=4, pady=(8, 4))
        ttk.Entry(custom, textvariable=self.custom_pose_name_var, width=22).pack(side=tk.LEFT, padx=(4, 6), pady=4)
        ttk.Button(custom, text="Save Custom", command=self.save_custom_pose).pack(side=tk.LEFT, padx=(0, 6), pady=4)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.columnconfigure(3, weight=1)

    def _build_log_and_yaml_panel(self, parent: ttk.Panedwindow) -> None:
        log_frame = ttk.LabelFrame(parent, text="Report / Log")
        yaml_frame = ttk.LabelFrame(parent, text="Current Pose YAML")
        parent.add(log_frame, weight=3)
        parent.add(yaml_frame, weight=2)

        self.log_widget = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.log_widget.configure(state=tk.DISABLED)

        self.yaml_widget = scrolledtext.ScrolledText(yaml_frame, wrap=tk.NONE, height=20)
        self.yaml_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.yaml_widget.configure(state=tk.DISABLED)

        button_bar = ttk.Frame(yaml_frame)
        button_bar.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(button_bar, text="Copy YAML to Clipboard", command=self.copy_yaml_to_clipboard).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_bar, text="Export Session Log", command=self.export_session_log).pack(side=tk.LEFT)

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

    def _save_servo_cfg(self) -> None:
        save_yaml_file(Path(self.args.servo_config), self.servo_cfg)

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
