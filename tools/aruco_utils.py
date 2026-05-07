#!/usr/bin/env python3
"""
aruco_utils.py

Shared camera-only helpers for ArUco diagnostic tools. This module must not
access serial, move the robot, or import motion helpers.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


DEFAULT_ARUCO_CONFIG = "ros2_ws/src/robot_arm_5dof/config/aruco_config.yaml"
DEFAULT_HOMOGRAPHY = "ros2_ws/src/robot_arm_5dof/config/homography.npy"
DEFAULT_ARUCO_HOMOGRAPHY_NPY = "ros2_ws/src/robot_arm_5dof/config/aruco_homography.npy"
DEFAULT_ARUCO_HOMOGRAPHY_YAML = "ros2_ws/src/robot_arm_5dof/config/aruco_homography.yaml"
DEFAULT_CAMERA_ID = 2
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_DURATION_SEC = 20.0
WINDOW_W = 960
WINDOW_H = 720


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def save_yaml(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def resolve_existing_path(path_value: str, config_path: str | None = None) -> Path:
    raw = Path(path_value).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(REPO_ROOT / raw)
        candidates.append(Path.cwd() / raw)
        if config_path is not None:
            candidates.append(Path(config_path).resolve().parent / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"File not found: {path_value}. Checked: {checked}")


def resolve_output_path(path_value: str) -> Path:
    raw = Path(path_value).expanduser()
    if raw.is_absolute():
        return raw
    return (REPO_ROOT / raw).resolve()


def load_aruco_config(path: str) -> dict[str, Any]:
    root = load_yaml(path)
    aruco_cfg = root.get("aruco", {})
    if not isinstance(aruco_cfg, dict):
        raise RuntimeError("[CONFIG][ERROR] aruco mapping missing")
    return root


def get_aruco_dict_name(aruco_root: dict[str, Any], override: str | None = None) -> str:
    cfg = aruco_root.get("aruco", {})
    return str(override or cfg.get("dictionary", "DICT_4X4_50"))


def get_predefined_dictionary(dict_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("[CONFIG][ERROR] OpenCV aruco module is unavailable in this build.")
    dict_id = getattr(cv2.aruco, dict_name, None)
    if dict_id is None:
        raise RuntimeError(f"[CONFIG][ERROR] Unknown ArUco dictionary {dict_name!r}")
    return cv2.aruco.getPredefinedDictionary(dict_id)


def build_detector(dictionary, profile: str = "default"):
    return build_detector_with_profile(dictionary, profile=profile)


def _set_if_hasattr(obj, name: str, value) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def build_detector_with_profile(dictionary, profile: str = "default"):
    params = cv2.aruco.DetectorParameters()
    profile_name = str(profile).strip().lower()

    if profile_name in {"relaxed", "aggressive"}:
        _set_if_hasattr(params, "cornerRefinementMethod", getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1))
        _set_if_hasattr(params, "cornerRefinementWinSize", 5)
        _set_if_hasattr(params, "adaptiveThreshWinSizeMin", 3)
        _set_if_hasattr(params, "adaptiveThreshWinSizeMax", 43 if profile_name == "relaxed" else 61)
        _set_if_hasattr(params, "adaptiveThreshWinSizeStep", 4)
        _set_if_hasattr(params, "minMarkerPerimeterRate", 0.015 if profile_name == "relaxed" else 0.01)
        _set_if_hasattr(params, "maxMarkerPerimeterRate", 5.0 if profile_name == "relaxed" else 6.0)
        _set_if_hasattr(params, "minDistanceToBorder", 2 if profile_name == "relaxed" else 1)
        _set_if_hasattr(params, "polygonalApproxAccuracyRate", 0.06 if profile_name == "relaxed" else 0.08)
        _set_if_hasattr(params, "minCornerDistanceRate", 0.03 if profile_name == "relaxed" else 0.02)
        _set_if_hasattr(params, "minOtsuStdDev", 3.0 if profile_name == "relaxed" else 1.0)

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return None


def _normalize_adaptive_block_size(value: int) -> int:
    block_size = max(3, int(value))
    if block_size % 2 == 0:
        block_size += 1
    return block_size


def preprocess_aruco_frame(
    frame: np.ndarray,
    mode: str = "none",
    adaptive_block_size: int = 31,
    adaptive_c: int = 7,
) -> np.ndarray:
    name = str(mode).strip().lower()
    if name == "none":
        return frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if name == "gray":
        return gray
    if name == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    if name == "adaptive":
        block_size = _normalize_adaptive_block_size(adaptive_block_size)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            int(adaptive_c),
        )
    if name == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(gray, -1, kernel)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def _offset_marker_arrays(marker_arrays, offset_x: int, offset_y: int):
    if marker_arrays is None:
        return None
    shifted = []
    for arr in marker_arrays:
        pts = np.asarray(arr, dtype=np.float32).copy()
        pts[..., 0] += float(offset_x)
        pts[..., 1] += float(offset_y)
        shifted.append(pts)
    return shifted


def detect_markers(
    frame: np.ndarray,
    dictionary,
    detector=None,
    profile: str = "default",
    preprocess: str = "none",
    roi: tuple[int, int, int, int] | None = None,
    adaptive_block_size: int = 31,
    adaptive_c: int = 7,
):
    processed_frame = preprocess_aruco_frame(
        frame,
        preprocess,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
    )
    detect_frame = processed_frame
    offset_x = 0
    offset_y = 0
    if roi is not None:
        x1, y1, x2, y2 = roi
        detect_frame = processed_frame[y1:y2, x1:x2]
        offset_x = x1
        offset_y = y1
    local_detector = detector
    if local_detector is None:
        local_detector = build_detector_with_profile(dictionary, profile=profile)
    if local_detector is not None:
        corners, ids, rejected = local_detector.detectMarkers(detect_frame)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(detect_frame, dictionary)
    corners = _offset_marker_arrays(corners, offset_x, offset_y)
    rejected = _offset_marker_arrays(rejected, offset_x, offset_y)
    return corners, ids, rejected, processed_frame


def marker_center(corners: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def corners_to_list(corners: np.ndarray) -> list[list[float]]:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in pts]


def marker_records(corners, ids, aruco_root: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = aruco_root.get("aruco", {})
    board_markers = cfg.get("board_markers", {})
    gripper_marker = cfg.get("gripper_marker", {})
    records: list[dict[str, Any]] = []
    if ids is None:
        return records
    for idx, marker_id in enumerate(ids.flatten().tolist()):
        c = corners[idx]
        u, v = marker_center(c)
        pts = np.asarray(c, dtype=np.float32).reshape(-1, 2)
        width = float(np.linalg.norm(pts[1] - pts[0]))
        height = float(np.linalg.norm(pts[2] - pts[1]))
        area = float(width * height)
        marker_role = None
        expected_board_x = None
        expected_board_y = None
        if marker_id == int(gripper_marker.get("id", -1)):
            marker_role = str(gripper_marker.get("role", "gripper_tcp_debug"))
        else:
            key = f"id_{marker_id}"
            entry = board_markers.get(key)
            if isinstance(entry, dict):
                marker_role = str(entry.get("role", key))
                expected_board_x = float(entry.get("board_x_cm"))
                expected_board_y = float(entry.get("board_y_cm"))
        records.append(
            {
                "id": int(marker_id),
                "role": marker_role,
                "pixel_u": u,
                "pixel_v": v,
                "pixel_corners": corners_to_list(c),
                "corners": pts,
                "approx_width_px": width,
                "approx_height_px": height,
                "approx_area_px": area,
                "expected_board_x_cm": expected_board_x,
                "expected_board_y_cm": expected_board_y,
            }
        )
    return records


def pixel_to_board(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    pt = np.array([[[u, v]]], dtype=np.float32)
    board_pt = cv2.perspectiveTransform(pt, H.astype(np.float32))
    return float(board_pt[0][0][0]), float(board_pt[0][0][1])


def add_board_coordinates(records: list[dict[str, Any]], H: np.ndarray | None) -> list[dict[str, Any]]:
    enriched = []
    for record in records:
        out = dict(record)
        if H is not None:
            bx, by = pixel_to_board(H, record["pixel_u"], record["pixel_v"])
            out["board_x_cm"] = bx
            out["board_y_cm"] = by
        enriched.append(out)
    return enriched


def find_record(records: list[dict[str, Any]], marker_id: int) -> dict[str, Any] | None:
    for record in records:
        if int(record["id"]) == int(marker_id):
            return record
    return None


def compute_board_error_stats(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    errors = []
    for record in records:
        if "board_x_cm" not in record:
            continue
        if record.get("expected_board_x_cm") is None or record.get("expected_board_y_cm") is None:
            continue
        dx = float(record["board_x_cm"]) - float(record["expected_board_x_cm"])
        dy = float(record["board_y_cm"]) - float(record["expected_board_y_cm"])
        dist = math.hypot(dx, dy)
        errors.append({"id": record["id"], "dx_cm": dx, "dy_cm": dy, "dist_cm": dist})
    if not errors:
        return None
    mean_dist = sum(e["dist_cm"] for e in errors) / len(errors)
    max_item = max(errors, key=lambda item: item["dist_cm"])
    return {
        "count": len(errors),
        "errors": errors,
        "mean_dist_cm": mean_dist,
        "max_dist_cm": float(max_item["dist_cm"]),
        "worst_marker_id": int(max_item["id"]),
    }


def draw_marker_overlay(
    frame: np.ndarray,
    records: list[dict[str, Any]],
    show_board_coords: bool = False,
    target_board_xy: tuple[float, float] | None = None,
    marker_board_xy: tuple[float, float] | None = None,
    tcp_board_xy: tuple[float, float] | None = None,
    error_text: str | None = None,
) -> np.ndarray:
    ann = frame.copy()
    for record in records:
        pts = np.asarray(record["corners"], dtype=np.int32).reshape((-1, 1, 2))
        color = (0, 255, 0) if record["id"] != 0 else (255, 180, 0)
        cv2.polylines(ann, [pts], True, color, 2)
        u = int(round(record["pixel_u"]))
        v = int(round(record["pixel_v"]))
        cv2.circle(ann, (u, v), 4, color, -1)
        label = f"ID {record['id']}"
        if record.get("role"):
            label += f" {record['role']}"
        if show_board_coords and "board_x_cm" in record:
            label += f" | {record['board_x_cm']:.1f},{record['board_y_cm']:.1f}cm"
        cv2.putText(ann, label, (u + 6, max(18, v - 6)), cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1)

    H_img, W_img = ann.shape[:2]
    cv2.rectangle(ann, (0, 0), (W_img, 28), (12, 12, 12), -1)
    cv2.putText(
        ann,
        "ArUco diagnostic only - no robot motion",
        (8, 19),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    if target_board_xy is not None or marker_board_xy is not None or tcp_board_xy is not None or error_text:
        lines = []
        if target_board_xy is not None:
            lines.append(f"target=({target_board_xy[0]:.2f},{target_board_xy[1]:.2f})cm")
        if marker_board_xy is not None:
            lines.append(f"marker=({marker_board_xy[0]:.2f},{marker_board_xy[1]:.2f})cm")
        if tcp_board_xy is not None:
            lines.append(f"tcp=({tcp_board_xy[0]:.2f},{tcp_board_xy[1]:.2f})cm")
        if error_text:
            lines.append(error_text)
        y = H_img - 12
        for line in reversed(lines):
            cv2.putText(ann, line, (8, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            y -= 18
    return ann


def rejected_candidate_records(rejected) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if rejected is None:
        return records
    for index, candidate in enumerate(rejected):
        pts = np.asarray(candidate, dtype=np.float32).reshape(-1, 2)
        u = float(np.mean(pts[:, 0]))
        v = float(np.mean(pts[:, 1]))
        width = float(np.linalg.norm(pts[1] - pts[0]))
        height = float(np.linalg.norm(pts[2] - pts[1]))
        records.append(
            {
                "index": index,
                "pixel_u": u,
                "pixel_v": v,
                "approx_width_px": width,
                "approx_height_px": height,
                "pixel_corners": [[float(x), float(y)] for x, y in pts],
                "corners": pts,
            }
        )
    return records


def draw_rejected_overlay(frame: np.ndarray, rejected_records: list[dict[str, Any]]) -> np.ndarray:
    ann = frame.copy()
    for record in rejected_records:
        pts = np.asarray(record["corners"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(ann, [pts], True, (0, 0, 255), 1)
        u = int(round(record["pixel_u"]))
        v = int(round(record["pixel_v"]))
        cv2.circle(ann, (u, v), 3, (0, 0, 255), -1)
        label = f"rej {record['index']}"
        cv2.putText(ann, label, (u + 4, max(16, v - 4)), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)
    return ann


def filter_rejected_candidate_records(
    records: list[dict[str, Any]],
    min_width_px: float,
    min_height_px: float,
    min_area_px: float,
    show_all: bool = False,
) -> list[dict[str, Any]]:
    if show_all:
        return list(records)
    filtered = []
    for record in records:
        if float(record["approx_width_px"]) < float(min_width_px):
            continue
        if float(record["approx_height_px"]) < float(min_height_px):
            continue
        area = float(record["approx_width_px"]) * float(record["approx_height_px"])
        if area < float(min_area_px):
            continue
        filtered.append(record)
    return filtered


def save_candidate_crops(frame: np.ndarray, records: list[dict[str, Any]], output_dir: str | Path, prefix: str) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    height, width = frame.shape[:2]
    for record in records:
        pts = np.asarray(record["corners"], dtype=np.float32)
        min_x = max(0, int(np.floor(np.min(pts[:, 0]) - 8)))
        min_y = max(0, int(np.floor(np.min(pts[:, 1]) - 8)))
        max_x = min(width, int(np.ceil(np.max(pts[:, 0]) + 8)))
        max_y = min(height, int(np.ceil(np.max(pts[:, 1]) + 8)))
        if max_x <= min_x or max_y <= min_y:
            continue
        crop = frame[min_y:max_y, min_x:max_x]
        identifier = record.get("id", record.get("index", "x"))
        out_path = out_dir / f"{prefix}_{identifier}.jpg"
        cv2.imwrite(str(out_path), crop)
        saved_paths.append(out_path)
    return saved_paths


def require_exact(label: str, prompt: str) -> bool:
    print(prompt)
    try:
        typed = input("> ").strip()
    except EOFError:
        return False
    return typed == label


def get_board_marker_points(aruco_root: dict[str, Any]) -> list[tuple[int, float, float]]:
    cfg = aruco_root.get("aruco", {})
    board_markers = cfg.get("board_markers", {})
    points = []
    for marker_id in (1, 2, 3, 4):
        entry = board_markers.get(f"id_{marker_id}", {})
        if not isinstance(entry, dict):
            continue
        points.append((marker_id, float(entry["board_x_cm"]), float(entry["board_y_cm"])))
    return points


def compute_homography_from_markers(records: list[dict[str, Any]], aruco_root: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    src_pts = []
    dst_pts = []
    used = []
    for marker_id, board_x, board_y in get_board_marker_points(aruco_root):
        record = find_record(records, marker_id)
        if record is None:
            continue
        src_pts.append([record["pixel_u"], record["pixel_v"]])
        dst_pts.append([board_x, board_y])
        used.append({"id": marker_id, "pixel_u": record["pixel_u"], "pixel_v": record["pixel_v"], "board_x_cm": board_x, "board_y_cm": board_y})
    if len(src_pts) != 4:
        raise RuntimeError("[VISION][ERROR] Need board markers IDs 1,2,3,4 to compute ArUco homography.")
    H = cv2.getPerspectiveTransform(np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32))
    return H.astype(np.float64), used


def homography_reprojection_stats(H: np.ndarray, used: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    for item in used:
        bx, by = pixel_to_board(H, item["pixel_u"], item["pixel_v"])
        dx = bx - item["board_x_cm"]
        dy = by - item["board_y_cm"]
        dist = math.hypot(dx, dy)
        errors.append({"id": int(item["id"]), "dx_cm": dx, "dy_cm": dy, "dist_cm": dist})
    mean_dist = sum(e["dist_cm"] for e in errors) / len(errors)
    max_item = max(errors, key=lambda entry: entry["dist_cm"])
    return {
        "errors": errors,
        "mean_dist_cm": mean_dist,
        "max_dist_cm": float(max_item["dist_cm"]),
        "worst_marker_id": int(max_item["id"]),
    }


def aruco_target_point_from_offset(marker_board_xy: tuple[float, float], aruco_root: dict[str, Any]) -> tuple[float, float] | None:
    cfg = aruco_root.get("aruco", {})
    offset = cfg.get("gripper_marker", {}).get("marker_to_tcp_offset_cm", {})
    ox = offset.get("x")
    oy = offset.get("y")
    if ox is None or oy is None:
        return None
    return float(marker_board_xy[0]) + float(ox), float(marker_board_xy[1]) + float(oy)
