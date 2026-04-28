#!/usr/bin/env python3
"""
calibrate_board_homography.py
────────────────────────────────────────────────────────────────────────────
Standalone Phase 3A camera-board homography calibration tool.

This tool computes pixel -> checkerboard board-coordinate homography in cm.
It does not access serial/ESP32, move the robot, run YOLO, implement IK, or
start autonomous sorting.

Click/order conventions:
  Manual mode outer corners:
    1. top-left      -> (0.0, 0.0) cm
    2. top-right     -> (27.0, 0.0) cm
    3. bottom-right  -> (27.0, 18.0) cm
    4. bottom-left   -> (0.0, 18.0) cm

Keyboard:
  q  quit without saving
  r  reset calibration
  s  save after explicit terminal confirmation
  v  validation click mode
  m  switch to manual mode
  a  retry auto detection
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


DEFAULT_BOARD_CONFIG = "ros2_ws/src/robot_arm_5dof/config/board_config.yaml"
DEFAULT_CAMERA_CONFIG = "ros2_ws/src/robot_arm_5dof/config/camera_config.yaml"
DEFAULT_OUTPUT_NPY = "ros2_ws/src/robot_arm_5dof/config/homography.npy"
DEFAULT_OUTPUT_YAML = "ros2_ws/src/robot_arm_5dof/config/homography.yaml"
DEFAULT_CAM = 2
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

WINDOW_NAME = "Phase 3A Board Homography Calibration"
WINDOW_W = 960
WINDOW_H = 720

POINT_COLORS = [
    (0, 255, 255),
    (0, 180, 255),
    (0, 120, 255),
    (0, 80, 255),
]
GRID_COLOR = (70, 220, 70)
BORDER_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
WARN_COLOR = (0, 0, 255)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def validate_board_config(config: dict[str, Any]) -> dict[str, Any]:
    board = config.get("board", {})
    expected = {
        "cols_squares": 9,
        "rows_squares": 6,
        "square_size_cm": 3.0,
        "width_cm": 27.0,
        "height_cm": 18.0,
        "origin": "top_left",
        "x_direction": "right",
        "y_direction": "down",
        "inner_corners_cols": 8,
        "inner_corners_rows": 5,
    }
    for key, value in expected.items():
        actual = board.get(key)
        if actual != value:
            print(f"[WARN] board.{key} is {actual!r}, expected {value!r}")

    return {
        "cols_squares": int(board.get("cols_squares", 9)),
        "rows_squares": int(board.get("rows_squares", 6)),
        "square_size_cm": float(board.get("square_size_cm", 3.0)),
        "width_cm": float(board.get("width_cm", 27.0)),
        "height_cm": float(board.get("height_cm", 18.0)),
        "inner_corners_cols": int(board.get("inner_corners_cols", 8)),
        "inner_corners_rows": int(board.get("inner_corners_rows", 5)),
    }


def load_camera_id(camera_config_path: str) -> int | None:
    if not os.path.exists(camera_config_path):
        print(f"[WARN] Camera config not found: {camera_config_path}")
        return None
    data = load_yaml(camera_config_path)
    camera = data.get("camera", {})
    camera_id = camera.get("camera_id")
    return int(camera_id) if camera_id is not None else None


def inner_corner_board_points(board: dict[str, Any]) -> np.ndarray:
    cols = board["inner_corners_cols"]
    rows = board["inner_corners_rows"]
    square = board["square_size_cm"]
    pts = []
    for y_idx in range(rows):
        for x_idx in range(cols):
            pts.append(((x_idx + 1) * square, (y_idx + 1) * square))
    return np.array(pts, dtype=np.float32)


def outer_corner_board_points(board: dict[str, Any]) -> np.ndarray:
    w = board["width_cm"]
    h = board["height_cm"]
    return np.array(
        [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)],
        dtype=np.float32,
    )


def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def compute_reprojection_error_cm(
    H: np.ndarray,
    src_px: np.ndarray,
    dst_board_cm: np.ndarray,
) -> tuple[float, float]:
    projected = transform_points(H, src_px)
    errors = np.linalg.norm(projected - dst_board_cm, axis=1)
    return float(np.mean(errors)), float(np.max(errors))


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _ccw(a, b, c) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a, b, c, d) -> bool:
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def validate_manual_polygon(points: list[tuple[float, float]]) -> bool:
    if len(points) != 4:
        print("[WARN] Manual calibration requires exactly 4 clicked corners.")
        return False
    if segments_intersect(points[0], points[1], points[2], points[3]):
        print("[WARN] Manual polygon appears self-crossing.")
        return False
    if segments_intersect(points[1], points[2], points[3], points[0]):
        print("[WARN] Manual polygon appears self-crossing.")
        return False
    area = polygon_area(points)
    if area < 1000.0:
        print(f"[WARN] Manual polygon area is too small: {area:.1f} px^2")
        return False
    return True


def detect_inner_corners(frame: np.ndarray, board: dict[str, Any]):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pattern = (board["inner_corners_cols"], board["inner_corners_rows"])

    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern)
        if ok:
            corners = corners.reshape(-1, 2).astype(np.float32)
            return True, corners, "findChessboardCornersSB"

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return False, None, "findChessboardCorners"

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners.reshape(-1, 2).astype(np.float32), "findChessboardCorners"


def compute_auto_homography(frame: np.ndarray, board: dict[str, Any]):
    ok, src_px, method = detect_inner_corners(frame, board)
    if not ok:
        print(f"[WARN] Auto detection failed using {method}.")
        return None

    dst_cm = inner_corner_board_points(board)
    H, mask = cv2.findHomography(src_px, dst_cm, method=0)
    if H is None:
        print("[WARN] cv2.findHomography failed.")
        return None
    mean_err, max_err = compute_reprojection_error_cm(H, src_px, dst_cm)
    print(
        f"[BOARD] Auto homography computed with {method}: "
        f"mean_error={mean_err:.3f} cm, max_error={max_err:.3f} cm"
    )
    return {
        "mode": "auto",
        "H": H,
        "source_points_px": src_px,
        "destination_points_board_cm": dst_cm,
        "reprojection_error_cm": {"mean": mean_err, "max": max_err},
        "method": method,
        "mask": mask,
    }


def compute_manual_homography(points: list[tuple[float, float]], board: dict[str, Any]):
    if not validate_manual_polygon(points):
        return None
    src_px = np.array(points, dtype=np.float32)
    dst_cm = outer_corner_board_points(board)
    H = cv2.getPerspectiveTransform(src_px, dst_cm)
    print("[BOARD] Manual homography computed from 4 outer corners.")
    return {
        "mode": "manual",
        "H": H,
        "source_points_px": src_px,
        "destination_points_board_cm": dst_cm,
        "reprojection_error_cm": None,
        "method": "manual_outer_corners",
    }


def project_grid(H: np.ndarray, board: dict[str, Any]) -> tuple[list[np.ndarray], bool]:
    H_inv = np.linalg.inv(H)
    w = board["width_cm"]
    h = board["height_cm"]
    cols = board["cols_squares"]
    rows = board["rows_squares"]
    square = board["square_size_cm"]
    lines = []

    for i in range(cols + 1):
        x = min(i * square, w)
        pts = transform_points(H_inv, np.array([(x, 0.0), (x, h)], dtype=np.float32))
        lines.append(pts)
    for j in range(rows + 1):
        y = min(j * square, h)
        pts = transform_points(H_inv, np.array([(0.0, y), (w, y)], dtype=np.float32))
        lines.append(pts)

    all_pts = np.vstack(lines)
    wildly_outside = bool(np.any(np.abs(all_pts) > 10000))
    return lines, wildly_outside


def draw_text_lines(frame: np.ndarray, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_COLOR, 1)
        y += 22


def draw_points(frame: np.ndarray, points: np.ndarray | list, labels: list[str] | None = None) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    for i, (x, y) in enumerate(pts):
        color = POINT_COLORS[i % len(POINT_COLORS)]
        cv2.circle(frame, (int(round(x)), int(round(y))), 5, color, -1)
        cv2.circle(frame, (int(round(x)), int(round(y))), 5, (255, 255, 255), 1)
        label = labels[i] if labels and i < len(labels) else str(i + 1)
        cv2.putText(
            frame,
            label,
            (int(round(x)) + 7, int(round(y)) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )


def draw_grid(frame: np.ndarray, H: np.ndarray, board: dict[str, Any]) -> bool:
    try:
        grid_lines, wildly_outside = project_grid(H, board)
    except np.linalg.LinAlgError:
        print("[WARN] Homography inverse failed; cannot draw grid.")
        return True

    for pts in grid_lines:
        p1 = tuple(np.round(pts[0]).astype(int))
        p2 = tuple(np.round(pts[1]).astype(int))
        cv2.line(frame, p1, p2, GRID_COLOR, 1)
    outer = transform_points(
        np.linalg.inv(H),
        outer_corner_board_points(board),
    )
    cv2.polylines(frame, [np.round(outer).astype(np.int32)], True, BORDER_COLOR, 2)
    return wildly_outside


def pixel_to_board(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pt = transform_points(H, np.array([(x, y)], dtype=np.float32))[0]
    return float(pt[0]), float(pt[1])


def update_board_config_homography_file(board_config_path: str, homography_path: str) -> None:
    path = Path(board_config_path)
    text = path.read_text(encoding="utf-8")
    new_line = f"  homography_file: {homography_path}          # fill after homography calibration"
    lines = text.splitlines()
    changed = False
    for i, line in enumerate(lines):
        if line.strip().startswith("homography_file:"):
            lines[i] = new_line
            changed = True
            break
    if not changed:
        lines.append(new_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    result: dict[str, Any],
    args: argparse.Namespace,
    board: dict[str, Any],
    actual_width: int,
    actual_height: int,
) -> bool:
    if args.no_save:
        print("[INFO] --no-save active; not writing homography files or board_config.yaml.")
        return False

    print("[SAFETY] This saves camera-board homography only. No robot motion is involved.")
    print("[SAFETY] Moving the camera after calibration invalidates this homography.")
    confirm = input("Type SAVE to write homography files and update board_config.yaml: ").strip()
    if confirm != "SAVE":
        print("[INFO] Save cancelled.")
        return False

    output_npy = Path(args.output_npy)
    output_yaml = Path(args.output_yaml)
    for path in (output_npy, output_yaml):
        if path.exists():
            overwrite = input(f"[WARN] {path} exists. Type OVERWRITE to replace: ").strip()
            if overwrite != "OVERWRITE":
                print("[INFO] Save cancelled.")
                return False

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_npy, result["H"])

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": result["mode"],
        "method": result.get("method"),
        "camera_id": args.cam,
        "requested_width": args.width,
        "requested_height": args.height,
        "actual_width": actual_width,
        "actual_height": actual_height,
        "board": {
            "cols_squares": board["cols_squares"],
            "rows_squares": board["rows_squares"],
            "square_size_cm": board["square_size_cm"],
            "width_cm": board["width_cm"],
            "height_cm": board["height_cm"],
            "inner_corners_cols": board["inner_corners_cols"],
            "inner_corners_rows": board["inner_corners_rows"],
        },
        "source_points_px": result["source_points_px"].tolist(),
        "destination_points_board_cm": result["destination_points_board_cm"].tolist(),
        "homography": result["H"].tolist(),
        "reprojection_error_cm": result.get("reprojection_error_cm"),
        "warning": "Moving the camera after calibration invalidates this homography.",
    }
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, default_flow_style=False, sort_keys=False)

    update_board_config_homography_file(args.board_config, str(output_npy))
    print(f"[BOARD] Saved homography matrix: {output_npy}")
    print(f"[BOARD] Saved homography metadata: {output_yaml}")
    print(f"[BOARD] Updated board_config.yaml homography_file: {output_npy}")
    return True


class CalibrationState:
    def __init__(self, initial_mode: str):
        self.mode = "auto" if initial_mode == "auto-then-manual" else initial_mode
        self.initial_mode = initial_mode
        self.manual_points: list[tuple[float, float]] = []
        self.validation_mode = False
        self.result: dict[str, Any] | None = None
        self.auto_attempted = False
        self.grid_warning_printed = False

    def reset(self) -> None:
        self.manual_points = []
        self.validation_mode = False
        self.result = None
        self.auto_attempted = False
        self.grid_warning_printed = False
        print("[INFO] Calibration reset.")


def make_mouse_callback(state: CalibrationState, board: dict[str, Any]):
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state.validation_mode:
            if state.result is None:
                print("[WARN] Compute homography before validation clicks.")
                return
            bx, by = pixel_to_board(state.result["H"], x, y)
            inside = 0.0 <= bx <= board["width_cm"] and 0.0 <= by <= board["height_cm"]
            status = "inside" if inside else "OUTSIDE"
            print(f"[BOARD] Pixel ({x}, {y}) -> board ({bx:.2f}, {by:.2f}) cm [{status}]")
            if not inside:
                print("[WARN] Validation point is outside [0,27] x [0,18] cm.")
            return

        if state.mode != "manual":
            print("[INFO] Press 'm' to switch to manual mode before clicking corners.")
            return
        if len(state.manual_points) >= 4:
            print("[INFO] Already have 4 manual points. Press 'r' to reset.")
            return
        state.manual_points.append((float(x), float(y)))
        idx = len(state.manual_points)
        print(f"[BOARD] Click {idx}/4 {labels[idx - 1]}: ({x}, {y})")
        if len(state.manual_points) == 4:
            state.result = compute_manual_homography(state.manual_points, board)

    return on_mouse


def run_tool(args: argparse.Namespace) -> int:
    board_config = load_yaml(args.board_config)
    board = validate_board_config(board_config)
    camera_config_id = load_camera_id(args.camera_config)
    if camera_config_id is not None and camera_config_id != args.cam:
        print(
            f"[WARN] camera_config.yaml camera_id={camera_config_id}, "
            f"but this run uses --cam {args.cam}. External overhead webcam default is 2."
        )

    print("[SAFETY] Phase 3A camera-board calibration only.")
    print("[SAFETY] No serial/ESP32 access, no robot motion, no IK, no sorting.")
    print("[BOARD] Camera must not move after calibration; recalibrate if it moves.")
    print("[BOARD] OpenCV inner corners for this 9x6 square board are 8x5.")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.cam}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width != args.width or actual_height != args.height:
        print(
            f"[WARN] Actual resolution {actual_width}x{actual_height} "
            f"differs from requested {args.width}x{args.height}."
        )
    print(f"[INFO] Camera {args.cam} opened ({actual_width}x{actual_height}).")

    state = CalibrationState(args.mode)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(state, board))

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Cannot read frame.")
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break
            continue

        if state.mode == "auto" and not state.auto_attempted:
            state.auto_attempted = True
            state.result = compute_auto_homography(frame, board)
            if state.result is None and state.initial_mode == "auto-then-manual":
                print("[INFO] Switching to manual mode after auto detection failure.")
                state.mode = "manual"

        display = frame.copy()
        if state.result is not None and args.show_grid:
            wildly_outside = draw_grid(display, state.result["H"], board)
            if wildly_outside and not state.grid_warning_printed:
                print("[WARN] Projected grid is wildly outside frame; check corner order/detection.")
                state.grid_warning_printed = True

        if state.result is not None:
            draw_points(display, state.result["source_points_px"])
        elif state.manual_points:
            draw_points(
                display,
                state.manual_points,
                ["TL", "TR", "BR", "BL"],
            )

        status = "READY" if state.result is not None else "NO HOMOGRAPHY"
        if state.validation_mode:
            status += " | VALIDATION CLICK MODE"
        help_lines = [
            f"Mode: {state.mode} ({state.initial_mode}) | {status}",
            "q quit | r reset | s save | v validation | m manual | a retry auto",
            "Manual click order: TL, TR, BR, BL outer corners",
            "No robot motion. Camera must stay fixed after calibration.",
        ]
        draw_text_lines(display, help_lines)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue
        if key == ord("q"):
            print("[INFO] Quit without saving.")
            break
        if key == ord("r"):
            state.reset()
            state.mode = "auto" if state.initial_mode == "auto-then-manual" else state.initial_mode
            continue
        if key == ord("m"):
            state.mode = "manual"
            state.manual_points = []
            state.result = None
            state.validation_mode = False
            print("[INFO] Switched to manual mode. Click TL, TR, BR, BL outer corners.")
            continue
        if key == ord("a"):
            state.mode = "auto"
            state.result = compute_auto_homography(frame, board)
            state.auto_attempted = True
            state.validation_mode = False
            continue
        if key == ord("v"):
            state.validation_mode = not state.validation_mode
            print(f"[INFO] Validation click mode: {state.validation_mode}")
            continue
        if key == ord("s"):
            if state.result is None:
                print("[WARN] No homography to save.")
                continue
            save_outputs(state.result, args, board, actual_width, actual_height)
            continue

    cap.release()
    cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    epilog = (
        "examples:\n"
        "  python tools/calibrate_board_homography.py --mode auto-then-manual --cam 2 --show-grid\n"
        "  python tools/calibrate_board_homography.py --mode manual --cam 2 --no-save --show-grid\n"
        "\n"
        "This tool only calibrates camera pixel -> board cm homography. It does not\n"
        "access ESP32/serial, move servos, run YOLO, implement IK, or sort objects.\n"
    )
    parser = argparse.ArgumentParser(
        description="Phase 3A manual/auto board homography calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "manual", "auto-then-manual"],
        default="auto-then-manual",
    )
    parser.add_argument("--cam", type=int, default=DEFAULT_CAM)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--board-config", default=DEFAULT_BOARD_CONFIG)
    parser.add_argument("--camera-config", default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--output-npy", default=DEFAULT_OUTPUT_NPY)
    parser.add_argument("--output-yaml", default=DEFAULT_OUTPUT_YAML)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show-grid", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_tool(args)


if __name__ == "__main__":
    sys.exit(main())
