#!/usr/bin/env python3
"""
test_yolo_board_mapping.py
────────────────────────────────────────────────────────────────────────────
Phase 3B standalone test:
  YOLO detections -> accepted bbox center pixel -> board coordinate in cm.

This tool does not access ESP32/serial, move the robot, implement IK, modify
homography, or start sorting. It reuses the conservative YOLO filters from
tools/test_yolo_camera.py and only adds homography-based board mapping.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from test_yolo_camera import (  # noqa: E402
    CLASS_COLORS,
    DEFAULT_ASPECT_MAX,
    DEFAULT_ASPECT_MIN,
    DEFAULT_CAKE_CONF,
    DEFAULT_CAKE_MINAREA,
    DEFAULT_CAM,
    DEFAULT_CONF,
    DEFAULT_CROSS_NMS_IOU,
    DEFAULT_DEVICE,
    DEFAULT_DONUT_CONF,
    DEFAULT_DONUT_MINAREA,
    DEFAULT_IMGSZ,
    DEFAULT_MAXAREA,
    DEFAULT_MAXDET,
    DEFAULT_MINAREA,
    DEFAULT_NMS_IOU,
    DEFAULT_STABLE_FRAMES,
    DEFAULT_CENTER_TOL,
    STATUS_COLORS,
    StabilityTracker,
    box_area,
    cross_group_nms,
    filter_detections,
    group_level_nms,
    load_model,
    parse_roi,
    print_environment,
    resolve_device,
    run_yolo,
)


DEFAULT_MODEL = "assets/best.pt"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_BOARD_CONFIG = "ros2_ws/src/robot_arm_5dof/config/board_config.yaml"

WINDOW_NAME = "YOLO Board Mapping Debug"
WINDOW_W = 960
WINDOW_H = 720


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def resolve_existing_path(path_value: str, config_path: str) -> Path:
    raw = Path(path_value).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(REPO_ROOT / raw)
        candidates.append(Path.cwd() / raw)
        candidates.append(Path(config_path).resolve().parent / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"File not found: {path_value}. Checked: {checked}")


def load_board_and_homography(board_config_path: str, homography_override: str | None):
    board_config = load_yaml(board_config_path)
    board = board_config.get("board", {})
    width_cm = float(board.get("width_cm", 27.0))
    height_cm = float(board.get("height_cm", 18.0))

    homography_value = homography_override or board.get("homography_file")
    if not homography_value:
        raise ValueError(
            "board.homography_file is empty. Run tools/calibrate_board_homography.py first "
            "or pass --homography."
        )
    homography_path = resolve_existing_path(str(homography_value), board_config_path)
    H = np.load(homography_path)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got shape {H.shape} from {homography_path}")
    return board_config, H.astype(np.float64), homography_path, width_cm, height_cm


def pixel_to_board(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    pt = np.array([[[u, v]]], dtype=np.float32)
    board_pt = cv2.perspectiveTransform(pt, H.astype(np.float32))
    return float(board_pt[0][0][0]), float(board_pt[0][0][1])


def annotate_board_coordinates(
    detections: list[dict[str, Any]],
    H: np.ndarray,
    width_cm: float,
    height_cm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inside = []
    outside = []
    for det in detections:
        bx, by = pixel_to_board(H, det["u"], det["v"])
        mapped = {
            **det,
            "board_x_cm": bx,
            "board_y_cm": by,
            "inside_board": 0.0 <= bx <= width_cm and 0.0 <= by <= height_cm,
        }
        if mapped["inside_board"]:
            inside.append(mapped)
        else:
            outside.append({
                **mapped,
                "status": "OUTSIDE_BOARD",
                "reject_reason": "OUTSIDE_BOARD",
            })
    return inside, outside


def draw_overlay(
    frame: np.ndarray,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    fps: float,
    n_raw: int,
    show_rejected: bool,
) -> np.ndarray:
    ann = frame.copy()
    H_img, W_img = ann.shape[:2]
    cv2.rectangle(ann, (0, 0), (W_img, 34), (12, 12, 12), -1)
    cv2.putText(
        ann,
        f"FPS:{fps:4.1f} RAW:{n_raw} ACC:{len(accepted)} REJ:{len(rejected)}",
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    def draw_one(det: dict[str, Any], rejected_det: bool) -> None:
        x1, y1, x2, y2 = det["bbox"]
        u, v = det["u"], det["v"]
        grp = det.get("group", "UNKNOWN")
        conf = det.get("conf", 0.0)
        raw = det.get("raw", "?")
        cls_col = CLASS_COLORS.get(grp, CLASS_COLORS["UNKNOWN"])
        if rejected_det:
            reason = det.get("reject_reason", "?")
            color = STATUS_COLORS["REJECTED"]
            if "board_x_cm" in det:
                label = (
                    f"{reason} | {grp} {conf:.2f} | u={u} v={v} | "
                    f"x={det['board_x_cm']:.1f} y={det['board_y_cm']:.1f}cm"
                )
            else:
                label = f"REJECTED:{reason} | {grp} {raw} {conf:.2f}"
        else:
            color = cls_col
            label = (
                f"{grp} | {raw} | {conf:.2f} | u={u} v={v} | "
                f"x={det['board_x_cm']:.1f} y={det['board_y_cm']:.1f}cm | "
                f"{det.get('status', '')}"
            )

        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        cv2.circle(ann, (u, v), 5, cls_col, -1)
        cv2.circle(ann, (u, v), 5, (255, 255, 255), 1)

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        bg_y1 = max(0, y1 - lh - 9)
        bg_x2 = min(W_img - 1, x1 + lw + 8)
        cv2.rectangle(ann, (x1, bg_y1), (bg_x2, y1), color, -1)
        cv2.putText(
            ann,
            label,
            (x1 + 4, y1 - 3),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    for det in accepted:
        draw_one(det, rejected_det=False)
    if show_rejected:
        for det in rejected:
            draw_one(det, rejected_det=True)

    if not accepted:
        cv2.putText(
            ann,
            "NO BOARD TARGET",
            (max(8, W_img // 2 - 180), H_img // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (0, 0, 255),
            2,
        )
    return ann


def run_raw_preview(cam: int, width: int, height: int, show: bool) -> None:
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera {cam} opened ({actual_w}x{actual_h}) RAW PREVIEW")
    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Cannot read frame")
                time.sleep(0.5)
                continue
            if show:
                cv2.imshow(WINDOW_NAME, frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped raw preview.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_mapping(args: argparse.Namespace) -> None:
    board_config, H, H_path, width_cm, height_cm = load_board_and_homography(
        args.board_config,
        args.homography,
    )
    board = board_config.get("board", {})
    print(f"[BOARD] Loaded board_config: {args.board_config}")
    print(f"[BOARD] Loaded homography: {H_path}")
    print(f"[BOARD] Board bounds: x=[0,{width_cm:g}] cm y=[0,{height_cm:g}] cm")
    print("[SAFETY] Phase 3B maps detections to board cm only. No robot motion, no serial, no sorting.")

    device = resolve_device(args.device)
    print_environment(device)
    model = load_model(args.model, device)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.cam}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera {args.cam} opened ({actual_w}x{actual_h})")
    if actual_w != args.width or actual_h != args.height:
        print(f"[WARN] Actual resolution differs from requested {args.width}x{args.height}")

    if args.roi:
        try:
            args.roi = parse_roi(args.roi)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        print(f"[INFO] ROI: {args.roi} (bbox centers are restored to full-frame pixels)")
    else:
        print("[WARN] Full-frame detection is for debug only; ROI is recommended.")

    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    tracker = StabilityTracker(
        stable_frames=args.stable_frames,
        center_tol=args.center_tolerance,
    )
    print("  FRAME  STATUS         GROUP   RAW              CONF       u       v     board_x     board_y  IN")
    print("  ------ -------------- ------- --------------- -------- ------- ------- ---------- ---------- ---")

    fps_s = 0.0
    t_prev = time.time()
    fc = 0
    saved = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Cannot read frame")
                time.sleep(0.5)
                continue
            raw_frame = frame.copy()
            now = time.time()
            fc += 1

            raw = run_yolo(model, frame, args.roi, args.conf, args.imgsz, args.max_det, device)
            filt = filter_detections(
                raw,
                conf_thr=args.conf,
                cake_conf=args.cake_conf,
                donut_conf=args.donut_conf,
                min_area=args.min_area,
                max_area=args.max_area,
                cake_min_area=args.cake_min_area,
                donut_min_area=args.donut_min_area,
                aspect_min=args.aspect_min,
                aspect_max=args.aspect_max,
            )
            acc = [d for d in filt if d["reject_reason"] is None]
            rej = [d for d in filt if d["reject_reason"] is not None]

            acc = group_level_nms(acc, args.group_nms_iou)
            acc, cross_rej = cross_group_nms(acc, args.cross_nms_iou)
            rej.extend(cross_rej)
            acc = tracker.update(acc)

            mapped_acc, outside_rej = annotate_board_coordinates(acc, H, width_cm, height_cm)
            rej.extend(outside_rej)

            fps_s = 0.9 * fps_s + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            for det in mapped_acc:
                print(
                    f"  {fc:06d} {det.get('status',''):<14} "
                    f"{det['group']:<7} {det['raw']:<15} "
                    f"conf={det['conf']:.3f}  "
                    f"u={det['u']:>5}  v={det['v']:>5}  "
                    f"x={det['board_x_cm']:>7.2f}  y={det['board_y_cm']:>7.2f}  YES"
                )
            for det in outside_rej:
                print(
                    f"  {fc:06d} OUTSIDE_BOARD  "
                    f"{det['group']:<7} {det['raw']:<15} "
                    f"conf={det['conf']:.3f}  "
                    f"u={det['u']:>5}  v={det['v']:>5}  "
                    f"x={det['board_x_cm']:>7.2f}  y={det['board_y_cm']:>7.2f}  NO"
                )
            if not raw:
                print(f"  {fc:06d} (no YOLO detections)")

            ann = draw_overlay(frame, mapped_acc, rej, fps_s, len(raw), args.show_rejected)
            if args.save_debug and not saved:
                cv2.imwrite("debug_board_mapping_raw.jpg", raw_frame)
                cv2.imwrite("debug_board_mapping_annotated.jpg", ann)
                print("[INFO] Saved debug_board_mapping_raw.jpg and debug_board_mapping_annotated.jpg")
                saved = True

            if args.show:
                cv2.imshow(WINDOW_NAME, ann)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Stopped board mapping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    epilog = (
        "examples:\n"
        "  /home/andra/envs/robot_yolo_env/bin/python tools/test_yolo_board_mapping.py --show --cam 2 --roi 100,250,620,470\n"
        "  /home/andra/envs/robot_yolo_env/bin/python tools/test_yolo_board_mapping.py --raw-preview --show\n"
        "\n"
        "This tool maps YOLO bbox centers to checkerboard coordinates only. It does not\n"
        "modify homography, access ESP32/serial, move the robot, run IK, or sort objects.\n"
    )
    parser = argparse.ArgumentParser(
        description="Standalone YOLO to board-coordinate mapping test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cam", type=int, default=DEFAULT_CAM)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--board-config", default=DEFAULT_BOARD_CONFIG)
    parser.add_argument("--homography")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["auto", "cpu", "0"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--show-rejected", action="store_true")
    parser.add_argument("--roi", type=str)
    parser.add_argument("--raw-preview", action="store_true")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--cake-conf", type=float, default=DEFAULT_CAKE_CONF)
    parser.add_argument("--donut-conf", type=float, default=DEFAULT_DONUT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAXDET)
    parser.add_argument("--min-area", type=int, default=DEFAULT_MINAREA)
    parser.add_argument("--max-area", type=int, default=DEFAULT_MAXAREA)
    parser.add_argument("--cake-min-area", type=int, default=DEFAULT_CAKE_MINAREA)
    parser.add_argument("--donut-min-area", type=int, default=DEFAULT_DONUT_MINAREA)
    parser.add_argument("--aspect-min", type=float, default=DEFAULT_ASPECT_MIN)
    parser.add_argument("--aspect-max", type=float, default=DEFAULT_ASPECT_MAX)
    parser.add_argument("--group-nms-iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--cross-nms-iou", type=float, default=DEFAULT_CROSS_NMS_IOU)
    parser.add_argument("--stable-frames", type=int, default=DEFAULT_STABLE_FRAMES)
    parser.add_argument("--center-tolerance", type=int, default=DEFAULT_CENTER_TOL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.conf = max(0.0, min(1.0, args.conf))
    if args.raw_preview:
        run_raw_preview(args.cam, args.width, args.height, args.show)
        return
    run_mapping(args)


if __name__ == "__main__":
    main()
