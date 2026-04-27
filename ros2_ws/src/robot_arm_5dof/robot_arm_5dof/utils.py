"""
Shared utilities for robot_arm_5dof ROS2 package.
"""

import os
import yaml
import logging
from typing import Optional, Dict, Any, List, Tuple

# -----------------------------------------------------------------------------
# Logging tags
# -----------------------------------------------------------------------------
LOG_VISION   = "[VISION]"
LOG_BOARD    = "[BOARD]"
LOG_IK       = "[IK]"
LOG_SERIAL   = "[SERIAL]"
LOG_GUI      = "[GUI]"
LOG_SAFETY   = "[SAFETY]"
LOG_TASK     = "[TASK]"
LOG_CALIB    = "[CALIB]"


def setup_logger(tag: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with a tag prefix."""
    logger = logging.getLogger(tag)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(f"%(name)s %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger


# -----------------------------------------------------------------------------
# Angle utilities
# -----------------------------------------------------------------------------

def clamp_angle(angle: float, min_angle: float, max_angle: float) -> float:
    """Clamp angle to [min, max] range."""
    return max(min_angle, min(max_angle, angle))


def clamp_all_angles(angles: List[float], limits: List[Tuple[float, float]]) -> List[float]:
    """Clamp a list of angles against a list of (min, max) tuples."""
    return [clamp_angle(a, lo, hi) for a, (lo, hi) in zip(angles, limits)]


def rad_to_deg(rad: float) -> float:
    """Radians to degrees."""
    return rad * 180.0 / 3.141592653589793


def deg_to_rad(deg: float) -> float:
    """Degrees to radians."""
    return deg * 3.141592653589793 / 180.0


# -----------------------------------------------------------------------------
# Config file helpers
# -----------------------------------------------------------------------------

def get_package_share_dir(package_name: str = "robot_arm_5dof") -> str:
    """Return the package share directory path (ROS2/colcon)."""
    # When running under colcon, this resolves correctly.
    # For development, walk up from this file's location.
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(pkg_root)


def _find_src_root() -> str:
    """
    Find the robot_arm_5dof package source root.

    This function exists because colcon installs the package to a different
    directory than the source tree, but config files live in the source tree.

    Search order:
      1. From the installed location (lib/python*/site-packages/robot_arm_5dof/),
         walk up 6 levels to the workspace, then check src/robot_arm_5dof/.
      2. Walk up from this file checking for package.xml (source tree case).

    Returns the path to 'src/robot_arm_5dof/'.
    """
    this_file = os.path.abspath(__file__)

    # Case: installed package (colcon ament_python layout).
    # File:  .../install/robot_arm_5dof/lib/python3.x/site-packages/robot_arm_5dof/utils.py
    # os.path.dirname applied N times from file:
    #   1: .../robot_arm_5dof/           (inside site-packages)
    #   2: .../site-packages/
    #   3: .../python3.x/
    #   4: .../lib/
    #   5: .../robot_arm_5dof/           (install/robot_arm_5dof)
    #   6: .../install/
    #   7: .../ros2_ws/                 <- workspace root (this is what we need)
    install_p = this_file
    for _ in range(7):
        install_p = os.path.dirname(install_p)
    # install_p is now the directory containing both install/ and src/
    src_via_install = os.path.join(install_p, "src", "robot_arm_5dof")
    if os.path.isdir(src_via_install):
        return src_via_install

    # Case: running from source tree. Walk up until we find package.xml.
    current = os.path.dirname(this_file)
    while True:
        if os.path.exists(os.path.join(current, "package.xml")):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # filesystem root
            break
        current = parent

    # Last resort: return the directory two levels up (source layout assumption).
    return os.path.dirname(os.path.dirname(this_file))


def get_config_path(filename: str) -> str:
    """
    Return the full path to a config file relative to the package source.

    Works in both source-tree and installed (colcon) contexts.
    """
    src_root = _find_src_root()
    return os.path.join(src_root, "config", filename)


def load_yaml_config(filename: str) -> Dict[str, Any]:
    """Load a YAML config file and return as dict."""
    path = get_config_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_yaml_config_optional(filename: str) -> Dict[str, Any]:
    """Load a YAML config file, return empty dict if not found."""
    try:
        return load_yaml_config(filename)
    except FileNotFoundError:
        return {}


def save_yaml_config(data: Dict[str, Any], filename: str) -> None:
    """Save a dict to a YAML config file, with a .bak backup of the existing file."""
    path = get_config_path(filename)
    bak_path = path + ".bak"
    if os.path.exists(path):
        import shutil
        shutil.copy2(path, bak_path)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


# -----------------------------------------------------------------------------
# HomographyMapper
# -----------------------------------------------------------------------------

import numpy as np

class HomographyMapper:
    """
    Maps pixel coordinates to board coordinates using a precomputed homography matrix.

    Pipeline:
        pixel (u, v)
            -> undistort using camera mtx + dist
            -> perspectiveTransform using H
            -> board coordinate (x_cm, y_cm)
    """

    def __init__(
        self,
        camera_mtx: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
    ):
        self.camera_mtx = camera_mtx
        self.dist_coeffs = dist_coeffs
        self.H = H

    @classmethod
    def from_npz(cls, calib_path: str, H_path: Optional[str] = None) -> "HomographyMapper":
        """
        Load camera calibration from .npz and optionally a saved homography matrix.

        Expected .npz contents: mtx, dist
        Expected .npy homography: 3x3 float array
        """
        import numpy as np
        data = np.load(calib_path)
        mtx = data["mtx"]
        dist = data["dist"]
        H = None
        if H_path and os.path.exists(H_path):
            H = np.load(H_path)
        return cls(camera_mtx=mtx, dist_coeffs=dist, H=H)

    def undistort_point(self, u: float, v: float) -> Tuple[float, float]:
        """Undistort a single pixel point using the camera calibration."""
        if self.camera_mtx is None or self.dist_coeffs is None:
            return u, v
        import cv2
        pts = np.array([[u, v]], dtype=np.float32)
        undist = cv2.undistortPoints(
            pts, self.camera_mtx, self.dist_coeffs,
            P=self.camera_mtx
        )
        return float(undist[0][0][0]), float(undist[0][0][1])

    def pixel_to_board(self, u: float, v: float) -> Tuple[float, float]:
        """
        Transform a pixel coordinate to board coordinate in cm.

        Requires H to be set.
        Returns (board_x_cm, board_y_cm).
        """
        if self.H is None:
            raise RuntimeError("Homography matrix not set. Compute H first.")
        import cv2
        pt = np.array([[u, v]], dtype=np.float32).reshape(-1, 1, 2)
        board_pt = cv2.perspectiveTransform(pt, self.H)
        return float(board_pt[0][0][0]), float(board_pt[0][0][1])

    def set_homography(self, H: np.ndarray) -> None:
        self.H = H


# -----------------------------------------------------------------------------
# ServoCommandBuilder
# -----------------------------------------------------------------------------

class ServoCommandBuilder:
    """
    Builds and validates ESP32 serial servo commands.

    Command format: S,ch1,ch2,ch3,ch4,ch5,ch6\\n
    All angles in degrees (0-180).
    """

    def __init__(self, servo_config: Dict[str, Any]):
        self._config = servo_config
        self._limits = self._build_limits()

    def _build_limits(self) -> List[Tuple[float, float]]:
        """Build list of (min, max) per channel in order CH1-CH6."""
        limits = []
        for i in range(1, 7):
            ch = f"ch{i}"
            entry = self._config.get("servos", {}).get(ch, {})
            lo = entry.get("min_angle_deg")
            hi = entry.get("max_angle_deg")
            if lo is None or hi is None:
                lo, hi = 0.0, 180.0
            limits.append((lo, hi))
        return limits

    def clamp_all(self, angles_deg: List[float]) -> List[float]:
        """Clamp all 6 channel angles to their configured limits."""
        if len(angles_deg) != 6:
            raise ValueError(f"Expected 6 angles, got {len(angles_deg)}")
        return clamp_all_angles(angles_deg, self._limits)

    def validate(self, angles_deg: List[float]) -> None:
        """Raise ValueError if any angle is out of physical range (0-180)."""
        if len(angles_deg) != 6:
            raise ValueError(f"Expected 6 angles, got {len(angles_deg)}")
        for i, a in enumerate(angles_deg):
            if not (0.0 <= a <= 180.0):
                raise ValueError(
                    f"CH{i+1} angle {a}° is outside physical range [0, 180]"
                )

    def build_command(self, angles_deg: List[float]) -> str:
        """
        Build a serial command string from raw angle values.

        Steps:
        1. Validate raw angles are within [0, 180]
        2. Clamp to configured min/max per channel
        3. Build "S,ch1,ch2,...,ch6\\n" string

        Returns command string ready to send over serial.
        """
        self.validate(angles_deg)
        clamped = self.clamp_all(angles_deg)
        return f"S,{clamped[0]:.1f},{clamped[1]:.1f},{clamped[2]:.1f}," \
               f"{clamped[3]:.1f},{clamped[4]:.1f},{clamped[5]:.1f}\n"


# -----------------------------------------------------------------------------
# Minimal placeholder node for build validation
# -----------------------------------------------------------------------------

def main(args=None):
    """Placeholder main for entry point registration."""
    pass