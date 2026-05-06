"""
robot_arm_5dof_yolo_ws/tools/

Root-level standalone scripts for staged robot-arm development and validation.

[SAFETY] Treat repository validation results as software-only unless a human
explicitly confirms physical robot motion in the current session.

Main tool groups:
  Vision / mapping
    test_yolo_camera.py
    calibrate_board_homography.py
    test_yolo_board_mapping.py
    validate_calibration.py

  IK / transforms
    test_board_to_robot_transform.py
    test_ik_dry_run.py
    analyze_ik_servo_convention.py
    sweep_ik_targets.py

  Serial / manual hardware diagnostics
    test_esp32_serial_home.py
    test_esp32_manual_move.py
    teach_servo_poses.py
    servo_pose_calibration_gui.py  # main calibration interface

  Guarded integrated tests
    test_ik_to_move_safe.py
    test_corrected_ik_to_move_safe.py
    test_yolo_to_ik_move_safe.py
    test_single_object_pick_place.py
    calibrate_robot_alignment.py

Usage examples:
  python tools/test_yolo_camera.py
  python tools/calibrate_board_homography.py --help
  python tools/test_ik_dry_run.py --help
  python tools/test_esp32_serial_home.py --help

[CONFIG] `servo_pose_calibration_gui.py` is the main calibration interface for
servo limits, gripper calibration, and taught pose calibration.
[POSE] Firmware HOME is not YAML HOME_SAFE.
[CONFIG] The GUI updates `servo_config.yaml` and taught pose YAML only; it does
not change firmware limits.
[SAFETY] Physical validation remains human-confirmed only.
"""
