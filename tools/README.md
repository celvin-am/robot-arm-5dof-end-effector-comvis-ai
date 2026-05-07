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
    test_aruco_detection.py
    calibrate_aruco_board_homography.py
    calibrate_aruco_tcp_offset.py
    test_aruco_gripper_target_error.py
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
    test_yolo_ik_single_object_sequence.py
    test_yolo_ik_one_cycle_sort.py
    test_yolo_ik_multi_object_sort.py
    calibrate_robot_alignment.py
    yolo_ik_sequence_utils.py

Usage examples:
  python tools/test_yolo_camera.py
  python tools/calibrate_board_homography.py --help
  python tools/test_ik_dry_run.py --help
  python tools/test_esp32_serial_home.py --help

[CONFIG] `servo_pose_calibration_gui.py` is the main calibration interface for
servo limits, gripper calibration, and taught pose calibration.
[IK_REF] Taught poses are for manual sequence validation.
[IK_REF] IK reference samples are coordinate-to-servo calibration data stored
in `ik_reference_samples.yaml`.
[IK_REF] `pre_pick` means a close-above-object pose, ready to grip.
[IK_REF] `safe_hover` means a higher travel-safe pose above the object or path.
[IK_REF] `pick` is an optional lower contact/grasp pose.
[IK_REF] `lift` is the raised pose after gripping.
[IK_REF] `custom` is a free reference mode.
[IK_REF] `gripper_state` records whether the reference was captured open, close,
half_open, or unknown.
[IK_REF] Do not use an old manual pose as exact IK ground truth unless
board_x/y/z was recorded with it.
[CONFIG] Servo direction observations are physical notes, not final IK
calibration.
[SAFETY] Use compare_ik_to_reference reasoning before changing live IK
conversion for CH2/CH3/CH5.
[VIS] Robot Pose Visualization is an approximate operator aid, not a camera
image and not guaranteed exact FK.
[POSE] OPEN_GRIPPER, CLOSE_SOFT, and CLOSE_FULL are action shortcuts that apply
the configured CH6 values to the current MOVE_SAFE pose.
[POSE] MOVE TO CURRENT sends the current slider values as a MOVE_SAFE command.
[POSE] Firmware HOME is not YAML HOME_SAFE.
[CONFIG] The GUI updates `servo_config.yaml` and taught pose YAML only; it does
not change firmware limits.
[SAFETY] Physical validation remains human-confirmed only.
[STATE] Single-object CAKE and DONUT semi-auto cycles are center-area
user-confirmed only.
[STATE] Multi-object sorter is guarded test mode, not final autonomy.
[CONFIG] Center-area validated semi-auto settings are stored in
`semi_auto_pick_place_config.yaml`.
[CONFIG] Validated semi-auto values currently include camera_id=2,
tcp_offset_mode=none, IK servo calibration enabled, z_mode correction enabled,
safe_hover/pre_pick/lift z heights, and class-specific grasp offsets.
[CONFIG] Semi-auto config also records current physical assumptions for object
height (0.005 m) and bowl/container height (0.045 m).
[POSE] Current drop still uses taught poses.
[CONFIG] Future IK-based drop may use the configured pick/place Z policies as
reference only.
[SAFETY] Full-board validation is still pending.
[SAFETY] Full autonomous multi-object sorting is not accepted until repeated
guarded tests pass.
[ARUCO] Board markers IDs 1-4 are diagnostic helpers for board validation and
alternative homography checks.
[ARUCO] Gripper marker ID 0 is for TCP/debug tracking only.
[ARUCO] ArUco does not replace YOLO for object detection.
[ARUCO] ArUco does not replace fixed Z calibration yet.
[ARUCO] No closed-loop control is implemented yet.
[ARUCO] A future phase may use ArUco gripper-vs-target error for correction
after IK motion, but that is not implemented here.
"""
