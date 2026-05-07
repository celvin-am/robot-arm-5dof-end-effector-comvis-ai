# AGENTS.md

## Project: Robot Arm 5 DOF + Gripper, YOLO Donut/Cake, ROS2, ESP32 Serial

This file is the operating guide for Claude Code or any coding agent working on this project. Treat this document as the primary project context. Do not make broad architectural changes without updating this file and `blueprint.md`.

---

## 1. Project Goal

Build a 5 DOF robot arm plus gripper for **class-based sorting** of donut and cake objects using:

- ROS2 as master software layer.
- ESP32 DevKit V1 as slave actuator controller.
- USB serial communication between ROS2 computer/Raspberry Pi and ESP32.
- Fixed overhead webcam, eye-to-hand configuration.
- YOLO `best.pt` model for donut/cake detection.
- Checkerboard workspace as coordinate reference.
- Homography-based pixel-to-board coordinate mapping.
- Position IK for the main arm positioning chain: CH1 base yaw, CH2 shoulder pitch, CH3 elbow pitch, and CH5 wrist pitch.
- CH4 wrist yaw / wrist rotate is treated as an end-effector orientation actuator for basic pick-and-place.
- Gripper open/close is a separate actuator, not part of the main IK chain.
- Class-based sorting: all CAKE objects go to CAKE_BOWL (Bowl 1), all DONUT objects go to DONUT_BOWL (Bowl 2).

The project must be developed in small, validated phases. Do not attempt to implement the full autonomous system in one pass.

Current guarded semi-auto status:

```text
- Single-object CAKE and DONUT center-area semi-auto cycles have user-confirmed
  validation inputs for camera_id=2, tcp_offset_mode=none, IK servo calibration,
  z_mode correction, z heights, class grasp offsets, and taught drop poses.
- This does NOT mean full-board or full autonomous validation.
- Multi-object sorting remains guarded test mode only, with rescans and explicit
  safety confirmations. It is not final accepted autonomy.
- ArUco board markers (IDs 1-4) and gripper marker (ID 0) are diagnostic
  helpers for board validation and TCP/debug tracking only. They do not
  replace YOLO object detection, fixed Z calibration, or checkerboard
  homography as the main accepted workflow.
```

---

## 2. Fixed System Decisions

These decisions are locked unless the human explicitly changes them.

### 2.1 Camera System

The camera system is **eye-to-hand**.

- Webcam is fixed overhead on a tripod.
- Camera does not move with the robot.
- Do not use TCP/end-effector camera assumptions.
- Do not use ray casting from end-effector/TCP to floor for the final system.

Correct vision pipeline:

```text
webcam frame
-> undistortion using camera calibration npz
-> YOLO best.pt detection
-> bounding box center pixel
-> homography pixel to checkerboard coordinate
-> board coordinate in cm
-> robot coordinate in meter
-> IK target
```

Incorrect final approach:

```text
bbox
-> ray from TCP camera
-> floor intersection
```

The old `ray_cast_to_floor()` logic in `viz_env_node.py` is only a legacy reference and must not be used for the final eye-to-hand system.

### 2.2 Workspace Checkerboard

Checkerboard nominal design:

```text
square size = 3 cm
columns     = 9 squares to the right
rows        = 6 squares downward
board size  = 27 cm x 18 cm
```

Current measured printed checkerboard used for calibration:

```text
width_x_cm  = 28.5
height_y_cm = 18.0
square_size_cm_x = 28.5 / 9 = 3.1666667
square_size_cm_y = 18.0 / 6 = 3.0
board_center = (14.25, 9.0) cm
```

Board frame convention:

```text
Origin board = top-left corner of checkerboard
X_board      = rightward direction
Y_board      = downward direction
Z_board      = normal to board surface
```

For OpenCV chessboard corner detection, remember that number of inner corners is not necessarily the same as number of squares. A 9 x 6 square board normally has 8 x 5 inner corners.

Important calibration rule:

```text
Current homography and board-to-robot work must use the actual measured print size,
not the nominal 27 cm x 18 cm design size.
Changing board dimensions means homography must be recalibrated before trusting
pixel-to-board coordinates.
```

ArUco diagnostic note:

```text
- ArUco board markers may be used to validate or compute an alternative
  homography, but they must not silently replace the current accepted
  checkerboard homography.
- ArUco gripper marker ID 0 may be used to estimate marker/TCP debug error in
  board coordinates.
- No closed-loop correction from ArUco is implemented yet.
```

Current semi-auto center-area validated offsets and heights:

```text
DONUT grasp offset: board_x +0.39 cm, board_y -0.10 cm
CAKE  grasp offset: board_x +0.39 cm, board_y -0.10 cm
safe_hover_z_m: 0.055
pre_pick_z_m:   0.015
lift_z_m:       0.125
```

Current physical height assumptions used for config reference:

```text
object height (cake/donut piece): 0.005 m
container / bowl height:          0.045 m
pre_pick_z_m is interpreted as TCP height from board
pre_pick clearance above object is roughly 0.010 m
```

Drop note:

```text
Current drop still uses taught poses.
Future IK-based drop may use configured container-height and drop-Z policy
references, but this is not yet the accepted motion path and does not imply
full autonomous validation.
```

### 2.3 YOLO Model

Final YOLO model: `best.pt`.

Detected raw classes:

```text
cake
cake 2
donut
donut 1
```

Logical grouping:

```text
cake + cake 2     -> CAKE
donut + donut 1   -> DONUT
```

Do not design the robot task as four separate object categories unless the human explicitly requests it.

### 2.4 Servo Mapping

Final servo mapping, confirmed from the current ESP32 firmware and current
project evidence. Treat any reported hardware-motion validation as unconfirmed
unless a human explicitly reports it in the current session:

| Channel | GPIO | Role | Servo | Motion | Function |
|---|---:|---|---|---|---|
| CH1 | 13 | J1 | MG996R | yaw | base rotation |
| CH2 | 14 | J2 | MG996R | pitch | shoulder up/down |
| CH3 | 27 | J3 | MG996R | pitch | elbow bend |
| CH4 | 26 | EE_ROTATE | MG90S | yaw / roll | wrist yaw / wrist rotate |
| CH5 | 25 | J4_POSITION | SG90 | pitch | wrist pitch, gripper up/down |
| CH6 | 33 | GRIPPER | MG90S | open-close | gripper |

Critical correction:

```text
CH4/GPIO26 is wrist yaw / wrist rotate.
CH5/GPIO25 is wrist pitch / gripper up-down.
```

For basic Cartesian position IK, the active positioning joints are:

```text
CH1 base yaw
CH2 shoulder pitch
CH3 elbow pitch
CH5 wrist pitch / gripper up-down
```

CH4 is an end-effector orientation actuator. It may be held at a calibrated neutral angle during early pick-and-place unless physical testing proves that CH4 significantly changes the TCP position.

CH6 is only the gripper open-close actuator.

The standalone ESP32/Arduino firmware is calibration evidence for channel mapping, GPIO mapping, safe limits, HOME_SAFE, serial protocol, and gripper open/close angles. Do not import any hardcoded pick/place sequence, timing sequence, fixed target positions, final sorting route, or autonomous motion logic from a standalone firmware sketch.

### 2.5 Kinematic Chain

Main positioning chain for position IK:

```text
J1 = base yaw                         -> CH1 / GPIO13
J2 = shoulder pitch                   -> CH2 / GPIO14
J3 = elbow pitch                      -> CH3 / GPIO27
J4 = wrist pitch / gripper up-down    -> CH5 / GPIO25
```

End-effector orientation actuator:

```text
EE_ROTATE = wrist yaw / wrist rotate  -> CH4 / GPIO26
```

Gripper actuator:

```text
GRIPPER = gripper open-close          -> CH6 / GPIO33
```

For basic pick-and-place position IK, use:

```text
T_position = T1 * T2 * T3 * T4 * T_tcp
```

Where:

```text
T1    = base yaw
T2    = shoulder pitch
T3    = elbow pitch
T4    = wrist pitch / gripper up-down
T_tcp = fixed transform from gripper body or wrist assembly to the actual TCP
```

Do not include CH6 in the kinematic chain.

CH4 may be included in URDF/FK for visualization and end-effector orientation, but it should not be an active position-IK variable during early pick-and-place. Keep CH4 at a calibrated neutral angle unless orientation control is explicitly required.

### 2.5.1 URDF and IKPy Mapping Rule

The current URDF may be used as a visual and IKPy baseline only after its wrist joint is interpreted correctly.

Required mapping for IKPy position solving:

```text
URDF base_joint -> CH1 / GPIO13 / base yaw
URDF shoulder   -> CH2 / GPIO14 / shoulder pitch
URDF elbow      -> CH3 / GPIO27 / elbow pitch
URDF wrist      -> CH5 / GPIO25 / wrist pitch / gripper up-down
```

Do not map the URDF `wrist` joint to CH4 unless the URDF joint is physically verified to be wrist rotate. For this project, the position-IK wrist axis must be CH5.

CH4 / GPIO26 / wrist rotate:

```text
- Not active in basic IKPy position IK.
- Keep at wrist_rotate_neutral_deg during pick-and-place.
- May be represented in URDF as an optional end-effector orientation joint.
- May be used later for gripper orientation if needed.
```

CH6 / GPIO33 / gripper:

```text
- Not active in IKPy position IK.
- Controlled separately as open/close.
```

IKPy active link mask must activate only the position chain joints and must not activate CH4 or CH6 for basic XYZ solving.

IKPy output must be converted to ESP32 command order:

```text
[CH1, CH2, CH3, CH4, CH5, CH6]
```

where:

```text
CH1 = IK base yaw result
CH2 = IK shoulder pitch result
CH3 = IK elbow pitch result
CH4 = calibrated wrist rotate neutral or requested rotate angle
CH5 = IK wrist pitch result
CH6 = gripper open/close angle
```

Never send raw IKPy radians directly to the ESP32. Convert using servo calibration:

```text
servo_deg = neutral_angle_deg + direction * rad_to_deg(joint_rad) + offset_deg
```

Then clamp to the configured safe min/max.

### 2.5.2 TCP Definition

TCP means Tool Center Point. For this project, the TCP should be defined as the effective gripper working point, normally:

```text
the center point between the two gripper fingers at the grasping area
```

The URDF must contain a fixed `tcp` link or equivalent terminal tool frame.

Recommended URDF structure:

```text
gripper_link -> fixed joint -> tcp
```

The fixed transform from `gripper_link` or wrist assembly to `tcp` must be measured physically. Do not guess it from the mesh origin.

### 2.5.3 Where URDF Dimensions Belong and What They Are For

URDF dimensions have two different roles. Do not mix them carelessly.

1. URDF geometry/origin data:

```text
- Stored directly in the URDF as <origin xyz="..." rpy="..."> and link/joint definitions.
- Used by robot_state_publisher, TF, RViz visualization, FK tree display, and IKPy chain loading.
- Should represent distances between actual joint rotation axes, not casing length or mesh appearance.
```

2. Calibrated kinematic parameters:

```text
- Stored separately in config/robot_kinematics.yaml.
- Used by geometric IK, FK validation, reachability checks, TCP offset, and debugging.
- Must be measured from the physical robot and may override or validate URDF values.
```

Suggested file:

```text
config/robot_kinematics.yaml
```

Suggested schema:

```yaml
kinematics:
  units: meter
  source: measured_physical_robot
  base_to_shoulder_m: null
  shoulder_to_elbow_m: null
  elbow_to_wrist_pitch_m: null
  wrist_pitch_to_wrist_rotate_m: null
  wrist_rotate_to_tcp_m: null
  wrist_pitch_to_tcp_m: null
  tcp_frame: tcp

ikpy:
  urdf_file: arduino_robot_arm.urdf
  base_link: base
  target_link: tcp
  active_joints:
    - base_joint      # CH1
    - shoulder        # CH2
    - elbow           # CH3
    - wrist           # CH5 wrist pitch
  inactive_actuators:
    ch4: wrist_rotate
    ch6: gripper
```

Current initial kinematic data, updated from URDF-derived estimates plus user physical measurement:

```yaml
kinematics_initial:
  units: meter
  status: initial_draft_not_final_calibration
  source:
    base_to_shoulder_m: urdf_estimate
    shoulder_to_elbow_m: urdf_estimate
    elbow_to_wrist_pitch_m: urdf_estimate
    wrist_pitch_to_wrist_rotate_m: user_physical_measurement
    wrist_rotate_to_tcp_m: user_physical_measurement
    wrist_pitch_to_tcp_direct_m: user_physical_measurement

  lengths_m:
    base_to_shoulder_m: 0.03798
    shoulder_to_elbow_m: 0.11716
    elbow_to_wrist_pitch_m: 0.12683
    wrist_pitch_to_wrist_rotate_m: 0.10
    wrist_rotate_to_tcp_m: 0.14
    wrist_pitch_to_tcp_direct_m: 0.12

  initial_position_ik_tool_length_m: 0.12
  initial_position_ik_tool_length_source: direct straight-line measurement from CH5 wrist pitch axis to TCP

  tcp_reference:
    tcp_definition: center_between_gripper_fingers
    gripper_position: closed
    ch4_wrist_rotate_mode: fixed_neutral_for_initial_ik
    ch4_neutral_deg: null   # candidate from HOME_SAFE is 95 deg, but still validate visually
    ch5_reference_deg: null # still must be defined by wrist pitch reference convention
    ch6_closed_deg: null    # still must be measured for donut/cake grip
```

For initial position IK, use `wrist_pitch_to_tcp_direct_m = 0.12`. Do not use `0.10 + 0.14` as the planar tool length unless the 3D URDF offsets and directions are modeled explicitly. The 0.10 m and 0.14 m values are useful for URDF/visual decomposition, not for the initial simplified IK tool length.

Function of these dimensions:

```text
- FK: estimate where the TCP is for given joint angles.
- IK: compute joint angles needed to reach target X/Y/Z.
- IKPy: build a correct chain from URDF to tcp.
- Safety: reject unreachable targets before servo movement.
- Calibration: compare expected TCP position vs real TCP position.
- GUI/RViz: show the robot and TCP consistently.
```

If URDF dimensions and measured physical dimensions disagree, physical measurement wins for control. URDF must then be updated or documented as visual-only.

### 2.6 Servo Installation Neutral Position

Servo horns/links should be installed when the servo has been commanded to approximately **90 degrees**, not 0 degrees.

Reason:

- 90 degrees is the neutral/midpoint position.
- It provides travel margin in both directions.
- 0 degrees is usually near a mechanical limit and can cause immediate collision or binding.

Required calibration utility:

```text
CENTER_ALL_SERVOS_TO_90
```

Only after centering should horn/link positions be mechanically attached.

### 2.7 Home Pose

Home pose is not “all servo angles = 0 degrees”.

Home pose is:

```text
HOME_SAFE / READY POSE
```

Home pose should follow this concept:

- Base faces the middle of the checkerboard.
- Shoulder is slightly raised.
- Elbow is bent.
- CH4 wrist rotate is neutral.
- CH5 wrist pitch points the gripper downward safely.
- Gripper is open.
- TCP/gripper is above the board and does not touch it.

Home must be calibrated on the physical robot and stored as a named pose.

Current HOME_SAFE validation status:

```yaml
HOME_SAFE:
  ch1: 90
  ch2: 130
  ch3: 130
  ch4: 95
  ch5: 60
  ch6: 45
  status: validated_for_idle_and_manual_testing
  validated_checks:
    no_mechanical_collision: true
    no_servo_hard_buzzing: true
    gripper_clear_of_board: true
    cable_not_tensioned: true
    repeatable_after_power_cycle: true
  not_yet_validated_for:
    - HOME_TO_HOVER_PICK_PATH
    - HOME_TO_DROP_ZONE_PATH
    - autonomous_pick_and_place
```

Do not treat HOME_SAFE as proof that all autonomous paths are safe. HOME_SAFE is currently valid for idle, manual testing, and calibration start pose only.

---

## Class-Based Sorting Task

The final robot task is **class-based sorting**, not single-object pick-and-place.

### Task Definition

All detected CAKE objects on the checkerboard must be sorted into **Bowl 1 (CAKE_BOWL)**.
All detected DONUT objects must be sorted into **Bowl 2 (DONUT_BOWL)**.
Bowls/containers are **fixed calibrated drop zones**, NOT YOLO-detected targets.
YOLO must NOT detect bowls.

### Task Flow

```
1. Detect all valid objects on the checkerboard.
2. Convert each pixel center → board coordinates → robot coordinates.
3. Build a pending object list.
4. Select one pending object (highest confidence, inside board ROI).
5. Pick the object.
6. Select drop zone by class group:
     CAKE  → CAKE_BOWL
     DONUT → DONUT_BOWL
7. Place the object into the correct bowl.
8. Mark the object as done.
9. Repeat until no pending valid objects remain.
10. Return to HOME_SAFE.
```

### Object Data Model

```yaml
object:
  id: int
  group: CAKE | DONUT              # logical class group
  raw_class: cake | cake 2 | donut | donut 1  # YOLO raw output
  confidence: float              # detection confidence
  pixel_u: int
  pixel_v: int
  board_x_cm: float           # from homography
  board_y_cm: float
  robot_x_m: float            # from board-to-robot transform
  robot_y_m: float
  status: pending | selected | picked | placed | done | failed
```

### Drop Zone Model

```yaml
drop_zones:
  cake:
    label: CAKE_BOWL
    container_id: 1
    board_x_cm: null        # fill after bowl placement calibration
    board_y_cm: null
    robot_x_m: null
    robot_y_m: null
    z_hover_m: null
    z_place_m: null
  donut:
    label: DONUT_BOWL
    container_id: 2
    board_x_cm: null
    board_y_cm: null
    robot_x_m: null
    robot_y_m: null
    z_hover_m: null
    z_place_m: null
```

### Target Selection Logic

```python
# Only select objects with status = pending
pending = [obj for obj in object_list if obj.status == "pending"]

# Prefer objects inside checkerboard ROI
valid = [obj for obj in pending if is_inside_board(obj)]

# Prefer higher confidence
valid.sort(key=lambda o: o.confidence, reverse=True)
selected = valid[0] if valid else None
```

Only one object is selected per pick cycle.

### Drop-Zone Selection Logic

```python
if selected is None:
    reject_target()
elif selected.group == "CAKE":
    drop_zone = drop_zones["cake"]
elif selected.group == "DONUT":
    drop_zone = drop_zones["donut"]
else:
    reject_target()
```

### Validity Requirements

A raw YOLO detection must pass ALL of the following before it becomes a pick target:

| # | Filter | Reason |
|---|---|---|
| 1 | Class group = CAKE or DONUT | Reject unknown class |
| 2 | Confidence >= group threshold (cake>=0.40, donut>=0.55) | Reject low-confidence detections |
| 3 | Bbox area >= group min area (cake>=900px², donut>=1500px²) | Reject tiny/sensor-noise detections |
| 4 | Aspect ratio in [0.4, 2.5] | Reject extreme/wrong-shape boxes |
| 5 | Inside checkerboard ROI | Reject objects outside workspace |
| 6 | Temporal stability (CANDIDATE → LOCKED) | Reject flickering detections |
| 7 | Board coordinate within [0, 27]cm × [0, 18]cm | Reject out-of-bounds |
| 8 | IK reachable (within workspace radius) | Reject unreachable targets |
| 9 | Drop zone is calibrated | Reject if bowl position unknown |

### Safety Rules

The robot must NOT move if ANY of these conditions are true:

```
- Object class is unknown or ungrouped
- Object is outside checkerboard ROI
- Object is not TARGET LOCKED (still CANDIDATE)
- Object board coordinate is invalid or out of bounds
- Drop zone is not calibrated
- IK solution fails (out of reach)
- Homography is not computed or invalid
- ESP32 is disconnected
- Emergency stop is active
- Robot is not at HOME_SAFE at start
```

### State Machine (Updated)

```text
IDLE
WAIT_DETECTION
SCAN_WORKSPACE
BUILD_OBJECT_LIST
SELECT_TARGET
MOVE_ABOVE_PICK
DESCEND_PICK
CLOSE_GRIPPER
LIFT_OBJECT
SELECT_DROP_ZONE
MOVE_ABOVE_DROP
DESCEND_DROP
OPEN_GRIPPER
LIFT_CLEAR
MARK_OBJECT_DONE
CHECK_REMAINING_OBJECTS
HOME_RETURN
ERROR_STOP
```

### GUI Dashboard Implications

Dashboard must display:

```
- Detected CAKE objects (orange bbox)
- Detected DONUT objects (green bbox)
- Object list panel: id, group, confidence, board_x, board_y, status
- Selected target highlighted
- CAKE_BOWL position marker
- DONUT_BOWL position marker
- Remaining pending count
- Current state machine state
- ESP32 / serial status
- YOLO filter stats: RAW count, ACC count, REJ count
```

### Phase 3 Implications (Homography is Critical)

Homography calibration in Phase 3 now serves FOUR purposes:

```
1. Object pixel center → board coordinates (cm)
2. Validate object is inside checkerboard ROI
3. CAKE_BOWL board coordinates
4. DONUT_BOWL board coordinates
5. Board-to-robot transform input
```

Without valid homography, none of these are possible.

---


## 3. Existing Source Files and How to Treat Them

### 3.1 `real_camera_node.py`

Use as a vision baseline.

Current behavior:

```text
OpenCV webcam -> YOLOv8 -> single-class filter -> EMA smoothing -> /simulated_yolo_bbox
```

Required final changes:

- Load `best.pt`.
- Support multi-class donut/cake detection.
- Group `cake/cake 2` and `donut/donut 1`.
- Add camera calibration loading from `kacamata_kamera.npz`.
- Add undistortion.
- Add homography or publish detections to a separate board mapper.
- Publish object target data, not only bbox normalized.

Do not keep the final API limited to one `geometry_msgs/Point` containing only bbox x/y/area.

### 3.2 `viz_env_node.py`

Use as GUI/visualizer baseline only.

Current behavior includes:

- Tkinter GUI.
- DH/FK display.
- Joint state visualization.
- Target display.
- YOLO bbox mini-view.
- Legacy ray casting from TCP camera.

Required final changes:

- Convert from eye-on-hand simulator to eye-to-hand dashboard.
- Remove or disable `ray_cast_to_floor()` as final target source.
- Add checkerboard map 9 x 6 squares.
- Display board coordinate and robot coordinate.
- Display robot base relative to board.
- Display donut/cake markers.
- Display drop zones.
- Add calibration tabs.
- Add servo and serial status panels.

### 3.3 `ibvs_controller_node.py`

Use as state machine reference.

Current state machine includes:

```text
IDLE
SEARCH
DIVING
GRABBING
LIFTING
DELIVERING
DROPPING
HOME_RETURN
```

Final state machine should be changed to:

```text
IDLE
WAIT_DETECTION
SCAN_WORKSPACE
BUILD_OBJECT_LIST
SELECT_TARGET
MOVE_ABOVE_PICK
DESCEND_PICK
CLOSE_GRIPPER
LIFT_OBJECT
SELECT_DROP_ZONE
MOVE_ABOVE_DROP
DESCEND_DROP
OPEN_GRIPPER
LIFT_CLEAR
MARK_OBJECT_DONE
CHECK_REMAINING_OBJECTS
HOME_RETURN
ERROR_STOP
```

Do not use arm sweeping as object search. The overhead camera sees the workspace.

### 3.4 `hardware_bridge_node.py`

Do not use as final hardware bridge.

It currently uses PCA9685. Final system must use:

```text
ROS2 -> USB serial -> ESP32 -> servo PWM
```

Create or refactor into:

```text
esp32_serial_bridge_node.py
```

### 3.5 `test_servo.py`

Use only as a conceptual reference for servo testing and smooth sweep movement.

It is not final because it uses PCA9685.

### 3.6 `uas.docx`

Use as kinematic reference.

Important content:

- Raspberry Pi/master and ESP32/slave architecture.
- Base yaw.
- Shoulder pitch.
- Elbow pitch.
- Wrist 1 yaw.
- Wrist 2 pitch.
- Gripper open-close.
- Gripper servo not included in main kinematic analysis.
- Modified DH baseline.

### 3.7 `arduino_robot_arm.urdf`

Use as a visual/IKPy baseline, not as a final truth source until validated.

Current repository note:

- No active URDF/Xacro is present in the repo right now.
- TCP is config-defined only for now.
- If a robot description is added later, it must add a fixed `tcp` link under
  the CH5 wrist-pitch chain and keep CH4 inactive for initial position IK.

Current treatment rules:

- The URDF can represent the position chain if its `wrist` joint is confirmed as CH5 wrist pitch.
- CH4 wrist rotate may be absent from the simplified IK chain and can be held neutral.
- CH6 gripper open-close may be absent from the IK chain and controlled separately.
- A fixed `tcp` frame must be added before using the URDF for final IKPy targets.
- Joint origins and axes must be validated against physical servo rotation axes.
- Link dimensions from URDF must be compared with physical measurements and stored in `config/robot_kinematics.yaml`.

---

## 4. Modified DH / URDF Kinematic Baseline

The project may use Modified DH, geometric IK, and/or IKPy/URDF. All of them must describe the same physical robot, otherwise the math is worthless.

### 4.1 Modified DH Baseline

Initial DH table from the project document:

| Link | a_i (cm) | alpha_i (deg) | d_i (cm) | theta_i |
|---:|---:|---:|---:|---|
| 1 | 0 | 0 | 5 | theta1 |
| 2 | 0 | 90 | 5.5 | theta2 + 90 deg |
| 3 | 12.5 | -180 | 0 | theta3 - 90 deg |
| 4 | 0 | 90 | 9 | theta4 |
| 5 | 0 | -90 | 3 | theta5 - 90 deg |

Use meter units internally in ROS2:

| Parameter | Value |
|---|---:|
| d1 | 0.050 m |
| d2 | 0.055 m |
| a3/link length baseline | 0.125 m |
| d4 | 0.090 m |
| d5 | 0.030 m |

Important:

- These are initial baseline values.
- They must be validated against the physical robot.
- Use distances between servo rotation axes, not casing or bracket length.
- For position IK, the effective wrist pitch axis is CH5, not CH4.
- CH4 wrist rotate is an end-effector orientation actuator unless later testing proves it must affect TCP position.

DH diagram rule:

```text
z axis of each DH frame follows the physical joint rotation axis.
```

Do not draw all z axes vertically.

### 4.2 URDF Dimension Rule

URDF dimensions belong in the URDF when they describe TF/joint/link geometry:

```text
<link>
<joint origin xyz="..." rpy="...">
<axis xyz="...">
```

These values are used for:

```text
robot_state_publisher
TF tree
RViz visualization
IKPy chain parsing
FK from URDF
TCP display
```

However, calibrated physical lengths must also be stored in:

```text
config/robot_kinematics.yaml
```

This config is used for:

```text
geometric IK
FK validation
reachability checks
TCP offset compensation
comparing URDF model vs real robot
```

Required physical measurements:

```text
base yaw axis -> shoulder pitch axis
shoulder pitch axis -> elbow pitch axis
elbow pitch axis -> wrist pitch axis / CH5
wrist pitch axis / CH5 -> wrist rotate axis / CH4
wrist rotate axis / CH4 -> TCP
or simplified: wrist pitch axis / CH5 -> TCP
```

If URDF and physical measurement disagree:

```text
physical measurement wins for control
URDF must be updated or marked visual-only
```

### 4.3 TCP Offset Rule

The TCP must be defined at the effective gripper working point.

For a two-finger gripper:

```text
TCP = center point between the two gripper fingers at the grasping area
```

Do not place TCP at:

```text
- servo center
- mesh origin
- arbitrary gripper body origin
- one finger tip only
```

For IKPy, the target link should be `tcp`, not just `gripper`.

---

## 5. Position Calibration Requirements

No autonomous pick-and-place code should be considered complete until these calibrations exist.

### 5.1 Servo Calibration

Required data per servo:

```text
channel
joint
servo_model
home_angle_deg
min_angle_deg
max_angle_deg
direction
angle_offset_deg
pulse_min_us
pulse_max_us
```

Suggested config file:

```text
config/servo_config.yaml
```

Draft schema:

```yaml
servos:
  ch1:
    joint: base_yaw
    model: MG996R
    gpio_pin: 13
    neutral_angle_deg: 90
    home_angle_deg: 90
    min_angle_deg: 40
    max_angle_deg: 140
    direction: 1
    offset_deg: 0
    pulse_min_us: 500
    pulse_max_us: 2500
  ch2:
    joint: shoulder_pitch
    model: MG996R
    gpio_pin: 14
    neutral_angle_deg: 90
    home_angle_deg: 130
    min_angle_deg: 40
    max_angle_deg: 140
    direction: 1
    offset_deg: 0
    pulse_min_us: 500
    pulse_max_us: 2500
  ch3:
    joint: elbow_pitch
    model: MG996R
    gpio_pin: 27
    neutral_angle_deg: 90
    home_angle_deg: 130
    min_angle_deg: 40
    max_angle_deg: 140
    direction: 1
    offset_deg: 0
    pulse_min_us: 500
    pulse_max_us: 2500
  ch4:
    joint: wrist_rotate
    model: MG90S
    gpio_pin: 26
    neutral_angle_deg: 90
    home_angle_deg: 95
    min_angle_deg: 40
    max_angle_deg: 140
    direction: 1
    offset_deg: 0
    role: end_effector_orientation
    use_in_position_ik: false
    pulse_min_us: 500
    pulse_max_us: 2500
  ch5:
    joint: wrist_pitch
    model: SG90
    gpio_pin: 25
    neutral_angle_deg: 90
    home_angle_deg: 60
    min_angle_deg: 40
    max_angle_deg: 140
    direction: 1
    offset_deg: 0
    role: position_ik
    use_in_position_ik: true
    pulse_min_us: 500
    pulse_max_us: 2500
  ch6:
    joint: gripper
    model: MG90S
    gpio_pin: 33
    neutral_angle_deg: 90
    home_angle_deg: 45
    open_angle_deg: 50
    close_angle_deg: 15
    min_angle_deg: 10
    max_angle_deg: 60
    direction: 1
    offset_deg: 0
    pulse_min_us: 500
    pulse_max_us: 2500
```

### 5.2 Pose Calibration

Required named poses:

```text
HOME_SAFE
READY_ABOVE_BOARD
HOVER_PICK_TEST
PICK_TEST
LIFT_TEST
PLACE_TEST
```

Suggested config file:

```text
config/pose_config.yaml
```

Draft schema:

```yaml
poses:
  HOME_SAFE:
    description: safe ready pose, gripper above board
    ch1: 90
    ch2: 130
    ch3: 130
    ch4: 95
    ch5: 60
    ch6: 45
    status: validated_for_idle_and_manual_testing
    not_yet_validated_for:
      - HOME_TO_HOVER_PICK_PATH
      - HOME_TO_DROP_ZONE_PATH
      - autonomous_pick_and_place
  READY_ABOVE_BOARD:
    description: robot ready above checkerboard workspace
    ch1: null
    ch2: null
    ch3: null
    ch4: null
    ch5: null
    ch6: null
```

Do not guess these values. They must be measured on the physical robot.

Pose finding rule:

```text
1. Start from validated HOME_SAFE.
2. Move one or two servo channels at a time using small increments.
3. Keep CH4 wrist rotate at neutral during position calibration.
4. Keep gripper open when searching hover and pick approach poses.
5. Save only poses that have been manually tested without collision, cable tension, or servo hard buzzing.
```

Manual pose milestones required before autonomous movement:

```text
HOME_SAFE -> READY_ABOVE_BOARD -> HOVER_PICK_TEST -> PICK_TEST -> LIFT_TEST -> HOVER_DROP_TEST -> PLACE_TEST -> LIFT_CLEAR -> HOME_SAFE
```

These named poses are not a replacement for IK. They are calibration and validation anchors used to verify IK, Z heights, and safe path segments.

### 5.3 Camera Calibration

Camera calibration file:

```text
kacamata_kamera.npz
```

Expected content:

```text
mtx  = camera intrinsic matrix
dist = distortion coefficients
```

Use this for undistortion, then compute board homography.

Suggested config file:

```text
config/camera_config.yaml
```

Draft schema:

```yaml
camera:
  camera_id: 0
  frame_width: 640
  frame_height: 480
  calibration_file: kacamata_kamera.npz
  use_undistortion: true
```

### 5.4 Checkerboard Homography Calibration

Goal:

```text
pixel coordinate -> board coordinate in cm
```

Validation points:

```text
top-left corner     -> approximately (0, 0) cm
top-right corner    -> approximately (27, 0) cm
bottom-left corner  -> approximately (0, 18) cm
bottom-right corner -> approximately (27, 18) cm
```

Suggested config file:

```text
config/board_config.yaml
```

Draft schema:

```yaml
board:
  square_size_cm: 3.0
  square_size_cm_x: 3.1666667
  square_size_cm_y: 3.0
  cols_squares: 9
  rows_squares: 6
  width_cm: 28.5
  height_cm: 18.0
  origin: top_left
  x_direction: right
  y_direction: down
  homography_file: config/homography.npy
```

Use nominal `square_size_cm` only for backward compatibility. Current
calibration must use the measured printed board width and height. If those
dimensions change, homography must be recalibrated before trusting
pixel-to-board coordinates.

### 5.5 Board-to-Robot Transform Calibration

Goal:

```text
board coordinate -> robot base coordinate
```

General transform:

```text
P_robot = R * P_board + T
```

Required parameters:

```text
robot_base_x_on_board_cm
robot_base_y_on_board_cm
robot_yaw_offset_deg
```

Suggested config file:

```text
config/robot_board_transform.yaml
```

Draft schema:

```yaml
board_to_robot:
  robot_base_x_board_cm: 14.25
  robot_base_y_board_cm: -9.0
  robot_yaw_offset_deg: -90.0
  units_output: meter
```

Use a rotation if the robot is not aligned with board axes. Do not assume perfect alignment unless tested.
The current measured-initial assumption is that the robot base is centered
above the printed board, 9 cm outside the top edge, and faces the board center.
Geometrically, the base-to-board-center direction lies along board Y+, but the
current transform implementation uses the sign convention where yaw -90 deg maps
board Y+ to positive robot X. Therefore the active config uses `-90.0`, not `+90.0`.

### 5.6 Pick/Place Height Calibration

Camera gives XY only. Z must be preset and calibrated.

Required parameters:

```text
Z_hover
Z_pick
Z_lift
Z_place
Z_clear
```

Suggested config file:

```text
config/pick_place_config.yaml
```

Draft schema:

```yaml
heights_m:
  z_hover: null
  z_pick: null
  z_lift: null
  z_place: null
  z_clear: null

# Class-based sorting — bowls are fixed calibrated drop zones, NOT YOLO-detected targets
drop_zones:
  cake:
    label: CAKE_BOWL
    container_id: 1
    board_x_cm: null        # fill after bowl placement calibration
    board_y_cm: null
    robot_x_m: null
    robot_y_m: null
    z_hover_m: null
    z_place_m: null
  donut:
    label: DONUT_BOWL
    container_id: 2
    board_x_cm: null
    board_y_cm: null
    robot_x_m: null
    robot_y_m: null
    z_hover_m: null
    z_place_m: null
```

---

## 6. ROS2 Node Architecture

Target node architecture:

```text
camera_yolo_node
board_mapper_node
robot_frame_transform_node
task_manager_node
ik_controller_node
esp32_serial_bridge_node
viz_env_node
```

### 6.1 `camera_yolo_node`

Responsibilities:

- Open webcam.
- Load `best.pt`.
- Detect cake/donut.
- Group raw classes.
- Publish detection data.

Initial output can use JSON in `std_msgs/String` to avoid custom message overhead during early prototype.

Suggested topic:

```text
/detections
```

Suggested JSON schema:

```json
{
  "stamp": 0.0,
  "objects": [
    {
      "class_group": "DONUT",
      "raw_class": "donut 1",
      "confidence": 0.87,
      "pixel_u": 342.0,
      "pixel_v": 216.0,
      "bbox": [300, 180, 380, 250]
    }
  ]
}
```

### 6.2 `board_mapper_node`

Responsibilities:

- Load camera calibration.
- Load or compute homography.
- Convert pixel center to board coordinate.
- Publish board targets.

Suggested topic:

```text
/board_objects
```

### 6.3 `robot_frame_transform_node`

Responsibilities:

- Load board-to-robot transform.
- Convert board coordinate cm to robot coordinate meter.
- Publish object targets in robot frame.

Suggested topic:

```text
/object_targets
```

### 6.4 `task_manager_node`

Responsibilities:

- Select target object.
- Lock target.
- Manage pick-and-place state machine.
- Send target poses to IK.
- Send gripper commands.

Final state machine:

```text
IDLE
WAIT_DETECTION
LOCK_TARGET
MOVE_ABOVE_PICK
DESCEND_PICK
CLOSE_GRIPPER
LIFT_OBJECT
MOVE_ABOVE_PLACE
DESCEND_PLACE
OPEN_GRIPPER
LIFT_CLEAR
HOME_RETURN
```

### 6.5 `ik_controller_node`

Responsibilities:

- Solve position IK for CH1, CH2, CH3, and CH5.
- Treat CH4 wrist rotate as an end-effector orientation actuator.
- Treat CH6 gripper as open-close only.
- Apply limits, direction, neutral angle, and offset from servo calibration.
- Output servo targets in ESP32 channel order: CH1, CH2, CH3, CH4, CH5, CH6.
- Use geometric IK or IKPy initially for position-only targets.
- Use DH/FK/URDF for visualization, TCP validation, and debugging.

For early physical control, prefer position-only IK:

```text
CH1 = atan2(y, x) base yaw
CH2-CH3 = planar shoulder/elbow IK
CH5 = wrist pitch compensation to keep the gripper directed safely
CH4 = calibrated wrist rotate neutral
CH6 = gripper open/close
```

Wrist compensation calibration note:

```text
CH5/GPIO25 is the wrist pitch axis that moves the gripper up/down.
CH4/GPIO26 is wrist yaw / wrist rotate and should not be used as the wrist pitch axis.
```

Exact sign, neutral angle, offset, and direction must be calibrated physically.

### 6.6 `esp32_serial_bridge_node`

Responsibilities:

- Read target servo angles from ROS2.
- Clamp angles to safe limits.
- Send serial command to ESP32.
- Parse ESP32 response.
- Publish ESP32 status.

Current ESP32 firmware serial command format:

```text
MOVE_SAFE 90 130 130 95 60 45\n
```

Meaning:

```text
CH1=90, CH2=130, CH3=130, CH4=95, CH5=60, CH6=45
```

Supported commands in the current firmware:

```text
PING
STATUS
HOME
MOVE_SAFE a1 a2 a3 a4 a5 a6
STOP
LIMITS
HELP
```

Expected response patterns:

```text
READY ESP32_ROBOT_ARM_SERIAL
PONG
STATUS READY/BUSY CH1=... CH2=... CH3=... CH4=... CH5=... CH6=...
LIMITS CH1=40..140 CH2=40..140 CH3=40..140 CH4=40..140 CH5=40..140 CH6=10..60
ACK HOME
DONE HOME
ACK MOVE_SAFE
DONE MOVE_SAFE
ACK STOP
DONE STOP
ERR BUSY ...
ERR MALFORMED MOVE_SAFE
ERR UNKNOWN ...
```

Do not switch to an `S,...` serial protocol unless the ESP32 firmware is intentionally changed and documented.

### 6.7 `viz_env_node`

Responsibilities:

- GUI dashboard.
- Eye-to-hand workspace display.
- Camera preview.
- Checkerboard map.
- Object markers.
- Robot base marker.
- Servo table.
- Calibration tabs.
- Logs.

Do not preserve old TCP camera FOV as the final model.

---

## 7. GUI Dashboard Requirements

GUI should be an engineering dashboard, not only a simulator.

Recommended initial framework:

```text
Tkinter, refactored from viz_env_node.py
```

Future optional framework:

```text
PySide6/PyQt after core system is stable
```

Required tabs:

```text
Dashboard
Servo Calibration
Pose Calibration
Board Calibration
Robot Transform
Logs & Debug
```

### 7.1 Dashboard Tab

Must show:

- Camera preview.
- YOLO boxes and class labels.
- Checkerboard map 9 x 6.
- Object markers for DONUT/CAKE.
- Board coordinate.
- Robot coordinate.
- Robot state.
- ESP32 status.
- Main buttons: HOME, START, STOP, E-STOP, OPEN, CLOSE.

### 7.2 Servo Calibration Tab

Must include:

- CH1-CH6 sliders.
- Center all servos to 90 degrees.
- Save home angle.
- Save min/max.
- Reverse direction checkbox.
- Individual servo test.

### 7.3 Pose Calibration Tab

Must include:

- Save HOME_SAFE.
- Save READY_ABOVE_BOARD.
- Save HOVER_PICK.
- Save PICK.
- Save LIFT.
- Save PLACE.

### 7.4 Board Calibration Tab

Must include:

- Camera view.
- Checkerboard corner detection or manual corner selection.
- Homography computation.
- Pixel-to-cm validation.

### 7.5 Robot Transform Tab

Must include:

- Robot base position on board.
- Board-to-robot yaw offset.
- Manual target test.

### 7.6 Logs Tab

Must include:

- ROS2 topic log.
- Serial messages.
- ESP32 responses.
- IK errors.
- Vision errors.

---

## 8. Development Rules for Claude Code

### 8.1 Do Not Make Large Unrequested Refactors

Never rewrite the entire system in one task.

Work phase by phase:

1. Documentation only.
2. Standalone tests.
3. Calibration utilities.
4. ROS2 nodes one by one.
5. Integration.
6. Full pick-and-place.

### 8.2 Keep Hardware Safe

Before sending servo movement commands:

- Confirm servo config exists.
- Clamp angles to min/max.
- Move slowly.
- Never jump from unknown pose to extreme angle.
- Provide emergency stop path.
- Default to HOME_SAFE, not all zero.

### 8.3 Use Config Files

Do not hardcode calibration values in node code once configs exist.

Use YAML for:

```text
servo_config.yaml
pose_config.yaml
camera_config.yaml
board_config.yaml
robot_board_transform.yaml
robot_kinematics.yaml
pick_place_config.yaml
```

### 8.4 Units

Use:

```text
cm for board calibration display
m for ROS2 kinematics and IK
rad for ROS2 joint states
deg for servo commands and GUI display
```

Always convert explicitly.

### 8.5 Preserve Original Files When Unsure

If refactoring major legacy files, create new files first:

```text
camera_yolo_node.py
board_mapper_node.py
esp32_serial_bridge_node.py
```

Do not delete original reference files unless explicitly instructed.

### 8.6 Comments and Logs

Use clear logs:

```text
[VISION]
[BOARD]
[IK]
[SERIAL]
[GUI]
[SAFETY]
```

Keep code comments concise and useful.

---

## 9. MCP Context7 Usage

The human has Context7 MCP available. Use it when implementation depends on current library/API details.

Use Context7 for:

- ROS2 `rclpy` patterns.
- ROS2 message definitions and launch files.
- OpenCV homography and camera calibration APIs.
- Ultralytics YOLO inference APIs.
- PySerial usage.
- Tkinter or PySide6 patterns if needed.
- Arduino/ESP32 servo libraries if firmware is requested.

Do not hallucinate exact library APIs. If unsure, query Context7.

Suggested Context7 lookups:

```text
rclpy publisher subscriber timer parameter Python ROS2
OpenCV findHomography perspectiveTransform undistort Python
Ultralytics YOLO Python predict results boxes names
pyserial Python read write timeout
Tkinter Canvas image update OpenCV frame
ESP32 Arduino Servo PWM serial parsing
```

If Context7 is unavailable, rely on official docs or keep code conservative and clearly mark uncertain assumptions.

---

## 10. Suggested Implementation Phases

### Phase 0: Documentation Lock

Create/update:

```text
blueprint.md
claude.md
```

No code changes except documentation.

### Phase 1: Servo Safety and Calibration

Create:

```text
config/servo_config.yaml
config/pose_config.yaml
tools/servo_center_test.py or ESP32 firmware test
```

Goal:

```text
center all servos at 90 deg
validate CH1-CH6
record min/max/home
```

### Phase 2: Vision Standalone

Create/update:

```text
camera_yolo_node.py or tools/test_yolo_camera.py
```

Goal:

```text
load best.pt
show webcam
show cake/donut detection
print raw_class, class_group, confidence, pixel center
```

### Phase 3: Camera Calibration and Homography

Create:

```text
board_mapper_node.py
tools/calibrate_board_homography.py
config/board_config.yaml
```

Goal:

```text
pixel -> board cm
validate 4 board corners
```

### Phase 4: Board-to-Robot Transform

Create:

```text
robot_frame_transform_node.py
config/robot_board_transform.yaml
```

Goal:

```text
board cm -> robot meters
manual target validation
```

### Phase 5: IK and Manual Motion

Create:

```text
ik_controller_node.py
config/robot_kinematics.yaml
```

Goal:

```text
move to manual X/Y/Z target without YOLO
```

### Phase 6: ESP32 Serial Bridge

Create:

```text
esp32_serial_bridge_node.py
firmware/esp32_servo_controller.ino
```

Goal:

```text
ROS2 sends servo angles
ESP32 moves CH1-CH6 smoothly
```

### Phase 7: GUI Dashboard

Refactor or create:

```text
viz_env_node.py or robot_dashboard_node.py
```

Goal:

```text
camera preview
board map
servo state
robot state
calibration controls
```

### Phase 8: Semi-Auto Pick

Goal:

```text
YOLO detects object
human confirms target
robot picks object
```

### Phase 9: Full Pick-and-Place

Goal:

```text
DONUT -> donut drop zone
CAKE -> cake drop zone
HOME after each cycle
```

---

## 11. Acceptance Criteria

### Vision Acceptance

- `best.pt` loads.
- Detects cake/donut on camera.
- Groups labels correctly.
- Pixel center is stable enough for mapping.

### Board Mapping Acceptance

- Corners map within reasonable tolerance:
  - top-left around `(0, 0)` cm
  - top-right around `(27, 0)` cm
  - bottom-left around `(0, 18)` cm
  - bottom-right around `(27, 18)` cm

### Servo Acceptance

- All servos can center at 90 degrees.
- All servos have safe min/max.
- HOME_SAFE is stored and repeatable.
- Gripper open/close is calibrated.

### Robot Transform Acceptance

- Board target maps to robot target consistently.
- Manual target test reaches expected region.

### IK Acceptance

- IK returns safe angles.
- Angles are within configured limits.
- Position IK controls CH1, CH2, CH3, and CH5.
- CH5 wrist pitch compensation is calibrated to keep the gripper directed safely.
- CH4 wrist rotate is calibrated separately as an end-effector orientation actuator.
- CH6 gripper open-close is calibrated separately.
- FK/URDF validation confirms that TCP reaches the expected target region.

### Full System Acceptance

- Empty workspace produces NO TARGET.
- CAKE on board is detected as CAKE.
- DONUT on board is detected as DONUT.
- Object outside checkerboard ROI is rejected.
- CAKE_BOWL position is calibrated.
- DONUT_BOWL position is calibrated.
- One CAKE object is picked and placed into Bowl 1.
- One DONUT object is picked and placed into Bowl 2.
- Multiple objects are processed one by one.
- Object is marked done after place.
- Robot returns to HOME_SAFE after sorting cycle completes.

---

## 12. Important Warnings

- Do not power servos from ESP32 USB.
- Use external servo power supply.
- Ensure common ground when required by the electronics design.
- Clamp servo commands.
- Never send unknown arbitrary angles to physical robot.
- Do not trust DH parameters until physical calibration confirms them.
- Do not trust camera calibration until homography validation succeeds.
- Do not run autonomous pick-and-place before manual pick succeeds.

---

## 13. Current Open Issues

These must be measured, validated, or decided physically:

Known initial data:

```yaml
HOME_SAFE:
  angles_deg: [90, 130, 130, 95, 60, 45]
  status: validated_for_idle_and_manual_testing

kinematics_initial_m:
  base_to_shoulder: 0.03798
  shoulder_to_elbow: 0.11716
  elbow_to_wrist_pitch: 0.12683
  wrist_pitch_to_wrist_rotate: 0.10
  wrist_rotate_to_tcp: 0.14
  wrist_pitch_to_tcp_direct: 0.12

initial_ik_policy:
  active_position_ik_channels: [CH1, CH2, CH3, CH5]
  fixed_for_position_ik: [CH4]
  gripper_only: CH6
```

Still open / must be measured:

```text
exact CH1-CH6 min/max safe angles after physical range testing
servo direction normal/reverse per channel
CH4 wrist rotate neutral angle, candidate 95 deg from HOME_SAFE but still validate
CH5 wrist pitch reference angle and reference meaning
CH6 open angle and closed angle per object class
final physical validation of URDF joint origins and axes against physical servo axes
final validation of link lengths against physical rotation axes
HOME_TO_HOVER_PICK path safety
HOME_TO_DROP_ZONE path safety
actual robot base location relative to checkerboard
actual robot yaw offset relative to board
valid pick height Z_pick_donut
valid pick height Z_pick_cake
safe hover height Z_hover
lift height Z_lift
place height Z_place
clear height Z_clear
drop zone donut location
drop zone cake location
```

---

## 14. Final Summary for Coding Agent

Build the system as a calibrated eye-to-hand robot arm dashboard and controller.

Do not treat this as an eye-on-hand IBVS robot.

The final system is:

```text
YOLO best.pt + overhead camera
-> checkerboard homography
-> board-to-robot transform
-> position IK using CH1, CH2, CH3, CH5
-> CH4 wrist rotate neutral/optional orientation
-> ESP32 serial CH1-CH6
-> pick-and-place donut/cake
```

Locked corrections:

```text
CH4/GPIO26 = wrist yaw / wrist rotate
CH5/GPIO25 = wrist pitch / gripper up-down
CH6/GPIO33 = gripper actuator only
Position IK uses CH1, CH2, CH3, and CH5
CH4 is treated as an end-effector orientation actuator for basic pick-and-place
TCP must be defined at the gripper working point, usually the center between the gripper fingers
servo horns installed at 90 deg neutral
home pose = HOME_SAFE, not all zero
PCA9685 code = reference only, not final hardware bridge
ray casting from TCP = legacy only, not final target localization
```
