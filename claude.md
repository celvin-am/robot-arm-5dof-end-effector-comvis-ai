# CLAUDE.md

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
- Inverse kinematics for 5 DOF arm movement.
- Gripper open/close as separate actuator, not part of main DH chain.
- Class-based sorting: all CAKE objects go to CAKE_BOWL (Bowl 1), all DONUT objects go to DONUT_BOWL (Bowl 2).

The project must be developed in small, validated phases. Do not attempt to implement the full autonomous system in one pass.

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

Checkerboard data is fixed:

```text
square size = 3 cm
columns     = 9 squares to the right
rows        = 6 squares downward
board size  = 27 cm x 18 cm
```

Board frame convention:

```text
Origin board = top-left corner of checkerboard
X_board      = rightward direction
Y_board      = downward direction
Z_board      = normal to board surface
```

For OpenCV chessboard corner detection, remember that number of inner corners is not necessarily the same as number of squares. A 9 x 6 square board normally has 8 x 5 inner corners.

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

Final servo mapping, updated from physical ESP32/Arduino calibration evidence:

| Channel | GPIO | Joint | Servo | Motion | Function |
|---|---:|---|---|---|---|
| CH1 | 13 | J1 | MG996R | yaw | base rotation |
| CH2 | 14 | J2 | MG996R | pitch | shoulder up/down |
| CH3 | 27 | J3 | MG996R | pitch | elbow bend |
| CH4 | 26 | J4 | MG90S | pitch | wrist pitch, gripper up/down |
| CH5 | 25 | J5 | SG90 | yaw | wrist yaw / wrist rotate |
| CH6 | 33 | J6 | MG90S | open-close | gripper |

Critical correction:

```text
CH4/J4 is wrist pitch. It moves the gripper up and down.
CH5/J5 is wrist yaw / wrist rotate.
```

The standalone ESP32/Arduino sketch is calibration evidence only for channel
mapping, GPIO mapping, safe limits, HOME_SAFE, and gripper open/close angles.
Do not import its hardcoded pick/place sequence, timing sequence, fixed target
positions, final sorting route, or autonomous motion logic.

### 2.5 Kinematic Chain

Main kinematic chain:

```text
J1 = base yaw
J2 = shoulder pitch
J3 = elbow pitch
J4 = wrist pitch
J5 = wrist yaw
```

Gripper actuator:

```text
J6 = gripper open-close
```

J6 must not be included in the main DH kinematic chain.

Use:

```text
T_robot = T1 * T2 * T3 * T4 * T5
```

Do not use:

```text
T_robot = T1 * T2 * T3 * T4 * T5 * T6
```

J6 is only an actuator command for open/close.

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
- Wrist yaw is neutral.
- Wrist pitch points gripper downward safely.
- Gripper is open.
- TCP/gripper is above the board and does not touch it.

Home must be calibrated on the physical robot and stored as a named pose.

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

---

## 4. Modified DH Baseline

The project uses Modified DH as the documentation and FK baseline.

Initial DH table from project document:

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

DH diagram rule:

```text
z axis of each DH frame follows the physical joint rotation axis.
```

Do not draw all z axes vertically.

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
    pulse_min_us: 600
    pulse_max_us: 2400
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
    pulse_min_us: 600
    pulse_max_us: 2400
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
    pulse_min_us: 600
    pulse_max_us: 2400
  ch4:
    joint: wrist_pitch
    model: MG90S
    gpio_pin: 26
    neutral_angle_deg: 90
    home_angle_deg: 95
    min_angle_deg: 40
    max_angle_deg: 140
    direction: 1
    offset_deg: 0
    pulse_min_us: 600
    pulse_max_us: 2400
  ch5:
    joint: wrist_yaw
    model: SG90
    gpio_pin: 25
    neutral_angle_deg: 90
    home_angle_deg: 60
    min_angle_deg: 40
    max_angle_deg: 140
    direction: 1
    offset_deg: 0
    pulse_min_us: 600
    pulse_max_us: 2400
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
    pulse_min_us: 600
    pulse_max_us: 2400
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
    description: safe ready pose, gripper open, above board
    ch1: 90
    ch2: 130
    ch3: 130
    ch4: 95
    ch5: 60
    ch6: 45
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
  cols_squares: 9
  rows_squares: 6
  width_cm: 27.0
  height_cm: 18.0
  origin: top_left
  x_direction: right
  y_direction: down
  homography_file: config/homography.npy
```

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
  robot_base_x_board_cm: null
  robot_base_y_board_cm: null
  robot_yaw_offset_deg: null
  units_output: meter
```

Use a rotation if the robot is not aligned with board axes. Do not assume perfect alignment unless tested.

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

- Solve IK for J1-J5.
- Apply limits.
- Output joint/servo targets.
- Use geometric IK initially if simpler and safer.
- DH/FK can be used for visualization and validation.

For early physical control, prefer simple geometric IK:

```text
J1 = atan2(y, x)
J2-J3 = planar 2-link IK
J4 = wrist pitch, calibrated to keep the gripper directed safely
J5 = wrist yaw/orientation
```

Wrist compensation calibration note:

```text
The old wrist-pitch compensation assumption is not valid for the physical mapping.
Recalibrate wrist compensation with CH4/J4 as wrist pitch.
Use CH5/J5 for wrist yaw/orientation.
```

Exact sign and offset must be calibrated physically.

### 6.6 `esp32_serial_bridge_node`

Responsibilities:

- Read target servo angles from ROS2.
- Clamp angles to safe limits.
- Send serial command to ESP32.
- Parse ESP32 response.
- Publish ESP32 status.

Suggested serial command:

```text
S,90,75,110,80,90,40\n
```

Meaning:

```text
CH1=90, CH2=75, CH3=110, CH4=80, CH5=90, CH6=40
```

Suggested responses:

```text
OK
ERR
BUSY
DONE
```

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
tools/test_homography.py
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
- Wrist compensation is recalibrated for physical CH4 wrist pitch.
- CH5 wrist yaw/orientation remains calibrated separately.

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

These must be measured or decided physically:

```text
exact CH1-CH6 home angles
exact CH1-CH6 min/max angles
servo direction normal/reverse per channel
actual link lengths between rotation axes
actual robot base location relative to checkerboard
actual robot yaw offset relative to board
valid pick height Z_pick
safe hover height Z_hover
drop zone donut location
drop zone cake location
gripper open and close angles
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
-> IK J1-J5
-> ESP32 serial CH1-CH6
-> pick-and-place donut/cake
```

Locked corrections:

```text
CH4/J4 = wrist pitch, gripper up/down
CH5/J5 = wrist yaw / wrist rotate
J6 = gripper actuator only
servo horns installed at 90 deg neutral
home pose = HOME_SAFE, not all zero
PCA9685 code = reference only, not final hardware bridge
ray casting from TCP = legacy only, not final target localization
```
