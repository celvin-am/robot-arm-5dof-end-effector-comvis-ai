# Phase 6 Serial Testing

> [SAFETY]
> Treat this phase as software-only validation unless a human explicitly confirms
> physical robot motion in the current session. Live command examples below are
> for a later manual hardware phase only.

Phase 6 is limited to a safe serial communication path between the laptop and
ESP32. This phase allows only `HOME_SAFE` and simple diagnostics.

## Scope

Allowed:
- `PING`
- `STATUS`
- `LIMITS`
- `HOME`
- `STOP`
- `HELP`

Not allowed in this phase:
- YOLO-controlled motion
- IK target motion
- Pick/place sequence
- Autonomous sorting
- Camera access
- Homography changes
- ROS2 integration
- GUI

## Physical Channel Mapping Used in Phase 6

This phase follows the current project mapping used for software validation:

- `CH1 GPIO13` = base yaw
- `CH2 GPIO14` = shoulder pitch
- `CH3 GPIO27` = elbow pitch
- `CH4 GPIO26` = wrist yaw / wrist rotate
- `CH5 GPIO25` = wrist pitch / gripper up-down
- `CH6 GPIO33` = gripper open-close

## Safety Rules

- Firmware clamps every requested angle to configured min/max limits.
- `HOME` is the only movement command enabled in this phase.
- Python test tooling requires explicit terminal confirmation before sending `HOME`.
- Validation must not send `HOME`.
- `STOP` is accepted while a `HOME` move is in progress.

## HOME_SAFE

- `CH1=90`
- `CH2=130`
- `CH3=130`
- `CH4=95`
- `CH5=60`
- `CH6=45`

## Later Manual Test

Example command for the later manual hardware phase only:

```bash
/home/andra/envs/robot_yolo_env/bin/python tools/test_esp32_serial_home.py --port /dev/ttyUSB0 --home
```
