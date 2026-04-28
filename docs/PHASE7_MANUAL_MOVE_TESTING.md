# Phase 7 Manual Move Testing

Phase 7 extends the safe serial path with guarded manual servo movement.

## Scope

Allowed firmware commands:
- `PING`
- `STATUS`
- `HOME`
- `MOVE_SAFE ch1 ch2 ch3 ch4 ch5 ch6`
- `STOP`
- `LIMITS`
- `HELP`

Still not allowed:
- YOLO-controlled motion
- IK target motion
- Pick/place sequence
- Autonomous sorting
- Camera access
- ROS2 integration
- GUI or hardware bridge integration

## Safety Rules

- `MOVE_SAFE` is manual only and must be typed explicitly from the test tool.
- Firmware clamps all channels to safe limits before motion.
- Movement remains smooth 1-degree stepping.
- BUSY state rejects everything except `STOP`.
- Python test tool requires explicit confirmation:
  - `Type HOME to send HOME.`
  - `Type MOVE to send MOVE_SAFE.`
- `--dry-run` never opens the serial port.

## Safe Limits

- `CH1 40..140`
- `CH2 40..140`
- `CH3 40..140`
- `CH4 40..140`
- `CH5 40..140`
- `CH6 10..60`

## HOME_SAFE

- `90 130 130 95 60 45`

## Recommended First Physical Test Pose

For the later manual hardware phase only:

- `90 125 125 95 60 45`

Example command, not run during validation:

```bash
/home/andra/envs/robot_yolo_env/bin/python tools/test_esp32_manual_move.py --port /dev/ttyUSB0 --move-safe 90 125 125 95 60 45
```
