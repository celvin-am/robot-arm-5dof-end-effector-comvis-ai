# Phase 12 Teach Poses

> [SAFETY]
> Treat this phase as software-only validation unless a human explicitly confirms
> physical robot motion in the current session. Live command examples below are
> for a later manual hardware phase only.

Phase 12 adds a guarded manual teach workflow so we can capture practical
working poses without relying on the current IK model.

## Scope

- uses ESP32 serial plus existing `HOME` and `MOVE_SAFE`
- no IK motion in this phase
- no YOLO, no camera, no homography changes
- no autonomous loop

## Tool

- `tools/teach_servo_poses.py`

## Inputs

- `ros2_ws/src/robot_arm_5dof/config/servo_config.yaml`
- `ros2_ws/src/robot_arm_5dof/config/pose_config.yaml`
- `ros2_ws/src/robot_arm_5dof/config/serial_config.yaml`

## Output

- `ros2_ws/src/robot_arm_5dof/config/taught_pick_place_poses.yaml`

## Workflow

1. Connect to ESP32 serial.
2. Start from `HOME_SAFE`.
3. Jog one channel at a time in small steps.
4. Send `MOVE_SAFE` only after explicit confirmation.
5. When the physical pose looks right, save the current angles as one of:
   - `OBJECT_HOVER`
   - `OBJECT_PICK`
   - `OBJECT_LIFT`
   - `CAKE_BOWL_HOVER`
   - `CAKE_BOWL_PLACE`
   - `DONUT_BOWL_HOVER`
   - `DONUT_BOWL_PLACE`
   - `HOME_SAFE`

## Interactive Commands

- `ch1 +`
- `ch1 -`
- `set ch3 120`
- `send`
- `home`
- `save OBJECT_HOVER`
- `poses`
- `reset`
- `step 5`
- `show`
- `help`
- `quit`

## Safety

- every `MOVE_SAFE` requires typing `MOVE`
- startup `HOME` requires typing `HOME`
- live teach mode requires `--yes-i-understand-hardware-risk`
- all channels are clamped to servo-config min/max
- `--dry-run` never opens serial
- no automatic motion loop is used

## Example Safe Commands

```bash
/home/andra/envs/robot_yolo_env/bin/python tools/teach_servo_poses.py --help
/home/andra/envs/robot_yolo_env/bin/python tools/teach_servo_poses.py --dry-run --port /dev/ttyUSB0
```

## Example Later Manual Session

Not run during validation:

```bash
/home/andra/envs/robot_yolo_env/bin/python tools/teach_servo_poses.py --yes-i-understand-hardware-risk --port /dev/ttyUSB0
```
