# Phase 8 IK to MOVE_SAFE Testing

Phase 8 bridges the dry-run IK result into a guarded `MOVE_SAFE` serial command.

## Scope

Allowed:
- compute IK from a known board or robot target
- validate `elbow_up` servo angles against configured limits
- send `MOVE_SAFE` only after explicit confirmation
- optionally send `HOME` before or after the manual test, also with confirmation

Not allowed:
- YOLO-controlled motion
- autonomous sorting
- pick/place sequence
- gripper close/open sequence
- camera access
- ROS2 integration
- GUI
- firmware changes
- homography or board mapping changes

## Current Safe Target

First dry-run selected target:

- `board_x = 7.0`
- `board_y = 9.0`
- `z = 0.12`
- `solution = elbow_up`
- `tcp_offset_mode = none`

## Safety Rules

- no serial command is sent unless `--send` is used
- `--dry-run` prints the intended `MOVE_SAFE` command only
- `HOME` requires typing `HOME`
- `MOVE_SAFE` requires typing `MOVE`
- `return-home` requires typing `HOME` again after the move
- `CH6` remains fixed at `HOME_SAFE = 45` in this phase
- gripper open/close control is not used yet

## Example Dry Run

```bash
/home/andra/envs/robot_yolo_env/bin/python tools/test_ik_to_move_safe.py --dry-run --board-x 7.0 --board-y 9.0 --z 0.12
```

## Example Later Manual Hardware Test

Not run during validation:

```bash
/home/andra/envs/robot_yolo_env/bin/python tools/test_ik_to_move_safe.py --send --port /dev/ttyUSB0 --board-x 7.0 --board-y 9.0 --z 0.12 --home-first --return-home
```
