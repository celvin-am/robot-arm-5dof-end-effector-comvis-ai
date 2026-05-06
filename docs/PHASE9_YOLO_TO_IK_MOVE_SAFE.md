# Phase 9 YOLO to IK MOVE_SAFE

> [SAFETY]
> Treat this phase as software-only validation unless a human explicitly confirms
> physical robot motion in the current session. Live command examples below are
> for a later manual hardware phase only.

Phase 9 is a single-target hover test that bridges the existing YOLO board
mapping pipeline into the guarded IK-to-`MOVE_SAFE` path.

## Scope

Allowed:
- detect one accepted `CAKE` or `DONUT` target
- map pixel center to board coordinates using the saved homography
- convert board coordinates to robot coordinates
- compute IK with the current validated dry-run settings
- print the final `MOVE_SAFE` command
- optionally send `HOME` and `MOVE_SAFE` only after explicit confirmation

Not allowed:
- autonomous sorting
- pick/place sequence
- gripper close/open
- drop to bowl
- multi-object autonomous loop
- ROS2 integration
- GUI
- firmware changes
- homography changes
- board mapping changes

## Current Hover Settings

- `solution = elbow_up`
- `tcp_offset_mode = none` (temporary manual-validation default only)
- `z hover default = 0.12`
- `CH6 = HOME_SAFE gripper angle = 45`

## Safety Rules

- no serial command is sent unless `--send` is used
- live motion requires `--home-first`
- live motion requires `--yes-i-understand-hardware-risk`
- `--dry-run` prints the intended `MOVE_SAFE` command only
- `HOME` requires typing `HOME`
- `MOVE_SAFE` requires typing `MOVE`
- `return-home` requires typing `HOME` again after the move
- no gripper open/close action is used in this phase
- the tool selects only one accepted inside-board target and then stops

## Example Later Commands

Dry run:

```bash
/home/andra/envs/robot_yolo_env/bin/python tools/test_yolo_to_ik_move_safe.py --dry-run --show
```

Later live test, not run during validation:

```bash
/home/andra/envs/robot_yolo_env/bin/python tools/test_yolo_to_ik_move_safe.py --send --home-first --yes-i-understand-hardware-risk --port /dev/ttyUSB0 --show --tcp-offset-mode none --return-home
```
