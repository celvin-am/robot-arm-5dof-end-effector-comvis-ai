# Phase 10 Single-Object Pick-Place

> [SAFETY]
> Treat this phase as software-only validation unless a human explicitly confirms
> physical robot motion in the current session. Live command examples below are
> for a later manual hardware phase only.

Phase 10 is a guarded single-object pick-place test built on the existing YOLO,
board-mapping, IK dry-run, and `MOVE_SAFE` serial path.

## Scope

Allowed:
- detect one stable inside-board `CAKE` or `DONUT`
- map object center to board and robot coordinates
- compute fixed hover, pick, lift, bowl hover, and bowl place poses
- generate `MOVE_SAFE` commands for those poses
- close and open the gripper by reusing `MOVE_SAFE` with different `CH6` values
- send the sequence only after explicit confirmation

Not allowed:
- autonomous sorting loop
- repeated multi-object processing
- ROS2 integration
- GUI
- firmware changes
- homography changes
- config writes

## Fixed Sequence

1. `HOME`
2. object hover
3. object pick
4. gripper close
5. lift
6. bowl hover
7. bowl place
8. gripper open
9. `HOME`

## Safety Rules

- default behavior is dry-run unless `--send` is passed
- live motion requires `--yes-i-understand-hardware-risk`
- `--send` requires typing `START`
- `--confirm-each-step` requires `ENTER` before every `MOVE_SAFE`
- each pose must pass IK and servo-limit validation before the sequence starts
- if the target changes or is lost before start, the live run aborts
- no gripper command is sent separately; gripper actions are encoded in `MOVE_SAFE`

## Example Commands

Dry run:

```bash
/home/andra/envs/robot_yolo_env/bin/python tools/test_single_object_pick_place.py --dry-run --target-group ANY
```

Later guarded live test, not run during validation:

```bash
/home/andra/envs/robot_yolo_env/bin/python tools/test_single_object_pick_place.py --send --yes-i-understand-hardware-risk --port /dev/ttyUSB0 --target-group ANY --confirm-each-step --tcp-offset-mode none
```
