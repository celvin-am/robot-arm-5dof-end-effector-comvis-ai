# Phase 11 Robot Alignment Correction

Phase 11 adds a guarded alignment-correction layer to reduce the miss between
the desired board target and the actual TCP/gripper position.

## Workflow

1. Move to safe hover calibration targets only after explicit confirmation.
2. Show camera preview.
3. Click the observed TCP/gripper center.
4. Convert the clicked pixel to board coordinates using the existing homography.
5. Compute board-space error:
   - `error_x_cm = target_board_x_cm - actual_tcp_board_x_cm`
   - `error_y_cm = target_board_y_cm - actual_tcp_board_y_cm`
6. Save either:
   - translation-only correction for 1 sample
   - affine correction for 3 or more samples
7. Use that correction in the corrected IK-to-`MOVE_SAFE` hover test.

## Tools

- `tools/calibrate_robot_alignment.py`
- `tools/test_corrected_ik_to_move_safe.py`

## Safety Rules

- no motion is sent unless `--send` is used
- every calibration move requires typing `MOVE`
- corrected hover test requires explicit `HOME` and `MOVE` confirmations
- no gripper close/open in the calibration tool
- no pick/place sequence in this phase

## Example Software-Safe Commands

```bash
python tools/calibrate_robot_alignment.py --help
python tools/test_corrected_ik_to_move_safe.py --dry-run --board-x 7.0 --board-y 9.0 --z 0.12
```

## Example Later Manual Commands

Not run during validation:

```bash
python tools/calibrate_robot_alignment.py --send --port /dev/ttyUSB0 --show --target-set basic
python tools/test_corrected_ik_to_move_safe.py --send --port /dev/ttyUSB0 --board-x 7.0 --board-y 9.0 --z 0.12 --home-first --return-home
```
