# Phase 11 Robot Alignment Correction

> [SAFETY]
> Treat this phase as software-only validation unless a human explicitly confirms
> physical robot motion in the current session. Live command examples below are
> for a later manual hardware phase only.

Phase 11 adds a guarded alignment-correction layer to reduce the miss between
the desired board target and the actual TCP/gripper position.

## Workflow

1. Move to safe hover calibration targets only after explicit confirmation.
2. Show camera preview.
3. Click the observed TCP/gripper center.
   Only click the actual TCP/gripper center. If gripper is not visible,
   reject or skip the sample.
4. Convert the clicked pixel to board coordinates using the existing homography.
5. Compute board-space error:
   - `error_x_cm = target_board_x_cm - actual_tcp_board_x_cm`
   - `error_y_cm = target_board_y_cm - actual_tcp_board_y_cm`
6. Reject clicks that convert outside the board bounds `[0,27] x [0,18] cm`.
7. Explicitly review each sample:
   - type `ACCEPT` to keep it
   - type `REJECT` to retry it, or skip it when `--skip-on-no-click` is used
8. Save only if at least one valid sample remains.
6. Save either:
   - translation-only correction for 1 or 2 valid samples
   - affine correction for 3 or more samples
9. Use that correction in the corrected IK-to-`MOVE_SAFE` hover test.

## Tools

- `tools/calibrate_robot_alignment.py`
- `tools/test_corrected_ik_to_move_safe.py`

## Corrected Hover Safety

`tools/test_corrected_ik_to_move_safe.py` now applies correction
incrementally instead of using the full saved correction at once.

- `--correction-gain` scales the saved correction before it is applied
  - default: `0.3`
- `--max-correction-cm` clamps the applied X and Y correction independently
  for safety
  - default: `5.0`

Applied correction formula:

```text
corrected_target = original_target + clamp(gain * raw_correction, +/- max_correction_cm)
```

The tool prints:

- raw correction
- correction gain
- applied correction
- corrected target

It also rejects corrected board targets outside `[0,27] x [0,18] cm` and
blocks serial send if servo limits fail.

## Safety Rules

- no motion is sent unless `--send` is used
- live calibration motion requires `--yes-i-understand-hardware-risk`
- every calibration move requires typing `MOVE`
- existing correction files require explicit overwrite confirmation
- corrected hover test requires `--home-first`, `--yes-i-understand-hardware-risk`, and explicit `HOME` / `MOVE` confirmations
- no gripper close/open in the calibration tool
- no pick/place sequence in this phase

## Example Software-Safe Commands

```bash
python tools/calibrate_robot_alignment.py --help
python tools/test_corrected_ik_to_move_safe.py --dry-run --board-x 7.0 --board-y 9.0 --z 0.12 --correction-gain 0.3
```

## Example Later Manual Commands

Not run during validation:

```bash
python tools/calibrate_robot_alignment.py --send --yes-i-understand-hardware-risk --port /dev/ttyUSB0 --show --target-set basic
python tools/test_corrected_ik_to_move_safe.py --send --home-first --yes-i-understand-hardware-risk --port /dev/ttyUSB0 --board-x 7.0 --board-y 9.0 --z 0.12 --return-home
```
