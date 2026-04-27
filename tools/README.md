"""
robot_arm_5dof_yolo_ws/tools/

Root-level standalone scripts (outside ros2_ws).
Used for vision calibration, homography testing, and ESP32 mock testing.

Phase 1 creates this directory with placeholder stubs only.
Real implementations start in Phase 2 (vision) and Phase 4 (ESP32 mock).

Files:
  test_yolo_camera.py    — Phase 2: standalone YOLO + webcam test
  validate_calibration.py — Phase 2: load kacamata_kamera.npz, test undistortion
  test_homography.py    — Phase 3: checkerboard homography calibration
  mock_esp32.py         — Phase 4: mock ESP32 serial device for testing

Usage (run from workspace root):
  python tools/test_yolo_camera.py
  python tools/validate_calibration.py
  python tools/test_homography.py
  python tools/mock_esp32.py
"""