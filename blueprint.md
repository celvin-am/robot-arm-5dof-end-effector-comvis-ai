# Blueprint Sistem Robot Arm 5 DOF + Gripper  
## ROS2 + ESP32 Serial + YOLO Donut/Cake + Eye-to-Hand Checkerboard

**Status dokumen:** Blueprint teknis final awal  
**Tujuan:** Menjadi acuan bersama sebelum implementasi kode, refactor node ROS2, firmware ESP32, GUI, dan kalibrasi posisi.  
**Catatan:** Blueprint ini mengunci rancangan sistem, tetapi nilai sudut servo, offset aktual, dan parameter kalibrasi fisik tetap harus divalidasi langsung pada robot.

---

## 1. Tujuan Sistem

Sistem yang dibangun adalah robot arm **5 DOF + gripper** untuk melakukan **class-based sorting** objek donut dan cake menggunakan kamera overhead, YOLO, ROS2, dan ESP32.

Semua objek CAKE di-sort ke **Bowl 1 (CAKE_BOWL)**.
Semua objek DONUT di-sort ke **Bowl 2 (DONUT_BOWL)**.
Bowl/containers adalah **fixed calibrated drop zones**, BUKAN YOLO-detected targets.
YOLO TIDAK boleh mendeteksi bowl.

Alur kerja utama:

```text
Webcam overhead
→ YOLO best.pt mendeteksi donut/cake
→ ambil titik tengah bbox objek
→ koreksi distorsi kamera
→ homography pixel ke checkerboard
→ koordinat board dalam cm
→ transformasi board ke koordinat robot
→ inverse kinematics 5 DOF
→ ROS2 mengirim sudut servo ke ESP32 via serial
→ ESP32 menggerakkan 6 servo
→ robot sort objek (CAKE→Bowl1, DONUT→Bowl2)
```

Sistem ini menggunakan pendekatan **eye-to-hand**, bukan eye-on-hand. Kamera diam di atas workspace dan tidak ikut bergerak bersama gripper.

---

## 2. Keputusan Arsitektur yang Dikunci

| Komponen | Keputusan Final |
|---|---|
| Tipe sistem kamera | Eye-to-hand, kamera fixed overhead |
| Objek target | Donut dan cake |
| Model YOLO | `best.pt` custom model |
| Workspace | Checkerboard 9 × 6 kotak |
| Ukuran kotak | 3 cm |
| Ukuran board | 27 cm × 18 cm |
| Robot | 5 DOF + 1 gripper |
| Controller utama | ROS2 pada laptop/Raspberry Pi |
| Actuator controller | ESP32 DevKit V1 |
| Komunikasi | Serial USB ROS2 ↔ ESP32 |
| Kinematika utama | J1 sampai J5 |
| Gripper | J6, tidak masuk DH utama |
| Drop zone | CAKE_BOWL (Bowl 1), DONUT_BOWL (Bowl 2), fixed calibrated positions, bukan YOLO target |
| GUI | Eye-to-Hand Robot Arm Dashboard |
| Baseline kode | Kode dosen dipakai sebagai referensi dan direfactor |

---

## 2b. Class-Based Sorting Task

### Task Flow

```
1. Deteksi semua objek valid di checkerboard.
2. Konversi pixel → board (cm) → robot (m).
3. Build pending object list.
4. Pilih satu objek (prioritas: di dalam ROI, confidence tertinggi).
5. Pick objek.
6. Drop zone berdasarkan class group:
     CAKE  → CAKE_BOWL (Bowl 1)
     DONUT → DONUT_BOWL (Bowl 2)
7. Place objek ke bowl yang sesuai.
8. Tandai objek sebagai done.
9. Ulangi sampai tidak ada objek pending.
10. HOME_SAFE.
```

### Object Data Model

```yaml
object:
  id: int
  group: CAKE | DONUT
  raw_class: cake | cake 2 | donut | donut 1
  confidence: float
  pixel_u: int
  pixel_v: int
  board_x_cm: float
  board_y_cm: float
  robot_x_m: float
  robot_y_m: float
  status: pending | selected | picked | placed | done | failed
```

### Drop Zone Model

Bowl adalah **fixed calibrated drop zones**, BUKAN YOLO-detected targets.

```yaml
drop_zones:
  cake:
    label: CAKE_BOWL
    container_id: 1
    board_x_cm: null   # kalibrasi manual
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

### State Machine

```
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

### Target Validity (Semua harus lulus)

| # | Filter | Alasan |
|---|---|---|
| 1 | group = CAKE atau DONUT | Tolak unknown class |
| 2 | confidence >= group threshold | Tolak noise |
| 3 | area >= group min area | Tolak noise sensor |
| 4 | aspect dalam [0.4, 2.5] | Tolak bentuk tidak wajar |
| 5 | Di dalam ROI checkerboard | Tolak objek di luar workspace |
| 6 | TARGET LOCKED (stabil temporal) | Tolak flickering |
| 7 | board coordinate dalam [0,27]×[0,18]cm | Tolak out-of-bounds |
| 8 | IK reachable | Tolak di luar jangkauan |
| 9 | Drop zone sudah dikalibrasi | Tolak bowl belum diketahui |

### Safety

Robot TIDAK boleh bergerak kalau:
- Class tidak dikenal
- Objek di luar ROI checkerboard
- Belum TARGET LOCKED
- Koordinat board invalid
- Drop zone belum dikalibrasi
- IK gagal
- Homography belum dihitung
- ESP32 terputus
- Emergency stop aktif

### Fase 3 Implikasi

Homography sekarang dipakai untuk:
1. Pixel → board (cm) objek
2. Validasi objek di dalam ROI checkerboard
3. Koordinat board untuk CAKE_BOWL
4. Koordinat board untuk DONUT_BOWL
5. Input board-to-robot transform

Tanpa homography valid, tidak ada satupun bisa berjalan.

---



## 3. Sumber Referensi Sistem

Sumber yang menjadi dasar rancangan:

| Sumber | Fungsi dalam Blueprint |
|---|---|
| `real_camera_node.py` | Basis pembacaan webcam dan YOLO |
| `viz_env_node.py` | Basis GUI/visualizer, tetapi harus direfactor menjadi eye-to-hand |
| `ibvs_controller_node.py` | Referensi state machine sorting |
| `hardware_bridge_node.py` | Referensi bridge hardware, diganti serial ESP32 |
| `test_servo.py` | Referensi uji servo dasar |
| `uas.docx` | Referensi DH Modified dan pembagian joint |
| `best.pt` | Model YOLO donut/cake |
| `kacamata_kamera.npz` | Parameter kalibrasi kamera: `mtx` dan `dist` |
| Foto robot fisik | Referensi bentuk joint aktual |
| Foto/sketsa DH | Referensi koreksi arah sumbu joint |

---

## 4. Hardware Robot

### 4.1 Mapping Servo Final

| Channel | Joint | Servo | Gerak | Fungsi |
|---|---|---|---|---|
| CH1 | J1 | MG996R | Yaw | Base putar kiri-kanan |
| CH2 | J2 | MG996R | Pitch | Shoulder/bahu naik-turun |
| CH3 | J3 | MG996R | Pitch | Elbow/siku menekuk |
| CH4 | J4 | MG90S | Yaw | Wrist yaw / rotasi pergelangan |
| CH5 | J5 | SG90 | Pitch | Wrist pitch / gripper atas-bawah |
| CH6 | J6 | MG90S | Open-close | Gripper buka-tutup |

Konfigurasi servo final:

```text
3 × MG996R  → J1, J2, J3
1 × MG90S   → J4
1 × SG90    → J5
1 × MG90S   → J6 gripper
```

Keputusan penting:

```text
J5 adalah wrist pitch.
J5 bergerak atas-bawah.
J5 bukan wrist yaw dan bukan wrist roll.
J6 hanya actuator buka-tutup gripper.
```

---

## 5. Struktur Joint Robot

Urutan joint final:

```text
J1 = Base yaw
J2 = Shoulder pitch
J3 = Elbow pitch
J4 = Wrist yaw
J5 = Wrist pitch
J6 = Gripper buka-tutup
```

Kinematika utama hanya menggunakan:

```text
J1, J2, J3, J4, J5
```

Gripper tidak masuk DH utama:

```text
T_robot = T1 × T2 × T3 × T4 × T5
```

Bukan:

```text
T_robot = T1 × T2 × T3 × T4 × T5 × T6
```

J6 hanya dikontrol sebagai servo buka/tutup.

---

## 6. Konsep DH dan Kinematika

### 6.1 Prinsip Frame DH

Aturan utama:

```text
Sumbu Z pada frame DH harus sejajar dengan sumbu rotasi joint.
```

Untuk robot ini:

| Frame | Origin | Arah Sumbu Z |
|---|---|---|
| `{0}` | Pusat base | Sumbu yaw base, vertikal |
| `{1}` | Pusat shoulder | Sumbu rotasi shoulder pitch |
| `{2}` | Pusat elbow | Sumbu rotasi elbow pitch |
| `{3}` | Pusat wrist yaw | Sumbu rotasi wrist yaw |
| `{4}` | Pusat wrist pitch | Sumbu rotasi wrist pitch |
| `{5}` | TCP / tool | Frame end-effector, bukan servo gripper |

Robot harus digambar sebagai **side-view serial arm**, bukan sebagai robot industri generik.

Bentuk rantai kinematik:

```text
Base
↓
Shoulder pitch
↓
Elbow pitch
↓
Wrist yaw
↓
Wrist pitch
↓
TCP / gripper
```

---

### 6.2 DH Modified Awal dari Dokumen UAS

Dokumen UAS menggunakan **Modified DH**, bukan Standard DH.

Tabel awal:

| Link | `aᵢ` cm | `αᵢ` | `dᵢ` cm | `θᵢ` |
|---:|---:|---:|---:|---|
| 1 | 0 | 0° | 5 | `θ1` |
| 2 | 0 | 90° | 5.5 | `θ2 + 90°` |
| 3 | 12.5 | -180° | 0 | `θ3 - 90°` |
| 4 | 0 | 90° | 9 | `θ4` |
| 5 | 0 | -90° | 3 | `θ5 - 90°` |

Konversi ke meter:

| Parameter | Nilai Meter |
|---|---:|
| `d1` | 0.050 m |
| `d2` | 0.055 m |
| `a3` / link utama | 0.125 m |
| `d4` | 0.090 m |
| `d5` | 0.030 m |

Status tabel:

```text
Baseline awal untuk FK, visualizer, dan dokumentasi.
Belum final sampai divalidasi dengan pengukuran fisik jarak antar sumbu servo.
```

---

### 6.3 Strategi IK

Untuk kontrol awal robot fisik, gunakan **IK geometrik sederhana** lebih dulu:

```text
J1 = arah XY target terhadap base
J2-J3 = planar 2-link IK untuk reach
J4 = wrist yaw sesuai kebutuhan orientasi gripper
J5 = wrist pitch compensation agar gripper mengarah ke bawah
J6 = gripper open-close
```

DH tetap dipakai untuk:

```text
- Forward kinematics
- Visualizer
- Dokumentasi laporan
- Validasi workspace
```

Alasan IK geometrik diprioritaskan:

```text
- Servo hobby punya backlash.
- Parameter fisik bisa berbeda dari CAD/dokumen.
- Pick-and-place hanya membutuhkan XY + Z preset.
- J5 dapat dipakai sebagai kompensasi orientasi gripper.
```

Konsep kompensasi J5:

```text
θ5 ≈ -(θ2 + θ3) + offset
```

Rumus ini belum final, harus dikalibrasi terhadap arah mounting servo.

---

## 7. Servo Mounting dan Posisi Awal Mekanik

### 7.1 Servo Dipasang pada 90° atau 0°?

Keputusan final:

```text
Horn/link servo sebaiknya dipasang saat servo sudah dikomando ke 90°.
```

Bukan saat servo berada pada 0°.

Alasan:

```text
90° adalah posisi tengah/neutral servo.
Memberi ruang gerak ke dua arah.
Lebih aman untuk joint yang butuh gerak naik-turun atau kiri-kanan.
0° biasanya dekat batas gerak dan berisiko langsung mentok.
```

Prosedur standar pemasangan:

```text
1. Lepas horn/link dari servo.
2. Power servo dengan supply stabil.
3. Kirim command ke servo: 90°.
4. Tunggu servo berhenti di posisi tengah.
5. Pasang horn/link sesuai pose mekanik netral.
6. Kencangkan baut horn.
7. Uji gerak kecil: 80° → 90° → 100°.
8. Baru perluas range secara bertahap.
```

---

## 8. Home Pose Standar

### 8.1 Definisi Home

Home bukan semua servo 0°.

Home adalah pose aman:

```text
HOME_SAFE / READY POSE
```

Karakter home standar:

```text
- Base menghadap tengah checkerboard.
- Shoulder agak naik.
- Elbow menekuk.
- Wrist yaw netral.
- Wrist pitch mengarahkan gripper ke bawah.
- Gripper terbuka.
- TCP/gripper berada aman di atas papan.
```

Home tidak boleh:

```text
- Tegap lurus penuh.
- Membungkuk terlalu rendah.
- Membuat gripper menyentuh papan.
- Memaksa servo berada dekat mechanical limit.
```

### 8.2 Pose Operasional

Pose utama yang harus disimpan:

| Pose | Fungsi |
|---|---|
| `HOME_SAFE` | Posisi aman saat startup dan selesai task |
| `READY_ABOVE_BOARD` | Siap kerja di atas checkerboard |
| `HOVER_PICK` | Posisi di atas objek sebelum turun |
| `PICK` | Posisi ambil objek |
| `LIFT_OBJECT` | Posisi naik setelah objek terambil |
| `HOVER_PLACE` | Posisi di atas drop zone |
| `PLACE` | Posisi melepas objek |
| `LIFT_CLEAR` | Posisi clear setelah melepas objek |

Format penyimpanan pose:

```yaml
poses:
  HOME_SAFE:
    ch1: null
    ch2: null
    ch3: null
    ch4: null
    ch5: null
    ch6: null
```

Nilai pose harus diperoleh dari kalibrasi fisik, bukan ditebak.

---

## 9. Workspace Checkerboard dan Kamera

### 9.1 Spesifikasi Board

```text
Ukuran 1 kotak = 3 cm
Jumlah kotak horizontal = 9
Jumlah kotak vertikal = 6
Ukuran total = 27 cm × 18 cm
```

Frame board:

```text
Origin board = pojok kiri atas checkerboard
X_board      = arah kanan
Y_board      = arah bawah
Z_board      = tegak lurus papan
```

### 9.2 Kamera

Kamera dipasang:

```text
- fixed overhead
- menggunakan tripod
- sejajar menghadap papan
- tidak bergerak bersama robot
```

Pipeline kamera:

```text
Frame kamera
→ undistort dengan mtx + dist
→ YOLO best.pt
→ ambil center bbox
→ homography ke board coordinate
```

---

## 10. YOLO Object Detection

Model `best.pt` mendeteksi:

```text
cake
cake 2
donut
donut 1
```

Untuk logika robot:

```text
cake + cake 2     → CAKE
donut + donut 1   → DONUT
```

Output vision final:

```text
class_group
raw_class
confidence
pixel_u
pixel_v
board_x_cm
board_y_cm
robot_x_m
robot_y_m
```

Controller tidak boleh hanya mengandalkan bbox normalized seperti sistem dosen. Target harus sudah dalam koordinat robot.

---

## 11. Kalibrasi Posisi

Kalibrasi posisi dibagi menjadi lima kategori:

```text
1. Kalibrasi servo
2. Kalibrasi home pose
3. Kalibrasi kamera dan checkerboard
4. Kalibrasi transform board ke robot
5. Kalibrasi tinggi pick/place
```

---

### 11.1 Kalibrasi Servo

Setiap servo harus memiliki data:

```text
channel
joint
servo_model
home_angle
min_angle
max_angle
direction
offset
pulse_min
pulse_max
```

Template tabel:

| CH | Joint | Servo | Home | Min | Max | Direction | Catatan |
|---|---|---|---:|---:|---:|---|---|
| CH1 | J1 Base yaw | MG996R | TBD | TBD | TBD | TBD | pusat base |
| CH2 | J2 Shoulder | MG996R | TBD | TBD | TBD | TBD | bahu |
| CH3 | J3 Elbow | MG996R | TBD | TBD | TBD | TBD | siku |
| CH4 | J4 Wrist yaw | MG90S | TBD | TBD | TBD | TBD | rotasi pergelangan |
| CH5 | J5 Wrist pitch | SG90 | TBD | TBD | TBD | TBD | atas-bawah gripper |
| CH6 | Gripper | MG90S | TBD | TBD | TBD | TBD | buka/tutup |

Prosedur:

```text
1. Center servo ke 90°.
2. Pasang horn/link.
3. Uji gerak kecil ±10°.
4. Tentukan arah normal/reverse.
5. Tentukan batas aman min/max.
6. Simpan home angle.
7. Jangan pernah langsung uji 0° atau 180° pada robot terpasang.
```

---

### 11.2 Kalibrasi Home Pose

Target:

```text
Menentukan nilai servo yang menghasilkan HOME_SAFE.
```

Langkah:

```text
1. Servo sudah dipasang pada posisi tengah 90°.
2. Gerakkan CH1 ke arah tengah board.
3. Atur CH2 agar shoulder naik aman.
4. Atur CH3 agar elbow menekuk.
5. Atur CH4 netral.
6. Atur CH5 agar gripper mengarah ke bawah.
7. Atur CH6 gripper terbuka.
8. Simpan sebagai HOME_SAFE.
```

Template:

| Pose | CH1 | CH2 | CH3 | CH4 | CH5 | CH6 |
|---|---:|---:|---:|---:|---:|---:|
| HOME_SAFE | TBD | TBD | TBD | TBD | TBD | TBD |
| READY_ABOVE_BOARD | TBD | TBD | TBD | TBD | TBD | TBD |
| HOVER_PICK_TEST | TBD | TBD | TBD | TBD | TBD | TBD |
| PICK_TEST | TBD | TBD | TBD | TBD | TBD | TBD |
| LIFT_TEST | TBD | TBD | TBD | TBD | TBD | TBD |

---

### 11.3 Kalibrasi Kamera dan Checkerboard

Target:

```text
Pixel kamera → koordinat board dalam cm.
```

Data yang harus disimpan:

```text
camera_id
resolution_width
resolution_height
camera_matrix mtx
distortion_coefficients dist
homography_matrix H
board_width_cm = 27
board_height_cm = 18
square_size_cm = 3
```

Validasi wajib:

| Titik Board | Koordinat yang Diharapkan |
|---|---|
| Pojok kiri atas | `(0, 0)` cm |
| Pojok kanan atas | `(27, 0)` cm |
| Pojok kiri bawah | `(0, 18)` cm |
| Pojok kanan bawah | `(27, 18)` cm |

Toleransi awal:

```text
Error mapping sebaiknya ≤ 1 cm untuk tahap awal.
```

---

### 11.4 Kalibrasi Board ke Robot

Target:

```text
Koordinat board → koordinat robot.
```

Transformasi umum:

```text
P_robot = R × P_board + T
```

Parameter:

```text
robot_base_x_board_cm
robot_base_y_board_cm
robot_yaw_offset_deg
```

Jika robot sejajar dengan board:

```text
X_robot = X_board - offset_x
Y_robot = Y_board - offset_y
```

Jika robot tidak sejajar:

```text
Gunakan rotasi R dan translasi T.
```

Prosedur kalibrasi:

```text
1. Tentukan titik pusat base robot pada koordinat board.
2. Tentukan arah depan robot terhadap board.
3. Ukur yaw offset antara X_robot dan X_board.
4. Uji 3 titik manual di board.
5. Bandingkan target robot dengan posisi fisik gripper.
6. Koreksi offset/rotasi sampai error kecil.
```

---

### 11.5 Kalibrasi Tinggi Pick dan Place

Kamera overhead hanya memberi XY. Z harus preset.

Parameter:

| Parameter | Fungsi | Nilai |
|---|---|---:|
| `Z_hover` | Tinggi aman sebelum turun | TBD |
| `Z_pick` | Tinggi saat mengambil objek | TBD |
| `Z_lift` | Tinggi setelah objek terambil | TBD |
| `Z_place` | Tinggi saat melepas objek | TBD |
| `Z_clear` | Tinggi aman setelah place | TBD |

Langkah:

```text
1. Gerakkan robot ke atas objek tanpa objek.
2. Turunkan perlahan sampai gripper hampir menyentuh objek.
3. Simpan posisi sebagai PICK_TEST.
4. Uji gripper close.
5. Naikkan ke LIFT_TEST.
6. Ulangi untuk donut dan cake.
```

---

## 12. ROS2 Node Final

Node final:

| Node | Fungsi |
|---|---|
| `camera_yolo_node` | Membaca webcam dan menjalankan YOLO `best.pt` |
| `board_mapper_node` | Undistort + homography pixel ke board |
| `robot_frame_transform_node` | Board coordinate ke robot coordinate |
| `task_manager_node` | State machine pick-and-place |
| `ik_controller_node` | Menghitung sudut J1–J5 |
| `esp32_serial_bridge_node` | Mengirim sudut servo ke ESP32 |
| `viz_env_node` | GUI eye-to-hand dashboard |

Untuk tahap awal, beberapa node boleh digabung agar lebih sederhana, tetapi rancangan logis tetap dipisahkan.

---

## 13. Topic ROS2 Usulan

### 13.1 Topic Vision

| Topic | Data | Fungsi |
|---|---|---|
| `/camera/frame` | Image | Frame kamera |
| `/detections` | Detection array | Hasil YOLO |
| `/object_targets` | Object target | Objek dalam koordinat board/robot |
| `/calibration_status` | String/status | Status homography |

### 13.2 Topic Robot

| Topic | Data | Fungsi |
|---|---|---|
| `/joint_states` | JointState | Sudut joint aktual/target |
| `/servo_targets` | Float array | Target servo CH1–CH6 |
| `/servo_states` | Float array/status | Status servo dari ESP32 |
| `/robot_status` | String | State robot |
| `/robot_command` | String | HOME, START, STOP, ESTOP |

### 13.3 Topic GUI

| Topic | Arah | Fungsi |
|---|---|---|
| `/manual_target` | GUI → controller | Target manual |
| `/gripper_command` | GUI → controller | Open/close gripper |
| `/calibration_command` | GUI → calibration node | Perintah kalibrasi |
| `/drop_zone_config` | GUI → task manager | Posisi drop zone |

---

## 14. Serial Protocol ROS2 ↔ ESP32

Format command awal:

```text
S,90,75,110,80,90,40\n
```

Makna:

| Posisi | Channel | Fungsi |
|---|---|---|
| 1 | CH1 | Base yaw |
| 2 | CH2 | Shoulder |
| 3 | CH3 | Elbow |
| 4 | CH4 | Wrist yaw |
| 5 | CH5 | Wrist pitch |
| 6 | CH6 | Gripper |

Balasan ESP32:

```text
OK
ERR
BUSY
DONE
```

Fitur firmware ESP32:

```text
- Attach 6 servo.
- Baca serial.
- Parse command.
- Validasi range servo.
- Smooth movement.
- Kirim ACK/status.
- Jalankan timeout safety.
```

---

## 15. State Machine Final

State final:

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
ERROR_STOP
```

Alur:

```text
HOME_SAFE
→ WAIT_DETECTION
→ LOCK_TARGET
→ MOVE_ABOVE_PICK
→ DESCEND_PICK
→ CLOSE_GRIPPER
→ LIFT_OBJECT
→ MOVE_ABOVE_PLACE
→ DESCEND_PLACE
→ OPEN_GRIPPER
→ LIFT_CLEAR
→ HOME_RETURN
```

State lama `SEARCH` dari sistem dosen tidak dipakai sebagai gerakan mencari objek, karena kamera overhead sudah melihat seluruh workspace.

---

## 16. GUI Final

GUI yang cocok:

```text
Eye-to-Hand Robot Arm Dashboard
```

Panel utama:

```text
1. Camera + YOLO View
2. Checkerboard Board Map
3. Robot Control Panel
4. Servo/Joint Monitor
5. Logs dan ESP32 Status
```

Tab final:

```text
Dashboard
Servo Calibration
Pose Calibration
Board Calibration
Robot Transform Calibration
Logs & Debug
```

Fitur minimum MVP:

```text
- Live camera preview
- BBox YOLO donut/cake
- Board map 9×6
- Object coordinate board/robot
- HOME button
- START PICK button
- STOP button
- E-STOP button
- Open/Close gripper
- Servo table CH1–CH6
- Serial status ESP32
```

GUI bisa dimulai dari `viz_env_node.py`, tetapi konsepnya harus berubah dari simulator eye-on-hand menjadi dashboard eye-to-hand.

---

## 17. Safety System

Safety wajib:

```text
- Servo dipasang pada posisi neutral 90°.
- Semua servo punya min/max angle.
- Semua command disaring oleh clamp.
- Gerakan servo harus smooth, tidak lompat.
- HOME_SAFE selalu tersedia.
- E-STOP langsung menghentikan task.
- Gripper tidak turun jika target tidak valid.
- Robot tidak bergerak jika homography belum valid.
- Robot tidak bergerak jika ESP32 disconnected.
- Serial timeout harus dianggap error.
```

---

## 18. File Konfigurasi Usulan

### 18.1 `servo_config.yaml`

```yaml
servos:
  ch1:
    joint: base_yaw
    model: MG996R
    home: null
    min: null
    max: null
    direction: normal
  ch2:
    joint: shoulder_pitch
    model: MG996R
    home: null
    min: null
    max: null
    direction: normal
  ch3:
    joint: elbow_pitch
    model: MG996R
    home: null
    min: null
    max: null
    direction: normal
  ch4:
    joint: wrist_yaw
    model: MG90S
    home: null
    min: null
    max: null
    direction: normal
  ch5:
    joint: wrist_pitch
    model: SG90
    home: null
    min: null
    max: null
    direction: normal
  ch6:
    joint: gripper
    model: MG90S
    open: null
    close: null
    min: null
    max: null
    direction: normal
```

### 18.2 `calibration_config.yaml`

```yaml
camera:
  camera_id: 0
  calibration_file: kacamata_kamera.npz
  width: null
  height: null

board:
  square_size_cm: 3
  cols: 9
  rows: 6
  width_cm: 27
  height_cm: 18
  origin: top_left

robot_transform:
  base_x_board_cm: null
  base_y_board_cm: null
  yaw_offset_deg: null

heights:
  z_hover: null
  z_pick: null
  z_lift: null
  z_place: null
  z_clear: null
```

---

## 19. Validasi Bertahap

Urutan validasi:

```text
1. Center semua servo ke 90°.
2. Pasang horn/link servo.
3. Test servo satu per satu.
4. Tentukan min/max tiap servo.
5. Simpan HOME_SAFE.
6. Test gripper open/close.
7. Test kamera.
8. Test YOLO best.pt.
9. Test undistortion.
10. Test homography checkerboard.
11. Test board coordinate.
12. Test board-to-robot transform.
13. Test IK ke target manual.
14. Test serial ROS2 → ESP32.
15. Test robot ke HOVER_PICK manual.
16. Test pick manual.
17. Test YOLO-guided pick.
18. Test full pick-and-place donut/cake.
```

---

## 20. Roadmap Implementasi

### Phase 0 — Fixasi dan Dokumentasi

Output:

```text
blueprint.md
servo_config.yaml draft
calibration_config.yaml draft
```

### Phase 1 — Servo dan ESP32

Output:

```text
ESP32 servo test
serial protocol test
servo calibration table
HOME_SAFE pose
```

### Phase 2 — Vision Standalone

Output:

```text
YOLO best.pt berjalan
class CAKE/DONUT terdeteksi
camera preview stabil
```

### Phase 3 — Board Mapping

Output:

```text
undistortion
homography
pixel → board cm
CAKE_BOWL board coordinate (calibrated)
DONUT_BOWL board coordinate (calibrated)
```

### Phase 4 — Board to Robot Transform

Output:

```text
board coordinate → robot coordinate
test 3 titik manual
```

### Phase 5 — IK dan Pose

Output:

```text
IK manual target
HOVER_PICK
PICK
LIFT
PLACE
```

### Phase 6 — ROS2 Integration

Output:

```text
camera_yolo_node
board_mapper_node
task_manager_node
esp32_serial_bridge_node
viz_env_node refactor
```

### Phase 7 — Full Pick-and-Place

Output:

```text
DONUT → drop zone donut
CAKE → drop zone cake
robot kembali HOME_SAFE
```

---

## 21. Open Issues yang Harus Diisi dari Kalibrasi Fisik

| Item | Status |
|---|---|
| Sudut home CH1–CH6 | Belum diukur |
| Min/max CH1–CH6 | Belum diukur |
| Direction normal/reverse | Belum diuji |
| Posisi robot base terhadap board | Belum dikalibrasi |
| Yaw offset board-to-robot | Belum dikalibrasi |
| Z_hover | Belum ditentukan |
| Z_pick | Belum ditentukan |
| Z_place | Belum ditentukan |
| Drop zone CAKE_BOWL board position | Belum dikalibrasi |
| Drop zone DONUT_BOWL board position | Belum dikalibrasi |
| Final DH hasil validasi fisik | Belum final |

---

## 22. Kesimpulan Blueprint

Blueprint final sistem:

```text
Robot arm 5 DOF + gripper
J1 = base yaw
J2 = shoulder pitch
J3 = elbow pitch
J4 = wrist yaw
J5 = wrist pitch atas-bawah
J6 = gripper buka-tutup

Kamera = overhead eye-to-hand
Workspace = checkerboard 27 cm × 18 cm
Model = YOLO best.pt untuk cake/donut
Master = ROS2 pada laptop/Raspberry Pi
Slave = ESP32 DevKit V1
Komunikasi = serial USB
Kinematika utama = J1–J5
Gripper = actuator tambahan, tidak masuk DH utama
Home = HOME_SAFE, bukan semua servo 0°
Servo mounting = center 90° sebelum pasang horn/link
```

Implementasi baru boleh dimulai setelah:

```text
servo center 90° selesai,
HOME_SAFE ditentukan,
YOLO valid,
homography valid,
board-to-robot transform valid,
dan ESP32 serial command stabil.
```

