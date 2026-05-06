#include <ESP32Servo.h>

/*
  ESP32 Robot Arm Serial Firmware - Calibration Jog Version

  Purpose:
  - Easy HOME / HOVER / PICK / LIFT / DROP pose calibration.
  - Jog selected channel using + and -.
  - Always report current status after motion.
  - Preserve existing protocol: HOME, STATUS, LIMITS, MOVE_SAFE, STOP.

  Final physical mapping:
  CH1 GPIO13 = base yaw
  CH2 GPIO14 = shoulder pitch
  CH3 GPIO27 = elbow pitch
  CH4 GPIO26 = wrist yaw / wrist rotate
  CH5 GPIO25 = wrist pitch / gripper up-down
  CH6 GPIO33 = gripper open-close

  Notes:
  - Servo angles are commanded angles, not encoder feedback.
  - CH4 is normally fixed neutral for initial position IK.
  - CH6 is gripper open-close only.
*/

namespace {

constexpr uint32_t BAUD_RATE = 115200;
constexpr size_t SERVO_COUNT = 6;
constexpr uint8_t SERVO_PINS[SERVO_COUNT] = {13, 14, 27, 26, 25, 33};

const char *CHANNEL_NAMES[SERVO_COUNT] = {
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_rotate",
    "wrist_pitch",
    "gripper"};

const char *CHANNEL_ROLES[SERVO_COUNT] = {
    "CH1 GPIO13 base yaw",
    "CH2 GPIO14 shoulder pitch",
    "CH3 GPIO27 elbow pitch",
    "CH4 GPIO26 wrist yaw / wrist rotate",
    "CH5 GPIO25 wrist pitch / gripper up-down",
    "CH6 GPIO33 gripper open-close"};

constexpr int MIN_ANGLES[SERVO_COUNT] = {40, 40, 40, 40, 40, 10};
constexpr int MAX_ANGLES[SERVO_COUNT] = {140, 140, 140, 140, 140, 60};

constexpr int HOME_ANGLES[SERVO_COUNT] = {90, 130, 130, 95, 60, 45};

// Installation center: do not force CH6 to 90 because gripper safe range is 10..60.
constexpr int CENTER90_ANGLES[SERVO_COUNT] = {90, 90, 90, 90, 90, 45};

constexpr size_t INPUT_BUFFER_SIZE = 128;

Servo servos[SERVO_COUNT];
int currentAngles[SERVO_COUNT];

bool busyState = false;
bool stopRequested = false;

char inputBuffer[INPUT_BUFFER_SIZE];
size_t inputLength = 0;

int stepDelayMs = 20;
int jogStepDeg = 5;
size_t selectedChannel = 0;  // 0-based index, default CH1.

int clampAngle(size_t index, int angle) {
  if (angle < MIN_ANGLES[index]) {
    return MIN_ANGLES[index];
  }
  if (angle > MAX_ANGLES[index]) {
    return MAX_ANGLES[index];
  }
  return angle;
}

void clearInputBuffer() {
  inputLength = 0;
  inputBuffer[0] = '\0';
}

void trimLine(char *line) {
  size_t start = 0;
  while (line[start] == ' ' || line[start] == '\t' || line[start] == '\r' || line[start] == '\n') {
    ++start;
  }

  size_t end = strlen(line);
  while (end > start) {
    char c = line[end - 1];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      --end;
    } else {
      break;
    }
  }

  if (start > 0) {
    memmove(line, line + start, end - start);
  }
  line[end - start] = '\0';
}

void toUpperInPlace(char *line) {
  for (size_t i = 0; line[i] != '\0'; ++i) {
    line[i] = static_cast<char>(toupper(static_cast<unsigned char>(line[i])));
  }
}

void printMap() {
  Serial.println("MAP BEGIN");
  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    Serial.print("  CH");
    Serial.print(i + 1);
    Serial.print(" GPIO");
    Serial.print(SERVO_PINS[i]);
    Serial.print(" ");
    Serial.println(CHANNEL_NAMES[i]);
  }
  Serial.println("MAP END");
}

void printLimits() {
  Serial.print("LIMITS ");
  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    Serial.print("CH");
    Serial.print(i + 1);
    Serial.print('=');
    Serial.print(MIN_ANGLES[i]);
    Serial.print("..");
    Serial.print(MAX_ANGLES[i]);
    if (i + 1 < SERVO_COUNT) {
      Serial.print(' ');
    }
  }
  Serial.println();
}

void printStatus() {
  Serial.print("STATUS ");
  Serial.print(busyState ? "BUSY " : "READY ");
  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    Serial.print("CH");
    Serial.print(i + 1);
    Serial.print('=');
    Serial.print(currentAngles[i]);
    if (i + 1 < SERVO_COUNT) {
      Serial.print(' ');
    }
  }
  Serial.print(" SELECTED=CH");
  Serial.print(selectedChannel + 1);
  Serial.print(" STEP=");
  Serial.print(jogStepDeg);
  Serial.print(" DELAY_MS=");
  Serial.print(stepDelayMs);
  Serial.println();
}

void printStatusVerbose() {
  Serial.println("STATUS_VERBOSE BEGIN");
  Serial.print("  state: ");
  Serial.println(busyState ? "BUSY" : "READY");
  Serial.print("  selected_channel: CH");
  Serial.println(selectedChannel + 1);
  Serial.print("  jog_step_deg: ");
  Serial.println(jogStepDeg);
  Serial.print("  step_delay_ms: ");
  Serial.println(stepDelayMs);

  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    Serial.print("  CH");
    Serial.print(i + 1);
    Serial.print(": gpio=");
    Serial.print(SERVO_PINS[i]);
    Serial.print(" name=");
    Serial.print(CHANNEL_NAMES[i]);
    Serial.print(" current=");
    Serial.print(currentAngles[i]);
    Serial.print(" min=");
    Serial.print(MIN_ANGLES[i]);
    Serial.print(" max=");
    Serial.print(MAX_ANGLES[i]);
    Serial.print(" home=");
    Serial.println(HOME_ANGLES[i]);
  }
  Serial.println("STATUS_VERBOSE END");
}

void printPoseYaml(const char *poseName) {
  Serial.print("POSE_YAML ");
  Serial.println(poseName);
  Serial.print(poseName);
  Serial.println(":");
  Serial.println("  description: captured from ESP32 commanded angles");
  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    Serial.print("  ch");
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.println(currentAngles[i]);
  }
}

void printReportAfterMotion(const char *label) {
  Serial.print("REPORT ");
  Serial.println(label);
  printStatus();
  printPoseYaml("current_pose");
}

void writeServoAngles(const int angles[SERVO_COUNT]) {
  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    int bounded = clampAngle(i, angles[i]);
    servos[i].write(bounded);
    currentAngles[i] = bounded;
  }
}

void printHelp() {
  Serial.println("HELP");
  Serial.println("  Basic:");
  Serial.println("    PING");
  Serial.println("    STATUS");
  Serial.println("    STATUS_VERBOSE");
  Serial.println("    MAP");
  Serial.println("    LIMITS");
  Serial.println("    HELP");
  Serial.println("  Motion:");
  Serial.println("    HOME");
  Serial.println("    CENTER90");
  Serial.println("    MOVE_SAFE a1 a2 a3 a4 a5 a6");
  Serial.println("    MOVE_ONE ch angle");
  Serial.println("    JOG ch delta");
  Serial.println("    STOP");
  Serial.println("  Easy calibration:");
  Serial.println("    MENU");
  Serial.println("    SELECT ch       or SEL ch");
  Serial.println("    NEXT            select next channel");
  Serial.println("    PREV            select previous channel");
  Serial.println("    STEP deg        set + / - jog step");
  Serial.println("    +               jog selected channel +STEP");
  Serial.println("    -               jog selected channel -STEP");
  Serial.println("    ++              jog selected channel +10");
  Serial.println("    --              jog selected channel -10");
  Serial.println("    POSE_YAML       print current pose as YAML");
  Serial.println("    OPEN            CH6 gripper to 50");
  Serial.println("    CLOSE_SOFT      CH6 gripper to 35");
  Serial.println("    CLOSE_FULL      CH6 gripper to 15");
  Serial.println("  Presets:");
  Serial.println("    1 or PRESET 1   HOME_SAFE");
  Serial.println("    2 or PRESET 2   CENTER90_INSTALL");
  Serial.println("    3 or PRESET 3   GRIPPER_OPEN_SAFE");
  Serial.println("    4 or PRESET 4   GRIPPER_CLOSE_SOFT");
  Serial.println("    5 or PRESET 5   GRIPPER_CLOSE_FULL");
  Serial.println("    6 or PRESET 6   JOG CH1 +10");
  Serial.println("    7 or PRESET 7   JOG CH1 -10");
  Serial.println("    8 or PRESET 8   JOG CH5 +10");
  Serial.println("    9 or PRESET 9   JOG CH5 -10");
  Serial.println("    10 or PRESET 10 STATUS_VERBOSE + POSE_YAML");
}

void printMenu() {
  Serial.println("MENU BEGIN");
  Serial.println("  Selected channel jog:");
  Serial.print("    selected = CH");
  Serial.print(selectedChannel + 1);
  Serial.print(" ");
  Serial.print(CHANNEL_NAMES[selectedChannel]);
  Serial.print(", step = ");
  Serial.print(jogStepDeg);
  Serial.println(" deg");
  Serial.println("    SELECT 1..6, NEXT, PREV, STEP n");
  Serial.println("    +   selected channel +STEP");
  Serial.println("    -   selected channel -STEP");
  Serial.println("    ++  selected channel +10");
  Serial.println("    --  selected channel -10");
  Serial.println();
  Serial.println("  Presets:");
  Serial.println("    1  HOME_SAFE");
  Serial.println("    2  CENTER90_INSTALL");
  Serial.println("    3  GRIPPER_OPEN_SAFE");
  Serial.println("    4  GRIPPER_CLOSE_SOFT");
  Serial.println("    5  GRIPPER_CLOSE_FULL");
  Serial.println("    6  JOG_CH1_BASE_PLUS_10");
  Serial.println("    7  JOG_CH1_BASE_MINUS_10");
  Serial.println("    8  JOG_CH5_WRIST_PLUS_10");
  Serial.println("    9  JOG_CH5_WRIST_MINUS_10");
  Serial.println("    10 STATUS_VERBOSE_AND_POSE_YAML");
  Serial.println("MENU END");
  printStatus();
}

void printPlan(const char *label, const int requestedAngles[SERVO_COUNT], const int targetAngles[SERVO_COUNT]) {
  Serial.print("PLAN ");
  Serial.println(label);

  Serial.print("FROM ");
  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    Serial.print("CH");
    Serial.print(i + 1);
    Serial.print('=');
    Serial.print(currentAngles[i]);
    if (i + 1 < SERVO_COUNT) {
      Serial.print(' ');
    }
  }
  Serial.println();

  Serial.print("REQUEST ");
  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    Serial.print("CH");
    Serial.print(i + 1);
    Serial.print('=');
    Serial.print(requestedAngles[i]);
    if (i + 1 < SERVO_COUNT) {
      Serial.print(' ');
    }
  }
  Serial.println();

  Serial.print("TARGET ");
  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    Serial.print("CH");
    Serial.print(i + 1);
    Serial.print('=');
    Serial.print(targetAngles[i]);
    if (i + 1 < SERVO_COUNT) {
      Serial.print(' ');
    }
  }
  Serial.println();

  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    if (requestedAngles[i] != targetAngles[i]) {
      Serial.print("WARN CLAMP CH");
      Serial.print(i + 1);
      Serial.print(' ');
      Serial.print(requestedAngles[i]);
      Serial.print("->");
      Serial.print(targetAngles[i]);
      Serial.print(" limit=");
      Serial.print(MIN_ANGLES[i]);
      Serial.print("..");
      Serial.println(MAX_ANGLES[i]);
    }
  }
}

void processBusyCommand(char *line) {
  trimLine(line);
  toUpperInPlace(line);
  if (strcmp(line, "STOP") == 0) {
    if (!stopRequested) {
      stopRequested = true;
      Serial.println("ACK STOP");
    }
  } else if (strcmp(line, "STATUS") == 0) {
    printStatus();
  } else if (line[0] != '\0') {
    Serial.print("ERR BUSY ");
    Serial.println(line);
  }
}

void pollSerialWhileBusy() {
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      inputBuffer[inputLength] = '\0';
      processBusyCommand(inputBuffer);
      clearInputBuffer();
      continue;
    }
    if (inputLength + 1 < INPUT_BUFFER_SIZE) {
      inputBuffer[inputLength++] = c;
      inputBuffer[inputLength] = '\0';
    } else {
      clearInputBuffer();
      Serial.println("ERR LINE_TOO_LONG");
    }
  }
}

void moveToAngles(const int requestedAngles[SERVO_COUNT], const char *label) {
  int targetAngles[SERVO_COUNT];
  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    targetAngles[i] = clampAngle(i, requestedAngles[i]);
  }

  printPlan(label, requestedAngles, targetAngles);

  busyState = true;
  stopRequested = false;

  bool anyChanged = true;
  while (anyChanged && !stopRequested) {
    anyChanged = false;
    for (size_t i = 0; i < SERVO_COUNT; ++i) {
      if (currentAngles[i] < targetAngles[i]) {
        ++currentAngles[i];
        anyChanged = true;
      } else if (currentAngles[i] > targetAngles[i]) {
        --currentAngles[i];
        anyChanged = true;
      }
      servos[i].write(currentAngles[i]);
    }
    pollSerialWhileBusy();
    delay(stepDelayMs);
  }

  busyState = false;
  if (stopRequested) {
    Serial.println("DONE STOP");
  } else {
    Serial.print("DONE ");
    Serial.println(label);
  }

  printReportAfterMotion(label);
}

bool parseMoveSafeAngles(const char *line, int angles[SERVO_COUNT]) {
  int parsed[SERVO_COUNT];
  char tail = '\0';
  int count = sscanf(
      line,
      "MOVE_SAFE %d %d %d %d %d %d %c",
      &parsed[0],
      &parsed[1],
      &parsed[2],
      &parsed[3],
      &parsed[4],
      &parsed[5],
      &tail);
  if (count != static_cast<int>(SERVO_COUNT)) {
    return false;
  }

  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    angles[i] = parsed[i];
  }
  return true;
}

bool parseTwoInts(const char *line, const char *prefix, int &a, int &b) {
  char tail = '\0';
  char format[32];
  snprintf(format, sizeof(format), "%s %%d %%d %%c", prefix);
  int count = sscanf(line, format, &a, &b, &tail);
  return count == 2;
}

bool parseOneInt(const char *line, const char *prefix, int &a) {
  char tail = '\0';
  char format[32];
  snprintf(format, sizeof(format), "%s %%d %%c", prefix);
  int count = sscanf(line, format, &a, &tail);
  return count == 1;
}

void copyCurrentTo(int angles[SERVO_COUNT]) {
  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    angles[i] = currentAngles[i];
  }
}

void moveOneChannel(int channel1Based, int angle, const char *label) {
  if (channel1Based < 1 || channel1Based > static_cast<int>(SERVO_COUNT)) {
    Serial.print("ERR BAD_CHANNEL ");
    Serial.println(channel1Based);
    return;
  }

  int requested[SERVO_COUNT];
  copyCurrentTo(requested);
  requested[channel1Based - 1] = angle;

  Serial.print("ACK ");
  Serial.println(label);
  moveToAngles(requested, label);
}

void jogChannel(int channel1Based, int delta, const char *label) {
  if (channel1Based < 1 || channel1Based > static_cast<int>(SERVO_COUNT)) {
    Serial.print("ERR BAD_CHANNEL ");
    Serial.println(channel1Based);
    return;
  }

  int requested[SERVO_COUNT];
  copyCurrentTo(requested);
  requested[channel1Based - 1] = currentAngles[channel1Based - 1] + delta;

  Serial.print("ACK ");
  Serial.println(label);
  moveToAngles(requested, label);
}

void selectChannel(int channel1Based) {
  if (channel1Based < 1 || channel1Based > static_cast<int>(SERVO_COUNT)) {
    Serial.print("ERR BAD_CHANNEL ");
    Serial.println(channel1Based);
    return;
  }
  selectedChannel = static_cast<size_t>(channel1Based - 1);
  Serial.print("SELECTED CH");
  Serial.print(selectedChannel + 1);
  Serial.print(" ");
  Serial.println(CHANNEL_NAMES[selectedChannel]);
  printStatus();
}

void handlePreset(int preset) {
  int requested[SERVO_COUNT];

  switch (preset) {
    case 1:
      Serial.println("ACK PRESET HOME_SAFE");
      moveToAngles(HOME_ANGLES, "PRESET_HOME_SAFE");
      return;
    case 2:
      Serial.println("ACK PRESET CENTER90_INSTALL");
      moveToAngles(CENTER90_ANGLES, "PRESET_CENTER90_INSTALL");
      return;
    case 3:
      copyCurrentTo(requested);
      requested[5] = 50;
      Serial.println("ACK PRESET GRIPPER_OPEN_SAFE");
      moveToAngles(requested, "PRESET_GRIPPER_OPEN_SAFE");
      return;
    case 4:
      copyCurrentTo(requested);
      requested[5] = 35;
      Serial.println("ACK PRESET GRIPPER_CLOSE_SOFT");
      moveToAngles(requested, "PRESET_GRIPPER_CLOSE_SOFT");
      return;
    case 5:
      copyCurrentTo(requested);
      requested[5] = 15;
      Serial.println("ACK PRESET GRIPPER_CLOSE_FULL");
      moveToAngles(requested, "PRESET_GRIPPER_CLOSE_FULL");
      return;
    case 6:
      jogChannel(1, 10, "PRESET_JOG_CH1_PLUS_10");
      return;
    case 7:
      jogChannel(1, -10, "PRESET_JOG_CH1_MINUS_10");
      return;
    case 8:
      jogChannel(5, 10, "PRESET_JOG_CH5_PLUS_10");
      return;
    case 9:
      jogChannel(5, -10, "PRESET_JOG_CH5_MINUS_10");
      return;
    case 10:
      Serial.println("ACK PRESET STATUS_VERBOSE_AND_POSE_YAML");
      printStatusVerbose();
      printPoseYaml("current_pose");
      return;
    default:
      Serial.print("ERR BAD_PRESET ");
      Serial.println(preset);
      return;
  }
}

void handleCommand(char *line) {
  trimLine(line);
  if (line[0] == '\0') {
    return;
  }

  // Keep exact + / - shortcuts before uppercasing.
  if (strcmp(line, "+") == 0) {
    jogChannel(static_cast<int>(selectedChannel + 1), jogStepDeg, "JOG_SELECTED_PLUS");
    return;
  }
  if (strcmp(line, "-") == 0) {
    jogChannel(static_cast<int>(selectedChannel + 1), -jogStepDeg, "JOG_SELECTED_MINUS");
    return;
  }
  if (strcmp(line, "++") == 0) {
    jogChannel(static_cast<int>(selectedChannel + 1), 10, "JOG_SELECTED_PLUS_10");
    return;
  }
  if (strcmp(line, "--") == 0) {
    jogChannel(static_cast<int>(selectedChannel + 1), -10, "JOG_SELECTED_MINUS_10");
    return;
  }

  toUpperInPlace(line);

  // Numeric preset shortcut: 1..10
  bool allDigits = true;
  for (size_t i = 0; line[i] != '\0'; ++i) {
    if (!isdigit(static_cast<unsigned char>(line[i]))) {
      allDigits = false;
      break;
    }
  }
  if (allDigits) {
    int preset = atoi(line);
    handlePreset(preset);
    return;
  }

  if (strcmp(line, "PING") == 0) {
    Serial.println("PONG");
    return;
  }

  if (strcmp(line, "STATUS") == 0) {
    printStatus();
    return;
  }

  if (strcmp(line, "STATUS_VERBOSE") == 0) {
    printStatusVerbose();
    return;
  }

  if (strcmp(line, "MAP") == 0) {
    printMap();
    return;
  }

  if (strcmp(line, "MENU") == 0) {
    printMenu();
    return;
  }

  if (strcmp(line, "HELP") == 0) {
    printHelp();
    return;
  }

  if (strcmp(line, "LIMITS") == 0) {
    printLimits();
    return;
  }

  if (strcmp(line, "POSE_YAML") == 0) {
    printPoseYaml("current_pose");
    return;
  }

  if (strcmp(line, "NEXT") == 0) {
    selectedChannel = (selectedChannel + 1) % SERVO_COUNT;
    Serial.print("SELECTED CH");
    Serial.print(selectedChannel + 1);
    Serial.print(" ");
    Serial.println(CHANNEL_NAMES[selectedChannel]);
    printStatus();
    return;
  }

  if (strcmp(line, "PREV") == 0) {
    selectedChannel = (selectedChannel + SERVO_COUNT - 1) % SERVO_COUNT;
    Serial.print("SELECTED CH");
    Serial.print(selectedChannel + 1);
    Serial.print(" ");
    Serial.println(CHANNEL_NAMES[selectedChannel]);
    printStatus();
    return;
  }

  int one = 0;
  if (parseOneInt(line, "SELECT", one) || parseOneInt(line, "SEL", one)) {
    selectChannel(one);
    return;
  }

  if (parseOneInt(line, "STEP", one)) {
    if (one < 1 || one > 30) {
      Serial.print("ERR BAD_STEP ");
      Serial.println(one);
      return;
    }
    jogStepDeg = one;
    Serial.print("STEP ");
    Serial.println(jogStepDeg);
    printStatus();
    return;
  }

  if (parseOneInt(line, "SET_STEP_DELAY", one)) {
    if (one < 1 || one > 200) {
      Serial.print("ERR BAD_STEP_DELAY ");
      Serial.println(one);
      return;
    }
    stepDelayMs = one;
    Serial.print("STEP_DELAY_MS ");
    Serial.println(stepDelayMs);
    printStatus();
    return;
  }

  if (strcmp(line, "STOP") == 0) {
    Serial.println("ACK STOP");
    stopRequested = true;
    if (!busyState) {
      Serial.println("DONE STOP");
    }
    return;
  }

  if (strcmp(line, "HOME") == 0) {
    if (busyState) {
      Serial.println("ERR BUSY HOME");
      return;
    }
    Serial.println("ACK HOME");
    moveToAngles(HOME_ANGLES, "HOME");
    return;
  }

  if (strcmp(line, "CENTER90") == 0) {
    if (busyState) {
      Serial.println("ERR BUSY CENTER90");
      return;
    }
    Serial.println("ACK CENTER90");
    moveToAngles(CENTER90_ANGLES, "CENTER90");
    return;
  }

  if (strcmp(line, "OPEN") == 0) {
    moveOneChannel(6, 50, "GRIPPER_OPEN");
    return;
  }

  if (strcmp(line, "CLOSE_SOFT") == 0) {
    moveOneChannel(6, 35, "GRIPPER_CLOSE_SOFT");
    return;
  }

  if (strcmp(line, "CLOSE_FULL") == 0) {
    moveOneChannel(6, 15, "GRIPPER_CLOSE_FULL");
    return;
  }

  int a = 0;
  int b = 0;
  if (parseTwoInts(line, "MOVE_ONE", a, b)) {
    if (busyState) {
      Serial.println("ERR BUSY MOVE_ONE");
      return;
    }
    moveOneChannel(a, b, "MOVE_ONE");
    return;
  }

  if (parseTwoInts(line, "JOG", a, b)) {
    if (busyState) {
      Serial.println("ERR BUSY JOG");
      return;
    }
    jogChannel(a, b, "JOG");
    return;
  }

  if (strncmp(line, "PRESET", 6) == 0) {
    if (parseOneInt(line, "PRESET", one)) {
      handlePreset(one);
      return;
    }
    Serial.println("ERR MALFORMED PRESET");
    return;
  }

  if (strncmp(line, "MOVE_SAFE", 9) == 0) {
    if (busyState) {
      Serial.println("ERR BUSY MOVE_SAFE");
      return;
    }

    int requestedAngles[SERVO_COUNT];
    if (!parseMoveSafeAngles(line, requestedAngles)) {
      Serial.println("ERR MALFORMED MOVE_SAFE");
      return;
    }

    Serial.println("ACK MOVE_SAFE");
    moveToAngles(requestedAngles, "MOVE_SAFE");
    return;
  }

  Serial.print("ERR UNKNOWN ");
  Serial.println(line);
}

void serviceSerial() {
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      inputBuffer[inputLength] = '\0';
      handleCommand(inputBuffer);
      clearInputBuffer();
      continue;
    }
    if (inputLength + 1 < INPUT_BUFFER_SIZE) {
      inputBuffer[inputLength++] = c;
      inputBuffer[inputLength] = '\0';
    } else {
      clearInputBuffer();
      Serial.println("ERR LINE_TOO_LONG");
    }
  }
}

}  // namespace

void setup() {
  Serial.begin(BAUD_RATE);
  delay(250);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    servos[i].setPeriodHertz(50);
    servos[i].attach(SERVO_PINS[i], 500, 2500);
    currentAngles[i] = HOME_ANGLES[i];
  }

  writeServoAngles(HOME_ANGLES);
  clearInputBuffer();

  Serial.println("READY ESP32_ROBOT_ARM_SERIAL_CALIBRATION_JOG");
  printHelp();
  printStatus();
}

void loop() {
  serviceSerial();
}