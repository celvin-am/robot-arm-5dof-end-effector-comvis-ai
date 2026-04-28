#include <ESP32Servo.h>

namespace {

constexpr uint32_t BAUD_RATE = 115200;
constexpr size_t SERVO_COUNT = 6;
constexpr uint8_t SERVO_PINS[SERVO_COUNT] = {13, 14, 27, 26, 25, 33};

// Phase 6 physical channel mapping:
// CH1 GPIO13 = base yaw
// CH2 GPIO14 = shoulder pitch
// CH3 GPIO27 = elbow pitch
// CH4 GPIO26 = wrist yaw / wrist rotate
// CH5 GPIO25 = wrist pitch / gripper up-down
// CH6 GPIO33 = gripper open-close
constexpr int MIN_ANGLES[SERVO_COUNT] = {40, 40, 40, 40, 40, 10};
constexpr int MAX_ANGLES[SERVO_COUNT] = {140, 140, 140, 140, 140, 60};
constexpr int HOME_ANGLES[SERVO_COUNT] = {90, 130, 130, 95, 60, 45};

constexpr int STEP_DELAY_MS = 20;
constexpr size_t INPUT_BUFFER_SIZE = 96;

Servo servos[SERVO_COUNT];
int currentAngles[SERVO_COUNT];
bool busyState = false;
bool stopRequested = false;
char inputBuffer[INPUT_BUFFER_SIZE];
size_t inputLength = 0;

int clampAngle(size_t index, int angle) {
  if (angle < MIN_ANGLES[index]) {
    return MIN_ANGLES[index];
  }
  if (angle > MAX_ANGLES[index]) {
    return MAX_ANGLES[index];
  }
  return angle;
}

void writeServoAngles(const int angles[SERVO_COUNT]) {
  for (size_t i = 0; i < SERVO_COUNT; ++i) {
    servos[i].write(clampAngle(i, angles[i]));
  }
}

void printHelp() {
  Serial.println("HELP PING STATUS HOME MOVE_SAFE STOP LIMITS HELP");
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
  Serial.println();
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

void processBusyCommand(char *line) {
  trimLine(line);
  toUpperInPlace(line);
  if (strcmp(line, "STOP") == 0) {
    if (!stopRequested) {
      stopRequested = true;
      Serial.println("ACK STOP");
    }
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
    delay(STEP_DELAY_MS);
  }

  busyState = false;
  if (stopRequested) {
    Serial.println("DONE STOP");
  } else {
    Serial.print("DONE ");
    Serial.println(label);
  }
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

void handleCommand(char *line) {
  trimLine(line);
  toUpperInPlace(line);

  if (line[0] == '\0') {
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

  if (strcmp(line, "HELP") == 0) {
    printHelp();
    return;
  }

  if (strcmp(line, "LIMITS") == 0) {
    printLimits();
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

  // Phase 6 assumes HOME_SAFE as the internal startup reference and writes
  // those clamped angles directly to avoid an unnecessary sweep on boot.
  writeServoAngles(HOME_ANGLES);
  clearInputBuffer();

  Serial.println("READY ESP32_ROBOT_ARM_SERIAL");
  printHelp();
  printStatus();
}

void loop() {
  serviceSerial();
}
