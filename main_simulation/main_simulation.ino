// intersection_controller.ino
// Single‑light multiplexed 4‑way traffic controller

const int RED_PIN = 2;
const int YEL_PIN = 3;
const int GRN_PIN = 4;

void setup() {
  pinMode(RED_PIN, OUTPUT);
  pinMode(YEL_PIN, OUTPUT);
  pinMode(GRN_PIN, OUTPUT);
  Serial.begin(9600);
  allRed();
}

// Bring all lights to red
void allRed() {
  digitalWrite(RED_PIN, HIGH);
  digitalWrite(YEL_PIN, LOW);
  digitalWrite(GRN_PIN, LOW);
}

// Run one R→Y→G→Y→R cycle for duration sec
void runCycle(int seconds) {
  allRed();
  delay(500);

  // Yellow before green
  digitalWrite(RED_PIN, LOW);
  digitalWrite(YEL_PIN, HIGH);
  delay(2000);

  // Green phase
  digitalWrite(YEL_PIN, LOW);
  digitalWrite(GRN_PIN, HIGH);
  delay(seconds * 1000);

  // Yellow after
  digitalWrite(GRN_PIN, LOW);
  digitalWrite(YEL_PIN, HIGH);
  delay(3000);

  // Back to red
  allRed();
}

void loop() {
  // Check for input
  if (Serial.available() > 0) {
    // Read until newline
    String cmd = Serial.readStringUntil('\n');  
    cmd.trim();  // remove any stray whitespace

    if (cmd.length() > 2 && cmd.charAt(1) == ':') {
      char dir = cmd.charAt(0);           // 'N', 'E', 'S', 'W'
      int sep = cmd.indexOf(':');
      int dur = cmd.substring(sep + 1).toInt();
      Serial.print("↪ Cmd received: "); Serial.println(cmd);

      // Optionally log direction:
      Serial.print("🔀 Direction: "); Serial.print(dir);
      Serial.print(" | Green time: "); Serial.print(dur); Serial.println("s");

      runCycle(dur);
      Serial.println("✅ Cycle complete, awaiting next command");
    } else {
      Serial.print("‼️ Invalid cmd: "); Serial.println(cmd);
    }
  }
}

