// Verdantix field node: ESP32 + HC-SR04 ultrasonic (bin fill level) + optional ACS712 current sensor.
// Posts JSON readings to the twin's ingest API every 10 s. Same payload as scripts/fake_sensor.py.
//
// Wiring (3.3 V logic):
//   HC-SR04 VCC -> 5V, GND -> GND, TRIG -> GPIO 5, ECHO -> GPIO 18 (through a 1k/2k divider)
//   ACS712 OUT  -> GPIO 34 (ADC), VCC -> 5V, GND -> GND   (set HAS_CURRENT to 1)
//
// Arduino IDE: board "ESP32 Dev Module". Libraries: WiFi, HTTPClient (bundled with the ESP32 core).
// At scale, swap HTTP for MQTT (PubSubClient) to a broker; a 20-line bridge forwards to /api/ingest.

#include <WiFi.h>
#include <HTTPClient.h>

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* INGEST_URL = "https://YOUR-USER-verdantix.hf.space/api/ingest";
const char* INGEST_TOKEN = "";                 // matches INGEST_TOKEN in .env, if set
const char* SENSOR_ID = "esp32-bin-01";
const char* BIN_ID = "bin-soe-1";              // asset id from backend/data/campus.geojson
const char* BUILDING_ID = "soe-a-block";

const int TRIG = 5, ECHO = 18, CURRENT_PIN = 34;
const float BIN_DEPTH_CM = 80.0;               // sensor to bin floor when empty
#define HAS_CURRENT 0

float readDistanceCm() {
  digitalWrite(TRIG, LOW); delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  long us = pulseIn(ECHO, HIGH, 30000);        // 30 ms timeout ~ 5 m
  return us == 0 ? BIN_DEPTH_CM : us * 0.0343 / 2.0;
}

float binFillPct() {
  float d = 0;
  for (int i = 0; i < 5; i++) { d += readDistanceCm(); delay(60); }   // average 5 pings
  d /= 5;
  float pct = 100.0 * (BIN_DEPTH_CM - d) / BIN_DEPTH_CM;
  return constrain(pct, 0, 100);
}

float currentKw() {                            // ACS712-20A on 230 V single phase, rough RMS
  long sum = 0; int n = 400;
  for (int i = 0; i < n; i++) { int v = analogRead(CURRENT_PIN) - 2048; sum += (long)v * v; }
  float vrms = sqrt((float)sum / n) * 3.3 / 4095.0;
  float amps = vrms / 0.100;                   // 100 mV per A
  return amps * 230.0 / 1000.0;
}

void post(const String& body) {
  if (WiFi.status() != WL_CONNECTED) WiFi.reconnect();
  HTTPClient http;
  http.begin(INGEST_URL);
  http.addHeader("Content-Type", "application/json");
  if (strlen(INGEST_TOKEN)) http.addHeader("X-Token", INGEST_TOKEN);
  int code = http.POST(body);
  Serial.printf("POST %d %s\n", code, body.c_str());
  http.end();
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(400); Serial.print("."); }
  Serial.println(" WiFi ok");
}

void loop() {
  String body = "[{\"sensor\":\"" + String(SENSOR_ID) + "\",\"asset\":\"" + BIN_ID +
                "\",\"metric\":\"bin_fill\",\"value\":" + String(binFillPct(), 1) + "}";
#if HAS_CURRENT
  body += ",{\"sensor\":\"" + String(SENSOR_ID) + "-ct\",\"asset\":\"" + BUILDING_ID +
          "\",\"metric\":\"power_kw\",\"value\":" + String(currentKw(), 2) + "}";
#endif
  body += "]";
  post(body);
  delay(10000);
}
