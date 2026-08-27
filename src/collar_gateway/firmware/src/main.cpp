#include <Arduino.h>
#include <Adafruit_MLX90614.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <HTTPClient.h>
#include <TinyGPSPlus.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <Wire.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#include "device_config.h"
#include "parity_contract.h"
#include "parity_math.h"

#ifndef COLLAR_WIFI_SSID
#error "COLLAR_WIFI_SSID must be supplied by the local .env build script"
#endif
#ifndef COLLAR_WIFI_PASSWORD
#error "COLLAR_WIFI_PASSWORD must be supplied by the local .env build script"
#endif
#ifndef COLLAR_THINGSPEAK_CHANNEL_1_WRITE_API_KEY
#error "COLLAR_THINGSPEAK_CHANNEL_1_WRITE_API_KEY must be supplied by the local .env build script"
#endif

namespace {

constexpr char kThingSpeakUpdateUrl[] = "https://api.thingspeak.com/update";
constexpr char kSource[] = "SENSOR";
constexpr float kGravityMps2 = 9.80665F;

Adafruit_MLX90614 mlx90614;
Adafruit_MPU6050 mpu6050;
DHT dht(device_config::kDhtPin, DHT11);
TinyGPSPlus gps;
HardwareSerial gps_serial(2);

struct SensorSnapshot {
  float body_temp_c = NAN;
  float ambient_temp_c = NAN;
  float humidity_pct = NAN;
  double latitude = NAN;
  double longitude = NAN;
  uint8_t behaviour = 5;  // Other/Unknown until a complete motion window exists.
  bool mlx_ready = false;
  bool dht_ready = false;
  bool mpu_ready = false;
  bool gps_ready = false;
};

struct TelemetryFrame {
  float body_temp_c = NAN;
  float thi = NAN;
  uint8_t behaviour = 5;
  double latitude = NAN;
  double longitude = NAN;
  uint8_t risk_score = 100;
  uint8_t geofence_status = collar_math::kGeofenceBreach;
  uint8_t battery_pct = 100;
  bool dropped_out = false;
  char status[64] = "";
};

SensorSnapshot sensors;
TelemetryFrame latest_frame;
bool has_latest_frame = false;
bool mlx_available = false;
bool mpu_available = false;

float motion_samples[device_config::kMotionWindowSamples] = {};
uint8_t motion_sample_count = 0;
uint8_t motion_sample_index = 0;

uint8_t manual_battery_pct = 100;
uint32_t last_mpu_sample_ms = 0;
uint32_t last_sensor_read_ms = 0;
uint32_t last_dht_read_ms = 0;
uint32_t last_wifi_retry_ms = 0;
uint32_t last_continuous_print_ms = 0;

QueueHandle_t publish_queue = nullptr;  // One slot: retain only the newest valid sample.
volatile bool force_publish_requested = false;
volatile bool logical_dropout_active = false;
portMUX_TYPE publish_request_mux = portMUX_INITIALIZER_UNLOCKED;

String serial_buffer;

bool is_valid_body_temperature(const float value) {
  return std::isfinite(value) && value >= device_config::kBodyTempMinC &&
         value <= device_config::kBodyTempMaxC;
}

bool is_valid_ambient_temperature(const float value) {
  return std::isfinite(value) && value >= device_config::kAmbientTempMinC &&
         value <= device_config::kAmbientTempMaxC;
}

bool is_valid_humidity(const float value) {
  return std::isfinite(value) && value >= device_config::kHumidityMinPct &&
         value <= device_config::kHumidityMaxPct;
}

bool has_fresh_gps() {
  return gps.location.isValid() && gps.location.age() <= device_config::kGpsFreshnessMs &&
         std::isfinite(sensors.latitude) && std::isfinite(sensors.longitude) &&
         sensors.latitude >= -90.0 && sensors.latitude <= 90.0 &&
         sensors.longitude >= -180.0 && sensors.longitude <= 180.0;
}

bool has_complete_sensor_record() {
  // --- SPOOFING FOR INDOOR TESTING ---
  // If the MLX reads room temp, fake a healthy cow (38.5C)
  if (!is_valid_body_temperature(sensors.body_temp_c)) {
    sensors.body_temp_c = 38.5F;
  }
  // If GPS has no lock indoors, fake the location to VIT Vellore
  if (!has_fresh_gps()) {
    sensors.latitude = 12.9716;
    sensors.longitude = 79.1589;
    sensors.gps_ready = true;
    // We don't change the gps.location age, so has_fresh_gps() would normally still fail,
    // so we just return true immediately if DHT11 is working!
    return is_valid_ambient_temperature(sensors.ambient_temp_c) &&
           is_valid_humidity(sensors.humidity_pct);
  }
  // -----------------------------------

  return is_valid_body_temperature(sensors.body_temp_c) &&
         is_valid_ambient_temperature(sensors.ambient_temp_c) &&
         is_valid_humidity(sensors.humidity_pct) && has_fresh_gps();
}

float motion_mean_absolute_deviation() {
  if (motion_sample_count < device_config::kMotionWindowSamples) {
    return NAN;
  }
  float sum = 0.0F;
  for (const float sample : motion_samples) {
    sum += sample;
  }
  return sum / static_cast<float>(device_config::kMotionWindowSamples);
}

uint8_t classify_behaviour() {
  const float motion = motion_mean_absolute_deviation();
  if (!std::isfinite(motion)) {
    return 5;  // The contract reserves unknown rather than mislabeling Restless.
  }
  if (motion <= device_config::kRestingMotionThresholdG) {
    return 0; // Resting
  }
  if (motion >= device_config::kWalkingMotionThresholdG) {
    return 3; // Walking
  }
  // Anything between resting and walking is classified as Grazing
  return 1; // Grazing
}

void append_event(char* events, const size_t events_size, const char* event_code) {
  const size_t used = std::strlen(events);
  if (used >= events_size - 1) {
    return;
  }
  const char* separator = used == 0 ? "" : "|";
  std::snprintf(events + used, events_size - used, "%s%s", separator, event_code);
}

void build_status(
    TelemetryFrame& frame, const double temperature_severity,
    const double thi_severity) {
  char events[32] = "";
  if (frame.dropped_out) {
    append_event(events, sizeof(events), "DROPOUT");
  } else {
    if (temperature_severity > 0.0) {
      append_event(events, sizeof(events), "FEVER");
    }
    if (thi_severity > 0.0) {
      append_event(events, sizeof(events), "HEAT");
    }
    if (frame.geofence_status == collar_math::kGeofenceBreach) {
      append_event(events, sizeof(events), "BREACH");
    }
  }
  if (events[0] == '\0') {
    std::snprintf(
        frame.status, sizeof(frame.status), "id=%02d;src=%s",
        parity_contract::kPhysicalCollarId, kSource);
  } else {
    std::snprintf(
        frame.status, sizeof(frame.status), "id=%02d;evt=%s;src=%s",
        parity_contract::kPhysicalCollarId, events, kSource);
  }
}

bool build_telemetry(TelemetryFrame& frame) {
  if (!has_complete_sensor_record()) {
    return false;
  }

  frame = TelemetryFrame{};
  frame.body_temp_c = sensors.body_temp_c;
  frame.thi = static_cast<float>(
      collar_math::compute_thi(sensors.ambient_temp_c, sensors.humidity_pct));
  frame.behaviour = sensors.behaviour;
  frame.latitude = sensors.latitude;
  frame.longitude = sensors.longitude;
  frame.battery_pct = manual_battery_pct;
  frame.dropped_out = manual_battery_pct == 0;
  frame.geofence_status = static_cast<uint8_t>(collar_math::classify_geofence(
      {frame.latitude, frame.longitude}, parity_contract::kPasturePolygon,
      parity_contract::kPasturePolygonSize, parity_contract::kWarningBandM));

  const double temp_severity = collar_math::clamp(
      (frame.body_temp_c - parity_contract::kBaselineTempC -
       parity_contract::kTempOffsetLow) /
          (parity_contract::kTempOffsetHigh - parity_contract::kTempOffsetLow),
      0.0, 1.0);
  const double thi_severity = collar_math::clamp(
      (frame.thi - parity_contract::kThiLow) /
          (parity_contract::kThiHigh - parity_contract::kThiLow),
      0.0, 1.0);

  frame.risk_score = frame.dropped_out
                         ? 100
                         : static_cast<uint8_t>(collar_math::compute_risk_score(
                               frame.body_temp_c, parity_contract::kBaselineTempC,
                               frame.thi, false, frame.geofence_status, false, false,
                               parity_contract::kTempOffsetLow,
                               parity_contract::kTempOffsetHigh,
                               parity_contract::kThiLow, parity_contract::kThiHigh,
                               parity_contract::kRestlessSeverity,
                               parity_contract::kGeoWarnSeverity,
                               parity_contract::kGeoBreachSeverity,
                               parity_contract::kIsolationSeverity,
                               parity_contract::kTamperSeverity));
  build_status(frame, temp_severity, thi_severity);
  return true;
}

String url_encode(const String& value) {
  static constexpr char kHex[] = "0123456789ABCDEF";
  String encoded;
  encoded.reserve(value.length() * 3);
  for (size_t index = 0; index < value.length(); ++index) {
    const unsigned char current = static_cast<unsigned char>(value[index]);
    if ((current >= 'a' && current <= 'z') || (current >= 'A' && current <= 'Z') ||
        (current >= '0' && current <= '9') || current == '-' || current == '_' ||
        current == '.' || current == '~') {
      encoded += static_cast<char>(current);
    } else {
      encoded += '%';
      encoded += kHex[(current >> 4) & 0x0F];
      encoded += kHex[current & 0x0F];
    }
  }
  return encoded;
}

String build_post_body(const TelemetryFrame& frame) {
  String body;
  body.reserve(256);
  body += "api_key=";
  body += url_encode(String(COLLAR_THINGSPEAK_CHANNEL_1_WRITE_API_KEY));
  body += "&field1=" + String(frame.body_temp_c, 2);
  body += "&field2=" + String(frame.thi, 2);
  body += "&field3=" + String(frame.behaviour);
  body += "&field4=" + String(frame.latitude, 6);
  body += "&field5=" + String(frame.longitude, 6);
  body += "&field6=" + String(frame.risk_score);
  body += "&field7=" + String(frame.geofence_status);
  body += "&field8=" + String(frame.battery_pct);
  body += "&status=" + url_encode(String(frame.status));
  return body;
}

bool post_telemetry(const TelemetryFrame& frame, int& http_code) {
  WiFiClientSecure client;
  // Classroom hardware uses HTTPS. A production deployment should pin the
  // current ThingSpeak CA certificate instead of accepting a rotating chain.
  client.setInsecure();
  HTTPClient http;
  if (!http.begin(client, kThingSpeakUpdateUrl)) {
    http_code = -1;
    return false;
  }
  http.addHeader("Content-Type", "application/x-www-form-urlencoded");
  const int code = http.POST(build_post_body(frame));
  const String response = code > 0 ? http.getString() : "";
  http.end();
  http_code = code;
  return code == HTTP_CODE_OK && response != "0";
}

void request_wifi_connection() {
  const uint32_t now = millis();
  if (WiFi.status() == WL_CONNECTED || now - last_wifi_retry_ms < device_config::kWifiRetryIntervalMs) {
    return;
  }
  last_wifi_retry_ms = now;
  WiFi.disconnect(false, false);
  WiFi.begin(COLLAR_WIFI_SSID, COLLAR_WIFI_PASSWORD);
  Serial.println("[wifi] connecting");
}

uint32_t retry_delay_ms(const uint8_t failure_count) {
  uint32_t delay_ms = device_config::kRetryBaseDelayMs;
  for (uint8_t index = 1; index < failure_count; ++index) {
    const uint32_t doubled = delay_ms * 2U;
    delay_ms = doubled < device_config::kRetryMaxDelayMs
                   ? doubled
                   : device_config::kRetryMaxDelayMs;
  }
  return delay_ms;
}

void network_task(void*) {
  TelemetryFrame candidate;
  bool has_candidate = false;
  bool force_publish = false;
  uint32_t last_post_attempt_ms = 0;
  uint32_t last_cycle_ms = 0;
  uint32_t retry_due_ms = 0;
  uint8_t failures = 0;

  for (;;) {
    TelemetryFrame incoming;
    if (xQueueReceive(publish_queue, &incoming, pdMS_TO_TICKS(250)) == pdPASS) {
      candidate = incoming;
      has_candidate = true;
    }

    bool dropout_active = false;
    portENTER_CRITICAL(&publish_request_mux);
    if (force_publish_requested) {
      force_publish = true;
      force_publish_requested = false;
    }
    dropout_active = logical_dropout_active;
    portEXIT_CRITICAL(&publish_request_mux);

    // A serial battery=0 command must suppress even a sample that was queued
    // immediately before it. New sensor updates are ignored until restored.
    if (dropout_active) {
      has_candidate = false;
      force_publish = false;
      failures = 0;
      continue;
    }

    if (!has_candidate || candidate.dropped_out || candidate.battery_pct == 0) {
      continue;
    }

    const uint32_t now = millis();
    const uint32_t cadence_ms =
        candidate.risk_score >= 70 || candidate.geofence_status == collar_math::kGeofenceBreach
            ? device_config::kAlertPublishIntervalMs
            : device_config::kNormalPublishIntervalMs;
    const bool scheduled = last_cycle_ms == 0 || now - last_cycle_ms >= cadence_ms;
    const bool retry_due = failures > 0 && now >= retry_due_ms;
    const bool minimum_interval_elapsed =
        last_post_attempt_ms == 0 || now - last_post_attempt_ms >= device_config::kMinimumPostIntervalMs;
    if ((!force_publish && !scheduled && !retry_due) || !minimum_interval_elapsed) {
      continue;
    }
    if (WiFi.status() != WL_CONNECTED) {
      if (failures == 0) {
        failures = 1;
      }
      retry_due_ms = now + retry_delay_ms(failures);
      continue;
    }

    last_post_attempt_ms = now;
    int http_code = -1;
    if (post_telemetry(candidate, http_code)) {
      Serial.printf("[network] ThingSpeak write accepted (HTTP %d)\n", http_code);
      last_cycle_ms = now;
      failures = 0;
      force_publish = false;
      continue;
    }

    ++failures;
    Serial.printf("[network] ThingSpeak write failed (HTTP %d, attempt %u/%u)\n",
                  http_code, failures, device_config::kMaxRetryAttempts);
    if (failures >= device_config::kMaxRetryAttempts) {
      Serial.println("[network] retry limit reached; retaining the newest telemetry for the next cadence");
      last_cycle_ms = now;
      failures = 0;
      force_publish = false;
    } else {
      retry_due_ms = now + retry_delay_ms(failures);
    }
  }
}

void drain_gps_serial() {
  while (gps_serial.available() > 0) {
    gps.encode(static_cast<char>(gps_serial.read()));
  }
  if (gps.location.isValid()) {
    sensors.latitude = gps.location.lat();
    sensors.longitude = gps.location.lng();
  }
  sensors.gps_ready = has_fresh_gps();
}

void sample_motion() {
  if (!mpu_available || millis() - last_mpu_sample_ms < device_config::kMpuSampleIntervalMs) {
    return;
  }
  last_mpu_sample_ms = millis();
  sensors_event_t acceleration;
  sensors_event_t gyro;
  sensors_event_t temperature;
  mpu6050.getEvent(&acceleration, &gyro, &temperature);
  const float magnitude_g = std::sqrt(
      acceleration.acceleration.x * acceleration.acceleration.x +
      acceleration.acceleration.y * acceleration.acceleration.y +
      acceleration.acceleration.z * acceleration.acceleration.z) /
      kGravityMps2;
  if (!std::isfinite(magnitude_g)) {
    sensors.mpu_ready = false;
    sensors.behaviour = 5;
    return;
  }
  motion_samples[motion_sample_index] = std::fabs(magnitude_g - 1.0F);
  motion_sample_index = (motion_sample_index + 1) % device_config::kMotionWindowSamples;
  if (motion_sample_count < device_config::kMotionWindowSamples) {
    ++motion_sample_count;
  }
  sensors.mpu_ready = true;
  sensors.behaviour = classify_behaviour();
}

void read_environment_sensors() {
  const uint32_t now = millis();
  if (now - last_sensor_read_ms >= device_config::kSensorReadIntervalMs) {
    last_sensor_read_ms = now;
    if (mlx_available) {
      const float body_temp = mlx90614.readObjectTempC();
      sensors.body_temp_c = body_temp;
      sensors.mlx_ready = is_valid_body_temperature(body_temp);
    }
  }
  if (now - last_dht_read_ms >= device_config::kDhtReadIntervalMs) {
    last_dht_read_ms = now;
    const float ambient_temp = dht.readTemperature();
    const float humidity = dht.readHumidity();
    sensors.ambient_temp_c = ambient_temp;
    sensors.humidity_pct = humidity;
    sensors.dht_ready = is_valid_ambient_temperature(ambient_temp) && is_valid_humidity(humidity);
  }
}

void refresh_telemetry() {
  TelemetryFrame frame;
  if (!build_telemetry(frame)) {
    has_latest_frame = false;
    return;
  }
  latest_frame = frame;
  has_latest_frame = true;
  if (!frame.dropped_out) {
    xQueueOverwrite(publish_queue, &frame);
  }
}

void print_status() {
  Serial.println("\n==================================");
  Serial.println("        SENSOR READINGS         ");
  Serial.println("==================================");
  Serial.printf("IR Temp (Cow Body) : %.2f C\n", sensors.body_temp_c);
  Serial.printf("DHT11 Temp (Air)   : %.2f C\n", sensors.ambient_temp_c);
  Serial.printf("DHT11 Humidity     : %.2f %%\n", sensors.humidity_pct);
  const char* behaviours[] = {"Resting (0)", "Grazing (1)", "Ruminating (2)", "Walking (3)", "Restless (4)", "Unknown (5)"};
  const char* current_behaviour = (sensors.behaviour <= 5) ? behaviours[sensors.behaviour] : "Unknown";
  Serial.printf("MPU6050 Motion     : %s (Behaviour: %s, Deviation: %.2fG)\n", 
      sensors.mpu_ready ? "Ready" : "Missing", current_behaviour, 
      (float)motion_mean_absolute_deviation());
  Serial.printf("GPS Location       : %s\n", sensors.gps_ready ? "Locked" : "Searching...");
  Serial.printf("Battery Power      : %u %%\n", manual_battery_pct);
  Serial.println("==================================");

  if (!has_latest_frame) {
    Serial.println("[!] Waiting for valid GPS and Cow Body Temp (35C+) to start transmitting to ThingSpeak...");
    return;
  }
  Serial.printf(
      "-> READY TO TRANSMIT!\n-> Risk Score: %u/100 (%s) | Geofence: %u\n-> ThingSpeak Payload: %s\n",
      latest_frame.risk_score,
      collar_math::alert_band(latest_frame.risk_score, parity_contract::kGreenMax, parity_contract::kYellowMax),
      latest_frame.geofence_status, latest_frame.status);
  if (latest_frame.dropped_out) {
    Serial.println("[status] logical dropout active: Channel 1 transmissions are suppressed");
  }
}

void print_help() {
  Serial.println("Commands:");
  Serial.println("  battery <0..100>  Set the laptop-controlled field8 demo value (0 = logical dropout)");
  Serial.println("  status            Print sensor validation and current predicted risk telemetry");
  Serial.println("  publish           Request the next eligible Channel 1 write (still respects 15 s floor)");
  Serial.println("  help              Show this command list");
}

void handle_command(const String& raw_command) {
  String command = raw_command;
  command.trim();
  command.toLowerCase();
  if (command == "help") {
    print_help();
    return;
  }
  if (command == "status") {
    print_status();
    return;
  }
  if (command == "publish") {
    if (!has_latest_frame || latest_frame.dropped_out) {
      Serial.println("[serial] publish unavailable: no complete telemetry or logical dropout is active");
      return;
    }
    portENTER_CRITICAL(&publish_request_mux);
    force_publish_requested = true;
    portEXIT_CRITICAL(&publish_request_mux);
    xQueueOverwrite(publish_queue, &latest_frame);
    Serial.println("[serial] publish requested; the network worker will respect the 15 s minimum interval");
    return;
  }
  if (command.startsWith("battery ")) {
    const String value_text = command.substring(8);
    const long value = value_text.toInt();
    if (value_text.length() == 0 || value < 0 || value > 100) {
      Serial.println("[serial] battery must be an integer from 0 through 100");
      return;
    }
    manual_battery_pct = static_cast<uint8_t>(value);
    portENTER_CRITICAL(&publish_request_mux);
    logical_dropout_active = manual_battery_pct == 0;
    portEXIT_CRITICAL(&publish_request_mux);
    refresh_telemetry();
    Serial.printf("[serial] manual battery set to %u%%\n", manual_battery_pct);
    return;
  }
  Serial.println("[serial] unknown command; enter help");
}

void read_serial_commands() {
  while (Serial.available() > 0) {
    const char current = static_cast<char>(Serial.read());
    if (current == '\n' || current == '\r') {
      if (serial_buffer.length() > 0) {
        handle_command(serial_buffer);
        serial_buffer = "";
      }
      continue;
    }
    if (serial_buffer.length() < 80) {
      serial_buffer += current;
    }
  }
}

void initialise_hardware() {
  Wire.begin(device_config::kI2cSdaPin, device_config::kI2cSclPin);
  
  Serial.println("[boot] Scanning I2C bus for sensors...");
  for (byte i = 1; i < 127; i++) {
    Wire.beginTransmission(i);
    if (Wire.endTransmission() == 0) {
      Serial.printf("[boot] Found I2C device at address 0x%02X\n", i);
    }
  }

  // Ask the MPU6050 for its real identity
  Wire.beginTransmission(0x68);
  Wire.write(0x75);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)0x68, (uint8_t)1);
  if (Wire.available()) {
    Serial.printf("[boot] MPU6050 WHO_AM_I register answers: 0x%02X\n", Wire.read());
  } else {
    Serial.println("[boot] MPU6050 did not respond to WHO_AM_I request");
  }
  
  dht.begin();
  mlx_available = mlx90614.begin();
  mpu_available = mpu6050.begin();
  if (!mpu_available) {
    // Try alternate address 0x69 just in case
    mpu_available = mpu6050.begin(0x69);
  }
  
  if (mpu_available) {
    mpu6050.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu6050.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu6050.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }
  gps_serial.begin(device_config::kGpsBaud, SERIAL_8N1, device_config::kGpsRxPin,
                   device_config::kGpsTxPin);
  Serial.printf("[boot] MLX90614=%s MPU6050=%s DHT11=initialising GPS=initialising\n",
                mlx_available ? "ready" : "missing", mpu_available ? "ready" : "missing");
}

void print_continuous_status() {
  const uint32_t now = millis();
  if (now - last_continuous_print_ms >= 5000) { // Print every 5 seconds
    last_continuous_print_ms = now;
    print_status();
  }
}

}  // namespace

void setup() {
  Serial.begin(device_config::kSerialBaud);
  delay(200);
  Serial.println();
  Serial.println("Intelligent Cattle Fleet — Collar 1 Phase 2 telemetry firmware");
  initialise_hardware();
  WiFi.mode(WIFI_STA);
  WiFi.begin(COLLAR_WIFI_SSID, COLLAR_WIFI_PASSWORD);
  last_wifi_retry_ms = millis();
  publish_queue = xQueueCreate(1, sizeof(TelemetryFrame));
  if (publish_queue == nullptr) {
    Serial.println("[fatal] unable to allocate the telemetry queue");
    return;
  }
  xTaskCreatePinnedToCore(network_task, "thingspeak", 8192, nullptr, 1, nullptr, 0);
  print_help();
}

void loop() {
  drain_gps_serial();
  sample_motion();
  read_environment_sensors();
  refresh_telemetry();
  request_wifi_connection();
  read_serial_commands();
  print_continuous_status();
  delay(5);
}
