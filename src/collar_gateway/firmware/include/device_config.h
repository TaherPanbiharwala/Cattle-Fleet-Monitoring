// Public, non-secret board configuration for the ESP32 DevKit V1 / WROOM-32.
#pragma once

#include <cstdint>

namespace device_config {

constexpr int kI2cSdaPin = 21;
constexpr int kI2cSclPin = 22;
constexpr int kDhtPin = 15;  // ADR-024: GPIO4 is dead on the reference board
constexpr int kGpsRxPin = 16;  // ESP32 RX <- NEO-6M TX
constexpr int kGpsTxPin = 17;  // ESP32 TX -> NEO-6M RX (optional)

constexpr uint32_t kSerialBaud = 115200;
constexpr uint32_t kGpsBaud = 9600;
constexpr uint32_t kMpuSampleIntervalMs = 100;  // 10 Hz, local only
constexpr uint32_t kSensorReadIntervalMs = 1000;
constexpr uint32_t kDhtReadIntervalMs = 2000;   // DHT11 minimum interval
constexpr uint32_t kGpsFreshnessMs = 5000;

constexpr uint32_t kNormalPublishIntervalMs = 30000;
constexpr uint32_t kAlertPublishIntervalMs = 15000;
constexpr uint32_t kMinimumPostIntervalMs = 15000;
constexpr uint32_t kWifiRetryIntervalMs = 10000;
constexpr uint32_t kRetryBaseDelayMs = 2000;
constexpr uint32_t kRetryMaxDelayMs = 16000;
constexpr uint8_t kMaxRetryAttempts = 4;

constexpr uint8_t kMotionWindowSamples = 50;  // 5 seconds at 10 Hz
constexpr float kRestingMotionThresholdG = 0.25F;  // ADR-024: recalibrated for MPU-6500 clone offset
constexpr float kWalkingMotionThresholdG = 0.50F;

constexpr float kBodyTempMinC = 35.0F;
constexpr float kBodyTempMaxC = 43.0F;
constexpr float kAmbientTempMinC = -20.0F;
constexpr float kAmbientTempMaxC = 60.0F;
constexpr float kHumidityMinPct = 0.0F;
constexpr float kHumidityMaxPct = 100.0F;

}  // namespace device_config
