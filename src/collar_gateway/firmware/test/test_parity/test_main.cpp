#include <cmath>

#include <unity.h>

#include "parity_contract.h"
#include "parity_math.h"

namespace {

void test_thi_vectors_match_contract() {
  for (std::size_t index = 0; index < parity_contract::kThiVectorCount; ++index) {
    const auto vector = parity_contract::kThiVectors[index];
    const double actual = collar_math::compute_thi(vector.ambient_temp_c, vector.humidity_pct);
    TEST_ASSERT_FLOAT_WITHIN(
        0.001F, static_cast<float>(vector.expected_thi), static_cast<float>(actual));
  }
}

void test_geofence_vectors_match_contract() {
  for (std::size_t index = 0; index < parity_contract::kGeofenceVectorCount; ++index) {
    const auto vector = parity_contract::kGeofenceVectors[index];
    const int actual = collar_math::classify_geofence(
        {vector.latitude, vector.longitude}, parity_contract::kPasturePolygon,
        parity_contract::kPasturePolygonSize, parity_contract::kWarningBandM);
    TEST_ASSERT_EQUAL_INT(vector.expected_status, actual);
  }
}

void test_risk_vectors_match_contract() {
  for (std::size_t index = 0; index < parity_contract::kRiskVectorCount; ++index) {
    const auto vector = parity_contract::kRiskVectors[index];
    const int actual = collar_math::compute_risk_score(
        vector.body_temp_c, parity_contract::kBaselineTempC, vector.thi,
        vector.restless, vector.geofence_status, vector.isolated, vector.tampered,
        parity_contract::kTempOffsetLow, parity_contract::kTempOffsetHigh,
        parity_contract::kThiLow, parity_contract::kThiHigh,
        parity_contract::kRestlessSeverity, parity_contract::kGeoWarnSeverity,
        parity_contract::kGeoBreachSeverity, parity_contract::kIsolationSeverity,
        parity_contract::kTamperSeverity);
    TEST_ASSERT_EQUAL_INT(vector.expected_score, actual);
  }
}

void test_alert_band_boundaries_match_contract() {
  TEST_ASSERT_EQUAL_STRING("green", collar_math::alert_band(0, parity_contract::kGreenMax,
                                                              parity_contract::kYellowMax));
  TEST_ASSERT_EQUAL_STRING("green", collar_math::alert_band(parity_contract::kGreenMax,
                                                              parity_contract::kGreenMax,
                                                              parity_contract::kYellowMax));
  TEST_ASSERT_EQUAL_STRING("yellow", collar_math::alert_band(parity_contract::kGreenMax + 1,
                                                               parity_contract::kGreenMax,
                                                               parity_contract::kYellowMax));
  TEST_ASSERT_EQUAL_STRING("red", collar_math::alert_band(parity_contract::kYellowMax + 1,
                                                            parity_contract::kGreenMax,
                                                            parity_contract::kYellowMax));
}

}  // namespace

#if defined(ARDUINO)
void setup() {
  UNITY_BEGIN();
  RUN_TEST(test_thi_vectors_match_contract);
  RUN_TEST(test_geofence_vectors_match_contract);
  RUN_TEST(test_risk_vectors_match_contract);
  RUN_TEST(test_alert_band_boundaries_match_contract);
  UNITY_END();
}

void loop() {}
#else
int main(int, char**) {
  UNITY_BEGIN();
  RUN_TEST(test_thi_vectors_match_contract);
  RUN_TEST(test_geofence_vectors_match_contract);
  RUN_TEST(test_risk_vectors_match_contract);
  RUN_TEST(test_alert_band_boundaries_match_contract);
  return UNITY_END();
}
#endif
