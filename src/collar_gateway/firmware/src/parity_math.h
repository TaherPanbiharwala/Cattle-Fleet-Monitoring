// Pure C++ implementation of the Phase 1 telemetry math contract.
#pragma once

#include <cmath>
#include <cstddef>

namespace collar_math {

struct Coord {
  double latitude;
  double longitude;
};

enum GeofenceStatus : int {
  kGeofenceInside = 0,
  kGeofenceWarning = 1,
  kGeofenceBreach = 2,
};

inline double clamp(const double value, const double lower, const double upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

inline double degrees_to_radians(const double degrees) {
  return degrees * 3.14159265358979323846 / 180.0;
}

inline double compute_thi(const double ambient_temp_c, const double humidity_pct) {
  return (1.8 * ambient_temp_c + 32.0) -
         (0.55 - 0.0055 * humidity_pct) * (1.8 * ambient_temp_c - 26.0);
}

inline double haversine_m(const Coord first, const Coord second, const double earth_radius_m) {
  const double lat1 = degrees_to_radians(first.latitude);
  const double lon1 = degrees_to_radians(first.longitude);
  const double lat2 = degrees_to_radians(second.latitude);
  const double lon2 = degrees_to_radians(second.longitude);
  const double dlat = lat2 - lat1;
  const double dlon = lon2 - lon1;
  const double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0) +
                   std::cos(lat1) * std::cos(lat2) *
                       std::sin(dlon / 2.0) * std::sin(dlon / 2.0);
  const double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
  return earth_radius_m * c;
}

inline bool point_in_polygon(const Coord point, const Coord* polygon, const std::size_t count) {
  bool inside = false;
  std::size_t previous = count - 1;
  for (std::size_t current = 0; current < count; ++current) {
    const Coord current_point = polygon[current];
    const Coord previous_point = polygon[previous];
    if (((current_point.longitude > point.longitude) !=
         (previous_point.longitude > point.longitude)) &&
        (point.latitude <
         (previous_point.latitude - current_point.latitude) *
                 (point.longitude - current_point.longitude) /
                 (previous_point.longitude - current_point.longitude) +
             current_point.latitude)) {
      inside = !inside;
    }
    previous = current;
  }
  return inside;
}

inline double point_to_segment_distance_m(
    const Coord point, const Coord start, const Coord end) {
  const double cosine_latitude = std::cos(degrees_to_radians(start.latitude));
  constexpr double kMetersPerDegreeLatitude = 111320.0;
  const double meters_per_degree_longitude = kMetersPerDegreeLatitude * cosine_latitude;
  const double point_x = (point.latitude - start.latitude) * kMetersPerDegreeLatitude;
  const double point_y = (point.longitude - start.longitude) * meters_per_degree_longitude;
  const double segment_x = (end.latitude - start.latitude) * kMetersPerDegreeLatitude;
  const double segment_y = (end.longitude - start.longitude) * meters_per_degree_longitude;
  const double segment_length_squared = segment_x * segment_x + segment_y * segment_y;
  if (segment_length_squared < 1e-12) {
    return std::sqrt(point_x * point_x + point_y * point_y);
  }
  const double projection = clamp(
      (point_x * segment_x + point_y * segment_y) / segment_length_squared, 0.0, 1.0);
  const double delta_x = point_x - projection * segment_x;
  const double delta_y = point_y - projection * segment_y;
  return std::sqrt(delta_x * delta_x + delta_y * delta_y);
}

inline int classify_geofence(
    const Coord point, const Coord* polygon, const std::size_t count,
    const double warning_band_m) {
  if (!point_in_polygon(point, polygon, count)) {
    return kGeofenceBreach;
  }
  double minimum_distance = INFINITY;
  for (std::size_t current = 0; current < count; ++current) {
    const std::size_t next = (current + 1) % count;
    const double distance = point_to_segment_distance_m(point, polygon[current], polygon[next]);
    if (distance < minimum_distance) {
      minimum_distance = distance;
    }
  }
  return minimum_distance <= warning_band_m ? kGeofenceWarning : kGeofenceInside;
}

inline int round_half_even(const double value) {
  const double floored = std::floor(value);
  const double fraction = value - floored;
  if (fraction < 0.5 - 1e-9) {
    return static_cast<int>(floored);
  }
  if (fraction > 0.5 + 1e-9) {
    return static_cast<int>(floored + 1.0);
  }
  const int lower = static_cast<int>(floored);
  return (lower % 2 == 0) ? lower : lower + 1;
}

inline int compute_risk_score(
    const double body_temp_c, const double baseline_temp_c, const double thi,
    const bool is_restless, const int geofence_status, const bool is_isolated,
    const bool is_tampered, const double temp_offset_low,
    const double temp_offset_high, const double thi_low, const double thi_high,
    const double restless_severity, const double geo_warning_severity,
    const double geo_breach_severity, const double isolation_severity,
    const double tamper_severity) {
  const double temperature = clamp(
      (body_temp_c - baseline_temp_c - temp_offset_low) /
          (temp_offset_high - temp_offset_low),
      0.0, 1.0);
  const double heat = clamp((thi - thi_low) / (thi_high - thi_low), 0.0, 1.0);
  const double geofence = geofence_status == kGeofenceBreach
                               ? geo_breach_severity
                               : (geofence_status == kGeofenceWarning
                                      ? geo_warning_severity
                                      : 0.0);
  const double product = (1.0 - temperature) * (1.0 - heat) *
                         (1.0 - (is_restless ? restless_severity : 0.0)) *
                         (1.0 - geofence) *
                         (1.0 - (is_isolated ? isolation_severity : 0.0)) *
                         (1.0 - (is_tampered ? tamper_severity : 0.0));
  return round_half_even(100.0 * (1.0 - product));
}

inline const char* alert_band(const int risk_score, const int green_max, const int yellow_max) {
  return risk_score <= green_max ? "green" : (risk_score <= yellow_max ? "yellow" : "red");
}

}  // namespace collar_math
