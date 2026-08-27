"""Generate C++ parity constants from the repository's tracked JSON contract."""

from pathlib import Path
import json

Import("env")


def cpp_bool(value: bool) -> str:
    return "true" if value else "false"


project_dir = Path(env["PROJECT_DIR"]).resolve()
contract_path = project_dir.parents[2] / "contracts" / "telemetry_parity_v1.json"
contract = json.loads(contract_path.read_text(encoding="utf-8"))

if contract.get("schema_version") != 1:
    raise RuntimeError("Unsupported telemetry parity contract schema version")

out_dir = project_dir / ".pio" / "generated"
out_dir.mkdir(parents=True, exist_ok=True)
header_path = out_dir / "parity_contract.h"

geofence = contract["geofence"]
risk = contract["risk"]
telemetry = contract["telemetry"]

lines = [
    "// Generated from contracts/telemetry_parity_v1.json. Do not edit.",
    "#pragma once",
    "#include <cstddef>",
    "#include \"parity_math.h\"",
    "namespace parity_contract {",
    "using Coord = collar_math::Coord;",
    "struct ThiVector { double ambient_temp_c; double humidity_pct; double expected_thi; };",
    "struct GeofenceVector { double latitude; double longitude; int expected_status; };",
    "struct RiskVector { double body_temp_c; double thi; bool restless; int geofence_status; bool isolated; bool tampered; int expected_score; };",
    f"constexpr int kPhysicalCollarId = {telemetry['physical_collar_id']};",
    f"constexpr const char kStatusSource[] = \"{telemetry['status_source']}\";",
    f"constexpr double kEarthRadiusM = {geofence['earth_radius_m']:.12g};",
    f"constexpr double kWarningBandM = {geofence['warning_band_m']:.12g};",
    "constexpr Coord kPasturePolygon[] = {",
]
for lat, lon in geofence["polygon"]:
    lines.append(f"    {{{lat:.12g}, {lon:.12g}}},")
lines += [
    "};",
    f"constexpr std::size_t kPasturePolygonSize = {len(geofence['polygon'])};",
    f"constexpr double kBaselineTempC = {risk['baseline_temp_c']:.12g};",
    f"constexpr double kTempOffsetLow = {risk['temp_offset_low']:.12g};",
    f"constexpr double kTempOffsetHigh = {risk['temp_offset_high']:.12g};",
    f"constexpr double kThiLow = {risk['thi_low']:.12g};",
    f"constexpr double kThiHigh = {risk['thi_high']:.12g};",
    f"constexpr double kRestlessSeverity = {risk['restless']:.12g};",
    f"constexpr double kGeoWarnSeverity = {risk['geo_warn']:.12g};",
    f"constexpr double kGeoBreachSeverity = {risk['geo_breach']:.12g};",
    f"constexpr double kIsolationSeverity = {risk['social_isolation']:.12g};",
    f"constexpr double kTamperSeverity = {risk['collar_tamper']:.12g};",
    f"constexpr int kGreenMax = {risk['green_max']};",
    f"constexpr int kYellowMax = {risk['yellow_max']};",
    "constexpr ThiVector kThiVectors[] = {",
]
for vector in contract["thi_vectors"]:
    lines.append(
        "    {"
        f"{vector['ambient_temp_c']:.12g}, {vector['humidity_pct']:.12g}, "
        f"{vector['expected_thi']:.12g}" "},"
    )
lines += [
    "};",
    f"constexpr std::size_t kThiVectorCount = {len(contract['thi_vectors'])};",
    "constexpr GeofenceVector kGeofenceVectors[] = {",
]
for vector in geofence["vectors"]:
    lines.append(
        f"    {{{vector['latitude']:.12g}, {vector['longitude']:.12g}, {vector['expected_status']}}},"
    )
lines += [
    "};",
    f"constexpr std::size_t kGeofenceVectorCount = {len(geofence['vectors'])};",
    "constexpr RiskVector kRiskVectors[] = {",
]
for vector in risk["vectors"]:
    lines.append(
        "    {"
        f"{vector['body_temp_c']:.12g}, {vector['thi']:.12g}, "
        f"{cpp_bool(vector['restless'])}, {vector['geofence_status']}, "
        f"{cpp_bool(vector['isolated'])}, {cpp_bool(vector['tampered'])}, "
        f"{vector['expected_score']}" "},"
    )
lines += [
    "};",
    f"constexpr std::size_t kRiskVectorCount = {len(risk['vectors'])};",
    "}  // namespace parity_contract",
    "",
]
header_path.write_text("\n".join(lines), encoding="utf-8")
env.Append(CPPPATH=[str(out_dir)])
