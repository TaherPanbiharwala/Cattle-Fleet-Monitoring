"""Load untracked project .env credentials into the local PlatformIO build."""

from pathlib import Path
import os

Import("env")

REQUIRED_KEYS = (
    "WIFI_SSID",
    "WIFI_PASSWORD",
    "THINGSPEAK_CHANNEL_1_WRITE_API_KEY",
)


def find_dotenv() -> Path | None:
    project_dir = Path(env["PROJECT_DIR"]).resolve()
    for directory in (project_dir, *project_dir.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


dotenv_path = find_dotenv()
values = parse_dotenv(dotenv_path) if dotenv_path is not None else {}
# Explicit environment variables make the build usable in CI without creating
# a local file. They take precedence over .env and are never printed.
for key in REQUIRED_KEYS:
    if os.environ.get(key):
        values[key] = os.environ[key]
missing = [key for key in REQUIRED_KEYS if not values.get(key)]
if missing:
    raise RuntimeError(
        "Missing firmware credentials. Copy .env.example to .env or provide "
        "these environment variables: " + ", ".join(missing)
    )

def cpp_string_literal(value: str) -> str:
    """Return a PlatformIO-safe macro value that remains a C++ string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    # PlatformIO/SCons consumes one layer of quotes while constructing -D flags.
    # The escaped quotes below reach the C++ preprocessor intact.
    return '\\"' + escaped + '\\"'


# Values are never printed; PlatformIO's .pio directory is ignored because it
# can contain local compile flags.
env.Append(
    CPPDEFINES=[
        ("COLLAR_WIFI_SSID", cpp_string_literal(values["WIFI_SSID"])),
        ("COLLAR_WIFI_PASSWORD", cpp_string_literal(values["WIFI_PASSWORD"])),
        (
            "COLLAR_THINGSPEAK_CHANNEL_1_WRITE_API_KEY",
            cpp_string_literal(values["THINGSPEAK_CHANNEL_1_WRITE_API_KEY"]),
        ),
    ]
)
