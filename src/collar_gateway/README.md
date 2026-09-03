# Collar Gateway — Deliverable 8

This directory contains the Phase 2 firmware for the one physical Collar-1
prototype. It is an ESP32 DevKit V1 / WROOM-32 PlatformIO project that emits
the same immutable ThingSpeak telemetry fields as the Phase 1 simulator.

It is not the Deliverable 9 raw-IMU gateway. The MPU6050 is sampled locally at
10 Hz only to produce a conservative temporary behaviour code; it does not
stream raw IMU frames to the laptop.

## Hardware wiring

For the full pin-by-pin connection diagram, power guidance, and bring-up
sequence, see [ESP32 Wiring Guide](ESP32_WIRING_GUIDE.md).

| Module | ESP32 connection | Notes |
|---|---|---|
| MLX90614 | I²C SDA GPIO21, SCL GPIO22 | Share the I²C bus with MPU6050. |
| MPU6050 | I²C SDA GPIO21, SCL GPIO22 | Used for local 10 Hz motion sampling. |
| DHT11 | DATA GPIO15 | Use the breakout module's pull-up, or add the module-recommended pull-up resistor. `GPIO4` was the original pin but is dead on some ESP32 boards (ADR-024). |
| NEO-6M | GPS TX → GPIO16, GPS RX ← GPIO17 | The RX connection is optional unless the module is configured. |

Connect all grounds. Power each breakout board at the voltage specified by its
manufacturer; do not feed 5 V logic into an ESP32 GPIO. The defaults are
defined in `firmware/include/device_config.h` and can be changed for a custom
carrier board.

## Local credentials and build

Create a Python 3.11+ virtual environment and install the repository tools once:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

From the repository root, copy `.env.example` to `.env` and set these values:

```dotenv
WIFI_SSID=your-wifi-name
WIFI_PASSWORD=your-wifi-password
THINGSPEAK_CHANNEL_1_WRITE_API_KEY=your-channel-1-write-key
```

The PlatformIO pre-build script reads the untracked root `.env`; it never
writes a credentials header or prints values. Install PlatformIO locally if it
is not already available, then run:

```bash
cd src/collar_gateway/firmware
pio test -e native
pio run -e esp32dev -t upload
pio device monitor -b 115200
```

`pio test -e native` runs the generated C++ THI, risk, alert-band, and
geofence parity vectors without needing the physical board. `pio run` requires
the `.env` values; the native parity test does not.

## Laptop-controlled battery demonstration

No battery ADC is connected. `field8` is a manually controlled demo value that
starts at `100` while the ESP32 is USB powered. In the serial monitor use:

```text
status
battery 67
publish
battery 0
help
```

`battery <0..100>` changes `field8`. `battery 0` starts a logical dropout:
the serial telemetry shows `evt=DROPOUT`, risk `100`, and Channel 1 posts stop.
This deliberately does not fabricate a fresh healthy record. Restoring a
non-zero value resumes normal telemetry after valid sensors and GPS are
available. `publish` requests the next eligible post but can never bypass the
15-second physical floor.

## Telemetry and simulator connection

The firmware validates MLX90614 body temperature, DHT11 values, and a fresh
GPS fix before it publishes a complete Channel 1 row. The output always uses:

```text
field1=body-temperature; field2=THI; field3=behaviour;
field4=latitude; field5=longitude; field6=predicted-risk-score;
field7=geofence; field8=manual-battery; status=id=01;...;src=SENSOR
```

**Indoor test fallback (ADR-024):** without a live MLX90614 reading or a
fresh GPS fix, the firmware substitutes a fixed healthy body temperature and
a fixed pasture coordinate so Channel 1 can still be exercised end-to-end
during development. A row built this way always posts `src=SPOOF` in place of
`src=SENSOR` and is flagged `[INDOOR TEST FALLBACK]` on the serial monitor —
it is never indistinguishable from a genuine sensor reading. Treat a feed
full of `src=SPOOF` rows as "the pipeline works," not as validated field data.

The local motion classifier returns Resting (`0`), Grazing (`1`, for
mid-range motion that isn't clearly resting or walking — ADR-024), Walking
(`3`), or Other/Unknown (`5`, only when a full 5-second motion window hasn't
been collected yet); it never treats uncertain data as Restless (`4`). Its
thresholds are intentionally provisional until Deliverables 9 and 10 provide
the raw-IMU collection and evaluated model pipeline.

For the Phase 1 HUD to show the physical collar, set
`thingspeak.channel_1_id` in `config/default_config.yaml`, add
`THINGSPEAK_READ_API_KEY` to the same local `.env`, then run:

```bash
.venv/bin/python src/main.py --mode live --config config/default_config.yaml --hud
```

Channel 1 samples display as fresh physical ID 1 only when every required
telemetry field is valid. GPS alone still anchors herd movement but does not
make Collar 1 telemetry fresh.

## Hardware verification checklist

1. Confirm the serial boot line detects MLX90614 and MPU6050; wait for DHT11
   and GPS to become valid with `status`.
2. Check a normal Channel 1 write occurs no more often than every 30 seconds.
3. Move outside the configured pasture to verify geofence breach risk and the
   15-second alert cadence.
4. Set `battery 0` and confirm writes stop; restore `battery 100` and confirm
   normal operation returns.
5. With the simulator in live HUD mode, confirm ID 1 moves from grey/stale to
   a fresh physical marker after the first complete Channel 1 row.
6. Check the boot log's MPU `WHO_AM_I` value. `0x70` means an MPU-6500 clone —
   see the wiring guide's MPU-6500 note before trusting motion data.
7. Before treating any test run as validated field data, confirm the `status`
   field says `src=SENSOR`, not `src=SPOOF` — the latter means one or more
   readings were the ADR-024 indoor fallback, not a live sensor.
