# ESP32 Collar Wiring Guide

This guide wires the Phase 2 Collar-1 firmware to an ESP32 DevKit V1 /
WROOM-32, MLX90614, DHT11, MPU6050, and NEO-6M. The pin assignments match
[`firmware/include/device_config.h`](firmware/include/device_config.h).

The collar has no battery ADC. Do not connect a battery-voltage signal to the
ESP32 for this build. The laptop serial monitor sets the required telemetry
`field8` value with `battery <0..100>`.

## Pin reference

| ESP32 pin | Connect to | Purpose |
|---|---|---|
| `3V3` | MLX90614 `VCC`, MPU6050 `VCC`, DHT11 `VCC` | 3.3 V sensor power. |
| `GND` | Every module `GND` | A shared ground is required. |
| `GPIO21` | MLX90614 `SDA` and MPU6050 `SDA` | Shared I²C data bus. |
| `GPIO22` | MLX90614 `SCL` and MPU6050 `SCL` | Shared I²C clock bus. |
| `GPIO4` | DHT11 `DATA` / `OUT` | Ambient temperature and humidity data. |
| `GPIO16` | NEO-6M `TX` | ESP32 UART2 receive; connect TX to RX. |
| `GPIO17` | NEO-6M `RX` (optional) | ESP32 UART2 transmit; required only to configure the GPS module. |
| USB | Laptop | Firmware upload, serial monitor, and manual battery control. |

`GPIO34` is deliberately unused. It was reserved for the original battery ADC
design and must not be wired for this no-ADC build.

## Connection diagram

```text
                         ESP32 DevKit V1 / WROOM-32
                      ┌─────────────────────────────┐
  MLX90614 SDA ───────┤ GPIO21                       │
  MPU6050  SDA ───────┤ GPIO21                       │
  MLX90614 SCL ───────┤ GPIO22                       │
  MPU6050  SCL ───────┤ GPIO22                       │
  DHT11 DATA ─────────┤ GPIO4                        │
  NEO-6M TX ──────────┤ GPIO16 (UART2 RX)            │
  NEO-6M RX ──────────┤ GPIO17 (UART2 TX, optional)  │
                      │                             │
  MLX + MPU + DHT VCC ┤ 3V3                           │
  All module grounds ─┤ GND                           │
                      │ USB ───── laptop             │
                      └─────────────────────────────┘
```

## Wire each module

### MLX90614 body-temperature sensor

| MLX90614 pin | ESP32 connection |
|---|---|
| `VCC` / `VIN` | `3V3` |
| `GND` | `GND` |
| `SDA` | `GPIO21` |
| `SCL` | `GPIO22` |

The firmware accepts an object temperature only from `35–43°C`. Pointing the
sensor at open air, a desk, or a distant object normally produces an invalid
body-temperature sample, so Channel 1 will not publish until the sensor sees a
plausible target.

### MPU6050 motion sensor

| MPU6050 pin | ESP32 connection |
|---|---|
| `VCC` | `3V3` |
| `GND` | `GND` |
| `SDA` | `GPIO21` |
| `SCL` | `GPIO22` |
| `INT` | Leave disconnected |

The MLX90614 and MPU6050 share the same I²C bus. Their default addresses are
different (`0x5A` and typically `0x68`), so they can coexist. Most breakout
boards include I²C pull-up resistors. If yours do not, add one 4.7–10 kΩ
pull-up from `SDA` to `3V3` and one from `SCL` to `3V3`; do not add many sets of
parallel pull-ups across multiple modules.

### DHT11 ambient sensor

| DHT11 pin | ESP32 connection |
|---|---|
| `VCC` | `3V3` |
| `GND` | `GND` |
| `DATA` / `OUT` | `GPIO4` |

For a bare four-pin DHT11, add a 4.7–10 kΩ resistor from `DATA` to `3V3`. Most
three-pin DHT11 modules already include this resistor. The firmware reads the
DHT11 every two seconds; temporary `status` output with `dht=no` during boot is
normal.

### NEO-6M GPS

| NEO-6M pin | ESP32 connection |
|---|---|
| `TX` | `GPIO16` |
| `RX` | `GPIO17` (optional) |
| `GND` | `GND` |
| `VCC` | The voltage specified by your exact GPS breakout board |

UART data is crossed: GPS `TX` goes to ESP32 receive (`GPIO16`), and GPS `RX`
goes to ESP32 transmit (`GPIO17`). The firmware listens at `9600` baud.

Do not assume every NEO-6M breakout has the same power input. Many common
boards accept 5 V through an onboard regulator; others require 3.3 V. Check
the board label or datasheet before connecting `VCC`. Crucially, the ESP32 is
not 5 V tolerant: GPS `TX` must be at or below 3.3 V. Use a level shifter or
voltage divider if your GPS board outputs 5 V logic.

The device requires a GPS fix no older than five seconds before posting a
complete Collar-1 row. For first acquisition, place the antenna outside with a
clear view of the sky and allow several minutes.

## Safe assembly order

1. Disconnect USB power from the ESP32.
2. Wire the shared `GND`, then MLX90614 and MPU6050 power and I²C lines.
3. Wire the DHT11 power and `DATA` line; add its pull-up only if your board
   lacks one.
4. Wire NEO-6M ground and UART `TX → GPIO16`. Add `GPIO17 → RX` only when you
   need GPS configuration.
5. Check every connection once more, then attach USB to the laptop. Use a
   stable USB port or a regulated supply: Wi‑Fi transmission causes short power
   bursts.
6. Build/upload the firmware and open the serial monitor:

   ```bash
   cd src/collar_gateway/firmware
   pio run -e esp32dev -t upload
   pio device monitor -b 115200
   ```

7. Enter `status`. Wait until MLX, DHT, MPU, and GPS report valid before
   expecting a ThingSpeak Channel 1 post.

## Laptop controls and expected output

At `115200` baud, the USB serial monitor accepts:

```text
status
battery 75
publish
battery 0
help
```

`battery 75` makes the next valid telemetry row report `field8=75`.
`battery 0` activates a logical dropout: it discards queued data and suppresses
future Channel 1 posts until you set a non-zero value. This is intentional—an
offline collar must not fabricate a fresh healthy record.

## Troubleshooting

| Symptom | Check |
|---|---|
| MLX90614 or MPU6050 is missing at boot | Verify `SDA` is GPIO21, `SCL` is GPIO22, both modules share `GND`, and the module is powered at 3.3 V. |
| `dht=no` remains in `status` | Verify `DATA` is GPIO4, the pin order on the DHT11, and the pull-up resistor. |
| GPS never becomes valid | Confirm GPS `TX → GPIO16`, shared ground, a clear sky view, and that the module uses 9600 baud. |
| ESP32 resets when Wi‑Fi starts | Use a stable USB cable/supply and verify there is no short between power and ground. |
| No ThingSpeak posts | A complete valid MLX, DHT, and fresh GPS record is required; then check Wi‑Fi and the local `.env` credentials. |
| Simulator HUD stays grey for ID 1 | Set `thingspeak.channel_1_id` and `THINGSPEAK_READ_API_KEY` for the simulator, then wait for its next Channel 1 sniff. |

## Configuration source

The firmware’s pin mappings, timing, and sensor limits are public constants in
[`firmware/include/device_config.h`](firmware/include/device_config.h). Change
that file—not the wiring guide—if you move to a custom ESP32 carrier board.
