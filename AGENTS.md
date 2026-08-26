# AGENTS.md — AI Agent Operating Manual & System Context
**Intelligent Cattle Fleet Management Platform**
*Target Audience: AI Coding Assistants (Claude Code — primary) & Human Developers*

---

## 1. Welcome & Primary Directive

You are working on the **Intelligent Cattle Fleet Management Platform**. This project bridges embedded wearable computing (ESP32 IoT collars) with cloud telemetry (ThingSpeak), digital-twin fleet simulation, and machine learning analytics.

> **PRIMARY DIRECTIVE:**
> Whenever you start a task in this repository, you must read:
> 1. [DECISION.md](file:///Users/taherpanbiharwala/Desktop/IoT/DECISION.md) — Complete historical & architectural decision records.
> 2. [Cattle_Fleet_Management_Master_PRD.md](file:///Users/taherpanbiharwala/Desktop/IoT/Cattle_Fleet_Management_Master_PRD.md) — System requirements, phased roadmap, and exact mathematical specifications.
> 3. [Cattle_Fleet_Management_HerdSimulator_PRD.md](file:///Users/taherpanbiharwala/Desktop/IoT/Cattle_Fleet_Management_HerdSimulator_PRD.md) — Digital-twin simulator component specification.

---

## 2. Project Summary & Phased Architecture

The platform is designed around the reality that only **one physical collar (ESP32 + sensors, BOM ₹2,450)** will initially be built, while fleet analytics require **20 cattle**. We bridge this gap using a **Python digital-twin simulator**.

```
┌────────────────────────────────────────────────────────┐
│  Phase 1 (MVP - Current Focus):                        │
│  Autonomous Herd Digital-Twin Simulator (20 Cattle)    │
│  • 1 Reserved Physical Identity (ID 1)                 │
│  • 19 Simulated Virtual Cattle (IDs 2-20)              │
│  • Deterministic Rule Scoring, Geofence, Physiology    │
│  • Multiplexed ThingSpeak Channel 2 Uplink (30s)       │
│  • Zero-Dependency Local Web HUD (Leaflet.js)          │
│  • High-Speed Dry-Run & 190-Pair Ground Truth Logging  │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 2: Physical Collar Parity & Wi-Fi Gateway       │
│  • ESP32 Firmware matching P1 Telemetry Contract       │
│  • 10 Hz Wi-Fi IMU Acquisition & Packet Gap Logging    │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 3: WASP-Lab IMU Dataset Behaviour ML            │
│  • 112 Spectral/Statistical Features per 5s Window     │
│  • Cow-Grouped Macro F1 Benchmarks (RF, GBT, CNN)      │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 4: Personalized Per-Cow Deviation Modeling      │
│  • Longitudinal Baseline Learning (7 healthy days)     │
│  • Robust z-score (Median & MAD) Anomaly Scoring       │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 5: Supervised Fever & Lameness Risk Research    │
│  • Label-gated risk predictions (Never a diagnosis)    │
└────────────────────────────────────────────────┘
```

---

## 3. Non-Negotiable Golden Rules

1. **Immutable Telemetry Contract:**
   All 8 fields and the reserved `status` field must strictly adhere to the contract. No module may alter field order, rename fields, or drop fields.
2. **ThingSpeak Rate-Limit Compliance:**
   - **Never** post faster than the 15-second physical floor to any ThingSpeak channel.
   - Default posting cadence for Channel 2 multiplexer is **30 seconds**.
   - Keep combined annual usage strictly below 3,000,000 writes (~8,200 writes/day).
3. **No Solar Recharge:**
   Collar hardware has no solar panels. Battery level (0–100%) must only deplete based on activity drain ($1\times$ base, $3\times$ during breach/alert bursts). When battery hits 0%, collar triggers dropout. Dropout telemetry must report as **stale and critical** (`risk_score=100`, red band) — never a fabricated healthy score just because no fresh data exists (see ADR-017; this was violated once already and fixed).
4. **Strict Mathematical Determinism:**
   Given the same configuration, scenario, and random seed (default `42`), the simulation must produce byte-identical normalized telemetry across runs.
5. **No Secrets in Code:**
   ThingSpeak API keys, Wi-Fi credentials, or tokens must only be loaded via environment variables or a `.env` file (never hardcoded, never logged).
6. **Safe Behaviour Classification:**
   Behaviour codes: `0=Resting, 1=Grazing, 2=Ruminating, 3=Walking, 4=Restless, 5=Other/Unknown`. **Never** map miscellaneous or low-confidence behaviours to `Restless (4)`.
7. **No Clinical / Veterinary Diagnosis Claims:**
   All health outputs must be presented as *"Predicted Risk Score"* or *"Anomaly Indicator"*. Never generate output claiming to provide a clinical diagnosis or medical treatment.
8. **Phase Isolation:**
   Phase 1 (MVP Digital Twin) must run independently without dependencies on Phase 2 (ESP32) or Phase 3 (ML models).

---

## 4. Telemetry & Data Specifications

### 4.1 ThingSpeak 8-Field Schema (Channels 1 & 2)

| Field | Meaning | Type | Unit / Encoding |
|---|---|---|---|
| `field1` | Body Temperature | float | °C (e.g. `38.6`) |
| `field2` | THI (Heat Stress) | float | Calculated index |
| `field3` | Behaviour Code | int | `0` Resting, `1` Grazing, `2` Ruminating, `3` Walking, `4` Restless, `5` Unknown |
| `field4` | GPS Latitude | float | Decimal degrees (e.g. `12.9716`) |
| `field5` | GPS Longitude | float | Decimal degrees (e.g. `79.1589`) |
| `field6` | Composite Risk Score | int | `0` to `100` |
| `field7` | Geofence Status | int | `0` Inside, `1` 10m Warning Zone, `2` Breach |
| `field8` | Battery Level | int | `0` to `100` (%) |
| `status` | Identity & Events | text | Semicolon-delimited: `id=XX;evt=YY;src=ZZ` (e.g. `id=07;evt=FEVER;src=RULE`) |

### 4.2 Mathematical Formulas

* **THI Formula:**
  $$\text{THI} = (1.8 \times T_{\text{amb}} + 32) - (0.55 - 0.0055 \times RH) \times (1.8 \times T_{\text{amb}} - 26)$$
* **Severity Components:**
  $$S_{\text{temp}} = \text{clamp}\left(\frac{T_{\text{body}} - \text{baseline} - 0.5}{1.5}, 0, 1\right)$$
  $$S_{\text{THI}} = \text{clamp}\left(\frac{\text{THI} - 68}{16}, 0, 1\right)$$
  $$S_{\text{restless}} = 0.35, \quad S_{\text{geo\_warn}} = 0.50, \quad S_{\text{geo\_breach}} = 0.90, \quad S_{\text{isol}} = 0.70, \quad S_{\text{tamper}} = 0.90$$
* **Composite Risk Score:**
  $$\text{risk} = \text{round}\left(100 \times \left(1 - \prod_{i} (1 - S_i)\right)\right) \in [0, 100]$$

### 4.3 Scenario JSON Contract

Declarative fault-injection scenarios (`--scenario config/scenarios/*.json`) use this shape, enforced by `scenario_runner.load_scenario()`, which fails fast on any violation before the run starts (see ADR-016):

```json
{
  "schema_version": 1,
  "scenario_id": "demo_scenario",
  "seed": 42,
  "events": [
    {
      "animal_id": 5,
      "type": "fever_onset",
      "start_sim_second": 300,
      "duration_seconds": 1200,
      "event_id": "demo-fever-1",
      "params": {"peak_offset_c": 1.8}
    }
  ]
}
```

| Field | Required | Notes |
|---|---|---|
| `type` | Yes | One of `fever_onset, heat_stress, geofence_breach, tamper, social_isolation, collar_dropout` |
| `start_sim_second` | Yes | When the event activates |
| `duration_seconds` | Yes, positive | Event auto-clears `duration_seconds` after activation |
| `event_id` | No | Auto-generated as `<scenario_id>-<index>` if omitted; must be unique in the file |
| `params` | No | Merged over per-type defaults in `scenario_runner._DEFAULT_PARAMS` |

**Validated before startup:** unknown `type`, duplicate `event_id`, invalid `animal_id` (when the loader is given the herd's valid ID range), non-positive `duration_seconds`, and any two events of the **same type on the same animal** with overlapping `[start_sim_second, start_sim_second + duration_seconds)` windows. Different event types on the same animal, or the same type on different animals, may overlap freely — that's what lets faults compose.

Events activated live via the CLI/API (not from a scenario file) carry no `duration_seconds` and run until an explicit `clear <id>` — auto-expiry only applies to the scripted timeline.

---

## 5. Repository Structure

```
IoT/
├── AGENTS.md                       # This operating manual
├── DECISION.md                     # Architecture decision records
├── Cattle_Fleet_Management_Master_PRD.md
├── Cattle_Fleet_Management_HerdSimulator_PRD.md
├── pyproject.toml                  # Python build & dependency spec
├── config/
│   ├── default_config.yaml         # Validated v1 YAML configuration
│   └── scenarios/
│       ├── demo_scenario.json      # 20-minute classroom walkthrough scenario
│       └── fault_injection.json    # Full test matrix of all 6 anomaly types
├── src/
│   ├── herd_simulator/
│   │   ├── __init__.py
│   │   ├── config.py               # Strict YAML schema validation
│   │   ├── utils/
│   │   │   ├── geo.py              # Haversine, Ray-casting polygon, 10m warning band
│   │   │   └── risk.py             # Deterministic severity product-rule engine
│   │   ├── models/
│   │   │   ├── animal.py           # Animal runtime state & per-run profile
│   │   │   ├── behaviour.py        # 5-state Markov state machine
│   │   │   ├── movement.py         # Centroid drift + flocking + Collar-1 anchoring
│   │   │   ├── physiology.py       # Baseline + diurnal curve + fever ramp
│   │   │   └── battery.py          # Activity-aware drain (no solar recharge)
│   │   ├── engine/
│   │   │   ├── simulator.py        # 1-second tick loop coordinator
│   │   │   ├── scenario_runner.py  # JSON event parser & event composition
│   │   │   ├── scheduler.py        # Round-robin multiplexer & priority queue
│   │   │   └── live_cli.py         # Terminal prompt for live fault injection
│   │   ├── services/
│   │   │   ├── logger.py           # Buffered per-run logging: 8 output files + ground truth
│   │   │   ├── replay.py           # Telemetry CSV reader + normalization for determinism
│   │   │   ├── thingspeak.py       # Ch.2 POST writer & Ch.1 GET sniffer (not yet built)
│   │   │   └── api_server.py       # REST API & static HUD server (not yet built)
│   │   └── web/                    # (not yet built)
│   │       ├── index.html          # Leaflet.js live map & telemetry HUD
│   │       ├── app.js
│   │       └── style.css
│   └── main.py                     # CLI entry point (not yet built — see ADR-017)
└── tests/
    ├── test_golden_vectors.py      # Golden vector parity tests (THI, Risk, Geo)
    ├── test_config.py              # YAML validation guard tests (severity relationship checks)
    ├── test_models.py              # Unit tests for physiology, behaviour, movement
    ├── test_scenario_engine.py     # Fault injection & overlap policy tests
    ├── test_scheduler.py           # 15s floor, starvation prevention, sweep checks
    ├── test_logger.py              # Per-run logging: buffered I/O, ground truth, determinism
    └── test_replay.py              # Replay reader, normalization, end-to-end determinism
```

---

## 6. Execution Modes & CLI Commands

The simulator supports multiple execution modes via `main.py`:

```bash
# 1. High-Speed Dry Run (runs a full 24h cycle in <10s locally, 0 ThingSpeak writes)
python src/main.py --mode dry-run --config config/default_config.yaml --duration-hours 24

# 2. Offline Mode with Local Web HUD (real-time 1s ticks, local Leaflet map on port 8000, 0 ThingSpeak writes)
python src/main.py --mode offline --config config/default_config.yaml --hud

# 3. Live Mode (real-time 1s ticks, posts to ThingSpeak Ch.2 every 30s, sniffs Ch.1)
python src/main.py --mode live --config config/default_config.yaml --scenario config/scenarios/demo_scenario.json --hud

# 4. Replay Mode (reproduces a previous run deterministically from local log)
python src/main.py --mode replay --log-dir logs/run_20260820_001/

# 5. Run Full Test Suite
pytest tests/ -v
```

---

## 7. Local REST API Endpoints

When the simulator runs with the local HUD server enabled (`--hud`, binds to `127.0.0.1:8000`):

* `GET /api/health` — System status, uptime, simulated second, memory, queue depth.
* `GET /api/state` — Complete snapshot of all 20 cattle (positions, states, risks, battery).
* `GET /api/history?id=<id>&limit=<n>` — Historical telemetry for a specific cow (1–10,000 records).
* `GET /api/queue` — Current round-robin transmission queue and priority slots.
* `POST /api/events` — Inject a live anomaly event (accepts JSON event payload, returns `201`).
* `DELETE /api/events/<event_id>` — Clear an active injected event (returns `204`).

Standard Error Format:
```json
{
  "code": "INVALID_EVENT_TYPE",
  "message": "Event type 'invalid_type' is not supported.",
  "details": {"supported_types": ["fever_onset", "heat_stress", "geofence_breach", "tamper", "social_isolation", "collar_dropout"]}
}
```

---

## 8. Coding & Engineering Guidelines

1. **Language & Tooling:** Python 3.11+, `pytest` for testing, `pyyaml` for configuration.
2. **Zero Dependency Principle:** The core simulation loop, math utilities, and local web server must use standard library modules (`http.server`, `math`, `json`, `urllib`, `dataclasses`, `typing`) where possible to guarantee zero-cost portability.
3. **Type Hinting:** All functions, methods, and dataclasses must use standard Python type annotations (`typing` / built-in generics).
4. **Error Handling:** Gracefully handle network timeouts and ThingSpeak HTTP failures with exponential backoff; never crash the internal 1-second simulation loop on network disconnects.
5. **Clean Logs:** Telemetry logs and ground-truth pairs must be written efficiently with buffered stream writers.
