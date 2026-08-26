# Architecture Decision Record (ADR) & Project Decision Log
**Intelligent Cattle Fleet Management Platform**
*Last Updated: 2026-08-27 · Status: Active / Approved*

---

## Executive Summary & Context

This document preserves the complete historical and technical rationale for all architectural, mathematical, hardware, and operational decisions taken for the **Intelligent Cattle Fleet Management Platform**. Any developer, researcher, or AI coding assistant (Claude Code — primary) must review this log before proposing or executing changes.

---

## Index of Decisions

- [ADR-001: 5-Phase Project Roadmap & MVP Scope Boundary](#adr-001-5-phase-project-roadmap--mvp-scope-boundary)
- [ADR-002: Fleet Population & Identity Allocation (20 Cattle)](#adr-002-fleet-population--identity-allocation-20-cattle)
- [ADR-003: Physical Collar Sensor & Hardware Baseline (BOM ₹2,450)](#adr-003-physical-collar-sensor--hardware-baseline-bom-2450)
- [ADR-004: ThingSpeak Free-Tier Multi-Channel Allocation & Rate Limits](#adr-004-thingspeak-free-tier-multi-channel-allocation--rate-limits)
- [ADR-005: Stable Telemetry Contract & Reserved Status Field Protocol](#adr-005-stable-telemetry-contract--reserved-status-field-protocol)
- [ADR-006: 6-State Behavioural Classification & Safe Code Mapping](#adr-006-6-state-behavioural-classification--safe-code-mapping)
- [ADR-007: Deterministic Product-Rule Risk Scoring Formula](#adr-007-deterministic-product-rule-risk-scoring-formula)
- [ADR-008: Geofence Spatial Geometry & 10-Meter Warning Band](#adr-008-geofence-spatial-geometry--10-meter-warning-band)
- [ADR-009: Activity-Aware Battery Drainage (Strictly No Solar Recharge)](#adr-009-activity-aware-battery-drainage-strictly-no-solar-recharge)
- [ADR-010: Collar-1 Sniffing & Synchronization (True Digital Twin Coupling)](#adr-010-collar-1-sniffing--synchronization-true-digital-twin-coupling)
- [ADR-011: Local Web Visualizer HUD Architecture (Zero-Dependency)](#adr-011-local-web-visualizer-hud-architecture-zero-dependency)
- [ADR-012: High-Speed Dry-Run Mode & Quota Preservation](#adr-012-high-speed-dry-run-mode--quota-preservation)
- [ADR-013: 190-Pair Ground-Truth Matrix Logging](#adr-013-190-pair-ground-truth-matrix-logging)
- [ADR-014: Dual Fault-Injection Engine (Declarative JSON + Live CLI)](#adr-014-dual-fault-injection-engine-declarative-json--live-cli)
- [ADR-015: Machine Learning Dataset, Evaluation, and Diagnostic Guardrails](#adr-015-machine-learning-dataset-evaluation-and-diagnostic-guardrails)
- [ADR-016: Scenario JSON Contract Correction (Deliverable #3 Fix)](#adr-016-scenario-json-contract-correction-deliverable-3-fix)
- [ADR-017: Pre-Landing Review Fixes — Deliverables #1-#3](#adr-017-pre-landing-review-fixes--deliverables-1-3)
- [ADR-018: Logging, Ground Truth & Replay Architecture (Deliverable #4)](#adr-018-logging-ground-truth--replay-architecture-deliverable-4)

---

## ADR-001: 5-Phase Project Roadmap & MVP Scope Boundary

* **Status:** Approved
* **Context:** The project spans hardware construction, digital-twin simulation, edge computing, machine learning, and veterinary analytics. Building everything simultaneously risks architectural deadlock.
* **Decision:** Decouple delivery into five sequential phases with clear gates:
  * **Phase 1 (P1 - MVP):** Autonomous herd digital-twin simulator running locally on Python 3.11 with ThingSpeak uplink, local HUD, and deterministic logging.
  * **Phase 2 (P2):** One physical ESP32 collar matching P1 telemetry parity + Wi-Fi raw IMU collection gateway.
  * **Phase 3 (P3):** WASP-lab public IMU dataset behavior classification & walking analytics benchmark.
  * **Phase 4 (P4):** Personalized per-cow deviation modeling (z-score on longitudinal features).
  * **Phase 5 (P5):** Supervised fever and lameness risk research (strictly label-gated).
* **Consequences:** P1 is completely self-contained. P2–P5 cannot modify the P1 telemetry schema or block P1 acceptance.

---

## ADR-002: Fleet Population & Identity Allocation (20 Cattle)

* **Status:** Approved
* **Context:** Fleet analytics (social grouping, automatic roll call, pairwise proximity, herd drift) require multi-animal scale, but the budget allows building only one physical collar.
* **Decision:** Fixed reference fleet of **20 cattle**:
  * **ID 1 (Physical Collar):** Reserved identity for the real ESP32 collar. Displays as offline/grey until live data arrives from ThingSpeak Channel 1.
  * **IDs 2–20 (Simulated Herd):** 19 virtual cattle simulated by the Python digital-twin process.
* **Cost Impact:** ₹0 additional hardware cost beyond the single physical collar.

---

## ADR-003: Physical Collar Sensor & Hardware Baseline (BOM ₹2,450)

* **Status:** Approved
* **Context:** Hardware constraints established in Review 2 Bill of Materials (BOM).
* **Decision:** Physical collar uses the following binding sensor suite:
  * **Compute:** ESP32-WROOM-32 (Dual-core, Wi-Fi/BLE)
  * **Body Temperature:** MLX90614 non-contact infrared sensor (I2C)
  * **Ambient Environment:** DHT11 temperature and relative humidity sensor
  * **Movement / IMU:** MPU6050 6-axis accelerometer + gyroscope (I2C)
  * **Location:** NEO-6M GPS module (UART)
  * **Battery Voltage:** ADC input on GPIO34 with voltage divider
* **Consequences:** The simulator must synthesize telemetry that exactly mimics the physical ranges and properties of these specific sensors.

---

## ADR-004: ThingSpeak Free-Tier Multi-Channel Allocation & Rate Limits

* **Status:** Approved
* **Context:** MathWorks ThingSpeak free-tier limits: 4 channels max, 15-second minimum write interval per channel, 3,000,000 messages/year ceiling (~8,219 messages/day).
* **Decision:** Channel topology allocation:
  * **Channel 1 (Physical Collar):** Dedicated to Animal ID 1. POST every 30s (steady state) or 15s (breach/alert).
  * **Channel 2 (Simulated Fleet):** Dedicated to multiplexing Animals IDs 2–20. POST 1 animal every 30s in round-robin sequence (full sweep = $19 \times 30\text{s} = 570\text{s} \approx 9.5\text{ min}$).
  * **Channel 3 (Configuration / Geofence):** Read-only for field devices/simulators to fetch geofence vertices and thresholds (reads do not consume message quota).
  * **Channel 4 (Spare):** Reserved for system diagnostics or future expansion.
* **Budget Validation:** $2,880 + 2,880 = 5,760\text{ writes/day} \approx 2,102,400\text{ writes/year}$ (well within the 3.0M ceiling, leaving ~30% headroom for alert bursts).

---

## ADR-005: Stable Telemetry Contract & Reserved Status Field Protocol

* **Status:** Approved
* **Context:** ThingSpeak channels support exactly 8 numeric/alphanumeric fields plus one 255-byte reserved `status` text field. Both physical and virtual cattle must use the exact same schema so downstream dashboards require only one parser.
* **Decision:** Telemetry schema:
  * `field1`: Body Temperature (°C, float)
  * `field2`: Temperature-Humidity Index (THI, float)
  * `field3`: Behaviour Code (0–5, integer)
  * `field4`: GPS Latitude (decimal degrees, float)
  * `field5`: GPS Longitude (decimal degrees, float)
  * `field6`: Composite Risk Score (0–100, integer)
  * `field7`: Geofence Status (`0`=Inside, `1`=10m Warning Zone, `2`=Breach)
  * `field8`: Battery Level (0–100%, integer)
  * `status`: Semicolon-delimited header: `id=XX;evt=YY;src=ZZ` (e.g. `id=07;evt=FEVER;src=RULE` or `id=14;evt=BREACH;src=RULE`).
* **Consequences:** No module or adapter is permitted to alter or expand this 8-field layout.

---

## ADR-006: 6-State Behavioural Classification & Safe Code Mapping

* **Status:** Approved
* **Context:** Animal activity must be classified consistently across simulation, physical IMU classifier, and public ML datasets.
* **Decision:** Standardized integer behaviour mapping:
  * `0`: Resting
  * `1`: Grazing
  * `2`: Ruminating
  * `3`: Walking
  * `4`: Restless
  * `5`: Other / Unknown / Miscellaneous (reserved for low-confidence ML output)
* **Critical Rule:** No ML adapter or preprocessor may map `Miscellaneous` or `Other` to `Restless (4)`. Restless is reserved strictly for high-arousal distress/anomaly states.

---

## ADR-007: Deterministic Product-Rule Risk Scoring Formula

* **Status:** Approved
* **Context:** Both the C++ ESP32 firmware and the Python simulator must generate identical risk scores given the same input signals.
* **Decision:** Compute severity components in $[0, 1]$:
  * Temperature severity: $S_{\text{temp}} = \text{clamp}\left(\frac{\text{body\_temp} - \text{baseline} - 0.5}{1.5}, 0, 1\right)$
  * THI severity: $S_{\text{THI}} = \text{clamp}\left(\frac{\text{THI} - 68}{16}, 0, 1\right)$ where $\text{THI} = (1.8 T + 32) - (0.55 - 0.0055 RH)(1.8 T - 26)$
  * Restless behaviour: $S_{\text{restless}} = 0.35$
  * Geofence warning: $S_{\text{geo\_warn}} = 0.50$
  * Geofence breach: $S_{\text{geo\_breach}} = 0.90$
  * Social isolation: $S_{\text{isol}} = 0.70$
  * Collar tamper: $S_{\text{tamper}} = 0.90$
  * Combined Risk Score:
    $$\text{risk} = \text{round}\left(100 \times \left(1 - \prod_{i} (1 - S_i)\right)\right) \in [0, 100]$$
* **Alert Bands:** `0–39` = Green (Normal), `40–69` = Yellow (Warning), `70–100` = Red (Critical Alert).

---

## ADR-008: Geofence Spatial Geometry & 10-Meter Warning Band

* **Status:** Approved
* **Context:** Simple containment checks often suffer from edge fluttering. Clear boundary definitions are required for safety alerts.
* **Decision:** 
  * The pasture is defined as a simple closed polygon with $\ge 3$ coordinates.
  * Ray-casting algorithm determines point inclusion.
  * A **10-meter inner buffer band** along the polygon perimeter is classified as `Geofence Status = 1` (Warning).
  * Any point outside the polygon is classified as `Geofence Status = 2` (Breach).
  * Coordinates inside the polygon and $>10\text{m}$ from the boundary are `Geofence Status = 0` (Inside).

---

## ADR-009: Activity-Aware Battery Drainage (Strictly No Solar Recharge)

* **Status:** Approved
* **Context:** Real IoT wearables drain battery non-linearly based on transmission frequency and active alerts. Solar harvesting was considered and evaluated.
* **Decision:** 
  * **NO solar recharge:** The collar hardware does not have solar panels. Battery level is strictly monotonically decreasing during a run.
  * **Activity-Aware Discharge Rates:**
    * Baseline discharge (resting/grazing, 30s cadence): $1\times$ base drain.
    * Active alert discharge (breach/warning, 15s bursts, active alarms): $3\times$ base drain.
  * **Dropout Threshold:** When battery reaches $0\%$, the device triggers `collar_dropout` and ceases transmission.

---

## ADR-010: Collar-1 Sniffing & Synchronization (True Digital Twin Coupling)

* **Status:** Approved (Antigravity Enhancement)
* **Context:** Virtual cows should not roam blindly detached from the real world if the physical collar is actively running in the field.
* **Decision:** 
  * The simulator background worker periodically issues a free `GET` to ThingSpeak Channel 1.
  * When valid GPS fixes exist for Collar 1, the simulated herd centroid smoothly anchors to Collar 1, causing virtual cattle to flock around the physical prototype.
  * If Collar 1 is offline or data is stale (>120s), the simulator seamlessly falls back to autonomous pasture drift without resetting state.

---

## ADR-011: Local Web Visualizer HUD Architecture (Zero-Dependency)

* **Status:** Approved (Antigravity Enhancement)
* **Context:** Demonstrations during academic reviews require visual map rendering without complex web stack setup (no heavy Node.js or React build step).
* **Decision:** 
  * Python's built-in `http.server` / `asyncio` binds to `127.0.0.1:8000`.
  * Serves a lightweight single-page HTML5/JS dashboard using local Leaflet.js assets.
  * Displays pasture boundaries, color-coded animal pins (Green/Yellow/Red), offline ID 1 marker, and real-time transmission queue HUD.

---

## ADR-012: High-Speed Dry-Run Mode & Quota Preservation

* **Status:** Approved (Antigravity Enhancement)
* **Context:** Running a 24-hour real-time simulation takes 24 hours and burns API write quotas during development.
* **Decision:** 
  * Provide a `--dry-run` CLI flag.
  * Disables network HTTP writes and time sleeps; advances simulation time in memory.
  * Executes a full 24-hour simulation cycle (86,400 ticks) in $<10\text{ seconds}$, outputting complete normalized CSV/JSON logs.

---

## ADR-013: 190-Pair Ground-Truth Matrix Logging

* **Status:** Approved (Antigravity Enhancement)
* **Context:** Downstream team members developing proximity graphs, social network clustering, and roll-call algorithms require benchmark ground truth to evaluate accuracy.
* **Decision:** 
  * At every complete simulation tick, log all $C(20, 2) = 190$ unordered pairwise distances (in meters) and active anomaly states to `ground_truth_pairs.csv`.

---

## ADR-014: Dual Fault-Injection Engine (Declarative JSON + Live CLI)

* **Status:** Approved (Antigravity Enhancement)
* **Context:** Testing requires automated reproducible scenarios as well as impromptu live demonstration commands.
* **Decision:** Support two complementary injection mechanisms:
  * **Declarative JSON scenarios:** Timed event scripts specifying `fever_onset`, `heat_stress`, `geofence_breach`, `tamper`, `social_isolation`, `collar_dropout`.
  * **Interactive Live CLI:** Terminal prompt listening for live commands (`fever <id>`, `breach <id>`, `tamper <id>`, `clear <id>`, `status`, `pause`, `resume`, `quit`) entering the same priority event queue.
  * **Event Composition Order:** Physiology $\rightarrow$ Movement $\rightarrow$ Social State $\rightarrow$ Collar Faults $\rightarrow$ Risk Calculation $\rightarrow$ Transmission.

---

## ADR-015: Machine Learning Dataset, Evaluation, and Diagnostic Guardrails

* **Status:** Approved
* **Context:** Academic rigor demands preventing data leakage and avoiding misleading clinical claims.
* **Decision:** 
  * Use the public **WASP-lab** cattle dataset (Morales-Vargas et al., 2025) for IMU behavior benchmarks.
  * **Validation Rule:** Use **cow-grouped cross-validation** (never split windows from the same cow or event across train and test sets).
  * **Diagnostic Safety Guardrail:** The system outputs "predicted risk score" or "anomaly indicator". It **never** outputs a clinical veterinary diagnosis or recommends medical treatment.

---

## ADR-016: Scenario JSON Contract Correction (Deliverable #3 Fix)

* **Status:** Approved
* **Context:** Deliverable #3's initial `scenario_runner.py` used a bare JSON array with `sim_second`/`event` fields and no `duration_seconds`, which did not match the scenario contract already specified in the Master PRD ("Scenario JSON contains `schema_version`, `scenario_id`, `seed`, and `events`. Each event contains `animal_id`, `type`, `start_sim_second`, `duration_seconds`, and typed parameters... Unknown types, duplicate event IDs, invalid cattle IDs, and non-positive duration fail before startup. The same event type cannot overlap itself for the same cow."). Without `duration_seconds`, a scripted event had no way to end on its own — `fever_onset` happened to taper off via its own onset/plateau/recovery ramp, but `heat_stress`, `geofence_breach`, `social_isolation`, and `tamper` would stay active for the rest of the run unless manually cleared.
* **Decision:**
  * Scenario JSON top-level shape is `{schema_version, scenario_id, seed, events}`; each event is `{animal_id, type, start_sim_second, duration_seconds, event_id?, params?}` (documented in full in AGENTS.md §4.3).
  * `load_scenario()` validates all four "fail before startup" rules — unknown `type`, duplicate `event_id`, invalid `animal_id` (when the caller supplies the herd's valid ID range), non-positive `duration_seconds` — plus same-type self-overlap across the scripted timeline, before returning.
  * `duration_seconds` is required for scenario-file events and drives auto-expiry (`expire_events()`, called once per tick from `simulator.tick()`). It is `None` for events activated live via CLI/API, which continue running until an explicit `clear` command — the self-overlap rule and auto-expiry govern the static scripted timeline, not interactive re-triggering of the same fault.
* **Consequences:** Any scenario JSON written against the old bare-array format must be rewritten to the wrapped format. `config/scenarios/demo_scenario.json` and `fault_injection.json` (Deliverable #7, not yet created) must be authored against the format documented here and in AGENTS.md §4.3, not the original Deliverable #3 implementation.

---

## ADR-017: Pre-Landing Review Fixes — Deliverables #1-#3

* **Status:** Approved
* **Context:** A structural pre-landing review across Deliverables #1-#3 (foundation, animal models, simulation engine — 4,766 lines total) evaluated the code against the Master PRD and AGENTS.md's own golden rules and surfaced seven issues.
* **Decision:**
  * **Fixed:**
    * `config.py` now rejects `risk.severity.temp_offset_high <= temp_offset_low` and `thi_high <= thi_low` at load time. `risk.py` divides by these differences with no zero-guard, so a misconfigured or swapped pair previously would not fail until the tick loop actually ran, crashing it with `ZeroDivisionError`.
    * Dropout telemetry (both the battery-exhaustion and `collar_dropout`-event paths in `simulator._tick_animal`) now reports `risk_score=100, alert_band="red"` instead of a fabricated `0`/`"green"` — matching the Master PRD's explicit requirement that "the HUD marks the cow stale and critical instead of fabricating a current score." `dropped_out=True` alone was not enough; a healthy-looking score actively contradicted it. Codified in AGENTS.md golden rule 3.
    * `movement.breach_excursion_target()`'s outward search now scales with the pasture polygon's own vertex-to-vertex diameter instead of a fixed 1km cap, which silently failed — returning a "breach target" still inside the polygon — for any configured pasture larger than that.
    * `live_cli.py` accepts `isolation` as an alias for `isolate`, matching the Master PRD's documented live-command verb (the shorter `isolate` still works — additive, not a rename).
    * `simulator.py`'s module docstring corrected to accurately describe how its 10-step tick loop maps onto ADR-014's actual 6-stage composition order (the docstring had drifted to a different, incorrect ordering that didn't match ADR-014's text).
    * `config.py` and `simulator.py`: removed 4 dead imports (`re`, `field`, `math`, `typing.Any`) left over from earlier drafts.
  * **Deferred, with rationale recorded here so it isn't silently re-discovered:**
    * ID 1 (the physical collar) still receives full physiology/behaviour/movement/battery/risk simulation every tick, contradicting "ID 1 is offline/grey until Channel 1 returns a fresh physical-collar record" (ADR-002). Not fixed now because the correct placeholder behavior depends on the not-yet-built Channel-1 sniffing integration (ADR-010) — building it prematurely risks a second rewrite once that integration lands. The scheduler already correctly excludes ID 1 from Channel-2 transmission, so nothing leaks to ThingSpeak today.
    * `src/main.py` (AGENTS.md §6's CLI entry point) does not exist. `create_simulator()`/`run_simulation()` are fully functional as library calls (proven by the dry-run integration tests) but none of AGENTS.md's five documented `--mode` invocations are runnable yet. This is net-new feature work, not a defect in existing code — scoped as its own follow-up deliverable step, not folded into a review pass.
  * **Left alone (not a gap):** 6 imports in `simulator.py` flagged unused by pyflakes (`get_queue_snapshot`, `get_all_active_for_animal`, `step_centroid_anchored`, `validate_body_temp`, `individual_offset_m`, `ActiveEvent`) — these read as intentional placeholders for the not-yet-built REST API, HUD, and Collar-1-sniffing wiring, not dead code, so they were not stripped.
* **Consequences:** Both deferred items are now explicit, tracked gaps rather than undocumented behavior. Whoever builds Channel-1 sniffing must also revisit ID 1's tick-loop treatment. Whoever builds the CLI entry point should treat `main.py` as net-new scope with its own review, not bundle it into an unrelated change. 197 tests pass (was 190 before this review).

---

## ADR-018: Logging, Ground Truth & Replay Architecture (Deliverable #4)

* **Status:** Approved
* **Context:** Deliverables #1-#3 built a fully functional simulation engine (20 cattle, 1-second tick loop, deterministic risk scoring, fault injection), but it wrote nothing to disk. The Master PRD requires per-run structured logging (8 output files), C(20,2)=190 pairwise ground-truth distances per tick, buffered I/O, a replay reader, and normalization for deterministic verification — all without modifying the core simulation loop's logic.
* **Decision:**
  * **Callback-based event lifecycle observability:** Added 3 new callback fields to `Simulator` (`on_event_expired`, `on_event_cleared`, `on_tick_complete`) alongside the existing 3 (`on_telemetry`, `on_transmit`, `on_event_activated`). The logger registers 6 closures via `wire_logger(sim, rl)`, observing the full lifecycle without coupling to the tick loop. This preserves the functional-style separation: the engine produces data, services consume it.
  * **Enriched return types on `expire_events()` and `clear_all_events()`:** Changed from `list[str]` / `int` to `list[tuple[str, int, str]]` (event_id, animal_id, event_type). The logger needs all three fields to write meaningful JSONL records. Minimal test impact (1 assertion changed).
  * **Scenario-sourced events now trigger `on_event_activated`:** Previously only CLI-injected events fired this callback. Fixed by adding the dispatch in `tick()`'s scenario-processing block, so the logger captures all activations regardless of source.
  * **`on_tick_complete` for ground truth batching:** `on_telemetry` fires per-animal, but ground truth requires all 20 positions simultaneously. `on_tick_complete(telemetry_batch, sim_second)` fires once at the end of each tick with the full batch, enabling `itertools.combinations` over sorted positions for the 190-pair Haversine matrix.
  * **Buffered I/O with configurable flush interval:** `BufferedWriter` accumulates lines in memory and flushes to disk every `buffer_size` lines (default 100 from `config/default_config.yaml`). This avoids per-row `write()` syscalls during high-speed dry-runs while keeping memory bounded.
  * **8 output files per run:** Each run creates `logs/run_{timestamp}_{run_id[:8]}/` containing: `manifest.json` (run identity + config hash), `config.snapshot.json` (full config at time of run), `animal_profiles.json` (per-cow baselines), `telemetry.csv` (17-column, 6dp floats), `events.jsonl` (activated/expired/cleared lifecycle), `transmissions.jsonl` (scheduler uplink records), `ground_truth_pairs.csv` (190 pairwise distances + anomaly flags), `summary.json` (aggregate counters).
  * **Normalization contract for determinism verification:** `normalize_telemetry_csv()` strips `run_id`, rounds all floats to 6 decimal places, and sorts by `(sim_second, animal_id)`. `normalize_manifest()` strips `run_id`, `start_time_iso`, and `config_hash`. Two dry-runs with identical seed/config/scenario produce byte-identical normalized output — proven by integration test.
  * **Replay reader (`replay.py`):** `load_replay()` yields typed `ReplayRow` objects sorted by `(sim_second, animal_id)`. Pipe-separated event codes are parsed into `list[str]`. The reader is the foundation for the future `--mode replay` CLI command (deferred to when `main.py` is built).
* **Consequences:** 246 tests pass (197 existing + 38 logger + 11 replay). The logging service is fully wired and functional via `create_run_logger()` + `wire_logger()` + `close_logger()` — these can be called from any entry point. `main.py` remains deferred (ADR-017). The replay reader is ready for the future replay mode but does not depend on it.
