# Architecture Decision Record (ADR) & Project Decision Log
**Intelligent Cattle Fleet Management Platform**
*Last Updated: 2026-08-28 · Status: Active / Approved*

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
- [ADR-019: ThingSpeak Client & Quota Enforcement (Deliverable #5)](#adr-019-thingspeak-client--quota-enforcement-deliverable-5)
- [ADR-020: REST API Server & Web HUD (Deliverable #6)](#adr-020-rest-api-server--web-hud-deliverable-6)
- [ADR-021: CLI Entry Point & MVP Finalization (Deliverable #7)](#adr-021-cli-entry-point--mvp-finalization-deliverable-7)
- [ADR-022: Post-Ship Live Boundary Hardening (Deliverables #4–#7 Follow-up)](#adr-022-post-ship-live-boundary-hardening-deliverables-4-7-follow-up)
- [ADR-023: Physical Collar Parity Firmware (Deliverable #8)](#adr-023-physical-collar-parity-firmware-deliverable-8)
- [ADR-024: Field Hardware Corrections & Indoor Test Fallback (Deliverable #8 Follow-up)](#adr-024-field-hardware-corrections--indoor-test-fallback-deliverable-8-follow-up)

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
  * **Battery Demo (Deliverable #8):** No ADC is fitted to the USB-powered classroom prototype. `field8` is a laptop-controlled manual percentage; GPIO34 is unused (superseded by ADR-023).
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

---

## ADR-019: ThingSpeak Client & Quota Enforcement (Deliverable #5)

* **Status:** Approved
* **Context:** Deliverables #1-4 built a complete simulation engine with logging but no actual HTTP communication with ThingSpeak. The scheduler decides which animal transmits and when, firing `sim.on_transmit(telemetry)`, but nothing consumed that callback to perform network I/O. The Master PRD requires Channel 2 POST writes, Channel 1 GET sniffing for Collar-1 GPS anchoring (ADR-010), quota enforcement against the 3M annual ceiling, exponential backoff on failures, and resilience ("ThingSpeak outage does not stop simulation").
* **Decision:**
  * **Background-threaded HTTP (`thingspeak.py`):** Two daemon threads — a writer thread draining a bounded `Queue(maxsize=100)` for Channel 2 POSTs, and a sniffer thread performing periodic Channel 1 GETs. The tick loop's `on_transmit` callback enqueues writes non-blockingly; the writer thread handles the actual HTTP, retries, and backoff. This ensures the 1-second tick loop never blocks on network I/O.
  * **Stdlib-only HTTP:** Uses `urllib.request` (no `requests` or `httpx`), consistent with the project's zero-dependency principle. Thin `_http_post` and `_http_get` wrappers provide clean mockability for testing.
  * **Fan-out callback composition:** `wire_thingspeak(sim, client)` must be called after `wire_logger()`. It captures the existing `on_transmit` (the logger's callback) via closure and wraps both into a single function — the logger records the transmission first, then the ThingSpeak client enqueues the HTTP write. No changes to the `Simulator` dataclass's single-slot `Optional[Callable]` type signatures were needed.
  * **Quota enforcement:** `QuotaState` tracks annual and daily write counts in memory. `check_quota()` disables writes before the configurable `annual_write_limit` (default 3,000,000) is reached and warns at `quota_warning_pct` (default 90%). Daily counter resets after 86,400 seconds. Not persisted across runs — acceptable for P1 MVP.
  * **Exponential backoff:** Failed POSTs retry up to 4 times with exponential delay (`base_s * 2^failures`, capped at `backoff_max_s`). Backoff sleeps use `Event.wait()` so they're interruptible on shutdown. ThingSpeak rate-limit responses (HTTP 200 with body `"0"`) are treated as retryable failures, matching ThingSpeak's documented behavior.
  * **Mode gating:** `enqueue_write()` returns immediately (no HTTP, no queue) in DRY_RUN and OFFLINE modes. `start()` skips thread creation when mode is not LIVE or when required API keys are missing. DRY_RUN determinism is preserved — the ThingSpeak integration causes no side effects outside LIVE mode.
  * **Collar-1 centroid anchoring (ADR-010 implementation):** The `Simulator` dataclass gains `collar1_anchor: Optional[Coord]` and `collar1_anchor_time: float`. The sniffer thread updates these via a thread-safe callback. `tick()` step 4 now conditionally uses `step_centroid_anchored()` when a fresh fix exists (within `channel_1_stale_threshold_s`), falling back to `step_centroid_autonomous()` otherwise. Only active in LIVE mode.
  * **Config extensions:** `ThingSpeakConfig` gains `channel_1_id`, `annual_write_limit`, and `quota_warning_pct`. Validated at load time (annual limit > 0, warning % in 1–100). Defaults make existing configs forward-compatible.
  * **Transmission result logging:** `log_write_result()` in `logger.py` writes HTTP outcome records (`success`, `rejected`, `retry_exhausted`, `dropped`, `quota_disabled`) to `transmissions.jsonl` with a `"type": "http_result"` discriminator, coexisting with the existing scheduler-level records.
  * **Credential safety:** API keys loaded from environment variables via the existing `load_env_credentials()`. Never logged — log messages use `key={'set' if key else 'missing'}`. Missing write key in LIVE mode logs a warning and skips the writer thread; the simulation continues.
* **Deferred (not addressed in this deliverable):**
  * ID 1's tick-loop behavior remains unchanged (still simulated each tick). The sniffer provides centroid anchoring for the herd, which is the correct ADR-010 behavior, but revisiting ID 1's own simulation is deferred until the sniffing integration is proven in the field (as noted in ADR-017).
  * `src/main.py` CLI entry point remains deferred (ADR-017).
* **Consequences:** 308 tests pass (246 existing + 62 new ThingSpeak tests). The `step_centroid_anchored` import in `simulator.py` (previously flagged as an intentional placeholder in ADR-017) is now in active use. The ThingSpeak client is fully wired via `ThingSpeakClient(cfg, creds, mode)` + `client.start()` + `wire_thingspeak(sim, client)` — callable from any entry point.

---

## ADR-020: REST API Server & Web HUD (Deliverable #6)

* **Status:** Approved
* **Context:** Deliverables #1-5 built a complete simulation engine with logging and ThingSpeak uplink, but no way to visualize or interact with a running simulation from a browser. The Master PRD specifies a local REST API and Leaflet.js web HUD (AGENTS.md §7, HerdSimulator PRD FR-34). `HudConfig` already exists in `config.py` with validated `host`, `port`, and `poll_interval_ms`. The scheduler's `get_queue_snapshot()` and scenario runner's `activate_event()` / `clear_event_by_id()` provide the API surface needed for queue inspection and live event injection.
* **Decision:**
  * **Stdlib-only `ThreadingHTTPServer` in a daemon thread (`api_server.py`):** Consistent with the zero-dependency principle. `HudServer.start()` creates the server on a daemon thread; `HudServer.stop()` calls `shutdown()`. The daemon thread ensures the simulation exits cleanly even if the server hangs.
  * **`HudState` dataclass for thread-safe shared state:** Holds a reference to `sim`, `run_id`, `start_time`, `latest_telemetry: list[AnimalTelemetry]`, `history: dict[int, collections.deque(maxlen=10_000)]`, a `threading.Lock`, and an optional `ThingSpeakClient`. The lock protects mutable collections (history buffer, latest_telemetry snapshot). Atomic reads of `sim` fields rely on the GIL.
  * **Fan-out callback chaining on `on_tick_complete`:** `wire_api_server(sim, hud_state)` must be called after `wire_logger` and `wire_thingspeak`. It captures the existing `on_tick_complete` callback via closure and wraps both — the upstream callback (logger/ThingSpeak) runs first, then the HUD state update appends to history deques and refreshes `latest_telemetry` under the lock. HUD-specific logic is wrapped in `try/except` so a failure does not break the upstream logger's ground-truth callback.
  * **Direct `activate_event()` for synchronous event injection:** `POST /api/events` calls `activate_event()` and `enqueue_priority()` directly rather than going through the `CLICommand` queue. This allows the 201 response to return the assigned `event_id` synchronously. The call is protected by `hud_state.lock` for thread safety on `EventState._next_id` increment.
  * **`_clear_event_with_info` helper for DELETE:** `clear_event_by_id()` returns only `bool`, but the `on_event_cleared` callback needs `(event_id, animal_id, event_type)`. A local helper looks up the event metadata in `event_state.active` before clearing, returning `Optional[tuple[str, int, str]]`.
  * **Per-animal history ring buffer:** `dict[int, collections.deque(maxlen=10_000)]` populated by the `on_tick_complete` callback. `GET /api/history?id=<id>&limit=<n>` reads from this buffer under the lock. The 10,000-entry cap bounds memory at ~2.8 hours of telemetry per animal.
  * **6 REST endpoints (per AGENTS.md §7):** `GET /api/health` (uptime, sim_second, memory, queue depth), `GET /api/state` (full 20-cattle snapshot with pasture polygon, ambient weather, active events), `GET /api/history` (per-animal historical telemetry), `GET /api/queue` (scheduler snapshot with ThingSpeak quota), `POST /api/events` (inject live anomaly, returns 201 with event_id), `DELETE /api/events/<id>` (clear active event, returns 204). All JSON responses include `schema_version`, `run_id`, `sim_second`. Errors use the standard format: `{"code": "...", "message": "...", "details": {...}}`.
  * **Static file serving with directory traversal protection:** Requests not matching `/api/*` are served from `src/herd_simulator/web/`. `/` maps to `index.html`. Path traversal is blocked by resolving the requested path and verifying it falls within the `web/` directory via `Path.resolve()`.
  * **Leaflet 1.9.4 vendor bundle:** `leaflet.js`, `leaflet.css`, and marker images committed as static assets in `web/leaflet/`. No CDN dependency — consistent with the zero-dependency principle. OSM tiles are the only external network request (degrades gracefully offline to a grey map).
  * **Dark-themed single-page HUD (`index.html`, `app.js`, `style.css`):** Leaflet map with circle markers colored by alert band (green/yellow/red, grey for dropout). Sidebar with environment panel, event list with clear buttons, event injection form, queue status, and a 20-row animal table. Client-side polling at `poll_interval_ms` (default 2000ms). Toast notifications for event injection/clearing feedback. Connection status indicator dot.
* **Deferred (not addressed in this deliverable):**
  * ID 1 behavior (physical collar parity) remains simulated — deferred per ADR-017.
  * `src/main.py` CLI entry point remains deferred (ADR-017).
* **Consequences:** 370 tests pass (308 existing + 62 new API server tests across 14 test classes). The API server is fully wired via `create_hud_state(sim, run_id)` + `HudServer(hud_state, cfg).start()` + `wire_api_server(sim, hud_state)` — callable from any entry point. Tests use port 0 (OS-assigned) to eliminate CI conflicts.
 
---

## ADR-021: CLI Entry Point & MVP Finalization (Deliverable #7)

* **Status:** Approved
* **Context:** Deliverables #1-#6 implemented all core simulator subsystems (math utilities, animal state models, 1-second simulation engine, structured logging, ThingSpeak client, and REST API/HUD). However, `src/main.py` was deferred in ADR-017, canonical scenario files (`demo_scenario.json` and `fault_injection.json`) were missing, and end-to-end verification across all 16 MVP Acceptance Criteria was needed to finalize Phase 1.
* **Decision:**
  * **Unified CLI Entry Point (`src/main.py`):** Built with standard library `argparse`, supporting all 4 execution modes (`dry-run`, `offline`, `live`, `replay`) and full flags (`--config`, `--scenario`, `--duration-hours`, `--hud`, `--port`, `--seed`, `--log-dir`, `--playback-speed`, `--verbose`).
  * **Strict Lifecycle & Boot Wiring Order:**
    1. Parse args & configure logging.
    2. Auto-load `.env` variables via stdlib `env_loader.py` without third-party dependencies.
    3. Load & validate YAML configuration with CLI overrides applied via `dataclasses.replace`.
    4. Load & validate scenario JSON against herd ID bounds (`1..n_total`).
    5. Initialize deterministic animal profiles and `Simulator` instance.
    6. Initialize and wire `RunLogger` (manifest, snapshot, profiles, telemetry, events, transmissions, ground truth).
    7. Initialize `ThingSpeakClient` (gated to `LIVE` mode) and wire transmission & result callbacks.
    8. Initialize `HudServer` and wire `on_tick_complete` state updates (if `--hud` enabled).
    9. Spawn interactive CLI daemon thread if in interactive terminal (`sys.stdin.isatty()`).
    10. Register `SIGINT` and `SIGTERM` signal handlers for graceful shutdown.
    11. Execute `run_simulation(sim, duration_seconds)`.
    12. Execute `finally` cleanup: write `summary.json`, flush/close buffered writers, stop ThingSpeak and HUD threads.
  * **Interactive HUD Replay (`run_replay`):** Replay mode parses prior `telemetry.csv` files via `load_replay()`, groups rows by `sim_second`, and streams them into `HudState` under thread lock with configurable playback speed, allowing visual re-examination of past runs in the Leaflet HUD without re-running simulation physics.
  * **Canonical Scenario Suite:**
    - `config/scenarios/demo_scenario.json`: 20-minute classroom walkthrough demonstrating all 5 visible anomaly faults (`fever_onset`, `geofence_breach`, `social_isolation`, `tamper`, `heat_stress`, `collar_dropout`) with spaced activation and overlap composition.
    - `config/scenarios/fault_injection.json`: Full 10-event test matrix covering all 6 fault types, sequential re-activation, boundary edge cattle IDs (2 and 20), and multi-sweep execution.
  * **Comprehensive E2E Acceptance Verification (`test_e2e_acceptance.py`):** 16 automated test cases explicitly mapped 1:1 against the Master PRD's MVP Acceptance Criteria (AC-1 through AC-16), proving byte-identical dry-run determinism, sweep coverage, 15s floor enforcement, quota safety, fault composition, and zero credential leakage.
* **Consequences:** Phase 1 (MVP) is 100% complete and fully verified. 402 tests pass across 12 test suites.

---

## ADR-022: Post-Ship Live Boundary Hardening (Deliverables #4–#7 Follow-up)

* **Status:** Approved
* **Date:** 2026-08-27
* **Supersedes:** The deferred ID 1 behavior in ADR-017, ADR-019, and ADR-020; the GIL-only state-safety assumption in ADR-020; and the cleanup ordering described in ADR-021.
* **Context:** The Deliverables #4–#7 review found several integration-boundary defects despite the MVP subsystems being individually complete: the physical collar identity could still receive synthetic simulation, REST and sniffer threads could race the tick loop, POST retries could bypass the ThingSpeak rate floor, shutdown could close loggers before background writers finished, and live API events could accidentally use scripted-event expiry semantics.
* **Decision:**
  * **Physical Collar-1 boundary:** ID 1 uses a dedicated physical-collar tick path. It never receives synthetic physiology, behavior, movement, battery, or risk updates. Before a complete fresh Channel 1 row arrives, local state exposes an explicit stale/critical placeholder; a GPS-only fix is used only for herd anchoring. A complete fresh row supplies ID 1's actual telemetry and clears the stale state. The local `stale` flag is presentation metadata and is not added to the immutable ThingSpeak eight-field contract.
  * **Shared-state synchronization:** `Simulator.state_lock` protects the simulation tick, REST reads and event mutations, and Channel 1 callback updates. HUD history/latest snapshots retain a separate HUD lock, with callbacks arranged to avoid reverse lock acquisition.
  * **Transport pacing and shutdown:** The ThingSpeak writer applies a global wall-clock gate of at least 15 seconds before every POST attempt, including retries. Shutdown signals workers first, joins the writer/sniffer threads, drains pending work, and only then closes the logger so final HTTP outcomes are recorded.
  * **Dropout scheduling:** A dropout candidate is skipped before consuming a scheduler slot; the skip is recorded in `transmissions.jsonl` as a `type: "skipped"` record and does not increment transmission totals. The ThingSpeak enqueue boundary also defensively refuses dropout telemetry.
  * **Live event semantics and wiring:** REST-injected events reject `duration_seconds` with `DURATION_NOT_ALLOWED` and remain active until DELETE/clear. The main callback wiring uses the correct event argument order, replay loads the saved configuration snapshot, and the HUD renders stale physical data grey.
* **Consequences:** The live integration now preserves physical-vs-simulated identity, deterministic state access, rate-limit compliance, and lossless shutdown logging across Deliverables #4–#7. The complete suite passes with **411 tests** after this hardening, including regression coverage for each boundary above. Commit `6095f04` contains the implementation.

---

## ADR-023: Physical Collar Parity Firmware (Deliverable #8)

* **Status:** Implemented / Approved
* **Date:** 2026-08-27
* **Supersedes:** ADR-003's GPIO34 battery-ADC requirement for the USB-powered classroom prototype only.
* **Context:** Phase 1 has a hardened Channel 1 sniffer and HUD boundary, but no physical publisher. The project needs one ESP32 Collar-1 prototype that emits the same immutable eight fields and `status` contract without adding a battery voltage divider or creating a dependency from P1 onto the firmware.
* **Decision:**
  * **Platform and hardware:** Deliverable 8 is a PlatformIO/Arduino project targeting `esp32dev` (ESP32 DevKit V1 / WROOM-32). It integrates MLX90614 on I²C (`SDA=GPIO21`, `SCL=GPIO22`), DHT11 (`DATA=GPIO4`), MPU6050 on the shared I²C bus, and NEO-6M GPS (`TX→GPIO16`, optional `RX←GPIO17`). The pin map is public configuration in `src/collar_gateway/firmware/include/device_config.h` and is documented in `src/collar_gateway/ESP32_WIRING_GUIDE.md`.
  * **Telemetry completeness:** MLX90614 body temperature, DHT11 ambient temperature/humidity, and a fresh GPS fix are required before a complete Channel 1 row is posted. The MPU6050 is sampled locally at 10 Hz in a five-second window; the temporary classifier emits only Resting (`0`), Walking (`3`), or Other/Unknown (`5`). It must never map uncertain motion to Restless (`4`).
  * **Parity and status:** The firmware computes THI, geofence status, alert band, and product-rule predicted risk from `contracts/telemetry_parity_v1.json`, which is also checked from Python and generated into the firmware build. It posts `id=01;...;src=SENSOR`; measured temperature/THI/breach conditions can add the machine event codes `FEVER`, `HEAT`, and `BREACH`. Narrative UI output remains “Predicted Risk Score,” never a diagnosis.
  * **Battery exception:** No ADC or GPIO34 wiring is used. `field8` starts at `100` and is set manually from the laptop over USB serial with `battery <0..100>`. It is demonstration metadata, not a voltage reading. `battery 0` creates a logical dropout (`risk_score=100`, `evt=DROPOUT` in local status), clears pending writes, and suppresses Channel 1 transmission until restored to a non-zero value.
  * **Transport and secrets:** Channel 1 posts at 30 seconds normally and 15 seconds when critical/breached; the background worker applies the 15-second wall-clock floor before every retry and keeps only the latest valid telemetry. `WIFI_SSID`, `WIFI_PASSWORD`, and `THINGSPEAK_CHANNEL_1_WRITE_API_KEY` are loaded from an untracked root `.env` or explicit build environment variables and never logged. `requirements.txt` supplies PyYAML, pytest, and PlatformIO for a reproducible laptop setup.
* **Deferred:** Deliverable 9 remains responsible for raw 10 Hz IMU streaming to the laptop, packet-gap logging, and any trained behaviour model. The classroom firmware currently accepts the ESP32 TLS certificate chain without pinning; pin a current ThingSpeak CA certificate before any production deployment.
* **Consequences:** A physical Collar 1 can now replace the simulator's stale placeholder once a complete Channel 1 row is received, while P1 remains runnable independently. The firmware compiles for `esp32dev`, native C++ parity tests cover THI/risk/geofence/alert vectors, and the Python Channel 1 regression suite accepts `src=SENSOR` rows with the manual `field8` value.

---

## ADR-024: Field Hardware Corrections & Indoor Test Fallback (Deliverable #8 Follow-up)

* **Status:** Approved
* **Date:** 2026-08-28
* **Supersedes:** ADR-023's `DATA=GPIO4` DHT11 pin assignment, its motion-classification thresholds and mid-range fallback code, and its assumption of a genuine MPU-6050 chip. Does **not** supersede AGENTS.md golden rule 3's anti-fabrication requirement — the indoor test fallback below is a narrowly-scoped, distinctly-labeled exception to it, not a repeal.
* **Context:** Bringing up Deliverable 8 on the actual purchased hardware surfaced three issues ADR-023 did not anticipate: (1) the supplied GY-521 breakout is an MPU-6500 clone, not a genuine MPU-6050 — it answers `WHO_AM_I` with `0x70` instead of `0x68`, and its factory-uncalibrated accelerometer offsets exceed 0.20 g at rest, well past the original resting/walking thresholds; (2) `GPIO4` proved dead on this specific ESP32 dev board; (3) requiring a live GPS fix and a live MLX90614 reading before any Channel 1 row publishes made it impossible to exercise the ThingSpeak pipeline indoors during development, before a cow or open sky was available.
* **Decision:**
  * **MPU-6500 clone compatibility:** `initialise_hardware()` now reads the `WHO_AM_I` register (0x75) directly over I²C at boot and logs the raw value, so a clone chip is diagnosed from the serial monitor instead of silently misbehaving. `mpu6050.begin()` also retries at the alternate address `0x69` if `0x68` fails. A board that reports `0x70` additionally requires manually patching the developer's local PlatformIO-managed `Adafruit_MPU6050.h` to accept that device ID — a local library-cache edit documented in `ESP32_WIRING_GUIDE.md`, not a change this repository can carry, since the library isn't vendored.
  * **Motion thresholds recalibrated:** `kRestingMotionThresholdG` raised from `0.05` to `0.25` and `kWalkingMotionThresholdG` from `0.15` to `0.50` in `device_config.h`, to absorb the clone sensor's factory offset instead of misreading a stationary cow as constantly walking.
  * **Mid-range behaviour fallback changed from Unknown to Grazing:** `classify_behaviour()`'s mid-range case (motion between the resting and walking thresholds, once a full 5-second window exists) now returns Grazing (`1`) instead of Other/Unknown (`5`), so ordinary in-range motion isn't spuriously flagged low-confidence. Unknown (`5`) is still returned whenever a full motion window hasn't been collected yet. This does not touch AGENTS.md golden rule 6 — mapping to Restless (`4`) remains forbidden in both cases.
  * **DHT11 migrated to GPIO15:** `GPIO4` is dead on the reference board; `kDhtPin` is now `15`. `ESP32_WIRING_GUIDE.md` and `README.md` are updated to match; `GPIO4` is no longer part of the pin map.
  * **Indoor test fallback for MLX90614 and GPS — narrowly-scoped exception to golden rule 3:** `has_complete_sensor_record()` substitutes a fixed `38.5°C` body temperature when the MLX90614 reading is out of physiological range, and a fixed pasture coordinate (`12.9716, 79.1589`) when there is no fresh GPS fix, so Channel 1 can be exercised end-to-end without a live cow or open sky during development. This is needed and is not being deferred. However, every row built from a substituted value sets `TelemetryFrame.spoofed = true`, which `build_status()` surfaces as `src=SPOOF` in place of `src=SENSOR` — a fallback row is never posted with a status string indistinguishable from a genuine one. The serial `status` output and the new continuous print (below) likewise tag a substituted reading `[INDOOR TEST FALLBACK]`. This exception applies only to the physical-collar firmware's own local fallback; it does not touch the P1 simulator, and it does not weaken the dropout/staleness rule from ADR-017 or the ID-1-never-simulated boundary from ADR-022 — both still hold for a `src=SPOOF` row exactly as for a missing one.
  * **Continuous serial diagnostics:** the serial monitor now prints a full sensor-reading block (raw body/ambient temperature, humidity, motion deviation, GPS lock state, battery, with `[INDOOR TEST FALLBACK]` markers where relevant) every 5 seconds, independent of the ThingSpeak publish cadence, in addition to the existing `status` command's compact validity line.
* **Consequences:** Deliverable 8 now runs on the actual supplied hardware (MPU-6500 clone, dead GPIO4) and can be exercised indoors during development. `src=SPOOF` is a new, permanent value the Channel 1 `status` field may carry. Nothing on the Python side parses or validates `src` today (the sniffer only reads GPS fields for herd anchoring), so this required no simulator changes — but any future consumer of Channel 1's `status` field (dashboards, Deliverable 9+ tooling) must treat `SPOOF` as non-genuine telemetry, not a sensor fault, and must not average or train on it alongside genuine `SENSOR` rows.
