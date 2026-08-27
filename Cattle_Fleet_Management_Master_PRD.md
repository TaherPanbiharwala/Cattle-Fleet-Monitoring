# Intelligent Cattle Fleet Management Platform

## Context

This greenfield student project demonstrates fleet-scale cattle monitoring when only one physical collar can eventually be built. The MVP is a laptop-run digital twin containing 19 simulated cattle plus the reserved identity of one physical collar. Later phases add the physical collar, public-dataset behaviour and walking-pattern ML, personalized per-cow deviation, and fever and lameness prediction research without changing the telemetry contract.

## Current State

Verified on 2026-08-20:

| Component | Current state | Evidence |
|---|---|---|
| Simulator requirements | Draft PRD exists; no implementation | `Cattle_Fleet_Management_HerdSimulator_PRD.md` |
| Physical collar | Not built | User-confirmed |
| Simulator software | Not built | Workspace contains no source code |
| ThingSpeak channels | Not configured | User-confirmed |
| ML pipeline | Not built | Workspace contains no model code |
| Project data | None collected | User-confirmed |
| Public behaviour data | Available externally | WASP-lab dataset |
| Fever labels | Not available | Not present in selected dataset |
| Lameness labels | Not available | Not present in selected dataset |

The existing PRD fixes 20 cattle identities (ID 1 physical, IDs 2-20 simulated), Python, three ThingSpeak channels, 30-second normal and 15-second alert cadence, local logging, dry-run, fault injection, map HUD, ground truth, the MLX90614, DHT11, MPU6050, NEO-6M, and ADC GPIO34 sources, and no solar recharge.

ThingSpeak's free option permits four channels, three million messages per year, up to eight fields per message, and one channel update every 15 seconds.

## Product Goals

1. Demonstrate a believable 20-animal fleet using one future physical collar and 19 virtual animals.
2. Produce deterministic, replayable telemetry for fleet analytics and demonstrations.
3. Expose fever, isolation, breach, tamper, heat stress, and dropout faults.
4. Preserve one telemetry contract from rule-based simulation through ML integration.
5. Train behaviour recognition using public IMU datasets before original cow data exists.
6. Establish an individual healthy baseline when longitudinal cow data becomes available.
7. Produce model-predicted fever and lameness risk without presenting it as a veterinary diagnosis.
8. Support lightweight ESP32 inference only after laptop models meet quality and resource gates.

## Users

Primary users are the student development team and faculty reviewers. Researchers, farmers, and veterinarians are future secondary users. The MVP does not serve production farms or make treatment decisions.

## Delivery Phases

| Phase | Deliverable | Binding status |
|---|---|---|
| P1 | Autonomous herd digital-twin simulator | MVP; build first |
| P2 | One physical collar with simulator parity | Next |
| P3 | Public-dataset behaviour and walking analysis | Later |
| P4 | Personalized per-cow deviation model | Later |
| P5 | Fever and lameness risk research | Later, label-gated |

P2-P5 do not block P1 acceptance.

## Stable Telemetry Contract

Behaviour codes are: 0 resting, 1 grazing, 2 ruminating, 3 walking, 4 restless, and 5 other/unknown. No adapter may map miscellaneous behaviour to restless.

| Field | Meaning | Type |
|---|---|---|
| `field1` | Body temperature | Celsius float |
| `field2` | THI | Float |
| `field3` | Behaviour code | Integer 0-5 |
| `field4` | Latitude | Decimal degrees |
| `field5` | Longitude | Decimal degrees |
| `field6` | Risk score | Integer 0-100 |
| `field7` | Geofence status | 0 inside, 1 warning, 2 breach |
| `field8` | Battery | Percentage 0-100 |
| `status` | Identity, event, and score source | Semicolon-delimited text |

Status examples are `id=07;evt=NONE;src=RULE`, `id=07;evt=FEVER;src=RULE`, and `id=07;evt=DEVIATION;src=ML`. Scores 0-39 are green, 40-69 yellow, and 70-100 red. P1 uses deterministic rules; P4 and later use calibrated personalized prediction scores.

## P1: Digital-Twin Simulator

### Population and time

1. Default herd size is 20: one reserved physical identity and 19 simulated animals.
2. Simulated IDs are stable integers 2-20.
3. ID 1 is offline/grey until Channel 1 returns a fresh physical-collar record.
4. Internal state updates and local telemetry snapshots occur every simulated second.
5. ThingSpeak writes occur every 30 wall-clock seconds normally and no faster than every 15 seconds.
6. Dry-run mode removes sleeping and network requests.
7. One seed controls all stochastic operations.
8. The same seed, configuration, and scenario reproduce equivalent normalized telemetry.

### Configuration and reproducibility contract

Configuration uses versioned YAML validated before startup. The root keys are `schema_version`, `simulation`, `herd`, `geofence`, `behaviour`, `physiology`, `battery`, `scenarios`, `telemetry`, `thingspeak`, `server`, and `logging`. Required MVP defaults are schema version 1, herd size 20, physical ID 1, simulated IDs 2-20, one-second simulated ticks, 30-second normal writes, 15-second alert/minimum writes, `127.0.0.1` server binding, and deterministic seed 42 unless overridden. Unknown keys and invalid bounds fail validation; secrets are supplied only through environment variables.

Normalized replay removes wall-clock timestamps and network-response metadata, sorts records by simulated second and animal ID, and rounds floating-point telemetry to six decimal places. Given the same configuration, seed, and scenario, the normalized manifests, profiles, telemetry, events, and ground-truth files must be byte-identical.

### Animal profiles and behaviour

Each cow has a persisted run profile containing identity, temperature baseline, natural variability, behaviour tendencies, preferred centroid offset, walking-speed range, cohesion strength, starting battery, and fault modifiers. Profiles are generated once and written to `animal_profiles.json`.

The state machine supports resting, grazing, ruminating, walking, and restless. Transitions depend on time of day; active fever, heat, and isolation raise restless probability. Invalid transitions are rejected. Code 5 is reserved for uncertain model output and is not ordinarily simulated.

### Movement

1. The geofence is a simple closed latitude/longitude polygon with at least three distinct vertices. The ten-metre band immediately inside its boundary is the warning zone; any point outside the polygon is a breach. The centroid performs a bounded random walk outside the warning band during normal operation.
2. Each cow follows with behaviour-dependent offset and speed.
3. Resting cattle remain stationary except for GPS noise.
4. Walking cattle move faster than grazing cattle.
5. Isolation progressively increases centroid distance.
6. Breach creates a continuous path through warning and breach zones.
7. Teleportation is prohibited except in explicit tests.
8. Fresh physical-collar GPS may smoothly anchor the centroid.
9. Stale Channel 1 data returns to autonomous movement without resetting state.

### Physiology

Body temperature combines personal baseline, diurnal variation, bounded noise, and fever ramp. Shared ambient temperature and humidity follow a configurable diurnal curve. THI is `(1.8*T+32)-(0.55-0.0055*RH)*(1.8*T-26)`. Fever injection has onset, plateau, and recovery. Heat injection changes ambient conditions rather than THI directly. Values outside configured physical bounds are rejected.

### Deterministic P1 risk score

Severity components in `[0,1]` are:

- Temperature: `clamp((body_temp - baseline - 0.5) / 1.5)`
- THI: `clamp((THI - 68) / 16)`
- Restless: `0.35`
- Geofence warning: `0.50`
- Geofence breach: `0.90`
- Isolation: `0.70`
- Tamper: `0.90`

`risk = round(100 * (1 - product(1 - severity_i)))`. Dropout suppresses transmission; the HUD marks the cow stale and critical instead of fabricating a current score.

### Battery

Battery starts from configuration, uses base drain normally and three times base drain in alert/breach operation, never increases, and causes dropout at zero. Dry-run and real-time modes produce the same trajectory.

### Scenario contract

Scenario JSON contains `schema_version`, `scenario_id`, `seed`, and `events`. Each event contains `animal_id`, `type`, `start_sim_second`, `duration_seconds`, and typed parameters. Supported types are `fever_onset`, `heat_stress`, `geofence_breach`, `tamper`, `social_isolation`, and `collar_dropout`. Unknown types, duplicate event IDs, invalid cattle IDs, and non-positive duration fail before startup. The same event type cannot overlap itself for the same cow. Different event types may overlap and compose in the fixed order physiology, movement, social state, collar faults, risk calculation, and transmission. Dropout suppresses transmissions while internal state and other active events continue. This composition rule is the complete overlap policy.

Live commands support fever, heat, breach, tamper, isolation, dropout, clear, status, pause, resume, and quit. They enter the same event engine as scripted events.

### Scheduler

Channel 2 posts one cow per write in normal ID order. New events receive the next eligible slot without permanently starving ordinary cattle. Failed writes use bounded exponential backoff and never block the queue. All successful, rejected, retried, and skipped writes are logged. One scheduler exclusively owns Channel 2 writes.

### Local API and HUD

The server binds to `127.0.0.1` by default and exposes `GET /api/health`, `GET /api/state`, `GET /api/history?id=<id>&limit=<n>`, `GET /api/queue`, `POST /api/events`, and `DELETE /api/events/<event_id>`. JSON uses snake-case keys and includes `schema_version`, `run_id`, and `sim_second` wherever state is returned. History defaults to 100 records and accepts 1-10,000. Event creation accepts the scenario-event schema without `start_sim_second` when immediate, returns `201` with the assigned event ID, and rejects invalid events with `422`. Successful deletion returns `204`; a missing event returns `404`. All errors use `{\"code\": \"...\", \"message\": \"...\", \"details\": {...}}`. The HUD shows the polygon, animal markers, offline ID 1, behaviour, temperature, THI, risk and source, geofence, battery, events, record age, queue, and ThingSpeak message count. Leaflet assets are bundled locally.

### Logs

Each run creates `manifest.json`, `config.snapshot.json`, `animal_profiles.json`, `telemetry.csv`, `events.jsonl`, `transmissions.jsonl`, `ground_truth_pairs.csv`, and `summary.json`. Telemetry includes schema/run/time IDs, simulated second, animal identity, physical/simulated flag, body and ambient temperatures, humidity, THI, behaviour, coordinates, risk and source, geofence, battery, and active events. Ground truth uses one row per unordered pair and complete tick, giving 190 pairs for 20 identities.

Execution modes are normal, offline, dry-run, replay, and compressed demo.

## P2: Physical Collar

The binding sensor set is ESP32, MLX90614, DHT11, MPU6050, NEO-6M, and ADC GPIO34. Collar 1 emits the same eight fields and behaviour enum. In data-collection mode, MPU6050 accelerometer and gyroscope data stream at 10 Hz to the laptop over persistent Wi-Fi. Each frame contains schema version, collar ID, cow ID, UTC timestamp, monotonic sequence, sample rate, three acceleration values, and three gyroscope values. Disconnected samples may be dropped; gaps must be logged, and affected windows excluded. ThingSpeak receives only compact telemetry. Secrets stay outside source control.

Training and initial inference run on the laptop. ESP32 deployment is allowed only after accuracy, memory, latency, and power gates pass.

## P3: Behaviour and Walking ML

The WASP-lab adapter parses `<event_id>_<behavior>_<cow_id>_<YYYYMMDD>_<HHMMSS>.csv`. Walking maps to 3, Grazing to 1, Resting to 0, and Miscellaneous to 5. The common input is body-frame acceleration and gyroscope only, allowing later transfer to MPU6050.

Data is validated at 10 Hz, segmented into five-second windows with 50% overlap, and never split across train/test by overlapping event or cow. Acceleration and gyroscope magnitudes are calculated. Fourteen statistical and spectral features across eight signals produce 112 features. Invalid timestamps, non-finite values, gaps, and windows below 45 valid samples are rejected.

Required comparisons are multinomial logistic regression, Random Forest, RBF SVM, gradient-boosted trees, and an experimental small 1D CNN. Primary model selection uses cow-grouped macro F1. Random-window paper reproduction is secondary and labelled non-primary.

Release gates are grouped macro F1 at least 0.85, walking and miscellaneous recall at least 0.75, full per-class metrics, confusion matrix, latency, dataset hash, feature manifest, artifact, and model card.

Walking outputs are bout duration and frequency, dominant step frequency, acceleration and gyroscope RMS, variance, jerk, periodicity, GPS speed, and personal deviation. Neck IMU data is not presented as direct left/right hoof asymmetry.

## P4: Personalized Per-Cow Deviation

Public clips train the general classifier; longitudinal data is required for personal baselines. The simulator uses its first complete simulated day. A physical cow requires seven healthy days and 1,000 valid windows. Before that, status is `BASELINE_LEARNING` and no health prediction is issued.

Baselines are conditioned by hour, use median and median absolute deviation, exclude known anomaly periods, do not adapt during alerts, and quarantine new data before admission. Unknown cows may use a herd baseline but are marked unpersonalized.

For feature `j`, `z_j = (x_j - median_j) / max(1.4826*MAD_j, epsilon_j)`. `D` is the median of the three largest absolute z-scores. `risk = round(100 * (1 - exp(-max(D-1,0)/3)))`.

P4 must detect at least 90% of simulator-injected fever, isolation, and walking deviations, produce no more than one false warning per simulated cow per day, reproduce artifacts with the same data and seed, and expose the top three contributing deviations.

## P5: Fever and Lameness Research

Fever prediction uses temperature history, personal baseline, ambient conditions, THI, behaviour, and quality flags. Synthetic fever tests integration only. Supervised release requires a compatible labelled dataset. Output is “predicted fever risk,” not diagnosis. Targets are sensitivity at least 0.85, specificity at least 0.75, and no more than one false critical alert per cow per day.

The selected public data has behaviours associated with lameness but no lame-cow labels. Behaviour classification cannot be reported as lameness classification. Future lameness features include behaviour durations, walking cadence, gait periodicity, speed, transitions, and deviation history. Supervised training requires cow-level veterinary or established locomotion-score labels. Candidate models are regularized logistic regression, gradient-boosted trees, and a temporal convolutional model when enough longitudinal data exists. Select the simplest model meeting cow-grouped AUROC at least 0.80, sensitivity at least 0.80, specificity at least 0.70, calibrated probabilities, and model-card requirements. No model recommends treatment.

## Non-Functional Requirements

The MVP uses no paid service; runs on Python 3.11 on Windows, macOS, and Linux; is seeded and reproducible; sustains one-second updates for 20 cattle; runs a simulated day in under ten seconds in dry-run; continues through network failure; keeps secrets out of outputs; exposes health, queue, retries, gaps, and quota; preserves the field contract; uses cow-grouped ML evaluation; explains score sources; and labels health predictions experimental.

## MVP Acceptance Criteria

1. A seeded 24-hour dry run creates 19 profiles and complete telemetry.
2. The same run reproduces equivalent normalized output.
3. At least 99% of sweeps contain all simulated IDs.
4. No Channel 2 writes occur less than 15 seconds apart.
5. Combined steady-state Channel 1 and 2 use remains below three million annual messages.
6. All six events validate, execute, clear, and appear in logs.
7. Priority events reach the next eligible slot without starvation.
8. Geofence, THI, and risk golden vectors pass.
9. Dropout suppresses transmission and produces stale HUD state.
10. ThingSpeak outage does not stop simulation.
11. Replay reproduces telemetry order.
12. HUD shows 19 simulated cattle and one offline/live physical identity.
13. Ground truth contains 190 unordered pairs per complete 20-animal tick.
14. A classroom scenario demonstrates fever, breach, isolation, tamper, and dropout within 20 minutes.
15. No credentials appear in output or errors.
16. Automated tests pass on supported systems.

## Testing Plan

| Layer | Coverage | Minimum count |
|---|---|---:|
| Unit | THI, Haversine, polygon, risk, battery, transitions, validation | 30 |
| Property | Coordinate, score, battery, and schedule bounds | 8 |
| Integration | Simulator, scenarios, scheduler, logs, replay, APIs | 15 |
| Network | Rate limit, retry, timeout, stale/invalid responses | 8 |
| End-to-end | Dry run, offline demo, faults, replay, HUD | 6 |
| ML adapter | Filename, columns, windows, gaps, labels | 12 |
| ML evaluation | Groups, leakage, metrics, reproducibility | 8 |
| Firmware contract | Golden vectors and telemetry parity | 10 |

## Failure and Rollback

Invalid configuration and polygon fail before startup. ThingSpeak failures continue locally. Stale Channel 1 returns autonomous movement. Wi-Fi gaps are logged and excluded. Malformed datasets are quarantined. Unknown labels require explicit mapping. Missing baselines report learning state. Missing or incompatible model artifacts fall back to rule scoring. Battery zero causes dropout. HUD failure does not stop simulation. Quota warning disables writes before the configured ceiling.

Every later phase is configuration-gated. P1 rules remain available after ML. Models are versioned and activated through an explicit pointer. Failed ML rolls back to the prior artifact or rule score. ThingSpeak can be disabled independently. Output schemas are versioned and original runs immutable.

## Child Issues and Sequence

| # | Deliverable | Priority | Human effort | Dependency |
|---:|---|---|---:|---|
| 1 | Python foundation, schemas, configuration, utility math | Critical | 1-2 days | None |
| 2 | Animal, behaviour, movement, physiology, battery | Critical | 2-3 days | #1 |
| 3 | Scenario engine, CLI, risk, scheduler | Critical | 2 days | #1-2 |
| 4 | Logging, ground truth, replay, dry-run | Critical | 2 days | #2-3 |
| 5 | ThingSpeak and quota enforcement | High | 1-2 days | #3-4 |
| 6 | HUD and live faults | High | 2 days | #3-4 |
| 7 | Demo scenarios and MVP verification | Critical | 1-2 days | #1-6 |
| 8 | Physical collar parity | Later | 3-5 days | MVP |
| 9 | Wi-Fi IMU acquisition and gap logging | Later | 2-3 days | #8 |
| 10 | Dataset adapter and behaviour benchmark | Later | 3-5 days | #9 contract |
| 11 | Personalized deviation model | Later | 3-5 days | #10 |
| 12 | Fever and lameness research | Later | Dataset-dependent | #10-11 |

## Planned Repository Structure

`Cattle_Fleet_Management_Master_PRD.md`, `pyproject.toml`, versioned config and scenario folders, `src/herd_simulator`, `src/collar_gateway`, `src/dataset_adapters`, `src/ml`, `src/web`, and unit, integration, end-to-end, and ML tests. The current simulator PRD remains as source history.

## Out of MVP Scope

Physical collar construction, ML training/deployment, lameness and fever classification, clinical validation, diagnosis/treatment advice, mobile app, production farm deployment, multi-farm tenancy, paid cloud, solar charging, and raw IMU streaming through ThingSpeak.

## Definition of Done

P1 is implementable without new interface decisions; every P1 acceptance criterion is testable; later phases cannot alter the ThingSpeak contract; the public data has a leakage-safe adapter; personal deviation is separate from general behaviour; health prediction remains label-gated; every phase has entry, exit, and rollback gates; and reviewers can distinguish built MVP features from the research roadmap.


## Research and Platform References

1. Morales-Vargas, D., Guarda-Vera, M., Iglesias-Quilodrán, D., Cancino-Baier, D., and Muñoz-Poblete, C. (2025). “A dataset for detecting walking, grazing, and resting behaviors in free-grazing cattle using IoT collar IMU signals.” *Frontiers in Veterinary Science*, 12:1630083. https://doi.org/10.3389/fvets.2025.1630083
2. WASP-lab public dataset repository: https://github.com/WASP-lab/db-cow-walking
3. ThingSpeak licensing and message limits: https://thingspeak.mathworks.com/pages/license_faq
4. ThingSpeak channel field documentation: https://www.mathworks.com/help/thingspeak/channel-control.html

