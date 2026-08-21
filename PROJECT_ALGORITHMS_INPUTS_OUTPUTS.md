# Intelligent Cattle Fleet Management Platform (P1 Digital Twin Simulator)
## Core Algorithms, Mathematical Models, Expected Inputs & Outputs

*Project: Edge IoT & Digital-Twin Herd Simulator for Cattle Fleet Management*  
*Architecture Reference: ADR-001 through ADR-015, Master PRD & HerdSimulator PRD*

---

## 1. System Overview & Problem Formulation

In livestock fleet management, monitoring a large herd (e.g., 20 cattle) requires comprehensive telemetry (body temperature, heat stress, behavioural activity, GPS coordinates, geofence status, battery, and anomaly alerts). Because deploying 20 physical hardware collars is cost-prohibitive in early development, this platform combines **1 physical ESP32 prototype collar (ID 1)** with **19 digital-twin simulated cattle (IDs 2–20)**.

The simulator generates synthetic sensor telemetry that strictly obeys real-world physical constraints, mathematical equations, and rate limits of cloud infrastructure (MathWorks ThingSpeak Free Tier), multiplexing 19 virtual cows over a single cloud channel while maintaining identical schema parity with physical hardware.

---

## 2. Core Algorithms & Mathematical Formulations

### Algorithm 1: Composite Product-Rule Risk Scoring Engine (ADR-007)
* **Purpose:** Combines disparate multi-modal sensor signals (fever, heat stress, restlessness, geofence breach/warning, social isolation, tamper) into a unified $[0, 100]$ integer risk score without arbitrary additive thresholds.
* **Mathematical Formulation:**
  Each signal is mapped to an independent severity component $S_i \in [0, 1]$:
  1. **Temperature Severity ($S_{\text{temp}}$):**
     $$S_{\text{temp}} = \text{clamp}\left(\frac{T_{\text{body}} - T_{\text{baseline}} - 0.5}{1.5}, 0, 1\right)$$
  2. **THI Heat Stress Severity ($S_{\text{THI}}$):**
     $$S_{\text{THI}} = \text{clamp}\left(\frac{\text{THI} - 68}{16}, 0, 1\right)$$
     $$\text{where } \text{THI} = (1.8 \times T_{\text{amb}} + 32) - (0.55 - 0.0055 \times RH) \times (1.8 \times T_{\text{amb}} - 26)$$
  3. **Behavioural Distress ($S_{\text{restless}}$):** $0.35$ if state is *Restless*, else $0.0$
  4. **Geofence Warning ($S_{\text{geo\_warn}}$):** $0.50$ if inside $10\text{m}$ warning boundary, else $0.0$
  5. **Geofence Breach ($S_{\text{geo\_breach}}$):** $0.90$ if outside pasture polygon, else $0.0$
  6. **Social Isolation ($S_{\text{isol}}$):** $0.70$ if straggler distance exceeds threshold, else $0.0$
  7. **Collar Tamper ($S_{\text{tamper}}$):** $0.90$ if collar unlatched/removed, else $0.0$

  **Unified Risk Score Calculation (Probability of At Least One Failure):**
  $$\text{Risk Score} = \text{round}\left(100 \times \left(1 - \prod_{i \in \{\text{temp}, \text{THI}, \text{restless}, \text{geo}, \text{isol}, \text{tamper}\}} (1 - S_i)\right)\right)$$
* **Classification Output:**
  * **$0 \le \text{Risk} \le 39$:** `Green` (Normal)
  * **$40 \le \text{Risk} \le 69$:** `Yellow` (Warning / Elevated Attention)
  * **$70 \le \text{Risk} \le 100$:** `Red` (Critical Alert / Immediate Intervention)

---

### Algorithm 2: Geospatial Geofence & 10-Meter Warning Band Classifier (ADR-008)
* **Purpose:** Determine if an animal is inside the designated pasture, in a pre-breach warning buffer zone, or in full breach.
* **Pseudocode / Steps:**
  1. **Haversine Great-Circle Distance:**
     $$a = \sin^2\left(\frac{\Delta \text{lat}}{2}\right) + \cos(\text{lat}_1)\cos(\text{lat}_2)\sin^2\left(\frac{\Delta \text{lon}}{2}\right)$$
     $$d = 2 R \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1-a}\right), \quad R = 6,371,000\text{ m}$$
  2. **Ray-Casting Algorithm:** Cast a horizontal ray from point $P(\text{lat}, \text{lon})$ across polygon edges. Count edge intersections. If intersection count is odd, point is inside; if even, point is outside.
  3. **Perpendicular Distance to Polygon Perimeter:**
     Compute minimum Euclidean distance $d_{\text{min}}$ (in local projection meters) from point $P$ to each line segment $(V_i, V_{i+1})$ of the polygon.
  4. **Status Assignment:**
     $$\text{Geofence Status} = \begin{cases} 
     2 \text{ (Breach)}, & \text{if not } \text{PointInPolygon}(P, \text{Pasture}) \\
     1 \text{ (Warning Zone)}, & \text{if } \text{PointInPolygon}(P, \text{Pasture}) \text{ and } d_{\text{min}} \le 10.0\text{ m} \\
     0 \text{ (Inside Safe Zone)}, & \text{otherwise}
     \end{cases}$$

---

### Algorithm 3: 5-State Time-of-Day Weighted Markov Behaviour Model (ADR-006)
* **Purpose:** Simulates realistic cattle diurnal activity transitions without generating illegal state transitions.
* **State Space:** $\mathcal{S} = \{\text{Resting (0)}, \text{Grazing (1)}, \text{Ruminating (2)}, \text{Walking (3)}, \text{Restless (4)}\}$. (State 5 is reserved for low-confidence ML output).
* **Allowed Transition Graph:**
  * $\text{Resting} \rightarrow \{\text{Grazing}, \text{Restless}\}$
  * $\text{Grazing} \rightarrow \{\text{Ruminating}, \text{Walking}\}$
  * $\text{Ruminating} \rightarrow \{\text{Resting}\}$
  * $\text{Walking} \rightarrow \{\text{Grazing}\}$
  * $\text{Restless} \rightarrow \{\text{Resting}\}$
* **Time-of-Day Modulation ($H = \text{hour} \pmod{24}$):**
  * **Dawn/Dusk ($H \in \{5,6,7, 17,18,19\}$):** Grazing and Walking transition probabilities multiplied by $1.5\times$.
  * **Midday/Night ($H \in \{0,1,2,3, 11,12,13, 22,23\}$):** Resting and Ruminating transition probabilities multiplied by $1.5\times$.
* **Anomaly Boost:** When an anomaly (fever, heat stress, isolation) is active, an additive $+0.30$ probability is injected onto the $\text{Resting} \rightarrow \text{Restless}$ edge.

---

### Algorithm 4: Herd Kinematics, Flocking & Collar-1 Sniffing (ADR-010)
* **Purpose:** Simulates cohesive herd movement and coupling to the real physical collar.
* **Steps:**
  1. **Centroid Update:**
     * *Autonomous Mode:* Centroid performs a bounded 2D random walk within the pasture polygon at speed $v_{\text{centroid}} \approx 0.02\text{ m/s}$.
     * *Anchored Mode (Collar-1 Sniffing):* Simulator performs periodic HTTP GET to ThingSpeak Channel 1. If physical collar GPS fix is fresh ($<120\text{s}$ old), centroid steps smoothly toward Collar 1 coordinates:
       $$\vec{C}_{t+1} = \text{MoveToward}(\vec{C}_t, \vec{P}_{\text{collar1}}, \text{max\_step} = v_{\text{drift}} \times \Delta t)$$
  2. **Individual Cattle Positioning (Cohesion & Modulation):**
     Each cow $i$ maintains a preferred angular bearing $\theta_i$ and offset distance $r_i \le r_{\text{max}} = 30\text{m}$.
     $$\vec{P}_i = \vec{C} + \text{PolarToOffset}(r_i, \theta_i) + \vec{\epsilon}_{\text{GPS}}$$
     Ground speed is modulated by behaviour: $\text{Resting} (0\text{ m/s}) < \text{Ruminating} (0.01\text{ m/s}) < \text{Grazing} (0.08\text{ m/s}) < \text{Restless} (0.25\text{ m/s}) < \text{Walking} (0.8\text{–}1.4\text{ m/s})$.
  3. **Anomaly Excursions:**
     * *Isolation:* Position drifts outward along vector $\vec{P}_i - \vec{C}$ by $0.15\text{ m/s}$ up to $120\text{m}$.
     * *Breach:* Animal walks directly toward the nearest polygon boundary and steps outside.

---

### Algorithm 5: Activity-Aware Non-Linear Battery Drain (ADR-009)
* **Purpose:** Models physical battery discharge without solar recharging (strict monotonic depletion).
* **Mathematical Formula:**
  $$\text{Rate} = \begin{cases}
  0.5\% \text{ per hour} & \text{(Baseline resting/grazing, 30s cadence)} \\
  1.5\% \text{ per hour } (3\times \text{ multiplier}) & \text{(Alert/breach bursts, 15s cadence)}
  \end{cases}$$
  $$\text{Battery}_{t+1} = \max\left(0.0, \text{Battery}_t - \text{Rate} \times \frac{\Delta t}{3600}\right)$$
  $$\text{If } \text{Battery}_{t+1} == 0.0 \implies \text{Trigger } \texttt{collar\_dropout} \text{ and cease all transmissions.}$$

---

### Algorithm 6: Round-Robin Multiplexed Scheduler with Priority Anomaly Jump (ADR-004)
* **Purpose:** Maximizes coverage of 19 virtual cattle over 1 ThingSpeak channel while respecting the strict 15-second physical write floor.
* **Mechanism:**
  * Base state: Queue $\mathcal{Q} = [2, 3, 4, \dots, 20]$ operates round-robin at $30\text{s}$ interval (full sweep = $19 \times 30\text{s} = 570\text{s} \approx 9.5\text{ min}$).
  * Anomaly injection (Fever, Breach, Tamper, Isolation, Heat Stress): The affected animal ID immediately jumps to the head of priority queue $\mathcal{P}$.
  * Next transmission slot selects from $\mathcal{P}$ first (cadence compressed to $15\text{s}$ during alerts); if $\mathcal{P}$ is empty, selects from $\mathcal{Q}$.
  * Enforces timestamp check: $\text{Timestamp}_{\text{now}} - \text{Timestamp}_{\text{last\_write}} \ge 15.0\text{ seconds}$ before issuing HTTP POST.

---

### Algorithm 7: 190-Pair Ground Truth Spatial Proximity Engine (ADR-013)
* **Purpose:** Logs exact pairwise ground truth physical distances for every combination $C(20, 2) = \frac{20 \times 19}{2} = 190$ animal pairs at every simulation second.
* **Outputs:** Pairwise distance matrix in meters + health state flags, enabling evaluation of downstream clustering, social grouping, and disease transmission contact tracing algorithms.

---

## 3. Comprehensive Input & Output Specifications

### 3.1 System Inputs

| Input Category | Source / Interface | Format / Type | Description / Example Values |
|---|---|---|---|
| **Configuration Parameters** | `config/default_config.yaml` | YAML file | Pasture boundary coordinates $[[\text{lat}, \text{lon}], \dots]$, baseline temp mean ($38.6^\circ\text{C}$), noise std ($0.05$), battery discharge rates, ThingSpeak channel IDs and cadences ($30\text{s}, 15\text{s}$). |
| **Declarative Scenarios** | `config/scenarios/*.json` | JSON file | Timed event schedule: `[{"sim_second": 300, "animal_id": 5, "event": "fever_onset", "params": {"peak_offset_c": 1.8}}]` |
| **Interactive Live CLI** | Terminal stdin (`live_cli.py`) | Text string command | Operator fault triggers: `fever 5`, `breach 14`, `tamper 7`, `isolate 10`, `dropout 3`, `clear 5`, `status`, `pause`, `resume`. |
| **Real Collar Telemetry (Herd Sniffing)** | ThingSpeak Channel 1 | HTTP GET JSON | Live GPS latitude & longitude from the physical ESP32 prototype collar to anchor virtual herd flocking. |
| **Environmental Conditions** | Diurnal synthesized model / Weather API | Float values | Ambient temperature ($24.0^\circ\text{C} \text{ night} \dots 32.0^\circ\text{C} \text{ day}$) and Relative Humidity ($65\% \pm 5\%$). |
| **RNG Seed** | Config / CLI flag | Integer (`seed=42`) | Deterministic seed to guarantee byte-identical replay across runs. |

---

### 3.2 System Outputs

#### A. Cloud Telemetry: ThingSpeak Channel 2 Uplink (Immutable 8-Field Contract)
Every write to ThingSpeak Channel 2 contains exactly 8 numeric fields and 1 reserved text header:

| Field Name | Meaning | Data Type | Units / Encoding | Valid Range | Example |
|---|---|---|---|---|---|
| `field1` | Body Temperature | Float | Degrees Celsius ($^\circ\text{C}$) | $35.0 \dots 43.0$ | `39.8` |
| `field2` | Heat Stress Index (THI) | Float | Dimensionless index | $50.0 \dots 100.0$ | `78.4` |
| `field3` | Behaviour State | Integer | Enum code | $0\dots 4$ ($0=\text{Rest}, 1=\text{Graze}, 2=\text{Ruminate}, 3=\text{Walk}, 4=\text{Restless}$) | `4` |
| `field4` | GPS Latitude | Float | Decimal degrees | WGS-84 coordinate | `12.97145` |
| `field5` | GPS Longitude | Float | Decimal degrees | WGS-84 coordinate | `79.15923` |
| `field6` | Composite Risk Score | Integer | Severity scale | $0 \dots 100$ ($0\text{-}39\text{ Green}, 40\text{-}69\text{ Yellow}, 70\text{-}100\text{ Red}$) | `84` |
| `field7` | Geofence Status | Integer | Status code | $0=\text{Inside}, 1=\text{10m Warning}, 2=\text{Breach}$ | `0` |
| `field8` | Battery Level | Integer | Percentage ($\%$) | $0 \dots 100$ | `94` |
| `status` | Reserved Header | String | Semicolon text format | `id=XX;evt=YY;src=ZZ` | `id=05;evt=FEVER;src=RULE` |

---

#### B. Local Web Visualizer HUD (`http://127.0.0.1:8000`)
* **Leaflet.js Map:** Renders pasture polygon, 10m inner warning perimeter, and 20 real-time color-coded pins:
  * 🟢 **Green Pin:** Normal condition ($\text{Risk} \le 39$)
  * 🟡 **Yellow Pin:** Warning condition ($40 \le \text{Risk} \le 69$)
  * 🔴 **Red Pin:** Critical alert ($\text{Risk} \ge 70$, fever, breach, or tamper)
  * ⚫ **Grey / Strikethrough:** Physical Collar 1 Offline or Battery Dropout ($0\%$)
* **Live Transmission Queue HUD:** Shows current round-robin queue order, next transmission countdown, and priority slots.
* **REST API Endpoints:**
  * `GET /api/health` $\rightarrow$ System uptime, sim second, memory, queue depth.
  * `GET /api/state` $\rightarrow$ Complete JSON array of all 20 cattle states.
  * `GET /api/queue` $\rightarrow$ Transmission queue state.
  * `POST /api/events` $\rightarrow$ Live anomaly injection API.

---

#### C. Local File System & Analytical Logs (`logs/run_<timestamp>/`)

1. **`telemetry.csv`:** Full historical per-second time series for all 20 cattle:
   ```csv
   sim_second,animal_id,is_physical,body_temp_c,thi,behaviour,lat,lon,risk_score,alert_band,geofence_status,battery_pct
   300,5,0,39.82,74.12,4,12.97145,79.15923,84,red,0,98.45
   ```
2. **`ground_truth_pairs.csv`:** All 190 pairwise distances per tick:
   ```csv
   sim_second,animal_a,animal_b,distance_m,a_anomaly,b_anomaly
   300,2,3,14.28,0,0
   300,2,4,8.91,0,0
   300,2,5,35.12,0,1
   ... (exactly 190 rows per second)
   ```
3. **`manifest.json` & `summary.json`:** Run metadata, random seed, execution duration, total message count, and validation digest for deterministic replay verification.

---

## 4. Summary Table: Input $\rightarrow$ Algorithm $\rightarrow$ Output Flow

```
┌────────────────────────────────────────────────────────┐
│                     INPUT SIGNALS                      │
│  • Config YAML (Pasture vertices, physiology, rates)   │
│  • Scenario JSON & Live CLI Commands (Faults)          │
│  • Diurnal Weather (Temp, Humidity)                    │
│  • Collar 1 Sniffer (ThingSpeak Ch.1 GET)              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   CORE ALGORITHMS                      │
│  1. Physiology & Fever Ramp Engine                     │
│  2. Markov Behaviour State Machine (Diurnal Weighted)  │
│  3. Centroid Drift, Flocking & GPS Kinematics          │
│  4. Ray-Casting & 10m Geofence Buffer Classifier       │
│  5. Product-Rule Severity & Risk Calculation           │
│  6. Activity-Aware Monotonic Battery Drain             │
│  7. Round-Robin & Priority Multiplexer Scheduler       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                     OUTPUT SINKS                       │
│  • ThingSpeak Ch.2: 8-Field Telemetry + status Header  │
│  • Web HUD: Leaflet.js Map + Risk Status Pins          │
│  • REST API: /api/state, /api/health, /api/queue       │
│  • Log Files: telemetry.csv, ground_truth_pairs.csv    │
└────────────────────────────────────────────────────────┘
```
