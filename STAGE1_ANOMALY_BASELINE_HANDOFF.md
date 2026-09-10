# Stage 1 Personal-Baseline Anomaly Detection — Handoff

**For:** whoever's picking up the per-cow statistical baseline / anomaly-detection half of Stage 1.
**Your job, in one sentence:** for each cow, learn what's "normal" for *her specifically* from the MmCows dataset, then flag and score windows where she deviates from her own baseline using SPC/CUSUM.
**Last updated:** 2026-09-10
**Status:** genuinely unstarted — unlike the behavior-classifier half, there is no existing code for this anywhere in the repo. You're building from zero.

---

## 1. The 60-second version of what this project is

Same project as [`STAGE1_HANDOFF.md`](STAGE1_HANDOFF.md) — read that doc's Section 1 if you haven't, it's the same context and I won't repeat it here. Short version: Stage 1 (classical ML/stats, no LLM) produces a structured record saying "this cow looks different from her own normal, here's specifically what changed." Stage 1 has **two independent halves**:

- **Behavior classifier** (WASP-lab / `db-cow-walking` dataset) — someone else's task, don't worry about it. See [`STAGE1_HANDOFF.md`](STAGE1_HANDOFF.md) if you're curious what they're doing.
- **Personal-baseline anomaly detection** (MmCows dataset) — **this is you.**

Stage 2 (already built — an LLM layer that explains what you detect) doesn't need your attention either. It's built and tested against a mock version of your output; once your real code lands, someone swaps the mock for your real output and nothing else changes downstream. That's the whole point of the frozen schema in Section 4 below — get your output into that shape and everything else just works.

## 2. What already exists — nothing, this is genuinely new

Checked as of 2026-09-10: there is no MmCows code anywhere in this repo. No adapter, no baseline computation, no CUSUM implementation. This is different from the behavior-classifier handoff, where someone had already written a first pass — here you're starting from the dataset itself.

The **old** version of this project's roadmap (`Cattle_Fleet_Management_Master_PRD.md`, Phase 4 — now superseded, don't build it) used a different statistical approach (median/MAD robust z-scores) and different data sources (the fleet simulator / physical collar). **That formula and those data sources are explicitly dropped** — this task uses SPC/CUSUM over MmCows instead, full stop. If you ever see references to "P4" or a median-MAD formula elsewhere in this repo's docs, that's the old approach; ignore it.

## 3. The dataset

**MmCows** (Vu et al. 2024) — [github.com/neis-lab/mmcows](https://github.com/neis-lab/mmcows). 14 continuous days per cow, persistent cow IDs. You need exactly three of its sensor streams, plus one optional:

| Stream | What it actually is | Resolution | Use it for |
|---|---|---|---|
| `cbt` | Core body temperature | 1 minute | `cbt_deviation_sigma`, `cbt_cusum_value` |
| `ankle` | Leg **orientation** — lying vs. standing state. **Not raw acceleration** — this is the one correction the PRD explicitly calls out (Section 9), because it's an easy thing to get wrong writing ingestion code. | 1 minute | `lying_time_pct_24h`, `lying_time_deviation_sigma` |
| `thi` | Ambient temperature-humidity index | 1 minute | Context only — never itself a deviation signal, but read Section 5 below, it matters a lot for how you present `cbt` deviations |
| `immu` *(optional)* | Neck accelerometer + magnetometer — actual raw movement magnitude, if you want finer activity detail beyond the lying/standing state `ankle` gives you | 100 ms | `activity_magnitude_deviation_sigma` — do this second, after cbt/lying_time work |

**Explicitly excluded:** milk yield (not enough data to validate as a signal), visual/camera data (a much bigger download, no role in this design), and cow IDs `T13`/`T14` (these are stationary reference tags in the dataset, not actual cows — filter them out or your baselines will be nonsense).

I haven't explored MmCows' actual file layout/column format myself — it's not downloaded anywhere in this repo yet. Check the dataset's own repo/documentation for exact file structure when you start.

## 4. The output shape you're building toward

The frozen schema is [`cattle-anomaly-assistant/shared/schemas/anomaly_record.py`](cattle-anomaly-assistant/shared/schemas/anomaly_record.py) — don't edit it, just produce values that fit it. The fields that are your responsibility, per cow-window:

```
cbt_c                                float   — the raw reading
cbt_deviation_sigma                  float   — (cbt_c - her_baseline_mean) / her_baseline_std
cbt_cusum_value                      float   — see Section 6
lying_time_pct_24h                   float   — % of the last 24h classified "lying" via ankle orientation
lying_time_deviation_sigma           float   — same z-score idea, over lying_time_pct
thi                                  float   — pass the raw ambient THI through, no transformation
activity_magnitude_deviation_sigma   float | null   — only if you get to immu; null otherwise
anomaly_flag                         bool    — your deterministic decision (Section 7)
anomaly_score                        float [0,1]
driving_signals                      list[str]   — which of the above actually triggered (Section 7)
```

`driving_signals` uses a specific short vocabulary that Stage 2's retrieval already matches on exactly — use these literal strings, nothing else: `"cbt"`, `"lying_time"`, `"activity_magnitude"`. (There's also `"herd_isolation"` and `"behavior_state"` in the vocabulary — the first needs a fleet proximity graph that doesn't exist for MmCows, the second is the other collaborator's job. Neither is yours.)

**Not your job either:** actually wiring your output into a real `AnomalyRecord` object (a file called `to_anomaly_record.py`) — that's a later integration step, done once both Stage 1 halves have real, validated output. Get baseline + CUSUM working and validated against injected shifts (Section 6) first.

## 5. Why THI matters even though it's "just context"

This is the one place the PRD calls out multi-signal reasoning as a hard requirement (FR-10) for the LLM layer downstream, and it depends entirely on your output being right: a `cbt` deviation alongside elevated `thi` reads as *"probably heat stress, environmental explanation available"*; the same `cbt` deviation with normal `thi` reads as *"points toward the animal specifically."* You don't need to do this reasoning yourself — the LLM does — but you must **always populate `thi`** on every record (not just when `cbt` is a driving signal), or that reasoning has nothing to work with. Don't treat it as optional just because it's "context only."

## 6. How to actually do the detection — SPC/CUSUM

This is genuinely underspecified in the PRD beyond "per-cow baseline (μ, σ) and CUSUM implementation" — here's a concrete starting approach. Treat the specifics (windows, parameters) as a first draft to validate against Section 7's injection tests and adjust, not gospel.

**Step 1 — per-cow baseline.** For each cow and each signal (`cbt`, `lying_time_pct`, later `activity_magnitude`), pick a reference window from her 14-day record and compute her personal mean (μ) and standard deviation (σ) over it. MmCows has no illness ground truth (per the PRD, this is true of both datasets in this whole project — not a gap, it's *why* this is anomaly detection and not diagnosis), so there's no labeled "healthy period" to draw the baseline from. A reasonable starting split: **days 1–7 as the baseline reference window, days 8–14 as the monitoring window** where you actually compute deviations against that baseline. Document whatever split you pick — it's a real design decision, not a detail.

**Step 2 — deviation z-score.** For any later window's value `x`: `deviation_sigma = (x - μ) / σ`. This directly becomes `cbt_deviation_sigma` / `lying_time_deviation_sigma`.

**Step 3 — CUSUM.** A z-score alone reacts to single-window noise; CUSUM is specifically good at catching a *small, sustained* shift (like "grazing dropped for 3 days"), which is exactly the pattern this whole project cares about. Two-sided CUSUM (you need both directions — e.g. lying time can go up *or* down):

```
C+_t = max(0, C+_{t-1} + z_t - k)
C-_t = max(0, C-_{t-1} - z_t - k)
```

where `z_t` is that window's deviation-sigma from Step 2, and `C+`/`C-` both start at 0. `k` (the "slack") and the alarm threshold `h` are the two parameters you have to pick — standard starting points from SPC literature are **k = 0.5** and **h = 4 or 5** (both in units of σ), tuned to detect roughly a 1σ sustained shift without too many false alarms. An alarm fires when either `C+_t` or `C-_t` exceeds `h`. Store whichever of `C+`/`C-` is currently larger as `cbt_cusum_value` (or track both if you want the sign information — the schema only asks for one number, but nothing stops you from keeping more internally).

**Step 4 — anomaly_flag / anomaly_score / driving_signals** (also not specified by the PRD — propose-and-validate, same as above): a reasonable starting rule is `anomaly_flag = True` if *any* monitored signal's CUSUM crossed `h`; add that signal's short name to `driving_signals`. For `anomaly_score`, a simple squashing of the largest active CUSUM value into `[0,1]` (e.g. something like `1 - exp(-max_cusum / h)`, so it approaches 1 as the alarm gets stronger and approaches 0 near baseline) is a sane starting point — just don't reuse the *old* Master PRD's specific formula (`1 - exp(-max(D-1,0)/3)`), since that was defined over median/MAD z-scores, not CUSUM, and that whole formula is superseded per Section 2 above.

For a sense of what magnitude of shift is worth catching, Stage 2's knowledge base already has literature-derived reference numbers seeded (not a spec for your detector, just a sanity check that you're in the right ballpark): a sustained lying-time increase of ~2σ over 2+ days, a `cbt` rise of ~2σ, a grazing-time drop of ~30% over 3+ days (that last one's the behavior classifier's signal, not yours, but it's the same category of shift size). See [`cattle-anomaly-assistant/stage2_rag_assistant/kb/seed_data/shift_categories.yaml`](cattle-anomaly-assistant/stage2_rag_assistant/kb/seed_data/shift_categories.yaml) if you want to see the actual numbers.

## 7. Validating your own detector — the injection harness

Before this is trustworthy, you need to prove it actually catches known shifts, since there's no ground truth to check against otherwise. Build a small harness (PRD calls this task **M1c**, tightly coupled to your work here — do it as part of the same effort):

1. Take a real, unperturbed multi-day MmCows sequence for one cow.
2. Inject a synthetic shift of a *known* type and magnitude — e.g., add +2σ to her `cbt` readings for 3 consecutive days while leaving `thi` untouched, or reduce her `lying_time_pct` by some amount for a stretch of days.
3. Run your baseline + CUSUM detector over the injected sequence.
4. Check: did `anomaly_flag` turn `True` roughly when the injection started? Does `driving_signals` correctly name the signal you injected into, and not others?

This harness pays off twice — it's your own correctness check now, and it gets reused later (by someone else, at Stage 2's evaluation stage) to build the golden test set the whole system gets scored against. Worth building it properly the first time.

## 8. What's explicitly not your problem right now

- The WASP-lab behavior classifier — a completely different dataset and a different collaborator's job. See [`STAGE1_HANDOFF.md`](STAGE1_HANDOFF.md) if curious.
- `src/herd_simulator/` and `src/collar_gateway/` — the unrelated cattle-fleet IoT simulator and physical-collar firmware. Ignore completely.
- `cattle-anomaly-assistant/stage2_rag_assistant/` — the already-built LLM/RAG explanation layer. Done, tested against mock data, not your concern.
- `herd_isolation_score` — needs a fleet proximity graph that doesn't exist for MmCows data. Leave it `null`.
- Wiring your output into `AnomalyRecord` for real (`to_anomaly_record.py`) — a later joint step, not your first deliverable.

## 9. What you'll need, practically

Python + `numpy` (mean/std/cumulative-sum math) is genuinely enough for the statistics here — CUSUM is arithmetic, not a trained model, so none of the ML tooling (`scikit-learn`, `torch`) the behavior-classifier half needs applies to you. `pandas` will make wrangling MmCows' per-cow time series far less painful than hand-rolling it. Like the rest of `cattle-anomaly-assistant/`, build this as its own self-contained piece — don't import from `src/`, and once you have real dependencies, add them to that subproject's `requirements.txt`, not the repo root's.

## 10. Reading order

1. This document.
2. [`LLM_Diagnostic_Assistant_PRD.md`](LLM_Diagnostic_Assistant_PRD.md) Section 2 (background/datasets), Section 6.1 (the exact `AnomalyRecord` fields), Section 9's correction note on `ankle` vs. `immu`, and Section 11 (how the injection harness gets reused for evaluation later).
3. [`LLM_ASSISTANT_STATUS.md`](LLM_ASSISTANT_STATUS.md) §5's "M1b" and "M1c" sections — the task checklist this handoff is based on, kept current as work progresses.
4. [`AGENTS.md`](AGENTS.md) §3 — golden rules; secrets-via-env and no-diagnosis-claims both apply here same as everywhere else in this repo.

## 11. Reporting progress

Same as the other handoff: [`LLM_ASSISTANT_STATUS.md`](LLM_ASSISTANT_STATUS.md) is the shared tracker across every tool and every person working on this project. Tick boxes in §5's M1b/M1c sections and add a dated line to the progress log in §8 as you go — that's how anyone else picking this up later (including whoever eventually writes `to_anomaly_record.py`) knows what actually happened, what you decided (baseline window, k/h values, the anomaly-score formula), and why.
