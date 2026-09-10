# Cattle Anomaly Assistant

Stage 1 (classical ML anomaly detection) + Stage 2 (RAG/LLM explanation) for the cattle-collar project. Structure follows [`LLM_Diagnostic_Assistant_PRD.md`](../LLM_Diagnostic_Assistant_PRD.md) Section 13.

This is a self-contained subproject inside the larger `IoT` repo — its own `pyproject.toml`/`requirements.txt`, own `tests/`. It does not import from `../src/` (the herd-simulator/collar-firmware platform); anything reused from there (e.g. the WASP-lab behavior classifier) is copied in as a starting point, not imported cross-folder.

**Before doing anything here, read:**
1. [`../LLM_Diagnostic_Assistant_PRD.md`](../LLM_Diagnostic_Assistant_PRD.md) — the frozen spec.
2. [`../LLM_ASSISTANT_STATUS.md`](../LLM_ASSISTANT_STATUS.md) — current milestone/progress status, open questions, and the onboarding block to paste into a fresh Codex/Claude Code/Antigravity session.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -v
```

## Layout

- `shared/schemas/` — the frozen `AnomalyRecord` / `AnomalyExplanationQuery` / `AnomalyExplanationResponse` / `GoldenCase` pydantic models (PRD Section 6). Both stages import from here; never duplicate a model.
- `stage2_rag_assistant/mock/generate_mock_records.py` — produces schema-valid mock `AnomalyRecord`s so Stage 2 can be built before Stage 1 exists (PRD Section 6.2). Swap for `stage1_anomaly_detection/to_anomaly_record.py`'s real output at the PRD Section 14 integration point.
- `stage1_anomaly_detection/`, `stage2_rag_assistant/{kb,pipeline,calibration,eval,api}/` — not created yet; see `LLM_ASSISTANT_STATUS.md` for what's next.
