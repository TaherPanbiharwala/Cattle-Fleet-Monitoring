# Cattle Anomaly Assistant

Stage 1 (classical ML anomaly detection) + Stage 2 (RAG/LLM explanation) for the cattle-collar project. Structure follows [`LLM_Diagnostic_Assistant_PRD.md`](../LLM_Diagnostic_Assistant_PRD.md) Section 13.

This is a self-contained subproject inside the larger `IoT` repo — its own `pyproject.toml`/`requirements.txt`, own `tests/`. It does not import from `../src/` (the herd-simulator/collar-firmware platform); anything reused from there (e.g. the WASP-lab behavior classifier) is copied in as a starting point, not imported cross-folder.

**Before doing anything here, read:**
1. [`../LLM_Diagnostic_Assistant_PRD.md`](../LLM_Diagnostic_Assistant_PRD.md) — the frozen spec.
2. [`../LLM_ASSISTANT_STATUS.md`](../LLM_ASSISTANT_STATUS.md) — current milestone/progress status, open questions, and the onboarding block to paste into a fresh Codex/Claude Code/Antigravity session.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # includes ragas — real weight (numpy/pandas/langchain-core/instructor)
.venv/bin/python -m pytest tests/ -v        # 132/132, all against fake LLM/judge clients, zero API keys
.venv/bin/python -m stage2_rag_assistant.kb.build_kb --overwrite   # build the local KB (gitignored, not needed for tests)
.venv/bin/python -m stage2_rag_assistant.eval.run_eval             # run the Layer 2 eval, writes a report under eval/reports/
.venv/bin/python -m stage2_rag_assistant.calibration.tune_threshold   # prototype tau bucketing, writes a report under calibration/reports/
```

If anything importing `ragas` fails on a fresh install, read [`stage2_rag_assistant/eval/_ragas_compat.py`](stage2_rag_assistant/eval/_ragas_compat.py)'s docstring first — it's a documented upstream packaging bug, not a setup mistake.

## Layout

- `shared/schemas/` — the frozen `AnomalyRecord` / `AnomalyExplanationQuery` / `AnomalyExplanationResponse` / `GoldenCase` pydantic models (PRD Section 6), all inheriting a `StrictModel` base (`extra="forbid"`, `AwareDatetime` timestamps). Both stages import from here; never duplicate a model.
- `stage2_rag_assistant/mock/generate_mock_records.py` — produces schema-valid mock `AnomalyRecord`s so Stage 2 could be built before Stage 1 exists (PRD Section 6.2). Swap for `stage1_anomaly_detection/to_anomaly_record.py`'s real output at the PRD Section 14 integration point.
- `stage2_rag_assistant/config/` — Stage 2 pipeline config (`default.yaml` + pydantic loader). `llm.provider: fake` by default — no real LLM provider chosen yet (PRD Open Question 2, deliberately deferred).
- `stage2_rag_assistant/llm/` — the `LLMClient` protocol, a factory (`fake` works today, any real provider raises `NotImplementedError` until one is chosen), and `providers/fake_provider.py`.
- `stage2_rag_assistant/kb/` — the SQLite knowledge base: `schema.sql`, an editable `seed_data/shift_categories.yaml` (the actual source of truth — edit this, not the built DB), and `build_kb.py`. The built `kb.sqlite3` is gitignored.
- `stage2_rag_assistant/pipeline/` — the 5 pipeline stages (`record_assembler`, `intent_router`, `retriever`, `generator`, `fallback_gate`) plus `response_builder`/`audit_log`, wired end to end by `orchestrator.run_pipeline()`.
- `stage2_rag_assistant/eval/` — the Layer 2 (Ragas) eval harness: a fake judge LLM/embedding (`ragas_llm.py`/`ragas_embeddings.py`, same deferred-provider policy as `llm/`), field mapping + scoring (`metrics.py`), a 24-case golden set under `golden/` (17 mock-derived + 7 hand-written adversarial — `real_cases.jsonl`/`injected_cases.jsonl` need real Stage 1 output and are deliberately not created yet), and `run_eval.py`. Read `_ragas_compat.py` before touching anything that imports `ragas` directly.
- `stage2_rag_assistant/golden_loading.py` — shared golden-set loading (`load_cases`/`query_for`), used by both `eval/` and `calibration/` — lives here rather than nested under either since both depend on it equally.
- `stage2_rag_assistant/calibration/` — the τ-calibration prototype (PRD Section 10 steps 1-3 only): a deterministic calibration/test split of the golden set, confidence bucketing, and the LLM's own per-bucket empirical accuracy (`bucketing.py`), run via `tune_threshold.py`. Never writes to `config/default.yaml`'s real `tau` — steps 4-6 need real Stage 1 output (Integration Point), not this milestone.
- `stage1_anomaly_detection/`, `stage2_rag_assistant/api/` — not created yet; see `LLM_ASSISTANT_STATUS.md` for what's next.
