"""Atomic, derived-only JSON artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .errors import BehaviorError, fail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail("RUNTIME_CONTEXT_INVALID", "The required JSONL input file does not exist.", path=str(path))
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                fail("RUNTIME_CONTEXT_INVALID", "Input contains invalid JSONL.", path=str(path), line=number)
            if not isinstance(value, dict):
                fail("RUNTIME_CONTEXT_INVALID", "Each JSONL line must be an object.", path=str(path), line=number)
            records.append(value)
    return records


def atomic_output_dir(output_dir: Path, writer: Any) -> Path:
    """Run ``writer(temp_dir)`` then atomically publish a new output directory."""

    final_dir = output_dir.expanduser().resolve()
    if final_dir.exists():
        fail("OUTPUT_EXISTS", "--output-dir must name a new, nonexistent directory.", output_dir=str(final_dir))
    try:
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.tmp-", dir=final_dir.parent))
    except OSError as exc:
        fail("OUTPUT_WRITE_ERROR", "Could not create a temporary output directory.", output_dir=str(final_dir), error=str(exc))
    try:
        writer(temp_dir)
        os.replace(temp_dir, final_dir)
    except BehaviorError:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        fail("OUTPUT_WRITE_ERROR", "Could not publish output atomically.", output_dir=str(final_dir), error=str(exc))
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return final_dir


def assert_new_output_dir(output_dir: Path) -> None:
    """Fail early so expensive benchmark work never targets an existing result."""

    final_dir = output_dir.expanduser().resolve()
    if final_dir.exists():
        fail("OUTPUT_EXISTS", "--output-dir must name a new, nonexistent directory.", output_dir=str(final_dir))
