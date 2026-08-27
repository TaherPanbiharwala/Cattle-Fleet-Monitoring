"""
env_loader.py — Lightweight standard library .env file loader.

Loads KEY=VALUE pairs into os.environ without external dependencies.
Respects existing environment variables (uses setdefault).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_dotenv(dotenv_path: Optional[str | Path] = None) -> bool:
    """Load key-value pairs from a .env file into os.environ.

    Args:
        dotenv_path: Path to the .env file. If None, looks for .env in the
                     current working directory or project root.

    Returns:
        True if a .env file was found and loaded, False otherwise.
    """
    if dotenv_path is None:
        target = Path(".env")
        if not target.exists():
            # Check parent directories up to workspace root
            curr = Path.cwd()
            for parent in [curr] + list(curr.parents):
                candidate = parent / ".env"
                if candidate.exists():
                    target = candidate
                    break
    else:
        target = Path(dotenv_path)

    if not target.exists() or not target.is_file():
        return False

    try:
        content = target.read_text(encoding="utf-8")
    except OSError:
        return False

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()

        # Strip optional surrounding single or double quotes
        if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
            val = val[1:-1]

        if key:
            os.environ.setdefault(key, val)

    return True
