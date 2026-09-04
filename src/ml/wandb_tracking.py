"""Optional, aggregate-only Weights & Biases tracking for Phase 3.

This module intentionally has no module-level W&B import.  The rest of the
benchmark can run locally with ``disabled`` tracking and Phase 1 never needs
the W&B dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class WandbSettings:
    """Public experiment metadata; API keys stay in process environment only."""

    mode: str = "disabled"
    project: str = "cattle-fleet-phase3"
    entity: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = (
        "phase3",
        "wasp",
        "kaggle",
        "cow-grouped",
        "public-benchmark",
    )

    def __post_init__(self) -> None:
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("W&B mode must be one of: online, offline, disabled.")


class RunLike(Protocol):
    def log(self, data: dict[str, Any]) -> None: ...

    def finish(self) -> None: ...


class _DisabledRun:
    def log(self, data: dict[str, Any]) -> None:
        del data

    def finish(self) -> None:
        return None


class WandbTracker:
    """Small adapter that limits W&B payloads to aggregate benchmark data."""

    def __init__(self, settings: WandbSettings) -> None:
        self.settings = settings
        self._wandb: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.mode != "disabled"

    def _sdk(self) -> Any:
        if self._wandb is None:
            try:
                import wandb
            except ImportError as exc:  # pragma: no cover - optional dependency boundary
                raise RuntimeError(
                    "W&B tracking was requested but the 'ml' optional dependencies are not installed."
                ) from exc
            self._wandb = wandb
        return self._wandb

    def start_fold(
        self,
        *,
        model_name: str,
        held_out_cow_id: str,
        config: dict[str, Any],
    ) -> RunLike:
        if not self.enabled:
            return _DisabledRun()
        return self._sdk().init(
            project=self.settings.project,
            entity=self.settings.entity,
            group=self.settings.group,
            name=f"{model_name}-cow-{held_out_cow_id}",
            job_type="outer-fold",
            tags=list(self.settings.tags),
            config=config,
            mode=self.settings.mode,
        )

    def start_summary(self, config: dict[str, Any]) -> RunLike:
        if not self.enabled:
            return _DisabledRun()
        return self._sdk().init(
            project=self.settings.project,
            entity=self.settings.entity,
            group=self.settings.group,
            name="benchmark-summary",
            job_type="benchmark-summary",
            tags=list(self.settings.tags),
            config=config,
            mode=self.settings.mode,
        )

    def log_aggregate_artifact(
        self,
        run: RunLike,
        *,
        name: str,
        artifact_type: str,
        files: tuple[Path, ...],
        metadata: dict[str, Any],
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Upload only reviewed reports/manifests/models, never source samples."""

        if not self.enabled:
            return
        prohibited_tokens = ("raw", "window", "sample", "telemetry")
        for path in files:
            name_lower = path.name.casefold()
            if any(token in name_lower for token in prohibited_tokens):
                raise ValueError("W&B artifacts must not include raw IMU data.")
            if path.suffix.casefold() == ".csv" and not (
                "metrics" in name_lower or "confusion" in name_lower
            ):
                raise ValueError("W&B CSV artifacts may contain only aggregate metrics or confusion matrices.")
        artifact = self._sdk().Artifact(name=name, type=artifact_type, metadata=metadata)
        for path in files:
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact, aliases=list(aliases))
