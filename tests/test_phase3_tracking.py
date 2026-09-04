"""Tests for the dependency-isolated, aggregate-only W&B boundary."""

from __future__ import annotations

import pytest

from ml.wandb_tracking import WandbSettings, WandbTracker


class _FakeArtifact:
    def __init__(self, *, name: str, type: str, metadata: dict) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str]] = []

    def add_file(self, source: str, name: str) -> None:
        self.files.append((source, name))


class _FakeRun:
    def __init__(self) -> None:
        self.logged_artifacts: list[tuple[_FakeArtifact, list[str]]] = []

    def log(self, data: dict) -> None:
        del data

    def log_artifact(self, artifact: _FakeArtifact, aliases: list[str]) -> None:
        self.logged_artifacts.append((artifact, aliases))

    def finish(self) -> None:
        return None


class _FakeWandb:
    Artifact = _FakeArtifact

    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


def test_disabled_wandb_tracking_needs_no_wandb_installation() -> None:
    tracker = WandbTracker(WandbSettings(mode="disabled"))

    run = tracker.start_fold(model_name="random_forest", held_out_cow_id="cow-1", config={"seed": 42})
    run.log({"macro_f1": 0.9})
    run.finish()


def test_wandb_settings_validate_modes() -> None:
    with pytest.raises(ValueError, match="mode"):
        WandbSettings(mode="not-a-mode")


def test_offline_tracking_uses_required_tags_and_rejects_raw_artifacts(tmp_path) -> None:
    tracker = WandbTracker(WandbSettings(mode="offline", group="benchmark-group"))
    fake_wandb = _FakeWandb()
    tracker._wandb = fake_wandb
    run = tracker.start_summary({"seed": 42})

    assert fake_wandb.init_calls[0]["mode"] == "offline"
    assert fake_wandb.init_calls[0]["group"] == "benchmark-group"
    assert set(WandbSettings().tags).issubset(set(fake_wandb.init_calls[0]["tags"]))

    metrics = tmp_path / "fold_metrics.csv"
    metrics.write_text("macro_f1\n0.9\n", encoding="utf-8")
    tracker.log_aggregate_artifact(
        run,
        name="phase3-benchmark-report",
        artifact_type="benchmark-report",
        files=(metrics,),
        metadata={"raw_samples_embedded": False},
    )
    artifact, aliases = fake_wandb.run.logged_artifacts[0]
    assert artifact.metadata == {"raw_samples_embedded": False}
    assert artifact.files == [(str(metrics), "fold_metrics.csv")]
    assert aliases == []

    prohibited = tmp_path / "raw_windows.csv"
    prohibited.write_text("not uploaded\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw IMU"):
        tracker.log_aggregate_artifact(
            run,
            name="not-used",
            artifact_type="benchmark-report",
            files=(prohibited,),
            metadata={},
        )
