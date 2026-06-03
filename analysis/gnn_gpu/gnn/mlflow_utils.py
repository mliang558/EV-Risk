"""Optional MLflow experiment tracking for GNN pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MlflowTracker:
    """Thin wrapper: no-op if MLflow disabled or not installed."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        experiment_name: str = "ev-charging-resilience-gnn",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        self.enabled = enabled
        self._run = None
        self._mlflow = None
        if not enabled:
            return
        try:
            import mlflow
        except ImportError:
            print("[MLflow] pip install mlflow to enable tracking; continuing without it.", flush=True)
            self.enabled = False
            return

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
        print(
            f"[MLflow] run_id={self._run.info.run_id} | experiment={experiment_name} | "
            f"uri={mlflow.get_tracking_uri()}",
            flush=True,
        )

    @property
    def run_id(self) -> str | None:
        if self._run is None:
            return None
        return self._run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled or self._mlflow is None:
            return
        clean = {k: str(v) for k, v in params.items()}
        self._mlflow.log_params(clean)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        p = Path(path)
        if not p.exists():
            return
        if p.is_dir():
            self._mlflow.log_artifacts(str(p), artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(str(p), artifact_path=artifact_path)

    def log_model_pytorch(self, model: Any, artifact_path: str = "model") -> None:
        if not self.enabled or self._mlflow is None:
            return
        try:
            self._mlflow.pytorch.log_model(model, artifact_path=artifact_path)
        except Exception as e:
            print(f"[MLflow] could not log pytorch model: {e}", flush=True)

    def log_tags(self, tags: dict[str, str]) -> None:
        if not self.enabled or self._mlflow is None:
            return
        self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def end(self) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.end_run()

    def __enter__(self) -> "MlflowTracker":
        return self

    def __exit__(self, *args) -> None:
        self.end()


def log_stratum_metrics(tracker: MlflowTracker, props: dict[str, float], prefix: str) -> None:
    """Log urban/suburban/rural fractions with a prefix, e.g. train_urban_pct."""
    if not tracker.enabled:
        return
    metrics = {f"{prefix}_{k}_pct": float(v) for k, v in props.items()}
    tracker.log_metrics(metrics)
