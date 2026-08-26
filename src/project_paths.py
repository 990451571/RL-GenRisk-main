from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return configured_path("data_root", project_root() / "data", "RL_GENRISK_DATA_ROOT")


def codex_output_root() -> Path:
    return configured_path(
        "codex_output_root",
        Path("/mnt/e/codex_file") if os.name != "nt" else Path("E:/codex_file"),
        "RL_GENRISK_CODEX_OUTPUT_ROOT",
    )


def configured_path(key: str, default: str | Path | None = None, env_var: str | None = None) -> Path | None:
    value = os.getenv(env_var) if env_var else None
    if not value:
        value = load_local_paths().get(key)
    if not value:
        return Path(default) if default is not None else None
    path = Path(expand_windows_path_for_wsl(str(value)))
    if path.is_absolute():
        return path
    return project_root() / path


def train_label_path() -> Path:
    return configured_path(
        "train_label_path",
        project_root() / "experiments" / "protocol_B" / "train_driver_genes.csv",
        "RL_GENRISK_TRAIN_LABEL_PATH",
    )


def val_label_path() -> Path:
    return configured_path(
        "val_label_path",
        project_root() / "experiments" / "protocol_B" / "validation_driver_genes.csv",
        "RL_GENRISK_VAL_LABEL_PATH",
    )


def load_local_paths() -> dict[str, str]:
    config_path = project_root() / "config" / "local_paths.yaml"
    if not config_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        if value:
            values[key.strip()] = value
    return values


def expand_windows_path_for_wsl(value: str) -> str:
    if os.name == "nt":
        return value
    if len(value) >= 3 and value[1:3] in {":\\", ":/"}:
        drive = value[0].lower()
        rest = value[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return value


def describe_path_resolution() -> str:
    return (
        "Path precedence: explicit CLI argument > RL_GENRISK_* environment "
        "variable > config/local_paths.yaml > project-relative defaults. "
        "Train/Validation labels fail fast if the resolved files are missing; "
        "Test or external-holdout paths are never used as fallbacks."
    )
