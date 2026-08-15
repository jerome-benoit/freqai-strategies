import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import sqlite3
import stat
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from pandas import DataFrame

_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_SCHEMA_VERSION = 3
_PACKAGE_NAMES = (
    "cmaes",
    "contourpy",
    "cycler",
    "datasieve",
    "fonttools",
    "freqtrade",
    "gymnasium",
    "kiwisolver",
    "matplotlib",
    "numpy",
    "optuna-dashboard",
    "optunahub",
    "pandas",
    "pyparsing",
    "scipy",
    "scikit-learn",
    "stable-baselines3",
    "sb3-contrib",
    "optuna",
    "torch",
)
_IMPORTED_MODULE_DISTRIBUTIONS = {
    "gymnasium": "gymnasium",
    "matplotlib": "matplotlib",
    "sb3_contrib": "sb3-contrib",
    "scipy": "scipy",
    "stable_baselines3": "stable-baselines3",
}


class _DurabilityIndeterminateError(OSError):
    """Report a visible replace whose directory durability could not be sealed."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            "ReforceXY artifact replacement is visible but its directory durability "
            f"is indeterminate: {path}"
        )
        self.path = path
        self.replacement_visible = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"ReforceXY artifact is not a regular file: {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def _sanitize_pair(pair: str) -> str:
    sanitized = pair.replace("/", "_").replace(":", "_")
    if not sanitized or sanitized in {".", ".."}:
        raise ValueError("ReforceXY pair cannot be converted to a safe artifact key")
    return sanitized


def _simple_relative_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty relative filename")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {".", ".."}:
        raise RuntimeError(f"{label} must be one traversal-free filename")
    return path.name


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _durable_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        fd = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            with os.fdopen(fd, "wb", closefd=True) as destination:
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.replace(temporary_path, path)
        try:
            _fsync_directory(path.parent)
        except BaseException as error:
            raise _DurabilityIndeterminateError(path) from error
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _durable_json_create(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(fd, "wb", closefd=True) as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)


def _read_regular_json_with_evidence(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"ReforceXY artifact is not a regular file: {path}")
        payload = bytearray()
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            payload.extend(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"ReforceXY artifact changed while reading: {path}")
        value = json.loads(payload)
    finally:
        os.close(fd)
    if not isinstance(value, dict):
        raise RuntimeError(f"ReforceXY JSON artifact must contain an object: {path}")
    return value, {
        "name": path.name,
        "bytes": after.st_size,
        "sha256": digest.hexdigest(),
    }


def _read_regular_json(path: Path) -> dict[str, Any]:
    value, _evidence = _read_regular_json_with_evidence(path)
    return value


_LOADED_SOURCE_SHA256 = _sha256_file(Path(__file__))


def _require_manifest_schema(manifest: Mapping[str, Any]) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version != _MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            "ReforceXY run manifest schema is missing or unsupported: "
            f"expected {_MANIFEST_SCHEMA_VERSION}, got {schema_version!r}"
        )


def _reproduction_id(manifest: Mapping[str, Any]) -> str:
    _require_manifest_schema(manifest)
    reproducibility_inputs = manifest.get("reproducibility_inputs")
    if not isinstance(reproducibility_inputs, Mapping):
        raise RuntimeError(
            "ReforceXY run manifest reproducibility_inputs must be a mapping"
        )
    return _sha256_bytes(
        _json_bytes(
            {
                "schema_version": manifest["schema_version"],
                "reproducibility_inputs": reproducibility_inputs,
            }
        )
    )


def _require_manifest_integrity(manifest: Mapping[str, Any]) -> None:
    expected = _reproduction_id(manifest)
    if manifest.get("reproduction_id") != expected:
        raise RuntimeError(
            "ReforceXY run manifest reproduction_id does not match its inputs"
        )


def _imported_module_evidence(
    module_name: str,
    distribution_name: str,
    metadata_version: str,
) -> dict[str, Any]:
    module = import_module(module_name)
    imported_version = getattr(module, "__version__", None)
    imported_path = getattr(module, "__file__", None)
    if not isinstance(imported_version, str) or not imported_version:
        raise RuntimeError(
            f"Imported module {module_name!r} does not expose a non-empty __version__"
        )
    if not isinstance(imported_path, str) or not imported_path:
        raise RuntimeError(
            f"Imported module {module_name!r} does not expose a non-empty __file__"
        )
    return {
        "distribution": distribution_name,
        "metadata_version": metadata_version,
        "imported_version": imported_version,
        "imported_path": str(Path(imported_path).resolve()),
        "version_mismatch": imported_version != metadata_version,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if (
        callable(value)
        and hasattr(value, "__module__")
        and hasattr(value, "__qualname__")
    ):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def _required_environment(
    model_source_path: Path,
    loaded_model_source_sha256: str,
) -> tuple[str, str, str]:
    git_commit = os.environ.get("REFORCEXY_GIT_COMMIT", "").lower()
    if not _GIT_COMMIT_PATTERN.fullmatch(git_commit):
        raise RuntimeError(
            "REFORCEXY_GIT_COMMIT must contain the clean 40-character Git commit. "
            "Start ReforceXY through ./docker-compose.sh."
        )

    freqtrade_image = os.environ.get("REFORCEXY_FREQTRADE_IMAGE", "").lower()
    if not _IMAGE_DIGEST_PATTERN.fullmatch(freqtrade_image):
        raise RuntimeError(
            "REFORCEXY_FREQTRADE_IMAGE must be an immutable image reference "
            "(repository@sha256:digest). Start ReforceXY through ./docker-compose.sh."
        )
    runtime_image_id = os.environ.get("REFORCEXY_RUNTIME_IMAGE_ID", "").lower()
    if not _IMAGE_ID_PATTERN.fullmatch(runtime_image_id):
        raise RuntimeError(
            "REFORCEXY_RUNTIME_IMAGE_ID must contain the exact local ReforceXY image ID. "
            "Start ReforceXY through ./docker-compose.sh."
        )
    expected_model_source_sha256 = os.environ.get(
        "REFORCEXY_MODEL_SOURCE_SHA256", ""
    ).lower()
    expected_manifest_source_sha256 = os.environ.get(
        "REFORCEXY_MANIFEST_SOURCE_SHA256", ""
    ).lower()
    current_model_source_sha256 = _sha256_file(model_source_path)
    current_manifest_source_sha256 = _sha256_file(Path(__file__))
    if (
        loaded_model_source_sha256 != expected_model_source_sha256
        or current_model_source_sha256 != expected_model_source_sha256
        or _LOADED_SOURCE_SHA256 != expected_manifest_source_sha256
        or current_manifest_source_sha256 != expected_manifest_source_sha256
    ):
        raise RuntimeError(
            "ReforceXY source changed after the container was created. "
            "Recreate it through ./docker-compose.sh up -d."
        )
    return git_commit, freqtrade_image, runtime_image_id


def _dataframe_evidence(dataframe: DataFrame) -> dict[str, Any]:
    metadata = {
        "columns": [str(column) for column in dataframe.columns],
        "dtypes": [str(dtype) for dtype in dataframe.dtypes],
        "index_names": [
            str(name) if name is not None else None for name in dataframe.index.names
        ],
        "shape": list(dataframe.shape),
    }
    digest = hashlib.sha256(_json_bytes(metadata))
    row_hashes = pd.util.hash_pandas_object(
        dataframe,
        index=True,
        categorize=True,
    )
    digest.update(row_hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    return {
        **metadata,
        "hash_algorithm": "sha256(metadata || pandas_hash_pandas_object_uint64)",
        "sha256": digest.hexdigest(),
    }


def _state_file_evidence(
    paths: Sequence[Path], snapshot_directory: Path
) -> list[dict[str, Any]]:
    evidence = []
    for path in sorted(set(paths), key=str):
        item: dict[str, Any] = {
            "name": path.name,
            "exists": path.is_file(),
        }
        if path.is_file():
            snapshot_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            snapshot_path = snapshot_directory / path.name
            if path.suffix == ".sqlite":
                source_uri = f"{path.resolve().as_uri()}?mode=ro"
                with (
                    sqlite3.connect(source_uri, uri=True) as source,
                    sqlite3.connect(snapshot_path) as destination,
                ):
                    source.backup(destination)
                    integrity_result = destination.execute(
                        "PRAGMA quick_check"
                    ).fetchone()
                    if integrity_result != ("ok",):
                        raise RuntimeError(
                            f"Optuna SQLite snapshot failed integrity check: "
                            f"{integrity_result!r}"
                        )
            else:
                shutil.copy2(path, snapshot_path)
            _fsync_file(snapshot_path)
            _fsync_directory(snapshot_directory)
            item.update(
                {
                    "bytes": snapshot_path.stat().st_size,
                    "sha256": _sha256_file(snapshot_path),
                    "snapshot_relative_to_run_inputs": str(
                        Path("optuna-state-before-run") / path.name
                    ),
                }
            )
        evidence.append(item)
    return evidence


def _training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    exchange = config.get("exchange", {})
    return _json_safe(
        {
            "exchange": {
                "name": exchange.get("name"),
                "pair_whitelist": exchange.get("pair_whitelist"),
            },
            "fee": config.get("fee"),
            "freqai": config.get("freqai", {}),
            "freqaimodel": config.get("freqaimodel"),
            "stake_amount": config.get("stake_amount"),
            "strategy": config.get("strategy"),
            "timeframe": config.get("timeframe"),
            "trading_mode": config.get("trading_mode"),
        }
    )


def _refresh_reproduction_id(manifest: dict[str, Any]) -> None:
    manifest["reproduction_id"] = _reproduction_id(manifest)


def create_run_manifest(
    *,
    config: Mapping[str, Any],
    dataframes: Mapping[str, DataFrame],
    data_path: Path,
    environment_parameters: Mapping[str, Any],
    model_parameters: Mapping[str, Any],
    model_source_path: Path,
    pair: str,
    run_instance_id: str,
    full_path: Path,
    hyperopt_enabled: bool,
    loaded_model_source_sha256: str,
    optuna_seed: int,
    model_seed: int,
    n_envs: int,
    n_eval_envs: int,
    optunahub_registry_ref: str,
) -> dict[str, Any]:
    git_commit, freqtrade_image, runtime_image_id = _required_environment(
        model_source_path,
        loaded_model_source_sha256,
    )
    training_config = _training_config(config)
    data_split_parameters = config.get("freqai", {}).get("data_split_parameters", {})
    feature_parameters = config.get("freqai", {}).get("feature_parameters", {})
    if feature_parameters.get("noise_standard_deviation", 0) not in (0, 0.0):
        raise RuntimeError(
            "Reproducibility requires "
            "freqai.feature_parameters.noise_standard_deviation=0 because "
            "DataSieve 0.1.9 uses an unscoped NumPy random generator"
        )
    svm_parameters = feature_parameters.get(
        "svm_params",
        {"shuffle": False, "nu": 0.01},
    )
    if (
        feature_parameters.get("use_SVM_to_remove_outliers", False)
        and svm_parameters.get("shuffle", False)
        and svm_parameters.get("random_state") is None
    ):
        raise RuntimeError(
            "Reproducibility requires feature_parameters.svm_params.random_state "
            "when the SVM outlier extractor shuffles its input"
        )
    if (
        data_split_parameters.get("shuffle", False)
        and data_split_parameters.get("random_state") is None
    ):
        raise RuntimeError(
            "Reproducibility requires freqai.data_split_parameters.random_state "
            "when the train/test split is shuffled"
        )
    if feature_parameters.get("shuffle_after_split", False):
        raise RuntimeError(
            "Reproducibility requires "
            "freqai.feature_parameters.shuffle_after_split=false because Freqtrade "
            "uses an unscoped Python random generator for this operation"
        )
    if (
        config.get("freqai", {})
        .get("rl_config", {})
        .get("randomize_starting_position", False)
    ):
        raise RuntimeError(
            "Reproducibility requires "
            "freqai.rl_config.randomize_starting_position=false because the "
            "Freqtrade environment uses an unscoped Python random generator"
        )
    pair_key = _sanitize_pair(pair)
    optuna_state_paths = [
        full_path / f"optuna-{pair_key}.sqlite",
        full_path / f"optuna-{pair_key}.log",
        full_path / f"hyperopt-best-params-{pair_key}.json",
        full_path / "optuna-retrain-counters.json",
    ]
    package_metadata = {package: version(package) for package in _PACKAGE_NAMES}
    runtime = {
        "freqtrade_image": freqtrade_image,
        "imported_modules": {
            module_name: _imported_module_evidence(
                module_name,
                distribution_name,
                package_metadata[distribution_name],
            )
            for module_name, distribution_name in sorted(
                _IMPORTED_MODULE_DISTRIBUTIONS.items()
            )
        },
        "optunahub_registry_ref": optunahub_registry_ref,
        "packages": package_metadata,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "reforcexy_image_id": runtime_image_id,
    }
    source = {
        "git_commit": git_commit,
        "files": {
            model_source_path.name: _sha256_file(model_source_path),
            Path(__file__).name: _sha256_file(Path(__file__)),
        },
    }
    configuration = {
        "sha256": _sha256_bytes(_json_bytes(training_config)),
        "value": training_config,
    }
    seeds = {
        "data_split_random_state": config.get("freqai", {})
        .get("data_split_parameters", {})
        .get("random_state"),
        "configured_model_seed": model_seed,
        "evaluation_environment": {
            "formula": "final_model_seed + 10000 + environment_rank",
            "rank_count": n_eval_envs,
        },
        "final_model_seed": model_seed,
        "optuna_sampler_seed": optuna_seed,
        "svm_outlier_random_state": svm_parameters.get("random_state"),
        "optuna_trial": {
            "enabled": hyperopt_enabled,
            "formula": "configured_model_seed",
        },
        "training_environment": {
            "formula": "final_model_seed + environment_rank",
            "rank_count": n_envs,
        },
        "holdout_environment": {
            "formula": "final_model_seed + 20000 + environment_rank",
            "rank_count": n_eval_envs,
        },
    }
    data = {
        name: _dataframe_evidence(dataframe)
        for name, dataframe in sorted(dataframes.items())
    }
    reproducibility_inputs = {
        "configuration": configuration,
        "data": data,
        "environment_parameters": _json_safe(environment_parameters),
        "configured_model_parameters": _json_safe(model_parameters),
        "optuna_state_before_run": _state_file_evidence(
            optuna_state_paths,
            data_path
            / "reproducibility-inputs"
            / run_instance_id
            / "optuna-state-before-run",
        ),
        "runtime": runtime,
        "source": source,
        "seeds": seeds,
    }
    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "status": "started",
        "started_at": _utc_now(),
        "run": {
            "data_path": str(data_path),
            "pair": pair,
            "run_instance_id": run_instance_id,
        },
        "reproduction_id": "",
        "reproducibility_inputs": reproducibility_inputs,
    }
    _refresh_reproduction_id(manifest)
    return manifest


def write_run_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    _require_manifest_integrity(manifest)
    _durable_json_replace(path, manifest)


def finalize_run_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    status: str,
    duration_seconds: float,
    optuna_status: str,
    resolved_model_parameters: Mapping[str, Any],
) -> None:
    _require_manifest_schema(manifest)
    if status not in {"failed", "interrupted"}:
        raise ValueError(
            "Non-consumable finalization accepts only failed or interrupted status"
        )
    manifest["status"] = status
    manifest["finished_at"] = _utc_now()
    manifest["result"] = {
        "duration_seconds": duration_seconds,
        "optuna_status": optuna_status,
        "resolved_model_parameters": _json_safe(resolved_model_parameters),
    }
    write_run_manifest(path, manifest)


def set_resolved_run_inputs(
    path: Path,
    manifest: dict[str, Any],
    *,
    dataframes: Mapping[str, DataFrame],
    environment_parameters: Mapping[str, Any],
    execution_environment: Mapping[str, Any],
    resolved_model_parameters: Mapping[str, Any],
) -> None:
    _require_manifest_integrity(manifest)
    manifest["reproducibility_inputs"]["data"].update(
        {
            name: _dataframe_evidence(dataframe)
            for name, dataframe in sorted(dataframes.items())
        }
    )
    manifest["reproducibility_inputs"]["environment_parameters"] = _json_safe(
        environment_parameters
    )
    manifest["reproducibility_inputs"]["execution_environment"] = _json_safe(
        execution_environment
    )
    manifest["reproducibility_inputs"]["seeds"]["final_model_seed"] = (
        resolved_model_parameters.get(
            "seed",
            manifest["reproducibility_inputs"]["seeds"]["configured_model_seed"],
        )
    )
    manifest["reproducibility_inputs"]["resolved_model_parameters"] = _json_safe(
        resolved_model_parameters
    )
    _refresh_reproduction_id(manifest)
    write_run_manifest(path, manifest)
