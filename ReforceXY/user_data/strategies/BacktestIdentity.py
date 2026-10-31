"""Implementation identity for native result and FreqAI prediction caches.

Keep both standalone strategy copies identical. Native resolver search and class
source metadata are private SDK contracts; recheck them on Freqtrade upgrades.
"""

import json
import os
import platform
import site
import sys
import sysconfig
from collections.abc import Iterator, Mapping
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from inspect import getfile
from pathlib import Path
from typing import Any, Final

import freqtrade
from freqtrade.exceptions import OperationalException
from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver

_IMPLEMENTATION_KEY: Final = "_freqai_strategies_implementation"
_COMMON_DEPENDENCIES: Final = (
    "freqtrade",
    "datasieve",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "optuna",
    "optuna-integration",
    "optunahub",
    "cmaes",
    "torch",
    "botorch",
    "gpytorch",
    "filelock",
    "joblib",
    "cloudpickle",
    "TA-Lib",
    "technical",
    "ft-pandas-ta",
)
_REGRESSION_DEPENDENCIES: Final = ("xgboost", "lightgbm", "catboost", "ngboost", "scikit-image")
_RL_DEPENDENCIES: Final = ("gymnasium", "stable-baselines3", "sb3-contrib")


def _installed_roots() -> set[Path]:
    """Locate interpreter libraries without excluding sibling project directories."""
    paths = sysconfig.get_paths()
    roots = {
        Path(value).resolve()
        for key in ("stdlib", "platstdlib", "purelib", "platlib")
        if (value := paths.get(key))
    }
    roots.update(
        Path(path).resolve() for path in site.getsitepackages([sys.prefix, sys.base_prefix])
    )
    roots.add(Path(site.getusersitepackages()).resolve())
    roots.add(Path(freqtrade.__file__).resolve().parent)
    return roots


def _python_sources(root: Path, excluded: set[Path]) -> Iterator[Path]:
    """Walk source files only; directory links cannot expand or cycle the boundary."""

    def fail(error: OSError) -> None:
        raise error

    for directory, children, files in os.walk(root, onerror=fail):
        parent = Path(directory)
        children[:] = [
            name
            for name in children
            if name not in ("__pycache__", ".git")
            and not (child := parent / name).is_symlink()
            and not (child / "pyvenv.cfg").is_file()
            and not any(child.resolve().is_relative_to(path) for path in excluded)
        ]
        for name in files:
            path = parent / name
            if (
                path.suffix == ".py"
                and not any(path.resolve().is_relative_to(directory) for directory in excluded)
                and path.is_file()
            ):
                yield path


def implementation_fingerprint(implementations: Mapping[str, type]) -> str:
    """Hash selected classes, bounded local Python trees and release dependencies.

    Every Python file inside local implementation/MRO directories contributes,
    including unused or dynamically imported helpers. Installed libraries never
    create scan roots. Resolver snapshots override disk source, even through
    file aliases. Absolute paths locate files but are not part of the identity.
    """
    classes = {role: list(leaf.__mro__[:-1]) for role, leaf in implementations.items()}
    excluded = _installed_roots()
    snapshots: dict[Path, set[bytes]] = {}
    class_sources: dict[tuple[str, int], tuple[Path, bytes | None]] = {}
    roots: dict[Path, list[tuple[str, int, str, str]]] = {}
    for role, lineage in classes.items():
        leaf = lineage[0]
        for index, cls in enumerate(lineage):
            if cls.__module__ in ("abc", "builtins") or cls.__module__.startswith("freqtrade."):
                continue
            snapshot = cls.__dict__.get("__source__")
            filename = cls.__dict__.get("__file__")
            if snapshot is None and filename is None and cls.__module__ == leaf.__module__:
                snapshot = leaf.__dict__.get("__source__")
                filename = leaf.__dict__.get("__file__")
            if filename is None:
                try:
                    filename = getfile(cls)
                except TypeError as error:
                    if cls.__module__ in sys.builtin_module_names:
                        continue
                    raise OperationalException(
                        f"Backtest cache: cannot locate Python source for {cls.__qualname__}"
                    ) from error
            path = Path(filename).resolve()
            if any(path.is_relative_to(directory) for directory in excluded):
                continue
            if path.suffix != ".py":
                raise OperationalException(
                    f"Backtest cache: Python source is required for {cls.__qualname__}"
                )
            payload = snapshot.encode() if snapshot is not None else None
            class_sources[(role, index)] = (path, payload)
            roots.setdefault(path.parent, []).append(
                (role, index, cls.__module__, cls.__qualname__)
            )
            if payload is not None:
                snapshots.setdefault(path, set()).add(payload)

    source_cache: dict[Path, tuple[bytes, ...]] = {}
    sources: set[tuple[str, bytes]] = set()

    def payloads(path: Path) -> tuple[bytes, ...]:
        path = path.resolve()
        if path not in source_cache:
            source_cache[path] = (
                tuple(sorted(snapshots[path])) if path in snapshots else (path.read_bytes(),)
            )
        return source_cache[path]

    for (role, index), (path, snapshot) in class_sources.items():
        label = json.dumps(("class", role, index))
        for payload in (snapshot,) if snapshot is not None else payloads(path):
            sources.add((label, payload))
    for root, anchors in roots.items():
        anchor = min(anchors)
        paths = set(_python_sources(root, excluded))
        paths.update(path for path in snapshots if path.is_relative_to(root))
        for path in paths:
            label = json.dumps(("tree", anchor, path.relative_to(root).as_posix()))
            for payload in payloads(path):
                sources.add((label, payload))
    sources.add(("identity-helper", Path(__file__).read_bytes()))
    rl = any(
        cls.__module__.startswith("freqtrade.freqai.RL.")
        for lineage in classes.values()
        for cls in lineage
    )
    dependencies = _COMMON_DEPENDENCIES + (_RL_DEPENDENCIES if rl else _REGRESSION_DEPENDENCIES)
    versions: dict[str, str | None] = {}
    for name in dependencies:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    configuration = {
        "python": platform.python_version(),
        "sdk": freqtrade.__version__,
        "versions": versions,
        "implementations": {
            role: [(cls.__module__, cls.__qualname__) for cls in lineage]
            for role, lineage in classes.items()
        },
    }
    digest = sha256(json.dumps(configuration, sort_keys=True).encode())
    for label, payload in sorted(sources):
        digest.update(json.dumps((label, len(payload))).encode())
        digest.update(payload)
    return digest.hexdigest()


def bind_backtest_implementation(strategy: Any) -> None:
    """Extend native result identity without model construction or storage changes."""
    config = strategy.config
    model_name = config.get("freqaimodel")
    if not config.get("freqai", {}).get("enabled", False) or not model_name:
        config.pop(_IMPLEMENTATION_KEY, None)
        return
    if model_name == "BaseRegressionModel":
        raise OperationalException(
            "BaseRegressionModel is a baseclass and cannot be used directly. "
            "Please choose an existing child class or inherit from this baseclass."
        )
    paths = FreqaiModelResolver.build_search_paths(config, FreqaiModelResolver.user_subdir)
    for path in paths:
        try:
            model, _ = FreqaiModelResolver._search_object(
                directory=path, object_name=model_name, add_source=True
            )
        except FileNotFoundError:
            continue
        if model is not None:
            config[_IMPLEMENTATION_KEY] = implementation_fingerprint(
                {
                    "strategy": type(strategy),
                    "model": model,
                }
            )
            return
    raise OperationalException(
        f"Impossible to load FreqaiModel '{model_name}'. "
        "This class does not exist or contains Python code errors."
    )
