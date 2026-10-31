"""Implementation identity for native result and FreqAI prediction caches.

Keep both standalone strategy copies identical. Native resolver search and class
source metadata are private SDK contracts; recheck them on Freqtrade upgrades.
"""

import ast
import json
import platform
import sys
from collections.abc import Iterator, Mapping
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from importlib.util import resolve_name
from inspect import getfile, isfunction, unwrap
from pathlib import Path
from types import ModuleType
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
    "pandas-ta",
)
_REGRESSION_DEPENDENCIES: Final = ("xgboost", "lightgbm", "catboost", "ngboost", "scikit-image")
_RL_DEPENDENCIES: Final = ("gymnasium", "stable-baselines3", "sb3-contrib")


def _import_namespaces(source: bytes, package: str) -> Iterator[dict[str, Any]]:
    """Find loaded imports even when a native-resolved leaf owns no methods."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                if not package:
                    continue
                try:
                    name = resolve_name("." * node.level + name, package)
                except ImportError:
                    continue
            names = [name, *(f"{name}.{alias.name}" for alias in node.names)]
        else:
            continue
        for name in names:
            module = sys.modules.get(name)
            if module is not None:
                yield vars(module)


def _class_namespaces(cls: type) -> Iterator[dict[str, Any]]:
    """Use actual function globals: native-resolved modules may be unregistered."""
    for member in vars(cls).values():
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        elif isinstance(member, property):
            member = member.fget
        if callable(member):
            member = unwrap(member)
        if isfunction(member):
            yield member.__globals__
    module = sys.modules.get(cls.__module__)
    # Data-only imported classes have no function globals. Reject name collisions.
    if module is not None and vars(module).get(cls.__name__) is cls:
        yield vars(module)


def implementation_fingerprint(implementations: Mapping[str, type]) -> str:
    """Hash selected classes, reachable local helpers and release dependencies.

    Paths locate sources but are not part of identity. Resolver snapshots take
    precedence over disk source. Local helper traversal stays inside directories
    containing selected non-SDK MRO implementations; it never scans user data or
    installed packages. Version metadata does not detect same-version SDK/binary
    modifications. File hashes are not memoized across strategy constructions.
    """
    classes = {role: list(leaf.__mro__[:-1]) for role, leaf in implementations.items()}
    local_classes = [
        cls
        for lineage in classes.values()
        for cls in lineage
        if cls.__module__ != "abc" and not cls.__module__.startswith("freqtrade.")
    ]
    sources: dict[tuple[str, bytes], None] = {}
    source_cache: dict[tuple[str, str | Path], bytes] = {}
    snapshots: dict[Path, str] = {}
    imports: dict[tuple[bytes, str], tuple[dict[str, Any], ...]] = {}
    directories: set[Path] = set()
    namespaces: list[dict[str, Any]] = []

    def add_source(path: str | Path, snapshot: str | None = None) -> bytes:
        key = ("snapshot", snapshot) if snapshot is not None else ("file", Path(path).resolve())
        if key not in source_cache:
            source_cache[key] = (
                snapshot.encode() if snapshot is not None else Path(path).read_bytes()
            )
        sources[(Path(path).name, source_cache[key])] = None
        return source_cache[key]

    def imported_namespaces(payload: bytes, package: str) -> tuple[dict[str, Any], ...]:
        key = (payload, package)
        if key not in imports:
            imports[key] = tuple(_import_namespaces(payload, package))
        return imports[key]

    for lineage in classes.values():
        leaf = lineage[0]
        for cls in lineage:
            if cls not in local_classes:
                continue
            snapshot = cls.__dict__.get("__source__")
            path = cls.__dict__.get("__file__")
            if snapshot is None and path is None and cls.__module__ == leaf.__module__:
                snapshot = leaf.__dict__.get("__source__")
                path = leaf.__dict__.get("__file__")
            if snapshot is None and path is None:
                path = getfile(cls)
            payload = add_source(path or cls.__module__, snapshot)
            if path is not None:
                resolved = Path(path).resolve()
                directories.add(resolved.parent)
                if snapshot is not None:
                    snapshots[resolved] = snapshot
            namespaces.extend(_class_namespaces(cls))
            namespaces.extend(imported_namespaces(payload, cls.__module__.rpartition(".")[0]))

    visited: set[int] = set()
    while namespaces:
        namespace = namespaces.pop()
        if id(namespace) in visited:
            continue
        visited.add(id(namespace))
        filename = namespace.get("__file__")
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not any(path.is_relative_to(directory) for directory in directories):
            continue
        payload = add_source(path, snapshots.get(path))
        namespaces.extend(imported_namespaces(payload, namespace.get("__package__") or ""))
        for value in tuple(namespace.values()):
            if callable(value):
                value = unwrap(value)
            if isfunction(value):
                namespaces.append(value.__globals__)
            elif isinstance(value, ModuleType):
                namespaces.append(vars(value))
            elif isinstance(value, type):
                namespaces.extend(_class_namespaces(value))

    add_source(__file__)
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
