"""CLI integration test for reward_space_analysis.

Purpose
-------
Execute a bounded, optionally shuffled subset of parameter combinations for `reward_space_analysis.py` to verify end-to-end execution.

Key features
------------
* Deterministic sampling with optional shuffling (`--shuffle_seed`).
* Optional duplication of first N scenarios under strict diagnostics (`--strict_sample`).
* Per-scenario timing and aggregate statistics (mean / min / max / median / p95 seconds).
* Warning counting based on header lines plus a breakdown of distinct warning headers.
* Log tail truncation controlled via `--tail_chars` (characters) or full logs via `--full_logs`.
* Direct CLI forwarding of bootstrap resample count to the child process.

Usage
-----
python test_cli.py --num_samples 50 --out_dir ../sample_run_output \
    --shuffle_seed 123 --strict_sample 3 --bootstrap_resamples 200

JSON Summary fields
-------------------
- total, successes[], failures[]
- mean_seconds, max_seconds, min_seconds, median_seconds, p95_seconds
- warnings_breakdown
- seeds (sampling/configuration seeds)
- metadata (timestamp_utc, python_version, platform, git_commit, schema_version=2, per_scenario_timeout)
- interrupted (optional)
Exit codes
----------
0: success, 1: failures present, 130: interrupted (partial summary written).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

try:
    from typing import NotRequired, Required  # Python >=3.11
except ImportError:
    from typing_extensions import NotRequired, Required  # Python <3.11

ConfigTuple = Tuple[str, str, float, int, int, int]

SUMMARY_FILENAME = "reward_space_cli_results.json"


class ScenarioResult(TypedDict):
    config: ConfigTuple
    status: str
    stdout: str
    stderr: str
    strict: bool
    seconds: Optional[float]
    warnings: int


class SummaryResult(TypedDict, total=False):
    # Required keys
    total: Required[int]
    successes: Required[List[ScenarioResult]]
    failures: Required[List[ScenarioResult]]
    mean_seconds: Required[Optional[float]]
    max_seconds: Required[Optional[float]]
    min_seconds: Required[Optional[float]]
    median_seconds: Required[Optional[float]]
    p95_seconds: Required[Optional[float]]

    # Extension keys
    warnings_breakdown: NotRequired[Dict[str, int]]
    seeds: NotRequired[Dict[str, Any]]
    metadata: NotRequired[Dict[str, Any]]
    interrupted: NotRequired[bool]


_WARN_HEADER_RE = re.compile(r"^\s*(?:[A-Za-z]+Warning|WARNING)\b:?", re.IGNORECASE)


def _is_warning_header(line: str) -> bool:
    l = line.strip()
    if not l:
        return False
    if "warnings.warn" in l.lower():
        return False
    return bool(_WARN_HEADER_RE.search(l))


def build_arg_matrix(
    max_scenarios: int = 40,
    shuffle_seed: Optional[int] = None,
) -> List[ConfigTuple]:
    exit_potential_modes = [
        "canonical",
        "non_canonical",
        "progressive_release",
        "retain_previous",
        "spike_cancel",
    ]
    exit_attenuation_modes = ["linear", "sqrt", "power", "half_life", "legacy"]
    potential_gammas = [0.0, 0.5, 0.95, 0.999]
    hold_enabled = [0, 1]
    entry_additive_enabled = [0, 1]
    exit_additive_enabled = [0, 1]

    product_iter = itertools.product(
        exit_potential_modes,
        exit_attenuation_modes,
        potential_gammas,
        hold_enabled,
        entry_additive_enabled,
        exit_additive_enabled,
    )

    full: List[ConfigTuple] = list(product_iter)
    full = [c for c in full if not (c[0] == "canonical" and (c[4] == 1 or c[5] == 1))]
    if shuffle_seed is not None:
        rnd = random.Random(shuffle_seed)
        rnd.shuffle(full)
    if max_scenarios >= len(full):
        return full
    step = len(full) / max_scenarios
    idx_pos = step / 2.0  # Centered sampling
    selected: List[ConfigTuple] = []
    selected_indices: set[int] = set()
    for _ in range(max_scenarios):
        idx = int(round(idx_pos))
        if idx < 0:
            idx = 0
        elif idx >= len(full):
            idx = len(full) - 1
        if idx in selected_indices:
            left = idx - 1
            right = idx + 1
            while True:
                if left >= 0 and left not in selected_indices:
                    idx = left
                    break
                if right < len(full) and right not in selected_indices:
                    idx = right
                    break
                left -= 1
                right += 1
                if left < 0 and right >= len(full):
                    # All indices taken; fallback to current idx
                    break
        selected.append(full[idx])
        selected_indices.add(idx)
        idx_pos += step
    return selected


def run_scenario(
    script: Path,
    out_dir: Path,
    idx: int,
    num_samples: int,
    conf: ConfigTuple,
    strict: bool,
    bootstrap_resamples: int,
    timeout: int,
    skip_feature_analysis: bool = False,
    skip_partial_dependence: bool = False,
    unrealized_pnl: bool = False,
    full_logs: bool = False,
    params: Optional[List[str]] = None,
    tail_chars: int = 5000,
) -> ScenarioResult:
    (
        exit_potential_mode,
        exit_attenuation_mode,
        potential_gamma,
        hold_enabled,
        entry_additive_enabled,
        exit_additive_enabled,
    ) = conf
    scenario_dir = out_dir / f"scenario_{idx:02d}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--num_samples",
        str(num_samples),
        "--out_dir",
        str(scenario_dir),
        "--exit_potential_mode",
        exit_potential_mode,
        "--exit_attenuation_mode",
        exit_attenuation_mode,
        "--potential_gamma",
        str(potential_gamma),
        "--hold_potential_enabled",
        str(hold_enabled),
        "--entry_additive_enabled",
        str(entry_additive_enabled),
        "--exit_additive_enabled",
        str(exit_additive_enabled),
        "--seed",
        str(100 + idx),
    ]
    # Forward bootstrap resamples explicitly
    cmd += ["--bootstrap_resamples", str(bootstrap_resamples)]
    if skip_feature_analysis:
        cmd.append("--skip_feature_analysis")
    if skip_partial_dependence:
        cmd.append("--skip_partial_dependence")
    if unrealized_pnl:
        cmd.append("--unrealized_pnl")
    if strict:
        cmd.append("--strict_diagnostics")
    if params:
        cmd += ["--params"] + list(params)
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "config": conf,
            "status": "timeout",
            "stderr": "<timeout>",
            "stdout": "",
            "strict": strict,
            "seconds": None,
            "warnings": 0,
        }
    status = "ok" if proc.returncode == 0 else f"error({proc.returncode})"
    end = time.perf_counter()
    if proc.returncode != 0:
        cmd_str = " ".join(cmd)
        stderr_head_lines = proc.stderr.splitlines()[:3]
        stderr_head = "\n".join(stderr_head_lines)
        print(f"[error details] command: {cmd_str}")
        if stderr_head:
            print(f"[error details] stderr head:\n{stderr_head}")
        else:
            print("[error details] stderr is empty.")
    combined = proc.stdout.splitlines() + proc.stderr.splitlines()
    warnings = sum(1 for line in combined if _is_warning_header(line))
    if full_logs:
        stdout_out = proc.stdout
        stderr_out = proc.stderr
    else:
        if tail_chars == 0:
            stdout_out = ""
            stderr_out = ""
        else:
            stdout_out = proc.stdout[-tail_chars:]
            stderr_out = proc.stderr[-tail_chars:]
    return {
        "config": conf,
        "status": status,
        "stdout": stdout_out,
        "stderr": stderr_out,
        "strict": strict,
        "seconds": round(end - start, 4),
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=40,
        help="Number of synthetic samples per scenario (minimum 4 for feature analysis)",
    )
    parser.add_argument(
        "--skip_feature_analysis",
        action="store_true",
        help="Forward --skip_feature_analysis to child process to skip feature importance and model-based analysis for all scenarios.",
    )
    parser.add_argument(
        "--skip_partial_dependence",
        action="store_true",
        help="Forward --skip_partial_dependence to child process to skip partial dependence computation.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="sample_run_output",
        help="Output parent directory",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="If set, shuffle full scenario space before sampling a diverse subset",
    )
    parser.add_argument(
        "--strict_sample",
        type=int,
        default=0,
        help="Duplicate the first N scenarios executed again with --strict_diagnostics",
    )
    parser.add_argument(
        "--max_scenarios",
        type=int,
        default=40,
        help="Maximum number of (non-strict) scenarios before strict duplication",
    )
    parser.add_argument(
        "--bootstrap_resamples",
        type=int,
        default=120,
        help="Number of bootstrap resamples to pass to child processes (speed/perf tradeoff)",
    )
    parser.add_argument(
        "--per_scenario_timeout",
        type=int,
        default=600,
        help="Timeout (seconds) per child process (default: 600)",
    )
    parser.add_argument(
        "--full_logs",
        action="store_true",
        help="If set, store full stdout/stderr (may be large) instead of tail truncation.",
    )
    parser.add_argument(
        "--unrealized_pnl",
        action="store_true",
        help="Forward --unrealized_pnl to child process to exercise hold Φ(s) path.",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Forward parameter overrides to child process via --params, e.g. action_masking=0",
    )
    parser.add_argument(
        "--tail_chars",
        type=int,
        default=5000,
        help="Characters to keep from stdout/stderr tail when not storing full logs.",
    )
    args = parser.parse_args()

    # Basic validation
    if args.max_scenarios <= 0:
        parser.error("--max_scenarios must be > 0")
    if args.num_samples < 4 and not args.skip_feature_analysis:
        parser.error("--num_samples must be >= 4 unless --skip_feature_analysis is set")
    if args.strict_sample < 0:
        parser.error("--strict_sample must be >= 0")
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap_resamples must be > 0")
    if args.tail_chars < 0:
        parser.error("--tail_chars must be >= 0")
    if args.per_scenario_timeout <= 0:
        parser.error("--per_scenario_timeout must be > 0")

    script = Path(__file__).parent / "reward_space_analysis.py"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_arg_matrix(max_scenarios=args.max_scenarios, shuffle_seed=args.shuffle_seed)

    # Validate --params basic KEY=VALUE format
    valid_params: List[str] = []
    invalid_params: List[str] = []
    for p in args.params:
        if "=" in p:
            valid_params.append(p)
        else:
            invalid_params.append(p)
    if invalid_params:
        msg = f"Warning: ignoring malformed --params entries: {invalid_params}"
        print(msg, file=sys.stderr)
        print(f"{msg}")
    args.params = valid_params

    # Prepare list of (conf, strict)
    scenario_pairs: List[Tuple[ConfigTuple, bool]] = [(c, False) for c in scenarios]
    indices = {conf: idx for idx, conf in enumerate(scenarios, start=1)}
    n_duplicated = max(0, min(args.strict_sample, len(scenarios)))
    if n_duplicated > 0:
        print(f"Duplicating first {n_duplicated} scenarios with --strict_diagnostics")
    for c in scenarios[:n_duplicated]:
        scenario_pairs.append((c, True))

    results: List[ScenarioResult] = []
    total = len(scenario_pairs)
    interrupted = False
    try:
        for i, (conf, strict) in enumerate(scenario_pairs, start=1):
            res = run_scenario(
                script=script,
                out_dir=out_dir,
                idx=i,
                num_samples=args.num_samples,
                conf=conf,
                strict=strict,
                bootstrap_resamples=args.bootstrap_resamples,
                timeout=args.per_scenario_timeout,
                skip_feature_analysis=args.skip_feature_analysis,
                skip_partial_dependence=args.skip_partial_dependence,
                unrealized_pnl=args.unrealized_pnl,
                full_logs=args.full_logs,
                params=args.params,
                tail_chars=args.tail_chars,
            )
            results.append(res)
            status = res["status"]
            strict_str = f"[strict duplicate_of={indices.get(conf, '?')}]" if strict else ""
            secs = res.get("seconds")
            secs_str = f" {secs:.2f}s" if secs is not None else ""
            print(f"[{i}/{total}] {conf} {strict_str} -> {status}{secs_str}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received: writing partial summary...")

    successes = [r for r in results if r["status"] == "ok"]
    failures = [r for r in results if r["status"] != "ok"]
    durations: List[float] = [
        float(r["seconds"]) for r in results if isinstance(r["seconds"], float)
    ]
    if durations:
        _sorted = sorted(durations)
        median_seconds = statistics.median(_sorted)
        n = len(_sorted)
        if n == 1:
            p95_seconds = _sorted[0]
        else:
            pos = 0.95 * (n - 1)
            i0 = int(math.floor(pos))
            i1 = int(math.ceil(pos))
            if i0 == i1:
                p95_seconds = _sorted[i0]
            else:
                w = pos - i0
                p95_seconds = _sorted[i0] + (_sorted[i1] - _sorted[i0]) * w
    else:
        median_seconds = None
        p95_seconds = None
    summary: SummaryResult = {
        "total": len(results),
        "successes": successes,
        "failures": failures,
        "mean_seconds": round(sum(durations) / len(durations), 4) if durations else None,
        "max_seconds": max(durations) if durations else None,
        "min_seconds": min(durations) if durations else None,
        "median_seconds": median_seconds,
        "p95_seconds": p95_seconds,
    }
    # Build warnings breakdown
    warnings_breakdown: Dict[str, int] = {}
    for r in results:
        text = (r["stderr"] + "\n" + r["stdout"]).splitlines()
        for line in text:
            if _is_warning_header(line):
                fp = " ".join(line.strip().split())[:160]
                warnings_breakdown[fp] = warnings_breakdown.get(fp, 0) + 1

    # Collect environment + reproducibility metadata
    def _git_hash() -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            if proc.returncode == 0:
                return proc.stdout.strip() or None
        except Exception:
            return None
        return None

    summary.update(
        {
            "warnings_breakdown": warnings_breakdown,
            "seeds": {
                "shuffle_seed": args.shuffle_seed,
                "strict_sample": args.strict_sample,
                "max_scenarios": args.max_scenarios,
                "bootstrap_resamples": args.bootstrap_resamples,
            },
            "metadata": {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "git_commit": _git_hash(),
                "schema_version": 2,
                "per_scenario_timeout": args.per_scenario_timeout,
            },
        }
    )
    if interrupted:
        summary["interrupted"] = True
    # Atomic write to avoid corrupt partial files
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="_tmp_summary_", dir=str(out_dir))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        os.replace(tmp_path, out_dir / SUMMARY_FILENAME)
    except Exception:
        # Best effort fallback
        try:
            Path(out_dir / SUMMARY_FILENAME).write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    else:
        # Defensive cleanup: remove temp file if atomic replace did not clean up
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    print("Summary written to", out_dir / SUMMARY_FILENAME)
    if not interrupted and summary["failures"]:
        print("Failures detected:")
        for f in summary["failures"]:
            print(f"  - {f['config']}: {f['status']}")
        sys.exit(1)
    if interrupted:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
