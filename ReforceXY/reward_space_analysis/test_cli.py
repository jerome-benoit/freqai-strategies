"""CLI integration smoke test for reward_space_analysis.

Purpose
-------
Execute a bounded, optionally shuffled subset of parameter combinations for `reward_space_analysis.py` to verify end-to-end execution (smoke / regression signal, not correctness proof).

Key features
------------
* Deterministic sampling with optional shuffling (`--shuffle_seed`).
* Optional duplication of first N scenarios under strict diagnostics (`--strict_sample`).
* Per-scenario timing and aggregate statistics (mean / min / max seconds).
* Simple warning counting + (patch adds) breakdown of distinct warning lines.
* Scenario list + seed metadata exported for reproducibility.
* Direct CLI forwarding of bootstrap resample count to child process.

Usage
-----
python test_cli.py --samples 50 --out_dir ../sample_run_output_smoke \
    --shuffle_seed 123 --strict_sample 3 --bootstrap_resamples 200

JSON Summary fields
-------------------
total, ok, failures[], warnings_total, warnings_breakdown, mean_seconds,
max_seconds, min_seconds, strict_duplicated, scenarios (list), seeds (metadata).

Exit codes
----------
0: success, 1: failures present, 130: interrupted (partial summary written).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

ConfigTuple = Tuple[str, str, float, int, int, int]


SUMMARY_FILENAME = "reward_space_cli_smoke_results.json"


class ScenarioResult(TypedDict):
    config: ConfigTuple
    status: str
    stdout: str
    stderr: str
    strict: bool
    seconds: Optional[float]
    warnings: int


class SummaryResult(TypedDict):
    total: int
    ok: int
    failures: List[ScenarioResult]
    warnings_total: int
    mean_seconds: Optional[float]
    max_seconds: Optional[float]
    min_seconds: Optional[float]
    strict_duplicated: int


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
    idx_pos = 0.0
    selected: List[ConfigTuple] = []
    for _ in range(max_scenarios):
        idx = int(idx_pos)
        if idx >= len(full):
            idx = len(full) - 1
        selected.append(full[idx])
        idx_pos += step
    return selected


def run_scenario(
    script: Path,
    out_dir: Path,
    idx: int,
    base_samples: int,
    conf: ConfigTuple,
    strict: bool,
    bootstrap_resamples: int,
    timeout: int,
    skip_feature_analysis: bool = False,
    unrealized_pnl: bool = False,
    action_masking: bool = False,
    store_full_logs: bool = False,
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
        str(base_samples),
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
    if unrealized_pnl:
        cmd.append("--unrealized_pnl")
    if action_masking:
        cmd.append("--action_masking")
    if strict:
        cmd.append("--strict_diagnostics")
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
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    warn_count = combined.count("warning")
    stdout_out = proc.stdout if store_full_logs else proc.stdout[-5000:]
    stderr_out = proc.stderr if store_full_logs else proc.stderr[-5000:]
    return {
        "config": conf,
        "status": status,
        "stdout": stdout_out,
        "stderr": stderr_out,
        "strict": strict,
        "seconds": round(end - start, 4),
        "warnings": warn_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=int,
        default=40,
        help="Number of synthetic samples per scenario (minimum 4 for feature analysis)",
    )
    parser.add_argument(
        "--skip_feature_analysis",
        action="store_true",
        help="Skip feature importance and model-based analysis for all scenarios.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="sample_run_output_smoke",
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
        "--store_full_logs",
        action="store_true",
        help="If set, store full stdout/stderr (may be large) instead of tail truncation.",
    )
    parser.add_argument(
        "--unrealized_pnl",
        action="store_true",
        help="Forward --unrealized_pnl to child process to exercise hold Φ(s) path.",
    )
    parser.add_argument(
        "--action_masking",
        action="store_true",
        help="Forward --action_masking to child process to exercise coercion path.",
    )
    args = parser.parse_args()

    # Basic validation
    if args.max_scenarios <= 0:
        parser.error("--max_scenarios must be > 0")
    if args.samples < 4 and not args.skip_feature_analysis:
        parser.error("--samples must be >= 4 unless --skip_feature_analysis is set")
    if args.strict_sample < 0:
        parser.error("--strict_sample must be >= 0")
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap_resamples must be > 0")

    script = Path(__file__).parent / "reward_space_analysis.py"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_arg_matrix(max_scenarios=args.max_scenarios, shuffle_seed=args.shuffle_seed)

    # Prepare list of (conf, strict_flag)
    scenario_pairs: List[Tuple[ConfigTuple, bool]] = [(c, False) for c in scenarios]
    strict_n = max(0, min(args.strict_sample, len(scenarios)))
    for c in scenarios[:strict_n]:
        scenario_pairs.append((c, True))

    results: List[ScenarioResult] = []
    total = len(scenario_pairs)
    interrupted = False
    try:
        for i, (conf, strict_flag) in enumerate(scenario_pairs, start=1):
            # Ensure child process sees the chosen bootstrap resamples via direct CLI args only
            res = run_scenario(
                script=script,
                out_dir=out_dir,
                idx=i,
                base_samples=args.samples,
                conf=conf,
                strict=strict_flag,
                bootstrap_resamples=args.bootstrap_resamples,
                timeout=args.per_scenario_timeout,
                skip_feature_analysis=args.skip_feature_analysis,
                unrealized_pnl=args.unrealized_pnl,
                action_masking=args.action_masking,
                store_full_logs=args.store_full_logs,
            )
            results.append(res)
            status = res["status"]
            tag = "[strict]" if strict_flag else ""
            secs = res.get("seconds")
            secs_str = f" {secs:.2f}s" if secs is not None else ""
            print(f"[{i}/{total}] {conf} {tag} -> {status}{secs_str}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received: writing partial summary...")

    ok = sum(1 for r in results if r["status"] == "ok")
    failures = [r for r in results if r["status"] != "ok"]
    total_warnings = sum(r["warnings"] for r in results)
    durations: List[float] = [
        float(r["seconds"]) for r in results if isinstance(r["seconds"], float)
    ]
    summary: SummaryResult = {
        "total": len(results),
        "ok": ok,
        "failures": failures,
        "warnings_total": total_warnings,
        "mean_seconds": round(sum(durations) / len(durations), 4) if durations else None,
        "max_seconds": max(durations) if durations else None,
        "min_seconds": min(durations) if durations else None,
        "strict_duplicated": strict_n,
    }
    # Build warning breakdown (simple line fingerprinting)
    warning_counts: Dict[str, int] = {}
    for r in results:
        text = (r["stderr"] + "\n" + r["stdout"]).splitlines()
        for line in text:
            if "warning" in line.lower():
                # Fingerprint: trim + collapse whitespace + limit length
                fp = " ".join(line.strip().split())[:160]
                warning_counts[fp] = warning_counts.get(fp, 0) + 1

    # Scenario export (list of configs only, excluding strict flag duplication detail)
    scenario_list = [list(c) for c, _ in scenario_pairs]

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

    summary_extra: Dict[str, Any] = {
        "warnings_breakdown": warning_counts,
        "scenarios": scenario_list,
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
            "schema_version": 1,
            "per_scenario_timeout": args.per_scenario_timeout,
        },
    }
    serializable: Dict[str, Any]
    if interrupted:
        serializable = {**summary, **summary_extra, "interrupted": True}
    else:
        serializable = {**summary, **summary_extra}
    # Atomic write to avoid corrupt partial files
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="_tmp_summary_", dir=str(out_dir))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)
        os.replace(tmp_path, out_dir / SUMMARY_FILENAME)
    except Exception:
        # Best effort fallback
        try:
            Path(out_dir / SUMMARY_FILENAME).write_text(
                json.dumps(serializable, indent=2), encoding="utf-8"
            )
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    else:
        if os.path.exists(
            tmp_path
        ):  # Defensive cleanup: remove temp file if atomic replace did not clean up
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
