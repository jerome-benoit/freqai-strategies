"""Real Freqtrade archives shared by the strategy replay regressions."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from freqtrade.data import btanalysis
from freqtrade.enums import RunMode
from freqtrade.optimize.optimize_reports import store_backtest_results

HAS_ARCHIVED_STRATEGY_REPLAY = hasattr(btanalysis, "backtest_strategy_dir")


@contextmanager
def archived_strategy(
    config: dict[str, Any], source: Path
) -> Iterator[tuple[dict[str, Any], Path]]:
    """Export and extract the main source exactly as native backtesting does."""
    name = config["strategy"]
    output = config["user_data_dir"] / "backtest_results"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": "archive-replay-regression",
        "backtest_start_time": 1761955200,
        "backtest_start_ts": 1761955200,
        "backtest_end_ts": 1762041600,
        "timeframe": config["timeframe"],
    }
    archive = store_backtest_results(
        {"exportdirectory": output, "original_config": config},
        {"metadata": {name: metadata}, "strategy": {}, "strategy_comparison": []},
        "2026-10-31_00-00-00",
        strategy_files={name: str(source)},
    )
    with btanalysis.backtest_strategy_dir(archive, name) as directory:
        replay = deepcopy(config)
        replay["runmode"] = RunMode.WEBSERVER
        replay["strategy_path"] = str(directory)
        replay["timerange"] = f"{metadata['backtest_start_ts']}-{metadata['backtest_end_ts']}"
        yield replay, directory / f"{archive.stem}_{name}.py"
