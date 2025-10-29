"""Pytest configuration: fixtures and RNG setup.

Helper assertion wrappers live in `reward_space_analysis.tests.helpers`.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from reward_space_analysis import calculate_reward  # retained if some fixtures rely on it


@pytest.fixture(scope="session")
def temp_output_dir():
    """Temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def setup_rng():
    """Configure RNG for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def base_reward_params():
    """Default reward parameters."""
    from reward_space_analysis import DEFAULT_MODEL_REWARD_PARAMETERS

    return DEFAULT_MODEL_REWARD_PARAMETERS.copy()
