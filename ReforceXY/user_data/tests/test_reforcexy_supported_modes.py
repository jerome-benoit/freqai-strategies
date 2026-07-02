import importlib
import inspect
import io
import logging
import math
import sys
from collections.abc import Callable


sys.path.insert(0, "/freqtrade/user_data")
reforcexy_module = importlib.import_module("freqaimodels.ReforceXY")
ReforceXY = reforcexy_module.ReforceXY
MyRLEnv = reforcexy_module.MyRLEnv
convert_optuna_params_to_model_params = (
    reforcexy_module.convert_optuna_params_to_model_params
)


def make_env() -> MyRLEnv:
    env = object.__new__(MyRLEnv)
    env.id = "supported-modes-check"
    env._exit_potential_decay = 0.25
    return env


def capture_reforcexy_warnings(action: Callable[[], None]) -> str:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("freqaimodels.ReforceXY")
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        action()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return stream.getvalue()


def test_supported_mode_sets_are_immutable_constants() -> None:
    # Given: ReforceXY exposes ordered tuples for user-facing valid-value messages.
    expected_model_types = ("PPO", "RecurrentPPO", "MaskablePPO", "DQN", "QRDQN")
    expected_exit_modes = (
        "canonical",
        "non_canonical",
        "progressive_release",
        "spike_cancel",
        "retain_previous",
    )
    expected_transform_functions = (
        "tanh",
        "softsign",
        "arctan",
        "sigmoid",
        "asinh",
        "clip",
    )

    # When/Then: adjacent immutable sets mirror the tuples without reordering messages.
    assert ReforceXY._MODEL_TYPES == expected_model_types
    assert ReforceXY._MODEL_TYPES_SET == frozenset(expected_model_types)
    assert ReforceXY._EXIT_POTENTIAL_MODES == expected_exit_modes
    assert ReforceXY._EXIT_POTENTIAL_MODES_SET == frozenset(expected_exit_modes)
    assert ReforceXY._TRANSFORM_FUNCTIONS == expected_transform_functions
    assert ReforceXY._TRANSFORM_FUNCTIONS_SET == frozenset(expected_transform_functions)


def test_supported_mode_helpers_do_not_rebuild_sets() -> None:
    # Given: supported modes are cached on the class.
    source = inspect.getsource(ReforceXY)

    # When/Then: obsolete repeated builders and direct set construction are absent.
    assert "def _model_types_set" not in source
    assert "def _exit_potential_modes_set" not in source
    assert "def _transform_functions_set" not in source
    assert "set(ReforceXY._MODEL_TYPES)" not in source
    assert "set(ReforceXY._EXIT_POTENTIAL_MODES)" not in source
    assert "set(ReforceXY._TRANSFORM_FUNCTIONS)" not in source


def test_model_type_membership_and_invalid_message_are_unchanged() -> None:
    # Given: model type membership is backed by the same public tuple values.
    for model_type in ReforceXY._MODEL_TYPES:
        # When/Then: every public model type remains accepted by membership checks.
        assert model_type in ReforceXY._MODEL_TYPES_SET

    try:
        # When: an unsupported model type reaches Optuna parameter conversion.
        convert_optuna_params_to_model_params("UnsupportedModel", {"learning_rate": 0.1})
    except ValueError as exc:
        # Then: the existing public error text remains unchanged.
        assert str(exc) == "Hyperopt [global]: model type 'UnsupportedModel' not supported"
    else:
        raise AssertionError("unsupported model type did not raise ValueError")


def test_transform_function_membership_and_invalid_message_are_unchanged() -> None:
    # Given: a lightweight environment instance and all supported transforms.
    env = make_env()

    for transform_function in ReforceXY._TRANSFORM_FUNCTIONS:
        # When/Then: every public transform remains accepted by membership checks and execution.
        assert transform_function in ReforceXY._TRANSFORM_FUNCTIONS_SET
        result = env._potential_transform(transform_function, 0.5)
        assert math.isfinite(result)

    def use_invalid_transform() -> None:
        env._potential_transform("invalid", 0.5)

    # When: an unsupported transform reaches runtime validation.
    warnings = capture_reforcexy_warnings(use_invalid_transform)

    # Then: the existing public warning still lists the same valid values in tuple order.
    assert "potential_transform='invalid' invalid; defaulting to 'tanh'." in warnings
    assert "Valid: tanh, softsign, arctan, sigmoid, asinh, clip" in warnings


def test_exit_potential_mode_membership_and_behavior_are_unchanged() -> None:
    # Given: a lightweight environment instance and all supported exit potential modes.
    env = make_env()
    expected_values = {
        "canonical": 0.0,
        "non_canonical": 0.0,
        "progressive_release": 1.5,
        "spike_cancel": 4.0,
        "retain_previous": 2.0,
    }

    for exit_potential_mode, expected_value in expected_values.items():
        # When: each mode computes its exit potential.
        env._exit_potential_mode = exit_potential_mode
        result = env._compute_exit_potential(prev_potential=2.0, gamma=0.5)

        # Then: every public mode remains accepted and preserves behavior.
        assert exit_potential_mode in ReforceXY._EXIT_POTENTIAL_MODES_SET
        assert result == expected_value

    # Then: invalid public warning text still uses the same tuple-ordered valid values.
    assert ", ".join(ReforceXY._EXIT_POTENTIAL_MODES) == (
        "canonical, non_canonical, progressive_release, spike_cancel, retain_previous"
    )


if __name__ == "__main__":
    test_supported_mode_sets_are_immutable_constants()
    test_supported_mode_helpers_do_not_rebuild_sets()
    test_model_type_membership_and_invalid_message_are_unchanged()
    test_transform_function_membership_and_invalid_message_are_unchanged()
    test_exit_potential_mode_membership_and_behavior_are_unchanged()
    print("supported mode checks passed")
