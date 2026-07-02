import importlib
import inspect
import sys
import types

from optuna.samplers import TPESampler


sys.path.insert(0, "/freqtrade/user_data")
reforcexy_module = importlib.import_module("freqaimodels.ReforceXY")
ReforceXY = reforcexy_module.ReforceXY


class AutoSampler:
    def __init__(self, seed: int) -> None:
        self.seed = seed


def make_model(sampler: str) -> ReforceXY:
    model = object.__new__(ReforceXY)
    model.rl_config_optuna = {"sampler": sampler, "seed": 7}
    model.optuna_n_startup_trials = 3
    return model


def test_create_sampler_returns_tpe_when_sampler_is_tpe() -> None:
    # Given: a ReforceXY model configured for the public tpe sampler.
    model = make_model("tpe")

    # When: the sampler factory is invoked.
    sampler = model.create_sampler()

    # Then: the factory returns Optuna's TPE sampler.
    assert isinstance(sampler, TPESampler)


def test_create_sampler_returns_auto_when_sampler_is_auto() -> None:
    # Given: a ReforceXY model configured for the public auto sampler.
    model = make_model("auto")
    original_load_module = reforcexy_module.optunahub.load_module
    reforcexy_module.optunahub.load_module = lambda _: types.SimpleNamespace(
        AutoSampler=AutoSampler
    )

    try:
        # When: the sampler factory is invoked.
        sampler = model.create_sampler()
    finally:
        reforcexy_module.optunahub.load_module = original_load_module

    # Then: the factory returns the AutoSampler provided by optunahub.
    assert isinstance(sampler, AutoSampler)


def test_create_sampler_rejects_invalid_public_sampler() -> None:
    # Given: a ReforceXY model configured with an unsupported public sampler.
    model = make_model("invalid")

    try:
        # When: the sampler factory is invoked.
        model.create_sampler()
    except ValueError as exc:
        # Then: public invalid config fails before typed exhaustive dispatch.
        message = str(exc)
        assert "Hyperopt [global]: unsupported sampler 'invalid'." in message
        assert "Valid: tpe, auto" in message
    else:
        raise AssertionError("unsupported sampler did not raise ValueError")


def test_create_sampler_uses_exhaustive_match_dispatch() -> None:
    # Given: ReforceXY exposes a finite SamplerType surface.
    source = inspect.getsource(ReforceXY.create_sampler)

    # When/Then: dispatch is encoded as a match with assert_never for type-level exhaustiveness.
    assert "match sampler:" in source
    assert 'case "tpe":' in source
    assert 'case "auto":' in source
    assert "assert_never(sampler)" in source


if __name__ == "__main__":
    test_create_sampler_returns_tpe_when_sampler_is_tpe()
    test_create_sampler_returns_auto_when_sampler_is_auto()
    test_create_sampler_rejects_invalid_public_sampler()
    test_create_sampler_uses_exhaustive_match_dispatch()
    print("sampler dispatch checks passed")
