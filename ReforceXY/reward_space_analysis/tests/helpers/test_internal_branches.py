from reward_space_analysis import RewardParams, _get_bool_param


def test_get_bool_param_none_and_invalid_literal():
    """Verify _get_bool_param handles None and invalid literals correctly.

    Tests edge case handling in boolean parameter parsing:
    - None values should coerce to False
    - Invalid string literals should trigger fallback to default value

    **Setup:**
    - Test cases: None value, invalid literal "not_a_bool"
    - Default value: True

    **Assertions:**
    - None coerces to False (covers _to_bool None path)
    - Invalid literal returns default (ValueError fallback path)
    """
    params_none: RewardParams = {"check_invariants": None}
    # None should coerce to False (coverage for _to_bool None path)
    assert _get_bool_param(params_none, "check_invariants", True) is False

    params_invalid: RewardParams = {"check_invariants": "not_a_bool"}
    # Invalid literal triggers ValueError in _to_bool; fallback returns default (True)
    assert _get_bool_param(params_invalid, "check_invariants", True) is True
