import datetime
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from decimal import Decimal
from typing import Literal, Protocol, Self, final


TakeProfitAttemptStatus = Literal["proposed", "submitting", "ambiguous"]
TakeProfitOperatorAction = Literal[
    "confirmed_no_remote_order",
    "confirmed_terminal_canceled_fill",
    "confirmed_terminal_zero_fill",
    "revalidated_state",
    "revalidated_terminal_order",
]
TakeProfitRecoveryAction = Literal[
    "confirm_terminal_canceled_fill",
    "confirm_terminal_zero_fill",
    "revalidate_state",
    "revalidate_terminal_order",
]
TakeProfitTradeAccountingAction = Literal[
    "native_freqtrade_lifecycle",
    "none",
    "recalculate_from_orders",
]
TakeProfitAttributionSource = Literal[
    "attempt",
    "operator",
    "recovered_untagged",
]


_OPERATOR_CONFIRMABLE_CANCELED_STATUSES = frozenset({"canceled", "cancelled"})
_RECOVERABLE_TERMINAL_ORDER_BLOCK_REASONS = frozenset(
    {
        "attributed_fill_regressed",
        "attributed_order_amount_changed",
        "attributed_order_missing",
        "attributed_order_provenance_changed",
        "attributed_order_reopened",
        "attributed_order_side_changed",
        "attributed_order_tag_changed",
        "confirmed_terminal_canceled_fill_changed",
        "confirmed_terminal_canceled_fill_order_missing",
        "confirmed_terminal_zero_fill_changed",
        "confirmed_terminal_zero_fill_order_missing",
        "terminal_trade_amount_mismatch",
        "unattributed_exit_fill",
        "unknown_attempt_order",
        "unknown_terminal_fill",
    }
)
_REVALIDATABLE_STATE_BLOCK_REASONS = frozenset(
    {
        "attributed_fill_regressed",
        "attributed_order_amount_changed",
        "attributed_order_missing",
        "attributed_order_provenance_changed",
        "attributed_order_reopened",
        "attributed_order_side_changed",
        "attributed_order_tag_changed",
        "confirmed_terminal_canceled_fill_changed",
        "confirmed_terminal_canceled_fill_order_missing",
        "confirmed_terminal_zero_fill_changed",
        "confirmed_terminal_zero_fill_order_missing",
        "closed_trade_exposure_mismatch",
        "closed_trade_has_open_exit_order",
        "duplicate_order_id",
        "entry_exposure_changed",
        "initial_exposure_mismatch",
        "trade_amount_mismatch",
        "unknown_attempt_order",
    }
)
_BLOCKED_REASONS = frozenset(
    {
        "ambiguous_recovered_exit",
        "attributed_fill_regressed",
        "attributed_order_amount_changed",
        "attributed_order_missing",
        "attributed_order_provenance_changed",
        "attributed_order_reopened",
        "attributed_order_side_changed",
        "attributed_order_tag_changed",
        "closed_trade_exposure_mismatch",
        "closed_trade_has_open_exit_order",
        "confirmed_terminal_canceled_fill_changed",
        "confirmed_terminal_canceled_fill_order_missing",
        "confirmed_terminal_zero_fill_changed",
        "confirmed_terminal_zero_fill_order_missing",
        "credited_fill_exceeds_initial_amount",
        "duplicate_exposure_adjustment",
        "duplicate_order_id",
        "entry_exposure_changed",
        "exit_order_role_changed",
        "exposure_adjustment_limit",
        "external_exit_fill",
        "initial_exposure_mismatch",
        "missing_entry_exposure",
        "multiple_attempt_orders",
        "open_entry_order",
        "partial_wallet_mutation",
        "partial_wallet_order_changed",
        "terminal_trade_amount_mismatch",
        "trade_amount_mismatch",
        "unattributed_exit_fill",
        "unknown_attempt_order",
        "unknown_external_exit_fill",
        "unknown_terminal_fill",
        "zero_fill_retry_limit_reached",
    }
)
_BLOCKED_ORDER_ID_REASONS = frozenset(
    {
        "ambiguous_recovered_exit",
        "confirmed_terminal_zero_fill_changed",
        "confirmed_terminal_zero_fill_order_missing",
        "confirmed_terminal_canceled_fill_changed",
        "confirmed_terminal_canceled_fill_order_missing",
        "attributed_fill_regressed",
        "attributed_order_amount_changed",
        "attributed_order_missing",
        "attributed_order_provenance_changed",
        "attributed_order_reopened",
        "attributed_order_side_changed",
        "attributed_order_tag_changed",
        "closed_trade_has_open_exit_order",
        "external_exit_fill",
        "multiple_attempt_orders",
        "partial_wallet_order_changed",
        "terminal_trade_amount_mismatch",
        "unknown_attempt_order",
        "unknown_external_exit_fill",
        "unknown_terminal_fill",
        "unattributed_exit_fill",
        "zero_fill_retry_limit_reached",
    }
)


class _Order(Protocol):
    @property
    def filled(self) -> float | None: ...

    @property
    def ft_fee_base(self) -> float | None: ...

    @property
    def funding_fee(self) -> float | None: ...

    @property
    def order_id(self) -> str: ...

    @property
    def order_date_utc(self) -> datetime.datetime: ...

    @property
    def order_filled_utc(self) -> datetime.datetime | None: ...

    @property
    def ft_order_side(self) -> str: ...

    @property
    def ft_order_tag(self) -> str | None: ...

    @property
    def ft_is_open(self) -> bool: ...

    @property
    def remaining(self) -> float | None: ...

    @property
    def safe_amount(self) -> float: ...

    @property
    def safe_amount_after_fee(self) -> float: ...

    @property
    def safe_price(self) -> float | None: ...

    @property
    def status(self) -> str | None: ...


class _Trade(Protocol):
    @property
    def is_open(self) -> bool: ...

    @property
    def amount(self) -> float: ...

    @property
    def entry_side(self) -> str: ...

    @property
    def exit_side(self) -> str: ...

    @property
    def funding_fee_running(self) -> float | None: ...

    @property
    def orders(self) -> Sequence[_Order]: ...

    @property
    def trade_direction(self) -> str: ...


@dataclass(frozen=True, slots=True)
class TakeProfitRecoveryInstruction:
    """Canonical recovery disposition for the current blocked state."""

    action: TakeProfitRecoveryAction
    expected_trade_amount: float
    order_id: str | None = None
    terminal_status: str | None = None
    trade_accounting_action: TakeProfitTradeAccountingAction = "none"

    def __post_init__(self) -> None:
        _non_negative_float(self.expected_trade_amount)
        if self.trade_accounting_action not in {
            "native_freqtrade_lifecycle",
            "none",
            "recalculate_from_orders",
        }:
            raise ValueError("Recovery has an invalid trade-accounting action")
        if self.trade_accounting_action != "none" and self.order_id is None:
            raise ValueError("Trade-accounting recovery has no order ID")
        if (self.trade_accounting_action == "native_freqtrade_lifecycle") != (
            not self.expected_trade_is_open
        ):
            raise ValueError("Recovery has an inconsistent trade lifecycle")
        if self.action in {
            "confirm_terminal_canceled_fill",
            "confirm_terminal_zero_fill",
        }:
            _non_empty_string(self.order_id)
            if self.terminal_status not in _OPERATOR_CONFIRMABLE_CANCELED_STATUSES:
                raise ValueError("Terminal proof instruction has an invalid status")
        elif self.action == "revalidate_terminal_order":
            _non_empty_string(self.order_id)
            if self.terminal_status is not None:
                raise ValueError("Recovery instruction has unexpected terminal fields")
        elif self.action == "revalidate_state":
            if any(
                value is not None for value in (self.order_id, self.terminal_status)
            ):
                raise ValueError("State revalidation instruction has evidence fields")
            if self.trade_accounting_action != "none":
                raise ValueError("State revalidation cannot repair trade accounting")
        else:
            raise ValueError(f"Invalid take-profit recovery action {self.action!r}")

    @property
    def expected_trade_is_open(self) -> bool:
        return not math.isclose(
            self.expected_trade_amount,
            0.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )


@dataclass(frozen=True, slots=True)
class TakeProfitAttempt:
    attempt_id: str
    stage: int
    status: TakeProfitAttemptStatus
    amount: float
    stake_amount: float
    exit_side: str
    tag: str
    created_at: str
    submitted_at: str | None
    recovery_deadline: str
    known_order_ids: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "stage": self.stage,
            "status": self.status,
            "amount": self.amount,
            "stake_amount": self.stake_amount,
            "exit_side": self.exit_side,
            "tag": self.tag,
            "created_at": self.created_at,
            "submitted_at": self.submitted_at,
            "recovery_deadline": self.recovery_deadline,
            "known_order_ids": list(self.known_order_ids),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _require_exact_mapping_keys(value, cls, "take-profit attempt")
        status = value.get("status")
        if status not in {"proposed", "submitting", "ambiguous"}:
            raise ValueError(f"Invalid take-profit attempt status {status!r}")
        submitted_at = value.get("submitted_at")
        if submitted_at is not None and not isinstance(submitted_at, str):
            raise ValueError("Invalid take-profit attempt submitted_at")
        known_order_ids = value.get("known_order_ids")
        if not isinstance(known_order_ids, list) or not all(
            isinstance(order_id, str) and order_id for order_id in known_order_ids
        ):
            raise ValueError("Invalid take-profit attempt known_order_ids")
        if len(known_order_ids) != len(set(known_order_ids)):
            raise ValueError("Duplicate take-profit attempt known order")
        attempt_id = _required_string(value, "attempt_id")
        stage = _required_non_negative_int(value, "stage")
        amount = _required_non_negative_float(value, "amount")
        stake_amount = _required_non_negative_float(value, "stake_amount")
        tag = _required_string(value, "tag")
        created_at = _required_string(value, "created_at")
        recovery_deadline = _required_string(value, "recovery_deadline")
        created_time = _parse_iso_datetime(created_at)
        recovery_time = _parse_iso_datetime(recovery_deadline)
        submitted_time = (
            _parse_iso_datetime(submitted_at) if submitted_at is not None else None
        )
        if amount == 0.0 or stake_amount == 0.0:
            raise ValueError("Take-profit attempt amounts must be positive")
        if status in {"submitting", "ambiguous"} and submitted_time is None:
            raise ValueError("Submitted take-profit attempt has no timestamp")
        if submitted_time is not None and submitted_time < created_time:
            raise ValueError("Take-profit attempt was submitted before creation")
        if recovery_time < (submitted_time or created_time):
            raise ValueError("Take-profit recovery deadline precedes submission")
        parsed_tag = TakeProfitStageManager.parse_order_tag(tag)
        if parsed_tag is None or parsed_tag[1] != stage:
            raise ValueError("Take-profit attempt tag does not match its identity")

        return cls(
            attempt_id=attempt_id,
            stage=stage,
            status=status,
            amount=amount,
            stake_amount=stake_amount,
            exit_side=_required_string(value, "exit_side"),
            tag=tag,
            created_at=created_at,
            submitted_at=submitted_at,
            recovery_deadline=recovery_deadline,
            known_order_ids=tuple(known_order_ids),
        )


@dataclass(frozen=True, slots=True)
class TakeProfitRetryGate:
    """Persisted backoff after causally attributed terminal zero fills."""

    stage: int
    consecutive_zero_fill_count: int
    last_order_id: str
    retry_not_before: str

    def __post_init__(self) -> None:
        _non_negative_int(self.stage)
        if self.consecutive_zero_fill_count <= 0:
            raise ValueError("Take-profit zero-fill retry count must be positive")
        _non_empty_string(self.last_order_id)
        _parse_iso_datetime(self.retry_not_before)

    def to_mapping(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "consecutive_zero_fill_count": self.consecutive_zero_fill_count,
            "last_order_id": self.last_order_id,
            "retry_not_before": self.retry_not_before,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _require_exact_mapping_keys(value, cls, "take-profit retry gate")
        return cls(
            stage=_required_non_negative_int(value, "stage"),
            consecutive_zero_fill_count=_required_non_negative_int(
                value,
                "consecutive_zero_fill_count",
            ),
            last_order_id=_required_string(value, "last_order_id"),
            retry_not_before=_required_string(value, "retry_not_before"),
        )


@dataclass(frozen=True, slots=True)
class TakeProfitExposureAdjustment:
    """Audited correction for Freqtrade's full-exit wallet fallback."""

    adjustment_id: str
    previous_amount: float
    adjusted_amount: float
    amount: float
    exit_credit_baseline: float
    exit_reason: str
    recorded_at: str
    retired_at: str | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.adjustment_id)
        previous_amount = _required_positive_value(
            self.previous_amount,
            "exposure adjustment previous amount",
        )
        adjusted_amount = _required_positive_value(
            self.adjusted_amount,
            "exposure adjustment adjusted amount",
        )
        amount = _required_positive_value(
            self.amount,
            "exposure adjustment amount",
        )
        _non_negative_float(self.exit_credit_baseline)
        _non_empty_string(self.exit_reason)
        recorded_at = _parse_iso_datetime(self.recorded_at)
        if adjusted_amount >= previous_amount:
            raise ValueError("Exposure adjustment must reduce the trade amount")
        if not math.isclose(
            previous_amount - adjusted_amount,
            amount,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Exposure adjustment amount is inconsistent")
        if adjusted_amount <= previous_amount * 0.98:
            raise ValueError("Exposure adjustment exceeds Freqtrade's wallet fallback")
        if self.retired_at is not None:
            retired_at = _parse_iso_datetime(self.retired_at)
            if retired_at < recorded_at:
                raise ValueError(
                    "Exposure adjustment was retired before it was recorded"
                )

    @property
    def is_active(self) -> bool:
        return self.retired_at is None

    def to_mapping(self) -> dict[str, object]:
        return {
            "adjustment_id": self.adjustment_id,
            "previous_amount": self.previous_amount,
            "adjusted_amount": self.adjusted_amount,
            "amount": self.amount,
            "exit_credit_baseline": self.exit_credit_baseline,
            "exit_reason": self.exit_reason,
            "recorded_at": self.recorded_at,
            "retired_at": self.retired_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _require_exact_mapping_keys(value, cls, "take-profit exposure adjustment")
        retired_at = value.get("retired_at")
        if retired_at is not None and not isinstance(retired_at, str):
            raise ValueError("Invalid exposure adjustment retirement timestamp")
        return cls(
            adjustment_id=_required_string(value, "adjustment_id"),
            previous_amount=_required_positive_float(value, "previous_amount"),
            adjusted_amount=_required_positive_float(value, "adjusted_amount"),
            amount=_required_positive_float(value, "amount"),
            exit_credit_baseline=_required_non_negative_float(
                value,
                "exit_credit_baseline",
            ),
            exit_reason=_required_string(value, "exit_reason"),
            recorded_at=_required_string(value, "recorded_at"),
            retired_at=retired_at,
        )


@dataclass(frozen=True, slots=True)
class TakeProfitStageDefinition:
    stage: int
    trigger_natr_fraction: float
    exit_fraction_of_remaining: float

    def __post_init__(self) -> None:
        _non_negative_int(self.stage)
        if (
            not math.isfinite(self.trigger_natr_fraction)
            or self.trigger_natr_fraction <= 0
        ):
            raise ValueError("Take-profit trigger NATR fraction must be positive")
        if (
            not math.isfinite(self.exit_fraction_of_remaining)
            or not 0.0 < self.exit_fraction_of_remaining <= 1.0
        ):
            raise ValueError("Take-profit exit fraction must be in range (0, 1]")

    def to_mapping(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "trigger_natr_fraction": self.trigger_natr_fraction,
            "exit_fraction_of_remaining": self.exit_fraction_of_remaining,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _require_exact_mapping_keys(value, cls, "take-profit stage definition")
        return cls(
            stage=_required_non_negative_int(value, "stage"),
            trigger_natr_fraction=_required_positive_float(
                value,
                "trigger_natr_fraction",
            ),
            exit_fraction_of_remaining=_required_positive_float(
                value,
                "exit_fraction_of_remaining",
            ),
        )


@dataclass(frozen=True, slots=True)
class TakeProfitFinalStageDefinition:
    stage: int
    trigger_natr_fraction: float

    def __post_init__(self) -> None:
        _non_negative_int(self.stage)
        if (
            not math.isfinite(self.trigger_natr_fraction)
            or self.trigger_natr_fraction <= 0
        ):
            raise ValueError("Final take-profit trigger NATR fraction must be positive")

    def to_mapping(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "trigger_natr_fraction": self.trigger_natr_fraction,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _require_exact_mapping_keys(value, cls, "final take-profit stage definition")
        return cls(
            stage=_required_non_negative_int(value, "stage"),
            trigger_natr_fraction=_required_positive_float(
                value,
                "trigger_natr_fraction",
            ),
        )


@dataclass(frozen=True, slots=True)
class TakeProfitPlanSignature:
    partial_stages: tuple[TakeProfitStageDefinition, ...]
    final_stage: TakeProfitFinalStageDefinition

    def __post_init__(self) -> None:
        stages = tuple(stage.stage for stage in self.partial_stages)
        if stages != tuple(sorted(stages)) or len(stages) != len(set(stages)):
            raise ValueError("Take-profit partial stages must be unique and sorted")
        expected_final_stage = max(stages, default=-1) + 1
        if self.final_stage.stage != expected_final_stage:
            raise ValueError(
                f"Invalid final take-profit stage {self.final_stage.stage!r}; "
                f"expected {expected_final_stage}"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "partial_stages": [stage.to_mapping() for stage in self.partial_stages],
            "final_stage": self.final_stage.to_mapping(),
        }

    def derive_stage_targets(
        self,
        initial_amount: float,
    ) -> tuple[tuple[int, float], ...]:
        remaining_amount = _non_negative_float(initial_amount)
        targets: list[tuple[int, float]] = []
        for definition in self.partial_stages:
            target_amount = remaining_amount * definition.exit_fraction_of_remaining
            targets.append((definition.stage, target_amount))
            remaining_amount -= target_amount
        return tuple(targets)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _require_exact_mapping_keys(value, cls, "take-profit plan signature")
        partial_raw = value.get("partial_stages")
        final_raw = value.get("final_stage")
        if not isinstance(partial_raw, list) or not all(
            isinstance(stage, Mapping) for stage in partial_raw
        ):
            raise ValueError("Invalid take-profit partial stage plan")
        if not isinstance(final_raw, Mapping):
            raise ValueError("Invalid final take-profit stage plan")
        return cls(
            partial_stages=tuple(
                TakeProfitStageDefinition.from_mapping(stage)
                for stage in partial_raw
                if isinstance(stage, Mapping)
            ),
            final_stage=TakeProfitFinalStageDefinition.from_mapping(final_raw),
        )


@dataclass(frozen=True, slots=True)
class TakeProfitOperatorResolution:
    action: TakeProfitOperatorAction
    resolved_at: str
    attempt_id: str | None = None
    order_id: str | None = None
    credited_amount: float | None = None
    terminal_status: str | None = None
    cleared_block_reason: str | None = None
    cleared_block_order_id: str | None = None

    def __post_init__(self) -> None:
        _parse_iso_datetime(self.resolved_at)
        if self.action == "confirmed_no_remote_order":
            _non_empty_string(self.attempt_id)
            if (
                self.order_id is not None
                or self.credited_amount is not None
                or self.terminal_status is not None
            ):
                raise ValueError("Ambiguous-attempt resolution has order fields")
        elif self.action == "confirmed_terminal_zero_fill":
            _non_empty_string(self.order_id)
            if (
                self.attempt_id is not None
                or self.credited_amount != 0.0
                or self.terminal_status not in _OPERATOR_CONFIRMABLE_CANCELED_STATUSES
            ):
                raise ValueError("Terminal zero-fill resolution has invalid fields")
        elif self.action == "confirmed_terminal_canceled_fill":
            _non_empty_string(self.order_id)
            if (
                self.attempt_id is not None
                or self.credited_amount is None
                or self.terminal_status not in _OPERATOR_CONFIRMABLE_CANCELED_STATUSES
            ):
                raise ValueError("Terminal canceled-fill resolution has invalid fields")
            _required_positive_value(
                self.credited_amount,
                "terminal canceled-fill resolution credit",
            )
        elif self.action == "revalidated_terminal_order":
            _non_empty_string(self.order_id)
            if (
                self.attempt_id is not None
                or self.credited_amount is None
                or self.terminal_status is not None
            ):
                raise ValueError("Terminal-fill resolution has invalid fields")
            _non_negative_float(self.credited_amount)
        elif self.action == "revalidated_state":
            if self.cleared_block_reason not in _REVALIDATABLE_STATE_BLOCK_REASONS:
                raise ValueError("State revalidation has an invalid block reason")
            if any(
                value is not None
                for value in (
                    self.attempt_id,
                    self.order_id,
                    self.credited_amount,
                    self.terminal_status,
                )
            ):
                raise ValueError("State revalidation has evidence fields")
            if self.cleared_block_reason in _BLOCKED_ORDER_ID_REASONS:
                _non_empty_string(self.cleared_block_order_id)
            elif self.cleared_block_order_id is not None:
                raise ValueError("State revalidation has an unexpected order ID")
        else:
            raise ValueError(f"Invalid take-profit operator action {self.action!r}")
        if self.action != "revalidated_state" and (
            self.cleared_block_reason is not None
            or self.cleared_block_order_id is not None
        ):
            raise ValueError("Operator resolution has unexpected block evidence")

    def to_mapping(self) -> dict[str, object]:
        return {
            "action": self.action,
            "attempt_id": self.attempt_id,
            "order_id": self.order_id,
            "credited_amount": self.credited_amount,
            "terminal_status": self.terminal_status,
            "cleared_block_reason": self.cleared_block_reason,
            "cleared_block_order_id": self.cleared_block_order_id,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _require_exact_mapping_keys(value, cls, "take-profit operator resolution")
        action = value.get("action")
        if action not in {
            "confirmed_no_remote_order",
            "confirmed_terminal_canceled_fill",
            "confirmed_terminal_zero_fill",
            "revalidated_state",
            "revalidated_terminal_order",
        }:
            raise ValueError(f"Invalid take-profit operator action {action!r}")
        attempt_id = value.get("attempt_id")
        order_id = value.get("order_id")
        credited_amount = value.get("credited_amount")
        terminal_status = value.get("terminal_status")
        cleared_block_reason = value.get("cleared_block_reason")
        cleared_block_order_id = value.get("cleared_block_order_id")
        return cls(
            action=action,
            resolved_at=_required_string(value, "resolved_at"),
            attempt_id=(
                _non_empty_string(attempt_id) if attempt_id is not None else None
            ),
            order_id=_non_empty_string(order_id) if order_id is not None else None,
            credited_amount=(
                _non_negative_float(credited_amount)
                if credited_amount is not None
                else None
            ),
            terminal_status=(
                _non_empty_string(terminal_status)
                if terminal_status is not None
                else None
            ),
            cleared_block_reason=(
                _non_empty_string(cleared_block_reason)
                if cleared_block_reason is not None
                else None
            ),
            cleared_block_order_id=(
                _non_empty_string(cleared_block_order_id)
                if cleared_block_order_id is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class TakeProfitZeroFillProof:
    """Immutable canonical snapshot accepted by an explicit operator action."""

    order_id: str
    order_date: str
    order_side: str
    order_tag: str | None
    terminal_status: str
    amount: float
    remaining: float
    fee_base: float
    funding_fee: float
    confirmed_at: str

    def __post_init__(self) -> None:
        _non_empty_string(self.order_id)
        order_date = _parse_iso_datetime(self.order_date)
        _non_empty_string(self.order_side)
        if self.order_tag is not None:
            _non_empty_string(self.order_tag)
        if self.terminal_status not in _OPERATOR_CONFIRMABLE_CANCELED_STATUSES:
            raise ValueError("Invalid operator-confirmed zero-fill status")
        amount = _required_positive_value(self.amount, "zero-fill proof amount")
        remaining = _non_negative_float(self.remaining)
        if not math.isclose(remaining, amount, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Terminal zero-fill proof has an inconsistent remainder")
        if _non_negative_float(self.fee_base) != 0.0:
            raise ValueError("Terminal zero-fill proof has a base fee")
        if _finite_float(self.funding_fee) != 0.0:
            raise ValueError("Terminal zero-fill proof has a funding fee")
        confirmed_at = _parse_iso_datetime(self.confirmed_at)
        if confirmed_at < order_date:
            raise ValueError("Terminal zero fill was confirmed before order creation")

    def to_mapping(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "order_date": self.order_date,
            "order_side": self.order_side,
            "order_tag": self.order_tag,
            "terminal_status": self.terminal_status,
            "amount": self.amount,
            "remaining": self.remaining,
            "fee_base": self.fee_base,
            "funding_fee": self.funding_fee,
            "confirmed_at": self.confirmed_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _require_exact_mapping_keys(value, cls, "terminal zero-fill proof")
        order_tag = value.get("order_tag")
        if order_tag is not None and not isinstance(order_tag, str):
            raise ValueError("Invalid terminal zero-fill order tag")
        return cls(
            order_id=_required_string(value, "order_id"),
            order_date=_required_string(value, "order_date"),
            order_side=_required_string(value, "order_side"),
            order_tag=order_tag,
            terminal_status=_required_string(value, "terminal_status"),
            amount=_required_positive_float(value, "amount"),
            remaining=_required_non_negative_float(value, "remaining"),
            fee_base=_required_non_negative_float(value, "fee_base"),
            funding_fee=_required_finite_float(value, "funding_fee"),
            confirmed_at=_required_string(value, "confirmed_at"),
        )


@dataclass(frozen=True, slots=True)
class TakeProfitCanceledFillProof:
    """Exact canonical snapshot required for a canceled positive fill."""

    order_id: str
    order_date: str
    order_side: str
    order_tag: str | None
    terminal_status: str
    amount: float
    filled: float
    remaining: float
    fee_base: float
    execution_price: float
    funding_fee: float
    filled_at: str
    confirmed_at: str

    def __post_init__(self) -> None:
        _non_empty_string(self.order_id)
        order_date = _parse_iso_datetime(self.order_date)
        _non_empty_string(self.order_side)
        if self.order_tag is not None:
            _non_empty_string(self.order_tag)
        if self.terminal_status not in _OPERATOR_CONFIRMABLE_CANCELED_STATUSES:
            raise ValueError("Invalid operator-confirmed canceled-fill status")
        amount = _required_positive_value(self.amount, "canceled-fill proof amount")
        filled = _required_positive_value(self.filled, "canceled-fill proof fill")
        remaining = _non_negative_float(self.remaining)
        fee_base = _non_negative_float(self.fee_base)
        _required_positive_value(
            self.execution_price,
            "canceled-fill proof execution price",
        )
        _finite_float(self.funding_fee)
        if filled > amount and not math.isclose(
            filled,
            amount,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Terminal canceled-fill proof exceeds its amount")
        if not math.isclose(
            min(filled, amount) + remaining,
            amount,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Terminal canceled-fill proof has an inconsistent remainder"
            )
        if fee_base >= filled or math.isclose(
            fee_base,
            filled,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Terminal canceled-fill proof fee consumes its fill")
        filled_at = _parse_iso_datetime(self.filled_at)
        confirmed_at = _parse_iso_datetime(self.confirmed_at)
        if filled_at < order_date or confirmed_at < filled_at:
            raise ValueError("Terminal canceled fill has an invalid timeline")

    @property
    def credited_amount(self) -> float:
        return min(self.filled, self.amount) - self.fee_base

    def to_mapping(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "order_date": self.order_date,
            "order_side": self.order_side,
            "order_tag": self.order_tag,
            "terminal_status": self.terminal_status,
            "amount": self.amount,
            "filled": self.filled,
            "remaining": self.remaining,
            "fee_base": self.fee_base,
            "execution_price": self.execution_price,
            "funding_fee": self.funding_fee,
            "filled_at": self.filled_at,
            "confirmed_at": self.confirmed_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _require_exact_mapping_keys(value, cls, "terminal canceled-fill proof")
        order_tag = value.get("order_tag")
        if order_tag is not None and not isinstance(order_tag, str):
            raise ValueError("Invalid terminal canceled-fill order tag")
        return cls(
            order_id=_required_string(value, "order_id"),
            order_date=_required_string(value, "order_date"),
            order_side=_required_string(value, "order_side"),
            order_tag=order_tag,
            terminal_status=_required_string(value, "terminal_status"),
            amount=_required_positive_float(value, "amount"),
            filled=_required_positive_float(value, "filled"),
            remaining=_required_non_negative_float(value, "remaining"),
            fee_base=_required_non_negative_float(value, "fee_base"),
            execution_price=_required_positive_float(value, "execution_price"),
            funding_fee=_required_finite_float(value, "funding_fee"),
            filled_at=_required_string(value, "filled_at"),
            confirmed_at=_required_string(value, "confirmed_at"),
        )


@dataclass(frozen=True, slots=True)
class TakeProfitOrderAttribution:
    """Audit evidence binding one canonical order to take-profit execution."""

    order_id: str
    attribution_sequence: int
    tag: str | None
    side: str
    stage: int
    attempt_id: str | None
    order_date: str
    maximum_amount: float
    order_amount: float
    source: TakeProfitAttributionSource
    attempt_submitted_at: str | None = None
    retry_delay_seconds: float | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.order_id)
        _non_negative_int(self.attribution_sequence)
        if self.tag is not None:
            _non_empty_string(self.tag)
        _non_empty_string(self.side)
        _non_negative_int(self.stage)
        if self.attempt_id is not None:
            _non_empty_string(self.attempt_id)
        order_date = _parse_iso_datetime(self.order_date)
        maximum_amount = _required_positive_value(
            self.maximum_amount,
            "attribution maximum amount",
        )
        order_amount = _required_positive_value(
            self.order_amount,
            "attribution order amount",
        )
        if order_amount > maximum_amount and not math.isclose(
            order_amount,
            maximum_amount,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Attributed order exceeds its maximum amount")
        if self.source not in {
            "attempt",
            "operator",
            "recovered_untagged",
        }:
            raise ValueError("Invalid take-profit attribution source")
        if self.source == "attempt":
            _non_empty_string(self.attempt_id)
            parsed_tag = TakeProfitStageManager.parse_order_tag(self.tag)
            if parsed_tag is None or parsed_tag[1] != self.stage:
                raise ValueError("Attempt attribution has an invalid stable tag")
        elif self.source == "recovered_untagged":
            _non_empty_string(self.attempt_id)
            if self.tag is not None:
                raise ValueError("Recovered untagged attribution has invalid evidence")
        else:
            parsed_tag = TakeProfitStageManager.parse_order_tag(self.tag)
            if parsed_tag is None or parsed_tag[1] != self.stage:
                raise ValueError("Operator attribution has an invalid tag")
        if self.attempt_id is None:
            if (
                self.attempt_submitted_at is not None
                or self.retry_delay_seconds is not None
            ):
                raise ValueError("Non-attempt attribution has retry evidence")
        else:
            if self.attempt_submitted_at is None:
                raise ValueError("Attempt attribution has no submission timestamp")
            attempt_submitted_at = _parse_iso_datetime(self.attempt_submitted_at)
            if self.retry_delay_seconds is None:
                raise ValueError("Attempt attribution has no retry delay")
            retry_delay_seconds = _required_positive_value(
                self.retry_delay_seconds,
                "attribution retry delay",
            )
            if not (
                attempt_submitted_at
                <= order_date
                <= attempt_submitted_at
                + datetime.timedelta(seconds=retry_delay_seconds)
            ):
                raise ValueError("Attributed order is outside its submission window")

    def to_mapping(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "attribution_sequence": self.attribution_sequence,
            "tag": self.tag,
            "side": self.side,
            "stage": self.stage,
            "attempt_id": self.attempt_id,
            "order_date": self.order_date,
            "maximum_amount": self.maximum_amount,
            "order_amount": self.order_amount,
            "source": self.source,
            "attempt_submitted_at": self.attempt_submitted_at,
            "retry_delay_seconds": self.retry_delay_seconds,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _require_exact_mapping_keys(value, cls, "take-profit order attribution")
        source = value.get("source")
        if source not in {
            "attempt",
            "operator",
            "recovered_untagged",
        }:
            raise ValueError("Invalid take-profit attribution source")
        attempt_id = value.get("attempt_id")
        attempt_submitted_at = value.get("attempt_submitted_at")
        retry_delay_seconds = value.get("retry_delay_seconds")
        return cls(
            order_id=_required_string(value, "order_id"),
            attribution_sequence=_required_non_negative_int(
                value,
                "attribution_sequence",
            ),
            tag=(
                _non_empty_string(value.get("tag"))
                if value.get("tag") is not None
                else None
            ),
            side=_required_string(value, "side"),
            stage=_required_non_negative_int(value, "stage"),
            attempt_id=(
                _non_empty_string(attempt_id) if attempt_id is not None else None
            ),
            order_date=_required_string(value, "order_date"),
            maximum_amount=_required_positive_float(value, "maximum_amount"),
            order_amount=_required_positive_float(value, "order_amount"),
            source=source,
            attempt_submitted_at=(
                _non_empty_string(attempt_submitted_at)
                if attempt_submitted_at is not None
                else None
            ),
            retry_delay_seconds=(
                _required_positive_value(
                    retry_delay_seconds,
                    "attribution retry delay",
                )
                if retry_delay_seconds is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class TakeProfitPartialWalletMutation:
    """Fail-closed evidence of Freqtrade's partial-exit wallet mutation."""

    attempt_id: str
    order_id: str | None
    expected_trade_amount: float
    requested_exit_amount: float
    observed_trade_amount: float
    requested_exit_shortfall_amount: float
    attempt_submitted_at: str

    def __post_init__(self) -> None:
        _non_empty_string(self.attempt_id)
        if self.order_id is not None:
            _non_empty_string(self.order_id)
        expected = _required_positive_value(
            self.expected_trade_amount,
            "partial wallet mutation expected amount",
        )
        requested = _required_positive_value(
            self.requested_exit_amount,
            "partial wallet mutation requested amount",
        )
        observed = _required_positive_value(
            self.observed_trade_amount,
            "partial wallet mutation observed amount",
        )
        shortfall = _required_positive_value(
            self.requested_exit_shortfall_amount,
            "partial wallet mutation requested-exit shortfall",
        )
        if (
            expected < requested
            and not math.isclose(
                expected,
                requested,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ) or observed >= expected:
            raise ValueError("Partial wallet mutation is not a partial-exit mutation")
        if observed >= requested or observed <= requested * 0.98:
            raise ValueError("Partial wallet mutation is outside the fallback window")
        if not math.isclose(
            requested - observed,
            shortfall,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Partial wallet mutation shortfall is inconsistent")
        _parse_iso_datetime(self.attempt_submitted_at)

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "order_id": self.order_id,
            "expected_trade_amount": self.expected_trade_amount,
            "requested_exit_amount": self.requested_exit_amount,
            "observed_trade_amount": self.observed_trade_amount,
            "requested_exit_shortfall_amount": (self.requested_exit_shortfall_amount),
            "attempt_submitted_at": self.attempt_submitted_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _require_exact_mapping_keys(value, cls, "partial wallet mutation")
        order_id = value.get("order_id")
        return cls(
            attempt_id=_required_string(value, "attempt_id"),
            order_id=_non_empty_string(order_id) if order_id is not None else None,
            expected_trade_amount=_required_positive_float(
                value,
                "expected_trade_amount",
            ),
            requested_exit_amount=_required_positive_float(
                value,
                "requested_exit_amount",
            ),
            observed_trade_amount=_required_positive_float(
                value,
                "observed_trade_amount",
            ),
            requested_exit_shortfall_amount=_required_positive_float(
                value,
                "requested_exit_shortfall_amount",
            ),
            attempt_submitted_at=_required_string(value, "attempt_submitted_at"),
        )


@dataclass(frozen=True, slots=True)
class TakeProfitStageState:
    version: int
    plan_signature: TakeProfitPlanSignature
    initial_amount: float
    stage_targets: tuple[tuple[int, float], ...]
    deferred_stages: tuple[int, ...]
    order_attributions: tuple[TakeProfitOrderAttribution, ...]
    credited_order_amounts: tuple[tuple[str, float], ...]
    non_take_profit_exit_order_amounts: tuple[tuple[str, float], ...]
    entry_order_amounts: tuple[tuple[str, float], ...]
    exposure_adjustments: tuple[TakeProfitExposureAdjustment, ...]
    compacted_exposure_adjustment_count: int
    compacted_exposure_adjustment_amount: float
    confirmed_zero_fill_orders: tuple[TakeProfitZeroFillProof, ...]
    confirmed_canceled_fill_orders: tuple[TakeProfitCanceledFillProof, ...]
    operator_resolutions: tuple[TakeProfitOperatorResolution, ...]
    retry_gate: TakeProfitRetryGate | None = None
    partial_wallet_mutation: TakeProfitPartialWalletMutation | None = None
    active_attempt: TakeProfitAttempt | None = None
    blocked_reason: str | None = None
    blocked_order_id: str | None = None

    SCHEMA_VERSION = 1
    MAX_OPERATOR_RESOLUTIONS = 32
    MAX_EXPOSURE_ADJUSTMENTS = 32
    MAX_CONSECUTIVE_ZERO_FILL_OUTCOMES = 5

    def to_mapping(self) -> dict[str, object]:
        return {
            "version": self.version,
            "plan_signature": self.plan_signature.to_mapping(),
            "initial_amount": self.initial_amount,
            "stage_targets": [
                {"stage": stage, "amount": amount}
                for stage, amount in self.stage_targets
            ],
            "deferred_stages": list(self.deferred_stages),
            "order_attributions": [
                attribution.to_mapping() for attribution in self.order_attributions
            ],
            "credited_order_amounts": [
                {"order_id": order_id, "amount": amount}
                for order_id, amount in self.credited_order_amounts
            ],
            "non_take_profit_exit_order_amounts": [
                {"order_id": order_id, "amount": amount}
                for order_id, amount in self.non_take_profit_exit_order_amounts
            ],
            "entry_order_amounts": [
                {"order_id": order_id, "amount": amount}
                for order_id, amount in self.entry_order_amounts
            ],
            "exposure_adjustments": [
                adjustment.to_mapping() for adjustment in self.exposure_adjustments
            ],
            "compacted_exposure_adjustment_count": (
                self.compacted_exposure_adjustment_count
            ),
            "compacted_exposure_adjustment_amount": (
                self.compacted_exposure_adjustment_amount
            ),
            "confirmed_zero_fill_orders": [
                proof.to_mapping() for proof in self.confirmed_zero_fill_orders
            ],
            "confirmed_canceled_fill_orders": [
                proof.to_mapping() for proof in self.confirmed_canceled_fill_orders
            ],
            "operator_resolutions": [
                resolution.to_mapping() for resolution in self.operator_resolutions
            ],
            "retry_gate": (
                self.retry_gate.to_mapping() if self.retry_gate is not None else None
            ),
            "partial_wallet_mutation": (
                self.partial_wallet_mutation.to_mapping()
                if self.partial_wallet_mutation is not None
                else None
            ),
            "active_attempt": (
                self.active_attempt.to_mapping()
                if self.active_attempt is not None
                else None
            ),
            "blocked_reason": self.blocked_reason,
            "blocked_order_id": self.blocked_order_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        version = _required_non_negative_int(value, "version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported take-profit state version {version!r}; "
                f"expected {cls.SCHEMA_VERSION}"
            )
        _require_exact_mapping_keys(value, cls, "take-profit state")

        plan_raw = value.get("plan_signature")
        if not isinstance(plan_raw, Mapping):
            raise ValueError("Invalid take-profit plan signature")
        plan_signature = TakeProfitPlanSignature.from_mapping(plan_raw)
        initial_amount = _required_positive_float(value, "initial_amount")

        stage_targets = _parse_amount_records(
            value.get("stage_targets"),
            identifier_key="stage",
            identifier_parser=_non_negative_int,
        )
        stages = tuple(stage for stage, _ in stage_targets)
        if len(stages) != len(set(stages)):
            raise ValueError("Duplicate take-profit stage target")
        planned_stages = tuple(stage.stage for stage in plan_signature.partial_stages)
        if stages != planned_stages:
            raise ValueError("Take-profit targets do not match the persisted plan")
        expected_targets = plan_signature.derive_stage_targets(initial_amount)
        if any(
            stage != expected_stage
            or not math.isclose(
                amount,
                expected_amount,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            for (stage, amount), (expected_stage, expected_amount) in zip(
                stage_targets,
                expected_targets,
                strict=True,
            )
        ):
            raise ValueError("Take-profit targets do not match the persisted plan")

        deferred_raw = value.get("deferred_stages")
        if not isinstance(deferred_raw, list):
            raise ValueError("Invalid take-profit deferred_stages")
        deferred_stages = tuple(_non_negative_int(stage) for stage in deferred_raw)
        if len(deferred_stages) != len(set(deferred_stages)) or not set(
            deferred_stages
        ).issubset(stages):
            raise ValueError("Invalid take-profit deferred stage")

        attributions_raw = value.get("order_attributions")
        if not isinstance(attributions_raw, list) or not all(
            isinstance(attribution, Mapping) for attribution in attributions_raw
        ):
            raise ValueError("Invalid take-profit order attributions")
        order_attributions = tuple(
            TakeProfitOrderAttribution.from_mapping(attribution)
            for attribution in attributions_raw
            if isinstance(attribution, Mapping)
        )
        attribution_order_ids = tuple(
            attribution.order_id for attribution in order_attributions
        )
        if len(attribution_order_ids) != len(set(attribution_order_ids)):
            raise ValueError("Duplicate attributed take-profit order")
        attribution_sequences = tuple(
            attribution.attribution_sequence for attribution in order_attributions
        )
        if sorted(attribution_sequences) != list(range(len(order_attributions))):
            raise ValueError("Invalid take-profit attribution sequence")
        attribution_attempt_ids = tuple(
            attribution.attempt_id
            for attribution in order_attributions
            if attribution.attempt_id is not None
        )
        if len(attribution_attempt_ids) != len(set(attribution_attempt_ids)):
            raise ValueError("Take-profit attempt is attributed to multiple orders")
        all_stages = {
            *planned_stages,
            plan_signature.final_stage.stage,
        }
        if any(
            attribution.stage not in all_stages for attribution in order_attributions
        ):
            raise ValueError("Attributed order has an invalid take-profit stage")

        credited_order_amounts = _parse_amount_records(
            value.get("credited_order_amounts"),
            identifier_key="order_id",
            identifier_parser=_non_empty_string,
        )
        if not {order_id for order_id, _ in credited_order_amounts}.issubset(
            attribution_order_ids
        ):
            raise ValueError("Credited take-profit order is not attributed")
        total_credited = math.fsum(amount for _, amount in credited_order_amounts)
        if total_credited > initial_amount and not math.isclose(
            total_credited,
            initial_amount,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Credited take-profit amount exceeds the initial amount")

        non_take_profit_exit_order_amounts = _parse_amount_records(
            value.get("non_take_profit_exit_order_amounts"),
            identifier_key="order_id",
            identifier_parser=_non_empty_string,
        )
        take_profit_order_ids = set(attribution_order_ids)
        non_take_profit_order_ids = {
            order_id for order_id, _ in non_take_profit_exit_order_amounts
        }
        if take_profit_order_ids & non_take_profit_order_ids:
            raise ValueError(
                "Exit order cannot be both take-profit and non-take-profit"
            )

        entry_order_amounts = _parse_amount_records(
            value.get("entry_order_amounts"),
            identifier_key="order_id",
            identifier_parser=_non_empty_string,
        )

        adjustments_raw = value.get("exposure_adjustments")
        if not isinstance(adjustments_raw, list) or not all(
            isinstance(adjustment, Mapping) for adjustment in adjustments_raw
        ):
            raise ValueError("Invalid take-profit exposure adjustments")
        exposure_adjustments = tuple(
            TakeProfitExposureAdjustment.from_mapping(adjustment)
            for adjustment in adjustments_raw
            if isinstance(adjustment, Mapping)
        )
        adjustment_ids = tuple(
            adjustment.adjustment_id for adjustment in exposure_adjustments
        )
        if len(adjustment_ids) != len(set(adjustment_ids)):
            raise ValueError("Duplicate take-profit exposure adjustment")
        if len(exposure_adjustments) > cls.MAX_EXPOSURE_ADJUSTMENTS:
            raise ValueError("Too many take-profit exposure adjustments")
        compacted_exposure_adjustment_count = _required_non_negative_int(
            value,
            "compacted_exposure_adjustment_count",
        )
        compacted_exposure_adjustment_amount = _required_non_negative_float(
            value,
            "compacted_exposure_adjustment_amount",
        )
        if (compacted_exposure_adjustment_count == 0) != (
            compacted_exposure_adjustment_amount == 0.0
        ):
            raise ValueError("Invalid compacted exposure adjustment aggregate")
        total_exit_credited = total_credited + math.fsum(
            amount for _, amount in non_take_profit_exit_order_amounts
        )
        if total_exit_credited > initial_amount and not math.isclose(
            total_exit_credited,
            initial_amount,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Credited exit amount exceeds the initial amount")
        active_adjustment_total = math.fsum(
            adjustment.amount
            for adjustment in exposure_adjustments
            if adjustment.is_active
        )
        if (
            total_exit_credited + active_adjustment_total > initial_amount
            and not math.isclose(
                total_exit_credited + active_adjustment_total,
                initial_amount,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Exposure credits exceed the initial amount")
        if any(
            adjustment.is_active
            and adjustment.exit_credit_baseline > total_exit_credited
            and not math.isclose(
                adjustment.exit_credit_baseline,
                total_exit_credited,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            for adjustment in exposure_adjustments
        ):
            raise ValueError("Active exposure adjustment has an invalid fill baseline")

        proofs_raw = value.get("confirmed_zero_fill_orders")
        if not isinstance(proofs_raw, list) or not all(
            isinstance(proof, Mapping) for proof in proofs_raw
        ):
            raise ValueError("Invalid take-profit confirmed zero-fill orders")
        confirmed_zero_fill_orders = tuple(
            TakeProfitZeroFillProof.from_mapping(proof)
            for proof in proofs_raw
            if isinstance(proof, Mapping)
        )
        proof_order_ids = tuple(proof.order_id for proof in confirmed_zero_fill_orders)
        if len(proof_order_ids) != len(set(proof_order_ids)):
            raise ValueError("Duplicate terminal zero-fill proof")

        canceled_proofs_raw = value.get("confirmed_canceled_fill_orders")
        if not isinstance(canceled_proofs_raw, list) or not all(
            isinstance(proof, Mapping) for proof in canceled_proofs_raw
        ):
            raise ValueError("Invalid take-profit confirmed canceled-fill orders")
        confirmed_canceled_fill_orders = tuple(
            TakeProfitCanceledFillProof.from_mapping(proof)
            for proof in canceled_proofs_raw
            if isinstance(proof, Mapping)
        )
        canceled_proof_order_ids = tuple(
            proof.order_id for proof in confirmed_canceled_fill_orders
        )
        if len(canceled_proof_order_ids) != len(set(canceled_proof_order_ids)):
            raise ValueError("Duplicate terminal canceled-fill proof")
        if set(canceled_proof_order_ids) & set(proof_order_ids):
            raise ValueError("Terminal order has conflicting operator proofs")

        resolutions_raw = value.get("operator_resolutions")
        if not isinstance(resolutions_raw, list) or not all(
            isinstance(resolution, Mapping) for resolution in resolutions_raw
        ):
            raise ValueError("Invalid take-profit operator resolutions")
        if len(resolutions_raw) > cls.MAX_OPERATOR_RESOLUTIONS:
            raise ValueError("Too many take-profit operator resolutions")
        operator_resolutions = tuple(
            TakeProfitOperatorResolution.from_mapping(resolution)
            for resolution in resolutions_raw
            if isinstance(resolution, Mapping)
        )
        resolved_attempt_ids = tuple(
            resolution.attempt_id
            for resolution in operator_resolutions
            if resolution.action == "confirmed_no_remote_order"
        )
        if len(resolved_attempt_ids) != len(set(resolved_attempt_ids)):
            raise ValueError("Duplicate take-profit operator resolution")

        retry_gate_raw = value.get("retry_gate")
        if retry_gate_raw is not None and not isinstance(retry_gate_raw, Mapping):
            raise ValueError("Invalid take-profit retry gate")
        retry_gate = (
            TakeProfitRetryGate.from_mapping(retry_gate_raw)
            if isinstance(retry_gate_raw, Mapping)
            else None
        )
        if retry_gate is not None:
            credited_amounts_by_order_id = dict(credited_order_amounts)
            retry_attribution = next(
                (
                    attribution
                    for attribution in order_attributions
                    if attribution.order_id == retry_gate.last_order_id
                ),
                None,
            )
            retry_credit = credited_amounts_by_order_id.get(retry_gate.last_order_id)
            if (
                retry_attribution is None
                or retry_attribution.stage != retry_gate.stage
                or retry_attribution.attempt_id is None
                or (retry_credit is not None and retry_credit != 0.0)
            ):
                raise ValueError(
                    "Take-profit retry gate has no matching zero-fill order"
                )
            if (
                retry_gate.consecutive_zero_fill_count
                > cls.MAX_CONSECUTIVE_ZERO_FILL_OUTCOMES
            ):
                raise ValueError("Take-profit retry gate exceeds its safety budget")
            suffix_attribution, suffix_count = (
                TakeProfitStageManager._derive_retry_suffix(
                    order_attributions,
                    credited_amounts_by_order_id,
                )
            )
            if retry_credit is None:
                credited_sequences = (
                    attribution.attribution_sequence
                    for attribution in order_attributions
                    if attribution.order_id in credited_amounts_by_order_id
                )
                if max(credited_sequences, default=-1) >= (
                    retry_attribution.attribution_sequence
                ):
                    raise ValueError("Uncredited retry outcome is not causally latest")
                expected_count = (
                    suffix_count + 1
                    if suffix_attribution is not None
                    and suffix_attribution.stage == retry_attribution.stage
                    else 1
                )
            else:
                if (
                    suffix_attribution is None
                    or suffix_attribution.order_id != retry_gate.last_order_id
                ):
                    raise ValueError(
                        "Take-profit retry gate is not the causal zero-fill suffix"
                    )
                expected_count = suffix_count
            if retry_gate.consecutive_zero_fill_count != expected_count:
                raise ValueError(
                    "Take-profit retry count does not match its causal suffix"
                )
            if (
                retry_attribution.attempt_submitted_at is None
                or retry_attribution.retry_delay_seconds is None
            ):
                raise ValueError("Take-profit retry gate has no timing evidence")
            minimum_retry_time = max(
                _parse_iso_datetime(retry_attribution.attempt_submitted_at),
                _parse_iso_datetime(retry_attribution.order_date),
            ) + datetime.timedelta(
                seconds=(
                    retry_attribution.retry_delay_seconds
                    * TakeProfitStageManager._retry_backoff_multiplier(expected_count)
                )
            )
            if _parse_iso_datetime(retry_gate.retry_not_before) < minimum_retry_time:
                raise ValueError(
                    "Take-profit retry deadline precedes its causal backoff"
                )
        active_raw = value.get("active_attempt")
        if active_raw is not None and not isinstance(active_raw, Mapping):
            raise ValueError("Invalid take-profit active_attempt")
        active_attempt = (
            TakeProfitAttempt.from_mapping(active_raw)
            if isinstance(active_raw, Mapping)
            else None
        )
        if active_attempt is not None and active_attempt.stage not in stages:
            if active_attempt.stage != plan_signature.final_stage.stage:
                raise ValueError("Invalid take-profit attempt stage")
        if (
            active_attempt is not None
            and active_attempt.status == "proposed"
            and active_attempt.stage != plan_signature.final_stage.stage
        ):
            raise ValueError("Only a final take-profit attempt can be proposed")
        if (
            active_attempt is not None
            and active_attempt.attempt_id in resolved_attempt_ids
        ):
            raise ValueError("Resolved take-profit attempt cannot remain active")
        blocked_reason = value.get("blocked_reason")
        if blocked_reason is not None and (
            not isinstance(blocked_reason, str)
            or blocked_reason not in _BLOCKED_REASONS
        ):
            raise ValueError("Invalid take-profit blocked_reason")
        blocked_order_id = value.get("blocked_order_id")
        if blocked_reason in _BLOCKED_ORDER_ID_REASONS:
            blocked_order_id = _non_empty_string(blocked_order_id)
        elif blocked_reason is None and blocked_order_id is not None:
            raise ValueError("Take-profit blocked_order_id requires a blocked state")
        elif blocked_order_id is not None:
            raise ValueError(
                "Take-profit blocked_order_id is invalid for its blocked state"
            )
        if blocked_reason == "zero_fill_retry_limit_reached":
            if (
                retry_gate is None
                or retry_gate.consecutive_zero_fill_count
                < cls.MAX_CONSECUTIVE_ZERO_FILL_OUTCOMES
                or blocked_order_id != retry_gate.last_order_id
            ):
                raise ValueError("Zero-fill retry limit has no matching retry gate")

        partial_wallet_raw = value.get("partial_wallet_mutation")
        if partial_wallet_raw is not None and not isinstance(
            partial_wallet_raw,
            Mapping,
        ):
            raise ValueError("Invalid take-profit partial wallet mutation")
        partial_wallet_mutation = (
            TakeProfitPartialWalletMutation.from_mapping(partial_wallet_raw)
            if isinstance(partial_wallet_raw, Mapping)
            else None
        )
        if partial_wallet_mutation is not None:
            if partial_wallet_mutation.order_id is None:
                if (
                    active_attempt is None
                    or active_attempt.attempt_id != partial_wallet_mutation.attempt_id
                    or active_attempt.status != "ambiguous"
                ):
                    raise ValueError("Unbound partial wallet attempt is not ambiguous")
                if blocked_reason is None:
                    raise ValueError("Unbound partial wallet mutation must be blocked")
                attempt_amount_matches = math.isclose(
                    partial_wallet_mutation.requested_exit_amount,
                    active_attempt.amount,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ) or (
                    math.isclose(
                        partial_wallet_mutation.observed_trade_amount,
                        active_attempt.amount,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        partial_wallet_mutation.expected_trade_amount,
                        partial_wallet_mutation.requested_exit_amount,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                )
                if not attempt_amount_matches or (
                    partial_wallet_mutation.attempt_submitted_at
                    != (active_attempt.submitted_at or active_attempt.created_at)
                ):
                    raise ValueError(
                        "Partial wallet mutation does not match its attempt"
                    )
            elif (
                blocked_reason == "partial_wallet_order_changed"
                and blocked_order_id != partial_wallet_mutation.order_id
            ):
                raise ValueError(
                    "Bound partial wallet mutation has invalid order block"
                )
            else:
                if active_attempt is not None:
                    raise ValueError(
                        "Bound partial wallet mutation has an active attempt"
                    )
                mutation_attribution = next(
                    (
                        attribution
                        for attribution in order_attributions
                        if attribution.order_id == partial_wallet_mutation.order_id
                    ),
                    None,
                )
                if (
                    mutation_attribution is None
                    or mutation_attribution.attempt_id
                    != partial_wallet_mutation.attempt_id
                    or not math.isclose(
                        mutation_attribution.maximum_amount,
                        partial_wallet_mutation.requested_exit_amount,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        "Bound partial wallet mutation has no matching attribution"
                    )
        elif blocked_reason == "partial_wallet_mutation":
            raise ValueError("Partial wallet block has no mutation evidence")

        active_attributions = (
            tuple(
                attribution
                for attribution in order_attributions
                if active_attempt is not None
                and attribution.attempt_id == active_attempt.attempt_id
            )
            if active_attempt is not None
            else ()
        )
        if active_attributions:
            raise ValueError("Attributed take-profit attempt cannot remain active")

        parsed_state = cls(
            version=cls.SCHEMA_VERSION,
            plan_signature=plan_signature,
            initial_amount=initial_amount,
            stage_targets=tuple(sorted(stage_targets)),
            deferred_stages=tuple(sorted(deferred_stages)),
            order_attributions=tuple(
                sorted(
                    order_attributions,
                    key=lambda attribution: attribution.order_id,
                )
            ),
            credited_order_amounts=tuple(sorted(credited_order_amounts)),
            non_take_profit_exit_order_amounts=tuple(
                sorted(non_take_profit_exit_order_amounts)
            ),
            entry_order_amounts=tuple(sorted(entry_order_amounts)),
            exposure_adjustments=exposure_adjustments,
            compacted_exposure_adjustment_count=(compacted_exposure_adjustment_count),
            compacted_exposure_adjustment_amount=(compacted_exposure_adjustment_amount),
            confirmed_zero_fill_orders=tuple(
                sorted(
                    confirmed_zero_fill_orders,
                    key=lambda proof: proof.order_id,
                )
            ),
            confirmed_canceled_fill_orders=tuple(
                sorted(
                    confirmed_canceled_fill_orders,
                    key=lambda proof: proof.order_id,
                )
            ),
            operator_resolutions=operator_resolutions,
            retry_gate=retry_gate,
            partial_wallet_mutation=partial_wallet_mutation,
            active_attempt=active_attempt,
            blocked_reason=blocked_reason,
            blocked_order_id=blocked_order_id,
        )
        if active_attempt is not None:
            TakeProfitStageManager._validate_attempt_bounds(
                parsed_state,
                active_attempt,
            )
        return parsed_state


@dataclass(frozen=True, slots=True)
class TakeProfitStageProgress:
    stage: int
    target_amount: float
    filled_amount: float
    remaining_amount: float
    is_complete: bool
    is_deferred: bool
    has_deferred_debt: bool


@final
class TakeProfitStageManager:
    """Reconcile take-profit stages with causally attributed Freqtrade orders."""

    STATE_KEY = "take_profit_state"
    _TAG_PREFIX = "take_profit"
    _AMOUNT_REL_TOL = 1e-9
    _AMOUNT_ABS_TOL = 1e-12
    _TERMINAL_STATUSES = frozenset(
        {"canceled", "cancelled", "closed", "expired", "rejected"}
    )
    _ZERO_FILL_RETRY_STATUSES = frozenset({"expired", "rejected"})

    @classmethod
    def order_tag(
        cls,
        trade_direction: str,
        stage: int,
    ) -> str:
        if stage < 0:
            raise ValueError(
                f"Invalid take-profit stage {stage!r}: must be non-negative"
            )
        return f"{cls._TAG_PREFIX}_{trade_direction}_{stage}"

    @classmethod
    def has_order_tag_prefix(cls, value: object) -> bool:
        return isinstance(value, str) and value.startswith(f"{cls._TAG_PREFIX}_")

    @classmethod
    def parse_order_tag(
        cls,
        tag: object,
    ) -> tuple[str, int] | None:
        if not isinstance(tag, str):
            return None
        prefix = f"{cls._TAG_PREFIX}_"
        if not tag.startswith(prefix):
            return None
        parts = tag.removeprefix(prefix).split("_")
        if len(parts) != 2:
            return None
        try:
            stage = int(parts[1])
        except ValueError:
            return None
        if stage < 0 or not parts[0]:
            return None
        return parts[0], stage

    @classmethod
    def initialize(
        cls,
        trade: _Trade,
        plan_signature: TakeProfitPlanSignature,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
        validate_trade_amount: bool = True,
        current_time: datetime.datetime | None = None,
    ) -> TakeProfitStageState:
        entry_order_amounts = cls._entry_order_amounts(trade)
        if entry_order_amounts:
            initial_amount = cls._canonical_amount_sum(
                entry_order_amounts.values(),
                amount_quantizer,
            )
            blocked_reason = None
        else:
            blocked_reason = "missing_entry_exposure"
            initial_amount = cls._canonical_amount(
                cls._safe_amount(trade.amount),
                amount_quantizer,
            )
        if initial_amount <= 0.0:
            raise ValueError("Cannot initialize take-profit without positive exposure")
        if any(
            order.ft_order_side == trade.entry_side and order.ft_is_open
            for order in trade.orders
        ):
            blocked_reason = "open_entry_order"

        state = TakeProfitStageState(
            version=TakeProfitStageState.SCHEMA_VERSION,
            plan_signature=plan_signature,
            initial_amount=initial_amount,
            stage_targets=plan_signature.derive_stage_targets(initial_amount),
            deferred_stages=(),
            order_attributions=(),
            credited_order_amounts=(),
            non_take_profit_exit_order_amounts=(),
            entry_order_amounts=tuple(sorted(entry_order_amounts.items())),
            exposure_adjustments=(),
            compacted_exposure_adjustment_count=0,
            compacted_exposure_adjustment_amount=0.0,
            confirmed_zero_fill_orders=(),
            confirmed_canceled_fill_orders=(),
            operator_resolutions=(),
            retry_gate=None,
            blocked_reason=blocked_reason,
        )
        return (
            state
            if state.blocked_reason is not None
            else cls._reconcile(
                trade,
                state,
                amount_quantizer=amount_quantizer,
                validate_trade_amount=validate_trade_amount,
                current_time=current_time,
            )
        )

    @classmethod
    def reconcile(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
        current_time: datetime.datetime | None = None,
    ) -> TakeProfitStageState:
        return cls._reconcile(
            trade,
            state,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=True,
            current_time=current_time,
        )

    @classmethod
    def _reconcile(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        *,
        amount_quantizer: Callable[[float], float] | None,
        validate_trade_amount: bool,
        current_time: datetime.datetime | None,
        allow_canonical_adjustment_retirement: bool = False,
    ) -> TakeProfitStageState:
        if current_time is not None and current_time.tzinfo is None:
            raise ValueError(
                "Take-profit reconciliation timestamp must be timezone-aware"
            )
        if state.blocked_reason is not None:
            mutation = state.partial_wallet_mutation
            can_refresh_unbound_wallet_mutation = (
                state.blocked_reason == "partial_wallet_mutation"
                and mutation is not None
                and mutation.order_id is None
            )
            can_refresh_retry_limit = (
                state.blocked_reason == "zero_fill_retry_limit_reached"
            )
            if not (can_refresh_unbound_wallet_mutation or can_refresh_retry_limit):
                return state
            state = replace(state, blocked_reason=None, blocked_order_id=None)

        orders_by_id: dict[str, _Order] = {}
        for order in trade.orders:
            if order.order_id in orders_by_id:
                return cls._block(state, "duplicate_order_id")
            orders_by_id[order.order_id] = order

        confirmed_zero_fill_orders = {
            proof.order_id: proof for proof in state.confirmed_zero_fill_orders
        }
        for proof in state.confirmed_zero_fill_orders:
            order = orders_by_id.get(proof.order_id)
            if order is None:
                return cls._block(
                    state,
                    "confirmed_terminal_zero_fill_order_missing",
                    order_id=proof.order_id,
                )
            if not cls._zero_fill_proof_matches(trade, order, proof):
                return cls._block(
                    state,
                    "confirmed_terminal_zero_fill_changed",
                    order_id=proof.order_id,
                )
        confirmed_canceled_fill_orders = {
            proof.order_id: proof for proof in state.confirmed_canceled_fill_orders
        }
        for proof in state.confirmed_canceled_fill_orders:
            order = orders_by_id.get(proof.order_id)
            if order is None:
                return cls._block(
                    state,
                    "confirmed_terminal_canceled_fill_order_missing",
                    order_id=proof.order_id,
                )
            if not cls._canceled_fill_proof_matches(trade, order, proof):
                return cls._block(
                    state,
                    "confirmed_terminal_canceled_fill_changed",
                    order_id=proof.order_id,
                )

        if any(
            order.ft_order_side == trade.entry_side and order.ft_is_open
            for order in trade.orders
        ):
            return cls._block(state, "entry_exposure_changed")
        current_entry_amounts = cls._entry_order_amounts(trade)
        persisted_entry_amounts = dict(state.entry_order_amounts)
        if current_entry_amounts.keys() != persisted_entry_amounts.keys() or any(
            not cls._amounts_close(amount, persisted_entry_amounts[order_id])
            for order_id, amount in current_entry_amounts.items()
        ):
            return cls._block(state, "entry_exposure_changed")
        try:
            canonical_initial_amount = cls._canonical_amount_sum(
                persisted_entry_amounts.values(),
                amount_quantizer,
            )
        except ValueError:
            return cls._block(state, "initial_exposure_mismatch")
        if not cls._amounts_close(canonical_initial_amount, state.initial_amount):
            return cls._block(state, "initial_exposure_mismatch")

        configured_stages = {
            *(stage for stage, _ in state.stage_targets),
            state.plan_signature.final_stage.stage,
        }
        order_attributions = {
            attribution.order_id: attribution
            for attribution in state.order_attributions
        }
        credited_order_amounts = dict(state.credited_order_amounts)
        non_take_profit_exit_order_amounts = dict(
            state.non_take_profit_exit_order_amounts
        )
        strategy_exit_orders = [
            order for order in trade.orders if order.ft_order_side == trade.exit_side
        ]
        for order in trade.orders:
            if order.ft_order_side in {trade.entry_side, trade.exit_side}:
                continue
            external_block = cls._external_exit_block(
                order,
            )
            if external_block is not None:
                return cls._block(
                    state,
                    external_block,
                    order_id=order.order_id,
                )

        for attribution in order_attributions.values():
            order_id = attribution.order_id
            order = orders_by_id.get(order_id)
            if order is None:
                return cls._block(
                    state,
                    "attributed_order_missing",
                    order_id=order_id,
                )
            if (
                attribution.side != trade.exit_side
                or order.ft_order_side != attribution.side
            ):
                return cls._block(
                    state,
                    "attributed_order_side_changed",
                    order_id=order_id,
                )
            if order.ft_order_tag != attribution.tag:
                return cls._block(
                    state,
                    "attributed_order_tag_changed",
                    order_id=order_id,
                )
            order_amount = cls._safe_amount(order.safe_amount)
            if order_amount > attribution.maximum_amount and not cls._amounts_close(
                order_amount,
                attribution.maximum_amount,
            ):
                return cls._block(
                    state,
                    "attributed_order_amount_changed",
                    order_id=order_id,
                )
            reconciled_attribution = cls._reconcile_attribution_order_amount(
                attribution,
                order_amount,
                amount_quantizer,
            )
            if reconciled_attribution is None:
                return cls._block(
                    state,
                    "attributed_order_amount_changed",
                    order_id=order_id,
                )
            attribution = reconciled_attribution
            order_attributions[order_id] = attribution
            if not cls._attribution_provenance_matches(
                trade,
                attribution,
                order,
                configured_stages,
            ):
                return cls._block(
                    state,
                    "attributed_order_provenance_changed",
                    order_id=order_id,
                )

        active_attempt = state.active_attempt
        partial_wallet_mutation = state.partial_wallet_mutation
        matching_attempt_orders: list[_Order] = []
        retry_gate = state.retry_gate
        if active_attempt is not None and active_attempt.status in {
            "submitting",
            "ambiguous",
        }:
            matching_attempt_orders = [
                order
                for order in strategy_exit_orders
                if order.order_id not in order_attributions
                and cls._is_recovered_attempt_candidate(
                    order,
                    active_attempt,
                    amount_quantizer,
                )
            ]
            if len(matching_attempt_orders) > 1:
                block_reason = (
                    "ambiguous_recovered_exit"
                    if all(
                        order.ft_order_tag is None for order in matching_attempt_orders
                    )
                    else "multiple_attempt_orders"
                )
                return cls._block(
                    state,
                    block_reason,
                    order_id=matching_attempt_orders[0].order_id,
                )
            if matching_attempt_orders:
                order = matching_attempt_orders[0]
                source: TakeProfitAttributionSource = (
                    "recovered_untagged" if order.ft_order_tag is None else "attempt"
                )
                order_attributions[order.order_id] = TakeProfitOrderAttribution(
                    order_id=order.order_id,
                    attribution_sequence=(
                        max(
                            (
                                attribution.attribution_sequence
                                for attribution in order_attributions.values()
                            ),
                            default=-1,
                        )
                        + 1
                    ),
                    tag=order.ft_order_tag,
                    side=order.ft_order_side,
                    stage=active_attempt.stage,
                    attempt_id=active_attempt.attempt_id,
                    order_date=order.order_date_utc.isoformat(),
                    maximum_amount=(
                        partial_wallet_mutation.requested_exit_amount
                        if partial_wallet_mutation is not None
                        and partial_wallet_mutation.attempt_id
                        == active_attempt.attempt_id
                        else active_attempt.amount
                    ),
                    order_amount=cls._safe_amount(order.safe_amount),
                    source=source,
                    attempt_submitted_at=(
                        active_attempt.submitted_at or active_attempt.created_at
                    ),
                    retry_delay_seconds=cls._attempt_retry_delay_seconds(
                        active_attempt
                    ),
                )
                if partial_wallet_mutation is not None:
                    if partial_wallet_mutation.order_id not in {
                        None,
                        order.order_id,
                    }:
                        causal_candidate = replace(
                            state,
                            order_attributions=tuple(
                                sorted(
                                    order_attributions.values(),
                                    key=lambda attribution: attribution.order_id,
                                )
                            ),
                            active_attempt=None,
                        )
                        return cls._block(
                            causal_candidate,
                            "partial_wallet_order_changed",
                            order_id=order.order_id,
                        )
                    partial_wallet_mutation = replace(
                        partial_wallet_mutation,
                        order_id=order.order_id,
                    )
                try:
                    expected_amount = cls.get_expected_trade_amount(
                        state,
                        amount_quantizer=amount_quantizer,
                    )
                    observed_amount = cls._canonical_amount(
                        trade.amount,
                        amount_quantizer,
                    )
                except ValueError:
                    causal_candidate = replace(
                        state,
                        order_attributions=tuple(
                            sorted(
                                order_attributions.values(),
                                key=lambda attribution: attribution.order_id,
                            )
                        ),
                        partial_wallet_mutation=partial_wallet_mutation,
                        active_attempt=None,
                    )
                    return cls._block(causal_candidate, "trade_amount_mismatch")
                if partial_wallet_mutation is None and cls._is_partial_wallet_mutation(
                    active_attempt,
                    expected_amount,
                    observed_amount,
                ):
                    partial_wallet_mutation = TakeProfitPartialWalletMutation(
                        attempt_id=active_attempt.attempt_id,
                        order_id=order.order_id,
                        expected_trade_amount=expected_amount,
                        requested_exit_amount=active_attempt.amount,
                        observed_trade_amount=observed_amount,
                        requested_exit_shortfall_amount=(
                            active_attempt.amount - observed_amount
                        ),
                        attempt_submitted_at=(
                            active_attempt.submitted_at or active_attempt.created_at
                        ),
                    )
                active_attempt = None

        for attribution in order_attributions.values():
            order = orders_by_id[attribution.order_id]
            if (
                attribution.attempt_id is None
                or order.ft_is_open
                or order.order_id in credited_order_amounts
            ):
                continue
            try:
                cls._validate_terminal_zero_fill(
                    order,
                    cls._ZERO_FILL_RETRY_STATUSES
                    | _OPERATOR_CONFIRMABLE_CANCELED_STATUSES,
                )
            except ValueError:
                continue
            retry_gate = cls._record_zero_fill_outcome(
                retry_gate,
                attribution,
                order,
                current_time=current_time,
            )

        candidate = replace(
            state,
            order_attributions=tuple(
                sorted(
                    order_attributions.values(),
                    key=lambda attribution: attribution.order_id,
                )
            ),
            retry_gate=retry_gate,
            partial_wallet_mutation=partial_wallet_mutation,
            active_attempt=active_attempt,
        )

        terminal_credits: dict[str, float] = {}
        for order in strategy_exit_orders:
            if order.ft_is_open:
                continue
            try:
                terminal_credits[order.order_id] = cls._terminal_credited_amount(
                    order,
                    confirmed_zero_fill_orders.get(order.order_id),
                    confirmed_canceled_fill_orders.get(order.order_id),
                )
            except ValueError:
                if partial_wallet_mutation is not None:
                    return cls._block(candidate, "partial_wallet_mutation")
                return cls._block(
                    candidate,
                    "unknown_terminal_fill",
                    order_id=order.order_id,
                )

        attributed_order_ids = set(order_attributions)
        matching_attempt_order_ids = {
            order.order_id for order in matching_attempt_orders
        }
        for order in strategy_exit_orders:
            if order.order_id in attributed_order_ids or (
                order.order_id in matching_attempt_order_ids
            ):
                continue
            if cls.has_order_tag_prefix(order.ft_order_tag):
                return cls._block(
                    candidate,
                    "unknown_attempt_order",
                    order_id=order.order_id,
                )

        for order_id in attributed_order_ids:
            order = orders_by_id[order_id]
            previous_amount = credited_order_amounts.get(order_id)
            if order.ft_is_open:
                if previous_amount is not None:
                    return cls._block(
                        candidate,
                        "attributed_order_reopened",
                        order_id=order_id,
                    )
                continue
            current_amount = terminal_credits[order_id]
            if (
                previous_amount is not None
                and current_amount < previous_amount
                and not (cls._amounts_close(current_amount, previous_amount))
            ):
                return cls._block(
                    candidate,
                    "attributed_fill_regressed",
                    order_id=order_id,
                )
            credited_order_amounts[order_id] = current_amount

        for order_id, previous_amount in non_take_profit_exit_order_amounts.items():
            order = orders_by_id.get(order_id)
            if order is None:
                return cls._block(
                    candidate,
                    "attributed_order_missing",
                    order_id=order_id,
                )
            if order.ft_order_side != trade.exit_side:
                return cls._block(
                    candidate,
                    "attributed_order_side_changed",
                    order_id=order_id,
                )
            if order.ft_is_open:
                return cls._block(
                    candidate,
                    "attributed_order_reopened",
                    order_id=order_id,
                )
            current_amount = terminal_credits[order_id]
            if current_amount < previous_amount and not cls._amounts_close(
                current_amount,
                previous_amount,
            ):
                return cls._block(
                    candidate,
                    "attributed_fill_regressed",
                    order_id=order_id,
                )
            non_take_profit_exit_order_amounts[order_id] = current_amount

        for order in strategy_exit_orders:
            if order.ft_is_open or order.order_id in attributed_order_ids:
                continue
            if order.order_id in credited_order_amounts:
                return cls._block(candidate, "exit_order_role_changed")
            current_amount = terminal_credits[order.order_id]
            if current_amount > 0.0 and not order.ft_order_tag:
                return cls._block(
                    candidate,
                    "unattributed_exit_fill",
                    order_id=order.order_id,
                )
            previous_amount = non_take_profit_exit_order_amounts.get(order.order_id)
            if (
                previous_amount is not None
                and current_amount < previous_amount
                and not cls._amounts_close(current_amount, previous_amount)
            ):
                return cls._block(
                    candidate,
                    "attributed_fill_regressed",
                    order_id=order.order_id,
                )
            non_take_profit_exit_order_amounts[order.order_id] = current_amount

        if set(credited_order_amounts) & set(non_take_profit_exit_order_amounts):
            return cls._block(candidate, "exit_order_role_changed")

        previous_credited_order_amounts = dict(state.credited_order_amounts)
        increased_positive_order_ids = {
            order_id
            for order_id, amount in credited_order_amounts.items()
            if amount > 0.0
            and amount > previous_credited_order_amounts.get(order_id, 0.0)
            and not cls._amounts_close(
                amount,
                previous_credited_order_amounts.get(order_id, 0.0),
            )
        }
        retry_gate = cls._reconcile_retry_gate(
            retry_gate,
            order_attributions.values(),
            credited_order_amounts,
            current_time=current_time,
        )
        candidate = replace(
            candidate,
            credited_order_amounts=tuple(sorted(credited_order_amounts.items())),
            non_take_profit_exit_order_amounts=tuple(
                sorted(non_take_profit_exit_order_amounts.items())
            ),
            retry_gate=retry_gate,
        )

        total_take_profit_credited = math.fsum(credited_order_amounts.values())
        total_exit_credited = total_take_profit_credited + math.fsum(
            non_take_profit_exit_order_amounts.values()
        )
        if total_exit_credited > state.initial_amount and not cls._amounts_close(
            total_exit_credited,
            state.initial_amount,
        ):
            return cls._block(candidate, "credited_fill_exceeds_initial_amount")

        exposure_adjustments = state.exposure_adjustments
        has_open_exit_order = any(
            order.ft_is_open
            for order in trade.orders
            if order.ft_order_side != trade.entry_side
        )
        if not trade.is_open and has_open_exit_order:
            open_exit_order_id = min(
                order.order_id
                for order in trade.orders
                if order.ft_order_side != trade.entry_side and order.ft_is_open
            )
            return cls._block(
                candidate,
                "closed_trade_has_open_exit_order",
                order_id=open_exit_order_id,
            )
        if validate_trade_amount and not has_open_exit_order:
            exit_credit_amounts = (
                *credited_order_amounts.values(),
                *non_take_profit_exit_order_amounts.values(),
            )
            active_adjustments = tuple(
                adjustment
                for adjustment in exposure_adjustments
                if adjustment.is_active
            )
            active_adjustment_total = math.fsum(
                adjustment.amount for adjustment in active_adjustments
            )
            if (
                total_exit_credited + active_adjustment_total > state.initial_amount
                and not cls._amounts_close(
                    total_exit_credited + active_adjustment_total,
                    state.initial_amount,
                )
            ):
                return cls._block(candidate, "credited_fill_exceeds_initial_amount")
            expected_with_adjustments = cls._canonical_expected_trade_amount(
                persisted_entry_amounts.values(),
                exit_credit_amounts,
                (adjustment.amount for adjustment in active_adjustments),
                amount_quantizer,
            )
            if trade.is_open:
                try:
                    observed_amount = cls._canonical_amount(
                        trade.amount,
                        amount_quantizer,
                    )
                except ValueError:
                    return cls._block(candidate, "trade_amount_mismatch")
            else:
                observed_amount = 0.0
                if not cls._amounts_close(expected_with_adjustments, 0.0):
                    return cls._block(candidate, "closed_trade_exposure_mismatch")
            if trade.is_open and cls._amounts_close(expected_with_adjustments, 0.0):
                if len(increased_positive_order_ids) == 1:
                    return cls._block(
                        candidate,
                        "terminal_trade_amount_mismatch",
                        order_id=next(iter(increased_positive_order_ids)),
                    )
                return cls._block(candidate, "trade_amount_mismatch")
            expected_without_adjustments = cls._canonical_expected_trade_amount(
                persisted_entry_amounts.values(),
                exit_credit_amounts,
                (),
                amount_quantizer,
            )
            if (
                trade.is_open
                and allow_canonical_adjustment_retirement
                and active_adjustments
                and cls._amounts_close(
                    observed_amount,
                    expected_without_adjustments,
                )
            ):
                retireable_adjustments = active_adjustments
                adjustments_were_retired = True
            else:
                retireable_adjustments = tuple(
                    adjustment
                    for adjustment in active_adjustments
                    if total_exit_credited > adjustment.exit_credit_baseline
                )
                adjustments_were_retired = False
            if retireable_adjustments and not adjustments_were_retired:
                expected_after_retirement = cls._canonical_expected_trade_amount(
                    persisted_entry_amounts.values(),
                    exit_credit_amounts,
                    (
                        adjustment.amount
                        for adjustment in active_adjustments
                        if adjustment not in retireable_adjustments
                    ),
                    amount_quantizer,
                )
                adjustments_were_retired = cls._amounts_close(
                    observed_amount,
                    expected_after_retirement,
                )
            if adjustments_were_retired:
                retirement_time = max(
                    (
                        *((current_time,) if current_time is not None else ()),
                        *(
                            order.order_filled_utc
                            for order in strategy_exit_orders
                            if not order.ft_is_open
                            and terminal_credits.get(order.order_id, 0.0) > 0.0
                            and order.order_filled_utc is not None
                        ),
                        *(
                            _parse_iso_datetime(adjustment.recorded_at)
                            for adjustment in retireable_adjustments
                        ),
                    )
                ).isoformat()
                retireable_ids = {
                    adjustment.adjustment_id for adjustment in retireable_adjustments
                }
                exposure_adjustments = cls._retire_exposure_adjustments(
                    exposure_adjustments,
                    retireable_ids,
                    retirement_time,
                )
            elif not cls._amounts_close(
                observed_amount,
                expected_with_adjustments,
            ):
                if (
                    partial_wallet_mutation is not None
                    and partial_wallet_mutation.order_id is not None
                ):
                    return cls._block(candidate, "partial_wallet_mutation")
                if partial_wallet_mutation is not None or (
                    cls._is_partial_wallet_mutation(
                        active_attempt,
                        expected_with_adjustments,
                        observed_amount,
                    )
                ):
                    if active_attempt is None:
                        raise AssertionError("Validated mutation has no active attempt")
                    partial_wallet_mutation = (
                        partial_wallet_mutation
                        or TakeProfitPartialWalletMutation(
                            attempt_id=active_attempt.attempt_id,
                            order_id=None,
                            expected_trade_amount=expected_with_adjustments,
                            requested_exit_amount=active_attempt.amount,
                            observed_trade_amount=observed_amount,
                            requested_exit_shortfall_amount=(
                                active_attempt.amount - observed_amount
                            ),
                            attempt_submitted_at=(
                                active_attempt.submitted_at or active_attempt.created_at
                            ),
                        )
                    )
                    return cls._block(
                        replace(
                            candidate,
                            active_attempt=replace(
                                active_attempt,
                                status="ambiguous",
                            ),
                            partial_wallet_mutation=partial_wallet_mutation,
                        ),
                        "partial_wallet_mutation",
                    )
                if len(increased_positive_order_ids) == 1:
                    order_id = next(iter(increased_positive_order_ids))
                    return cls._block(
                        candidate,
                        "terminal_trade_amount_mismatch",
                        order_id=order_id,
                    )
                return cls._block(candidate, "trade_amount_mismatch")

        (
            exposure_adjustments,
            compacted_adjustment_count,
            compacted_adjustment_amount,
        ) = cls._compact_exposure_adjustments(
            exposure_adjustments,
            state.compacted_exposure_adjustment_count,
            state.compacted_exposure_adjustment_amount,
        )

        reconciled = replace(
            candidate,
            exposure_adjustments=exposure_adjustments,
            compacted_exposure_adjustment_count=compacted_adjustment_count,
            compacted_exposure_adjustment_amount=compacted_adjustment_amount,
            partial_wallet_mutation=partial_wallet_mutation,
            active_attempt=active_attempt,
        )
        if (
            reconciled.partial_wallet_mutation is not None
            and not has_open_exit_order
            and validate_trade_amount
        ):
            return cls._block(reconciled, "partial_wallet_mutation")
        if (
            reconciled.retry_gate is not None
            and reconciled.retry_gate.consecutive_zero_fill_count
            >= TakeProfitStageState.MAX_CONSECUTIVE_ZERO_FILL_OUTCOMES
        ):
            return cls._block(
                reconciled,
                "zero_fill_retry_limit_reached",
                order_id=reconciled.retry_gate.last_order_id,
            )
        return reconciled

    @classmethod
    def reconcile_for_exit_confirmation(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        *,
        confirmed_amount: float,
        exit_reason: str,
        adjustment_id: str,
        current_time: datetime.datetime,
        allow_wallet_adjustment: bool,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> TakeProfitStageState:
        """Reconcile the amount mutation made by Freqtrade before full confirmation."""
        if current_time.tzinfo is None:
            raise ValueError("Exit confirmation timestamp must be timezone-aware")
        _non_empty_string(exit_reason)
        _non_empty_string(adjustment_id)
        confirmed_amount = _required_positive_value(
            confirmed_amount,
            "confirmed exit amount",
        )
        observed_amount = _required_positive_value(
            trade.amount,
            "observed trade amount",
        )
        if not cls._raw_and_canonical_amounts_match(
            confirmed_amount,
            observed_amount,
            amount_quantizer,
        ):
            return cls._block(state, "trade_amount_mismatch")

        candidate = cls._reconcile(
            trade,
            state,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=False,
            current_time=current_time,
        )
        if candidate.blocked_reason is not None:
            return candidate

        total_exit_credited = math.fsum(
            amount for _, amount in candidate.credited_order_amounts
        ) + math.fsum(
            amount for _, amount in candidate.non_take_profit_exit_order_amounts
        )
        retireable_adjustments = tuple(
            adjustment
            for adjustment in candidate.exposure_adjustments
            if adjustment.is_active
            and total_exit_credited > adjustment.exit_credit_baseline
        )
        if retireable_adjustments:
            retirement_time = max(
                current_time,
                *(
                    _parse_iso_datetime(adjustment.recorded_at)
                    for adjustment in retireable_adjustments
                ),
            ).isoformat()
            candidate = replace(
                candidate,
                exposure_adjustments=cls._retire_exposure_adjustments(
                    candidate.exposure_adjustments,
                    {adjustment.adjustment_id for adjustment in retireable_adjustments},
                    retirement_time,
                ),
            )
        active_adjustment_amounts = (
            adjustment.amount
            for adjustment in candidate.exposure_adjustments
            if adjustment.is_active
        )
        expected_amount = cls._raw_expected_trade_amount(
            (amount for _, amount in candidate.entry_order_amounts),
            (
                *(amount for _, amount in candidate.credited_order_amounts),
                *(amount for _, amount in candidate.non_take_profit_exit_order_amounts),
            ),
            active_adjustment_amounts,
            amount_quantizer,
        )
        if expected_amount < 0.0:
            return cls._block(candidate, "trade_amount_mismatch")
        if cls._raw_and_canonical_amounts_match(
            observed_amount,
            expected_amount,
            amount_quantizer,
        ):
            return cls.reconcile(
                trade,
                candidate,
                amount_quantizer=amount_quantizer,
            )

        if (
            not allow_wallet_adjustment
            or observed_amount >= expected_amount
            or observed_amount <= expected_amount * 0.98
        ):
            return cls._block(candidate, "trade_amount_mismatch")
        if any(
            adjustment.adjustment_id == adjustment_id
            for adjustment in candidate.exposure_adjustments
        ):
            return cls._block(candidate, "duplicate_exposure_adjustment")
        (
            compacted_adjustments,
            compacted_adjustment_count,
            compacted_adjustment_amount,
        ) = cls._compact_exposure_adjustments(
            candidate.exposure_adjustments,
            candidate.compacted_exposure_adjustment_count,
            candidate.compacted_exposure_adjustment_amount,
            required_slots=1,
        )
        candidate = replace(
            candidate,
            exposure_adjustments=compacted_adjustments,
            compacted_exposure_adjustment_count=compacted_adjustment_count,
            compacted_exposure_adjustment_amount=compacted_adjustment_amount,
        )
        if (
            len(candidate.exposure_adjustments)
            >= TakeProfitStageState.MAX_EXPOSURE_ADJUSTMENTS
        ):
            return cls._block(candidate, "exposure_adjustment_limit")

        adjustment = TakeProfitExposureAdjustment(
            adjustment_id=adjustment_id,
            previous_amount=expected_amount,
            adjusted_amount=observed_amount,
            amount=expected_amount - observed_amount,
            exit_credit_baseline=total_exit_credited,
            exit_reason=exit_reason,
            recorded_at=current_time.isoformat(),
        )
        adjusted_state = replace(
            candidate,
            exposure_adjustments=(
                *candidate.exposure_adjustments,
                adjustment,
            ),
        )
        return cls.reconcile(
            trade,
            adjusted_state,
            amount_quantizer=amount_quantizer,
        )

    @classmethod
    def get_stage_progress(
        cls,
        state: TakeProfitStageState,
        stage: int,
    ) -> TakeProfitStageProgress:
        targets = dict(state.stage_targets)
        if stage not in targets:
            raise ValueError(f"Unknown take-profit stage {stage!r}")
        ordered_stages = [
            configured_stage for configured_stage, _ in state.stage_targets
        ]
        stage_index = ordered_stages.index(stage)
        cumulative_target_before = math.fsum(
            targets[configured_stage]
            for configured_stage in ordered_stages[:stage_index]
        )
        cumulative_target = cumulative_target_before + targets[stage]
        cumulative_filled = math.fsum(
            amount for _, amount in state.credited_order_amounts
        )
        filled_amount = min(
            targets[stage],
            max(0.0, cumulative_filled - cumulative_target_before),
        )
        remaining_amount = max(0.0, cumulative_target - cumulative_filled)
        is_filled = cumulative_filled >= cumulative_target or cls._amounts_close(
            cumulative_filled,
            cumulative_target,
        )
        deferred_stages = set(state.deferred_stages)
        return TakeProfitStageProgress(
            stage=stage,
            target_amount=targets[stage],
            filled_amount=filled_amount,
            remaining_amount=0.0 if is_filled else remaining_amount,
            is_complete=is_filled,
            is_deferred=stage in deferred_stages,
            has_deferred_debt=any(
                deferred_stage <= stage for deferred_stage in deferred_stages
            )
            and remaining_amount > 0.0,
        )

    @classmethod
    def get_execution_stage(
        cls,
        state: TakeProfitStageState,
    ) -> int:
        """Return the next scheduled stage, skipping explicitly deferred work."""
        return cls._get_stage(state, skip_deferred=True)

    @classmethod
    def get_credited_stage(
        cls,
        state: TakeProfitStageState,
    ) -> int:
        """Return the stage reached exclusively through attributed CEX fills."""
        return cls._get_stage(state, skip_deferred=False)

    @classmethod
    def _get_stage(
        cls,
        state: TakeProfitStageState,
        *,
        skip_deferred: bool,
    ) -> int:
        cumulative_filled = math.fsum(
            amount for _, amount in state.credited_order_amounts
        )
        cumulative_target = 0.0
        deferred_stages = set(state.deferred_stages) if skip_deferred else set()
        for stage, target in state.stage_targets:
            cumulative_target += target
            is_filled = cumulative_filled >= cumulative_target or cls._amounts_close(
                cumulative_filled,
                cumulative_target,
            )
            if stage in deferred_stages or is_filled:
                continue
            return stage
        return state.plan_signature.final_stage.stage

    @classmethod
    def get_expected_trade_amount(
        cls,
        state: TakeProfitStageState,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> float:
        return cls._canonical_expected_trade_amount(
            (amount for _, amount in state.entry_order_amounts),
            (
                *(amount for _, amount in state.credited_order_amounts),
                *(amount for _, amount in state.non_take_profit_exit_order_amounts),
            ),
            (
                adjustment.amount
                for adjustment in state.exposure_adjustments
                if adjustment.is_active
            ),
            amount_quantizer,
        )

    @classmethod
    def get_trade_amount_delta(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> float:
        expected = cls.get_expected_trade_amount(
            state,
            amount_quantizer=amount_quantizer,
        )
        observed = (
            cls._canonical_amount(trade.amount, amount_quantizer)
            if trade.is_open
            else 0.0
        )
        return observed - expected

    @staticmethod
    def defer_stage(
        state: TakeProfitStageState,
        stage: int,
    ) -> TakeProfitStageState:
        if stage not in dict(state.stage_targets):
            raise ValueError(f"Unknown take-profit stage {stage!r}")
        return replace(
            state,
            deferred_stages=tuple(sorted({*state.deferred_stages, stage})),
        )

    @staticmethod
    def is_attempt_due(
        state: TakeProfitStageState,
        stage: int,
        current_time: datetime.datetime,
    ) -> bool:
        """Return whether the persisted zero-fill backoff allows a new attempt."""
        if current_time.tzinfo is None:
            raise ValueError("Take-profit retry timestamp must be timezone-aware")
        retry_gate = state.retry_gate
        retry_budget_exhausted = (
            retry_gate is not None
            and retry_gate.stage == stage
            and retry_gate.consecutive_zero_fill_count
            >= TakeProfitStageState.MAX_CONSECUTIVE_ZERO_FILL_OUTCOMES
        )
        return not retry_budget_exhausted and (
            retry_gate is None
            or retry_gate.stage != stage
            or current_time >= _parse_iso_datetime(retry_gate.retry_not_before)
        )

    @classmethod
    def start_attempt(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        *,
        stage: int,
        amount: float,
        stake_amount: float,
        attempt_id: str,
        status: TakeProfitAttemptStatus,
        current_time: datetime.datetime,
        recovery_deadline: datetime.datetime,
    ) -> TakeProfitStageState:
        attempt_id = _non_empty_string(attempt_id)
        historical_attempt_ids = {
            *(
                attribution.attempt_id
                for attribution in state.order_attributions
                if attribution.attempt_id is not None
            ),
            *(
                resolution.attempt_id
                for resolution in state.operator_resolutions
                if resolution.attempt_id is not None
            ),
        }
        if attempt_id in historical_attempt_ids:
            raise ValueError("Take-profit attempt ID was already used")
        if state.blocked_reason is not None:
            raise ValueError(
                "Cannot start a take-profit attempt while state is blocked"
            )
        if state.active_attempt is not None:
            raise ValueError("A take-profit attempt is already active")
        final_stage = state.plan_signature.final_stage.stage
        if stage not in dict(state.stage_targets) and stage != final_stage:
            raise ValueError(f"Unknown take-profit stage {stage!r}")
        if status not in {"proposed", "submitting"}:
            raise ValueError(f"Invalid initial take-profit attempt status {status!r}")
        if status == "proposed" and stage != final_stage:
            raise ValueError("Only a final take-profit attempt can be proposed")
        if current_time.tzinfo is None or recovery_deadline.tzinfo is None:
            raise ValueError("Take-profit attempt timestamps must be timezone-aware")
        if not cls.is_attempt_due(state, stage, current_time):
            raise ValueError("Take-profit zero-fill retry backoff is still active")
        if recovery_deadline < current_time:
            raise ValueError("Take-profit recovery deadline precedes creation")
        amount = cls._safe_amount(amount)
        stake_amount = cls._safe_amount(stake_amount)
        if amount == 0.0 or stake_amount == 0.0:
            raise ValueError("Take-profit attempt amounts must be positive")
        execution_stage = cls.get_execution_stage(state)
        if stage != execution_stage:
            raise ValueError(
                "Take-profit attempt stage does not match the execution stage"
            )
        attempt = TakeProfitAttempt(
            attempt_id=attempt_id,
            stage=stage,
            status=status,
            amount=amount,
            stake_amount=stake_amount,
            exit_side=trade.exit_side,
            tag=cls.order_tag(trade.trade_direction, stage),
            created_at=current_time.isoformat(),
            submitted_at=(current_time.isoformat() if status == "submitting" else None),
            recovery_deadline=recovery_deadline.isoformat(),
            known_order_ids=tuple(sorted(order.order_id for order in trade.orders)),
        )
        cls._validate_attempt_bounds(state, attempt)
        return replace(
            state,
            active_attempt=attempt,
        )

    @classmethod
    def _validate_attempt_bounds(
        cls,
        state: TakeProfitStageState,
        attempt: TakeProfitAttempt,
    ) -> None:
        if attempt.stage != cls.get_execution_stage(state):
            raise ValueError(
                "Take-profit attempt stage does not match the execution stage"
            )
        final_stage = state.plan_signature.final_stage.stage
        # A proposal can predate Freqtrade's wallet fallback. Bound it to orders,
        # not to an adjustment that may be recorded only after that proposal.
        order_backed_exposure = cls._canonical_expected_trade_amount(
            (amount for _, amount in state.entry_order_amounts),
            (
                *(amount for _, amount in state.credited_order_amounts),
                *(amount for _, amount in state.non_take_profit_exit_order_amounts),
            ),
            (),
            None,
        )
        maximum_amount = (
            order_backed_exposure
            if attempt.stage == final_stage
            else min(
                cls.get_stage_progress(state, attempt.stage).remaining_amount,
                order_backed_exposure,
            )
        )
        if attempt.amount > maximum_amount and not cls._amounts_close(
            attempt.amount,
            maximum_amount,
        ):
            raise ValueError("Take-profit attempt exceeds its executable exposure")

    @classmethod
    def _record_zero_fill_outcome(
        cls,
        retry_gate: TakeProfitRetryGate | None,
        attribution: TakeProfitOrderAttribution,
        order: _Order,
        *,
        current_time: datetime.datetime | None,
    ) -> TakeProfitRetryGate:
        if retry_gate is not None and retry_gate.last_order_id == order.order_id:
            return retry_gate
        if (
            attribution.attempt_submitted_at is None
            or attribution.retry_delay_seconds is None
        ):
            raise ValueError("Attributed order has no retry evidence")
        submitted_at = _parse_iso_datetime(attribution.attempt_submitted_at)
        base_delay = datetime.timedelta(seconds=attribution.retry_delay_seconds)
        order_date = order.order_date_utc
        if order_date.tzinfo is None:
            raise ValueError("Take-profit order timestamp must be timezone-aware")
        observed_at = max(
            submitted_at,
            order_date,
            current_time or order_date,
        )
        consecutive_zero_fill_count = (
            retry_gate.consecutive_zero_fill_count + 1
            if retry_gate is not None and retry_gate.stage == attribution.stage
            else 1
        )
        return TakeProfitRetryGate(
            stage=attribution.stage,
            consecutive_zero_fill_count=consecutive_zero_fill_count,
            last_order_id=order.order_id,
            retry_not_before=(
                observed_at
                + base_delay
                * cls._retry_backoff_multiplier(consecutive_zero_fill_count)
            ).isoformat(),
        )

    @classmethod
    def _reconcile_retry_gate(
        cls,
        retry_gate: TakeProfitRetryGate | None,
        attributions: Iterable[TakeProfitOrderAttribution],
        credited_order_amounts: Mapping[str, float],
        *,
        current_time: datetime.datetime | None,
    ) -> TakeProfitRetryGate | None:
        """Derive the causal zero-fill suffix from the complete credited ledger."""
        latest_zero_fill, consecutive_zero_fill_count = cls._derive_retry_suffix(
            attributions,
            credited_order_amounts,
        )
        if latest_zero_fill is None:
            return None
        if (
            latest_zero_fill.attempt_submitted_at is None
            or latest_zero_fill.retry_delay_seconds is None
        ):
            raise ValueError("Attributed zero-fill order has no retry evidence")

        base_delay = datetime.timedelta(seconds=latest_zero_fill.retry_delay_seconds)
        if (
            retry_gate is not None
            and retry_gate.last_order_id == latest_zero_fill.order_id
        ):
            previous_observed_at = _parse_iso_datetime(retry_gate.retry_not_before) - (
                base_delay
                * cls._retry_backoff_multiplier(retry_gate.consecutive_zero_fill_count)
            )
            observed_at = max(
                previous_observed_at,
                _parse_iso_datetime(latest_zero_fill.attempt_submitted_at),
                _parse_iso_datetime(latest_zero_fill.order_date),
            )
        else:
            order_date = _parse_iso_datetime(latest_zero_fill.order_date)
            observed_at = max(
                _parse_iso_datetime(latest_zero_fill.attempt_submitted_at),
                order_date,
                current_time or order_date,
            )
        return TakeProfitRetryGate(
            stage=latest_zero_fill.stage,
            consecutive_zero_fill_count=consecutive_zero_fill_count,
            last_order_id=latest_zero_fill.order_id,
            retry_not_before=(
                observed_at
                + base_delay
                * cls._retry_backoff_multiplier(consecutive_zero_fill_count)
            ).isoformat(),
        )

    @staticmethod
    def _derive_retry_suffix(
        attributions: Iterable[TakeProfitOrderAttribution],
        credited_order_amounts: Mapping[str, float],
    ) -> tuple[TakeProfitOrderAttribution | None, int]:
        outcomes = sorted(
            (
                attribution
                for attribution in attributions
                if attribution.order_id in credited_order_amounts
            ),
            key=lambda attribution: attribution.attribution_sequence,
        )
        consecutive_zero_fill_count = 0
        latest_zero_fill: TakeProfitOrderAttribution | None = None
        for attribution in outcomes:
            credited_amount = credited_order_amounts[attribution.order_id]
            if credited_amount > 0.0:
                consecutive_zero_fill_count = 0
                latest_zero_fill = None
            elif attribution.attempt_id is not None:
                if (
                    latest_zero_fill is None
                    or latest_zero_fill.stage != attribution.stage
                ):
                    consecutive_zero_fill_count = 1
                else:
                    consecutive_zero_fill_count += 1
                latest_zero_fill = attribution
        return latest_zero_fill, consecutive_zero_fill_count

    @staticmethod
    def _retry_backoff_multiplier(consecutive_zero_fill_count: int) -> int:
        if consecutive_zero_fill_count <= 0:
            raise ValueError("Take-profit zero-fill retry count must be positive")
        return min(2 ** (consecutive_zero_fill_count - 1), 8)

    @staticmethod
    def _attempt_retry_delay_seconds(attempt: TakeProfitAttempt) -> float:
        submitted_at = _parse_iso_datetime(attempt.submitted_at or attempt.created_at)
        retry_delay = (
            _parse_iso_datetime(attempt.recovery_deadline) - submitted_at
        ).total_seconds()
        return _required_positive_value(retry_delay, "attempt retry delay")

    @staticmethod
    def mark_submitting(
        state: TakeProfitStageState,
        attempt_id: str,
        current_time: datetime.datetime,
    ) -> TakeProfitStageState:
        attempt = state.active_attempt
        if attempt is None or attempt.attempt_id != attempt_id:
            return state
        if attempt.status != "proposed":
            return state
        if current_time.tzinfo is None:
            raise ValueError("Take-profit submission timestamp must be timezone-aware")
        created_at = _parse_iso_datetime(attempt.created_at)
        if current_time < created_at:
            raise ValueError("Take-profit attempt was submitted before creation")
        recovery_window = _parse_iso_datetime(attempt.recovery_deadline) - created_at
        return replace(
            state,
            active_attempt=replace(
                attempt,
                status="submitting",
                submitted_at=current_time.isoformat(),
                recovery_deadline=(current_time + recovery_window).isoformat(),
            ),
        )

    @classmethod
    def update_attempt_amount(
        cls,
        state: TakeProfitStageState,
        attempt_id: str,
        amount: float,
        stake_amount: float,
    ) -> TakeProfitStageState:
        attempt = state.active_attempt
        if attempt is None or attempt.attempt_id != attempt_id:
            return state
        if attempt.status != "proposed":
            raise ValueError("Submitted take-profit attempt amounts cannot be changed")
        amount = cls._safe_amount(amount)
        stake_amount = cls._safe_amount(stake_amount)
        if amount == 0.0 or stake_amount == 0.0:
            raise ValueError("Take-profit attempt amounts must be positive")
        if amount > attempt.amount and not cls._amounts_close(amount, attempt.amount):
            raise ValueError("Proposed take-profit attempt amount cannot increase")
        if stake_amount > attempt.stake_amount and not cls._amounts_close(
            stake_amount,
            attempt.stake_amount,
        ):
            raise ValueError("Proposed take-profit attempt stake cannot increase")
        updated_attempt = replace(
            attempt,
            amount=amount,
            stake_amount=stake_amount,
        )
        cls._validate_attempt_bounds(state, updated_attempt)
        return replace(
            state,
            active_attempt=updated_attempt,
        )

    @classmethod
    def mark_ambiguous(
        cls,
        state: TakeProfitStageState,
        attempt_id: str,
    ) -> TakeProfitStageState:
        attempt = state.active_attempt
        if attempt is None or attempt.attempt_id != attempt_id:
            return state
        if attempt.status == "ambiguous":
            return state
        if attempt.status != "submitting":
            raise ValueError(
                "Only a submitting take-profit attempt can become ambiguous"
            )
        return replace(
            state,
            active_attempt=replace(attempt, status="ambiguous"),
        )

    @staticmethod
    def discard_proposed_attempt(
        state: TakeProfitStageState,
    ) -> TakeProfitStageState:
        attempt = state.active_attempt
        if attempt is None or attempt.status != "proposed":
            return state
        return replace(state, active_attempt=None)

    @classmethod
    def resolve_ambiguous_attempt(
        cls,
        state: TakeProfitStageState,
        attempt_id: str,
        resolved_at: datetime.datetime,
    ) -> TakeProfitStageState:
        attempt = state.active_attempt
        if (
            attempt is None
            or attempt.attempt_id != attempt_id
            or attempt.status != "ambiguous"
        ):
            raise ValueError("No matching ambiguous take-profit attempt")
        if resolved_at.tzinfo is None:
            raise ValueError("Take-profit resolution timestamp must be timezone-aware")
        recovery_deadline = _parse_iso_datetime(attempt.recovery_deadline)
        if resolved_at < recovery_deadline:
            raise ValueError(
                "Take-profit attempt cannot be cleared before its recovery deadline"
            )
        resolution = TakeProfitOperatorResolution(
            action="confirmed_no_remote_order",
            attempt_id=attempt_id,
            resolved_at=resolved_at.isoformat(),
        )
        return replace(
            state,
            active_attempt=None,
            operator_resolutions=cls._append_operator_resolution(state, resolution),
        )

    @classmethod
    def revalidate_terminal_order(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        order_id: str,
        resolved_at: datetime.datetime,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> TakeProfitStageState:
        """Reconcile one repaired canonical order and record the operator action."""
        reconciled, credited_amount = cls._prepare_terminal_order_revalidation(
            trade,
            state,
            order_id,
            resolved_at,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=True,
        )
        resolution = TakeProfitOperatorResolution(
            action="revalidated_terminal_order",
            order_id=_non_empty_string(order_id),
            credited_amount=credited_amount,
            resolved_at=resolved_at.isoformat(),
        )
        return replace(
            reconciled,
            operator_resolutions=cls._append_operator_resolution(
                reconciled,
                resolution,
            ),
        )

    @classmethod
    def validate_terminal_order_revalidation(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        order_id: str,
        resolved_at: datetime.datetime,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> None:
        """Validate repaired terminal evidence without checking trade accounting."""
        cls._prepare_terminal_order_revalidation(
            trade,
            state,
            order_id,
            resolved_at,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=False,
        )

    @classmethod
    def _prepare_terminal_order_revalidation(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        order_id: str,
        resolved_at: datetime.datetime,
        *,
        amount_quantizer: Callable[[float], float] | None,
        validate_trade_amount: bool,
    ) -> tuple[TakeProfitStageState, float]:
        order_id = _non_empty_string(order_id)
        has_matching_block = (
            state.blocked_reason in _RECOVERABLE_TERMINAL_ORDER_BLOCK_REASONS
            and state.blocked_order_id == order_id
        )
        if not has_matching_block:
            raise ValueError("No matching terminal fill to revalidate")
        if resolved_at.tzinfo is None:
            raise ValueError("Take-profit resolution timestamp must be timezone-aware")

        matching_orders = [
            order for order in trade.orders if order.order_id == order_id
        ]
        if len(matching_orders) != 1:
            raise ValueError("Expected exactly one matching canonical order")
        order = matching_orders[0]
        if order.ft_order_side != trade.exit_side or order.ft_is_open:
            raise ValueError("Revalidated order is not a terminal strategy exit")
        cls._validate_resolution_timeline(order, resolved_at)
        state = cls._prepare_terminal_attribution(
            trade,
            state,
            order,
            amount_quantizer,
        )
        zero_fill_proof = next(
            (
                proof
                for proof in state.confirmed_zero_fill_orders
                if proof.order_id == order_id
            ),
            None,
        )
        matching_zero_fill_proof = (
            zero_fill_proof
            if zero_fill_proof is not None
            and cls._zero_fill_proof_matches(trade, order, zero_fill_proof)
            else None
        )
        status = (order.status or "").lower()
        filled = _non_negative_float(order.filled) if order.filled is not None else None
        if (
            status in _OPERATOR_CONFIRMABLE_CANCELED_STATUSES
            and filled is not None
            and filled > 0
        ):
            raise ValueError(
                "Canceled positive fill requires confirm_terminal_canceled_fill"
            )
        credited_amount = cls._terminal_credited_amount(
            order,
            matching_zero_fill_proof,
        )

        candidate = replace(
            state,
            confirmed_zero_fill_orders=tuple(
                proof
                for proof in state.confirmed_zero_fill_orders
                if proof.order_id != order_id
            ),
            credited_order_amounts=tuple(
                record
                for record in state.credited_order_amounts
                if record[0] != order_id
            ),
            non_take_profit_exit_order_amounts=tuple(
                record
                for record in state.non_take_profit_exit_order_amounts
                if record[0] != order_id
            ),
            confirmed_canceled_fill_orders=tuple(
                proof
                for proof in state.confirmed_canceled_fill_orders
                if proof.order_id != order_id
            ),
            blocked_reason=None,
            blocked_order_id=None,
        )
        reconciled = cls._reconcile(
            trade,
            candidate,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=validate_trade_amount,
            current_time=resolved_at,
            allow_canonical_adjustment_retirement=validate_trade_amount,
        )
        if reconciled.blocked_reason not in {
            None,
            "zero_fill_retry_limit_reached",
        }:
            raise ValueError(
                "Canonical order repair does not produce a valid take-profit state: "
                f"{reconciled.blocked_reason}"
            )

        attributed = any(
            attribution.order_id == order_id
            for attribution in reconciled.order_attributions
        )
        persisted_credit = (
            dict(reconciled.credited_order_amounts).get(order_id)
            if attributed
            else dict(reconciled.non_take_profit_exit_order_amounts).get(order_id)
        )
        if credited_amount > 0.0 and persisted_credit is None:
            raise ValueError("Positive terminal fill is not present in the exit ledger")
        if persisted_credit is not None and not cls._amounts_close(
            persisted_credit,
            credited_amount,
        ):
            raise ValueError("Revalidated exit credit does not match the order")

        return reconciled, credited_amount

    @classmethod
    def revalidate_state(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        resolved_at: datetime.datetime,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> TakeProfitStageState:
        """Clear an eligible latch only after the complete state validates exactly."""
        if state.blocked_reason not in _REVALIDATABLE_STATE_BLOCK_REASONS:
            raise ValueError("Take-profit block is not eligible for state revalidation")
        if resolved_at.tzinfo is None:
            raise ValueError("Take-profit resolution timestamp must be timezone-aware")
        if state.blocked_reason == "unknown_attempt_order":
            matching_orders = [
                order
                for order in trade.orders
                if order.order_id == state.blocked_order_id
            ]
            if len(matching_orders) != 1:
                raise ValueError("Expected exactly one matching canonical order")
            order = matching_orders[0]
            if not order.ft_is_open:
                raise ValueError("Unknown attempt order is no longer open")
            if order.ft_order_side != trade.exit_side:
                raise ValueError("Unknown attempt order is not a strategy exit")
            try:
                repaired_tag = _non_empty_string(order.ft_order_tag)
            except ValueError as error:
                raise ValueError(
                    "Unknown attempt order requires a non-empty exit tag"
                ) from error
            if cls.has_order_tag_prefix(repaired_tag):
                raise ValueError("Unknown attempt order still has a take-profit tag")
        candidate = replace(
            state,
            blocked_reason=None,
            blocked_order_id=None,
        )
        reconciled = cls._reconcile(
            trade,
            candidate,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=True,
            current_time=resolved_at,
        )
        if reconciled.blocked_reason not in {
            None,
            "zero_fill_retry_limit_reached",
        }:
            raise ValueError(
                "Repaired take-profit state remains invalid: "
                f"{reconciled.blocked_reason}"
            )
        resolution = TakeProfitOperatorResolution(
            action="revalidated_state",
            resolved_at=resolved_at.isoformat(),
            cleared_block_reason=state.blocked_reason,
            cleared_block_order_id=state.blocked_order_id,
        )
        resolved = replace(
            reconciled,
            operator_resolutions=cls._append_operator_resolution(
                reconciled,
                resolution,
            ),
        )
        if TakeProfitStageState.from_mapping(resolved.to_mapping()) != resolved:
            raise ValueError(
                "Revalidated take-profit state failed round-trip validation"
            )
        return resolved

    @classmethod
    def confirm_terminal_zero_fill(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        order_id: str,
        terminal_status: str,
        resolved_at: datetime.datetime,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> TakeProfitStageState:
        """Persist an exact operator-verified canceled zero-fill snapshot."""
        reconciled = cls._prepare_terminal_zero_fill_confirmation(
            trade,
            state,
            order_id,
            terminal_status,
            resolved_at,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=True,
        )
        order_id = _non_empty_string(order_id)
        terminal_status = _non_empty_string(terminal_status).lower()
        resolution = TakeProfitOperatorResolution(
            action="confirmed_terminal_zero_fill",
            order_id=order_id,
            credited_amount=0.0,
            terminal_status=terminal_status,
            resolved_at=resolved_at.isoformat(),
        )
        return replace(
            reconciled,
            operator_resolutions=cls._append_operator_resolution(
                reconciled,
                resolution,
            ),
        )

    @classmethod
    def validate_terminal_zero_fill_confirmation(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        order_id: str,
        terminal_status: str,
        resolved_at: datetime.datetime,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> None:
        """Validate zero-fill evidence without enforcing the trade-amount postcondition."""
        cls._prepare_terminal_zero_fill_confirmation(
            trade,
            state,
            order_id,
            terminal_status,
            resolved_at,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=False,
        )

    @classmethod
    def _prepare_terminal_zero_fill_confirmation(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        order_id: str,
        terminal_status: str,
        resolved_at: datetime.datetime,
        *,
        amount_quantizer: Callable[[float], float] | None,
        validate_trade_amount: bool,
    ) -> TakeProfitStageState:
        order_id = _non_empty_string(order_id)
        terminal_status = _non_empty_string(terminal_status).lower()
        if terminal_status not in _OPERATOR_CONFIRMABLE_CANCELED_STATUSES:
            raise ValueError("Terminal zero-fill status is not operator-recoverable")
        if (
            state.blocked_reason not in _RECOVERABLE_TERMINAL_ORDER_BLOCK_REASONS
            or state.blocked_order_id != order_id
        ):
            raise ValueError("No matching unknown terminal fill to confirm")
        if resolved_at.tzinfo is None:
            raise ValueError("Take-profit resolution timestamp must be timezone-aware")
        matching_orders = [
            order for order in trade.orders if order.order_id == order_id
        ]
        if len(matching_orders) != 1:
            raise ValueError("Expected exactly one matching canonical order")
        state = cls._prepare_terminal_attribution(
            trade,
            state,
            matching_orders[0],
            amount_quantizer,
        )
        proof = cls._build_zero_fill_proof(
            trade,
            matching_orders[0],
            terminal_status,
            resolved_at,
        )
        candidate = replace(
            state,
            confirmed_zero_fill_orders=tuple(
                sorted(
                    (
                        *(
                            existing_proof
                            for existing_proof in state.confirmed_zero_fill_orders
                            if existing_proof.order_id != order_id
                        ),
                        proof,
                    ),
                    key=lambda item: item.order_id,
                )
            ),
            confirmed_canceled_fill_orders=tuple(
                existing_proof
                for existing_proof in state.confirmed_canceled_fill_orders
                if existing_proof.order_id != order_id
            ),
            credited_order_amounts=tuple(
                record
                for record in state.credited_order_amounts
                if record[0] != order_id
            ),
            non_take_profit_exit_order_amounts=tuple(
                record
                for record in state.non_take_profit_exit_order_amounts
                if record[0] != order_id
            ),
            blocked_reason=None,
            blocked_order_id=None,
        )
        reconciled = cls._reconcile(
            trade,
            candidate,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=validate_trade_amount,
            current_time=resolved_at,
        )
        if reconciled.blocked_reason not in {
            None,
            "zero_fill_retry_limit_reached",
        }:
            raise ValueError(
                "Terminal zero-fill proof does not produce a valid state: "
                f"{reconciled.blocked_reason}"
            )
        return reconciled

    @classmethod
    def confirm_terminal_canceled_fill(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        order_id: str,
        terminal_status: str,
        resolved_at: datetime.datetime,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> TakeProfitStageState:
        """Persist exact operator proof for a canceled positive-fill order."""
        reconciled, proof = cls._prepare_terminal_canceled_fill_confirmation(
            trade,
            state,
            order_id,
            terminal_status,
            resolved_at,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=True,
        )
        order_id = _non_empty_string(order_id)
        terminal_status = _non_empty_string(terminal_status).lower()
        resolution = TakeProfitOperatorResolution(
            action="confirmed_terminal_canceled_fill",
            order_id=order_id,
            credited_amount=proof.credited_amount,
            terminal_status=terminal_status,
            resolved_at=resolved_at.isoformat(),
        )
        return replace(
            reconciled,
            operator_resolutions=cls._append_operator_resolution(
                reconciled,
                resolution,
            ),
        )

    @classmethod
    def validate_terminal_canceled_fill_confirmation(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        order_id: str,
        terminal_status: str,
        resolved_at: datetime.datetime,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> None:
        """Validate canceled-fill evidence without enforcing the amount postcondition."""
        cls._prepare_terminal_canceled_fill_confirmation(
            trade,
            state,
            order_id,
            terminal_status,
            resolved_at,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=False,
        )

    @classmethod
    def _prepare_terminal_canceled_fill_confirmation(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        order_id: str,
        terminal_status: str,
        resolved_at: datetime.datetime,
        *,
        amount_quantizer: Callable[[float], float] | None,
        validate_trade_amount: bool,
    ) -> tuple[TakeProfitStageState, TakeProfitCanceledFillProof]:
        order_id = _non_empty_string(order_id)
        terminal_status = _non_empty_string(terminal_status).lower()
        if terminal_status not in _OPERATOR_CONFIRMABLE_CANCELED_STATUSES:
            raise ValueError("Terminal canceled-fill status is not recoverable")
        if (
            state.blocked_reason not in _RECOVERABLE_TERMINAL_ORDER_BLOCK_REASONS
            or state.blocked_order_id != order_id
        ):
            raise ValueError("No matching canceled terminal fill to confirm")
        if resolved_at.tzinfo is None:
            raise ValueError("Take-profit resolution timestamp must be timezone-aware")
        matching_orders = [
            order for order in trade.orders if order.order_id == order_id
        ]
        if len(matching_orders) != 1:
            raise ValueError("Expected exactly one matching canonical order")
        state = cls._prepare_terminal_attribution(
            trade,
            state,
            matching_orders[0],
            amount_quantizer,
        )
        proof = cls._build_canceled_fill_proof(
            trade,
            matching_orders[0],
            terminal_status,
            resolved_at,
        )
        candidate = replace(
            state,
            confirmed_zero_fill_orders=tuple(
                existing_proof
                for existing_proof in state.confirmed_zero_fill_orders
                if existing_proof.order_id != order_id
            ),
            confirmed_canceled_fill_orders=tuple(
                sorted(
                    (
                        *(
                            existing_proof
                            for existing_proof in state.confirmed_canceled_fill_orders
                            if existing_proof.order_id != order_id
                        ),
                        proof,
                    ),
                    key=lambda item: item.order_id,
                )
            ),
            credited_order_amounts=tuple(
                record
                for record in state.credited_order_amounts
                if record[0] != order_id
            ),
            non_take_profit_exit_order_amounts=tuple(
                record
                for record in state.non_take_profit_exit_order_amounts
                if record[0] != order_id
            ),
            blocked_reason=None,
            blocked_order_id=None,
        )
        reconciled = cls._reconcile(
            trade,
            candidate,
            amount_quantizer=amount_quantizer,
            validate_trade_amount=validate_trade_amount,
            current_time=resolved_at,
            allow_canonical_adjustment_retirement=validate_trade_amount,
        )
        if reconciled.blocked_reason is not None:
            raise ValueError(
                "Terminal canceled-fill proof does not produce a valid state: "
                f"{reconciled.blocked_reason}"
            )
        return reconciled, proof

    @classmethod
    def get_recovery_instruction(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> TakeProfitRecoveryInstruction | None:
        """Return the single recovery action supported by canonical evidence."""
        if not trade.is_open:
            return None
        if state.partial_wallet_mutation is not None:
            return None
        resolved_at = datetime.datetime.now(datetime.timezone.utc)
        if state.blocked_reason in _REVALIDATABLE_STATE_BLOCK_REASONS:
            try:
                reconciled = cls.revalidate_state(
                    trade,
                    state,
                    resolved_at,
                    amount_quantizer=amount_quantizer,
                )
            except ValueError:
                pass
            else:
                expected_amount = cls.get_expected_trade_amount(
                    reconciled,
                    amount_quantizer=amount_quantizer,
                )
                return TakeProfitRecoveryInstruction(
                    action="revalidate_state",
                    expected_trade_amount=expected_amount,
                )

        order_id = state.blocked_order_id
        if order_id is None:
            return None
        matching_orders = [
            order for order in trade.orders if order.order_id == order_id
        ]
        if len(matching_orders) != 1:
            return None
        order = matching_orders[0]
        if order.ft_order_side != trade.exit_side or order.ft_is_open:
            return None
        terminal_status = (order.status or "").lower()
        if (
            state.blocked_reason in _RECOVERABLE_TERMINAL_ORDER_BLOCK_REASONS
            and terminal_status in _OPERATOR_CONFIRMABLE_CANCELED_STATUSES
            and order.filled is not None
        ):
            try:
                filled = _non_negative_float(order.filled)
                if filled == 0.0:
                    action: TakeProfitRecoveryAction = "confirm_terminal_zero_fill"
                    try:
                        reconciled = cls.confirm_terminal_zero_fill(
                            trade,
                            state,
                            order_id,
                            terminal_status,
                            resolved_at,
                            amount_quantizer=amount_quantizer,
                        )
                    except ValueError:
                        reconciled = cls._prepare_terminal_zero_fill_confirmation(
                            trade,
                            state,
                            order_id,
                            terminal_status,
                            resolved_at,
                            amount_quantizer=amount_quantizer,
                            validate_trade_amount=False,
                        )
                        accounting_is_valid = False
                    else:
                        accounting_is_valid = True
                else:
                    action = "confirm_terminal_canceled_fill"
                    try:
                        reconciled = cls.confirm_terminal_canceled_fill(
                            trade,
                            state,
                            order_id,
                            terminal_status,
                            resolved_at,
                            amount_quantizer=amount_quantizer,
                        )
                    except ValueError:
                        reconciled, _ = (
                            cls._prepare_terminal_canceled_fill_confirmation(
                                trade,
                                state,
                                order_id,
                                terminal_status,
                                resolved_at,
                                amount_quantizer=amount_quantizer,
                                validate_trade_amount=False,
                            )
                        )
                        accounting_is_valid = False
                    else:
                        accounting_is_valid = True
            except ValueError:
                return None
            try:
                trade_accounting_action, expected_trade_amount = (
                    cls._trade_accounting_recovery(
                        trade,
                        reconciled,
                        order,
                        accounting_is_valid=accounting_is_valid,
                        amount_quantizer=amount_quantizer,
                    )
                )
            except ValueError:
                return None
            return TakeProfitRecoveryInstruction(
                action=action,
                order_id=order_id,
                terminal_status=terminal_status,
                trade_accounting_action=trade_accounting_action,
                expected_trade_amount=expected_trade_amount,
            )
        if terminal_status in _OPERATOR_CONFIRMABLE_CANCELED_STATUSES:
            return None
        if state.blocked_reason in _RECOVERABLE_TERMINAL_ORDER_BLOCK_REASONS:
            try:
                try:
                    reconciled = cls.revalidate_terminal_order(
                        trade,
                        state,
                        order_id,
                        resolved_at,
                        amount_quantizer=amount_quantizer,
                    )
                except ValueError:
                    reconciled, _ = cls._prepare_terminal_order_revalidation(
                        trade,
                        state,
                        order_id,
                        resolved_at,
                        amount_quantizer=amount_quantizer,
                        validate_trade_amount=False,
                    )
                    accounting_is_valid = False
                else:
                    accounting_is_valid = True
            except ValueError:
                return None
            try:
                trade_accounting_action, expected_trade_amount = (
                    cls._trade_accounting_recovery(
                        trade,
                        reconciled,
                        order,
                        accounting_is_valid=accounting_is_valid,
                        amount_quantizer=amount_quantizer,
                    )
                )
            except ValueError:
                return None
            return TakeProfitRecoveryInstruction(
                action="revalidate_terminal_order",
                order_id=order_id,
                trade_accounting_action=trade_accounting_action,
                expected_trade_amount=expected_trade_amount,
            )
        return None

    @classmethod
    def _trade_accounting_recovery(
        cls,
        trade: _Trade,
        reconciled: TakeProfitStageState,
        order: _Order,
        *,
        accounting_is_valid: bool,
        amount_quantizer: Callable[[float], float] | None,
    ) -> tuple[TakeProfitTradeAccountingAction, float]:
        expected_amount = cls.get_expected_trade_amount(
            reconciled,
            amount_quantizer=amount_quantizer,
        )
        expected_trade_is_open = not cls._amounts_close(expected_amount, 0.0)
        if not expected_trade_is_open:
            expected_amount = 0.0
        filled = _non_negative_float(order.filled) if order.filled is not None else 0.0
        order_funding_fee = _finite_float(order.funding_fee or 0.0)
        _finite_float(trade.funding_fee_running or 0.0)
        has_order_funding_fee = not cls._amounts_close(order_funding_fee, 0.0)
        if filled == 0.0 and has_order_funding_fee:
            raise ValueError("Zero-fill recovery order contains funding fees")
        if filled > 0.0:
            _required_positive_value(
                order.safe_price,
                "terminal recovery order price",
            )
        if not trade.is_open:
            raise ValueError("Closed trades are inspect-only")
        if not expected_trade_is_open:
            if filled == 0.0:
                raise ValueError("Zero-fill recovery cannot close trade exposure")
            return "native_freqtrade_lifecycle", expected_amount
        if filled == 0.0 and accounting_is_valid:
            return "none", expected_amount
        return "recalculate_from_orders", expected_amount

    @staticmethod
    def _append_operator_resolution(
        state: TakeProfitStageState,
        resolution: TakeProfitOperatorResolution,
    ) -> tuple[TakeProfitOperatorResolution, ...]:
        return (*state.operator_resolutions, resolution)[
            -TakeProfitStageState.MAX_OPERATOR_RESOLUTIONS :
        ]

    @classmethod
    def _entry_order_amounts(cls, trade: _Trade) -> dict[str, float]:
        return {
            order.order_id: cls._safe_amount(order.safe_amount_after_fee)
            for order in trade.orders
            if order.ft_order_side == trade.entry_side
            and not order.ft_is_open
            and cls._safe_amount(order.safe_amount_after_fee) > 0.0
        }

    @classmethod
    def _terminal_credited_amount(
        cls,
        order: _Order,
        zero_fill_proof: TakeProfitZeroFillProof | None = None,
        canceled_fill_proof: TakeProfitCanceledFillProof | None = None,
    ) -> float:
        if order.ft_is_open:
            return 0.0

        amount = _non_negative_float(order.safe_amount)
        if amount == 0.0:
            raise ValueError("Terminal order has no amount")

        status = (order.status or "").lower()
        if status not in cls._TERMINAL_STATUSES:
            raise ValueError("Order is closed locally without a terminal status")

        remaining = (
            _non_negative_float(order.remaining)
            if order.remaining is not None
            else None
        )
        if order.filled is None:
            raise ValueError("Terminal order fill is unknown")
        filled = _non_negative_float(order.filled)

        if filled > amount and not cls._amounts_close(filled, amount):
            raise ValueError("Terminal order fill exceeds its amount")
        filled = min(filled, amount)
        if remaining is not None and (
            remaining > amount or not cls._amounts_close(filled + remaining, amount)
        ):
            raise ValueError("Terminal order fill and remainder are inconsistent")

        fee_base = (
            _non_negative_float(order.ft_fee_base)
            if order.ft_fee_base is not None
            else 0.0
        )
        if fee_base > filled and not cls._amounts_close(fee_base, filled):
            raise ValueError("Terminal order base fee exceeds its fill")

        if filled == 0.0:
            allowed_statuses = cls._ZERO_FILL_RETRY_STATUSES
            if zero_fill_proof is not None:
                if not cls._zero_fill_proof_fields_match(order, zero_fill_proof):
                    raise ValueError("Terminal zero-fill proof no longer matches")
                allowed_statuses = allowed_statuses | {zero_fill_proof.terminal_status}
            cls._validate_terminal_zero_fill(order, allowed_statuses)
            return 0.0

        _required_positive_value(
            order.safe_price,
            "terminal order execution price",
        )
        _finite_float(order.funding_fee or 0.0)
        filled_at = order.order_filled_utc
        order_date = order.order_date_utc
        if filled_at is None:
            raise ValueError("Terminal positive fill is not explicitly observed")
        if order_date.tzinfo is None or filled_at.tzinfo is None:
            raise ValueError("Terminal fill timeline must be timezone-aware")
        if filled_at < order_date:
            raise ValueError("Terminal fill precedes order creation")

        if status in _OPERATOR_CONFIRMABLE_CANCELED_STATUSES:
            if canceled_fill_proof is None or not cls._canceled_fill_proof_fields_match(
                order,
                canceled_fill_proof,
            ):
                raise ValueError("Canceled positive fill requires exact operator proof")
            return canceled_fill_proof.credited_amount
        if filled < amount and remaining is None:
            raise ValueError("Terminal partial fill has no reported remainder")
        if fee_base >= filled or cls._amounts_close(fee_base, filled):
            raise ValueError("Terminal order base fee consumes its entire fill")

        return filled - fee_base

    @classmethod
    def _external_exit_block(
        cls,
        order: _Order,
    ) -> str | None:
        try:
            amount = _required_positive_value(
                order.safe_amount,
                "external exit amount",
            )
            filled = (
                _non_negative_float(order.filled) if order.filled is not None else None
            )
            fee_base = (
                _non_negative_float(order.ft_fee_base)
                if order.ft_fee_base is not None
                else 0.0
            )
        except ValueError:
            return "unknown_external_exit_fill"
        if filled is not None and filled > 0.0:
            try:
                cls._terminal_credited_amount(order)
            except ValueError:
                return "unknown_external_exit_fill"
            return "external_exit_fill"
        if order.ft_is_open:
            if fee_base != 0.0:
                return "unknown_external_exit_fill"
            if order.remaining is not None:
                try:
                    remaining = _non_negative_float(order.remaining)
                except ValueError:
                    return "unknown_external_exit_fill"
                if not cls._amounts_close(remaining, amount):
                    return "unknown_external_exit_fill"
            return None
        try:
            credited_amount = cls._terminal_credited_amount(order)
        except ValueError:
            return "unknown_external_exit_fill"
        return "external_exit_fill" if credited_amount > 0.0 else None

    @classmethod
    def _build_zero_fill_proof(
        cls,
        trade: _Trade,
        order: _Order,
        terminal_status: str,
        confirmed_at: datetime.datetime,
    ) -> TakeProfitZeroFillProof:
        if order.ft_order_side != trade.exit_side:
            raise ValueError("Terminal zero-fill proof is not a strategy exit")
        cls._validate_resolution_timeline(order, confirmed_at)
        if (order.status or "").lower() != terminal_status:
            raise ValueError("Canonical order status does not match terminal-status")
        amount, remaining, fee_base, funding_fee = cls._validate_terminal_zero_fill(
            order,
            _OPERATOR_CONFIRMABLE_CANCELED_STATUSES,
        )
        return TakeProfitZeroFillProof(
            order_id=order.order_id,
            order_date=order.order_date_utc.isoformat(),
            order_side=order.ft_order_side,
            order_tag=order.ft_order_tag,
            terminal_status=terminal_status,
            amount=amount,
            remaining=remaining,
            fee_base=fee_base,
            funding_fee=funding_fee,
            confirmed_at=confirmed_at.isoformat(),
        )

    @classmethod
    def _zero_fill_proof_matches(
        cls,
        trade: _Trade,
        order: _Order,
        proof: TakeProfitZeroFillProof,
    ) -> bool:
        if order.ft_order_side != trade.exit_side:
            return False
        return cls._zero_fill_proof_fields_match(order, proof)

    @classmethod
    def _zero_fill_proof_fields_match(
        cls,
        order: _Order,
        proof: TakeProfitZeroFillProof,
    ) -> bool:
        try:
            amount, remaining, fee_base, funding_fee = cls._validate_terminal_zero_fill(
                order,
                {proof.terminal_status},
            )
        except ValueError:
            return False
        return (
            order.order_id == proof.order_id
            and order.order_date_utc == _parse_iso_datetime(proof.order_date)
            and order.ft_order_side == proof.order_side
            and order.ft_order_tag == proof.order_tag
            and (order.status or "").lower() == proof.terminal_status
            and amount == proof.amount
            and remaining == proof.remaining
            and fee_base == proof.fee_base
            and funding_fee == proof.funding_fee
        )

    @classmethod
    def _build_canceled_fill_proof(
        cls,
        trade: _Trade,
        order: _Order,
        terminal_status: str,
        confirmed_at: datetime.datetime,
    ) -> TakeProfitCanceledFillProof:
        if order.ft_order_side != trade.exit_side:
            raise ValueError("Terminal canceled-fill proof is not a strategy exit")
        cls._validate_resolution_timeline(order, confirmed_at)
        if (order.status or "").lower() != terminal_status:
            raise ValueError("Canonical order status does not match terminal-status")
        (
            amount,
            filled,
            remaining,
            fee_base,
            execution_price,
            funding_fee,
            filled_at,
        ) = cls._validate_terminal_canceled_fill(
            order,
            _OPERATOR_CONFIRMABLE_CANCELED_STATUSES,
        )
        return TakeProfitCanceledFillProof(
            order_id=order.order_id,
            order_date=order.order_date_utc.isoformat(),
            order_side=order.ft_order_side,
            order_tag=order.ft_order_tag,
            terminal_status=terminal_status,
            amount=amount,
            filled=filled,
            remaining=remaining,
            fee_base=fee_base,
            execution_price=execution_price,
            funding_fee=funding_fee,
            filled_at=filled_at.isoformat(),
            confirmed_at=confirmed_at.isoformat(),
        )

    @classmethod
    def _canceled_fill_proof_matches(
        cls,
        trade: _Trade,
        order: _Order,
        proof: TakeProfitCanceledFillProof,
    ) -> bool:
        return (
            order.ft_order_side == trade.exit_side
            and cls._canceled_fill_proof_fields_match(order, proof)
        )

    @classmethod
    def _canceled_fill_proof_fields_match(
        cls,
        order: _Order,
        proof: TakeProfitCanceledFillProof,
    ) -> bool:
        try:
            (
                amount,
                filled,
                remaining,
                fee_base,
                execution_price,
                funding_fee,
                filled_at,
            ) = cls._validate_terminal_canceled_fill(
                order,
                {proof.terminal_status},
            )
        except ValueError:
            return False
        return (
            order.order_id == proof.order_id
            and order.order_date_utc == _parse_iso_datetime(proof.order_date)
            and order.ft_order_side == proof.order_side
            and order.ft_order_tag == proof.order_tag
            and (order.status or "").lower() == proof.terminal_status
            and amount == proof.amount
            and filled == proof.filled
            and remaining == proof.remaining
            and fee_base == proof.fee_base
            and execution_price == proof.execution_price
            and funding_fee == proof.funding_fee
            and filled_at == _parse_iso_datetime(proof.filled_at)
        )

    @staticmethod
    def _validate_resolution_timeline(
        order: _Order,
        resolved_at: datetime.datetime,
    ) -> None:
        if resolved_at.tzinfo is None:
            raise ValueError("Take-profit resolution timestamp must be timezone-aware")
        order_date = order.order_date_utc
        if order_date.tzinfo is None:
            raise ValueError(
                "Canonical order creation timestamp must be timezone-aware"
            )
        if resolved_at < order_date:
            raise ValueError("Take-profit resolution precedes order creation")
        filled_at = order.order_filled_utc
        if filled_at is None:
            return
        if filled_at.tzinfo is None:
            raise ValueError("Canonical order fill timestamp must be timezone-aware")
        if filled_at < order_date or resolved_at < filled_at:
            raise ValueError("Canonical order has an invalid resolution timeline")

    @classmethod
    def _validate_terminal_canceled_fill(
        cls,
        order: _Order,
        allowed_statuses: set[str] | frozenset[str],
    ) -> tuple[float, float, float, float, float, float, datetime.datetime]:
        if order.ft_is_open:
            raise ValueError("Terminal canceled-fill order is open")
        status = (order.status or "").lower()
        if status not in allowed_statuses:
            raise ValueError(
                "Terminal canceled-fill status is not operator-recoverable"
            )
        amount = _required_positive_value(order.safe_amount, "terminal order amount")
        if order.filled is None:
            raise ValueError("Terminal canceled-fill order has no explicit fill")
        filled = _required_positive_value(order.filled, "terminal order fill")
        if filled > amount and not cls._amounts_close(filled, amount):
            raise ValueError("Terminal canceled fill exceeds its amount")
        filled = min(filled, amount)
        if order.remaining is None:
            raise ValueError("Terminal canceled-fill order has no explicit remainder")
        remaining = _non_negative_float(order.remaining)
        if not cls._amounts_close(filled + remaining, amount):
            raise ValueError("Terminal canceled-fill remainder is inconsistent")
        fee_base = (
            _non_negative_float(order.ft_fee_base)
            if order.ft_fee_base is not None
            else 0.0
        )
        if fee_base >= filled or cls._amounts_close(fee_base, filled):
            raise ValueError("Terminal canceled-fill base fee consumes its fill")
        execution_price = _required_positive_value(
            order.safe_price,
            "terminal canceled-fill execution price",
        )
        funding_fee = _finite_float(order.funding_fee or 0.0)
        filled_at = order.order_filled_utc
        if filled_at is None or filled_at.tzinfo is None:
            raise ValueError("Terminal canceled-fill order has no fill timestamp")
        return (
            amount,
            filled,
            remaining,
            fee_base,
            execution_price,
            funding_fee,
            filled_at,
        )

    @classmethod
    def _validate_terminal_zero_fill(
        cls,
        order: _Order,
        allowed_statuses: set[str] | frozenset[str],
    ) -> tuple[float, float, float, float]:
        if order.ft_is_open:
            raise ValueError("Terminal zero-fill order is open")
        status = (order.status or "").lower()
        if status not in allowed_statuses:
            raise ValueError("Terminal zero-fill status is not retryable")
        amount = _required_positive_value(order.safe_amount, "terminal order amount")
        if order.filled is None or _non_negative_float(order.filled) != 0.0:
            raise ValueError("Terminal zero-fill order has no explicit zero fill")
        if order.remaining is None:
            raise ValueError("Terminal zero-fill order has no explicit remainder")
        remaining = _non_negative_float(order.remaining)
        if not cls._amounts_close(remaining, amount):
            raise ValueError("Terminal zero-fill remainder is inconsistent")
        fee_base = (
            _non_negative_float(order.ft_fee_base)
            if order.ft_fee_base is not None
            else 0.0
        )
        if fee_base != 0.0:
            raise ValueError("Terminal zero-fill order has a base fee")
        funding_fee = _finite_float(order.funding_fee or 0.0)
        if funding_fee != 0.0:
            raise ValueError("Terminal zero-fill order has a funding fee")
        if order.order_filled_utc is not None:
            raise ValueError("Terminal zero-fill order has a fill timestamp")
        return amount, remaining, fee_base, funding_fee

    @staticmethod
    def _canonical_amount(
        value: object,
        amount_quantizer: Callable[[float], float] | None,
    ) -> float:
        amount = _non_negative_float(value)
        if amount_quantizer is not None:
            amount = _non_negative_float(amount_quantizer(amount))
        return amount

    @classmethod
    def _canonical_amount_sum(
        cls,
        amounts: Iterable[float],
        amount_quantizer: Callable[[float], float] | None,
    ) -> float:
        precise_amount = Decimal(0)
        for amount in amounts:
            precise_amount += Decimal(str(_non_negative_float(amount)))
        return cls._canonical_amount(float(precise_amount), amount_quantizer)

    @classmethod
    def _canonical_signed_amount_sum(
        cls,
        entry_amounts: Iterable[float],
        exit_amounts: Iterable[float],
        amount_quantizer: Callable[[float], float] | None,
    ) -> float:
        precise_amount = Decimal(0)
        for amount in entry_amounts:
            precise_amount += Decimal(str(_non_negative_float(amount)))
        for amount in exit_amounts:
            precise_amount -= Decimal(str(_non_negative_float(amount)))
        if precise_amount < 0:
            return float(precise_amount)
        return cls._canonical_amount(float(precise_amount), amount_quantizer)

    @staticmethod
    def _raw_signed_amount_sum(
        entry_amounts: Iterable[float],
        exit_amounts: Iterable[float],
    ) -> float:
        precise_amount = Decimal(0)
        for amount in entry_amounts:
            precise_amount += Decimal(str(_non_negative_float(amount)))
        for amount in exit_amounts:
            precise_amount -= Decimal(str(_non_negative_float(amount)))
        return float(precise_amount)

    @classmethod
    def _raw_expected_trade_amount(
        cls,
        entry_amounts: Iterable[float],
        exit_order_amounts: Iterable[float],
        exposure_adjustments: Iterable[float],
        amount_quantizer: Callable[[float], float] | None,
    ) -> float:
        """Mirror Freqtrade's order recalculation, then its raw wallet fallback."""
        canonical_order_amount = cls._canonical_signed_amount_sum(
            entry_amounts,
            exit_order_amounts,
            amount_quantizer,
        )
        if canonical_order_amount < 0.0:
            return canonical_order_amount
        return cls._raw_signed_amount_sum(
            (canonical_order_amount,),
            exposure_adjustments,
        )

    @classmethod
    def _canonical_expected_trade_amount(
        cls,
        entry_amounts: Iterable[float],
        exit_order_amounts: Iterable[float],
        exposure_adjustments: Iterable[float],
        amount_quantizer: Callable[[float], float] | None,
    ) -> float:
        expected_amount = cls._raw_expected_trade_amount(
            entry_amounts,
            exit_order_amounts,
            exposure_adjustments,
            amount_quantizer,
        )
        if expected_amount < 0.0:
            return expected_amount
        return cls._canonical_amount(expected_amount, amount_quantizer)

    @staticmethod
    def _retire_exposure_adjustments(
        adjustments: tuple[TakeProfitExposureAdjustment, ...],
        adjustment_ids: set[str],
        retired_at: str,
    ) -> tuple[TakeProfitExposureAdjustment, ...]:
        return tuple(
            replace(adjustment, retired_at=retired_at)
            if adjustment.adjustment_id in adjustment_ids
            else adjustment
            for adjustment in adjustments
        )

    @staticmethod
    def _compact_exposure_adjustments(
        adjustments: tuple[TakeProfitExposureAdjustment, ...],
        compacted_count: int,
        compacted_amount: float,
        *,
        required_slots: int = 0,
    ) -> tuple[tuple[TakeProfitExposureAdjustment, ...], int, float]:
        """Compact oldest retired records only, preserving aggregate audit data."""
        maximum_records = max(
            0,
            TakeProfitStageState.MAX_EXPOSURE_ADJUSTMENTS - required_slots,
        )
        remove_count = max(0, len(adjustments) - maximum_records)
        if remove_count == 0:
            return adjustments, compacted_count, compacted_amount
        retired_indexes = [
            index
            for index, adjustment in enumerate(adjustments)
            if not adjustment.is_active
        ][:remove_count]
        if len(retired_indexes) != remove_count:
            return adjustments, compacted_count, compacted_amount
        retired_index_set = set(retired_indexes)
        compacted_adjustments = tuple(
            adjustment
            for index, adjustment in enumerate(adjustments)
            if index not in retired_index_set
        )
        compacted_sum = math.fsum(
            adjustments[index].amount for index in retired_indexes
        )
        return (
            compacted_adjustments,
            compacted_count + remove_count,
            math.fsum((compacted_amount, compacted_sum)),
        )

    @classmethod
    def _attribution_provenance_matches(
        cls,
        trade: _Trade,
        attribution: TakeProfitOrderAttribution,
        order: _Order,
        configured_stages: set[int],
    ) -> bool:
        if attribution.stage not in configured_stages:
            return False
        if attribution.source == "recovered_untagged":
            has_valid_tag = attribution.tag is None
        else:
            parsed_tag = cls.parse_order_tag(attribution.tag)
            has_valid_tag = (
                parsed_tag is not None
                and parsed_tag[0] == trade.trade_direction
                and parsed_tag[1] == attribution.stage
            )
        return (
            has_valid_tag
            and order.order_date_utc.tzinfo is not None
            and order.order_date_utc == _parse_iso_datetime(attribution.order_date)
        )

    @staticmethod
    def _is_partial_wallet_mutation(
        attempt: TakeProfitAttempt | None,
        expected_amount: float,
        observed_amount: float,
    ) -> bool:
        return (
            attempt is not None
            and attempt.status in {"submitting", "ambiguous"}
            and (
                expected_amount > attempt.amount
                or TakeProfitStageManager._amounts_close(
                    expected_amount,
                    attempt.amount,
                )
            )
            and observed_amount < attempt.amount
            and observed_amount > attempt.amount * 0.98
        )

    @classmethod
    def _reconcile_attribution_order_amount(
        cls,
        attribution: TakeProfitOrderAttribution,
        order_amount: float,
        amount_quantizer: Callable[[float], float] | None,
    ) -> TakeProfitOrderAttribution | None:
        if cls._amounts_close(order_amount, attribution.order_amount):
            return attribution
        if attribution.source not in {"attempt", "recovered_untagged"}:
            return None
        try:
            canonical_attributed_amount = cls._canonical_amount(
                attribution.order_amount,
                amount_quantizer,
            )
        except ValueError:
            return None
        if (
            canonical_attributed_amount > 0.0
            and canonical_attributed_amount < attribution.order_amount
            and order_amount < attribution.order_amount
            and cls._amounts_close(order_amount, canonical_attributed_amount)
        ):
            return replace(attribution, order_amount=order_amount)
        return None

    @classmethod
    def _prepare_terminal_attribution(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        order: _Order,
        amount_quantizer: Callable[[float], float] | None,
    ) -> TakeProfitStageState:
        if state.blocked_order_id != order.order_id:
            raise ValueError("Canonical order does not match the blocked order")
        matching_attributions = [
            attribution
            for attribution in state.order_attributions
            if attribution.order_id == order.order_id
        ]
        if len(matching_attributions) > 1:
            raise ValueError("Expected exactly one matching order attribution")
        order_amount = cls._safe_amount(order.safe_amount)
        if order_amount <= 0.0:
            raise ValueError("Canonical order has no positive amount")
        order_date = order.order_date_utc
        if order_date.tzinfo is None:
            raise ValueError(
                "Canonical order creation timestamp must be timezone-aware"
            )

        if matching_attributions:
            attribution = matching_attributions[0]
            if order.ft_order_side != attribution.side:
                raise ValueError(
                    "Canonical order side must be repaired before revalidation"
                )
            if order.ft_order_tag != attribution.tag:
                raise ValueError(
                    "Canonical order tag must be repaired before revalidation"
                )
            if order_amount > attribution.maximum_amount and not cls._amounts_close(
                order_amount,
                attribution.maximum_amount,
            ):
                raise ValueError(
                    "Canonical order amount exceeds the attributed maximum"
                )
            if order_date != _parse_iso_datetime(attribution.order_date):
                if (
                    attribution.attempt_submitted_at is None
                    or attribution.retry_delay_seconds is None
                ):
                    raise ValueError(
                        "Attributed order date has no causal submission window"
                    )
                submitted_at = _parse_iso_datetime(attribution.attempt_submitted_at)
                if not (
                    submitted_at
                    <= order_date
                    <= submitted_at
                    + datetime.timedelta(seconds=attribution.retry_delay_seconds)
                ):
                    raise ValueError("Canonical order is outside its submission window")
            repaired_attribution = replace(
                attribution,
                order_date=order_date.isoformat(),
                order_amount=order_amount,
            )
            return replace(
                state,
                order_attributions=tuple(
                    repaired_attribution
                    if existing.order_id == order.order_id
                    else existing
                    for existing in state.order_attributions
                ),
            )

        parsed_tag = cls.parse_order_tag(order.ft_order_tag)
        configured_stages = {
            *(stage for stage, _ in state.stage_targets),
            state.plan_signature.final_stage.stage,
        }
        if (
            parsed_tag is None
            or parsed_tag[0] != trade.trade_direction
            or parsed_tag[1] not in configured_stages
        ):
            return state
        if order.ft_order_side != trade.exit_side:
            raise ValueError("Canonical order is not a strategy exit")
        stage = parsed_tag[1]
        causal_attempt = state.active_attempt
        if causal_attempt is not None and (
            causal_attempt.stage != stage
            or causal_attempt.tag != order.ft_order_tag
            or not cls._is_recovered_attempt_candidate(
                order,
                causal_attempt,
                amount_quantizer,
            )
        ):
            raise ValueError("Canonical order does not match the submitted attempt")
        operator_attribution = TakeProfitOrderAttribution(
            order_id=order.order_id,
            attribution_sequence=(
                max(
                    (
                        attribution.attribution_sequence
                        for attribution in state.order_attributions
                    ),
                    default=-1,
                )
                + 1
            ),
            tag=order.ft_order_tag,
            side=order.ft_order_side,
            stage=stage,
            attempt_id=(
                causal_attempt.attempt_id if causal_attempt is not None else None
            ),
            order_date=order_date.isoformat(),
            maximum_amount=(
                causal_attempt.amount if causal_attempt is not None else order_amount
            ),
            order_amount=order_amount,
            source="operator",
            attempt_submitted_at=(
                causal_attempt.submitted_at or causal_attempt.created_at
                if causal_attempt is not None
                else None
            ),
            retry_delay_seconds=(
                cls._attempt_retry_delay_seconds(causal_attempt)
                if causal_attempt is not None
                else None
            ),
        )
        return replace(
            state,
            order_attributions=tuple(
                sorted(
                    (*state.order_attributions, operator_attribution),
                    key=lambda attribution: attribution.order_id,
                )
            ),
            active_attempt=None if causal_attempt is not None else state.active_attempt,
        )

    @classmethod
    def _is_recovered_attempt_candidate(
        cls,
        order: _Order,
        attempt: TakeProfitAttempt,
        amount_quantizer: Callable[[float], float] | None,
    ) -> bool:
        order_amount = cls._safe_amount(order.safe_amount)
        canonical_attempt_amount = cls._canonical_amount(
            attempt.amount,
            amount_quantizer,
        )
        amount_matches = (
            cls._amounts_close(order_amount, attempt.amount)
            or attempt.amount * 0.98 < order_amount < attempt.amount
            or (
                canonical_attempt_amount > 0.0
                and order_amount < attempt.amount
                and cls._amounts_close(
                    order_amount,
                    canonical_attempt_amount,
                )
            )
        )
        if (
            order.order_id in attempt.known_order_ids
            or order.ft_order_side != attempt.exit_side
            or not amount_matches
            or attempt.submitted_at is None
        ):
            return False
        has_exact_tag = order.ft_order_tag == attempt.tag
        if not (has_exact_tag or order.ft_order_tag is None):
            return False
        submitted_at = _parse_iso_datetime(attempt.submitted_at)
        recovery_deadline = _parse_iso_datetime(attempt.recovery_deadline)
        order_date = order.order_date_utc
        return (
            order_date.tzinfo is not None
            and submitted_at <= order_date <= recovery_deadline
        )

    @staticmethod
    def _block(
        state: TakeProfitStageState,
        reason: str,
        *,
        order_id: str | None = None,
    ) -> TakeProfitStageState:
        if reason in _BLOCKED_ORDER_ID_REASONS:
            order_id = _non_empty_string(order_id)
        elif order_id is not None:
            order_id = _non_empty_string(order_id)
        if state.blocked_reason is not None:
            return state
        return replace(
            state,
            blocked_reason=reason,
            blocked_order_id=order_id,
        )

    @staticmethod
    def _safe_amount(value: object) -> float:
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return 0.0
        return amount if math.isfinite(amount) and amount > 0.0 else 0.0

    @classmethod
    def _amounts_close(cls, left: float, right: float) -> bool:
        return math.isclose(
            left,
            right,
            rel_tol=cls._AMOUNT_REL_TOL,
            abs_tol=cls._AMOUNT_ABS_TOL,
        )

    @classmethod
    def _raw_and_canonical_amounts_match(
        cls,
        left: float,
        right: float,
        amount_quantizer: Callable[[float], float] | None,
    ) -> bool:
        return cls._amounts_close(left, right) and cls._amounts_close(
            cls._canonical_amount(left, amount_quantizer),
            cls._canonical_amount(right, amount_quantizer),
        )


def _require_exact_mapping_keys(
    value: Mapping[str, object],
    mapping_type: type[object],
    label: str,
) -> None:
    expected_keys = {field.name for field in fields(mapping_type)}
    actual_keys = set(value)
    if actual_keys == expected_keys:
        return
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    details = []
    if missing:
        details.append(f"missing {missing!r}")
    if unexpected:
        details.append(f"unexpected {unexpected!r}")
    raise ValueError(f"Invalid {label} keys: {', '.join(details)}")


def _required_string(value: Mapping[str, object], key: str) -> str:
    return _non_empty_string(value.get(key))


def _required_non_negative_int(value: Mapping[str, object], key: str) -> int:
    return _non_negative_int(value.get(key))


def _required_non_negative_float(value: Mapping[str, object], key: str) -> float:
    return _non_negative_float(value.get(key))


def _required_positive_float(value: Mapping[str, object], key: str) -> float:
    return _required_positive_value(value.get(key), key)


def _required_finite_float(value: Mapping[str, object], key: str) -> float:
    return _finite_float(value.get(key))


def _required_positive_value(value: object, name: str) -> float:
    parsed = _non_negative_float(value)
    if parsed == 0.0:
        raise ValueError(f"Invalid positive float value for {name!r}")
    return parsed


def _non_empty_string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid non-empty string value {value!r}")
    return value


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid non-negative integer value {value!r}")
    return value


def _non_negative_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid non-negative float value {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid non-negative float value {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"Invalid non-negative float value {value!r}")
    return parsed


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Expected a finite number") from error
    if not math.isfinite(parsed):
        raise ValueError("Expected a finite number")
    return parsed


def _parse_iso_datetime(value: str) -> datetime.datetime:
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Invalid ISO datetime value {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"Timezone-naive datetime value {value!r}")
    return parsed


def _parse_amount_records(
    value: object,
    *,
    identifier_key: str,
    identifier_parser: Callable[[object], object],
) -> tuple[tuple, ...]:
    if not isinstance(value, list):
        raise ValueError(f"Invalid take-profit {identifier_key} amount records")
    records = []
    for record in value:
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid take-profit {identifier_key} amount record")
        expected_keys = {identifier_key, "amount"}
        if set(record) != expected_keys:
            raise ValueError(f"Invalid take-profit {identifier_key} amount record keys")
        records.append(
            (
                identifier_parser(record.get(identifier_key)),
                _non_negative_float(record.get("amount")),
            )
        )
    identifiers = [identifier for identifier, _ in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate take-profit {identifier_key} amount record")
    return tuple(records)
