import datetime
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal, Protocol, Self


TakeProfitAttemptStatus = Literal["proposed", "submitting", "ambiguous"]
TakeProfitOperatorAction = Literal[
    "confirmed_no_remote_order",
    "confirmed_terminal_zero_fill",
    "revalidated_terminal_fill",
]
TakeProfitOrderRole = Literal["strategy_exit", "external_exit"]


_OPERATOR_ZERO_FILL_STATUSES = frozenset({"canceled", "cancelled"})
_BLOCKED_ORDER_ID_REASONS = frozenset(
    {
        "confirmed_terminal_zero_fill_changed",
        "confirmed_terminal_zero_fill_order_missing",
        "multiple_attempt_orders",
        "unknown_attempt_order",
        "unknown_external_exit_fill",
        "unknown_terminal_fill",
    }
)


class _Order(Protocol):
    @property
    def filled(self) -> float | None: ...

    @property
    def ft_fee_base(self) -> float | None: ...

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
    def safe_filled(self) -> float: ...

    @property
    def safe_remaining(self) -> float: ...

    @property
    def status(self) -> str | None: ...


class _Trade(Protocol):
    @property
    def amount(self) -> float: ...

    @property
    def entry_side(self) -> str: ...

    @property
    def exit_side(self) -> str: ...

    @property
    def orders(self) -> Sequence[_Order]: ...

    @property
    def stake_amount(self) -> float: ...

    @property
    def trade_direction(self) -> str: ...


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
        if parsed_tag is None or parsed_tag[1] != stage or parsed_tag[2] != attempt_id:
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
                or self.terminal_status not in _OPERATOR_ZERO_FILL_STATUSES
            ):
                raise ValueError("Terminal zero-fill resolution has invalid fields")
        elif self.action == "revalidated_terminal_fill":
            _non_empty_string(self.order_id)
            if (
                self.attempt_id is not None
                or self.credited_amount is None
                or self.terminal_status is not None
            ):
                raise ValueError("Terminal-fill resolution has invalid fields")
            _non_negative_float(self.credited_amount)
        else:
            raise ValueError(f"Invalid take-profit operator action {self.action!r}")

    def to_mapping(self) -> dict[str, object]:
        return {
            "action": self.action,
            "attempt_id": self.attempt_id,
            "order_id": self.order_id,
            "credited_amount": self.credited_amount,
            "terminal_status": self.terminal_status,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        action = value.get("action")
        if action not in {
            "confirmed_no_remote_order",
            "confirmed_terminal_zero_fill",
            "revalidated_terminal_fill",
        }:
            raise ValueError(f"Invalid take-profit operator action {action!r}")
        attempt_id = value.get("attempt_id")
        order_id = value.get("order_id")
        credited_amount = value.get("credited_amount")
        terminal_status = value.get("terminal_status")
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
        )


@dataclass(frozen=True, slots=True)
class TakeProfitZeroFillProof:
    """Immutable canonical snapshot accepted by an explicit operator action."""

    order_id: str
    order_role: TakeProfitOrderRole
    order_side: str
    terminal_status: str
    amount: float
    remaining: float
    fee_base: float
    confirmed_at: str

    def __post_init__(self) -> None:
        _non_empty_string(self.order_id)
        if self.order_role not in {"strategy_exit", "external_exit"}:
            raise ValueError("Invalid terminal zero-fill order role")
        _non_empty_string(self.order_side)
        if self.terminal_status not in _OPERATOR_ZERO_FILL_STATUSES:
            raise ValueError("Invalid operator-confirmed zero-fill status")
        amount = _required_positive_value(self.amount, "zero-fill proof amount")
        remaining = _non_negative_float(self.remaining)
        if not math.isclose(remaining, amount, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Terminal zero-fill proof has an inconsistent remainder")
        if _non_negative_float(self.fee_base) != 0.0:
            raise ValueError("Terminal zero-fill proof has a base fee")
        _parse_iso_datetime(self.confirmed_at)

    def to_mapping(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "order_role": self.order_role,
            "order_side": self.order_side,
            "terminal_status": self.terminal_status,
            "amount": self.amount,
            "remaining": self.remaining,
            "fee_base": self.fee_base,
            "confirmed_at": self.confirmed_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        order_role = value.get("order_role")
        if order_role not in {"strategy_exit", "external_exit"}:
            raise ValueError("Invalid terminal zero-fill order role")
        return cls(
            order_id=_required_string(value, "order_id"),
            order_role=order_role,
            order_side=_required_string(value, "order_side"),
            terminal_status=_required_string(value, "terminal_status"),
            amount=_required_positive_float(value, "amount"),
            remaining=_required_non_negative_float(value, "remaining"),
            fee_base=_required_non_negative_float(value, "fee_base"),
            confirmed_at=_required_string(value, "confirmed_at"),
        )


@dataclass(frozen=True, slots=True)
class TakeProfitStageState:
    version: int
    plan_signature: TakeProfitPlanSignature
    initial_amount: float
    stage_targets: tuple[tuple[int, float], ...]
    deferred_stages: tuple[int, ...]
    attributed_order_ids: tuple[str, ...]
    credited_order_amounts: tuple[tuple[str, float], ...]
    entry_order_amounts: tuple[tuple[str, float], ...]
    confirmed_zero_fill_orders: tuple[TakeProfitZeroFillProof, ...]
    operator_resolutions: tuple[TakeProfitOperatorResolution, ...]
    active_attempt: TakeProfitAttempt | None = None
    blocked_reason: str | None = None
    blocked_order_id: str | None = None

    SCHEMA_VERSION = 4
    MAX_OPERATOR_RESOLUTIONS = 32

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
            "attributed_order_ids": list(self.attributed_order_ids),
            "credited_order_amounts": [
                {"order_id": order_id, "amount": amount}
                for order_id, amount in self.credited_order_amounts
            ],
            "entry_order_amounts": [
                {"order_id": order_id, "amount": amount}
                for order_id, amount in self.entry_order_amounts
            ],
            "confirmed_zero_fill_orders": [
                proof.to_mapping() for proof in self.confirmed_zero_fill_orders
            ],
            "operator_resolutions": [
                resolution.to_mapping() for resolution in self.operator_resolutions
            ],
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

        attributed_raw = value.get("attributed_order_ids")
        if not isinstance(attributed_raw, list) or not all(
            isinstance(order_id, str) and order_id for order_id in attributed_raw
        ):
            raise ValueError("Invalid take-profit attributed_order_ids")
        attributed_order_ids = tuple(attributed_raw)
        if len(attributed_order_ids) != len(set(attributed_order_ids)):
            raise ValueError("Duplicate attributed take-profit order")

        credited_order_amounts = _parse_amount_records(
            value.get("credited_order_amounts"),
            identifier_key="order_id",
            identifier_parser=_non_empty_string,
        )
        if not set(order_id for order_id, _ in credited_order_amounts).issubset(
            attributed_order_ids
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

        entry_order_amounts = _parse_amount_records(
            value.get("entry_order_amounts"),
            identifier_key="order_id",
            identifier_parser=_non_empty_string,
        )

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
        revalidated_order_ids = tuple(
            resolution.order_id
            for resolution in operator_resolutions
            if resolution.action == "revalidated_terminal_fill"
        )
        if len(revalidated_order_ids) != len(set(revalidated_order_ids)):
            raise ValueError("Duplicate take-profit terminal-fill resolution")

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
            and active_attempt.attempt_id in resolved_attempt_ids
        ):
            raise ValueError("Resolved take-profit attempt cannot remain active")

        blocked_reason = value.get("blocked_reason")
        if blocked_reason is not None and not isinstance(blocked_reason, str):
            raise ValueError("Invalid take-profit blocked_reason")
        blocked_order_id = value.get("blocked_order_id")
        if blocked_reason in _BLOCKED_ORDER_ID_REASONS:
            blocked_order_id = _non_empty_string(blocked_order_id)
        elif blocked_reason is None and blocked_order_id is not None:
            raise ValueError("Take-profit blocked_order_id requires a blocked state")
        elif blocked_order_id is not None:
            blocked_order_id = _non_empty_string(blocked_order_id)

        return cls(
            version=version,
            plan_signature=plan_signature,
            initial_amount=initial_amount,
            stage_targets=tuple(sorted(stage_targets)),
            deferred_stages=tuple(sorted(deferred_stages)),
            attributed_order_ids=tuple(sorted(attributed_order_ids)),
            credited_order_amounts=tuple(sorted(credited_order_amounts)),
            entry_order_amounts=tuple(sorted(entry_order_amounts)),
            confirmed_zero_fill_orders=tuple(
                sorted(
                    confirmed_zero_fill_orders,
                    key=lambda proof: proof.order_id,
                )
            ),
            operator_resolutions=operator_resolutions,
            active_attempt=active_attempt,
            blocked_reason=blocked_reason,
            blocked_order_id=blocked_order_id,
        )


@dataclass(frozen=True, slots=True)
class TakeProfitStageProgress:
    stage: int
    target_amount: float
    filled_amount: float
    remaining_amount: float
    is_complete: bool
    is_deferred: bool
    has_deferred_debt: bool


class TakeProfitStageManager:
    """Reconcile take-profit stages with causally attributed Freqtrade orders."""

    STATE_KEY = "take_profit_state"
    _TAG_PREFIX = "take_profit"
    _EMERGENCY_EXIT_TAG = "emergency_exit"
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
        attempt_id: str | None = None,
    ) -> str:
        if stage < 0:
            raise ValueError(
                f"Invalid take-profit stage {stage!r}: must be non-negative"
            )
        tag = f"{cls._TAG_PREFIX}_{trade_direction}_{stage}"
        if attempt_id is not None:
            if not attempt_id or "_" in attempt_id:
                raise ValueError(f"Invalid take-profit attempt ID {attempt_id!r}")
            tag = f"{tag}_{attempt_id}"
        return tag

    @classmethod
    def has_order_tag_prefix(cls, value: object) -> bool:
        return isinstance(value, str) and value.startswith(f"{cls._TAG_PREFIX}_")

    @classmethod
    def parse_order_tag(
        cls,
        tag: object,
    ) -> tuple[str, int, str | None] | None:
        if not isinstance(tag, str):
            return None
        parts = tag.split("_")
        if len(parts) not in {4, 5} or parts[:2] != ["take", "profit"]:
            return None
        try:
            stage = int(parts[3])
        except ValueError:
            return None
        if stage < 0 or not parts[2]:
            return None
        attempt_id = parts[4] if len(parts) == 5 and parts[4] else None
        return parts[2], stage, attempt_id

    @classmethod
    def initialize(
        cls,
        trade: _Trade,
        plan_signature: TakeProfitPlanSignature,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> TakeProfitStageState:
        configured_stages = {
            definition.stage for definition in plan_signature.partial_stages
        }
        strategy_exit_orders = [
            order for order in trade.orders if order.ft_order_side == trade.exit_side
        ]
        blocked_reason = None
        blocked_order_id = None
        if any(
            order.ft_order_side == trade.entry_side and order.ft_is_open
            for order in trade.orders
        ):
            blocked_reason = "open_entry_order"
        for order in trade.orders:
            if order.ft_order_side in {trade.entry_side, trade.exit_side}:
                continue
            external_block = cls._external_exit_block(order)
            if external_block is not None and blocked_reason is None:
                blocked_reason = external_block
                blocked_order_id = order.order_id

        terminal_credits: dict[str, float] = {}
        for order in strategy_exit_orders:
            if order.ft_is_open:
                continue
            try:
                terminal_credits[order.order_id] = cls._terminal_credited_amount(order)
            except ValueError:
                if blocked_reason is None:
                    blocked_reason = "unknown_terminal_fill"
                    blocked_order_id = order.order_id

        attributed_order_ids: set[str] = set()
        for order in strategy_exit_orders:
            parsed_tag = cls.parse_order_tag(order.ft_order_tag)
            if parsed_tag is None:
                continue
            direction, stage, attempt_id = parsed_tag
            if attempt_id is not None:
                if blocked_reason is None:
                    blocked_reason = "unknown_attempt_order"
                    blocked_order_id = order.order_id
                continue
            if (
                direction != trade.trade_direction
                or stage not in configured_stages
                or order.ft_order_side != trade.exit_side
            ):
                if terminal_credits.get(order.order_id, 0.0) > 0.0:
                    blocked_reason = blocked_reason or "unattributed_exit_fill"
                continue
            attributed_order_ids.add(order.order_id)

        for index, order in enumerate(strategy_exit_orders):
            if order.ft_order_tag != cls._EMERGENCY_EXIT_TAG:
                continue
            predecessor = strategy_exit_orders[index - 1] if index > 0 else None
            if predecessor is not None and cls._is_legacy_emergency_successor(
                trade,
                predecessor,
                order,
                configured_stages,
            ):
                attributed_order_ids.add(order.order_id)
            elif terminal_credits.get(order.order_id, 0.0) > 0.0:
                blocked_reason = blocked_reason or "ambiguous_legacy_exit"

        credited_order_amounts: dict[str, float] = {}
        for order in strategy_exit_orders:
            if order.order_id in attributed_order_ids and not order.ft_is_open:
                credited_order_amounts[order.order_id] = terminal_credits.get(
                    order.order_id,
                    0.0,
                )
            elif (
                order.order_id not in attributed_order_ids
                and terminal_credits.get(order.order_id, 0.0) > 0.0
            ):
                blocked_reason = blocked_reason or "unattributed_exit_fill"

        entry_order_amounts = cls._entry_order_amounts(trade)
        if entry_order_amounts:
            initial_amount = cls._canonical_amount_sum(
                entry_order_amounts.values(),
                amount_quantizer,
            )
        else:
            blocked_reason = blocked_reason or "missing_entry_exposure"
            initial_amount = cls._canonical_amount(
                cls._safe_amount(trade.amount),
                amount_quantizer,
            )
        if initial_amount <= 0.0:
            raise ValueError("Cannot initialize take-profit without positive exposure")
        stage_targets = plan_signature.derive_stage_targets(initial_amount)

        if not cls._trade_amount_is_consistent(
            trade,
            entry_order_amounts.values(),
            credited_order_amounts.values(),
            amount_quantizer,
        ):
            blocked_reason = blocked_reason or "trade_amount_mismatch"

        return TakeProfitStageState(
            version=TakeProfitStageState.SCHEMA_VERSION,
            plan_signature=plan_signature,
            initial_amount=initial_amount,
            stage_targets=stage_targets,
            deferred_stages=(),
            attributed_order_ids=tuple(sorted(attributed_order_ids)),
            credited_order_amounts=tuple(sorted(credited_order_amounts.items())),
            entry_order_amounts=tuple(sorted(entry_order_amounts.items())),
            confirmed_zero_fill_orders=(),
            operator_resolutions=(),
            blocked_reason=blocked_reason,
            blocked_order_id=blocked_order_id,
        )

    @classmethod
    def reconcile(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> TakeProfitStageState:
        if state.blocked_reason is not None:
            return state

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

        configured_stages = {stage for stage, _ in state.stage_targets}
        attributed_order_ids = set(state.attributed_order_ids)
        credited_order_amounts = dict(state.credited_order_amounts)
        strategy_exit_orders = [
            order for order in trade.orders if order.ft_order_side == trade.exit_side
        ]
        for order in trade.orders:
            if order.ft_order_side in {trade.entry_side, trade.exit_side}:
                continue
            external_block = cls._external_exit_block(
                order,
                confirmed_zero_fill_orders.get(order.order_id),
            )
            if external_block is not None:
                return cls._block(
                    state,
                    external_block,
                    order_id=order.order_id,
                )

        terminal_credits: dict[str, float] = {}
        for order in strategy_exit_orders:
            if order.ft_is_open:
                continue
            try:
                terminal_credits[order.order_id] = cls._terminal_credited_amount(
                    order,
                    confirmed_zero_fill_orders.get(order.order_id),
                )
            except ValueError:
                return cls._block(
                    state,
                    "unknown_terminal_fill",
                    order_id=order.order_id,
                )

        active_attempt = state.active_attempt
        matching_attempt_orders: list[_Order] = []
        if active_attempt is not None:
            same_attempt_orders = [
                order
                for order in strategy_exit_orders
                if (parsed_tag := cls.parse_order_tag(order.ft_order_tag)) is not None
                and parsed_tag[2] == active_attempt.attempt_id
            ]
            if len(same_attempt_orders) > 1:
                return cls._block(
                    state,
                    "multiple_attempt_orders",
                    order_id=same_attempt_orders[0].order_id,
                )
            matching_attempt_orders = [
                order
                for order in same_attempt_orders
                if order.ft_order_tag == active_attempt.tag
                and order.ft_order_side == active_attempt.exit_side
            ]

        for order in strategy_exit_orders:
            if order.order_id in attributed_order_ids:
                continue
            parsed_tag = cls.parse_order_tag(order.ft_order_tag)
            if parsed_tag is None:
                continue
            if order.order_id in confirmed_zero_fill_orders:
                continue
            if order in matching_attempt_orders:
                continue
            return cls._block(
                state,
                "unknown_attempt_order",
                order_id=order.order_id,
            )

        for index, order in enumerate(strategy_exit_orders):
            if order.order_id in attributed_order_ids:
                continue
            predecessor = strategy_exit_orders[index - 1] if index > 0 else None
            if (
                order.ft_order_tag == cls._EMERGENCY_EXIT_TAG
                and predecessor is not None
                and cls._is_legacy_emergency_successor(
                    trade,
                    predecessor,
                    order,
                    configured_stages,
                )
            ):
                attributed_order_ids.add(order.order_id)

        if active_attempt is not None:
            attributed_order_ids.update(
                order.order_id for order in matching_attempt_orders
            )

            if not matching_attempt_orders and active_attempt.status in {
                "submitting",
                "ambiguous",
            }:
                recovered_candidates = [
                    order
                    for order in strategy_exit_orders
                    if cls._is_recovered_attempt_candidate(order, active_attempt)
                ]
                if len(recovered_candidates) == 1:
                    matching_attempt_orders = recovered_candidates
                    attributed_order_ids.add(recovered_candidates[0].order_id)
                elif len(recovered_candidates) > 1:
                    return cls._block(state, "ambiguous_recovered_exit")

        for order_id in attributed_order_ids:
            order = orders_by_id.get(order_id)
            if order is None:
                return cls._block(state, "attributed_order_missing")
            if order.ft_order_side != trade.exit_side:
                return cls._block(state, "attributed_order_side_changed")
            previous_amount = credited_order_amounts.get(order_id)
            if order.ft_is_open:
                if previous_amount is not None:
                    return cls._block(state, "attributed_order_reopened")
                continue
            current_amount = terminal_credits[order_id]
            if (
                previous_amount is not None
                and current_amount < previous_amount
                and not (cls._amounts_close(current_amount, previous_amount))
            ):
                return cls._block(state, "attributed_fill_regressed")
            credited_order_amounts[order_id] = current_amount

        total_credited = math.fsum(credited_order_amounts.values())
        if total_credited > state.initial_amount and not cls._amounts_close(
            total_credited,
            state.initial_amount,
        ):
            return cls._block(state, "credited_fill_exceeds_initial_amount")

        for order in strategy_exit_orders:
            if (
                order.order_id not in attributed_order_ids
                and terminal_credits.get(order.order_id, 0.0) > 0.0
            ):
                return cls._block(state, "unattributed_exit_fill")

        if not cls._trade_amount_is_consistent(
            trade,
            persisted_entry_amounts.values(),
            credited_order_amounts.values(),
            amount_quantizer,
        ):
            return cls._block(state, "trade_amount_mismatch")

        if matching_attempt_orders:
            active_attempt = None

        return replace(
            state,
            attributed_order_ids=tuple(sorted(attributed_order_ids)),
            credited_order_amounts=tuple(sorted(credited_order_amounts.items())),
            active_attempt=active_attempt,
        )

    @classmethod
    def get_stage_progress(
        cls,
        trade: _Trade,
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
    def get_exit_stage(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
    ) -> int:
        cumulative_filled = math.fsum(
            amount for _, amount in state.credited_order_amounts
        )
        cumulative_target = 0.0
        deferred_stages = set(state.deferred_stages)
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
        return cls._canonical_signed_amount_sum(
            (amount for _, amount in state.entry_order_amounts),
            (amount for _, amount in state.credited_order_amounts),
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
        observed = cls._canonical_amount(trade.amount, amount_quantizer)
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
        if current_time.tzinfo is None or recovery_deadline.tzinfo is None:
            raise ValueError("Take-profit attempt timestamps must be timezone-aware")
        if recovery_deadline < current_time:
            raise ValueError("Take-profit recovery deadline precedes creation")
        amount = cls._safe_amount(amount)
        stake_amount = cls._safe_amount(stake_amount)
        if amount == 0.0 or stake_amount == 0.0:
            raise ValueError("Take-profit attempt amounts must be positive")
        tag = cls.order_tag(trade.trade_direction, stage, attempt_id)
        submitted_at = current_time.isoformat() if status == "submitting" else None
        return replace(
            state,
            active_attempt=TakeProfitAttempt(
                attempt_id=attempt_id,
                stage=stage,
                status=status,
                amount=amount,
                stake_amount=stake_amount,
                exit_side=trade.exit_side,
                tag=tag,
                created_at=current_time.isoformat(),
                submitted_at=submitted_at,
                recovery_deadline=recovery_deadline.isoformat(),
                known_order_ids=tuple(sorted(order.order_id for order in trade.orders)),
            ),
        )

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
        amount = cls._safe_amount(amount)
        stake_amount = cls._safe_amount(stake_amount)
        if amount == 0.0 or stake_amount == 0.0:
            raise ValueError("Take-profit attempt amounts must be positive")
        return replace(
            state,
            active_attempt=replace(
                attempt,
                amount=amount,
                stake_amount=stake_amount,
            ),
        )

    @staticmethod
    def mark_ambiguous(
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
        return replace(state, active_attempt=replace(attempt, status="ambiguous"))

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
    def revalidate_unknown_terminal_fill(
        cls,
        trade: _Trade,
        state: TakeProfitStageState,
        order_id: str,
        resolved_at: datetime.datetime,
        *,
        amount_quantizer: Callable[[float], float] | None = None,
    ) -> TakeProfitStageState:
        """Reconcile one repaired canonical order and record the operator action."""
        order_id = _non_empty_string(order_id)
        if (
            state.blocked_reason != "unknown_terminal_fill"
            or state.blocked_order_id != order_id
        ):
            raise ValueError("No matching unknown terminal fill to revalidate")
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
        credited_amount = cls._terminal_credited_amount(order)

        candidate = replace(
            state,
            blocked_reason=None,
            blocked_order_id=None,
        )
        reconciled = cls.reconcile(
            trade,
            candidate,
            amount_quantizer=amount_quantizer,
        )
        if reconciled.blocked_reason is not None:
            raise ValueError(
                "Canonical order repair does not produce a valid take-profit state: "
                f"{reconciled.blocked_reason}"
            )

        attributed = order_id in reconciled.attributed_order_ids
        persisted_credit = dict(reconciled.credited_order_amounts).get(order_id)
        if credited_amount > 0.0 and not attributed:
            raise ValueError("Positive terminal fill is not attributed to take-profit")
        if attributed and (
            persisted_credit is None
            or not cls._amounts_close(persisted_credit, credited_amount)
        ):
            raise ValueError("Revalidated take-profit credit does not match the order")

        resolution = TakeProfitOperatorResolution(
            action="revalidated_terminal_fill",
            order_id=order_id,
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
        order_id = _non_empty_string(order_id)
        terminal_status = _non_empty_string(terminal_status).lower()
        if terminal_status not in _OPERATOR_ZERO_FILL_STATUSES:
            raise ValueError("Terminal zero-fill status is not operator-recoverable")
        if (
            state.blocked_reason
            not in {"unknown_terminal_fill", "unknown_external_exit_fill"}
            or state.blocked_order_id != order_id
        ):
            raise ValueError("No matching unknown terminal fill to confirm")
        if resolved_at.tzinfo is None:
            raise ValueError("Take-profit resolution timestamp must be timezone-aware")
        if any(
            proof.order_id == order_id for proof in state.confirmed_zero_fill_orders
        ):
            raise ValueError("Terminal zero-fill order is already confirmed")

        matching_orders = [
            order for order in trade.orders if order.order_id == order_id
        ]
        if len(matching_orders) != 1:
            raise ValueError("Expected exactly one matching canonical order")
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
                    (*state.confirmed_zero_fill_orders, proof),
                    key=lambda item: item.order_id,
                )
            ),
            blocked_reason=None,
            blocked_order_id=None,
        )
        reconciled = cls.reconcile(
            trade,
            candidate,
            amount_quantizer=amount_quantizer,
        )
        if reconciled.blocked_reason in {
            "confirmed_terminal_zero_fill_changed",
            "confirmed_terminal_zero_fill_order_missing",
        }:
            raise ValueError("Terminal zero-fill proof failed its postcondition")

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

        if status == "rejected" or order.order_filled_utc is None:
            raise ValueError("Terminal positive fill is not explicitly observed")
        if filled < amount and remaining is None:
            raise ValueError("Terminal partial fill has no reported remainder")
        if fee_base >= filled or cls._amounts_close(fee_base, filled):
            raise ValueError("Terminal order base fee consumes its entire fill")

        return filled - fee_base

    @classmethod
    def _external_exit_block(
        cls,
        order: _Order,
        zero_fill_proof: TakeProfitZeroFillProof | None = None,
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
            credited_amount = cls._terminal_credited_amount(order, zero_fill_proof)
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
        if order.ft_order_side == trade.exit_side:
            order_role: TakeProfitOrderRole = "strategy_exit"
        elif order.ft_order_side != trade.entry_side:
            order_role = "external_exit"
        else:
            raise ValueError("Entry orders cannot be confirmed as terminal exits")
        if (order.status or "").lower() != terminal_status:
            raise ValueError("Canonical order status does not match terminal-status")
        amount, remaining, fee_base = cls._validate_terminal_zero_fill(
            order,
            _OPERATOR_ZERO_FILL_STATUSES,
        )
        return TakeProfitZeroFillProof(
            order_id=order.order_id,
            order_role=order_role,
            order_side=order.ft_order_side,
            terminal_status=terminal_status,
            amount=amount,
            remaining=remaining,
            fee_base=fee_base,
            confirmed_at=confirmed_at.isoformat(),
        )

    @classmethod
    def _zero_fill_proof_matches(
        cls,
        trade: _Trade,
        order: _Order,
        proof: TakeProfitZeroFillProof,
    ) -> bool:
        if proof.order_role == "strategy_exit":
            expected_side = trade.exit_side
        else:
            if order.ft_order_side in {trade.entry_side, trade.exit_side}:
                return False
            expected_side = proof.order_side
        if order.ft_order_side != expected_side:
            return False
        return cls._zero_fill_proof_fields_match(order, proof)

    @classmethod
    def _zero_fill_proof_fields_match(
        cls,
        order: _Order,
        proof: TakeProfitZeroFillProof,
    ) -> bool:
        try:
            amount, remaining, fee_base = cls._validate_terminal_zero_fill(
                order,
                {proof.terminal_status},
            )
        except ValueError:
            return False
        return (
            order.order_id == proof.order_id
            and order.ft_order_side == proof.order_side
            and (order.status or "").lower() == proof.terminal_status
            and amount == proof.amount
            and remaining == proof.remaining
            and fee_base == proof.fee_base
        )

    @classmethod
    def _validate_terminal_zero_fill(
        cls,
        order: _Order,
        allowed_statuses: set[str] | frozenset[str],
    ) -> tuple[float, float, float]:
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
        if order.order_filled_utc is not None:
            raise ValueError("Terminal zero-fill order has a fill timestamp")
        return amount, remaining, fee_base

    @classmethod
    def _canonical_amount(
        cls,
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

    @classmethod
    def _trade_amount_is_consistent(
        cls,
        trade: _Trade,
        entry_amounts: Iterable[float],
        credited_amounts: Iterable[float],
        amount_quantizer: Callable[[float], float] | None,
    ) -> bool:
        try:
            expected_amount = cls._canonical_signed_amount_sum(
                entry_amounts,
                credited_amounts,
                amount_quantizer,
            )
            observed_amount = cls._canonical_amount(
                trade.amount,
                amount_quantizer,
            )
        except ValueError:
            return False
        return expected_amount >= 0.0 and cls._amounts_close(
            observed_amount,
            expected_amount,
        )

    @classmethod
    def _is_legacy_emergency_successor(
        cls,
        trade: _Trade,
        predecessor: _Order,
        emergency_order: _Order,
        configured_stages: set[int],
    ) -> bool:
        parsed_tag = cls.parse_order_tag(predecessor.ft_order_tag)
        if parsed_tag is None:
            return False
        direction, stage, _ = parsed_tag
        remaining_amount = cls._safe_amount(predecessor.safe_remaining)
        return (
            direction == trade.trade_direction
            and stage in configured_stages
            and not predecessor.ft_is_open
            and predecessor.ft_order_side == trade.exit_side
            and emergency_order.ft_order_side == trade.exit_side
            and remaining_amount > 0.0
            and cls._amounts_close(
                cls._safe_amount(emergency_order.safe_amount),
                remaining_amount,
            )
        )

    @classmethod
    def _is_recovered_attempt_candidate(
        cls,
        order: _Order,
        attempt: TakeProfitAttempt,
    ) -> bool:
        if (
            order.order_id in attempt.known_order_ids
            or order.ft_order_tag is not None
            or order.ft_order_side != attempt.exit_side
            or not cls._amounts_close(
                cls._safe_amount(order.safe_amount),
                attempt.amount,
            )
            or attempt.submitted_at is None
        ):
            return False
        submitted_at = datetime.datetime.fromisoformat(attempt.submitted_at)
        recovery_deadline = datetime.datetime.fromisoformat(attempt.recovery_deadline)
        return submitted_at <= order.order_date_utc <= recovery_deadline

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


def _required_string(value: Mapping[str, object], key: str) -> str:
    return _non_empty_string(value.get(key))


def _required_non_negative_int(value: Mapping[str, object], key: str) -> int:
    return _non_negative_int(value.get(key))


def _required_non_negative_float(value: Mapping[str, object], key: str) -> float:
    return _non_negative_float(value.get(key))


def _required_positive_float(value: Mapping[str, object], key: str) -> float:
    return _required_positive_value(value.get(key), key)


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
    if isinstance(value, bool):
        raise ValueError(f"Invalid non-negative float value {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid non-negative float value {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"Invalid non-negative float value {value!r}")
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
