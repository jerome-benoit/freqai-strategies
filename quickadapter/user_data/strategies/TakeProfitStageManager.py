import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


class _Order(Protocol):
    ft_order_side: str
    ft_order_tag: str | None
    ft_is_open: bool
    safe_amount: float
    safe_filled: float


class _Trade(Protocol):
    amount: float
    exit_side: str
    orders: Sequence[_Order]
    stake_amount: float
    trade_direction: str


@dataclass(frozen=True, slots=True)
class TakeProfitStageProgress:
    stage: int
    target_amount: float
    filled_amount: float
    remaining_amount: float
    is_complete: bool


class TakeProfitStageManager:
    """Derive take-profit progress from persisted Freqtrade orders."""

    _TAG_PREFIX = "take_profit"
    _AMOUNT_REL_TOL = 1e-9
    _AMOUNT_ABS_TOL = 1e-12

    @classmethod
    def order_tag(cls, trade_direction: str, stage: int) -> str:
        if stage < 0:
            raise ValueError(f"Invalid take-profit stage {stage!r}: must be non-negative")
        return f"{cls._TAG_PREFIX}_{trade_direction}_{stage}"

    @classmethod
    def _stage_orders(cls, trade: _Trade, stage: int) -> list[_Order]:
        order_tag = cls.order_tag(trade.trade_direction, stage)
        return [
            order
            for order in trade.orders
            if order.ft_order_side == trade.exit_side
            and order.ft_order_tag == order_tag
        ]

    @staticmethod
    def _safe_amount(value: object) -> float:
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return 0.0
        return amount if math.isfinite(amount) and amount > 0.0 else 0.0

    @classmethod
    def get_stage_progress(
        cls,
        trade: _Trade,
        stage: int,
        stake_fraction: float,
    ) -> TakeProfitStageProgress:
        if not 0.0 < stake_fraction <= 1.0:
            raise ValueError(
                f"Invalid take-profit stake fraction {stake_fraction!r}: "
                "must be in range (0, 1]"
            )

        orders = cls._stage_orders(trade, stage)
        target_amount = next(
            (
                amount
                for order in orders
                if (amount := cls._safe_amount(order.safe_amount)) > 0.0
            ),
            cls._safe_amount(trade.amount) * stake_fraction,
        )
        filled_amount = sum(cls._safe_amount(order.safe_filled) for order in orders)
        remaining_amount = max(0.0, target_amount - filled_amount)
        is_filled = filled_amount >= target_amount or math.isclose(
            filled_amount,
            target_amount,
            rel_tol=cls._AMOUNT_REL_TOL,
            abs_tol=cls._AMOUNT_ABS_TOL,
        )
        is_complete = bool(orders) and is_filled and not any(
            order.ft_is_open for order in orders
        )

        return TakeProfitStageProgress(
            stage=stage,
            target_amount=target_amount,
            filled_amount=filled_amount,
            remaining_amount=remaining_amount,
            is_complete=is_complete,
        )

    @classmethod
    def get_exit_stage(
        cls,
        trade: _Trade,
        stage_stake_fractions: Mapping[int, float],
    ) -> int:
        for stage in sorted(stage_stake_fractions):
            progress = cls.get_stage_progress(
                trade,
                stage,
                stage_stake_fractions[stage],
            )
            if not progress.is_complete:
                return stage
        return max(stage_stake_fractions, default=-1) + 1

    @classmethod
    def get_remaining_stake_amount(
        cls,
        trade: _Trade,
        progress: TakeProfitStageProgress,
    ) -> float:
        trade_amount = cls._safe_amount(trade.amount)
        trade_stake_amount = cls._safe_amount(trade.stake_amount)
        if trade_amount == 0.0 or trade_stake_amount == 0.0:
            return 0.0
        remaining_stake_amount = (
            progress.remaining_amount * trade_stake_amount / trade_amount
        )
        return min(trade_stake_amount, remaining_stake_amount)
