"""Inspect or recover a fail-closed QuickAdapter take-profit state.

Run this tool only while the bot is stopped and only after checking the CEX order
history. Repair canonical Freqtrade order data before revalidating a terminal fill.
"""

import argparse
import copy
import datetime
import hashlib
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import NoReturn

from freqtrade.exchange import amount_to_contract_precision
from freqtrade.persistence import Trade, init_db
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


STRATEGIES_DIR = Path(__file__).resolve().parents[1] / "strategies"
sys.path.insert(0, str(STRATEGIES_DIR))

from TakeProfitStageManager import (  # noqa: E402
    TakeProfitRecoveryAction,
    TakeProfitRecoveryInstruction,
    TakeProfitStageManager,
    TakeProfitStageState,
)


_CONFIRMATION_VERSION = 1


class MaintenanceError(RuntimeError):
    """Refuse an unsafe or inconsistent maintenance request."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--trade-id", required=True, type=int)
    parser.add_argument("--expected-pair", required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "inspect",
        help="Show take-profit data without modifying it",
    )
    clear_parser = commands.add_parser(
        "clear-ambiguous",
        help="Clear an ambiguous attempt after proving no remote order exists",
    )
    clear_parser.add_argument("--attempt-id", required=True)
    clear_parser.add_argument("--apply", action="store_true")
    clear_parser.add_argument("--confirmation")
    revalidate_parser = commands.add_parser(
        "revalidate-terminal-order",
        help="Reconcile a repaired canonical terminal order",
    )
    revalidate_parser.add_argument("--order-id", required=True)
    revalidate_parser.add_argument("--apply", action="store_true")
    revalidate_parser.add_argument("--confirmation")
    revalidate_state_parser = commands.add_parser(
        "revalidate-state",
        help="Clear an eligible block after the complete state was repaired",
    )
    revalidate_state_parser.add_argument("--apply", action="store_true")
    revalidate_state_parser.add_argument("--confirmation")
    zero_fill_parser = commands.add_parser(
        "confirm-terminal-zero-fill",
        help="Confirm an exact canceled zero-fill order after CEX verification",
    )
    zero_fill_parser.add_argument("--order-id", required=True)
    zero_fill_parser.add_argument(
        "--terminal-status",
        required=True,
        choices=("canceled", "cancelled"),
    )
    zero_fill_parser.add_argument("--apply", action="store_true")
    zero_fill_parser.add_argument("--confirmation")
    canceled_fill_parser = commands.add_parser(
        "confirm-terminal-canceled-fill",
        help="Confirm an exact canceled positive-fill order after CEX verification",
    )
    canceled_fill_parser.add_argument("--order-id", required=True)
    canceled_fill_parser.add_argument(
        "--terminal-status",
        required=True,
        choices=("canceled", "cancelled"),
    )
    canceled_fill_parser.add_argument("--apply", action="store_true")
    canceled_fill_parser.add_argument("--confirmation")
    return parser


def _fail(message: str) -> NoReturn:
    raise MaintenanceError(message)


def _load_trade(args: argparse.Namespace) -> Trade:
    if args.trade_id <= 0:
        _fail("trade-id must be positive")
    if args.db_url in {"sqlite://", "sqlite:///", "sqlite:///:memory:"}:
        _fail("db-url must identify a persistent database")
    try:
        database_url = make_url(args.db_url)
    except ArgumentError as error:
        raise MaintenanceError("db-url is invalid") from error
    if database_url.drivername.startswith("sqlite"):
        database_path = Path(database_url.database or "")
        if not database_path.is_absolute():
            database_path = Path.cwd() / database_path
        if not database_path.is_file():
            _fail("SQLite database does not exist")

    init_db(args.db_url)
    trades = Trade.get_trades(Trade.id == args.trade_id).all()
    if len(trades) != 1:
        _fail(f"expected exactly one trade, found {len(trades)}")
    trade = trades[0]
    if trade.pair != args.expected_pair:
        _fail("trade pair does not match expected-pair")
    if trade.strategy != "QuickAdapterV3":
        _fail("trade does not belong to QuickAdapterV3")
    return trade


def _load_state(trade: Trade) -> TakeProfitStageState:
    raw_state = trade.get_custom_data(TakeProfitStageManager.STATE_KEY)
    if not isinstance(raw_state, dict):
        _fail("trade has no valid take-profit state mapping")
    try:
        state = TakeProfitStageState.from_mapping(raw_state)
        return TakeProfitStageManager.reconcile(
            trade,
            state,
            amount_quantizer=partial(_quantize_trade_amount, trade),
        )
    except ValueError as error:
        raise MaintenanceError(f"invalid take-profit state: {error}") from error


def _order_summary(order) -> dict[str, object]:
    return {
        "database_id": order.id,
        "trade_id": order.ft_trade_id,
        "order_id": order.order_id,
        "pair": order.ft_pair,
        "tag": order.ft_order_tag,
        "side": order.ft_order_side,
        "exchange_side": order.side,
        "status": order.status,
        "is_open": order.ft_is_open,
        "type": order.order_type,
        "symbol": order.symbol,
        "cancel_reason": order.ft_cancel_reason,
        "ft_amount": order.ft_amount,
        "ft_price": order.ft_price,
        "amount": order.amount,
        "safe_amount": order.safe_amount,
        "safe_amount_after_fee": order.safe_amount_after_fee,
        "filled": order.filled,
        "remaining": order.remaining,
        "fee_base": order.ft_fee_base,
        "price": order.price,
        "average": order.average,
        "stop_price": order.stop_price,
        "safe_price": order.safe_price,
        "cost": order.cost,
        "funding_fee": order.funding_fee,
        "order_date": order.order_date_utc.isoformat(),
        "filled_date": (
            order.order_filled_utc.isoformat()
            if order.order_filled_utc is not None
            else None
        ),
        "update_date": (
            order.order_update_date.isoformat()
            if order.order_update_date is not None
            else None
        ),
    }


def _confirmation_trade_summary(trade: Trade) -> dict[str, object]:
    """Return the raw trade fields that can affect recovery or accounting."""
    return {
        "id": trade.id,
        "exchange": trade.exchange,
        "pair": trade.pair,
        "base_currency": trade.base_currency,
        "stake_currency": trade.stake_currency,
        "is_open": trade.is_open,
        "fee_open": trade.fee_open,
        "fee_open_cost": trade.fee_open_cost,
        "fee_open_currency": trade.fee_open_currency,
        "fee_close": trade.fee_close,
        "fee_close_cost": trade.fee_close_cost,
        "fee_close_currency": trade.fee_close_currency,
        "open_rate": trade.open_rate,
        "open_rate_requested": trade.open_rate_requested,
        "open_trade_value": trade.open_trade_value,
        "close_rate": trade.close_rate,
        "close_rate_requested": trade.close_rate_requested,
        "realized_profit": trade.realized_profit,
        "close_profit": trade.close_profit,
        "close_profit_abs": trade.close_profit_abs,
        "stake_amount": trade.stake_amount,
        "max_stake_amount": trade.max_stake_amount,
        "amount": trade.amount,
        "amount_requested": trade.amount_requested,
        "open_date": trade.open_date_utc.isoformat(),
        "close_date": (
            trade.close_date_utc.isoformat()
            if trade.close_date_utc is not None
            else None
        ),
        "stop_loss": trade.stop_loss,
        "stop_loss_pct": trade.stop_loss_pct,
        "initial_stop_loss": trade.initial_stop_loss,
        "initial_stop_loss_pct": trade.initial_stop_loss_pct,
        "is_stop_loss_trailing": trade.is_stop_loss_trailing,
        "max_rate": trade.max_rate,
        "min_rate": trade.min_rate,
        "exit_reason": trade.exit_reason,
        "exit_order_status": trade.exit_order_status,
        "strategy": trade.strategy,
        "enter_tag": trade.enter_tag,
        "timeframe": trade.timeframe,
        "trading_mode": getattr(
            trade.trading_mode,
            "value",
            trade.trading_mode,
        ),
        "amount_precision": trade.amount_precision,
        "price_precision": trade.price_precision,
        "precision_mode": trade.precision_mode,
        "precision_mode_price": trade.precision_mode_price,
        "contract_size": trade.contract_size,
        "leverage": trade.leverage,
        "is_short": trade.is_short,
        "liquidation_price": trade.liquidation_price,
        "interest_rate": trade.interest_rate,
        "funding_fees": trade.funding_fees,
        "funding_fee_running": trade.funding_fee_running,
        "record_version": trade.record_version,
    }


def _quantize_trade_amount(trade: Trade, amount: float) -> float:
    return amount_to_contract_precision(
        amount,
        trade.amount_precision,
        trade.precision_mode,
        trade.contract_size,
    )


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        _fail(f"{label} is not a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise MaintenanceError(f"{label} is not a finite number") from error
    if not math.isfinite(number):
        _fail(f"{label} is not a finite number")
    return number


def _positive_number(value: object, label: str) -> float:
    number = _finite_number(value, label)
    if number <= 0.0:
        _fail(f"{label} must be positive")
    return number


def _evidence_confirmation_token(
    *,
    prefix: str,
    token_parts: Sequence[str],
    trade: Trade,
    state: TakeProfitStageState,
    instruction: dict[str, object],
) -> str:
    evidence = {
        "version": _CONFIRMATION_VERSION,
        "instruction": instruction,
        "trade": _confirmation_trade_summary(trade),
        "state": state.to_mapping(),
        "orders": [
            {"sequence": sequence, **_order_summary(order)}
            for sequence, order in enumerate(trade.orders)
        ],
    }
    try:
        canonical_evidence = json.dumps(
            evidence,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise MaintenanceError(
            "recovery evidence cannot be represented canonically"
        ) from error
    digest = hashlib.sha256(canonical_evidence).hexdigest()
    return ":".join((prefix, *token_parts, f"v{_CONFIRMATION_VERSION}", digest))


def _recovery_confirmation_token(
    trade: Trade,
    state: TakeProfitStageState,
    recovery: TakeProfitRecoveryInstruction,
) -> str:
    """Bind an operator confirmation to the complete inspected recovery evidence."""
    prefixes = {
        "confirm_terminal_canceled_fill": "CANCELED-FILL",
        "confirm_terminal_zero_fill": "ZERO-FILL",
        "revalidate_state": "REVALIDATE-STATE",
        "revalidate_terminal_order": "REVALIDATE-ORDER",
    }
    prefix = prefixes[recovery.action]
    token_parts = [str(trade.id)]
    if recovery.order_id is not None:
        token_parts.append(recovery.order_id)
    if recovery.terminal_status is not None:
        token_parts.append(recovery.terminal_status)
    return _evidence_confirmation_token(
        prefix=prefix,
        token_parts=token_parts,
        trade=trade,
        state=state,
        instruction={
            "action": recovery.action,
            "expected_trade_amount": recovery.expected_trade_amount,
            "order_id": recovery.order_id,
            "terminal_status": recovery.terminal_status,
            "trade_accounting_action": recovery.trade_accounting_action,
        },
    )


def _ambiguous_confirmation_token(
    trade: Trade,
    state: TakeProfitStageState,
    attempt_id: str,
) -> str:
    return _evidence_confirmation_token(
        prefix="NO-ORDER",
        token_parts=(str(trade.id), attempt_id),
        trade=trade,
        state=state,
        instruction={
            "action": "clear_ambiguous",
            "attempt_id": attempt_id,
            "trade_accounting_action": "none",
        },
    )


def _recovery_deadline_is_reached(
    recovery_deadline: str,
    current_time: datetime.datetime,
) -> bool:
    return current_time >= datetime.datetime.fromisoformat(recovery_deadline)


def _require_recovery_instruction(
    trade: Trade,
    state: TakeProfitStageState,
    *,
    action: TakeProfitRecoveryAction,
    order_id: str | None = None,
    terminal_status: str | None = None,
) -> TakeProfitRecoveryInstruction:
    recovery = TakeProfitStageManager.get_recovery_instruction(
        trade,
        state,
        amount_quantizer=partial(_quantize_trade_amount, trade),
    )
    if recovery is not None:
        recovery = _effective_recovery_instruction(trade, recovery)
    if (
        recovery is None
        or recovery.action != action
        or recovery.order_id != order_id
        or (
            terminal_status is not None
            and recovery.terminal_status != terminal_status.lower()
        )
    ):
        _fail("canonical evidence does not support the requested recovery")
    blocked_reason = _automatic_accounting_block_reason(trade, state, recovery)
    if blocked_reason is not None:
        try:
            _validate_trade_accounting_recovery(trade, recovery)
        except MaintenanceError as error:
            _fail(f"automatic recovery is unavailable: {blocked_reason}: {error}")
        _fail(f"automatic recovery is unavailable: {blocked_reason}")
    return recovery


def _validate_trade_lifecycle_for_command(
    args: argparse.Namespace,
    trade: Trade,
) -> None:
    if trade.is_open or args.command == "inspect":
        return
    _fail("closed trades are inspect-only; no take-profit state recovery is required")


def _require_current_confirmation(
    supplied_confirmation: str | None,
    expected_confirmation: str,
) -> None:
    if supplied_confirmation != expected_confirmation:
        _fail("confirmation is stale or invalid; rerun inspect")


def _require_spot_accounting_recalculation(trade: Trade) -> None:
    trading_mode = getattr(trade.trading_mode, "value", trade.trading_mode)
    if trading_mode != "spot":
        _fail(
            "offline trade-accounting recalculation is supported only in spot mode; "
            "use the native Freqtrade lifecycle for futures or margin"
        )


def _recovery_order(
    trade: Trade,
    recovery: TakeProfitRecoveryInstruction,
):
    if recovery.order_id is None:
        _fail("trade-accounting recovery has no canonical order")
    matching_orders = [
        order for order in trade.orders if order.order_id == recovery.order_id
    ]
    if len(matching_orders) != 1:
        _fail("trade-accounting recovery order is not unique")
    return matching_orders[0]


def _validate_recalculation_sources(trade: Trade) -> None:
    for order in trade.orders:
        if order.ft_is_open or not order.filled:
            continue
        _positive_number(
            order.safe_price,
            f"filled order {order.order_id!r} price",
        )


def _validate_funding_accounting(trade: Trade) -> None:
    filled_order_funding_fees = []
    for order in trade.orders:
        funding_fee = _finite_number(order.funding_fee or 0.0, "order funding fee")
        if order.ft_is_open or not order.filled:
            if not math.isclose(funding_fee, 0.0, rel_tol=1e-9, abs_tol=1e-12):
                _fail("open and zero-fill orders cannot contain funding fees")
            continue
        filled_order_funding_fees.append(funding_fee)
    running_funding_fee = _finite_number(
        trade.funding_fee_running or 0.0,
        "running trade funding fee",
    )
    observed_funding_fees = _finite_number(
        trade.funding_fees or 0.0,
        "trade funding fees",
    )
    trading_mode = getattr(trade.trading_mode, "value", trade.trading_mode)
    if trading_mode != "futures" and any(
        not math.isclose(value, 0.0, rel_tol=1e-9, abs_tol=1e-12)
        for value in (
            *filled_order_funding_fees,
            running_funding_fee,
            observed_funding_fees,
        )
    ):
        _fail("funding fees are supported only in futures mode")
    expected_funding_fees = math.fsum((*filled_order_funding_fees, running_funding_fee))
    if not math.isclose(
        observed_funding_fees,
        expected_funding_fees,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        _fail(
            "canonical trade funding fees do not match filled orders and "
            "running accrual"
        )


def _open_trade_accounting_fingerprint(
    trade: Trade,
) -> dict[str, float | bool | None]:
    return {
        "amount": trade.amount,
        "stake_amount": trade.stake_amount,
        "open_rate": trade.open_rate,
        "open_trade_value": trade.open_trade_value,
        "max_stake_amount": trade.max_stake_amount,
        "fee_open_cost": trade.fee_open_cost,
        "realized_profit": trade.realized_profit,
        "close_profit": trade.close_profit,
        "close_profit_abs": trade.close_profit_abs,
        "stop_loss": trade.stop_loss,
        "stop_loss_pct": trade.stop_loss_pct,
        "initial_stop_loss": trade.initial_stop_loss,
        "initial_stop_loss_pct": trade.initial_stop_loss_pct,
        "is_stop_loss_trailing": trade.is_stop_loss_trailing,
    }


def _reset_native_recalculation_outputs(trade: Trade) -> None:
    """Clear profit outputs that Freqtrade assigns only for non-zero exits."""
    trade.realized_profit = 0.0
    trade.close_profit = None
    trade.close_profit_abs = None


def _recalculated_trade_copy(
    trade: Trade,
    *,
    expected_trade_amount: float | None = None,
) -> Trade:
    loaded_order_count = len(trade.orders)
    try:
        canonical_trade = copy.deepcopy(trade)
        if len(canonical_trade.orders) != loaded_order_count:
            _fail("canonical trade clone lost order evidence")
        _reset_native_recalculation_outputs(canonical_trade)
        canonical_trade.recalc_trade_from_orders()
        if expected_trade_amount is not None:
            canonical_trade.amount = _quantize_trade_amount(
                canonical_trade,
                expected_trade_amount,
            )
    except MaintenanceError:
        raise
    except Exception as error:
        raise MaintenanceError(
            "canonical trade accounting cannot be rebuilt from orders"
        ) from error
    return canonical_trade


def _validate_canonical_open_trade_accounting(
    trade: Trade,
    *,
    expected_trade_amount: float,
) -> None:
    """Compare persisted accounting with the native rebuild plus ledger exposure."""
    observed = _open_trade_accounting_fingerprint(trade)
    canonical_trade = _recalculated_trade_copy(
        trade,
        expected_trade_amount=expected_trade_amount,
    )
    expected = _open_trade_accounting_fingerprint(canonical_trade)
    for field, expected_value in expected.items():
        observed_value = observed[field]
        if isinstance(observed_value, bool) or isinstance(expected_value, bool):
            if observed_value is not expected_value:
                _fail(f"canonical trade {field} does not match filled orders")
            continue
        if observed_value is None or expected_value is None:
            if observed_value != expected_value:
                _fail(f"canonical trade {field} does not match filled orders")
            continue
        observed_number = _finite_number(observed_value, f"trade {field}")
        expected_number = _finite_number(expected_value, f"rebuilt trade {field}")
        if not math.isclose(
            observed_number,
            expected_number,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            _fail(f"canonical trade {field} does not match filled orders")


def _validate_ambiguous_accounting_recovery(
    trade: Trade,
    state: TakeProfitStageState,
) -> None:
    if not trade.is_open:
        _fail("closed trades are inspect-only")
    _validate_recalculation_sources(trade)
    _validate_funding_accounting(trade)
    if state.partial_wallet_mutation is not None:
        _fail("partial wallet mutation requires a fresh wallet and CEX audit")
    expected_amount = TakeProfitStageManager.get_expected_trade_amount(
        state,
        amount_quantizer=partial(_quantize_trade_amount, trade),
    )
    _validate_canonical_open_trade_accounting(
        trade,
        expected_trade_amount=expected_amount,
    )


def _validate_trade_accounting_recovery(
    trade: Trade,
    recovery: TakeProfitRecoveryInstruction,
) -> None:
    _validate_recalculation_sources(trade)
    _validate_funding_accounting(trade)
    action = recovery.trade_accounting_action
    if action == "native_freqtrade_lifecycle":
        _fail("full-exposure recovery requires the native Freqtrade close lifecycle")
    if action == "none":
        if not trade.is_open or not recovery.expected_trade_is_open:
            _fail("canonical trade lifecycle does not match the recovery state")
        _validate_canonical_open_trade_accounting(
            trade,
            expected_trade_amount=recovery.expected_trade_amount,
        )
        return

    _require_spot_accounting_recalculation(trade)
    if not trade.is_open or not recovery.expected_trade_is_open:
        _fail("automatic trade-accounting recovery requires an open trade")
    order = _recovery_order(trade, recovery)
    if order.ft_is_open or order.ft_order_side != trade.exit_side:
        _fail("trade-accounting recovery order is not a terminal exit")

    if action != "recalculate_from_orders":
        _fail("unknown trade-accounting recovery action")
    canonical_trade = _recalculated_trade_copy(
        trade,
        expected_trade_amount=recovery.expected_trade_amount,
    )
    if canonical_trade.is_open != recovery.expected_trade_is_open:
        _fail("rebuilt trade accounting has an unexpected lifecycle state")
    _validate_funding_accounting(canonical_trade)
    _validate_canonical_open_trade_accounting(
        canonical_trade,
        expected_trade_amount=recovery.expected_trade_amount,
    )


def _apply_trade_accounting_recovery(
    trade: Trade,
    recovery: TakeProfitRecoveryInstruction,
) -> None:
    _validate_trade_accounting_recovery(trade, recovery)
    action = recovery.trade_accounting_action
    if action == "none":
        return
    try:
        _reset_native_recalculation_outputs(trade)
        trade.recalc_trade_from_orders()
        trade.amount = _quantize_trade_amount(
            trade,
            recovery.expected_trade_amount,
        )
        if trade.is_open != recovery.expected_trade_is_open:
            _fail("native trade accounting produced an unexpected lifecycle state")
        _validate_funding_accounting(trade)
        _validate_canonical_open_trade_accounting(
            trade,
            expected_trade_amount=recovery.expected_trade_amount,
        )
        _commit_trade_accounting()
    except MaintenanceError:
        Trade.session.rollback()
        raise
    except Exception as error:
        Trade.session.rollback()
        raise MaintenanceError("failed to update canonical trade accounting") from error


def _effective_recovery_instruction(
    trade: Trade,
    recovery: TakeProfitRecoveryInstruction,
) -> TakeProfitRecoveryInstruction:
    """Select a safe native rebuild when the manager cannot see full accounting."""
    if recovery.trade_accounting_action == "native_freqtrade_lifecycle":
        return recovery
    trading_mode = getattr(trade.trading_mode, "value", trade.trading_mode)
    if (
        recovery.trade_accounting_action == "recalculate_from_orders"
        and trading_mode == "spot"
    ):
        state_only_recovery = replace(
            recovery,
            trade_accounting_action="none",
        )
        try:
            _validate_trade_accounting_recovery(trade, state_only_recovery)
        except MaintenanceError:
            pass
        else:
            return state_only_recovery
    try:
        _validate_trade_accounting_recovery(trade, recovery)
    except MaintenanceError:
        if recovery.trade_accounting_action != "none" or recovery.order_id is None:
            return recovery
        recalculated_recovery = replace(
            recovery,
            trade_accounting_action="recalculate_from_orders",
        )
        try:
            _validate_trade_accounting_recovery(trade, recalculated_recovery)
        except MaintenanceError:
            return recovery
        return recalculated_recovery
    return recovery


def _automatic_accounting_block_reason(
    trade: Trade,
    state: TakeProfitStageState,
    recovery: TakeProfitRecoveryInstruction | None,
) -> str | None:
    if not trade.is_open:
        return "closed_trade_is_inspect_only"
    if state.partial_wallet_mutation is not None:
        return "fresh_wallet_and_cex_audit_required"
    recovery_order_id = (
        state.blocked_order_id
        if state.blocked_order_id is not None
        else recovery.order_id
        if recovery is not None
        else None
    )
    blocked_order = next(
        (
            order
            for order in trade.orders
            if recovery_order_id is not None and order.order_id == recovery_order_id
        ),
        None,
    )
    try:
        blocked_fill = (
            _finite_number(blocked_order.filled or 0.0, "blocked order fill")
            if blocked_order is not None and not blocked_order.ft_is_open
            else 0.0
        )
    except MaintenanceError:
        blocked_fill = 0.0
    has_positive_blocked_fill = blocked_fill > 0.0
    trading_mode = getattr(trade.trading_mode, "value", trade.trading_mode)
    if (
        recovery is not None
        and recovery.trade_accounting_action == "native_freqtrade_lifecycle"
    ):
        return "positive_fill_requires_native_freqtrade_lifecycle"
    if has_positive_blocked_fill and trading_mode != "spot":
        return "positive_fill_requires_native_freqtrade_lifecycle"
    if blocked_order is not None:
        try:
            blocked_order_funding = _finite_number(
                blocked_order.funding_fee or 0.0,
                "blocked order funding fee",
            )
        except MaintenanceError:
            return "automatic_trade_accounting_precondition_failed"
        if (blocked_order.ft_is_open or not blocked_fill) and not math.isclose(
            blocked_order_funding,
            0.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            return "automatic_trade_accounting_precondition_failed"
    if recovery is None:
        return (
            "automatic_trade_accounting_precondition_failed"
            if has_positive_blocked_fill
            else None
        )
    try:
        _validate_trade_accounting_recovery(trade, recovery)
    except MaintenanceError:
        if trading_mode != "spot":
            return "non_spot_trade_accounting_not_replay_safe"
        return "automatic_trade_accounting_precondition_failed"
    return None


def _validate_state_roundtrip(state: TakeProfitStageState) -> dict[str, object]:
    payload = state.to_mapping()
    try:
        validated_state = TakeProfitStageState.from_mapping(payload)
    except ValueError as error:
        raise MaintenanceError(
            f"take-profit state failed round-trip validation: {error}"
        ) from error
    if validated_state != state or validated_state.to_mapping() != payload:
        _fail("take-profit state failed full round-trip validation")
    return payload


def _persist_state(
    trade: Trade,
    state: TakeProfitStageState,
) -> TakeProfitStageState:
    """Prevalidate the complete payload before the only custom-state write."""
    payload = _validate_state_roundtrip(state)
    try:
        trade.set_custom_data(TakeProfitStageManager.STATE_KEY, payload)
        persisted_payload = trade.get_custom_data(TakeProfitStageManager.STATE_KEY)
        if persisted_payload != payload:
            _fail("take-profit state persistence verification failed")
        persisted_state = _load_state(trade)
        if persisted_state != state:
            _fail("take-profit state postcondition failed after commit")
        return persisted_state
    except MaintenanceError:
        Trade.session.rollback()
        raise
    except Exception as error:
        Trade.session.rollback()
        raise MaintenanceError(
            "take-profit state persistence failed; inspect the trade before retrying"
        ) from error


def _summary(
    trade: Trade,
    state: TakeProfitStageState,
    *,
    command: str,
    would_apply: bool = False,
) -> dict[str, object]:
    attempt = state.active_attempt
    amount_quantizer = partial(_quantize_trade_amount, trade)
    expected_trade_amount = TakeProfitStageManager.get_expected_trade_amount(
        state,
        amount_quantizer=amount_quantizer,
    )
    trade_amount_delta = TakeProfitStageManager.get_trade_amount_delta(
        trade,
        state,
        amount_quantizer=amount_quantizer,
    )
    recovery = TakeProfitStageManager.get_recovery_instruction(
        trade,
        state,
        amount_quantizer=amount_quantizer,
    )
    if recovery is not None:
        recovery = _effective_recovery_instruction(trade, recovery)
    clearable_attempt = (
        attempt
        if attempt is not None
        and attempt.status == "ambiguous"
        and recovery is None
        and state.blocked_reason is None
        and state.partial_wallet_mutation is None
        else None
    )
    recovery_blocked_reason = _automatic_accounting_block_reason(
        trade,
        state,
        recovery,
    )
    clear_blocked_reason = None
    if clearable_attempt is not None:
        try:
            _validate_ambiguous_accounting_recovery(trade, state)
        except MaintenanceError:
            clear_blocked_reason = (
                "closed_trade_is_inspect_only"
                if not trade.is_open
                else "automatic_trade_accounting_precondition_failed"
            )
    clear_confirmation_is_available = (
        clearable_attempt is not None
        and clear_blocked_reason is None
        and _recovery_deadline_is_reached(
            clearable_attempt.recovery_deadline,
            datetime.datetime.now(datetime.timezone.utc),
        )
    )
    recovery_is_available = recovery is not None and recovery_blocked_reason is None
    return {
        "command": command,
        "would_apply": would_apply,
        "trade": {
            "id": trade.id,
            "pair": trade.pair,
            "direction": trade.trade_direction,
            "is_open": trade.is_open,
            "strategy": trade.strategy,
            "amount": trade.amount,
            "stake_amount": trade.stake_amount,
        },
        "state": {
            "version": state.version,
            "blocked_reason": state.blocked_reason,
            "blocked_order_id": state.blocked_order_id,
            "active_attempt": attempt.to_mapping() if attempt is not None else None,
            "clear_eligible_at": (
                clearable_attempt.recovery_deadline
                if clearable_attempt is not None
                else None
            ),
            "required_clear_confirmation": (
                _ambiguous_confirmation_token(
                    trade,
                    state,
                    clearable_attempt.attempt_id,
                )
                if clear_confirmation_is_available and clearable_attempt is not None
                else None
            ),
            "recovery_action": recovery.action if recovery is not None else None,
            "recovery_trade_accounting_action": (
                recovery.trade_accounting_action if recovery is not None else None
            ),
            "recovery_expected_trade_amount": (
                recovery.expected_trade_amount if recovery is not None else None
            ),
            "recovery_expected_trade_is_open": (
                recovery.expected_trade_is_open if recovery is not None else None
            ),
            "required_state_revalidation_confirmation": (
                _recovery_confirmation_token(trade, state, recovery)
                if recovery_is_available
                and recovery is not None
                and recovery.action == "revalidate_state"
                else None
            ),
            "required_terminal_order_revalidation_confirmation": (
                _recovery_confirmation_token(trade, state, recovery)
                if recovery_is_available
                and recovery is not None
                and recovery.action == "revalidate_terminal_order"
                else None
            ),
            "required_terminal_zero_fill_confirmation": (
                _recovery_confirmation_token(trade, state, recovery)
                if recovery_is_available
                and recovery is not None
                and recovery.action == "confirm_terminal_zero_fill"
                else None
            ),
            "required_terminal_canceled_fill_confirmation": (
                _recovery_confirmation_token(trade, state, recovery)
                if recovery_is_available
                and recovery is not None
                and recovery.action == "confirm_terminal_canceled_fill"
                else None
            ),
            "recovery_blocked_reason": (
                clear_blocked_reason
                if clear_blocked_reason is not None
                else recovery_blocked_reason
            ),
            "ledger": {
                "initial_amount": state.initial_amount,
                "expected_trade_amount": expected_trade_amount,
                "trade_amount_delta": trade_amount_delta,
                "total_take_profit_credited_amount": sum(
                    amount for _, amount in state.credited_order_amounts
                ),
                "total_non_take_profit_exit_amount": sum(
                    amount for _, amount in state.non_take_profit_exit_order_amounts
                ),
                "active_exposure_adjustment_amount": sum(
                    adjustment.amount
                    for adjustment in state.exposure_adjustments
                    if adjustment.is_active
                ),
                "compacted_exposure_adjustment_count": (
                    state.compacted_exposure_adjustment_count
                ),
                "compacted_exposure_adjustment_amount": (
                    state.compacted_exposure_adjustment_amount
                ),
                "stage_targets": [
                    {"stage": stage, "amount": amount}
                    for stage, amount in state.stage_targets
                ],
                "deferred_stages": list(state.deferred_stages),
                "order_attributions": [
                    attribution.to_mapping() for attribution in state.order_attributions
                ],
                "credited_order_amounts": [
                    {"order_id": order_id, "amount": amount}
                    for order_id, amount in state.credited_order_amounts
                ],
                "non_take_profit_exit_order_amounts": [
                    {"order_id": order_id, "amount": amount}
                    for order_id, amount in state.non_take_profit_exit_order_amounts
                ],
                "exposure_adjustments": [
                    adjustment.to_mapping() for adjustment in state.exposure_adjustments
                ],
                "entry_order_amounts": [
                    {"order_id": order_id, "amount": amount}
                    for order_id, amount in state.entry_order_amounts
                ],
                "confirmed_zero_fill_orders": [
                    proof.to_mapping() for proof in state.confirmed_zero_fill_orders
                ],
                "confirmed_canceled_fill_orders": [
                    proof.to_mapping() for proof in state.confirmed_canceled_fill_orders
                ],
                "partial_wallet_mutation": (
                    state.partial_wallet_mutation.to_mapping()
                    if state.partial_wallet_mutation is not None
                    else None
                ),
                "retry_gate": (
                    state.retry_gate.to_mapping()
                    if state.retry_gate is not None
                    else None
                ),
            },
            "operator_resolutions": [
                resolution.to_mapping() for resolution in state.operator_resolutions
            ],
        },
        "orders": [_order_summary(order) for order in trade.orders],
    }


def _print_result(result: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"command: {result['command']}")
    print(f"would_apply: {result['would_apply']}")
    print(json.dumps(result["trade"], sort_keys=True))
    print(json.dumps(result["state"], sort_keys=True))
    for order in result["orders"]:
        print(json.dumps(order, sort_keys=True))


def _validate_ambiguous_attempt(
    trade: Trade,
    state: TakeProfitStageState,
    attempt_id: str,
) -> None:
    if not attempt_id:
        _fail("attempt-id must be non-empty")
    if state.blocked_reason is not None:
        _fail(f"take-profit state is blocked: {state.blocked_reason}")
    if state.partial_wallet_mutation is not None:
        _fail("partial wallet mutation requires a fresh wallet and CEX audit")
    attempt = state.active_attempt
    if (
        attempt is None
        or attempt.attempt_id != attempt_id
        or attempt.status != "ambiguous"
    ):
        _fail("no matching ambiguous attempt")
    parsed_tag = TakeProfitStageManager.parse_order_tag(attempt.tag)
    if parsed_tag != (trade.trade_direction, attempt.stage):
        _fail("active attempt tag does not match the trade")

    reconciled_state = TakeProfitStageManager.reconcile(
        trade,
        state,
        amount_quantizer=lambda amount: _quantize_trade_amount(trade, amount),
    )
    if reconciled_state != state:
        _fail(
            "database orders change the persisted state; restart normal reconciliation "
            "or investigate the reported block before clearing anything"
        )

    if not _recovery_deadline_is_reached(
        attempt.recovery_deadline,
        datetime.datetime.now(datetime.timezone.utc),
    ):
        _fail(f"ambiguous attempt cannot be cleared before {attempt.recovery_deadline}")


def _clear_ambiguous(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    _validate_ambiguous_attempt(trade, state, args.attempt_id)
    _validate_ambiguous_accounting_recovery(trade, state)
    expected_confirmation = _ambiguous_confirmation_token(
        trade,
        state,
        args.attempt_id,
    )
    if not args.apply:
        return _summary(
            trade,
            state,
            command="clear-ambiguous",
            would_apply=True,
        )
    _require_current_confirmation(args.confirmation, expected_confirmation)

    try:
        candidate_state = state
        candidate_state = TakeProfitStageManager.reconcile(
            trade,
            candidate_state,
            amount_quantizer=partial(_quantize_trade_amount, trade),
        )
        if candidate_state.blocked_reason is not None:
            _fail(
                "canonical trade and state do not validate before ambiguity "
                "resolution: "
                f"{candidate_state.blocked_reason}"
            )
        resolved_state = TakeProfitStageManager.resolve_ambiguous_attempt(
            candidate_state,
            args.attempt_id,
            datetime.datetime.now(datetime.timezone.utc),
        )
        resolved_state = TakeProfitStageManager.reconcile(
            trade,
            resolved_state,
            amount_quantizer=partial(_quantize_trade_amount, trade),
        )
        if resolved_state.blocked_reason is not None:
            _fail(
                "canonical trade and state do not validate after ambiguity "
                "resolution: "
                f"{resolved_state.blocked_reason}"
            )
        _validate_state_roundtrip(resolved_state)
    except MaintenanceError:
        Trade.session.rollback()
        raise
    except ValueError as error:
        Trade.session.rollback()
        raise MaintenanceError(
            f"ambiguous attempt cannot be cleared: {error}"
        ) from error
    except Exception as error:
        Trade.session.rollback()
        raise MaintenanceError("failed to prepare ambiguity resolution") from error
    persisted_state = _persist_state(trade, resolved_state)
    return _summary(trade, persisted_state, command="clear-ambiguous")


def _revalidate_terminal_order(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    if not args.order_id:
        _fail("order-id must be non-empty")
    recovery = _require_recovery_instruction(
        trade,
        state,
        action="revalidate_terminal_order",
        order_id=args.order_id,
    )
    expected_confirmation = _recovery_confirmation_token(trade, state, recovery)
    _validate_trade_accounting_recovery(trade, recovery)
    if args.apply:
        _require_current_confirmation(args.confirmation, expected_confirmation)
    resolved_at = datetime.datetime.now(datetime.timezone.utc)
    amount_quantizer = partial(_quantize_trade_amount, trade)
    try:
        if not args.apply:
            TakeProfitStageManager.validate_terminal_order_revalidation(
                trade,
                state,
                args.order_id,
                resolved_at,
                amount_quantizer=amount_quantizer,
            )
            return _summary(
                trade,
                state,
                command="revalidate-terminal-order",
                would_apply=True,
            )
        _apply_trade_accounting_recovery(trade, recovery)
        resolved_state = TakeProfitStageManager.revalidate_terminal_order(
            trade,
            state,
            args.order_id,
            resolved_at,
            amount_quantizer=amount_quantizer,
        )
        _validate_state_roundtrip(resolved_state)
    except ValueError as error:
        Trade.session.rollback()
        raise MaintenanceError(
            f"terminal order cannot be revalidated: {error}"
        ) from error
    except Exception as error:
        Trade.session.rollback()
        raise MaintenanceError(
            "failed to prepare terminal-order revalidation"
        ) from error

    persisted_state = _persist_state(trade, resolved_state)
    return _summary(trade, persisted_state, command="revalidate-terminal-order")


def _revalidate_state(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    recovery = _require_recovery_instruction(
        trade,
        state,
        action="revalidate_state",
    )
    _validate_trade_accounting_recovery(trade, recovery)
    expected_confirmation = _recovery_confirmation_token(trade, state, recovery)
    if args.apply:
        _require_current_confirmation(args.confirmation, expected_confirmation)
    try:
        resolved_state = TakeProfitStageManager.revalidate_state(
            trade,
            state,
            datetime.datetime.now(datetime.timezone.utc),
            amount_quantizer=partial(_quantize_trade_amount, trade),
        )
        _validate_state_roundtrip(resolved_state)
    except ValueError as error:
        raise MaintenanceError(f"state cannot be revalidated: {error}") from error

    if not args.apply:
        return _summary(
            trade,
            state,
            command="revalidate-state",
            would_apply=True,
        )
    persisted_state = _persist_state(trade, resolved_state)
    return _summary(trade, persisted_state, command="revalidate-state")


def _confirm_terminal_zero_fill(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    return _confirm_terminal_fill(
        args,
        trade,
        state,
        command="confirm-terminal-zero-fill",
        recovery_action="confirm_terminal_zero_fill",
        failure_label="terminal zero fill",
        confirm=TakeProfitStageManager.confirm_terminal_zero_fill,
        validate=TakeProfitStageManager.validate_terminal_zero_fill_confirmation,
    )


def _confirm_terminal_canceled_fill(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    return _confirm_terminal_fill(
        args,
        trade,
        state,
        command="confirm-terminal-canceled-fill",
        recovery_action="confirm_terminal_canceled_fill",
        failure_label="terminal canceled fill",
        confirm=TakeProfitStageManager.confirm_terminal_canceled_fill,
        validate=(TakeProfitStageManager.validate_terminal_canceled_fill_confirmation),
    )


def _confirm_terminal_fill(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
    *,
    command: str,
    recovery_action: TakeProfitRecoveryAction,
    failure_label: str,
    confirm: Callable[..., TakeProfitStageState],
    validate: Callable[..., None],
) -> dict[str, object]:
    if not args.order_id:
        _fail("order-id must be non-empty")
    recovery = _require_recovery_instruction(
        trade,
        state,
        action=recovery_action,
        order_id=args.order_id,
        terminal_status=args.terminal_status,
    )
    expected_confirmation = _recovery_confirmation_token(trade, state, recovery)
    _validate_trade_accounting_recovery(trade, recovery)
    if args.apply:
        _require_current_confirmation(args.confirmation, expected_confirmation)
    resolved_at = datetime.datetime.now(datetime.timezone.utc)
    amount_quantizer = partial(_quantize_trade_amount, trade)
    try:
        if not args.apply:
            validate(
                trade,
                state,
                args.order_id,
                args.terminal_status,
                resolved_at,
                amount_quantizer=amount_quantizer,
            )
            return _summary(
                trade,
                state,
                command=command,
                would_apply=True,
            )
        _apply_trade_accounting_recovery(trade, recovery)
        resolved_state = confirm(
            trade,
            state,
            args.order_id,
            args.terminal_status,
            resolved_at,
            amount_quantizer=amount_quantizer,
        )
        _validate_state_roundtrip(resolved_state)
    except ValueError as error:
        Trade.session.rollback()
        raise MaintenanceError(
            f"{failure_label} cannot be confirmed: {error}"
        ) from error
    except Exception as error:
        Trade.session.rollback()
        raise MaintenanceError(f"failed to prepare {failure_label} recovery") from error

    persisted_state = _persist_state(trade, resolved_state)
    return _summary(trade, persisted_state, command=command)


def _commit_trade_accounting() -> None:
    try:
        Trade.commit()
    except Exception as error:
        Trade.session.rollback()
        raise MaintenanceError("failed to commit canonical trade accounting") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        trade = _load_trade(args)
        state = _load_state(trade)
        _validate_trade_lifecycle_for_command(args, trade)
        if args.command == "inspect":
            result = _summary(trade, state, command="inspect")
        elif args.command == "clear-ambiguous":
            result = _clear_ambiguous(args, trade, state)
        elif args.command == "revalidate-terminal-order":
            result = _revalidate_terminal_order(args, trade, state)
        elif args.command == "revalidate-state":
            result = _revalidate_state(args, trade, state)
        elif args.command == "confirm-terminal-zero-fill":
            result = _confirm_terminal_zero_fill(args, trade, state)
        elif args.command == "confirm-terminal-canceled-fill":
            result = _confirm_terminal_canceled_fill(args, trade, state)
        else:
            raise AssertionError(f"Unhandled command {args.command!r}")
        _print_result(result, as_json=args.as_json)
        return 0
    except MaintenanceError as error:
        if args.as_json:
            print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        else:
            print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
