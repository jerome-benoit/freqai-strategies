"""Inspect or recover a fail-closed QuickAdapter take-profit state.

Run this tool only while the bot is stopped and only after checking the CEX order
history. Repair canonical Freqtrade order data before revalidating a terminal fill.
"""

import argparse
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
        "revalidate-terminal",
        help="Reconcile a repaired canonical terminal order",
    )
    revalidate_parser.add_argument("--order-id", required=True)
    revalidate_parser.add_argument("--apply", action="store_true")
    revalidate_parser.add_argument("--confirmation")
    zero_fill_parser = commands.add_parser(
        "confirm-terminal-zero",
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
        "confirm-terminal-canceled",
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
    restore_partial_wallet_parser = commands.add_parser(
        "restore-partial-wallet-mutation",
        help="Restore native trade accounting for a bound terminal order",
    )
    restore_partial_wallet_parser.add_argument("--order-id", required=True)
    restore_partial_wallet_parser.add_argument("--apply", action="store_true")
    restore_partial_wallet_parser.add_argument("--confirmation")
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
    if not trade.is_open:
        _fail("trade is not open")
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


def _quantize_trade_amount(trade: Trade, amount: float) -> float:
    return amount_to_contract_precision(
        amount,
        trade.amount_precision,
        trade.precision_mode,
        trade.contract_size,
    )


def _recovery_confirmation_token(
    trade: Trade,
    state: TakeProfitStageState,
    recovery: TakeProfitRecoveryInstruction,
) -> str:
    """Bind an operator confirmation to the complete inspected recovery evidence."""
    prefixes = {
        "confirm_terminal_canceled": "CANCELED-FILL",
        "confirm_terminal_zero": "NO-FILL",
        "restore_partial_wallet_mutation": "RESTORE-PARTIAL-WALLET",
        "revalidate_terminal": "RECONCILE",
    }
    prefix = prefixes[recovery.action]
    # Recalculated accounting outputs are intentionally absent so a token remains
    # valid if the process stops after the first commit. The ordered source orders
    # and stable calculation inputs below still bind the operation exactly.
    evidence = {
        "version": 1,
        "instruction": {
            "action": recovery.action,
            "attempt_id": recovery.attempt_id,
            "order_id": recovery.order_id,
            "terminal_status": recovery.terminal_status,
        },
        "trade": {
            "id": trade.id,
            "pair": trade.pair,
            "exchange": trade.exchange,
            "strategy": trade.strategy,
            "is_open": trade.is_open,
            "is_short": trade.is_short,
            "direction": trade.trade_direction,
            "entry_side": trade.entry_side,
            "exit_side": trade.exit_side,
            "trading_mode": getattr(
                trade.trading_mode,
                "value",
                trade.trading_mode,
            ),
            "leverage": trade.leverage,
            "fee_open": trade.fee_open,
            "fee_close": trade.fee_close,
            "amount_precision": trade.amount_precision,
            "precision_mode": trade.precision_mode,
            "contract_size": trade.contract_size,
            "price_precision": trade.price_precision,
            "precision_mode_price": trade.precision_mode_price,
            "stop_loss_pct": trade.stop_loss_pct,
        },
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
    token_parts = [prefix, str(trade.id)]
    if recovery.attempt_id is not None:
        token_parts.append(recovery.attempt_id)
    token_parts.append(recovery.order_id)
    if recovery.terminal_status is not None:
        token_parts.append(recovery.terminal_status)
    token_parts.extend(("v1", digest))
    return ":".join(token_parts)


def _require_recovery_instruction(
    trade: Trade,
    state: TakeProfitStageState,
    *,
    action: TakeProfitRecoveryAction,
    order_id: str,
    terminal_status: str | None = None,
) -> TakeProfitRecoveryInstruction:
    recovery = TakeProfitStageManager.get_recovery_instruction(
        trade,
        state,
        amount_quantizer=partial(_quantize_trade_amount, trade),
    )
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
    return recovery


def _require_current_confirmation(
    supplied_confirmation: str | None,
    expected_confirmation: str,
) -> None:
    if supplied_confirmation != expected_confirmation:
        _fail("confirmation is stale or invalid; rerun inspect")


def _require_replayable_recalculation(trade: Trade) -> None:
    trading_mode = getattr(trade.trading_mode, "value", trade.trading_mode)
    if trading_mode == "margin":
        _fail(
            "automatic order recalculation is not replay-safe in margin mode; "
            "automatic state recovery is unavailable; reconcile and close the "
            "position manually"
        )


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
    clearable_attempt = (
        attempt
        if attempt is not None
        and attempt.status == "ambiguous"
        and recovery is None
        and (
            state.blocked_reason is None
            or (
                state.blocked_reason == "partial_wallet_mutation"
                and state.partial_wallet_mutation is not None
                and state.partial_wallet_mutation.order_id is None
                and state.partial_wallet_mutation.attempt_id == attempt.attempt_id
            )
        )
        and (
            state.partial_wallet_mutation is None
            or state.partial_wallet_mutation.order_id is None
        )
        else None
    )
    trading_mode = getattr(trade.trading_mode, "value", trade.trading_mode)
    requires_margin_recalculation = (
        trading_mode == "margin" and state.partial_wallet_mutation is not None
    )
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
                f"NO-ORDER:{trade.id}:{clearable_attempt.attempt_id}"
                if clearable_attempt is not None and not requires_margin_recalculation
                else None
            ),
            "required_revalidation_confirmation": (
                _recovery_confirmation_token(trade, state, recovery)
                if recovery is not None and recovery.action == "revalidate_terminal"
                else None
            ),
            "required_zero_fill_confirmation": (
                _recovery_confirmation_token(trade, state, recovery)
                if recovery is not None
                and recovery.action == "confirm_terminal_zero"
                and not requires_margin_recalculation
                else None
            ),
            "required_canceled_fill_confirmation": (
                _recovery_confirmation_token(trade, state, recovery)
                if recovery is not None
                and recovery.action == "confirm_terminal_canceled"
                and not requires_margin_recalculation
                else None
            ),
            "required_partial_wallet_restoration_confirmation": (
                _recovery_confirmation_token(trade, state, recovery)
                if recovery is not None
                and recovery.action == "restore_partial_wallet_mutation"
                and not requires_margin_recalculation
                else None
            ),
            "recovery_blocked_reason": (
                "margin_recalculation_not_replay_safe"
                if requires_margin_recalculation
                else None
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
    partial_wallet_mutation = state.partial_wallet_mutation
    if (
        partial_wallet_mutation is not None
        and partial_wallet_mutation.order_id is not None
    ):
        _fail(
            "ambiguous attempt is bound to a canonical order; use the terminal "
            "recovery action reported by inspect"
        )
    has_recoverable_partial_wallet_mutation = (
        state.blocked_reason == "partial_wallet_mutation"
        and partial_wallet_mutation is not None
        and partial_wallet_mutation.attempt_id == attempt_id
    )
    if state.blocked_reason is not None and not has_recoverable_partial_wallet_mutation:
        _fail(f"take-profit state is blocked: {state.blocked_reason}")
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

    recovery_deadline = datetime.datetime.fromisoformat(attempt.recovery_deadline)
    if datetime.datetime.now(datetime.timezone.utc) < recovery_deadline:
        _fail(f"ambiguous attempt cannot be cleared before {attempt.recovery_deadline}")


def _clear_ambiguous(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    _validate_ambiguous_attempt(trade, state, args.attempt_id)
    partial_wallet_mutation = state.partial_wallet_mutation
    if partial_wallet_mutation is not None:
        _require_replayable_recalculation(trade)
    expected_confirmation = f"NO-ORDER:{trade.id}:{args.attempt_id}"
    if not args.apply:
        return _summary(
            trade,
            state,
            command="clear-ambiguous",
            would_apply=True,
        )
    if args.confirmation != expected_confirmation:
        _fail(f"confirmation must be exactly {expected_confirmation!r}")

    try:
        candidate_state = state
        if partial_wallet_mutation is not None:
            trade.recalc_trade_from_orders()
            _validate_ambiguous_attempt(trade, state, args.attempt_id)
            observed_amount = _quantize_trade_amount(trade, trade.amount)
            expected_amount = _quantize_trade_amount(
                trade,
                partial_wallet_mutation.expected_trade_amount,
            )
            if not math.isclose(
                observed_amount,
                expected_amount,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                _fail(
                    "recalculated trade amount does not restore the partial "
                    "wallet mutation baseline"
                )
            candidate_state = replace(
                state,
                partial_wallet_mutation=None,
                blocked_reason=None,
                blocked_order_id=None,
            )
        candidate_state = TakeProfitStageManager.reconcile(
            trade,
            candidate_state,
            amount_quantizer=partial(_quantize_trade_amount, trade),
        )
        if candidate_state.blocked_reason is not None:
            _fail(
                "recalculated trade does not validate before ambiguity resolution: "
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
                "recalculated trade does not validate after ambiguity resolution: "
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
    if partial_wallet_mutation is not None:
        _commit_recalculated_trade()
    persisted_state = _persist_state(trade, resolved_state)
    return _summary(trade, persisted_state, command="clear-ambiguous")


def _revalidate_terminal(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    if not args.order_id:
        _fail("order-id must be non-empty")
    recovery = _require_recovery_instruction(
        trade,
        state,
        action="revalidate_terminal",
        order_id=args.order_id,
    )
    expected_confirmation = _recovery_confirmation_token(trade, state, recovery)
    if args.apply:
        _require_current_confirmation(args.confirmation, expected_confirmation)
    try:
        resolved_state = TakeProfitStageManager.revalidate_unknown_terminal_fill(
            trade,
            state,
            args.order_id,
            datetime.datetime.now(datetime.timezone.utc),
            amount_quantizer=lambda amount: _quantize_trade_amount(trade, amount),
        )
    except ValueError as error:
        raise MaintenanceError(
            f"terminal fill cannot be revalidated: {error}"
        ) from error

    if not args.apply:
        return _summary(
            trade,
            state,
            command="revalidate-terminal",
            would_apply=True,
        )
    persisted_state = _persist_state(trade, resolved_state)
    return _summary(trade, persisted_state, command="revalidate-terminal")


def _confirm_terminal_zero(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    return _confirm_terminal(
        args,
        trade,
        state,
        command="confirm-terminal-zero",
        recovery_action="confirm_terminal_zero",
        failure_label="terminal zero fill",
        confirm=TakeProfitStageManager.confirm_terminal_zero_fill,
        validate=TakeProfitStageManager.validate_terminal_zero_fill_confirmation,
    )


def _confirm_terminal_canceled(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    return _confirm_terminal(
        args,
        trade,
        state,
        command="confirm-terminal-canceled",
        recovery_action="confirm_terminal_canceled",
        failure_label="terminal canceled fill",
        confirm=TakeProfitStageManager.confirm_terminal_canceled_fill,
        validate=(TakeProfitStageManager.validate_terminal_canceled_fill_confirmation),
    )


def _confirm_terminal(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
    *,
    command: str,
    recovery_action: str,
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
    bound_mutation = state.partial_wallet_mutation
    requires_trade_recalculation = (
        bound_mutation is not None and bound_mutation.order_id == args.order_id
    )
    if requires_trade_recalculation:
        _require_replayable_recalculation(trade)
    if args.apply:
        _require_current_confirmation(args.confirmation, expected_confirmation)
    resolved_at = datetime.datetime.now(datetime.timezone.utc)
    amount_quantizer = partial(_quantize_trade_amount, trade)
    try:
        if requires_trade_recalculation and not args.apply:
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
        if requires_trade_recalculation:
            trade.recalc_trade_from_orders()
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

    if not args.apply:
        return _summary(
            trade,
            state,
            command=command,
            would_apply=True,
        )
    if requires_trade_recalculation:
        _commit_recalculated_trade()

    persisted_state = _persist_state(trade, resolved_state)
    return _summary(trade, persisted_state, command=command)


def _commit_recalculated_trade() -> None:
    try:
        Trade.commit()
    except Exception as error:
        Trade.session.rollback()
        raise MaintenanceError(
            "failed to commit recalculated canonical trade accounting"
        ) from error


def _restore_partial_wallet_mutation(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    if not args.order_id:
        _fail("order-id must be non-empty")
    mutation = state.partial_wallet_mutation
    if mutation is None or mutation.order_id != args.order_id:
        _fail("no matching bound partial-wallet mutation")
    _require_replayable_recalculation(trade)
    recovery = _require_recovery_instruction(
        trade,
        state,
        action="restore_partial_wallet_mutation",
        order_id=args.order_id,
    )
    expected_confirmation = _recovery_confirmation_token(trade, state, recovery)
    if args.apply:
        _require_current_confirmation(args.confirmation, expected_confirmation)
    amount_quantizer = partial(_quantize_trade_amount, trade)
    if not args.apply:
        try:
            TakeProfitStageManager.validate_partial_wallet_mutation_restoration(
                trade,
                state,
                args.order_id,
                amount_quantizer=amount_quantizer,
            )
        except ValueError as error:
            raise MaintenanceError(
                f"partial-wallet mutation cannot be restored: {error}"
            ) from error
        return _summary(
            trade,
            state,
            command="restore-partial-wallet-mutation",
            would_apply=True,
        )

    try:
        trade.recalc_trade_from_orders()
        resolved_state = TakeProfitStageManager.restore_partial_wallet_mutation(
            trade,
            state,
            args.order_id,
            datetime.datetime.now(datetime.timezone.utc),
            amount_quantizer=amount_quantizer,
        )
        _validate_state_roundtrip(resolved_state)
    except ValueError as error:
        Trade.session.rollback()
        raise MaintenanceError(
            f"partial-wallet mutation cannot be restored: {error}"
        ) from error
    except Exception as error:
        Trade.session.rollback()
        raise MaintenanceError(
            "failed to prepare partial-wallet mutation restoration"
        ) from error
    _commit_recalculated_trade()
    persisted_state = _persist_state(trade, resolved_state)
    return _summary(
        trade,
        persisted_state,
        command="restore-partial-wallet-mutation",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        trade = _load_trade(args)
        state = _load_state(trade)
        if args.command == "inspect":
            result = _summary(trade, state, command="inspect")
        elif args.command == "clear-ambiguous":
            result = _clear_ambiguous(args, trade, state)
        elif args.command == "revalidate-terminal":
            result = _revalidate_terminal(args, trade, state)
        elif args.command == "confirm-terminal-zero":
            result = _confirm_terminal_zero(args, trade, state)
        elif args.command == "confirm-terminal-canceled":
            result = _confirm_terminal_canceled(args, trade, state)
        else:
            result = _restore_partial_wallet_mutation(args, trade, state)
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
