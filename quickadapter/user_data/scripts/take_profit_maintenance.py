"""Inspect or recover a fail-closed QuickAdapter take-profit state.

Run this tool only while the bot is stopped and only after checking the CEX order
history. Repair canonical Freqtrade order data before revalidating a terminal fill.
"""

import argparse
import datetime
import json
import sys
from collections.abc import Sequence
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
        source_version = raw_state.get("version")
        if isinstance(source_version, bool) or not isinstance(source_version, int):
            raise ValueError("Take-profit state version must be an integer")
        state = TakeProfitStageState.from_mapping(raw_state)
        return TakeProfitStageManager.reconcile_migrated_state(
            trade,
            state,
            source_version,
            amount_quantizer=partial(_quantize_trade_amount, trade),
        )
    except ValueError as error:
        raise MaintenanceError(f"invalid take-profit state: {error}") from error


def _order_summary(order) -> dict[str, object]:
    return {
        "order_id": order.order_id,
        "tag": order.ft_order_tag,
        "side": order.ft_order_side,
        "status": order.status,
        "is_open": order.ft_is_open,
        "amount": order.amount,
        "filled": order.filled,
        "remaining": order.remaining,
        "fee_base": order.ft_fee_base,
        "order_date": order.order_date_utc.isoformat(),
        "filled_date": (
            order.order_filled_utc.isoformat()
            if order.order_filled_utc is not None
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
    blocked_order = next(
        (order for order in trade.orders if order.order_id == state.blocked_order_id),
        None,
    )
    zero_fill_status = (
        (blocked_order.status or "").lower() if blocked_order is not None else None
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
                attempt.recovery_deadline
                if attempt is not None and attempt.status == "ambiguous"
                else None
            ),
            "required_clear_confirmation": (
                f"NO-ORDER:{trade.id}:{attempt.attempt_id}"
                if attempt is not None and attempt.status == "ambiguous"
                else None
            ),
            "required_revalidation_confirmation": (
                f"RECONCILE:{trade.id}:{state.blocked_order_id}"
                if state.blocked_reason
                in {"unknown_terminal_fill", "unattributed_exit_fill"}
                and state.blocked_order_id is not None
                else None
            ),
            "required_zero_fill_confirmation": (
                f"NO-FILL:{trade.id}:{state.blocked_order_id}:{zero_fill_status}"
                if state.blocked_reason
                in {"unknown_terminal_fill", "unknown_external_exit_fill"}
                and state.blocked_order_id is not None
                and zero_fill_status in {"canceled", "cancelled"}
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
                "stage_targets": [
                    {"stage": stage, "amount": amount}
                    for stage, amount in state.stage_targets
                ],
                "deferred_stages": list(state.deferred_stages),
                "attributed_order_ids": list(state.attributed_order_ids),
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
            },
            "operator_resolutions": [
                resolution.to_mapping() for resolution in state.operator_resolutions
            ],
        },
        "orders": [
            _order_summary(order)
            for order in trade.orders
            if order.ft_order_side != trade.entry_side
        ],
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
    if not attempt_id or "_" in attempt_id:
        _fail("attempt-id must be non-empty and contain no underscore")
    if state.blocked_reason is not None:
        _fail(f"take-profit state is blocked: {state.blocked_reason}")
    attempt = state.active_attempt
    if (
        attempt is None
        or attempt.attempt_id != attempt_id
        or attempt.status != "ambiguous"
    ):
        _fail("no matching ambiguous attempt")
    parsed_tag = TakeProfitStageManager.parse_order_tag(attempt.tag)
    if parsed_tag != (trade.trade_direction, attempt.stage, attempt.attempt_id):
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

    resolved_state = TakeProfitStageManager.resolve_ambiguous_attempt(
        state,
        args.attempt_id,
        datetime.datetime.now(datetime.timezone.utc),
    )
    trade.set_custom_data(
        TakeProfitStageManager.STATE_KEY,
        resolved_state.to_mapping(),
    )
    persisted_state = _load_state(trade)
    if persisted_state != resolved_state:
        _fail("take-profit state postcondition failed after commit")
    return _summary(trade, persisted_state, command="clear-ambiguous")


def _revalidate_terminal(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    if not args.order_id:
        _fail("order-id must be non-empty")
    expected_confirmation = f"RECONCILE:{trade.id}:{args.order_id}"
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
    if args.confirmation != expected_confirmation:
        _fail(f"confirmation must be exactly {expected_confirmation!r}")

    trade.set_custom_data(
        TakeProfitStageManager.STATE_KEY,
        resolved_state.to_mapping(),
    )
    persisted_state = _load_state(trade)
    if persisted_state != resolved_state:
        _fail("take-profit state postcondition failed after commit")
    return _summary(trade, persisted_state, command="revalidate-terminal")


def _confirm_terminal_zero(
    args: argparse.Namespace,
    trade: Trade,
    state: TakeProfitStageState,
) -> dict[str, object]:
    if not args.order_id:
        _fail("order-id must be non-empty")
    expected_confirmation = f"NO-FILL:{trade.id}:{args.order_id}:{args.terminal_status}"
    try:
        resolved_state = TakeProfitStageManager.confirm_terminal_zero_fill(
            trade,
            state,
            args.order_id,
            args.terminal_status,
            datetime.datetime.now(datetime.timezone.utc),
            amount_quantizer=lambda amount: _quantize_trade_amount(trade, amount),
        )
    except ValueError as error:
        raise MaintenanceError(
            f"terminal zero fill cannot be confirmed: {error}"
        ) from error

    if not args.apply:
        return _summary(
            trade,
            state,
            command="confirm-terminal-zero",
            would_apply=True,
        )
    if args.confirmation != expected_confirmation:
        _fail(f"confirmation must be exactly {expected_confirmation!r}")

    trade.set_custom_data(
        TakeProfitStageManager.STATE_KEY,
        resolved_state.to_mapping(),
    )
    persisted_state = _load_state(trade)
    if persisted_state != resolved_state:
        _fail("take-profit state postcondition failed after commit")
    return _summary(trade, persisted_state, command="confirm-terminal-zero")


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
        else:
            result = _confirm_terminal_zero(args, trade, state)
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
