import os, time
from typing import Dict
from models import Decision, PositionMemo
from utils import tg

# Tracks 1 open position memo per symbol (in-memory)
OPEN: Dict[str, PositionMemo] = {}
LAST_DAILY_RESET = 0
DAILY_PNL = 0.0


def _reset_daily_if_needed():
    global LAST_DAILY_RESET, DAILY_PNL
    now = time.gmtime()
    midnight = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, 0))
    if LAST_DAILY_RESET == 0:
        LAST_DAILY_RESET = midnight
    elif time.time() - LAST_DAILY_RESET >= 86400:
        LAST_DAILY_RESET = midnight
        DAILY_PNL = 0.0


def has_open_position(ex, symbol: str) -> bool:
    """Exchange-truth: returns True if there is a non-zero position on symbol."""
    try:
        positions = ex.fetch_positions([symbol])
        for p in positions:
            amt = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)
            if abs(amt) > 0:
                return True
    except Exception:
        pass
    return symbol in OPEN


def _qty_from_notional(ex, symbol: str, notional_usdt: float) -> float:
    """Convert notional to qty, snap to lot size; return 0 if below exchange minQty."""
    mkt = ex.market(symbol)  # has filters/limits
    price = ex.fetch_ticker(symbol)["last"]
    raw_qty = notional_usdt / price

    min_amt = float(mkt.get("limits", {}).get("amount", {}).get("min") or 0)

    # Use ccxt precision rounding
    try:
        qty = float(ex.amount_to_precision(symbol, raw_qty))
    except Exception:
        qty = raw_qty

    if min_amt and qty < min_amt:
        return 0.0
    return qty


def _ensure_symbol_settings(ex, symbol: str):
    lev = int(os.getenv("LEVERAGE", "20"))
    mode = os.getenv("MARGIN_MODE", "isolated")
    try:
        ex.set_margin_mode(mode, symbol)
    except Exception:
        pass
    try:
        ex.set_leverage(lev, symbol)
    except Exception:
        pass
    # Ensure ONE-WAY mode (False). If your account is in hedge and you want to keep it,
    # we can add positionSide fields instead.
    try:
        ex.set_position_mode(False)
    except Exception:
        pass


def _place_brackets(ex, decision: Decision, qty: float):
    """
    Place reduce-only TP/SL as market-triggered orders on Binance USDM.
    We avoid 'closePosition' and only use 'reduceOnly' to prevent -1106.
    """
    reduce_side = "sell" if decision.side == "long" else "buy"
    base_params = {"reduceOnly": True, "workingType": "MARK_PRICE"}

    # TAKE PROFIT
    try:
        ex.create_order(
            decision.symbol,
            "TAKE_PROFIT_MARKET",
            reduce_side,
            qty,                # amount required when not using 'closePosition'
            None,
            {**base_params, "stopPrice": float(decision.tp)},
        )
    except Exception as e:
        tg(f"⚠️ TP order error {decision.symbol}: {e}")

    # STOP LOSS
    try:
        ex.create_order(
            decision.symbol,
            "STOP_MARKET",
            reduce_side,
            qty,
            None,
            {**base_params, "stopPrice": float(decision.sl)},
        )
    except Exception as e:
        tg(f"⚠️ SL order error {decision.symbol}: {e}")


def execute(ex, decision: Decision):
    """Places entry + brackets, sends Telegram logs."""
    _reset_daily_if_needed()

    # One-position-per-symbol
    if has_open_position(ex, decision.symbol):
        tg(f"⏸️ Skip {decision.symbol}: already have an open position.")
        return

    # Daily loss cap guard
    daily_cap = float(os.getenv("DAILY_LOSS_CAP_USDT", "30"))
    from builtins import abs as _abs
    if DAILY_PNL <= -_abs(daily_cap):
        tg(f"🛑 <b>Daily loss cap reached</b>. Skipping trade for {decision.symbol}.")
        return

    _ensure_symbol_settings(ex, decision.symbol)

    qty = _qty_from_notional(ex, decision.symbol, decision.size_usdt)
    if qty <= 0:
        tg(f"⏸️ Skip {decision.symbol}: notional {decision.size_usdt} USDT is below exchange minimum (minQty). "
           f"Increase RISK_NOTIONAL_MAP for this symbol.")
        return

    # ---- ENTRY ----
    try:
        side = "buy" if decision.side == "long" else "sell"
        order = ex.create_order(decision.symbol, decision.entry_type, side, qty)
        entry_price = float(order.get("price") or ex.fetch_ticker(decision.symbol)["last"])
        OPEN[decision.symbol] = PositionMemo(
            symbol=decision.symbol, side=decision.side, qty=qty,
            entry_price=entry_price, opened_at=time.time()
        )
        tg(
            f"🚀 <b>ENTRY</b> {decision.symbol} {decision.side.upper()} qty={qty:.6f}\n"
            f"SL={decision.sl:.4f}  TP={decision.tp:.4f}\n"
            f"conf={decision.confidence:.2f}  reason={decision.reason}"
        )
    except Exception as e:
        tg(f"❌ Entry failed {decision.symbol}: {e}")
        return

    # ---- BRACKETS ----
    _place_brackets(ex, decision, qty)


def poll_positions_and_report(ex):
    """Call this every ~10s: if position closed, infer exit and PnL; cancel sibling if needed."""
    global DAILY_PNL
    for sym, memo in list(OPEN.items()):
        try:
            positions = ex.fetch_positions([sym])
            pos = None
            for p in positions:
                amt = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)
                if abs(amt) > 0:
                    pos = p
                    break
            if pos is None:
                px = float(ex.fetch_ticker(sym)["last"])
                pnl = (px - memo.entry_price) * memo.qty if memo.side == "long" else (memo.entry_price - px) * memo.qty
                DAILY_PNL += pnl
                hold = int(time.time() - memo.opened_at)
                tg(
                    f"✅ <b>EXIT</b> {sym} {memo.side.UPPER()}  PnL: {pnl:+.2f}  Hold: {hold}s\n"
                    f"Today PnL: {DAILY_PNL:+.2f}"
                )
                del OPEN[sym]
        except Exception:
            continue
