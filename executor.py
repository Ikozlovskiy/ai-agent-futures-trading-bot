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
    midnight = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0,0,0,0,0,0))
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
    price = ex.fetch_ticker(symbol)["last"]
    qty = notional_usdt / price
    return float(ex.amount_to_precision(symbol, qty))

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

def _place_brackets(ex, decision: Decision, qty: float):
    reduce_side = "sell" if decision.side == "long" else "buy"
    params = {"reduceOnly": True, "closePosition": True, "workingType": "MARK_PRICE"}
    # TP
    try:
        ex.create_order(decision.symbol, "take_profit_market", reduce_side, None, None,
                        {**params, "stopPrice": float(decision.tp)})
    except Exception:
        ex.create_order(decision.symbol, "market", reduce_side, None, None,
                        {**params, "stopPrice": float(decision.tp), "type":"TAKE_PROFIT_MARKET"})
    # SL
    try:
        ex.create_order(decision.symbol, "stop_market", reduce_side, None, None,
                        {**params, "stopPrice": float(decision.sl)})
    except Exception:
        ex.create_order(decision.symbol, "market", reduce_side, None, None,
                        {**params, "stopPrice": float(decision.sl), "type":"STOP_MARKET"})

def execute(ex, decision: Decision):
    """Places entry + brackets, sends Telegram logs."""
    _reset_daily_if_needed()

    # One-position-per-symbol
    if has_open_position(ex, decision.symbol):
        tg(f"⏸️ Skip {decision.symbol}: already have an open position.")
        return

    # Daily loss cap guard
    daily_cap = float(os.getenv("DAILY_LOSS_CAP_USDT", "30"))
    from builtins import abs as _abs  # avoid shadowing
    if DAILY_PNL <= -_abs(daily_cap):
        tg(f"🛑 <b>Daily loss cap reached</b>. Skipping trade for {decision.symbol}.")
        return

    _ensure_symbol_settings(ex, decision.symbol)
    qty = _qty_from_notional(ex, decision.symbol, decision.size_usdt)
    if qty <= 0:
        tg(f"⚠️ Qty <= 0 for {decision.symbol}. Skip.")
        return

    try:
        side = "buy" if decision.side == "long" else "sell"
        order = ex.create_order(decision.symbol, decision.entry_type, side, qty)
        entry_price = float(order.get("price") or ex.fetch_ticker(decision.symbol)["last"])
        OPEN[decision.symbol] = PositionMemo(
            symbol=decision.symbol, side=decision.side, qty=qty,
            entry_price=entry_price, opened_at=time.time()
        )
        tg(f"🚀 <b>ENTRY</b> {decision.symbol} {decision.side.upper()} qty={qty:.6f}\n"
           f"SL={decision.sl:.4f}  TP={decision.tp:.4f}\n"
           f"conf={decision.confidence:.2f}  reason={decision.reason}")
        _place_brackets(ex, decision, qty)
    except Exception as e:
        tg(f"❌ Entry failed {decision.symbol}: {e}")

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
                tg(f"✅ <b>EXIT</b> {sym} {memo.side.upper()}  PnL: {pnl:+.2f}  Hold: {hold}s\n"
                   f"Today PnL: {DAILY_PNL:+.2f}")
                del OPEN[sym]
        except Exception:
            continue
