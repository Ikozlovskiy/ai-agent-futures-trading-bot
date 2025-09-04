import os, time
from typing import Dict, Optional
from models import Decision, PositionMemo
from utils import tg

# Tracks 1 open position memo per symbol (in-memory)
OPEN: Dict[str, PositionMemo] = {}
# Tracks bracket order ids per symbol: {"tp": "...", "sl": "..."}
BRACKETS: Dict[str, Dict[str, str]] = {}

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


def _is_pos_open_amount(ex, symbol: str, amt: float) -> bool:
    """Treat tiny dust as 0 to avoid false 'open position'."""
    try:
        m = ex.market(symbol)
        min_amt = float(m.get("limits", {}).get("amount", {}).get("min") or 0)
    except Exception:
        min_amt = 0.0
    eps = max(min_amt / 10.0, 1e-9)
    return abs(amt) > eps


def has_open_position(ex, symbol: str) -> bool:
    """Exchange-truth: returns True if there is a non-zero position on symbol."""
    try:
        positions = ex.fetch_positions([symbol])
        for p in positions:
            amt = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)
            if _is_pos_open_amount(ex, symbol, amt):
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


def _place_brackets(ex, decision: Decision, qty: float) -> Dict[str, Optional[str]]:
    """
    Place reduce-only TP/SL as market-triggered orders on Binance USDM.
    Returns order ids dict: {"tp": "...", "sl": "..."} (ids may be None on error).
    """
    reduce_side = "sell" if decision.side == "long" else "buy"
    base_params = {"reduceOnly": True, "workingType": "MARK_PRICE"}

    ids = {"tp": None, "sl": None}

    # TAKE PROFIT
    try:
        tp = ex.create_order(
            decision.symbol,
            "TAKE_PROFIT_MARKET",
            reduce_side,
            qty,
            None,
            {**base_params, "stopPrice": float(decision.tp)},
        )
        ids["tp"] = str(tp.get("id") or tp.get("orderId") or "")
    except Exception as e:
        tg(f"⚠️ TP order error {decision.symbol}: {e}")

    # STOP LOSS
    try:
        sl = ex.create_order(
            decision.symbol,
            "STOP_MARKET",
            reduce_side,
            qty,
            None,
            {**base_params, "stopPrice": float(decision.sl)},
        )
        ids["sl"] = str(sl.get("id") or sl.get("orderId") or "")
    except Exception as e:
        tg(f"⚠️ SL order error {decision.symbol}: {e}")

    return ids


def _cancel_open_brackets(ex, symbol: str):
    """Cancel any open TP/SL reduce-only orders for a symbol using stored ids."""
    if symbol not in BRACKETS:
        return
    ids = BRACKETS[symbol]
    try:
        open_orders = ex.fetch_open_orders(symbol)
    except Exception:
        open_orders = []

    def cancel(order_id: str):
        if not order_id:
            return
        for oo in open_orders:
            oid = str(oo.get("id") or oo.get("orderId") or "")
            if oid == str(order_id):
                try:
                    ex.cancel_order(order_id, symbol)
                    tg(f"🔁 Canceled sibling order {order_id} for {symbol}")
                except Exception as ce:
                    tg(f"⚠️ Could not cancel sibling {order_id} for {symbol}: {ce}")

    cancel(ids.get("tp", ""))
    cancel(ids.get("sl", ""))
    BRACKETS.pop(symbol, None)


def _try_detect_exit_by_orders(ex, symbol: str) -> bool:
    """
    If either TP or SL order id is 'closed', treat position as exited and return True.
    This complements position-based detection and ensures we log PnL promptly.
    """
    ids = BRACKETS.get(symbol)
    if not ids:
        return False

    def is_closed(oid: Optional[str]) -> bool:
        if not oid:
            return False
        try:
            o = ex.fetch_order(oid, symbol)
            return str(o.get("status", "")).lower() in ("closed", "filled")
        except Exception:
            return False

    return is_closed(ids.get("tp")) or is_closed(ids.get("sl"))


def execute(ex, decision: Decision):
    """Places entry + brackets, sends Telegram logs."""
    _reset_daily_if_needed()

    if has_open_position(ex, decision.symbol):
        tg(f"⏸️ Skip {decision.symbol}: already have an open position.")
        return

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
    ids = _place_brackets(ex, decision, qty)
    BRACKETS[decision.symbol] = ids


def poll_positions_and_report(ex):
    """
    Every ~10s: detect exit (by position OR by bracket order fill),
    send PnL TG, cancel sibling, clear state.
    """
    global DAILY_PNL
    for sym, memo in list(OPEN.items()):
        try:
            # 1) Fast path: did one of our bracket orders fill?
            exited_by_order = _try_detect_exit_by_orders(ex, sym)

            # 2) Position-based check (robust to dust)
            positions = ex.fetch_positions([sym])
            pos_open = False
            for p in positions:
                amt = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)
                if _is_pos_open_amount(ex, sym, amt):
                    pos_open = True
                    break

            if exited_by_order or not pos_open:
                # infer PnL from last price at the moment of detection
                px = float(ex.fetch_ticker(sym)["last"])
                pnl = (px - memo.entry_price) * memo.qty if memo.side == "long" \
                    else (memo.entry_price - px) * memo.qty
                DAILY_PNL += pnl
                hold = int(time.time() - memo.opened_at)

                tg(
                    f"✅ <b>EXIT</b> {sym} {memo.side.upper()}  PnL: {pnl:+.2f}  Hold: {hold}s\n"
                    f"Today PnL: {DAILY_PNL:+.2f}"
                )

                # cancel any leftover TP/SL and clear
                try:
                    _cancel_open_brackets(ex, sym)
                finally:
                    OPEN.pop(sym, None)

        except Exception as e:
            tg(f"⚠️ poll error while checking {sym}: {e}")
            continue
