import os, time
from typing import Dict, Optional, List, Tuple
from models import Decision, PositionMemo
from utils import tg

# Tracks 1 open position memo per symbol (in-memory)
OPEN: Dict[str, PositionMemo] = {}
# Tracks bracket order ids per symbol: {"tp": "...", "sl": "..."}
BRACKETS: Dict[str, Dict[str, str]] = {}

# Dynamic re-arm tracking
PROGRESS_STAGE: Dict[str, int] = {}   # symbol -> current stage index (-1 before first)
LAST_REARM_AT: Dict[str, float] = {}  # symbol -> last re-arm time
LAST_ROI_TELL: Dict[str, float] = {}  # symbol -> last time we reported progress heartbeat

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

    ids: Dict[str, Optional[str]] = {"tp": None, "sl": None}

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
        # Still attempt to cancel any stray reduce-only orders:
        try:
            open_orders = ex.fetch_open_orders(symbol)
        except Exception:
            open_orders = []
        for oo in open_orders:
            typ = str(oo.get("type", "")).upper()
            reduce_only = bool(oo.get("info", {}).get("reduceOnly")) or bool(oo.get("reduceOnly"))
            if reduce_only and typ in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                try:
                    ex.cancel_order(oo["id"], symbol)
                except Exception:
                    pass
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


# ----------------------------
# Dynamic TP/SL (ROI/ROE ladder)
# ----------------------------

def _parse_pct_list(env_name: str) -> List[float]:
    raw = (os.getenv(env_name) or "").strip()
    if not raw:
        return []
    out: List[float] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(float(x))
        except Exception:
            pass
    return out


def _current_price(ex, sym: str) -> float:
    """Use mark price if available (safer), else last."""
    t = ex.fetch_ticker(sym)
    if (os.getenv("DYN_USE_MARK", "true").lower() == "true"):
        try:
            mp = float(t.get("info", {}).get("markPrice") or 0.0)
            if mp > 0:
                return mp
        except Exception:
            pass
    return float(t["last"])


def _favorable_progress_pct(entry: float, current: float, side: str) -> float:
    """
    Returns progress in % according to DYN_METRIC:
      - PRICE_PCT: raw price % move in your favor
      - ROE: price % * leverage
      - UPNL_PCT: unrealized PnL % vs notional (qty*entry)
    """
    entry = float(entry); current = float(current)
    if entry <= 0 or current <= 0:
        return 0.0

    # raw price move %
    if side == "long":
        price_pct = (current / entry - 1.0) * 100.0
    else:
        price_pct = (entry / current - 1.0) * 100.0

    metric = (os.getenv("DYN_METRIC", "PRICE_PCT") or "PRICE_PCT").upper()
    if metric == "ROE":
        lev = float(os.getenv("LEVERAGE", "20") or 20)
        return price_pct * lev
    elif metric == "UPNL_PCT":
        # PnL % vs notional ~ (current-entry)/entry for long (qty cancels)
        upnl_pct = ((current - entry) / entry) * 100.0 if side == "long" else ((entry - current) / entry) * 100.0
        return upnl_pct
    else:
        return price_pct


def _price_from_roi(entry: float, pct: float, side: str) -> float:
    """
    Convert a target ROI% (price % vs entry) into an absolute price for SL/TP.
    Long: +pct -> entry * (1+pct/100)
    Short: +pct -> entry / (1+pct/100)
    """
    entry = float(entry); pct = float(pct)
    if side == "long":
        return entry * (1.0 + pct / 100.0)
    else:
        return entry / (1.0 + pct / 100.0)


def _rearm_brackets_abs(ex, symbol: str, side: str, qty: float,
                        new_sl: Optional[float], new_tp: Optional[float]) -> Dict[str, Optional[str]]:
    """
    Cancel existing reduce-only TP/SL and place new ones at absolute prices.
    Returns new ids dict {"tp": "...", "sl": "..."} (ids may be None).
    """
    try:
        open_orders = ex.fetch_open_orders(symbol)
    except Exception:
        open_orders = []

    # cancel existing reduce-only STOP/TP orders
    for oo in open_orders:
        typ = str(oo.get("type", "")).upper()
        reduce_only = bool(oo.get("info", {}).get("reduceOnly")) or bool(oo.get("reduceOnly"))
        if reduce_only and typ in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            try:
                ex.cancel_order(oo["id"], symbol)
            except Exception as e:
                tg(f"⚠️ Could not cancel old bracket {oo.get('id')} on {symbol}: {e}")

    reduce_side = "sell" if side == "long" else "buy"
    base = {"reduceOnly": True, "workingType": "MARK_PRICE"}

    new_ids: Dict[str, Optional[str]] = {"tp": None, "sl": None}

    # place new TP
    if new_tp is not None:
        try:
            tp = ex.create_order(symbol, "TAKE_PROFIT_MARKET", reduce_side, qty, None,
                                 {**base, "stopPrice": float(new_tp)})
            new_ids["tp"] = str(tp.get("id") or tp.get("orderId") or "")
        except Exception as e:
            tg(f"⚠️ TP re-arm error {symbol}: {e}")

    # place new SL
    if new_sl is not None:
        try:
            sl = ex.create_order(symbol, "STOP_MARKET", reduce_side, qty, None,
                                 {**base, "stopPrice": float(new_sl)})
            new_ids["sl"] = str(sl.get("id") or sl.get("orderId") or "")
        except Exception as e:
            tg(f"⚠️ SL re-arm error {symbol}: {e}")

    return new_ids


def _maybe_dynamic_rearm(ex, sym: str, memo: PositionMemo):
    """
    If progress crosses the next configured stage, move SL/TP per DYN_* envs.
    Progress metric is controlled by DYN_METRIC (PRICE_PCT|ROE|UPNL_PCT).
    """
    stages = _parse_pct_list("DYN_ROI_STAGES")
    if not stages:
        return  # feature off

    sl_targets = _parse_pct_list("DYN_SL_AT_STAGE")
    tp_targets = _parse_pct_list("DYN_TP_AT_STAGE")

    # Validate lengths
    if sl_targets and len(sl_targets) != len(stages):
        tg("⚠️ DYN_SL_AT_STAGE length != DYN_ROI_STAGES; ignoring dynamic SL.")
        sl_targets = []
    if tp_targets and len(tp_targets) != len(stages):
        tg("⚠️ DYN_TP_AT_STAGE length != DYN_ROI_STAGES; ignoring dynamic TP.")
        tp_targets = []

    cur_px = _current_price(ex, sym)
    progress = _favorable_progress_pct(memo.entry_price, cur_px, memo.side)

    # Optional heartbeat every ~60s when DEBUG_DECISIONS=true
    if (os.getenv("DEBUG_DECISIONS", "false").lower() == "true"):
        now = time.time()
        if now - LAST_ROI_TELL.get(sym, 0) > 60:
            nxt = stages[min(PROGRESS_STAGE.get(sym, -1) + 1, len(stages) - 1)]
            tg(f"📈 {sym} progress={progress:.2f}% (metric={os.getenv('DYN_METRIC','PRICE_PCT')}) | next stage={nxt}%")
            LAST_ROI_TELL[sym] = now

    cur_ix = PROGRESS_STAGE.get(sym, -1)
    next_ix = cur_ix + 1
    if next_ix >= len(stages):
        return  # at max stage

    if progress < stages[next_ix] - 1e-9:
        return  # haven't reached next stage

    # Cooldown to avoid thrash
    min_gap = int(os.getenv("DYN_MIN_REARM_SECONDS", "20") or 20)
    if time.time() - LAST_REARM_AT.get(sym, 0.0) < min_gap:
        return

    # Compute new absolute prices (percent vs entry)
    new_sl = None
    if sl_targets:
        new_sl = _price_from_roi(memo.entry_price, sl_targets[next_ix], memo.side)

    new_tp = None
    if tp_targets:
        new_tp = _price_from_roi(memo.entry_price, tp_targets[next_ix], memo.side)

    # Cancel old brackets and place new ones
    new_ids = _rearm_brackets_abs(ex, sym, memo.side, memo.qty, new_sl, new_tp)
    old = BRACKETS.get(sym, {})
    BRACKETS[sym] = {
        "tp": new_ids.get("tp") or old.get("tp"),
        "sl": new_ids.get("sl") or old.get("sl"),
    }

    PROGRESS_STAGE[sym] = next_ix
    LAST_REARM_AT[sym] = time.time()
    tg(
        f"🔧 {sym} dynamic re-arm → stage {next_ix+1}/{len(stages)} "
        f"(progress={progress:.2f}%, metric={os.getenv('DYN_METRIC','PRICE_PCT')}) "
        f"{'SL→'+str(round(new_sl, 6)) if new_sl is not None else ''} "
        f"{'TP→'+str(round(new_tp, 6)) if new_tp is not None else ''}"
    )


# ----------------------------
# ROE-based initial brackets
# ----------------------------

def _roe_brackets(entry_price: float, side: str) -> Tuple[float, float]:
    """
    Return (SL, TP) prices based on ROE % targets, independent of ATR.
    ROE_SL/ROE_TP are specified as ROE percent (e.g., -5 and 10).
    Conversion to price move uses leverage: price% = ROE% / leverage.
    """
    lev = float(os.getenv("LEVERAGE", "20") or 20)
    # ROE percents (signed)
    sl_roe = float(os.getenv("ROE_SL", "-5") or -5.0)
    tp_roe = float(os.getenv("ROE_TP", "10") or 10.0)

    # Convert to price move %
    sl_price_pct = sl_roe / lev
    tp_price_pct = tp_roe / lev

    if side == "long":
        sl = entry_price * (1.0 + sl_price_pct / 100.0)
        tp = entry_price * (1.0 + tp_price_pct / 100.0)
    else:
        # For shorts, +ROE means price down. Using reciprocal formula:
        sl = entry_price / (1.0 + sl_price_pct / 100.0)
        tp = entry_price / (1.0 + tp_price_pct / 100.0)
    return float(sl), float(tp)


# ----------------------------
# Main execution & polling
# ----------------------------

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

        # If BRACKET_MODE=ROE, overwrite initial decision SL/TP using ROE targets
        if (os.getenv("BRACKET_MODE", "ATR").upper() == "ROE"):
            sl_px, tp_px = _roe_brackets(entry_price, decision.side)
            decision.sl, decision.tp = sl_px, tp_px

        OPEN[decision.symbol] = PositionMemo(
            symbol=decision.symbol, side=decision.side, qty=qty,
            entry_price=entry_price, opened_at=time.time()
        )
        # init dynamic ladder trackers
        PROGRESS_STAGE[decision.symbol] = -1
        LAST_REARM_AT[decision.symbol] = 0.0
        LAST_ROI_TELL[decision.symbol] = 0.0

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
    adjust dynamic TP/SL while open, send PnL TG, cancel sibling, clear state.
    """
    global DAILY_PNL
    for sym, memo in list(OPEN.items()):
        try:
            # While position is open, consider dynamic re-arm based on configured stages
            try:
                _maybe_dynamic_rearm(ex, sym, memo)
            except Exception as e_dyn:
                tg(f"⚠️ dynamic re-arm error {sym}: {e_dyn}")

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
                    PROGRESS_STAGE.pop(sym, None)
                    LAST_REARM_AT.pop(sym, None)
                    LAST_ROI_TELL.pop(sym, None)

        except Exception as e:
            tg(f"⚠️ poll error while checking {sym}: {e}")
            continue
