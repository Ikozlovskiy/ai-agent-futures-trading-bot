import os, time
from typing import Dict, Optional, List, Tuple

import numpy as np

from models import Decision, PositionMemo
from utils import tg
from datahub import fetch_candles

# Tracks 1 open position memo per symbol (in-memory)
OPEN: Dict[str, PositionMemo] = {}
# Tracks bracket order ids per symbol: {"tp": "...", "sl": "..."} or {"tp1": "...", "tp2": "...", "sl": "..."}
BRACKETS: Dict[str, Dict[str, str]] = {}
# Tracks remaining qty for LADDER mode (decreases as TPs fill)
LADDER_REMAINING_QTY: Dict[str, float] = {}
# Tracks TP prices for progressive SL management
LADDER_TP_PRICES: Dict[str, Dict[str, float]] = {}  # symbol -> {tp1: price, tp2: price, tp3: price}

# Dynamic re-arm tracking
PROGRESS_STAGE: Dict[str, int] = {}   # symbol -> current stage index (-1 before first)
LAST_REARM_AT: Dict[str, float] = {}  # symbol -> last re-arm time
LAST_ROI_TELL: Dict[str, float] = {}  # symbol -> last time we reported progress heartbeat

LAST_DAILY_RESET = 0
DAILY_PNL = 0.0

# Scalper-specific tracking (for dynamic exits)
SCALPER_ENTRY_TIME: Dict[str, float] = {}  # symbol -> entry timestamp
SCALPER_HIGHEST_PROFIT: Dict[str, float] = {}  # symbol -> highest profit seen
SCALPER_CONSECUTIVE_AGAINST: Dict[str, int] = {}  # symbol -> count of candles against us

# Cooldown tracking (persists after trade exits)
LAST_TRADE_EXIT_TIME: Dict[str, float] = {}  # symbol -> last exit timestamp
LAST_TRADE_WAS_LOSS: Dict[str, bool] = {}  # symbol -> True if last trade was a loss


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
    # Use 2x minQty as threshold to catch rounding dust (typically minQty is 0.001)
    # This ensures amounts like 0.001-0.002 are treated as dust and ignored
    eps = max(min_amt * 2.0, 1e-6)
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


# -------- Tick-size / safe stop helpers --------

def _binance_tick_size(ex, symbol: str) -> float:
    """
    Get price tick size for a symbol from ccxt market info.
    Falls back to precision if filters not present.
    """
    try:
        m = ex.market(symbol)
        flt = m.get("info", {}).get("filters", [])
        for f in flt:
            if f.get("filterType") in ("PRICE_FILTER", "PRICE_FILTER "):
                ts = float(f.get("tickSize") or 0)
                if ts > 0:
                    return ts
        # Fallback to precision -> 10^-precision
        prec = m.get("precision", {}).get("price")
        if prec is not None:
            return 10 ** (-int(prec))
    except Exception:
        pass
    return 1e-6  # last resort


def _snap_to_tick(ex, symbol: str, px: float) -> float:
    """Round price to the exchange tick using ccxt helpers when available."""
    try:
        return float(ex.price_to_precision(symbol, px))
    except Exception:
        ts = _binance_tick_size(ex, symbol)
        return round(px / ts) * ts


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


def _safe_stop_price(ex, sym: str, side: str, px: float, kind: str) -> float:
    """
    Make a stopPrice valid for Binance USDM:
    - Snap to tick size
    - Keep at least STOP_TICKS_AWAY ticks away from current mark price on the correct side
    - Optionally enforce a minimal pct buffer (STOP_BUFFER_PCT_MIN)
    """
    if px is None or px <= 0:
        return px

    cur = _current_price(ex, sym)
    tick = _binance_tick_size(ex, sym)
    ticks_away = int(os.getenv("STOP_TICKS_AWAY", "2") or 2)
    pct_min = float(os.getenv("STOP_BUFFER_PCT_MIN", "0.0003") or 0.0003)  # 0.03%

    # Choose a guardrail based on order kind & side
    if kind == "SL":
        target = cur - ticks_away * tick if side == "long" else cur + ticks_away * tick
        guard = cur * (1 - pct_min) if side == "long" else cur * (1 + pct_min)
        # take the safer (further) one
        if side == "long":
            target = min(target, guard)
            if px >= target:
                px = target
        else:
            target = max(target, guard)
            if px <= target:
                px = target
    else:  # "TP"
        target = cur + ticks_away * tick if side == "long" else cur - ticks_away * tick
        guard = cur * (1 + pct_min) if side == "long" else cur * (1 - pct_min)
        if side == "long":
            target = max(target, guard)
            if px <= target:
                px = target
        else:
            target = min(target, guard)
            if px >= target:
                px = target

    return _snap_to_tick(ex, sym, px)


# -------- Dynamic TP/SL (ROI/ROE ladder) --------

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


def _favorable_progress_pct(entry: float, current: float, side: str) -> float:
    """
    Returns progress in % according to DYN_METRIC:
      - PRICE_PCT: raw price % move in your favor
      - ROE: price % * leverage (matches Binance UI ROE)
      - UPNL_PCT: unrealized PnL % vs notional
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
        upnl_pct = ((current - entry) / entry) * 100.0 if side == "long" else ((entry - current) / entry) * 100.0
        return upnl_pct
    else:
        return price_pct


def _price_from_roi(entry: float, pct: float, side: str) -> float:
    """
    Convert a target ROI% (price % vs entry) into an absolute price for SL/TP.
    For longs: +pct moves price up, -pct moves price down
    For shorts: +pct moves price down, -pct moves price up
    Long: entry * (1 + pct/100)
    Short: entry * (1 - pct/100)
    """
    entry = float(entry); pct = float(pct)
    if side == "long":
        return entry * (1.0 + pct / 100.0)
    else:
        # For shorts, positive pct means price moves down (favorable for profit)
        # and negative pct means price moves up (unfavorable, for SL)
        return entry * (1.0 - pct / 100.0)


def _rearm_brackets_abs(ex, symbol: str, side: str, qty: float,
                        new_sl: Optional[float], new_tp: Optional[float]) -> Dict[str, Optional[str]]:
    """
    Cancel existing reduce-only TP/SL and place new ones at absolute prices (tick-safe).
    Returns new ids dict {"tp": "...", "sl": "..."} (ids may be None).
    Validates that new SL is more favorable than current price (safety check).
    """
    tg(f"🔧 {symbol} _rearm_brackets_abs called | Side: {side} | Qty: {qty:.6f} | New SL: {new_sl} | New TP: {new_tp}")

    try:
        open_orders = ex.fetch_open_orders(symbol)
    except Exception:
        open_orders = []

    # Get current price for validation
    cur_price = _current_price(ex, symbol)
    tg(f"📊 {symbol} Current price: {cur_price:.6f} (mark price)")

    # Validate new SL is in the correct direction (more favorable than current price)
    original_sl = new_sl
    if new_sl is not None:
        if side == "long":
            # For longs, SL should be below current price
            if new_sl >= cur_price:
                tg(f"⚠️ {symbol} SL validation: Requested SL {new_sl:.6f} >= current {cur_price:.6f} (LONG)")
                # Get entry price from memo if available
                memo = OPEN.get(symbol)
                if memo:
                    # Check if entry is below current (profitable position)
                    if memo.entry_price < cur_price:
                        # Use the MINIMUM of requested SL and a safe distance below current
                        safe_sl = min(new_sl, cur_price * 0.999)  # 0.1% below current
                        if safe_sl > memo.entry_price:
                            new_sl = safe_sl
                            tg(f"✅ {symbol} Adjusted SL to {new_sl:.6f} (safe distance below current)")
                        else:
                            new_sl = memo.entry_price
                            tg(f"✅ {symbol} Adjusted SL to breakeven: {new_sl:.6f}")
                    else:
                        # Entry is above current (losing position) - skip SL adjustment
                        tg(f"⚠️ {symbol} Position is losing. Skipping SL adjustment.")
                        new_sl = None
                else:
                    tg(f"⚠️ {symbol} No memo found, skipping SL adjustment")
                    new_sl = None
        else:
            # For shorts, SL should be above current price
            if new_sl <= cur_price:
                tg(f"⚠️ {symbol} SL validation: Requested SL {new_sl:.6f} <= current {cur_price:.6f} (SHORT)")
                # Get entry price from memo if available
                memo = OPEN.get(symbol)
                if memo:
                    # Check if entry is above current (profitable position)
                    if memo.entry_price > cur_price:
                        # Use the MAXIMUM of requested SL and a safe distance above current
                        safe_sl = max(new_sl, cur_price * 1.001)  # 0.1% above current
                        if safe_sl < memo.entry_price:
                            new_sl = safe_sl
                            tg(f"✅ {symbol} Adjusted SL to {new_sl:.6f} (safe distance above current)")
                        else:
                            new_sl = memo.entry_price
                            tg(f"✅ {symbol} Adjusted SL to breakeven: {new_sl:.6f}")
                    else:
                        # Entry is below current (losing position) - skip SL adjustment
                        tg(f"⚠️ {symbol} Position is losing. Skipping SL adjustment.")
                        new_sl = None
                else:
                    tg(f"⚠️ {symbol} No memo found, skipping SL adjustment")
                    new_sl = None

        if original_sl != new_sl:
            tg(f"🔧 {symbol} SL adjusted: {original_sl:.6f} → {new_sl if new_sl else 'None'}")

    # Cancel existing reduce-only STOP orders (but preserve LADDER TPs)
    # For LADDER mode, only cancel SL during rearm (TPs remain active)
    is_ladder = symbol in LADDER_REMAINING_QTY
    if is_ladder:
        tg(f"🔧 {symbol} LADDER mode active - preserving TP orders, updating SL only")
    else:
        tg(f"🔧 {symbol} Single TP/SL mode - replacing both SL and TP")

    cancelled_count = 0
    for oo in open_orders:
        typ = str(oo.get("type", "")).upper()
        reduce_only = bool(oo.get("info", {}).get("reduceOnly")) or bool(oo.get("reduceOnly"))

        # For LADDER mode: only cancel STOP orders, preserve TAKE_PROFIT orders
        if is_ladder:
            should_cancel = reduce_only and typ in ("STOP_MARKET", "STOP", "STOP_LOSS")
        else:
            should_cancel = reduce_only and typ in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT", "STOP_LOSS")

        if should_cancel:
            try:
                oid = str(oo.get("id") or oo.get("orderId"))
                # Try Algo Order DELETE first
                try:
                    ex.request(
                        path='algoOrder',
                        api='fapiPrivate',
                        method='DELETE',
                        params={'symbol': symbol.replace('/', ''), 'algoId': oid}
                    )
                    cancelled_count += 1
                except Exception:
                    # Fallback to legacy cancel
                    ex.cancel_order(oid, symbol)
                    cancelled_count += 1
            except Exception as e:
                tg(f"⚠️ Could not cancel old bracket {oo.get('id')} on {symbol}: {e}")

    if cancelled_count > 0:
        tg(f"🗑️ Cancelled {cancelled_count} old bracket order(s) for {symbol}")

    reduce_side = "sell" if side == "long" else "buy"
    new_ids: Dict[str, Optional[str]] = {"tp": None, "sl": None}

    # place new TP
    if new_tp is not None:
        safe_tp = _safe_stop_price(ex, symbol, side, float(new_tp), "TP")
        new_ids["tp"] = _create_stop_order(ex, symbol, reduce_side, qty, safe_tp, "TAKE_PROFIT")
        if new_ids["tp"]:
            tg(f"📍 New TP placed for {symbol}: {safe_tp:.6f} (ID: {new_ids['tp']})")

    # place new SL
    if new_sl is not None:
        safe_sl = _safe_stop_price(ex, symbol, side, float(new_sl), "SL")
        new_ids["sl"] = _create_stop_order(ex, symbol, reduce_side, qty, safe_sl, "STOP")
        if new_ids["sl"]:
            tg(f"📍 New SL placed for {symbol}: {safe_sl:.6f} (ID: {new_ids['sl']})")

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
    metric = os.getenv("DYN_METRIC", "PRICE_PCT")

    # Heartbeat with progress tracking (log every ~60s)
    now = time.time()
    if now - LAST_ROI_TELL.get(sym, 0) > 60:
        cur_stage = PROGRESS_STAGE.get(sym, -1)
        next_stage_idx = cur_stage + 1
        if next_stage_idx < len(stages):
            nxt = stages[next_stage_idx]
            tg(f"📈 {sym} | Progress: {progress:.2f}% ({metric}) | Entry: {memo.entry_price:.6f} | Current: {cur_px:.6f} | Stage: {cur_stage+1}/{len(stages)} | Next: {nxt}%")
        else:
            tg(f"📈 {sym} | Progress: {progress:.2f}% ({metric}) | Entry: {memo.entry_price:.6f} | Current: {cur_px:.6f} | Stage: MAX ({cur_stage+1}/{len(stages)})")
        LAST_ROI_TELL[sym] = now

    cur_ix = PROGRESS_STAGE.get(sym, -1)
    next_ix = cur_ix + 1
    if next_ix >= len(stages):
        return  # at max stage

    if progress < stages[next_ix] - 1e-9:
        return  # haven't reached next stage

    # Cooldown to avoid thrash
    min_gap = int(os.getenv("DYN_MIN_REARM_SECONDS", "20") or 20)
    time_since_rearm = time.time() - LAST_REARM_AT.get(sym, 0.0)
    if time_since_rearm < min_gap:
        tg(f"⏱️ {sym} rearm cooldown active ({int(time_since_rearm)}s / {min_gap}s)")
        return

    # Log rearm trigger
    tg(
        f"🎯 {sym} REARM TRIGGERED!\n"
        f"Progress: {progress:.2f}% ({metric}) crossed stage {next_ix+1} threshold ({stages[next_ix]}%)\n"
        f"Entry: {memo.entry_price:.6f} | Current: {cur_px:.6f} | Side: {memo.side.upper()}"
    )

    # Compute new absolute prices (percent vs entry)
    new_sl = None
    if sl_targets:
        new_sl = _price_from_roi(memo.entry_price, sl_targets[next_ix], memo.side)
        tg(f"🔧 {sym} Calculated new SL: {new_sl:.6f} (target: {sl_targets[next_ix]}% price move)")

    new_tp = None
    if tp_targets:
        new_tp = _price_from_roi(memo.entry_price, tp_targets[next_ix], memo.side)
        tg(f"🔧 {sym} Calculated new TP: {new_tp:.6f} (target: {tp_targets[next_ix]}% price move)")

    # For LADDER mode, use remaining qty; otherwise use full qty
    is_ladder = sym in LADDER_REMAINING_QTY
    rearm_qty = LADDER_REMAINING_QTY.get(sym, memo.qty)
    bracket_mode = os.getenv("BRACKET_MODE", "ATR").upper()
    tg(f"🔧 {sym} Rearm mode: {bracket_mode} {'(LADDER active)' if is_ladder else '(Single TP/SL)'} | Qty: {rearm_qty:.6f}")

    # Cancel old brackets and place new ones (tick-safe)
    tg(f"🔄 {sym} Cancelling old brackets and placing new ones...")
    new_ids = _rearm_brackets_abs(ex, sym, memo.side, rearm_qty, new_sl, new_tp)
    old = BRACKETS.get(sym, {})
    BRACKETS[sym] = {
        "tp": new_ids.get("tp") or old.get("tp"),
        "sl": new_ids.get("sl") or old.get("sl"),
    }

    PROGRESS_STAGE[sym] = next_ix
    LAST_REARM_AT[sym] = time.time()

    # Log completion with details
    tg(
        f"✅ {sym} REARM COMPLETE → Stage {next_ix+1}/{len(stages)}\n"
        f"Progress: {progress:.2f}% ({metric})\n"
        f"{'New SL: '+str(round(new_sl, 6)) if new_sl is not None else 'SL: None'}"
        f"{' | ' if (new_sl and new_tp) else ''}"
        f"{'New TP: '+str(round(new_tp, 6)) if new_tp is not None else 'TP: None'}\n"
        f"Entry: {memo.entry_price:.6f} | Current: {cur_px:.6f}"
    )


# -------- ROE-based initial brackets --------

def _roe_brackets(entry_price: float, side: str) -> Tuple[float, float]:
    """
    Return (SL, TP) prices based on ROE % targets, independent of ATR.
    ROE_SL/ROE_TP are specified as ROE percent (e.g., -5 and 10).
    Conversion to price move uses leverage: price% = ROE% / leverage.
    For longs: positive % moves price up, negative % moves price down
    For shorts: positive % moves price down, negative % moves price up
    """
    lev = float(os.getenv("LEVERAGE", "20") or 20)
    sl_roe = float(os.getenv("ROE_SL", "-5") or -5.0)
    tp_roe = float(os.getenv("ROE_TP", "10") or 10.0)

    sl_price_pct = sl_roe / lev
    tp_price_pct = tp_roe / lev

    if side == "long":
        sl = entry_price * (1.0 + sl_price_pct / 100.0)
        tp = entry_price * (1.0 + tp_price_pct / 100.0)
    else:
        # For shorts: negative pct (SL) moves price UP, positive pct (TP) moves price DOWN
        sl = entry_price * (1.0 - sl_price_pct / 100.0)
        tp = entry_price * (1.0 - tp_price_pct / 100.0)
    return float(sl), float(tp)


def _fixed_pct_brackets(entry_price: float, side: str) -> Tuple[float, float]:
    """
    Return (SL, TP) prices based on fixed price percentages vs entry.
    Defaults: TP +12.5%, SL -5.0% (symmetric for shorts).
    Env overrides:
      FIXED_TP_PCT (default 12.5)
      FIXED_SL_PCT (default 5.0)
    """
    tp_pct = float(os.getenv("FIXED_TP_PCT", "12.5") or 12.5)
    sl_pct = float(os.getenv("FIXED_SL_PCT", "5.0") or 5.0)
    entry = float(entry_price)

    if side == "long":
        sl = entry * (1.0 - sl_pct / 100.0)
        tp = entry * (1.0 + tp_pct / 100.0)
    else:
        sl = entry * (1.0 + sl_pct / 100.0)
        tp = entry * (1.0 - tp_pct / 100.0)
    return float(sl), float(tp)


def _ladder_brackets(entry_price: float, side: str) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Return (SL, [(TP1_price, qty_pct), (TP2_price, qty_pct), ...]) for LADDER mode.
    TP levels and quantities are read from LADDER_TP_PCT and LADDER_TP_QTY_PCT.
    SL is read from LADDER_SL_PCT.
    All percentages are interpreted as ROE% and converted to price% using leverage.
    Returns: (sl_price, [(tp_price, portion_pct), ...])
    """
    tp_pcts_str = os.getenv("LADDER_TP_PCT", "7,14,21").strip()
    qty_pcts_str = os.getenv("LADDER_TP_QTY_PCT", "50,30,20").strip()
    sl_roe = float(os.getenv("LADDER_SL_PCT", "-7") or -7.0)

    # Get leverage to convert ROE% to price%
    lev = float(os.getenv("LEVERAGE", "20") or 20)

    tp_roes = [float(x.strip()) for x in tp_pcts_str.split(",") if x.strip()]
    qty_pcts = [float(x.strip()) for x in qty_pcts_str.split(",") if x.strip()]

    if len(tp_roes) != len(qty_pcts):
        tg(f"⚠️ LADDER_TP_PCT and LADDER_TP_QTY_PCT must have same length. Using defaults.")
        tp_roes = [7.0, 14.0, 21.0]
        qty_pcts = [50.0, 30.0, 20.0]

    if abs(sum(qty_pcts) - 100.0) > 0.1:
        tg(f"⚠️ LADDER_TP_QTY_PCT must sum to 100. Current sum: {sum(qty_pcts)}")

    entry = float(entry_price)

    # Convert ROE% to price%: price% = ROE% / leverage
    sl_price_pct = sl_roe / lev

    # Calculate SL price
    if side == "long":
        sl = entry * (1.0 + sl_price_pct / 100.0)
    else:
        sl = entry * (1.0 - sl_price_pct / 100.0)

    # Calculate TP prices (convert each ROE% to price%)
    tp_levels = []
    for tp_roe, qty_pct in zip(tp_roes, qty_pcts):
        tp_price_pct = tp_roe / lev
        if side == "long":
            tp_price = entry * (1.0 + tp_price_pct / 100.0)
        else:
            tp_price = entry * (1.0 - tp_price_pct / 100.0)
        tp_levels.append((float(tp_price), float(qty_pct)))

    return float(sl), tp_levels


# -------- Algo Order API helpers --------

def _create_stop_order(ex, symbol: str, side: str, qty: float, stop_price: float, order_type: str) -> Optional[str]:
    """
    Create TP/SL order using proven STOP_MARKET/TAKE_PROFIT_MARKET orders.
    These are fully queryable and work reliably with rearm.

    order_type: 'TAKE_PROFIT' or 'STOP'
    Returns orderId or None
    """
    try:
        order_label = "TAKE PROFIT" if order_type == "TAKE_PROFIT" else "STOP LOSS"

        # Order parameters
        order_params = {
            'stopPrice': float(stop_price),
            'reduceOnly': True,
            'workingType': 'MARK_PRICE',
        }

        # Determine order type
        order_type_str = 'STOP_MARKET' if order_type == 'STOP' else 'TAKE_PROFIT_MARKET'

        tg(f"📤 Creating {order_type_str}: {symbol} {side.upper()} qty={qty:.6f} @ {stop_price:.6f}")

        # Create order using CCXT
        order = ex.create_order(
            symbol=symbol,
            type=order_type_str.lower(),
            side=side.lower(),
            amount=qty,
            params=order_params
        )

        order_id = str(order.get('id') or order.get('orderId') or '')
        if not order_id:
            tg(f"❌ No order ID returned for {order_label}")
            return None

        # Verify order exists and wasn't rejected
        try:
            time.sleep(0.3)  # Small delay for order to register
            verification_order = ex.fetch_order(order_id, symbol)
            status = str(verification_order.get('status', '')).lower()

            if status == 'rejected':
                tg(f"❌ {order_label} was REJECTED by exchange: {order_id}")
                return None

            tg(f"✅ {order_label} placed & verified: {order_id} @ {stop_price:.6f}")
            return order_id

        except Exception as verify_error:
            # If we can't verify, assume it exists (it probably does)
            tg(f"⚠️ Could not verify {order_label} {order_id}: {verify_error}")
            tg(f"   Assuming order exists and will retry verification on next poll")
            return order_id

    except Exception as e:
        tg(f"❌ Failed to create {order_label}: {e}")
        import traceback
        tg(f"🔍 Traceback: {traceback.format_exc()}")
        return None


# -------- Order placement & polling --------

def _place_brackets(ex, decision: Decision, qty: float) -> Dict[str, Optional[str]]:
    """
    Place reduce-only TP/SL using proven STOP_MARKET/TAKE_PROFIT_MARKET orders (tick-safe).
    Returns order ids dict: {"tp": "...", "sl": "..."} (ids may be None on error).
    """
    reduce_side = "sell" if decision.side == "long" else "buy"
    ids: Dict[str, Optional[str]] = {"tp": None, "sl": None}

    tg(f"📋 Placing brackets for {decision.symbol}: Side={decision.side}, Qty={qty:.6f}")
    tg(f"   TP target: {decision.tp:.6f}, SL target: {decision.sl:.6f}")

    # TAKE PROFIT
    tp_px = _safe_stop_price(ex, decision.symbol, decision.side, float(decision.tp), "TP")
    ids["tp"] = _create_stop_order(ex, decision.symbol, reduce_side, qty, tp_px, "TAKE_PROFIT")

    # STOP LOSS
    sl_px = _safe_stop_price(ex, decision.symbol, decision.side, float(decision.sl), "SL")
    ids["sl"] = _create_stop_order(ex, decision.symbol, reduce_side, qty, sl_px, "STOP")

    # Verify orders were actually created
    if ids["tp"]:
        tg(f"✅ TP order confirmed: ID={ids['tp']}")
    else:
        tg(f"❌ TP order failed to create!")

    if ids["sl"]:
        tg(f"✅ SL order confirmed: ID={ids['sl']}")
    else:
        tg(f"❌ SL order failed to create!")

    # If both orders failed, warn loudly
    if not ids["tp"] and not ids["sl"]:
        tg(f"🚨 WARNING: Both TP and SL orders failed for {decision.symbol}! Position is UNPROTECTED!")

    return ids


def _place_ladder_brackets(ex, symbol: str, side: str, qty: float, sl_price: float, 
                           tp_levels: List[Tuple[float, float]]) -> Dict[str, Optional[str]]:
    """
    Place multiple reduce-only TP orders and one SL for LADDER mode using Algo Order API.
    tp_levels: [(tp_price, qty_pct), ...] where qty_pct is percentage of total position
    Special handling: qty_pct=100 for last level means "close ALL remaining" (prevents dust)
    Returns: {"tp1": "id1", "tp2": "id2", ..., "sl": "sl_id"}
    """
    reduce_side = "sell" if side == "long" else "buy"
    ids: Dict[str, Optional[str]] = {}

    # Store TP prices for later SL adjustments
    LADDER_TP_PRICES[symbol] = {}

    # Calculate actual remaining quantity for each TP level
    # If qty_pct is 100, it means "close remaining position" (not 100% of original)
    remaining_qty = qty
    adjusted_levels = []

    for i, (tp_price, qty_pct) in enumerate(tp_levels):
        is_last = (i == len(tp_levels) - 1)

        if qty_pct >= 100 or is_last:
            # Last TP or qty_pct=100: close ALL remaining to avoid dust
            tp_qty = remaining_qty
            actual_pct = (remaining_qty / qty) * 100  # For logging
            adjusted_levels.append((tp_price, tp_qty, actual_pct, True))  # True = "close remaining"
            remaining_qty = 0
            break
        else:
            # Normal TP: close specified % of ORIGINAL position
            tp_qty = qty * (qty_pct / 100.0)
            # Snap to lot size
            try:
                tp_qty = float(ex.amount_to_precision(symbol, tp_qty))
            except Exception:
                pass
            adjusted_levels.append((tp_price, tp_qty, qty_pct, False))
            remaining_qty -= tp_qty

    # Place multiple TP orders with calculated quantities
    for i, (tp_price, tp_qty, display_pct, is_final) in enumerate(adjusted_levels, start=1):
        if tp_qty <= 0:
            tg(f"⚠️ TP{i} qty too small, skipping")
            continue

        tp_px = _safe_stop_price(ex, symbol, side, float(tp_price), "TP")
        order_id = _create_stop_order(ex, symbol, reduce_side, tp_qty, tp_px, "TAKE_PROFIT")
        ids[f"tp{i}"] = order_id
        # Store the TP price for progressive SL management
        LADDER_TP_PRICES[symbol][f"tp{i}"] = tp_px
        if order_id:
            label = f"CLOSE ALL REMAINING" if is_final else f"{display_pct:.1f}%"
            tg(f"📊 TP{i} placed: {tp_qty:.6f} @ {tp_px:.6f} ({label})")

    # Place SL for full original position
    sl_px = _safe_stop_price(ex, symbol, side, float(sl_price), "SL")
    ids["sl"] = _create_stop_order(ex, symbol, reduce_side, qty, sl_px, "STOP")

    return ids


def _try_detect_exit_by_orders(ex, symbol: str) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Check if any TP or SL order is 'closed' by monitoring POSITION CHANGES.
    For LADDER mode: handle partial fills and update remaining qty + adjust SL qty.

    NOTE: Algo orders are invisible to queries but WILL execute. We detect fills
    by comparing expected vs actual position quantity, not by querying order status.

    Returns: (fully_exited, partial_qty_closed, exit_price) or (False, None, None)
    """

    ids = BRACKETS.get(symbol)
    if not ids:
        return False, None, None

    # SAFETY CHECK: Verify position still exists before checking orders
    # This catches edge cases where position closed but we haven't detected it yet
    memo = OPEN.get(symbol)
    debug = os.getenv("SCALP_DEBUG", "false").lower() in ("true", "1", "yes")

    if memo:
        try:
            positions = ex.fetch_positions([symbol])
            pos_exists = False
            for p in positions:
                amt = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)
                if _is_pos_open_amount(ex, symbol, amt):
                    pos_exists = True
                    break

            if not pos_exists:
                tg(f"✅ {symbol} Position closed (TP or SL hit)")
                # Position closed - cancel any remaining orders (silently)
                for key, order_id in list(ids.items()):
                    if order_id:
                        is_legacy = str(order_id).startswith("LEGACY_")
                        actual_id = str(order_id).replace("LEGACY_", "") if is_legacy else str(order_id)
                        try:
                            if is_legacy:
                                ex.cancel_order(actual_id, symbol)
                            else:
                                ex.request(
                                    path='algoOrder',
                                    api='fapiPrivate',
                                    method='DELETE',
                                    params={'symbol': symbol.replace('/', ''), 'algoId': actual_id}
                                )
                            if debug:
                                tg(f"🗑️ Cancelled orphaned {key.upper()} order")
                        except Exception:
                            # Order already filled/cancelled or invisible - ignore
                            pass
                # Get last price for PnL calculation
                px = float(ex.fetch_ticker(symbol)["last"])
                return True, None, px
        except Exception as e:
            if debug:
                tg(f"⚠️ Error during position safety check for {symbol}: {e}")

    # NEW: Check actual position quantity to detect partial fills
    memo = OPEN.get(symbol)
    if memo and symbol in LADDER_REMAINING_QTY:
        expected_qty = LADDER_REMAINING_QTY[symbol]
        try:
            positions = ex.fetch_positions([symbol])
            actual_qty = 0.0
            for p in positions:
                amt = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)
                if _is_pos_open_amount(ex, symbol, amt):
                    actual_qty = abs(amt)
                    break

            # If actual position is significantly less than expected, a TP was hit!
            qty_diff = expected_qty - actual_qty
            if qty_diff > 1e-6:  # Some quantity was closed
                tg(f"🎯 {symbol} Position decreased by {qty_diff:.6f} - TP was hit!")

                # Determine which TP was hit based on qty closed
                # Calculate which TP level this corresponds to
                qty_pcts_str = os.getenv("LADDER_TP_QTY_PCT", "33,33,100").strip()
                qty_pcts = [float(x.strip()) for x in qty_pcts_str.split(",") if x.strip()]

                original_qty = memo.qty
                pct_closed = (qty_diff / original_qty) * 100

                tg(f"📊 {symbol} Closed {pct_closed:.1f}% of position ({qty_diff:.6f} / {original_qty:.6f})")

                # Find which TP this corresponds to
                tp_num = None
                cumulative = 0
                for i, qty_pct in enumerate(qty_pcts, start=1):
                    cumulative += qty_pct
                    # Check if this matches the expected quantity for this TP
                    expected_tp_qty = original_qty * (qty_pct / 100.0)
                    if abs(qty_diff - expected_tp_qty) < expected_tp_qty * 0.1:  # Within 10% tolerance
                        tp_num = i
                        break

                if tp_num is None:
                    # If we can't determine exact TP, estimate based on cumulative %
                    cumulative = 0
                    for i, qty_pct in enumerate(qty_pcts, start=1):
                        if cumulative < pct_closed <= cumulative + qty_pct + 5:  # 5% tolerance
                            tp_num = i
                            break
                        cumulative += qty_pct

                if tp_num:
                    tg(f"🎯 Detected TP{tp_num} fill based on position quantity change")

                    # Update remaining qty
                    LADDER_REMAINING_QTY[symbol] = actual_qty
                    remaining = actual_qty

                    # Get current price for PnL calculation
                    px = float(ex.fetch_ticker(symbol)["last"])

                    # Calculate partial PnL
                    partial_pnl = (px - memo.entry_price) * qty_diff if memo.side == "long" else (memo.entry_price - px) * qty_diff
                    tg(f"🎯 <b>TP{tp_num} HIT (position-based)</b> {symbol}  |  Closed: {qty_diff:.6f} @ ~{px:.6f}  |  Partial PnL: {partial_pnl:+.2f} USDT")
                    tg(f"📊 Remaining position: {remaining:.6f}")

                    # Remove the filled TP from tracking
                    tp_key = f"tp{tp_num}"
                    if tp_key in ids:
                        ids.pop(tp_key, None)
                        tg(f"🗑️ Removed {tp_key} from BRACKETS")

                    # NOW TRIGGER SL ADJUSTMENT
                    tg(f"🎯 LADDER SL ADJUSTMENT TRIGGERED by TP{tp_num} position-based detection")

                    try:
                        sl_id = ids.get("sl")
                        if not sl_id:
                            tg(f"⚠️ {symbol} No SL order found to adjust")
                        elif remaining <= 1e-6:
                            tg(f"✅ {symbol} Position fully closed")
                            BRACKETS[symbol] = ids
                            return True, None, None
                        else:
                            reduce_side = "sell" if memo.side == "long" else "buy"

                            tg(f"📋 {symbol} Current state before SL adjustment:")
                            tg(f"   Entry: {memo.entry_price:.6f} | Side: {memo.side.upper()}")
                            tg(f"   Remaining qty: {remaining:.6f} | Old SL ID: {sl_id}")

                            # Determine new SL price based on TP level hit
                            if tp_num == 1:
                                sl_price = float(memo.entry_price)
                                sl_label = "BREAKEVEN (Entry Price)"
                                tg(f"🔄 TP1 filled → Moving SL to BREAKEVEN at {sl_price:.6f}")
                            elif tp_num == 2:
                                tp_prices = LADDER_TP_PRICES.get(symbol, {})
                                sl_price = tp_prices.get("tp1", memo.entry_price)
                                sl_label = f"TP1 Price ({sl_price:.6f})"
                                tg(f"🔄 TP2 filled → Moving SL to TP1 price at {sl_price:.6f}")
                            else:
                                tp_prices = LADDER_TP_PRICES.get(symbol, {})
                                prev_tp_key = f"tp{tp_num - 1}"
                                sl_price = tp_prices.get(prev_tp_key, memo.entry_price)
                                sl_label = f"{prev_tp_key.upper()} Price ({sl_price:.6f})"
                                tg(f"🔄 TP{tp_num} filled → Moving SL to {prev_tp_key.upper()} price at {sl_price:.6f}")

                            # Cancel old SL
                            tg(f"🗑️ Cancelling old SL order (ID: {sl_id})...")
                            cancelled_successfully = False
                            try:
                                ex.request(
                                    path='algoOrder',
                                    api='fapiPrivate',
                                    method='DELETE',
                                    params={'symbol': symbol.replace('/', ''), 'algoId': str(sl_id)}
                                )
                                cancelled_successfully = True
                                tg(f"✅ Old SL cancelled via Algo API")
                            except Exception as e1:
                                tg(f"⚠️ Algo API cancel failed: {e1}, trying legacy...")
                                try:
                                    ex.cancel_order(sl_id, symbol)
                                    cancelled_successfully = True
                                    tg(f"✅ Old SL cancelled via legacy API")
                                except Exception as e2:
                                    tg(f"❌ Legacy cancel also failed: {e2}")

                            if not cancelled_successfully:
                                tg(f"⚠️ Could not cancel old SL, but will try placing new one anyway")

                            # Place new SL with reduced qty at new price
                            tg(f"📍 Placing new SL: qty={remaining:.6f}, price={sl_price:.6f}, label={sl_label}")
                            sl_px_safe = _safe_stop_price(ex, symbol, memo.side, sl_price, "SL")
                            tg(f"   Tick-safe price: {sl_px_safe:.6f}")

                            new_sl_id = _create_stop_order(ex, symbol, reduce_side, remaining, sl_px_safe, "STOP")
                            if new_sl_id:
                                ids["sl"] = new_sl_id
                                BRACKETS[symbol] = ids
                                tg(f"🔄 Updated BRACKETS dict with new SL ID: {new_sl_id}")
                                tg(
                                    f"✅ <b>LADDER SL MOVED</b>\n"
                                    f"   {symbol} | TP{tp_num} filled → SL → {sl_label}\n"
                                    f"   New SL: {sl_px_safe:.6f} | Qty: {remaining:.6f} | ID: {new_sl_id}"
                                )
                            else:
                                tg(f"❌ Failed to place new SL order")
                    except Exception as e:
                        tg(f"❌ LADDER SL adjustment failed for {symbol} after TP{tp_num}: {e}")
                        import traceback
                        tg(f"🔍 Traceback: {traceback.format_exc()}")

                    # Update global BRACKETS
                    BRACKETS[symbol] = ids
                    tg(f"📋 BRACKETS updated after position-based detection: {list(ids.keys())}")
                    return False, qty_diff, px  # Partial exit

        except Exception as e:
            tg(f"⚠️ Error checking position quantity for {symbol}: {e}")

    def is_closed(oid: Optional[str]) -> Tuple[bool, Optional[float], Optional[float]]:
        """Returns (is_closed, filled_qty, avg_price)"""
        if not oid:
            return False, None, None

        # All orders are now standard STOP_MARKET/TAKE_PROFIT_MARKET - fully queryable
        try:
            o = ex.fetch_order(oid, symbol)
            status = str(o.get("status", "")).lower()
            order_type = str(o.get("type", "")).upper()

            if status in ("closed", "filled"):
                filled = float(o.get("filled", 0) or 0)
                avg_px = float(o.get("average") or o.get("price") or 0)
                tg(f"📊 Order {oid} filled: type={order_type}, qty={filled:.6f}, price={avg_px:.6f}")
                return True, filled, avg_px
            elif status in ("cancelled", "expired"):
                return False, None, None
            else:
                return False, None, None
        except Exception as e:
            # Order doesn't exist - might be filled and already cleared
            # Don't spam logs, this is checked frequently
            return False, None, None

    # Check SL - if hit, full exit
    sl_closed, sl_qty, sl_px = is_closed(ids.get("sl"))
    if sl_closed:
        tg(f"🛑 <b>STOP LOSS HIT</b> for {symbol} @ {sl_px:.6f} (qty: {sl_qty:.6f})")

        # CRITICAL FIX: Cancel all remaining TP orders when SL is hit
        # This prevents orphaned TP orders (especially TP3 at 100%) from opening unwanted positions
        tp_cancel_count = 0
        for key in list(ids.keys()):
            if not key.startswith("tp"):
                continue
            tp_id = ids.get(key)
            if tp_id:
                tg(f"🔄 Attempting to cancel {key.upper()} order (ID: {tp_id})...")
                cancelled = False
                try:
                    ex.request(
                        path='algoOrder',
                        api='fapiPrivate',
                        method='DELETE',
                        params={'symbol': symbol.replace('/', ''), 'algoId': str(tp_id)}
                    )
                    cancelled = True
                    tp_cancel_count += 1
                    tg(f"✅ Cancelled {key.upper()} via Algo API")
                except Exception as e1:
                    tg(f"⚠️ Algo API cancel failed for {key.upper()}: {e1}")
                    try:
                        ex.cancel_order(tp_id, symbol)
                        cancelled = True
                        tp_cancel_count += 1
                        tg(f"✅ Cancelled {key.upper()} via legacy API")
                    except Exception as e2:
                        tg(f"❌ Failed to cancel {key.upper()} via both APIs: {e2}")

                if cancelled:
                    # Remove from tracking to prevent re-attempts
                    ids.pop(key, None)

        if tp_cancel_count > 0:
            tg(f"🗑️ Successfully cancelled {tp_cancel_count} TP order(s) after SL hit")
        else:
            tg(f"⚠️ No TP orders found to cancel (may have been cancelled already)")

        # Update BRACKETS dict to reflect cancelled TPs
        BRACKETS[symbol] = ids
        tg(f"✅ SL exit confirmed for {symbol} - all TPs cancelled, position closed")
        return True, sl_qty, sl_px

    # Check TP orders (could be tp1, tp2, etc. for LADDER mode or just tp)
    has_ladder = any(k.startswith("tp") and k != "tp" for k in ids.keys())

    if has_ladder:
        # LADDER mode: check each TP level
        for key in list(ids.keys()):
            if not key.startswith("tp"):
                continue
            closed, qty, px = is_closed(ids.get(key))
            if closed:
                # Get memo for PnL calculation
                memo = OPEN.get(symbol)

                # Calculate PnL for this partial exit
                if memo:
                    partial_pnl = (px - memo.entry_price) * qty if memo.side == "long" else (memo.entry_price - px) * qty
                    tg(f"🎯 <b>{key.upper()} HIT</b> {symbol}  |  Closed: {qty:.6f} @ {px:.6f}  |  Partial PnL: {partial_pnl:+.2f} USDT")

                # Partial TP hit - update remaining qty and adjust SL progressively
                if symbol in LADDER_REMAINING_QTY:
                    LADDER_REMAINING_QTY[symbol] -= qty
                    remaining = LADDER_REMAINING_QTY[symbol]

                    # Report remaining qty
                    tg(f"📊 Remaining position: {remaining:.6f} (closed {(qty/(qty+remaining)*100):.1f}% at this level)")

                    # Remove this TP from tracking
                    ids.pop(key, None)

                    # Determine which TP was hit and adjust SL accordingly
                    tp_num = int(key.replace("tp", ""))

                    tg(f"🎯 LADDER SL ADJUSTMENT TRIGGERED by {key.upper()} fill")

                    try:
                        sl_id = ids.get("sl")
                        if not sl_id:
                            tg(f"⚠️ {symbol} No SL order found to adjust")
                        elif remaining <= 0:
                            tg(f"⚠️ {symbol} No remaining position to protect")
                        else:
                            memo = OPEN.get(symbol)
                            if not memo:
                                tg(f"⚠️ {symbol} No position memo found")
                            else:
                                reduce_side = "sell" if memo.side == "long" else "buy"

                                tg(f"📋 {symbol} Current state before SL adjustment:")
                                tg(f"   Entry: {memo.entry_price:.6f} | Side: {memo.side.upper()}")
                                tg(f"   Remaining qty: {remaining:.6f} | Old SL ID: {sl_id}")

                                # Determine new SL price based on TP level hit
                                if tp_num == 1:
                                    # TP1 hit: Move SL to breakeven (entry price)
                                    sl_price = float(memo.entry_price)
                                    sl_label = "BREAKEVEN (Entry Price)"
                                    tg(f"🔄 TP1 filled → Moving SL to BREAKEVEN at {sl_price:.6f}")
                                elif tp_num == 2:
                                    # TP2 hit: Move SL to TP1 price
                                    tp_prices = LADDER_TP_PRICES.get(symbol, {})
                                    sl_price = tp_prices.get("tp1", memo.entry_price)
                                    sl_label = f"TP1 Price ({sl_price:.6f})"
                                    tg(f"🔄 TP2 filled → Moving SL to TP1 price at {sl_price:.6f}")
                                else:
                                    # TP3+ hit: Move SL to previous TP price
                                    tp_prices = LADDER_TP_PRICES.get(symbol, {})
                                    prev_tp_key = f"tp{tp_num - 1}"
                                    sl_price = tp_prices.get(prev_tp_key, memo.entry_price)
                                    sl_label = f"{prev_tp_key.upper()} Price ({sl_price:.6f})"
                                    tg(f"🔄 TP{tp_num} filled → Moving SL to {prev_tp_key.upper()} price at {sl_price:.6f}")

                                # Cancel old SL
                                tg(f"🗑️ Cancelling old SL order (ID: {sl_id})...")
                                cancelled_successfully = False
                                try:
                                    ex.request(
                                        path='algoOrder',
                                        api='fapiPrivate',
                                        method='DELETE',
                                        params={'symbol': symbol.replace('/', ''), 'algoId': str(sl_id)}
                                    )
                                    cancelled_successfully = True
                                    tg(f"✅ Old SL cancelled via Algo API")
                                except Exception as e1:
                                    tg(f"⚠️ Algo API cancel failed: {e1}, trying legacy...")
                                    try:
                                        ex.cancel_order(sl_id, symbol)
                                        cancelled_successfully = True
                                        tg(f"✅ Old SL cancelled via legacy API")
                                    except Exception as e2:
                                        tg(f"❌ Legacy cancel also failed: {e2}")

                                if not cancelled_successfully:
                                    tg(f"⚠️ Could not cancel old SL, but will try placing new one anyway")

                                # Place new SL with reduced qty at new price
                                tg(f"📍 Placing new SL: qty={remaining:.6f}, price={sl_price:.6f}, label={sl_label}")
                                sl_px_safe = _safe_stop_price(ex, symbol, memo.side, sl_price, "SL")
                                tg(f"   Tick-safe price: {sl_px_safe:.6f}")

                                new_sl_id = _create_algo_order(ex, symbol, reduce_side, remaining, sl_px_safe, "STOP")
                                if new_sl_id:
                                    ids["sl"] = new_sl_id
                                    # *** CRITICAL FIX: Update global BRACKETS dict immediately ***
                                    BRACKETS[symbol] = ids
                                    tg(f"🔄 Updated BRACKETS dict with new SL ID: {new_sl_id}")
                                    tg(
                                        f"✅ <b>LADDER SL MOVED</b>\n"
                                        f"   {symbol} | {key.upper()} filled → SL → {sl_label}\n"
                                        f"   New SL: {sl_px_safe:.6f} | Qty: {remaining:.6f} | ID: {new_sl_id}"
                                    )
                                else:
                                    tg(f"❌ Failed to place new SL order")
                    except Exception as e:
                        tg(f"❌ LADDER SL adjustment failed for {symbol} after {key.upper()}: {e}")
                        import traceback
                        tg(f"🔍 Traceback: {traceback.format_exc()}")

                    # Check if all TPs are filled (full exit)
                    if remaining < 1e-6 or not any(k.startswith("tp") for k in ids.keys()):
                        tg(f"✅ All TPs hit for {symbol} - position fully closed")
                        # Update global BRACKETS before returning
                        BRACKETS[symbol] = ids
                        return True, None, None

                    # *** CRITICAL FIX: Update global BRACKETS before returning ***
                    BRACKETS[symbol] = ids
                    tg(f"📋 BRACKETS updated: {list(ids.keys())}")
                    return False, qty, px  # Partial exit, continue monitoring

        return False, None, None
    else:
        # Single TP mode
        tp_closed, tp_qty, tp_px = is_closed(ids.get("tp"))
        if tp_closed:
            return True, tp_qty, tp_px
        return False, None, None


def _cancel_open_brackets(ex, symbol: str):
    """Cancel any open TP/SL reduce-only orders for a symbol using standard cancel_order."""

    # Cancel all known orders via standard API
    ids = BRACKETS.get(symbol, {})
    cancelled_count = 0
    for key, order_id in ids.items():
        if not order_id or str(order_id).startswith("MANUAL"):
            continue

        try:
            ex.cancel_order(order_id, symbol)
            tg(f"🔁 Cancelled {key.upper()} order {order_id} for {symbol}")
            cancelled_count += 1
        except Exception:
            # Already filled or cancelled - ignore
            pass

    # 2. Cleanup any stray reduce-only orders (belt and suspenders)
    try:
        open_orders = ex.fetch_open_orders(symbol)
        for oo in open_orders:
            typ = str(oo.get("type", "")).upper()
            reduce_only = bool(oo.get("info", {}).get("reduceOnly")) or bool(oo.get("reduceOnly"))
            if reduce_only and typ in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT", "STOP_LOSS"):
                try:
                    ex.cancel_order(oo["id"], symbol)
                    tg(f"🧹 Cleaned up stray order {oo['id']} for {symbol}")
                    cancelled_count += 1
                except Exception:
                    pass
    except Exception:
        pass

    if cancelled_count > 0:
        tg(f"✅ Cancelled {cancelled_count} bracket order(s) for {symbol}")

    BRACKETS.pop(symbol, None)


# ========== SCALPER DYNAMIC EXITS ==========

def _is_scalper_position(sym: str) -> bool:
    """Check if position is from scalper strategy."""
    memo = OPEN.get(sym)
    if not memo:
        return False
    # Check if reason contains scalper strategy indicator
    reason = getattr(memo, 'reason', {})
    if isinstance(reason, dict):
        return reason.get('strategy') == 'MultiConfluenceScalper'
    return False


def maybe_scalper_trailing_stop(ex, sym: str, memo: PositionMemo) -> bool:
    """
    Scalper trailing stop: After profit threshold, trail SL by ATR.
    Returns True if SL was updated.
    """
    if not _is_scalper_position(sym):
        return False

    # Check if trailing is enabled
    if not _env_bool("SCALP_ENABLE_TRAILING", True):
        return False

    # Get current price
    try:
        cur_price = _current_price(ex, sym)
    except Exception:
        return False

    # Calculate current profit %
    if memo.side == "long":
        profit_pct = (cur_price - memo.entry_price) / memo.entry_price * 100
    else:
        profit_pct = (memo.entry_price - cur_price) / memo.entry_price * 100

    # Check if we've hit profit threshold to start trailing
    trail_threshold = float(os.getenv("SCALP_TRAIL_AFTER_PROFIT_PCT", "0.5") or 0.5)
    if profit_pct < trail_threshold:
        return False

    # Track highest profit seen
    if sym not in SCALPER_HIGHEST_PROFIT:
        SCALPER_HIGHEST_PROFIT[sym] = profit_pct
    else:
        SCALPER_HIGHEST_PROFIT[sym] = max(SCALPER_HIGHEST_PROFIT[sym], profit_pct)

    # Get 1m ATR for trailing distance
    try:
        ohlcv = fetch_candles(ex, sym, timeframe="1m", limit=30)
        arr = np.asarray(ohlcv, dtype=float)
        _, _, h, l, c, _ = arr.T

        # Calculate ATR
        tr = np.maximum(h[1:] - l[1:], 
                        np.maximum(np.abs(h[1:] - c[:-1]), 
                                  np.abs(l[1:] - c[:-1])))
        atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

        # Trail distance = ATR * multiplier
        trail_mult = float(os.getenv("SCALP_TRAIL_ATR_MULT", "0.5") or 0.5)
        trail_distance = atr_val * trail_mult

        # Calculate new SL
        if memo.side == "long":
            new_sl = cur_price - trail_distance
            # Only move SL up, never down
            current_sl = BRACKETS.get(sym, {}).get("sl")
            if current_sl:
                try:
                    # Fetch current SL price from order
                    order = ex.fetch_order(current_sl, sym)
                    current_sl_price = float(order.get("stopPrice") or order.get("triggerPrice") or 0)
                    if new_sl <= current_sl_price:
                        return False  # Don't move SL down
                except Exception:
                    pass

            # Ensure SL is below current price
            if new_sl >= cur_price * 0.999:
                new_sl = cur_price * 0.999
        else:
            new_sl = cur_price + trail_distance
            # Only move SL down, never up
            current_sl = BRACKETS.get(sym, {}).get("sl")
            if current_sl:
                try:
                    order = ex.fetch_order(current_sl, sym)
                    current_sl_price = float(order.get("stopPrice") or order.get("triggerPrice") or 0)
                    if new_sl >= current_sl_price:
                        return False  # Don't move SL up
                except Exception:
                    pass

            # Ensure SL is above current price
            if new_sl <= cur_price * 1.001:
                new_sl = cur_price * 1.001

        # Update SL (cancel old, place new)
        remaining_qty = LADDER_REMAINING_QTY.get(sym, memo.qty)
        new_ids = _rearm_brackets_abs(ex, sym, memo.side, remaining_qty, new_sl, None)

        if new_ids.get("sl"):
            old_brackets = BRACKETS.get(sym, {})
            old_brackets["sl"] = new_ids["sl"]
            BRACKETS[sym] = old_brackets
            tg(f"📈 Scalper trailing SL updated for {sym}: ${new_sl:.2f} (profit: {profit_pct:.2f}%)")
            return True

    except Exception as e:
        tg(f"⚠️ Scalper trailing stop error for {sym}: {e}")

    return False


def maybe_scalper_momentum_exit(ex, sym: str, memo: PositionMemo) -> bool:
    """
    Scalper momentum exit: Exit if consecutive candles close against position.
    Returns True if position should be closed.
    """
    if not _is_scalper_position(sym):
        return False

    # Get threshold
    threshold = int(os.getenv("SCALP_MOMENTUM_EXIT_CANDLES", "2") or 2)
    debug = os.getenv("SCALP_DEBUG", "false").lower() in ("true", "1", "yes")

    try:
        # Fetch recent 1m candles (need current + threshold closed candles)
        ohlcv = fetch_candles(ex, sym, timeframe="1m", limit=threshold + 5)
        arr = np.asarray(ohlcv, dtype=float)
        _, o, _, _, c, _ = arr.T

        if len(c) < threshold + 2:
            if debug:
                tg(f"📊 {sym} Momentum: Not enough candles (have {len(c)}, need {threshold + 2})")
            return False

        # Check last N CLOSED candles (skip -1 which is forming)
        # Use indices -2, -3, -4, etc. for closed candles
        # We want CONSECUTIVE candles from most recent backward
        against_count = 0
        candle_details = []

        for i in range(threshold):
            idx = -(i + 2)  # Start from -2 (last closed), then -3, -4, etc.
            if abs(idx) > len(c):
                break

            candle_open = float(o[idx])
            candle_close = float(c[idx])
            candle_body_pct = abs(candle_close - candle_open) / candle_open * 100

            if memo.side == "long":
                # For longs, bearish candles (close < open) are against us
                is_against = candle_close < candle_open
                candle_type = "🔴 BEAR" if is_against else "🟢 BULL"
            else:
                # For shorts, bullish candles (close > open) are against us
                is_against = candle_close > candle_open
                candle_type = "🟢 BULL" if is_against else "🔴 BEAR"

            candle_details.append(f"{candle_type} ({candle_body_pct:.2f}%)")

            if is_against:
                against_count += 1
            else:
                # Found a favorable candle - momentum broken, stop counting
                if debug:
                    tg(f"📊 {sym} Momentum broken at candle {i+1}: found favorable {candle_type}")
                break

        SCALPER_CONSECUTIVE_AGAINST[sym] = against_count

        # Always log momentum status in debug mode
        if debug:
            tg(f"📊 {sym} Momentum: {against_count}/{threshold} consecutive against | {memo.side.upper()} | Recent: {' ← '.join(candle_details)}")

        if against_count >= threshold:
            tg(
                f"🛑 <b>MOMENTUM EXIT TRIGGERED</b> {sym}\n"
                f"Position: {memo.side.upper()} | Consecutive against: {against_count}/{threshold}\n"
                f"Recent candles (newest→oldest): {' ← '.join(candle_details[:threshold])}\n"
                f"🔪 Cutting loss to preserve capital"
            )
            return True

    except Exception as e:
        tg(f"⚠️ Scalper momentum check error for {sym}: {e}")
        if debug:
            import traceback
            tg(f"🔍 Traceback: {traceback.format_exc()}")

    return False


def maybe_scalper_time_exit(ex, sym: str, memo: PositionMemo) -> bool:
    """
    Scalper time exit: Exit if position open > X minutes with minimal profit.
    Returns True if position should be closed.
    """
    if not _is_scalper_position(sym):
        return False

    # Get time limit
    time_limit_min = int(os.getenv("SCALP_TIME_EXIT_MINUTES", "15") or 15)
    time_limit_sec = time_limit_min * 60

    # Check entry time
    entry_time = SCALPER_ENTRY_TIME.get(sym, memo.opened_at)
    time_in_trade = time.time() - entry_time

    if time_in_trade < time_limit_sec:
        return False

    # Check current profit
    try:
        cur_price = _current_price(ex, sym)
        if memo.side == "long":
            profit_pct = (cur_price - memo.entry_price) / memo.entry_price * 100
        else:
            profit_pct = (memo.entry_price - cur_price) / memo.entry_price * 100

        # Only exit if profit is minimal (< 0.3%)
        if profit_pct < 0.3:
            tg(f"⏰ Scalper time exit triggered for {sym}: {time_in_trade/60:.1f} min in trade with only {profit_pct:.2f}% profit")
            return True

    except Exception:
        pass

    return False


def maybe_scalper_breakeven(ex, sym: str, memo: PositionMemo) -> bool:
    """
    Scalper breakeven: Move SL to breakeven after profit threshold.
    Returns True if SL was updated to breakeven.
    """
    if not _is_scalper_position(sym):
        return False

    # Get breakeven threshold
    be_threshold = float(os.getenv("SCALP_BREAKEVEN_AT_PCT", "0.5") or 0.5)

    try:
        cur_price = _current_price(ex, sym)

        # Calculate current profit
        if memo.side == "long":
            profit_pct = (cur_price - memo.entry_price) / memo.entry_price * 100
        else:
            profit_pct = (memo.entry_price - cur_price) / memo.entry_price * 100

        if profit_pct < be_threshold:
            return False

        # Check if SL is already at or better than breakeven
        current_sl_brackets = BRACKETS.get(sym, {})
        if current_sl_brackets.get("sl"):
            try:
                order = ex.fetch_order(current_sl_brackets["sl"], sym)
                current_sl_price = float(order.get("stopPrice") or order.get("triggerPrice") or 0)

                # Check if already at breakeven
                if memo.side == "long" and current_sl_price >= memo.entry_price * 0.999:
                    return False  # Already at breakeven
                if memo.side == "short" and current_sl_price <= memo.entry_price * 1.001:
                    return False
            except Exception:
                pass

        # Move SL to breakeven
        new_sl = memo.entry_price
        remaining_qty = LADDER_REMAINING_QTY.get(sym, memo.qty)

        new_ids = _rearm_brackets_abs(ex, sym, memo.side, remaining_qty, new_sl, None)

        if new_ids.get("sl"):
            old_brackets = BRACKETS.get(sym, {})
            old_brackets["sl"] = new_ids["sl"]
            BRACKETS[sym] = old_brackets
            tg(f"🔒 Scalper breakeven SL set for {sym} @ ${new_sl:.2f} (profit: {profit_pct:.2f}%)")
            return True

    except Exception as e:
        tg(f"⚠️ Scalper breakeven error for {sym}: {e}")

    return False

def is_cooldown_active(symbol: str) -> Tuple[bool, str]:
    """
    Check if symbol is in cooldown period.
    Returns: (is_active, reason_message)
    """
    now = time.time()

    # Get cooldown settings
    post_trade_cooldown = int(os.getenv("SCALP_POST_TRADE_COOLDOWN_SEC", "900") or 900)
    min_loss_cooldown = int(os.getenv("SCALP_MIN_LOSS_COOLDOWN_SEC", "1800") or 1800)

    # Check if we have a recent trade
    last_exit = LAST_TRADE_EXIT_TIME.get(symbol, 0)
    if last_exit == 0:
        return False, ""

    time_since_exit = now - last_exit

    # General cooldown (always applies)
    if time_since_exit < post_trade_cooldown:
        remaining = int(post_trade_cooldown - time_since_exit)
        return True, f"General cooldown: {int(time_since_exit)}s / {post_trade_cooldown}s ({remaining}s remaining)"

    # Loss-specific cooldown (additional, only after losses)
    if LAST_TRADE_WAS_LOSS.get(symbol, False):
        if time_since_exit < min_loss_cooldown:
            remaining = int(min_loss_cooldown - time_since_exit)
            return True, f"LOSS cooldown: {int(time_since_exit)}s / {min_loss_cooldown}s ({remaining}s remaining)"

    return False, ""



def execute(ex, decision: Decision):
    """Places entry + brackets, sends Telegram logs."""
    _reset_daily_if_needed()

    # Check cooldown FIRST
    cooldown_active, cooldown_msg = is_cooldown_active(decision.symbol)
    if cooldown_active:
        debug = os.getenv("SCALP_DEBUG", "false").lower() in ("true", "1", "yes")
        if debug:
            tg(f"⏸️ {decision.symbol} trade blocked: {cooldown_msg}")
        return

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

        # Override brackets by mode (FIXED_PCT | ROE | LADDER | ATR[default uses decision values])
        mode = (os.getenv("BRACKET_MODE", "ATR") or "ATR").upper()
        is_ladder_mode = False
        ladder_sl = None
        ladder_tp_levels = None

        if mode == "FIXED_PCT":
            sl_px, tp_px = _fixed_pct_brackets(entry_price, decision.side)
            decision.sl, decision.tp = sl_px, tp_px
        elif mode == "ROE":
            sl_px, tp_px = _roe_brackets(entry_price, decision.side)
            decision.sl, decision.tp = sl_px, tp_px
        elif mode == "LADDER":
            is_ladder_mode = True
            ladder_sl, ladder_tp_levels = _ladder_brackets(entry_price, decision.side)

            decision.sl = ladder_sl  # For logging purposes

        # Estimate taker fee rate (prefer exchange market info; fallback to env or default 0.0005)
        try:
            mkt = ex.market(decision.symbol)
            fee_rate = float(mkt.get("taker") or os.getenv("FEE_TAKER_PCT") or 0.0005)
        except Exception:
            fee_rate = float(os.getenv("FEE_TAKER_PCT") or 0.0005)
        try:
            fee_rate = float(fee_rate)
        except Exception:
            fee_rate = 0.0005
        entry_fee = abs(entry_price * qty) * float(fee_rate)

        OPEN[decision.symbol] = PositionMemo(
            symbol=decision.symbol, side=decision.side, qty=qty,
            entry_price=entry_price, opened_at=time.time(),
            taker_fee_rate=float(fee_rate), entry_fee=float(entry_fee)
        )
        # init dynamic ladder trackers
        PROGRESS_STAGE[decision.symbol] = -1
        LAST_REARM_AT[decision.symbol] = 0.0
        LAST_ROI_TELL[decision.symbol] = 0.0

        # Init scalper trackers if this is a scalper trade
        if isinstance(decision.reason, dict) and decision.reason.get('strategy') == 'MultiConfluenceScalper':
            SCALPER_ENTRY_TIME[decision.symbol] = time.time()
            SCALPER_HIGHEST_PROFIT[decision.symbol] = 0.0
            SCALPER_CONSECUTIVE_AGAINST[decision.symbol] = 0

        if is_ladder_mode:
            LADDER_REMAINING_QTY[decision.symbol] = qty
            tp_info = ", ".join([f"TP{i+1}={tp_price:.6f}({qty_pct}%)" 
                                for i, (tp_price, qty_pct) in enumerate(ladder_tp_levels)])
            tg(
                f"🚀 <b>ENTRY</b> {decision.symbol} {decision.side.upper()} qty={qty:.6f}\n"
                f"SL={ladder_sl:.6f}  {tp_info}\n"
                f"conf={decision.confidence:.2f}  reason={decision.reason}"
            )
        else:
            tg(
                f"🚀 <b>ENTRY</b> {decision.symbol} {decision.side.upper()} qty={qty:.6f}\n"
                f"SL={decision.sl:.6f}  TP={decision.tp:.6f}\n"
                f"conf={decision.confidence:.2f}  reason={decision.reason}"
            )
    except Exception as e:
        tg(f"❌ Entry failed {decision.symbol}: {e}")
        return

    # ---- BRACKETS ----
    if is_ladder_mode:
        # LADDER mode: Place multiple TPs + 1 SL (works for all patterns including FVG)
        ids = _place_ladder_brackets(ex, decision.symbol, decision.side, qty, ladder_sl, ladder_tp_levels)
    else:
        # Single TP/SL mode (ATR, FIXED_PCT, ROE when not LADDER)
        ids = _place_brackets(ex, decision, qty)
    BRACKETS[decision.symbol] = ids


def poll_positions_and_report(ex):
    """
    Every ~10s: detect exit (by position OR by bracket order fill),
    adjust dynamic TP/SL while open, send PnL TG, cancel sibling, clear state.
    Handles LADDER mode with partial exits.
    """
    global DAILY_PNL

    for sym, memo in list(OPEN.items()):
        try:
            # Check if this is a scalper position and apply scalper-specific exits
            if _is_scalper_position(sym):
                # 1. Check momentum exit (highest priority - cut losers fast)
                if maybe_scalper_momentum_exit(ex, sym, memo):
                    # Close position immediately with market order
                    try:
                        qty = LADDER_REMAINING_QTY.get(sym, memo.qty)
                        side = "sell" if memo.side == "long" else "buy"

                        tg(f"🚨 EXECUTING MOMENTUM EXIT for {sym}: Placing market order {side.upper()} {qty:.6f}")

                        # Cancel all TP/SL orders first to avoid conflicts
                        try:
                            _cancel_open_brackets(ex, sym)
                            tg(f"✅ Cancelled all brackets before momentum exit")
                        except Exception as cancel_err:
                            tg(f"⚠️ Error cancelling brackets (will continue anyway): {cancel_err}")

                        # Place market order to close position
                        order = ex.create_order(sym, "market", side, qty, params={"reduceOnly": True})
                        fill_price = float(order.get("price") or order.get("average") or ex.fetch_ticker(sym)["last"])

                        tg(f"✅ Momentum exit market order FILLED: {sym} {side.upper()} {qty:.6f} @ ${fill_price:.2f}")

                        # Calculate PnL for this forced exit
                        gross_pnl = (fill_price - memo.entry_price) * qty if memo.side == "long" else (memo.entry_price - fill_price) * qty
                        tg(f"📊 Momentum exit PnL: {gross_pnl:+.2f} USDT (before fees)")

                        # Position will be cleaned up in next poll iteration

                    except Exception as e:
                        tg(f"❌ CRITICAL: Failed to execute momentum exit for {sym}: {e}")
                        import traceback
                        tg(f"🔍 Traceback: {traceback.format_exc()}")

                        # Try emergency fallback: cancel all orders and try again
                        try:
                            tg(f"🆘 Attempting emergency position close...")
                            _cancel_open_brackets(ex, sym)
                            time.sleep(0.5)
                            emergency_order = ex.create_order(sym, "market", side, qty)
                            tg(f"✅ Emergency close successful")
                        except Exception as emergency_err:
                            tg(f"❌ Emergency close also failed: {emergency_err}")

                # 2. Check time exit (if momentum didn't trigger)
                elif maybe_scalper_time_exit(ex, sym, memo):
                    # Close position at market
                    try:
                        qty = LADDER_REMAINING_QTY.get(sym, memo.qty)
                        side = "sell" if memo.side == "long" else "buy"
                        ex.create_order(sym, "market", side, qty, params={"reduceOnly": True})
                        tg(f"⏰ Scalper time exit executed for {sym}")
                    except Exception as e:
                        tg(f"⚠️ Failed to execute time exit for {sym}: {e}")

                # 3. Try breakeven management
                else:
                    try:
                        maybe_scalper_breakeven(ex, sym, memo)
                    except Exception as e_be:
                        tg(f"⚠️ scalper breakeven error {sym}: {e_be}")

                    # 4. Try trailing stop
                    try:
                        maybe_scalper_trailing_stop(ex, sym, memo)
                    except Exception as e_trail:
                        tg(f"⚠️ scalper trailing error {sym}: {e_trail}")

            # Standard dynamic re-arm (for non-scalper positions or as fallback)
            else:
                try:
                    _maybe_dynamic_rearm(ex, sym, memo)
                except Exception as e_dyn:
                    tg(f"⚠️ dynamic re-arm error {sym}: {e_dyn}")

            # 1) Fast path: did one of our bracket orders fill?
            exited_by_order, partial_qty, partial_px = _try_detect_exit_by_orders(ex, sym)

            # Log when order fills are detected
            if exited_by_order:
                if partial_qty and partial_px:
                    tg(f"⚡ TP/SL order filled for {sym}: qty={partial_qty:.6f}, price={partial_px:.6f}")
                else:
                    tg(f"⚡ TP/SL order filled for {sym}")

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

                # For ladder mode, use actual remaining qty
                exit_qty = LADDER_REMAINING_QTY.get(sym, memo.qty)

                gross_pnl = (px - memo.entry_price) * memo.qty if memo.side == "long" \
                    else (memo.entry_price - px) * memo.qty

                # Estimate taker fees on entry and exit (market orders); use memo rate or fallback
                try:
                    fee_rate = float(memo.taker_fee_rate)
                    if fee_rate <= 0:
                        raise ValueError("no memo fee_rate")
                except Exception:
                    try:
                        mkt = ex.market(sym)
                        fee_rate = float(mkt.get("taker") or os.getenv("FEE_TAKER_PCT") or 0.0005)
                    except Exception:
                        fee_rate = float(os.getenv("FEE_TAKER_PCT") or 0.0005)
                # entry fee stored in memo; exit fee based on current price and qty
                entry_fee = float(getattr(memo, "entry_fee", 0.0) or 0.0)
                exit_fee = abs(px * memo.qty) * float(fee_rate)
                net_pnl = gross_pnl - entry_fee - exit_fee

                DAILY_PNL += net_pnl
                hold = int(time.time() - memo.opened_at)

                # Update cooldown tracking
                LAST_TRADE_EXIT_TIME[sym] = time.time()
                LAST_TRADE_WAS_LOSS[sym] = net_pnl < 0

                loss_emoji = "❌" if net_pnl < 0 else "✅"
                cooldown_type = "LOSS" if net_pnl < 0 else "WIN"

                tg(
                    f"{loss_emoji} <b>EXIT</b> {sym} {memo.side.upper()}  Net PnL: {net_pnl:+.2f}  (Gross: {gross_pnl:+.2f}, Fees: {-(entry_fee+exit_fee):+.2f})  Hold: {hold}s\n"
                    f"Today PnL: {DAILY_PNL:+.2f}\n"
                    f"Cooldown: {cooldown_type} mode activated"
                )

                # cancel any leftover TP/SL and clear
                try:
                    _cancel_open_brackets(ex, sym)
                except Exception as e_cancel:
                    tg(f"⚠️ Error cancelling brackets for {sym}: {e_cancel}")
                finally:
                    # Clear all tracking state
                    OPEN.pop(sym, None)
                    BRACKETS.pop(sym, None)  # Ensure BRACKETS is cleared too
                    PROGRESS_STAGE.pop(sym, None)
                    LAST_REARM_AT.pop(sym, None)
                    LAST_ROI_TELL.pop(sym, None)
                    LADDER_REMAINING_QTY.pop(sym, None)
                    LADDER_TP_PRICES.pop(sym, None)

                    # Clear scalper-specific state
                    SCALPER_ENTRY_TIME.pop(sym, None)
                    SCALPER_HIGHEST_PROFIT.pop(sym, None)
                    SCALPER_CONSECUTIVE_AGAINST.pop(sym, None)

                    tg(f"🧹 Cleared all tracking state for {sym}")

        except Exception as e:
            tg(f"⚠️ poll error while checking {sym}: {e}")
            continue
