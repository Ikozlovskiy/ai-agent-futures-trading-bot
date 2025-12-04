import os, time
from typing import Dict, Optional, List, Tuple
from models import Decision, PositionMemo
from utils import tg

# Tracks 1 open position memo per symbol (in-memory)
OPEN: Dict[str, PositionMemo] = {}
# Tracks bracket order ids per symbol: {"tp": "...", "sl": "..."} or {"tp1": "...", "tp2": "...", "sl": "..."}
BRACKETS: Dict[str, Dict[str, str]] = {}
# Tracks remaining qty for LADDER mode (decreases as TPs fill)
LADDER_REMAINING_QTY: Dict[str, float] = {}

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
    try:
        open_orders = ex.fetch_open_orders(symbol)
    except Exception:
        open_orders = []

    # Get current price for validation
    cur_price = _current_price(ex, symbol)

    # Validate new SL is in the correct direction (more favorable than current price)
    if new_sl is not None:
        if side == "long":
            # For longs, SL should be below current price
            if new_sl >= cur_price:
                tg(f"⚠️ {symbol} SL validation failed: SL {new_sl:.6f} >= current {cur_price:.6f} (LONG). Setting to entry or below.")
                # Get entry price from memo if available
                memo = OPEN.get(symbol)
                if memo and memo.entry_price < cur_price:
                    new_sl = memo.entry_price  # Move to breakeven as safety
                else:
                    new_sl = cur_price * 0.995  # 0.5% below current as last resort
        else:
            # For shorts, SL should be above current price
            if new_sl <= cur_price:
                tg(f"⚠️ {symbol} SL validation failed: SL {new_sl:.6f} <= current {cur_price:.6f} (SHORT). Setting to entry or above.")
                # Get entry price from memo if available
                memo = OPEN.get(symbol)
                if memo and memo.entry_price > cur_price:
                    new_sl = memo.entry_price  # Move to breakeven as safety
                else:
                    new_sl = cur_price * 1.005  # 0.5% above current as last resort

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
        safe_tp = _safe_stop_price(ex, symbol, side, float(new_tp), "TP")
        try:
            tp = ex.create_order(symbol, "TAKE_PROFIT_MARKET", reduce_side, qty, None,
                                 {**base, "stopPrice": float(safe_tp)})
            new_ids["tp"] = str(tp.get("id") or tp.get("orderId") or "")
        except Exception as e:
            tg(f"⚠️ TP re-arm error {symbol}: {e}")

    # place new SL
    if new_sl is not None:
        safe_sl = _safe_stop_price(ex, symbol, side, float(new_sl), "SL")
        try:
            sl = ex.create_order(symbol, "STOP_MARKET", reduce_side, qty, None,
                                 {**base, "stopPrice": float(safe_sl)})
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

    # Cancel old brackets and place new ones (tick-safe)
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


# -------- Order placement & polling --------

def _place_brackets(ex, decision: Decision, qty: float) -> Dict[str, Optional[str]]:
    """
    Place reduce-only TP/SL as market-triggered orders on Binance USDM (tick-safe).
    Returns order ids dict: {"tp": "...", "sl": "..."} (ids may be None on error).
    """
    reduce_side = "sell" if decision.side == "long" else "buy"
    base_params = {"reduceOnly": True, "workingType": "MARK_PRICE"}

    ids: Dict[str, Optional[str]] = {"tp": None, "sl": None}

    # TAKE PROFIT
    try:
        tp_px = _safe_stop_price(ex, decision.symbol, decision.side, float(decision.tp), "TP")
        tp = ex.create_order(
            decision.symbol,
            "TAKE_PROFIT_MARKET",
            reduce_side,
            qty,
            None,
            {**base_params, "stopPrice": float(tp_px)},
        )
        ids["tp"] = str(tp.get("id") or tp.get("orderId") or "")
    except Exception as e:
        tg(f"⚠️ TP order error {decision.symbol}: {e}")

    # STOP LOSS
    try:
        sl_px = _safe_stop_price(ex, decision.symbol, decision.side, float(decision.sl), "SL")
        sl = ex.create_order(
            decision.symbol,
            "STOP_MARKET",
            reduce_side,
            qty,
            None,
            {**base_params, "stopPrice": float(sl_px)},
        )
        ids["sl"] = str(sl.get("id") or sl.get("orderId") or "")
    except Exception as e:
        tg(f"⚠️ SL order error {decision.symbol}: {e}")

    return ids


def _place_ladder_brackets(ex, symbol: str, side: str, qty: float, sl_price: float, 
                           tp_levels: List[Tuple[float, float]]) -> Dict[str, Optional[str]]:
    """
    Place multiple reduce-only TP orders and one SL for LADDER mode.
    tp_levels: [(tp_price, qty_pct), ...] where qty_pct is percentage of total position
    Returns: {"tp1": "id1", "tp2": "id2", ..., "sl": "sl_id"}
    """
    reduce_side = "sell" if side == "long" else "buy"
    base_params = {"reduceOnly": True, "workingType": "MARK_PRICE"}

    ids: Dict[str, Optional[str]] = {}

    # Place multiple TP orders
    for i, (tp_price, qty_pct) in enumerate(tp_levels, start=1):
        tp_qty = qty * (qty_pct / 100.0)
        # Snap qty to lot size
        try:
            tp_qty = float(ex.amount_to_precision(symbol, tp_qty))
        except Exception:
            pass

        if tp_qty <= 0:
            tg(f"⚠️ TP{i} qty too small, skipping")
            continue

        try:
            tp_px = _safe_stop_price(ex, symbol, side, float(tp_price), "TP")
            tp = ex.create_order(
                symbol,
                "TAKE_PROFIT_MARKET",
                reduce_side,
                tp_qty,
                None,
                {**base_params, "stopPrice": float(tp_px)},
            )
            ids[f"tp{i}"] = str(tp.get("id") or tp.get("orderId") or "")
            tg(f"📊 TP{i} placed: {tp_qty:.6f} @ {tp_px:.6f} ({qty_pct}%)")
        except Exception as e:
            tg(f"⚠️ TP{i} order error {symbol}: {e}")
            ids[f"tp{i}"] = None

    # Place SL for full remaining position
    try:
        sl_px = _safe_stop_price(ex, symbol, side, float(sl_price), "SL")
        sl = ex.create_order(
            symbol,
            "STOP_MARKET",
            reduce_side,
            qty,
            None,
            {**base_params, "stopPrice": float(sl_px)},
        )
        ids["sl"] = str(sl.get("id") or sl.get("orderId") or "")
    except Exception as e:
        tg(f"⚠️ SL order error {symbol}: {e}")
        ids["sl"] = None

    return ids


def _try_detect_exit_by_orders(ex, symbol: str) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Check if any TP or SL order is 'closed'.
    For LADDER mode: handle partial fills and update remaining qty + adjust SL qty.
    Returns: (fully_exited, partial_qty_closed, exit_price) or (False, None, None)
    """
    ids = BRACKETS.get(symbol)
    if not ids:
        return False, None, None

    def is_closed(oid: Optional[str]) -> Tuple[bool, Optional[float], Optional[float]]:
        """Returns (is_closed, filled_qty, avg_price)"""
        if not oid:
            return False, None, None
        try:
            o = ex.fetch_order(oid, symbol)
            status = str(o.get("status", "")).lower()
            if status in ("closed", "filled"):
                filled = float(o.get("filled", 0) or 0)
                avg_px = float(o.get("average") or o.get("price") or 0)
                return True, filled, avg_px
        except Exception:
            pass
        return False, None, None

    # Check SL - if hit, full exit
    sl_closed, sl_qty, sl_px = is_closed(ids.get("sl"))
    if sl_closed:
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
                # Partial TP hit - update remaining qty and adjust SL
                if symbol in LADDER_REMAINING_QTY:
                    LADDER_REMAINING_QTY[symbol] -= qty
                    remaining = LADDER_REMAINING_QTY[symbol]

                    # Report partial exit
                    tg(f"📉 Partial TP hit: {symbol} {key.upper()} closed {qty:.6f} @ {px:.6f}. Remaining: {remaining:.6f}")

                    # Remove this TP from tracking
                    ids.pop(key, None)

                    # Adjust SL order quantity and MOVE TO BREAKEVEN after first TP
                    try:
                        sl_id = ids.get("sl")
                        if sl_id and remaining > 0:
                            # Cancel old SL
                            try:
                                ex.cancel_order(sl_id, symbol)
                            except Exception:
                                pass

                            # Place new SL with reduced qty at BREAKEVEN (entry price)
                            memo = OPEN.get(symbol)
                            if memo:
                                reduce_side = "sell" if memo.side == "long" else "buy"
                                # MOVE SL TO BREAKEVEN (0% ROI) = entry price
                                sl_price = float(memo.entry_price)

                                sl_px_safe = _safe_stop_price(ex, symbol, memo.side, sl_price, "SL")
                                new_sl = ex.create_order(
                                    symbol, "STOP_MARKET", reduce_side, remaining, None,
                                    {"reduceOnly": True, "workingType": "MARK_PRICE", "stopPrice": float(sl_px_safe)}
                                )
                                ids["sl"] = str(new_sl.get("id") or new_sl.get("orderId") or "")
                                tg(f"🔒 SL moved to BREAKEVEN: {sl_px_safe:.6f} (entry price, 0% ROI)")
                    except Exception as e:
                        tg(f"⚠️ Failed to adjust SL after partial TP: {e}")

                    # Check if all TPs are filled (full exit)
                    if remaining < 1e-6 or not any(k.startswith("tp") for k in ids.keys()):
                        return True, None, None

                    return False, qty, px  # Partial exit, continue monitoring

        return False, None, None
    else:
        # Single TP mode
        tp_closed, tp_qty, tp_px = is_closed(ids.get("tp"))
        if tp_closed:
            return True, tp_qty, tp_px
        return False, None, None


def _cancel_open_brackets(ex, symbol: str):
    """Cancel any open TP/SL reduce-only orders for a symbol using stored ids."""
    if symbol not in BRACKETS:
        # Still attempt to cancel stray reduce-only orders:
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

    # Cancel all bracket orders (handles both single TP and multiple TP1, TP2, TP3, etc.)
    for key, order_id in ids.items():
        cancel(order_id)

    BRACKETS.pop(symbol, None)


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
        ids = _place_ladder_brackets(ex, decision.symbol, decision.side, qty, ladder_sl, ladder_tp_levels)
    else:
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
            # While position is open, consider dynamic re-arm based on configured stages
            # (Note: dynamic re-arm is incompatible with LADDER mode - skip if ladder active)
            if sym not in LADDER_REMAINING_QTY:
                try:
                    _maybe_dynamic_rearm(ex, sym, memo)
                except Exception as e_dyn:
                    tg(f"⚠️ dynamic re-arm error {sym}: {e_dyn}")

            # 1) Fast path: did one of our bracket orders fill?
            exited_by_order, partial_qty, partial_px = _try_detect_exit_by_orders(ex, sym)

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

                tg(
                    f"✅ <b>EXIT</b> {sym} {memo.side.upper()}  Net PnL: {net_pnl:+.2f}  (Gross: {gross_pnl:+.2f}, Fees: {-(entry_fee+exit_fee):+.2f})  Hold: {hold}s\n"
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
                    LADDER_REMAINING_QTY.pop(sym, None)

        except Exception as e:
            tg(f"⚠️ poll error while checking {sym}: {e}")
            continue
