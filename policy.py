import time
import os
import numpy as np
from typing import Optional, Dict, Tuple, List
from models import Decision

# =========================
# Common helpers
# =========================

def _atr_brackets(price: float, atr_val: float, side: str, sl_mult: float, tp_mult: float):
    if side == "long":
        sl = price - sl_mult * atr_val
        tp = price + tp_mult * atr_val
    else:
        sl = price + sl_mult * atr_val
        tp = price - tp_mult * atr_val
    return float(sl), float(tp)

def _fractals_swings(ts: np.ndarray, h: np.ndarray, l: np.ndarray, lookback: int = 120,
                     left: int = 2, right: int = 2) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Very light swing detector (fractals) over the last `lookback` bars.
    Returns two lists [(ts, price), ...]: swing_highs, swing_lows
    """
    start = max(0, len(ts) - lookback)
    highs, lows = [], []
    for i in range(start + left, len(ts) - right):
        hh = True
        ll = True
        for j in range(1, left + 1):
            if not (h[i] >= h[i - j]): hh = False
            if not (l[i] <= l[i - j]): ll = False
        for j in range(1, right + 1):
            if not (h[i] >= h[i + j]): hh = False
            if not (l[i] <= l[i + j]): ll = False
        if hh:
            highs.append((int(ts[i]), float(h[i])))
        if ll:
            lows.append((int(ts[i]), float(l[i])))
    return highs, lows

def _fit_line(swings: List[Tuple[int, float]]) -> Optional[Tuple[float, float]]:
    """
    Fit y = a*x + b through swings using simple least-squares on integer index to keep units stable.
    (We use integer indices rather than timestamps to avoid float overflows.)
    Returns (a, b) where x is the swing index (0..n-1).
    """
    if len(swings) < 2:
        return None
    ys = np.array([p for _, p in swings], dtype=float)
    xs = np.arange(len(ys), dtype=float)
    xbar = xs.mean()
    ybar = ys.mean()
    den = ((xs - xbar) ** 2).sum()
    if den == 0:
        return None
    a = (((xs - xbar) * (ys - ybar)).sum()) / den
    b = ybar - a * xbar
    return a, b

def _line_value(ab: Tuple[float, float], idx: int) -> float:
    a, b = ab
    return a * float(idx) + b

# =========================
# Combo A (existing)
# =========================

def _sweep_and_reclaim(snapshot: Dict, side: str) -> bool:
    c = snapshot["c"]; l = snapshot["l"]; h = snapshot["h"]; ema9 = snapshot["ema9"]
    i = -2
    if side == "long":
        return (l[i] < l[i-1]) and (c[i] > ema9[i])
    else:
        return (h[i] > h[i-1]) and (c[i] < ema9[i])

def decide_combo_A(symbol: str, htf_map: Dict, itf_setup: Dict, ltf_snapshot: Dict,
                   size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    if not itf_setup or not htf_map:
        return None
    side = itf_setup.get("side")
    if side not in ("long","short"):
        return None
    if not _sweep_and_reclaim(ltf_snapshot, side):
        return None

    c = ltf_snapshot["c"]; price = float(c[-2])
    atr_val = float(itf_setup.get("atr") or 0.0)
    if atr_val <= 0:
        return None
    sl, tp = _atr_brackets(price, atr_val, side, sl_mult, tp_mult)

    lo, hi = itf_setup["zone"]
    zone_mid = 0.5 * (lo + hi); band = max(hi - lo, 1e-9)
    zone_factor = min(1.0, abs(price - zone_mid) / band)
    rsi = float(ltf_snapshot["rsi14"][-2]); rsi_factor = min(1.0, abs(rsi - 50.0) / 50.0)
    confidence = round(0.4 * zone_factor + 0.6 * rsi_factor, 3)

    return Decision(
        symbol=symbol,
        side=side,
        entry_type="market",
        size_usdt=size_usdt,
        sl=sl,
        tp=tp,
        confidence=confidence,
        reason={"htf": f"{htf_map.get('bias')} @ SD", "itf":"15m retest", "ltf":"sweep&reclaim"},
        valid_until=time.time() + 60
    )

# =========================
# Combo B (DISABLED per request)
# =========================
# (kept for reference; router won't call it)

def _orb_break(snapshot: Dict, start_epoch: int, orb_minutes: int) -> Optional[Tuple[str, float]]:
    ts, h, l, c = snapshot["ts"], snapshot["h"], snapshot["l"], snapshot["c"]
    start_ms = start_epoch * 1000
    end_ms = start_ms + orb_minutes * 60 * 1000
    if ts[-2] < end_ms:
        return None
    mask = (ts >= start_ms) & (ts < end_ms)
    if mask.sum() < 1:
        return None
    or_high = float(np.max(h[mask])); or_low  = float(np.min(l[mask]))
    if c[-2] > or_high:
        return ("long", or_high)
    if c[-2] < or_low:
        return ("short", or_low)
    return None

def decide_combo_B(symbol: str, htf_map: Dict, itf_setup: Dict, ltf_snapshot: Dict,
                   size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    return None  # disabled

# =========================
# NEW Combo C1: Trendline Break + EMA(9) Pullback
# Uses 15m data from ltf_snapshot (your ITF builder should pass 15m into snapshot for combos)
# =========================

def decide_combo_C1(symbol: str, htf_map: Dict, itf_setup: Dict, ltf_snapshot: Dict,
                    size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    # need ATR from ITF (already computed upstream)
    atr_val = float(itf_setup.get("atr") or 0.0)
    if atr_val <= 0:
        return None

    ts = ltf_snapshot["ts"]; h = ltf_snapshot["h"]; l = ltf_snapshot["l"]; c = ltf_snapshot["c"]; ema9 = ltf_snapshot["ema9"]
    if len(ts) < 60:
        return None

    # build swings and fit a trendline (prefer lows if up-move has more lows; else highs)
    highs, lows = _fractals_swings(ts, h, l, lookback=120, left=2, right=2)
    use_lows = len(lows) >= len(highs)
    swings = lows if use_lows else highs
    line = _fit_line(swings)
    if not line:
        return None

    # compare last close to the trendline value at the "virtual" last swing index
    # we map current bar to index len(swings) (next point on the fitted line)
    line_y = _line_value(line, len(swings))
    last_close = float(c[-2])

    broke_up = (use_lows and last_close > line_y)
    broke_down = ((not use_lows) and last_close < line_y)
    side = "long" if broke_up else ("short" if broke_down else None)
    if not side:
        return None

    # EMA(9) pullback confirmation: close near ema9 and on the correct side
    tol = 0.003  # 0.3%
    if side == "long":
        valid = (abs(last_close - float(ema9[-2])) / max(1e-9, last_close)) < 2*tol and last_close >= float(ema9[-2])
    else:
        valid = (abs(last_close - float(ema9[-2])) / max(1e-9, last_close)) < 2*tol and last_close <= float(ema9[-2])
    if not valid:
        return None

    sl, tp = _atr_brackets(last_close, atr_val, side, sl_mult, tp_mult)

    return Decision(
        symbol=symbol,
        side=side,
        entry_type="market",
        size_usdt=size_usdt,
        sl=sl,
        tp=tp,
        confidence=0.55,
        reason={"combo":"C1","trendline_on": "lows" if use_lows else "highs"},
        valid_until=time.time() + 60
    )

# =========================
# NEW Combo C2: Consolidation Breakout + ATR Expansion
# Box from recent 15m range; breakout candle body/range filter; ATR provides brackets
# =========================

def decide_combo_C2(symbol: str, htf_map: Dict, itf_setup: Dict, ltf_snapshot: Dict,
                    size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    atr_val = float(itf_setup.get("atr") or 0.0)
    if atr_val <= 0:
        return None

    ts = ltf_snapshot["ts"]; h = ltf_snapshot["h"]; l = ltf_snapshot["l"]; c = ltf_snapshot["c"]; o = ltf_snapshot["o"]
    if len(ts) < 80:
        return None

    # Build a "box" from last 40 bars (tight consolidation if width < ~1.2%)
    window = slice(max(0, len(ts) - 40), len(ts) - 2)
    box_hi = float(np.max(h[window])); box_lo = float(np.min(l[window]))
    mid = (box_hi + box_lo) / 2.0
    width = (box_hi - box_lo) / max(mid, 1e-9)
    if width >= 0.012:   # ~1.2% tightness gate
        return None

    # Breakout on the latest closed bar
    body = abs(float(c[-2]) - float(o[-2])); rng = float(h[-2]) - float(l[-2]) or 1e-9
    body_ratio = body / rng
    broke_up = (float(c[-2]) > box_hi) and (body_ratio > 0.6)
    broke_dn = (float(c[-2]) < box_lo) and (body_ratio > 0.6)
    side = "long" if broke_up else ("short" if broke_dn else None)
    if not side:
        return None

    price = float(c[-2])
    sl, tp = _atr_brackets(price, atr_val, side, sl_mult, tp_mult)

    return Decision(
        symbol=symbol,
        side=side,
        entry_type="market",
        size_usdt=size_usdt,
        sl=sl,
        tp=tp,
        confidence=0.55,
        reason={"combo":"C2","box":[round(box_lo,6), round(box_hi,6)], "box_width_pct": round(width*100,3)},
        valid_until=time.time() + 60
    )

# =========================
# Standalone 15m: Break & Retest (BR)
# =========================

def decide_standalone_BR(symbol: str, htf_map: Dict, itf_setup: Dict, ltf_snapshot: Dict,
                         size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    """
    Use 15m data (provided via ltf_snapshot here).
    1) Find recent swing high/low as the level.
    2) Confirm breakout (close beyond by tiny buffer).
    3) Retest within Y% + rejection close.
    """
    atr_val = float(itf_setup.get("atr") or 0.0)
    if atr_val <= 0:
        return None

    ts = ltf_snapshot["ts"]; h = ltf_snapshot["h"]; l = ltf_snapshot["l"]; c = ltf_snapshot["c"]; o = ltf_snapshot["o"]
    if len(ts) < 80:
        return None

    # level from previous swings
    highs, lows = _fractals_swings(ts, h, l, lookback=100)
    if not highs and not lows:
        return None
    level_high = float(highs[-1][1]) if highs else None
    level_low  = float(lows[-1][1])  if lows  else None

    buf = 0.0015  # 0.15% breakout buffer
    # Check last 10 bars for break->retest sequence
    for i in range(len(ts)-12, len(ts)-2):
        # long version
        if level_high and float(c[i]) > level_high * (1 + buf):
            # retest in next bars: low back within 0.2% of level and close strong
            near = abs(float(l[i+1]) - level_high) / max(level_high, 1e-9) < 0.002
            if near and float(c[i+1]) > float(o[i+1]):
                price = float(c[i+1])
                sl, tp = _atr_brackets(price, atr_val, "long", sl_mult, tp_mult)
                return Decision(symbol=symbol, side="long", entry_type="market",
                                size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.53,
                                reason={"standalone":"BR","level":round(level_high,6)}, valid_until=time.time()+60)
        # short version
        if level_low and float(c[i]) < level_low * (1 - buf):
            near = abs(float(h[i+1]) - level_low) / max(level_low, 1e-9) < 0.002
            if near and float(c[i+1]) < float(o[i+1]):
                price = float(c[i+1])
                sl, tp = _atr_brackets(price, atr_val, "short", sl_mult, tp_mult)
                return Decision(symbol=symbol, side="short", entry_type="market",
                                size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.53,
                                reason={"standalone":"BR","level":round(level_low,6)}, valid_until=time.time()+60)

    return None

# =========================
# Standalone 15m: Pullback to EMA(9) (EMA9)
# =========================

def decide_standalone_EMA9(symbol: str, htf_map: Dict, itf_setup: Dict, ltf_snapshot: Dict,
                           size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    """
    Simple trend-following pullback:
    - Trend up if price > ema9 and ema9 rising (last > prev)
    - Wait for a touch/near-touch then a strong close away from ema
    """
    atr_val = float(itf_setup.get("atr") or 0.0)
    if atr_val <= 0:
        return None

    c = ltf_snapshot["c"]; o = ltf_snapshot["o"]; ema9 = ltf_snapshot["ema9"]
    if len(c) < 20:
        return None

    prox = abs(float(c[-2]) - float(ema9[-2])) / max(1e-9, float(c[-2]))
    ema_rising = float(ema9[-2]) > float(ema9[-3])
    ema_falling = float(ema9[-2]) < float(ema9[-3])

    # Long pullback
    if float(c[-3]) > float(ema9[-3]) and ema_rising and prox < 0.003 and float(c[-2]) > float(o[-2]):
        price = float(c[-2])
        sl, tp = _atr_brackets(price, atr_val, "long", sl_mult, tp_mult)
        return Decision(symbol=symbol, side="long", entry_type="market",
                        size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.52,
                        reason={"standalone":"EMA9","dir":"up"}, valid_until=time.time()+60)

    # Short pullback
    if float(c[-3]) < float(ema9[-3]) and ema_falling and prox < 0.003 and float(c[-2]) < float(o[-2]):
        price = float(c[-2])
        sl, tp = _atr_brackets(price, atr_val, "short", sl_mult, tp_mult)
        return Decision(symbol=symbol, side="short", entry_type="market",
                        size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.52,
                        reason={"standalone":"EMA9","dir":"down"}, valid_until=time.time()+60)

    return None

# =========================
# Router
# =========================

def decide(symbol: str, combo: str, htf_map: Dict, itf_setup: Dict, ltf_snapshot: Dict,
           size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    """
    combo codes:
      A    -> Combo A (Supply/Demand + BOS/CHOCH → Retest → LTF trigger)
      C1   -> Combo C1 (Trendline Break + EMA9 Pullback)
      C2   -> Combo C2 (Consolidation Breakout + ATR expansion)
      BR   -> Standalone 15m Break & Retest
      EMA9 -> Standalone 15m Pullback to 9 EMA
      B    -> (disabled)
    """
    code = (combo or "A").upper()
    if code == "A":
        return decide_combo_A(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    if code == "C1":
        return decide_combo_C1(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    if code == "C2":
        return decide_combo_C2(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    if code == "BR":
        return decide_standalone_BR(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    if code == "EMA9":
        return decide_standalone_EMA9(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    # if code == "B":  # disabled
    #     return decide_combo_B(...)
    return None
