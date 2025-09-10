import time, os
import numpy as np
from typing import Optional, Dict, Tuple, List
from models import Decision

# =========================
# Debug helper (TG only when enabled)
# =========================
DEBUG = (os.getenv("DEBUG_DECISIONS", "false").lower() == "true")
try:
    from utils import tg as _dbg_tg
except Exception:
    def _dbg_tg(*args, **kwargs): pass

def _dbg(msg: str):
    if DEBUG:
        _dbg_tg("🔎 " + msg)

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

def _safe_arrays(snap: Dict, keys: List[str]) -> bool:
    if not isinstance(snap, dict):
        return False
    for k in keys:
        arr = snap.get(k)
        if arr is None or len(arr) < 5:
            return False
    return True

def _fractals_swings(ts: np.ndarray, h: np.ndarray, l: np.ndarray, lookback: int = 120,
                     left: int = 2, right: int = 2) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    if ts is None or h is None or l is None or len(ts) == 0:
        return [], []
    start = max(0, len(ts) - lookback)
    highs, lows = [], []
    for i in range(start + left, len(ts) - right):
        hh = True; ll = True
        for j in range(1, left + 1):
            if not (h[i] >= h[i - j]): hh = False
            if not (l[i] <= l[i - j]): ll = False
        for j in range(1, right + 1):
            if not (h[i] >= h[i + j]): hh = False
            if not (l[i] <= l[i + j]): ll = False
        if hh: highs.append((int(ts[i]), float(h[i])))
        if ll: lows.append((int(ts[i]), float(l[i])))
    return highs, lows

def _fit_line(swings: List[Tuple[int, float]]) -> Optional[Tuple[float, float]]:
    if len(swings) < 2: return None
    ys = np.array([p for _, p in swings], dtype=float)
    xs = np.arange(len(ys), dtype=float)
    xbar = xs.mean(); ybar = ys.mean()
    den = ((xs - xbar) ** 2).sum()
    if den == 0: return None
    a = (((xs - xbar) * (ys - ybar)).sum()) / den
    b = ybar - a * xbar
    return a, b

def _line_value(ab: Tuple[float, float], idx: int) -> float:
    a,b = ab
    return a * float(idx) + b

def _get_atr_from_itf(itf_setup: Optional[Dict]) -> float:
    try:
        return float((itf_setup or {}).get("atr") or 0.0)
    except Exception:
        return 0.0

# =========================
# HYBRID/LTF-aware brackets (optional; used by caller elsewhere)
# =========================
def _last_ltf_swing(ltf_snapshot: Dict, side: str) -> Optional[float]:
    try:
        ts, h, l = ltf_snapshot["ts"], ltf_snapshot["h"], ltf_snapshot["l"]
        for i in range(len(ts)-3, 2, -1):
            if side == "long":
                if l[i] < l[i-1] and l[i] < l[i+1]:
                    return float(l[i])
            else:
                if h[i] > h[i-1] and h[i] > h[i+1]:
                    return float(h[i])
    except Exception:
        pass
    return None

def _compute_brackets(price: float, side: str, itf_setup: Optional[Dict], ltf_snapshot: Dict) -> Optional[Tuple[float,float]]:
    src = (os.getenv("ATR_SOURCE","ITF") or "ITF").upper()
    atr_itf = float((itf_setup or {}).get("atr") or 0.0)
    price = float(price)
    sl_mult = float(os.getenv("ATR_SL_MULT","1.5") or 1.5)
    tp_mult = float(os.getenv("ATR_TP_MULT","3.0") or 3.0)

    if src == "ITF":
        if atr_itf <= 0: return None
        return _atr_brackets(price, atr_itf, side, sl_mult, tp_mult)

    atr_ltf = float((ltf_snapshot or {}).get("atr_ltf") or 0.0)
    if src == "LTF":
        atr_use = atr_ltf if atr_ltf > 0 else atr_itf
        if atr_use <= 0: return None
        return _atr_brackets(price, atr_use, side, sl_mult, tp_mult)

    # HYBRID
    if atr_itf <= 0: return None
    floor_mult = float(os.getenv("HYBRID_SL_FLOOR_ITF_ATR_MULT","0.6") or 0.6)
    wick_buf   = float(os.getenv("HYBRID_WICK_BUFFER_PCT","0.0007") or 0.0007)
    tp_itf_mult= float(os.getenv("HYBRID_TP_ITF_ATR_MULT","1.0") or 1.0)

    swing = _last_ltf_swing(ltf_snapshot, side)
    if swing is not None:
        if side == "long":
            sl_swing = min(swing, price) * (1.0 - wick_buf)
            sl_dist  = price - sl_swing
        else:
            sl_swing = max(swing, price) * (1.0 + wick_buf)
            sl_dist  = sl_swing - price
    else:
        sl_dist = 0.0

    sl_floor = sl_mult * atr_itf * floor_mult
    effective_sl_dist = max(sl_dist, sl_floor)

    if side == "long":
        sl = price - effective_sl_dist
        tp = price + (tp_mult * atr_itf * tp_itf_mult)
    else:
        sl = price + effective_sl_dist
        tp = price - (tp_mult * atr_itf * tp_itf_mult)
    return float(sl), float(tp)

# =========================
# Combo A (S/D + BOS/CHOCH → retest → LTF trigger)
# =========================

def _sweep_and_reclaim(snapshot: Dict, side: str) -> bool:
    c = snapshot["c"]; l = snapshot["l"]; h = snapshot["h"]; ema9 = snapshot["ema9"]
    i = -2
    if side == "long":
        return (l[i] < l[i-1]) and (c[i] > ema9[i])
    else:
        return (h[i] > h[i-1]) and (c[i] < ema9[i])

def decide_combo_A(symbol: str, htf_map: Dict, itf_setup: Optional[Dict], ltf_snapshot: Dict,
                   size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    if not isinstance(itf_setup, dict) or not isinstance(htf_map, dict):
        _dbg(f"{symbol} A: itf/htf missing")
        return None
    if not _safe_arrays(ltf_snapshot, ["c","l","h","ema9","rsi14"]):
        _dbg(f"{symbol} A: ltf arrays missing")
        return None

    side = itf_setup.get("side")
    if side not in ("long","short"):
        _dbg(f"{symbol} A: side not set")
        return None
    if not _sweep_and_reclaim(ltf_snapshot, side):
        _dbg(f"{symbol} A: sweep&reclaim fail")
        return None

    price = float(ltf_snapshot["c"][-2])
    br = _compute_brackets(price, side, itf_setup, ltf_snapshot)
    if not br: return None
    sl, tp = br

    zone = itf_setup.get("zone")
    if not zone or len(zone) != 2:
        _dbg(f"{symbol} A: zone missing")
        return None
    lo, hi = zone
    zone_mid = 0.5 * (lo + hi); band = max(hi - lo, 1e-9)
    zone_factor = min(1.0, abs(price - zone_mid) / band)
    rsi = float(ltf_snapshot["rsi14"][-2]); rsi_factor = min(1.0, abs(rsi - 50.0) / 50.0)
    confidence = round(0.4 * zone_factor + 0.6 * rsi_factor, 3)

    return Decision(symbol=symbol, side=side, entry_type="market",
                    size_usdt=size_usdt, sl=sl, tp=tp, confidence=confidence,
                    reason={"htf": f"{htf_map.get('bias')} @ SD", "itf":"retest", "ltf":"sweep&reclaim"},
                    valid_until=time.time()+60)

# =========================
# Combo C1 (Trendline Break + EMA9 Pullback)
# =========================

def decide_combo_C1(symbol: str, htf_map: Dict, itf_setup: Optional[Dict], ltf_snapshot: Dict,
                    size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    atr_val = _get_atr_from_itf(itf_setup)
    if atr_val <= 0:
        _dbg(f"{symbol} C1: atr 0")
        return None
    if not _safe_arrays(ltf_snapshot, ["ts","h","l","c","ema9"]):
        _dbg(f"{symbol} C1: ltf arrays missing")
        return None

    ts = ltf_snapshot["ts"]; h = ltf_snapshot["h"]; l = ltf_snapshot["l"]; c = ltf_snapshot["c"]; ema9 = ltf_snapshot["ema9"]
    if len(ts) < 60:
        return None

    highs, lows = _fractals_swings(ts, h, l, lookback=120, left=2, right=2)
    use_lows = len(lows) >= len(highs)
    swings = lows if use_lows else highs
    line = _fit_line(swings)
    if not line:
        _dbg(f"{symbol} C1: no trendline")
        return None

    line_y = _line_value(line, len(swings))
    last_close = float(c[-2])
    broke_up = (use_lows and last_close > line_y)
    broke_down = ((not use_lows) and last_close < line_y)
    side = "long" if broke_up else ("short" if broke_down else None)
    if not side:
        _dbg(f"{symbol} C1: no break")
        return None

    tol = float(os.getenv("C1_EMA_PROX_TOL","0.003") or 0.003)
    ok_prox = (abs(last_close - float(ema9[-2])) / max(1e-9, last_close)) < 2*tol
    ok_side = (last_close >= float(ema9[-2])) if side == "long" else (last_close <= float(ema9[-2]))
    if not (ok_prox and ok_side):
        _dbg(f"{symbol} C1: ema prox fail tol={tol}")
        return None

    br = _compute_brackets(last_close, side, itf_setup, ltf_snapshot)
    if not br: return None
    sl, tp = br

    return Decision(symbol=symbol, side=side, entry_type="market",
                    size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.55,
                    reason={"combo":"C1","trendline_on":"lows" if use_lows else "highs"},
                    valid_until=time.time()+60)

# =========================
# Combo C2 (Consolidation Breakout + ATR Expansion)
# =========================

def decide_combo_C2(symbol: str, htf_map: Dict, itf_setup: Optional[Dict], ltf_snapshot: Dict,
                    size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    atr_val = _get_atr_from_itf(itf_setup)
    if atr_val <= 0:
        _dbg(f"{symbol} C2: atr 0")
        return None
    if not _safe_arrays(ltf_snapshot, ["ts","h","l","c","o"]):
        _dbg(f"{symbol} C2: ltf arrays missing")
        return None

    ts = ltf_snapshot["ts"]; h = ltf_snapshot["h"]; l = ltf_snapshot["l"]; c = ltf_snapshot["c"]; o = ltf_snapshot["o"]
    if len(ts) < 80:
        return None

    width_max = float(os.getenv("C2_BOX_TIGHTNESS","0.012") or 0.012)
    win = slice(max(0, len(ts)-40), len(ts)-2)
    box_hi = float(np.max(h[win])); box_lo = float(np.min(l[win]))
    mid = (box_hi + box_lo)/2.0
    width = (box_hi - box_lo)/max(mid, 1e-9)
    if width >= width_max:
        _dbg(f"{symbol} C2: box too wide {width:.4f}>{width_max}")
        return None

    body = abs(float(c[-2]) - float(o[-2])); rng = float(h[-2]) - float(l[-2]) or 1e-9
    body_ratio = body / rng
    need = float(os.getenv("C2_BODY_RATIO","0.6") or 0.6)
    broke_up = (float(c[-2]) > box_hi) and (body_ratio > need)
    broke_dn = (float(c[-2]) < box_lo) and (body_ratio > need)
    side = "long" if broke_up else ("short" if broke_dn else None)
    if not side:
        _dbg(f"{symbol} C2: no breakout body_ratio={body_ratio:.2f} need>{need}")
        return None

    price = float(c[-2])
    br = _compute_brackets(price, side, itf_setup, ltf_snapshot)
    if not br: return None
    sl, tp = br

    return Decision(symbol=symbol, side=side, entry_type="market",
                    size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.55,
                    reason={"combo":"C2","box":[round(box_lo,6), round(box_hi,6)], "box_width_pct": round(width*100,3)},
                    valid_until=time.time()+60)

# =========================
# Standalone BR (Break & Retest on ITF)
# =========================

def decide_standalone_BR(symbol: str, htf_map: Dict, itf_setup: Optional[Dict], ltf_snapshot: Dict,
                         size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    atr_val = _get_atr_from_itf(itf_setup)
    if atr_val <= 0:
        _dbg(f"{symbol} BR: atr 0")
        return None
    if not _safe_arrays(ltf_snapshot, ["ts","h","l","c","o"]):
        _dbg(f"{symbol} BR: ltf arrays missing")
        return None

    ts = ltf_snapshot["ts"]; h = ltf_snapshot["h"]; l = ltf_snapshot["l"]; c = ltf_snapshot["c"]; o = ltf_snapshot["o"]
    if len(ts) < 80:
        return None

    buf = float(os.getenv("BR_BREAK_BUFFER","0.0015") or 0.0015)
    retest_tol = float(os.getenv("BR_RETEST_TOL","0.002") or 0.002)
    retest_bars = int(os.getenv("BR_RETEST_BARS","2") or 2)
    min_body_ratio = float(os.getenv("BR_MIN_BODY_RATIO","0.0") or 0.0)

    highs, lows = _fractals_swings(ts, h, l, lookback=100)
    if not highs and not lows:
        _dbg(f"{symbol} BR: no swings")
        return None
    level_high = float(highs[-1][1]) if highs else None
    level_low  = float(lows[-1][1])  if lows  else None

    start_i = max(10, len(ts) - 12)
    for i in range(start_i, len(ts) - 2):
        # LONG
        if level_high and float(c[i]) > level_high * (1.0 + buf):
            for j in range(1, retest_bars + 1):
                k = i + j
                if k >= len(ts) - 1: break
                near = abs(float(l[k]) - level_high) / max(level_high, 1e-9) < retest_tol
                body = abs(float(c[k]) - float(o[k])); rng = max(float(h[k]) - float(l[k]), 1e-9)
                strong = (min_body_ratio == 0.0) or (body / rng >= min_body_ratio)
                if near and (float(c[k]) > float(o[k])) and strong:
                    price = float(c[k])
                    br = _compute_brackets(price, "long", itf_setup, ltf_snapshot)
                    if not br: return None
                    sl, tp = br
                    return Decision(symbol=symbol, side="long", entry_type="market",
                                    size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.53,
                                    reason={"standalone":"BR","level":round(level_high,6),
                                            "break_i": i, "retest_i": k,
                                            "buf": buf, "retest_tol": retest_tol, "retest_bars": retest_bars},
                                    valid_until=time.time()+60)
        # SHORT
        if level_low and float(c[i]) < level_low * (1.0 - buf):
            for j in range(1, retest_bars + 1):
                k = i + j
                if k >= len(ts) - 1: break
                near = abs(float(h[k]) - level_low) / max(level_low, 1e-9) < retest_tol
                body = abs(float(c[k]) - float(o[k])); rng = max(float(h[k]) - float(l[k]), 1e-9)
                strong = (min_body_ratio == 0.0) or (body / rng >= min_body_ratio)
                if near and (float(c[k]) < float(o[k])) and strong:
                    price = float(c[k])
                    br = _compute_brackets(price, "short", itf_setup, ltf_snapshot)
                    if not br: return None
                    sl, tp = br
                    return Decision(symbol=symbol, side="short", entry_type="market",
                                    size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.53,
                                    reason={"standalone":"BR","level":round(level_low,6),
                                            "break_i": i, "retest_i": k,
                                            "buf": buf, "retest_tol": retest_tol, "retest_bars": retest_bars},
                                    valid_until=time.time()+60)
    _dbg(f"{symbol} BR: no valid break→retest found")
    return None

# =========================
# Standalone EMA9 (ITF pullback)
# =========================

def decide_standalone_EMA9(symbol: str, htf_map: Dict, itf_setup: Optional[Dict], ltf_snapshot: Dict,
                           size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    atr_val = _get_atr_from_itf(itf_setup)
    if atr_val <= 0:
        _dbg(f"{symbol} EMA9: atr 0")
        return None
    if not _safe_arrays(ltf_snapshot, ["c","o","ema9"]):
        _dbg(f"{symbol} EMA9: arrays missing")
        return None
    c = ltf_snapshot["c"]; o = ltf_snapshot["o"]; ema9 = ltf_snapshot["ema9"]
    if len(c) < 20: return None

    tol = float(os.getenv("EMA9_PROX_TOL","0.003") or 0.003)
    prox = abs(float(c[-2]) - float(ema9[-2])) / max(1e-9, float(c[-2]))
    ema_rising  = float(ema9[-2]) > float(ema9[-3])
    ema_falling = float(ema9[-2]) < float(ema9[-3])

    # Long
    if float(c[-3]) > float(ema9[-3]) and ema_rising and prox < tol and float(c[-2]) > float(o[-2]):
        price = float(c[-2])
        br = _compute_brackets(price, "long", itf_setup, ltf_snapshot)
        if not br: return None
        sl, tp = br
        return Decision(symbol=symbol, side="long", entry_type="market",
                        size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.52,
                        reason={"standalone":"EMA9","dir":"up","prox":prox,"tol":tol},
                        valid_until=time.time()+60)

    # Short
    if float(c[-3]) < float(ema9[-3]) and ema_falling and prox < tol and float(c[-2]) < float(o[-2]):
        price = float(c[-2])
        br = _compute_brackets(price, "short", itf_setup, ltf_snapshot)
        if not br: return None
        sl, tp = br
        return Decision(symbol=symbol, side="short", entry_type="market",
                        size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.52,
                        reason={"standalone":"EMA9","dir":"down","prox":prox,"tol":tol},
                        valid_until=time.time()+60)

    _dbg(f"{symbol} EMA9: no valid pullback")
    return None

# =========================
# Router (supports multiple strategies per symbol)
# =========================

def _dispatch_one(code: str, symbol: str, htf_map: Dict, itf_setup: Optional[Dict], ltf_snapshot: Dict,
                  size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    c = code.upper()
    if c == "A":    return decide_combo_A(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    if c == "C1":   return decide_combo_C1(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    if c == "C2":   return decide_combo_C2(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    if c == "BR":   return decide_standalone_BR(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    if c == "EMA9": return decide_standalone_EMA9(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    return None

def decide(symbol: str, combo: str, htf_map: Dict, itf_setup: Dict, ltf_snapshot: Dict,
           size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    """
    combo can be a single code ("A") or a '+'-separated list ("EMA9+BR+C2").
    We try in order and return the first non-None decision.
    """
    if not combo:
        combo = "A"
    for code in str(combo).split("+"):
        code = code.strip()
        if not code: continue
        d = _dispatch_one(code, symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
        if d is not None:
            return d
    return None
