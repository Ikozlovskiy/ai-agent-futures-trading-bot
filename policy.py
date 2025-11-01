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

def _dbg_kv(ctx: str, data: dict):
    if DEBUG:
        try:
            from pprint import pformat
            _dbg_tg("🔎 " + ctx + "\n" + pformat(data))
        except Exception:
            _dbg("ppfail " + ctx)

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
# Extra utilities (trend/vol/ATR)
# =========================

def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    if arr is None or len(arr) < span + 5:
        return np.array([])
    alpha = 2 / (span + 1.0)
    out = np.zeros_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def _true_range(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev_close = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    return tr

def _median_last(arr: np.ndarray, n: int) -> float:
    n = min(n, len(arr))
    if n <= 1: return float(arr[-1]) if len(arr) else 0.0
    return float(np.median(arr[-n:]))

def _atr_from_tr(tr: np.ndarray, period: int = 14) -> float:
    if len(tr) < period + 2: return 0.0
    return float(np.mean(tr[-period:]))

def _regime_ok(itf: Dict, ltf: Dict) -> bool:
    """Basic trend & vol regime filter applied before combos fire."""
    if not all(k in itf for k in ["o","h","l","c"]): return False
    c = itf["c"]; o = itf["o"]; h = itf["h"]; l = itf["l"]
    if len(c) < 220: return False

    ema_fast = _ema(c, int(os.getenv("REGIME_EMA_FAST","50")))
    ema_slow = _ema(c, int(os.getenv("REGIME_EMA_SLOW","200")))
    if len(ema_fast) < len(c) or len(ema_slow) < len(c): return False

    trend = (ema_fast[-2] > ema_slow[-2] and ema_fast[-2] > ema_fast[-6]) or \
            (ema_fast[-2] < ema_slow[-2] and ema_fast[-2] < ema_fast[-6])

    tr = _true_range(o,h,l,c)
    atr_itf = _atr_from_tr(tr, 14)
    regime_atr_ok = atr_itf >= _median_last(tr, 100) * float(os.getenv("REGIME_ATR_MULT","0.9"))

    fcap = os.getenv("REGIME_FUNDING_ABS","") or os.getenv("FUNDING_ABS_CAP","")
    funding_ok = True
    if fcap:
        try:
            cap = float(fcap)
            f = abs(float(itf.get("funding", 0.0)))
            funding_ok = f <= cap / 100.0
        except Exception:
            funding_ok = True

    return bool(trend and regime_atr_ok and funding_ok)

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
# Combo A (HTF S/D + BOS/CHOCH → ITF B&R → LTF trigger)
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
    _dbg_kv(f"[A] start {symbol}", {
        "have_htf": isinstance(htf_map, dict),
        "have_itf": isinstance(itf_setup, dict),
        "ltf_ok": _safe_arrays(ltf_snapshot, ["o","h","l","c","ema9"])
    })
    if not isinstance(itf_setup, dict) or not isinstance(htf_map, dict):
        _dbg("[A] missing itf/htf"); return None
    if not _safe_arrays(ltf_snapshot, ["o","h","l","c","ema9"]):
        _dbg("[A] ltf arrays not safe"); return None
    if not _regime_ok(itf_setup, ltf_snapshot):
        _dbg("[A] regime not ok"); return None

    # 1) HTF zone (selected on ITF pass) & directional side
    zone = itf_setup.get("zone")      # expected (lo, hi)
    side = itf_setup.get("side")      # "long" or "short"
    if (not isinstance(zone, (list,tuple)) or len(zone) != 2) or side not in ("long","short"):
        _dbg("[A] missing zone/side"); return None
    zlo, zhi = float(zone[0]), float(zone[1])
    z_mid = 0.5 * (zlo + zhi); z_h = max(zhi - zlo, 1e-9)

    c_itf = float(itf_setup.get("price") or 0.0) or float(ltf_snapshot["c"][-2])
    touch_tol = float(os.getenv("A_SD_TOUCH_TOL","0.25"))
    near_zone = abs(c_itf - z_mid) <= touch_tol * z_h
    _dbg_kv(f"[A] HTF/zone {symbol}", {
        "side": side, "zone": (zlo, zhi), "c_itf": c_itf, "z_mid": z_mid,
        "touch_tol": touch_tol, "near_zone": near_zone
    })

    # 2) ITF break & retest quality (in ATR units)
    atr_itf = _get_atr_from_itf(itf_setup)
    if atr_itf <= 0:
        _dbg("[A] atr_itf <= 0"); return None
    bos_req = float(os.getenv("A_BOS_MIN_ATR","0.8"))
    broke_ok = bool(itf_setup.get("broke_level", False))
    broke_dist = float(itf_setup.get("break_close_dist_atr", 0.0))
    retest_tol_atr = float(os.getenv("A_RETEST_TOL","0.30"))
    retest_ok = bool(itf_setup.get("retested", False))
    retest_dist_atr = float(itf_setup.get("retest_dist_atr", 1e9))
    _dbg_kv(f"[A] ITF break/retest {symbol}", {
        "bos_req": bos_req, "broke_ok": broke_ok, "broke_dist_atr": broke_dist,
        "retest_ok": retest_ok, "retest_dist_atr": retest_dist_atr, "retest_tol_atr": retest_tol_atr
    })
    if not (broke_ok and broke_dist >= bos_req):
        _dbg("[A] BOS condition failed"); return None
    if not (retest_ok and retest_dist_atr <= retest_tol_atr):
        _dbg("[A] Retest condition failed"); return None

    # 3) LTF micro trigger with body/“sweep” constraints
    o = ltf_snapshot["o"]; h = ltf_snapshot["h"]; l = ltf_snapshot["l"]; c = ltf_snapshot["c"]; ema9 = ltf_snapshot["ema9"]
    tr_ltf = _true_range(o,h,l,c)
    body = abs(float(c[-2]) - float(o[-2])); rng = max(float(tr_ltf[-2]), 1e-9)
    body_ratio = body / rng
    min_body = float(os.getenv("A_LTF_BODY_MIN","0.60"))
    wick_req = float(os.getenv("A_LTF_SWIPE_WICK","0.35"))

    if side == "long":
        swept = (float(l[-2]) < float(l[-3])) and (float(c[-2]) > float(ema9[-2]))
        wick_ok = (float(h[-2]) - float(c[-2])) / max(rng,1e-9) <= (1 - wick_req)
        trigger_ok = swept and body_ratio >= min_body and wick_ok
    else:
        swept = (float(h[-2]) > float(h[-3])) and (float(c[-2]) < float(ema9[-2]))
        wick_ok = (float(c[-2]) - float(l[-2])) / max(rng,1e-9) <= (1 - wick_req)
        trigger_ok = swept and body_ratio >= min_body and wick_ok

    _dbg_kv(f"[A] LTF trigger {symbol}", {
        "body_ratio": round(body_ratio,3), "min_body": min_body,
        "wick_req": wick_req, "swept": swept, "trigger_ok": trigger_ok, "near_zone": near_zone
    })
    if not (near_zone and trigger_ok):
        _dbg("[A] LTF trigger or near_zone failed"); return None

    price = float(c[-2])
    sl, tp = _atr_brackets(price, atr_itf, side, sl_mult, tp_mult)
    _dbg_kv(f"[A] DECISION {symbol}", {"price": price, "sl": sl, "tp": tp, "side": side})
    return Decision(
        symbol=symbol, side=side, entry_type="market",
        size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.60,
        reason={"combo":"A","zone_touch":round(abs(c_itf - z_mid)/max(z_h,1e-9),3),
                "bos_atr":round(broke_dist,2),"retest_atr":round(retest_dist_atr,2),
                "ltf_body":round(body_ratio,2)},
        valid_until=time.time()+60
    )

# =========================
# Combo C1 (Trendline Break + EMA9 Pullback)
# =========================

def decide_combo_C1(symbol: str, htf_map: Dict, itf_setup: Optional[Dict], ltf_snapshot: Dict,
                    size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    _dbg_kv(f"[C1] start {symbol}", {
        "ltf_ok": _safe_arrays(ltf_snapshot, ["ts","o","h","l","c","ema9"])
    })
    if not _safe_arrays(ltf_snapshot, ["ts","o","h","l","c","ema9"]):
        _dbg("[C1] ltf arrays not safe"); return None
    if not _regime_ok(itf_setup or {}, ltf_snapshot):
        _dbg("[C1] regime not ok"); return None

    ts = ltf_snapshot["ts"]; o = ltf_snapshot["o"]; h = ltf_snapshot["h"]; l = ltf_snapshot["l"]; c = ltf_snapshot["c"]; ema9 = ltf_snapshot["ema9"]
    if len(c) < 220: 
        _dbg("[C1] not enough candles"); 
        return None

    # Trend via EMAs
    ema50 = _ema(c, 50); ema200 = _ema(c, 200)
    if len(ema50) < len(c) or len(ema200) < len(c): 
        _dbg("[C1] EMA arrays too short"); 
        return None
    up = (ema50[-2] > ema200[-2] and ema50[-2] > ema50[-6])
    dn = (ema50[-2] < ema200[-2] and ema50[-2] < ema50[-6])

    highs, lows = _fractals_swings(ts, h, l, lookback=140)
    min_sw = int(os.getenv("C1_MIN_SWINGS","4"))
    tr = _true_range(o,h,l,c); atr_ltf = _atr_from_tr(tr, 14)
    break_buf_atr = float(os.getenv("C1_BREAK_BUF_ATR","0.6"))
    prox_tol = float(os.getenv("C1_PULLBACK_PROX","0.004"))
    min_body = float(os.getenv("C1_MIN_BODY","0.55"))
    vol_mult = float(os.getenv("C1_VOL_MULT","1.20"))

    _dbg_kv(f"[C1] trend {symbol}", {
        "up": up, "dn": dn, "len_highs": len(highs), "len_lows": len(lows),
        "min_sw": min_sw
    })
    _dbg_kv(f"[C1] params {symbol}", {
        "break_buf_atr": break_buf_atr, "prox_tol": prox_tol,
        "min_body": min_body, "vol_mult": vol_mult
    })

    # LONG
    if up and len(lows) >= min_sw:
        swings = lows
        line = _fit_line(swings)
        if not line: 
            _dbg("[C1] long: cannot fit line"); 
            return None
        line_y = _line_value(line, len(swings)-1)
        broke = float(c[-2]) > max(line_y, float(c[-3]))
        broke_ok = broke and (float(c[-2]) - float(lows[-1][1])) >= break_buf_atr * max(atr_ltf, 1e-9)

        prox = abs(float(c[-2]) - float(ema9[-2])) / max(1e-9, float(c[-2]))
        body = abs(float(c[-2])-float(o[-2])); rng = max(float(tr[-2]),1e-9)
        body_ok = (body / rng) >= min_body

        vol_ok = True
        if "v" in ltf_snapshot:
            vol_ok = float(ltf_snapshot["v"][-2]) >= _median_last(ltf_snapshot["v"], 50) * vol_mult

        _dbg_kv(f"[C1] long check {symbol}", {
            "broke_ok": broke_ok, "prox": round(prox,5), "body_ok": body_ok, "vol_ok": vol_ok
        })
        if broke_ok and prox <= prox_tol and float(c[-2]) > float(o[-2]) and body_ok and vol_ok:
            price = float(c[-2])
            atr_itf = _get_atr_from_itf(itf_setup) or atr_ltf
            sl, tp = _atr_brackets(price, atr_itf, "long", sl_mult, tp_mult)
            _dbg_kv(f"[C1] DECISION long {symbol}", {"price": price, "sl": sl, "tp": tp})
            return Decision(symbol=symbol, side="long", entry_type="market",
                            size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.59,
                            reason={"combo":"C1","dir":"up","break_buf_atr":break_buf_atr}, valid_until=time.time()+60)

    # SHORT
    if dn and len(highs) >= min_sw:
        swings = highs
        line = _fit_line(swings)
        if not line: 
            _dbg("[C1] short: cannot fit line"); 
            return None
        line_y = _line_value(line, len(swings)-1)
        broke = float(c[-2]) < min(line_y, float(c[-3]))
        broke_ok = (float(highs[-1][1]) - float(c[-2])) >= break_buf_atr * max(atr_ltf, 1e-9)

        prox = abs(float(c[-2]) - float(ema9[-2])) / max(1e-9, float(c[-2]))
        body = abs(float(c[-2])-float(o[-2])); rng = max(float(tr[-2]),1e-9)
        body_ok = (body / rng) >= min_body

        vol_ok = True
        if "v" in ltf_snapshot:
            vol_ok = float(ltf_snapshot["v"][-2]) >= _median_last(ltf_snapshot["v"], 50) * vol_mult

        _dbg_kv(f"[C1] short check {symbol}", {
            "broke_ok": broke_ok, "prox": round(prox,5), "body_ok": body_ok, "vol_ok": vol_ok
        })
        if broke_ok and prox <= prox_tol and float(c[-2]) < float(o[-2]) and body_ok and vol_ok:
            price = float(c[-2])
            atr_itf = _get_atr_from_itf(itf_setup) or atr_ltf
            sl, tp = _atr_brackets(price, atr_itf, "short", sl_mult, tp_mult)
            _dbg_kv(f"[C1] DECISION short {symbol}", {"price": price, "sl": sl, "tp": tp})
            return Decision(symbol=symbol, side="short", entry_type="market",
                            size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.59,
                            reason={"combo":"C1","dir":"down","break_buf_atr":break_buf_atr}, valid_until=time.time()+60)

    return None

# =========================
# Combo C2 (Consolidation Breakout + ATR Expansion)
# =========================

def decide_combo_C2(symbol: str, htf_map: Dict, itf_setup: Optional[Dict], ltf_snapshot: Dict,
                    size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    _dbg_kv(f"[C2] start {symbol}", {
        "ltf_ok": _safe_arrays(ltf_snapshot, ["ts","o","h","l","c"]),
        "itf_ok": _safe_arrays(itf_setup or {}, ["o","h","l","c"])
    })
    if not _safe_arrays(ltf_snapshot, ["ts","o","h","l","c"]):
        _dbg("[C2] ltf arrays not safe"); return None
    if not _safe_arrays(itf_setup or {}, ["o","h","l","c"]):
        _dbg("[C2] itf arrays not safe"); return None
    if not _regime_ok(itf_setup or {}, ltf_snapshot):
        _dbg("[C2] regime not ok"); return None

    # ITF box
    oI, hI, lI, cI = itf_setup["o"], itf_setup["h"], itf_setup["l"], itf_setup["c"]
    box_n = int(os.getenv("C2_BOX_BARS","36"))
    if len(cI) < box_n + 20: 
        _dbg("[C2] not enough ITF bars for box"); 
        return None
    hi = float(np.max(hI[-box_n:])); lo = float(np.min(lI[-box_n:]))
    box_h = hi - lo
    trI = _true_range(oI,hI,lI,cI)
    atrI = _atr_from_tr(trI, 14)
    if atrI <= 0: 
        _dbg("[C2] atrI <= 0"); 
        return None
    _dbg_kv(f"[C2] ITF box {symbol}", {
        "box_n": box_n, "hi": hi, "lo": lo, "box_h": box_h, "atrI": atrI,
        "box_atr_max": float(os.getenv("C2_BOX_ATR_MAX","0.70"))
    })
    if box_h > float(os.getenv("C2_BOX_ATR_MAX","0.70")) * atrI:
        _dbg("[C2] box too wide vs ATR"); 
        return None

    # LTF breakout + expansion
    o = ltf_snapshot["o"]; h = ltf_snapshot["h"]; l = ltf_snapshot["l"]; c = ltf_snapshot["c"]
    trL = _true_range(o,h,l,c)
    exp_ok = float(trL[-2]) >= _median_last(trL, 50) * float(os.getenv("C2_EXP_TR_MULT","1.15"))
    vol_ok = True
    if "v" in ltf_snapshot:
        vol_ok = float(ltf_snapshot["v"][-2]) >= _median_last(ltf_snapshot["v"], 50) * float(os.getenv("C2_VOL_MULT","1.25"))
    _dbg_kv(f"[C2] LTF expansion {symbol}", {
        "exp_ok": exp_ok, "vol_ok": vol_ok,
        "exp_mult": float(os.getenv("C2_EXP_TR_MULT","1.15")),
        "vol_mult": float(os.getenv("C2_VOL_MULT","1.25"))
    })

    buf_atr = float(os.getenv("C2_BREAK_BUF_ATR","0.7"))
    # Long break
    if float(c[-2]) > hi + buf_atr * atrI and exp_ok and vol_ok:
        tol = float(os.getenv("C2_RETEST_TOL","0.25"))
        retest_ok = abs(float(l[-1]) - hi) <= tol * atrI if len(l) >= 2 else True
        if retest_ok:
            price = float(c[-2]); side = "long"
            sl, tp = _atr_brackets(price, atrI, side, sl_mult, tp_mult)
            _dbg_kv(f"[C2] DECISION long {symbol}", {"price": price, "sl": sl, "tp": tp})
            return Decision(symbol=symbol, side=side, entry_type="market",
                            size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.60,
                            reason={"combo":"C2","dir":"up","box_h_atr":round(box_h/atrI,2)}, valid_until=time.time()+60)

    # Short break
    if float(c[-2]) < lo - buf_atr * atrI and exp_ok and vol_ok:
        tol = float(os.getenv("C2_RETEST_TOL","0.25"))
        retest_ok = abs(float(h[-1]) - lo) <= tol * atrI if len(h) >= 2 else True
        if retest_ok:
            price = float(c[-2]); side = "short"
            sl, tp = _atr_brackets(price, atrI, side, sl_mult, tp_mult)
            _dbg_kv(f"[C2] DECISION short {symbol}", {"price": price, "sl": sl, "tp": tp})
            return Decision(symbol=symbol, side=side, entry_type="market",
                            size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.60,
                            reason={"combo":"C2","dir":"down","box_h_atr":round(box_h/atrI,2)}, valid_until=time.time()+60)

    return None

# =========================
# Standalone BR (Break & Retest on ITF)
# =========================

def decide_standalone_BR(symbol: str, htf_map: Dict, itf_setup: Optional[Dict], ltf_snapshot: Dict,
                         size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    _dbg_kv(f"[BR] start {symbol}", {
        "atr": _get_atr_from_itf(itf_setup),
        "ltf_ok": _safe_arrays(ltf_snapshot, ["ts","o","h","l","c"])
    })
    atr_val = _get_atr_from_itf(itf_setup)
    if atr_val <= 0:
        _dbg("[BR] atr <= 0"); return None
    if not _safe_arrays(ltf_snapshot, ["ts","o","h","l","c"]):
        _dbg("[BR] ltf arrays not safe"); return None

    ts = ltf_snapshot["ts"]; o = ltf_snapshot["o"]; h = ltf_snapshot["h"]; l = ltf_snapshot["l"]; c = ltf_snapshot["c"]
    if len(ts) < 160:
        _dbg("[BR] not enough candles"); 
        return None

    # Trend filter (EMA50 vs EMA200)
    ema50 = _ema(c, 50); ema200 = _ema(c, 200)
    if len(ema50) < len(c) or len(ema200) < len(c):
        _dbg("[BR] EMA arrays too short"); 
        return None
    trend_up = (ema50[-2] > ema200[-2] and ema50[-2] > ema50[-6])
    trend_dn = (ema50[-2] < ema200[-2] and ema50[-2] < ema50[-6])

    # parameters
    buf = float(os.getenv("BR_BREAK_BUFFER","0.0012") or 0.0012)      # 0.12%
    retest_tol = float(os.getenv("BR_RETEST_TOL","0.0025") or 0.0025) # 0.25%
    retest_bars = int(os.getenv("BR_RETEST_BARS","3") or 3)
    min_body_ratio = float(os.getenv("BR_MIN_BODY_RATIO","0.60") or 0.60)
    vol_mult = float(os.getenv("BR_VOL_MULT","1.15") or 1.15)

    _dbg_kv(f"[BR] params {symbol}", {
        "buf": buf, "retest_tol": retest_tol, "retest_bars": retest_bars,
        "min_body_ratio": min_body_ratio, "vol_mult": vol_mult
    })

    highs, lows = _fractals_swings(ts, h, l, lookback=120)
    if not highs and not lows:
        _dbg("[BR] no recent highs/lows swings"); 
        return None
    level_high = float(highs[-1][1]) if highs else None
    level_low  = float(lows[-1][1])  if lows  else None
    _dbg_kv(f"[BR] levels {symbol}", {"level_high": level_high, "level_low": level_low})

    v_med = None
    if "v" in ltf_snapshot:
        v_med = _median_last(ltf_snapshot["v"], 50)

    start_i = max(10, len(ts) - 16)
    for i in range(start_i, len(ts) - 2):
        # LONG
        if trend_up and level_high and float(c[i]) > level_high * (1.0 + buf):
            for j in range(1, retest_bars + 1):
                k = i + j
                if k >= len(ts) - 1: break
                near = abs(float(l[k]) - level_high) / max(level_high, 1e-9) < retest_tol
                body = abs(float(c[k]) - float(o[k])); rng = max(float(h[k]) - float(l[k]), 1e-9)
                strong = (body / rng) >= min_body_ratio and (float(c[k]) > float(o[k]))
                volpass = True
                if v_med is not None:
                    volpass = float(ltf_snapshot["v"][k]) > v_med * vol_mult
                _dbg_kv(f"[BR] long retest chk {symbol}", {
                    "near": near, "strong": strong, "volpass": volpass, "k": k
                })
                if near and strong and volpass:
                    price = float(c[k])
                    sl, tp = _atr_brackets(price, atr_val, "long", sl_mult, tp_mult)
                    _dbg_kv(f"[BR] DECISION long {symbol}", {"price": price, "sl": sl, "tp": tp})
                    return Decision(symbol=symbol, side="long", entry_type="market",
                                    size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.58,
                                    reason={"standalone":"BR","level":round(level_high,6),
                                            "break_i": i, "retest_i": k},
                                    valid_until=time.time()+60)
        # SHORT
        if trend_dn and level_low and float(c[i]) < level_low * (1.0 - buf):
            for j in range(1, retest_bars + 1):
                k = i + j
                if k >= len(ts) - 1: break
                near = abs(float(h[k]) - level_low) / max(level_low, 1e-9) < retest_tol
                body = abs(float(c[k]) - float(o[k])); rng = max(float(h[k]) - float(l[k]), 1e-9)
                strong = (body / rng) >= min_body_ratio and (float(c[k]) < float(o[k]))
                volpass = True
                if v_med is not None:
                    volpass = float(ltf_snapshot["v"][k]) > v_med * vol_mult
                _dbg_kv(f"[BR] short retest chk {symbol}", {
                    "near": near, "strong": strong, "volpass": volpass, "k": k
                })
                if near and strong and volpass:
                    price = float(c[k])
                    sl, tp = _atr_brackets(price, atr_val, "short", sl_mult, tp_mult)
                    _dbg_kv(f"[BR] DECISION short {symbol}", {"price": price, "sl": sl, "tp": tp})
                    return Decision(symbol=symbol, side="short", entry_type="market",
                                    size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.58,
                                    reason={"standalone":"BR","level":round(level_low,6),
                                            "break_i": i, "retest_i": k},
                                    valid_until=time.time()+60)
    return None

# =========================
# Standalone EMA9 (ITF/LTF pullback with filters)
# =========================

def decide_standalone_EMA9(symbol: str, htf_map: Dict, itf_setup: Optional[Dict], ltf_snapshot: Dict,
                           size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    _dbg_kv(f"[EMA9] start {symbol}", {
        "atr": _get_atr_from_itf(itf_setup),
        "ltf_ok": _safe_arrays(ltf_snapshot, ["o","h","l","c","ema9"])
    })
    atr_val = _get_atr_from_itf(itf_setup)
    if atr_val <= 0:
        _dbg("[EMA9] atr <= 0"); return None
    if not _safe_arrays(ltf_snapshot, ["o","h","l","c","ema9"]):
        _dbg("[EMA9] ltf arrays not safe"); return None
    c = ltf_snapshot["c"]; o = ltf_snapshot["o"]; h = ltf_snapshot["h"]; l = ltf_snapshot["l"]; ema9 = ltf_snapshot["ema9"]
    if len(c) < 200: 
        _dbg("[EMA9] not enough candles"); 
        return None

    # Trend filter on LTF: EMA50 vs EMA200 + slope
    ema50 = _ema(c, 50); ema200 = _ema(c, 200)
    if len(ema50) < len(c) or len(ema200) < len(c):
        _dbg("[EMA9] EMA arrays too short"); 
        return None
    trend_up = (ema50[-2] > ema200[-2]) and (ema50[-2] > ema50[-6]) and (ema200[-2] >= ema200[-6]*0.999)
    trend_dn = (ema50[-2] < ema200[-2]) and (ema50[-2] < ema50[-6]) and (ema200[-2] <= ema200[-6]*1.001)

    tol = float(os.getenv("EMA9_PROX_TOL","0.003") or 0.003)
    prox = abs(float(c[-2]) - float(ema9[-2])) / max(1e-9, float(c[-2]))
    body = abs(float(c[-2]) - float(o[-2])); tr = _true_range(o,h,l,c)
    rng = max(float(tr[-2]),1e-9)
    body_ratio = body / rng
    min_body = float(os.getenv("EMA9_MIN_BODY_RATIO","0.55") or 0.55)

    tr_ok = float(tr[-2]) > _median_last(tr, 30) * float(os.getenv("EMA9_TR_MULT","1.0") or 1.0)
    v_ok = True
    if "v" in ltf_snapshot:
        v = ltf_snapshot["v"]
        v_ok = float(v[-2]) > _median_last(v, 30) * float(os.getenv("EMA9_VOL_MULT","1.1") or 1.1)

    _dbg_kv(f"[EMA9] trend {symbol}", {
        "trend_up": trend_up, "trend_dn": trend_dn
    })
    _dbg_kv(f"[EMA9] filters {symbol}", {
        "prox": prox, "tol": tol, "body_ratio": round(body_ratio,3), "min_body": min_body,
        "tr_ok": tr_ok, "v_ok": v_ok
    })

    # Long
    if trend_up and prox < tol and float(c[-2]) > float(o[-2]) and body_ratio >= min_body and tr_ok and v_ok:
        price = float(c[-2])
        br = _compute_brackets(price, "long", itf_setup, ltf_snapshot)
        if not br: 
            _dbg("[EMA9] brackets not available (long)"); 
            return None
        sl, tp = br
        _dbg_kv(f"[EMA9] DECISION {symbol}", {"side": "long", "price": price, "sl": sl, "tp": tp})
        return Decision(symbol=symbol, side="long", entry_type="market",
                        size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.56,
                        reason={"standalone":"EMA9","dir":"up","prox":prox,"body":round(body_ratio,2)},
                        valid_until=time.time()+60)

    # Short
    if trend_dn and prox < tol and float(c[-2]) < float(o[-2]) and body_ratio >= min_body and tr_ok and v_ok:
        price = float(c[-2])
        br = _compute_brackets(price, "short", itf_setup, ltf_snapshot)
        if not br: 
            _dbg("[EMA9] brackets not available (short)"); 
            return None
        sl, tp = br
        _dbg_kv(f"[EMA9] DECISION {symbol}", {"side": "short", "price": price, "sl": sl, "tp": tp})
        return Decision(symbol=symbol, side="short", entry_type="market",
                        size_usdt=size_usdt, sl=sl, tp=tp, confidence=0.56,
                        reason={"standalone":"EMA9","dir":"down","prox":prox,"body":round(body_ratio,2)},
                        valid_until=time.time()+60)

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
