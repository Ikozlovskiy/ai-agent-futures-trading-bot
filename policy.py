import time
import os
import numpy as np
from typing import Optional, Dict, Tuple
from models import Decision

def _atr_brackets(price: float, atr_val: float, side: str, sl_mult: float, tp_mult: float):
    if side == "long":
        sl = price - sl_mult * atr_val
        tp = price + tp_mult * atr_val
    else:
        sl = price + sl_mult * atr_val
        tp = price - tp_mult * atr_val
    return float(sl), float(tp)

# ====== Combo A ======
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

# ====== Combo B ======
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
    if not htf_map or not itf_setup:
        return None
    from datahub import utc_anchor_for_session
    anchor = utc_anchor_for_session(os.getenv("SESSION_START_UTC", "08:00"))
    orb = _orb_break(ltf_snapshot, anchor, int(os.getenv("ORB_MINUTES", "15")))
    if not orb:
        return None
    side, trigger = orb

    vwap_arr = htf_map.get("vwap")
    bias_ok = True
    if isinstance(vwap_arr, np.ndarray) and not np.isnan(vwap_arr[-2]):
        px = float(ltf_snapshot["c"][-2]); v = float(vwap_arr[-2])
        bias_ok = (px >= v) if side == "long" else (px <= v)
    if not bias_ok:
        return None

    atr_val = float(itf_setup.get("atr") or 0.0)
    if atr_val <= 0:
        return None
    price = float(ltf_snapshot["c"][-2])
    sl, tp = _atr_brackets(price, atr_val, side, sl_mult, tp_mult)

    box = itf_setup.get("box")
    conf = 0.55
    if box:
        box_h = box[1] - box[0]
        conf = min(0.95, max(0.5, (box_h / max(atr_val, 1e-9)) * 0.2 + 0.5))
    if isinstance(vwap_arr, np.ndarray) and not np.isnan(vwap_arr[-2]):
        conf = min(0.99, conf + 0.1 * (abs(price - float(vwap_arr[-2])) / max(price, 1e-9)))

    return Decision(
        symbol=symbol,
        side=side,
        entry_type="market",
        size_usdt=size_usdt,
        sl=sl,
        tp=tp,
        confidence=round(conf, 3),
        reason={"htf":"Anchored VWAP/value", "itf":"15m consolidation", "ltf":"ORB break"},
        valid_until=time.time() + 60
    )

# ====== Router ======
def decide(symbol: str, combo: str, htf_map: Dict, itf_setup: Dict, ltf_snapshot: Dict,
           size_usdt: float, sl_mult: float, tp_mult: float) -> Optional[Decision]:
    if combo.upper() == "A":
        return decide_combo_A(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    if combo.upper() == "B":
        return decide_combo_B(symbol, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
    return None
