import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from datetime import datetime, timezone

from utils import tg
from datahub import fetch_candles, opening_range, utc_anchor_for_session


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _pct(x: float, y: float) -> float:
    if y == 0:
        return 0.0
    return (x / y - 1.0) * 100.0


def detect_fvgs(high: np.ndarray, low: np.ndarray, lookback: int = 400) -> List[Dict]:
    """
    Classic ICT FVG on 1m:
      - Bullish FVG at i if low[i] > high[i-2], gap = [high[i-2], low[i]]
      - Bearish FVG at i if high[i] < low[i-2], gap = [high[i], low[i-2]]
    Returns a list of dicts with keys: {'i','side','lo','hi'}
    """
    n = len(high)
    out: List[Dict] = []
    start = max(2, n - lookback)
    for i in range(start, n):
        try:
            # bullish
            if low[i] > high[i - 2]:
                gap_lo = float(high[i - 2])
                gap_hi = float(low[i])
                out.append({"i": i, "side": "long", "lo": gap_lo, "hi": gap_hi})
            # bearish
            if high[i] < low[i - 2]:
                gap_lo = float(high[i])      # lower bound of gap
                gap_hi = float(low[i - 2])   # upper bound of gap
                out.append({"i": i, "side": "short", "lo": gap_lo, "hi": gap_hi})
        except Exception:
            continue
    return out


def _overlaps_interval(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    return not (hi1 < lo2 or hi2 < lo1)


def find_touch_and_confirm(
    fvgs: List[Dict],
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    or_levels: Optional[Tuple[float, float]] = None,
    require_break_within: Optional[int] = None,
    allow_outside_or: bool = True,
) -> Optional[Dict]:
    """
    From detected FVGs, find the latest side-specific setup:
      - Touch: first candle whose range overlaps the FVG gap after FVG forms.
      - Confirm: next candle closes above (long) / below (short) the touching candle high/low.
      - Optional: if confirm occurs before OR break, require break within N bars else ignore.
    Returns dict {'side','fvg_i','touch_i','confirm_i','entry','sl','rr','tp','pattern'} or None.
    """
    if not fvgs or len(c) < 5:
        return None

    # Use only the most recent FVG per side to reduce noise.
    by_side = {"long": None, "short": None}
    for f in fvgs:
        s = f["side"]
        if by_side[s] is None or f["i"] > by_side[s]["i"]:
            by_side[s] = f

    candidates: List[Dict] = []
    for side in ("long", "short"):
        f = by_side.get(side)
        if not f:
            continue
        i0 = int(f["i"])
        gap_lo = float(f["lo"])
        gap_hi = float(f["hi"])

        # If restricting by OR: accept FVG if it intersects the OR range, or allow outside if configured.
        if or_levels is not None and not allow_outside_or:
            orh, orl = or_levels
            if not _overlaps_interval(gap_lo, gap_hi, float(min(orh, orl)), float(max(orh, orl))):
                continue

        touch_i = None
        for t in range(i0 + 1, len(c) - 1):
            # Candle [t] touches the FVG gap by overlap of [low, high] with [gap_lo, gap_hi]
            if _overlaps_interval(float(l[t]), float(h[t]), gap_lo, gap_hi):
                touch_i = t
                break
        if touch_i is None:
            continue

        # Confirmation: next candle closes above/below the touching candle's extreme
        confirm_i = None
        if side == "long":
            trig = float(h[touch_i])
            for k in range(touch_i + 1, len(c)):
                if float(c[k]) > trig:
                    confirm_i = k
                    break
        else:
            trig = float(l[touch_i])
            for k in range(touch_i + 1, len(c)):
                if float(c[k]) < trig:
                    confirm_i = k
                    break
        if confirm_i is None:
            continue

        # OR break logic classification
        pattern = "unknown"
        broke_before_confirm = False
        broke_after_confirm = False
        if or_levels is not None:
            orh, orl = or_levels
            # first break index after FVG forms
            brk_idx = None
            if side == "long":
                for j in range(i0, len(c)):
                    if float(c[j]) > float(orh):
                        brk_idx = j
                        break
            else:
                for j in range(i0, len(c)):
                    if float(c[j]) < float(orl):
                        brk_idx = j
                        break
            if brk_idx is not None:
                if brk_idx <= touch_i:
                    broke_before_confirm = True
                elif brk_idx > confirm_i:
                    broke_after_confirm = True

            # If confirm occurs before OR break, optionally require break within N bars
            if require_break_within is not None and not broke_before_confirm:
                ok = False
                for j in range(confirm_i + 1, min(confirm_i + 1 + require_break_within, len(c))):
                    if (side == "long" and float(c[j]) > float(orh)) or (side == "short" and float(c[j]) < float(orl)):
                        ok = True
                        break
                if not ok:
                    continue

            if broke_before_confirm:
                pattern = "break→retrace→confirm"
            else:
                pattern = "touch→confirm→break"

        # Entry/SL/TP preview according to spec (SL on touch candle ± 10% of its range)
        entry = float(c[confirm_i])
        rng_touch = max(1e-9, float(h[touch_i]) - float(l[touch_i]))
        buf = float(os.getenv("FVG_SL_TOUCH_BUF_PCT", "0.10") or 0.10)  # 10% candle range buffer
        if side == "long":
            sl = float(l[touch_i]) - buf * rng_touch
            risk = max(1e-9, entry - sl)
            rr = float(os.getenv("FVG_RR", "2.0") or 2.0)
            tp = entry + rr * risk
        else:
            sl = float(h[touch_i]) + buf * rng_touch
            risk = max(1e-9, sl - entry)
            rr = float(os.getenv("FVG_RR", "2.0") or 2.0)
            tp = entry - rr * risk

        candidates.append({
            "side": side,
            "fvg_i": i0,
            "touch_i": touch_i,
            "confirm_i": confirm_i,
            "entry": entry,
            "sl": float(sl),
            "tp": float(tp),
            "rr": float(rr),
            "pattern": pattern,
        })

    # Pick the most recent confirmed candidate
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x["confirm_i"])[-1]


class NyOpenFVGInspector:
    """
    Stateless analyzer for OR + FVG touch/confirm on 1m. Intended for logging and live testing.
    """

    def __init__(self,
                 or_start_hhmm_utc: str = None,
                 or_minutes: int = None,
                 allow_outside_or: bool = None,
                 require_break_within: Optional[int] = None):
        self.or_start_hhmm_utc = or_start_hhmm_utc or os.getenv("OR_START_UTC", "14:30")
        self.or_minutes = int(or_minutes or int(os.getenv("OR_MINUTES", "5") or 5))
        self.allow_outside_or = bool(allow_outside_or if allow_outside_or is not None
                                     else _env_bool("FVG_ALLOW_OUTSIDE", True))
        self.require_break_within = require_break_within

    def analyze_symbol(self, ex, symbol: str, limit: int = 500) -> Optional[Dict]:
        """
        Fetch 1m candles, compute OR (by UTC anchor), detect FVGs strictly inside the OR window,
        and determine touch/confirm signal. Trading is only allowed after the first 1m candle
        following OR is fully formed.
        Returns a payload dict with diagnostics for logging.
        """
        ohlcv = fetch_candles(ex, symbol, timeframe="1m", limit=limit)
        arr = np.asarray(ohlcv, dtype=float)
        ts, o, h, l, c, v = arr.T

        # OR computation via UTC anchor (session/timezone integration to be added later)
        start_epoch = utc_anchor_for_session(self.or_start_hhmm_utc)
        or_res = opening_range(ts, h, l, start_epoch, self.or_minutes)
        orh, orl = (float("nan"), float("nan"))
        or_ready = False
        or_open_i = None
        or_close_i = None
        post_first_candle_i = None
        if or_res:
            orh, orl = or_res
            # Derive 1m indices for OR window: [or_open_ms, or_close_ms)
            or_open_ms = float(start_epoch) * 1000.0
            or_close_ms = float(start_epoch + 60 * self.or_minutes) * 1000.0
            # index of first bar at or after OR open
            or_open_i = next((i for i in range(len(ts)) if ts[i] >= or_open_ms), None)
            # index of first bar at or after OR close (this is the first post-OR candle)
            or_close_i = next((i for i in range(len(ts)) if ts[i] >= or_close_ms), None)
            # First candle after OR is formed if we have progressed at least one bar beyond it
            if or_open_i is not None and or_close_i is not None and (or_close_i - or_open_i) >= 2:
                if or_close_i < len(ts) - 1:
                    post_first_candle_i = or_close_i
                    or_ready = True

        # Detect FVGs only from bars that lie inside the OR window (no pre-OR FVGs)
        fvgs: List[Dict] = []
        if or_ready and or_open_i is not None and or_close_i is not None:
            h_or = h[or_open_i:or_close_i]
            l_or = l[or_open_i:or_close_i]
            in_or = detect_fvgs(h_or, l_or, lookback=len(h_or))
            for f in in_or:
                fvgs.append({
                    "i": int(f["i"]) + int(or_open_i),  # remap to full-series index
                    "side": f["side"],
                    "lo": f["lo"],
                    "hi": f["hi"],
                })

        # Build signal only when OR is ready and confirmation occurs after the first post-OR bar
        signal = None
        if or_ready and fvgs:
            cand = find_touch_and_confirm(
                fvgs, o, h, l, c,
                or_levels=(orh, orl),
                require_break_within=self.require_break_within,
                allow_outside_or=self.allow_outside_or,
            )
            if cand is not None and post_first_candle_i is not None:
                if int(cand.get("confirm_i", -1)) > int(post_first_candle_i):
                    signal = cand

        payload = {
            "symbol": symbol,
            "now": int(time.time()),
            "or_ready": bool(or_ready),
            "or_high": float(orh) if or_res else None,
            "or_low": float(orl) if or_res else None,
            "or_open_i": int(or_open_i) if or_open_i is not None else None,
            "or_close_i": int(or_close_i) if or_close_i is not None else None,
            "first_after_or_i": int(post_first_candle_i) if post_first_candle_i is not None else None,
            "last_close": float(c[-1]) if len(c) else None,
            "fvgs_found": len(fvgs),
            "last_fvg": fvgs[-1] if fvgs else None,
            "signal": signal,
            "last_ts": int(ts[-1]) if len(ts) else None,
            # arrays for human-readable bar details
            "ts": ts,
            "o": o, "h": h, "l": l, "c": c,
        }
        return payload
    # ... existing code ...

    def log_payload(self, payload: Dict):
        """Send a compact informative log to Telegram."""
        sym = payload["symbol"]
        or_info = "OR: pending"
        if payload["or_ready"]:
            or_info = f"ORH={payload['or_high']:.6f} ORL={payload['or_low']:.6f}"
        last_fvg = payload.get("last_fvg")
        fvg_info = "FVGs=0"
        if last_fvg:
            fvg_info = f"FVGs={payload['fvgs_found']} last[{last_fvg['side']}]: [{last_fvg['lo']:.6f},{last_fvg['hi']:.6f}]"
        sig = payload.get("signal")
        if sig:
            if sig["side"] == "long":
                br = f"entry={sig['entry']:.6f} sl={sig['sl']:.6f} tp={sig['tp']:.6f}"
            else:
                br = f"entry={sig['entry']:.6f} sl={sig['sl']:.6f} tp={sig['tp']:.6f}"

            # Human-readable bar details (UTC) with safe bounds (no raw indices)
            ts_arr = payload.get("ts")
            o_arr = payload.get("o")
            h_arr = payload.get("h")
            l_arr = payload.get("l")
            c_arr = payload.get("c")
            if ts_arr is None: ts_arr = []
            if o_arr is None: o_arr = []
            if h_arr is None: h_arr = []
            if l_arr is None: l_arr = []
            if c_arr is None: c_arr = []

            def iso(ms_val):
                try:
                    return datetime.fromtimestamp(float(ms_val)/1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                except Exception:
                    return "n/a"

            def bar_line(i: int) -> str:
                try:
                    if isinstance(i, (int, np.integer)) and 0 <= i < len(ts_arr):
                        return f"{iso(ts_arr[i])} | O={float(o_arr[i]):.6f} H={float(h_arr[i]):.6f} L={float(l_arr[i]):.6f} C={float(c_arr[i]):.6f}"
                    return "unavailable"
                except Exception:
                    return "unavailable"

            i_f = int(sig.get("fvg_i", -1))
            i_t = int(sig.get("touch_i", -1))
            i_c = int(sig.get("confirm_i", -1))

            fvg_bar = bar_line(i_f)
            touch_bar = bar_line(i_t)
            confirm_bar = bar_line(i_c)
            bars_block = f"FVG bar:    {fvg_bar}\nTouch bar:  {touch_bar}\nConfirm bar:{confirm_bar}"

            tg(
                f"📊 NY-Open FVG {sym}\n"
                f"{or_info}\n"
                f"{fvg_info}\n"
                f"✅ Confirmed [{sig['side'].upper()}] {sig['pattern']} (RR={sig['rr']})\n"
                f"{br}\n"
                f"{bars_block}"
            )
        else:
            tg(f"📊 NY-Open FVG {sym} | {or_info} | {fvg_info} | no confirm yet")