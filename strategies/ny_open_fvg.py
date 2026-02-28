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
    v: np.ndarray,
    or_levels: Optional[Tuple[float, float]] = None,
    allow_outside_or: bool = True,
) -> Optional[Dict]:
    """
    Enhanced FVG entry with rejection requirement, OR break validation, and VOLUME confirmation:
      - Touch: candle whose range overlaps the FVG gap after FVG forms.
      - Rejection: candle must close outside the gap in the trade's direction.
        * For longs: close above the FVG gap (close > gap_hi) AND above OR high
        * For shorts: close below the FVG gap (close < gap_lo) AND below OR low
      - Volume: rejection candle must show volume expansion vs recent average
      - Entry: close of the rejection candle.
      - SL/TP: based on FVG gap boundaries and configured RR ratio.
    Returns dict {'side','fvg_i','touch_i','confirm_i','entry','sl','rr','tp','pattern','gap_size','vol_expansion'} or None.
    """
    if not fvgs or len(c) < 5 or len(v) < 5:
        return None

    # Get OR boundaries for validation
    orh, orl = (None, None)
    if or_levels is not None:
        orh, orl = or_levels

    # Volume filter configuration
    rejection_vol_mult = float(os.getenv("FVG_REJECTION_VOL_MULT", "1.3") or 1.3)
    formation_vol_mult = float(os.getenv("FVG_FORMATION_VOL_MULT", "0") or 0)  # 0 = disabled
    vol_lookback = int(os.getenv("FVG_VOL_LOOKBACK", "20") or 20)

    # CRITICAL FIX: Continue checking ALL FVGs and pick the MOST RECENT valid one
    # This ensures we don't lock onto an early invalid FVG and miss better setups
    candidates: List[Dict] = []

    for f in fvgs:
        side = f["side"]
        i0 = int(f["i"])
        gap_lo = float(f["lo"])
        gap_hi = float(f["hi"])
        gap_size = gap_hi - gap_lo

        # Optional: Check volume on FVG formation candle
        if formation_vol_mult > 0 and i0 >= vol_lookback:
            formation_vol = float(v[i0])
            avg_vol_at_formation = float(np.mean(v[max(0, i0 - vol_lookback):i0]))
            if avg_vol_at_formation > 0 and formation_vol < avg_vol_at_formation * formation_vol_mult:
                # FVG formed on weak volume - skip
                continue

        # If restricting by OR: accept FVG if it intersects the OR range, or allow outside if configured.
        if or_levels is not None and not allow_outside_or:
            if not _overlaps_interval(gap_lo, gap_hi, float(min(orh, orl)), float(max(orh, orl))):
                continue

        # Find rejection: candle that touches the FVG and closes outside it in the trade's direction
        # CRITICAL: Also validate entry is OUTSIDE the opening range (break requirement)
        # NEW: Also validate VOLUME EXPANSION on rejection candle
        rejection_i = None
        rejection_vol_expansion = 0.0

        for t in range(i0 + 1, len(c)):
            # Candle [t] must touch the FVG gap by overlap of [low, high] with [gap_lo, gap_hi]
            if _overlaps_interval(float(l[t]), float(h[t]), gap_lo, gap_hi):
                close_price = float(c[t])

                # Check rejection based on side WITH OR BREAK VALIDATION
                rejection_confirmed = False
                if side == "long":
                    # For longs: close must be above the FVG gap high AND above OR high
                    if close_price > gap_hi:
                        # Validate entry is OUTSIDE (above) the opening range
                        if orh is None or close_price > orh:
                            rejection_confirmed = True
                else:  # side == "short"
                    # For shorts: close must be below the FVG gap low AND below OR low
                    if close_price < gap_lo:
                        # Validate entry is OUTSIDE (below) the opening range
                        if orl is None or close_price < orl:
                            rejection_confirmed = True

                if rejection_confirmed:
                    # VOLUME VALIDATION: Rejection candle must show volume expansion
                    if t >= vol_lookback:
                        rejection_vol = float(v[t])
                        # Calculate average volume from recent bars (exclude current bar)
                        avg_vol = float(np.mean(v[max(0, t - vol_lookback):t]))

                        if avg_vol > 1e-9:  # Avoid division by zero
                            vol_expansion = rejection_vol / avg_vol

                            # Check if volume expansion meets threshold
                            if vol_expansion >= rejection_vol_mult:
                                rejection_i = t
                                rejection_vol_expansion = vol_expansion
                                break
                            # else: volume too weak, continue searching

        if rejection_i is None:
            continue

        # Entry at close of rejection candle
        entry = float(c[rejection_i])

        # Pattern indicates rejection confirmation with OR break
        pattern = "fvg_rejection_break"

        # SL/TP calculation based on FVG gap boundaries and RR ratio
        rr_cfg = float(os.getenv("FVG_RR", "2.0") or 2.0)

        if side == "long":
            # For longs: SL below FVG gap low, TP based on RR
            sl = gap_lo
            risk = max(1e-9, entry - sl)
            tp = entry + rr_cfg * risk
        else:
            # For shorts: SL above FVG gap high, TP based on RR
            sl = gap_hi
            risk = max(1e-9, sl - entry)
            tp = entry - rr_cfg * risk

        candidates.append({
            "side": side,
            "fvg_i": i0,
            "touch_i": rejection_i,
            "confirm_i": rejection_i,  # Same as rejection for compatibility
            "entry": entry,
            "sl": float(sl),
            "tp": float(tp),
            "rr": float(rr_cfg),
            "pattern": pattern,
            "gap_size": float(gap_size),
            "gap_lo": gap_lo,
            "gap_hi": gap_hi,
            "vol_expansion": float(rejection_vol_expansion),
            "rejection_vol_mult_required": float(rejection_vol_mult),
        })

    # Pick the MOST RECENT valid rejected FVG (latest confirmation index)
    # This ensures we use the freshest setup, not the first one found
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x["confirm_i"])[-1]


class NyOpenFVGInspector:
    """
    Stateless analyzer for OR + FVG touch/confirm on configurable timeframes. Intended for logging and live testing.
    """

    def __init__(self,
                 or_start_hhmm_utc: str = None,
                 or_minutes: int = None,
                 or_timeframe: str = None,
                 fvg_timeframe: str = None,
                 allow_outside_or: bool = None,
                 require_break_within: Optional[int] = None):
        self.or_start_hhmm_utc = or_start_hhmm_utc or os.getenv("OR_START_UTC", "14:30")
        self.or_minutes = int(or_minutes or int(os.getenv("OR_MINUTES", "15") or 15))
        self.or_timeframe = or_timeframe or os.getenv("OR_TIMEFRAME", "15m")
        self.fvg_timeframe = fvg_timeframe or os.getenv("FVG_TIMEFRAME", "5m")
        self.allow_outside_or = bool(allow_outside_or if allow_outside_or is not None
                                     else _env_bool("FVG_ALLOW_OUTSIDE", True))
        self.require_break_within = require_break_within
        # Scan window around OR for FVG detection (in candles of fvg_timeframe)
        self.scan_pre_min = int(os.getenv("FVG_SCAN_PRE_OR_MIN", "0") or 0)
        self.scan_post_min = int(os.getenv("FVG_SCAN_POST_OR_MIN", "60") or 60)

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string (e.g., '5m', '15m', '1h') to minutes."""
        try:
            if timeframe.endswith('m'):
                return int(timeframe[:-1])
            elif timeframe.endswith('h'):
                return int(timeframe[:-1]) * 60
            elif timeframe.endswith('d'):
                return int(timeframe[:-1]) * 1440
            else:
                return 5  # default to 5 minutes
        except Exception:
            return 5

    def analyze_symbol(self, ex, symbol: str, limit: int = 500) -> Optional[Dict]:
        """
        Fetch candles using FVG timeframe (e.g., 5m), compute OR (by UTC anchor), detect FVGs,
        and determine touch/confirm signal. Trading is only allowed after the first candle
        following OR is fully formed.
        Returns a payload dict with diagnostics for logging.
        """
        ohlcv = fetch_candles(ex, symbol, timeframe=self.fvg_timeframe, limit=limit)
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
            # Derive indices for OR window: [or_open_ms, or_close_ms)
            or_open_ms = float(start_epoch) * 1000.0
            or_close_ms = float(start_epoch + 60 * self.or_minutes) * 1000.0
            # index of first bar at or after OR open
            or_open_i = next((i for i in range(len(ts)) if ts[i] >= or_open_ms), None)
            # index of first bar at or after OR close (this is the first post-OR candle)
            or_close_i = next((i for i in range(len(ts)) if ts[i] >= or_close_ms), None)
            # First candle after OR is formed if we have progressed at least one bar beyond it
            if or_open_i is not None and or_close_i is not None and (or_close_i - or_open_i) >= 1:
                if or_close_i < len(ts) - 1:
                    post_first_candle_i = or_close_i
                    or_ready = True

        # Detect FVGs in a configurable window around OR (pre/post minutes converted to bars)
        fvgs: List[Dict] = []
        if or_ready and or_open_i is not None and or_close_i is not None:
            # Convert minutes to bars based on FVG timeframe
            tf_minutes = self._timeframe_to_minutes(self.fvg_timeframe)
            pre_bars = int(max(0, self.scan_pre_min // tf_minutes)) if tf_minutes > 0 else 0
            post_bars = int(max(0, self.scan_post_min // tf_minutes)) if tf_minutes > 0 else 12
            scan_start = max(0, int(or_open_i) - pre_bars)
            scan_end = min(len(c), int(or_close_i) + post_bars)
            if scan_end > scan_start:
                h_scan = h[scan_start:scan_end]
                l_scan = l[scan_start:scan_end]
                in_scan = detect_fvgs(h_scan, l_scan, lookback=len(h_scan))
                for f in in_scan:
                    fvgs.append({
                        "i": int(f["i"]) + int(scan_start),  # remap to full-series index
                        "side": f["side"],
                        "lo": f["lo"],
                        "hi": f["hi"],
                    })

        # Build signal only when OR is ready and confirmation occurs after the first post-OR bar
        signal = None
        if or_ready and fvgs:
            cand = find_touch_and_confirm(
                fvgs, o, h, l, c, v,
                or_levels=(orh, orl),
                allow_outside_or=self.allow_outside_or,
            )
            if cand is not None and post_first_candle_i is not None:
                if int(cand.get("confirm_i", -1)) > int(post_first_candle_i):
                    signal = cand

        # Add FVG summary statistics
        fvg_summary = None
        if fvgs:
            long_fvgs = [f for f in fvgs if f["side"] == "long"]
            short_fvgs = [f for f in fvgs if f["side"] == "short"]
            fvg_summary = {
                "total": len(fvgs),
                "long": len(long_fvgs),
                "short": len(short_fvgs),
                "avg_gap_size": sum(f["hi"] - f["lo"] for f in fvgs) / len(fvgs) if fvgs else 0,
            }

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
            "fvg_summary": fvg_summary,
            "signal": signal,
            "last_ts": int(ts[-1]) if len(ts) else None,
            # arrays for human-readable bar details
            "ts": ts,
            "o": o, "h": h, "l": l, "c": c,
        }
        return payload
    # ... existing code ...

    def log_payload(self, payload: Dict, debug: bool = True):
        """Send a comprehensive informative log to Telegram with enhanced FVG diagnostics."""
        sym = payload["symbol"]
        or_high = payload.get("or_high")
        or_low = payload.get("or_low")

        or_info = "OR: pending"
        if payload["or_ready"] and or_high is not None and or_low is not None:
            or_range = or_high - or_low
            or_info = f"ORH={or_high:.2f} ORL={or_low:.2f} (range: ${or_range:.2f})"

        last_fvg = payload.get("last_fvg")
        fvg_info = "FVGs=0"
        if last_fvg:
            fvg_gap = last_fvg['hi'] - last_fvg['lo']
            fvg_info = f"FVGs={payload['fvgs_found']} | Last[{last_fvg['side']}]: ${last_fvg['lo']:.2f}-${last_fvg['hi']:.2f} (gap: ${fvg_gap:.2f})"

        sig = payload.get("signal")

        # In production mode (debug=False), only show OR updates and heartbeat without FVG details
        if not debug and not sig:
            tg(f"📊 NY-Open FVG {sym} | {or_info} | Monitoring active")
            return

        if sig:
            entry = sig['entry']
            sl = sig['sl']
            tp = sig['tp']
            side = sig['side']
            gap_size = sig.get('gap_size', 0)
            gap_lo = sig.get('gap_lo', 0)
            gap_hi = sig.get('gap_hi', 0)

            # Calculate key metrics
            risk_usd = abs(entry - sl)
            reward_usd = abs(tp - entry)

            # Validate entry is outside OR
            entry_position = "INVALID"
            if or_high is not None and or_low is not None:
                if side == "long":
                    if entry > or_high:
                        entry_position = f"✅ ABOVE OR (${entry - or_high:.2f} above ORH)"
                    else:
                        entry_position = f"❌ INSIDE OR (${or_high - entry:.2f} below ORH)"
                else:  # short
                    if entry < or_low:
                        entry_position = f"✅ BELOW OR (${or_low - entry:.2f} below ORL)"
                    else:
                        entry_position = f"❌ INSIDE OR (${entry - or_low:.2f} above ORL)"

            vol_expansion = sig.get('vol_expansion', 0)
            vol_mult_required = sig.get('rejection_vol_mult_required', 0)
            vol_info = f"\n📊 Volume: {vol_expansion:.2f}x avg (required: {vol_mult_required:.2f}x)" if vol_expansion > 0 else ""

            br = f"Entry: ${entry:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f}\nRisk: ${risk_usd:.2f} | Reward: ${reward_usd:.2f} | RR: {sig['rr']:.1f}:1{vol_info}"

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
                        return f"{iso(ts_arr[i])} | O={float(o_arr[i]):.2f} H={float(h_arr[i]):.2f} L={float(l_arr[i]):.2f} C={float(c_arr[i]):.2f}"
                    return "unavailable"
                except Exception:
                    return "unavailable"

            i_f = int(sig.get("fvg_i", -1))
            i_t = int(sig.get("touch_i", -1))

            # Calculate FVG age (bars between formation and rejection)
            fvg_age = i_t - i_f if (i_f >= 0 and i_t >= 0) else 0

            fvg_bar = bar_line(i_f)
            touch_bar = bar_line(i_t)
            bars_block = f"FVG Formation:   {fvg_bar}\nRejection/Entry: {touch_bar}\nFVG Age: {fvg_age} bars ({fvg_age} minutes on {self.fvg_timeframe})"

            tg(
                f"🎯 NY-Open FVG {sym} [{side.upper()}]\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 {or_info}\n"
                f"📈 FVG Gap: ${gap_lo:.2f}-${gap_hi:.2f} (size: ${gap_size:.2f})\n"
                f"📍 {entry_position}\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ {sig['pattern'].upper()} CONFIRMED\n"
                f"{br}\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"{bars_block}"
            )
        else:
            # Enhanced monitoring log with more context
            last_close = payload.get("last_close")
            close_info = ""
            if last_close and or_high and or_low:
                if last_close > or_high:
                    close_info = f"| Price: ${last_close:.2f} (${last_close - or_high:.2f} above OR)"
                elif last_close < or_low:
                    close_info = f"| Price: ${last_close:.2f} (${or_low - last_close:.2f} below OR)"
                else:
                    close_info = f"| Price: ${last_close:.2f} (inside OR)"

            tg(f"📊 NY-Open FVG {sym} | {or_info} | {fvg_info} {close_info} | Waiting for rejection")