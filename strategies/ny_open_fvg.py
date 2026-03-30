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


def detect_fvgs(high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int = 400) -> List[Dict]:
    """
    Classic ICT FVG (Fair Value Gap) detection with ATR-based minimum gap size filter:

    FVG forms when there's a gap between candle i-2 and candle i, with candle i-1 in between.

    BEARISH FVG: Gap down movement
      - Condition: high[i] < low[i-2] (candle 3's high is below candle 1's low)
      - Gap zone: [high[i], low[i-2]] (between candle 3 high and candle 1 low)
      - Example: C1.low=2051.48, C2=middle, C3.high=2049.96 → gap=[2049.96, 2051.48]

    BULLISH FVG: Gap up movement
      - Condition: low[i] > high[i-2] (candle 3's low is above candle 1's high)
      - Gap zone: [high[i-2], low[i]] (between candle 1 high and candle 3 low)

    Returns a list of dicts with keys: {'i','side','lo','hi'}

    IMPORTANT: Gap must have minimum size to be valid (filters out insignificant gaps using ATR)
    """
    from datahub import atr as calc_atr

    n = len(high)
    out: List[Dict] = []
    start = max(2, n - lookback)

    # ATR-based minimum gap size filter
    min_atr_mult = float(os.getenv("MIN_FVG_ATR_MULTIPLIER", "0.5") or 0.5)
    atr_period = int(os.getenv("FVG_ATR_PERIOD", "14") or 14)

    # Calculate ATR for the entire series
    atr_vals = calc_atr(high, low, close, period=atr_period)

    # Fallback filters (kept for compatibility)
    min_gap_pct = float(os.getenv("FVG_MIN_GAP_PCT", "0.03") or 0.03)  # 0.03% minimum gap size
    min_gap_points = float(os.getenv("FVG_MIN_GAP_POINTS", "0.01") or 0.01)  # Minimum absolute gap size

    for i in range(start, n):
        try:
            # Get ATR at formation candle
            atr_at_i = float(atr_vals[i]) if i < len(atr_vals) else 0.0
            min_gap_size_atr = atr_at_i * min_atr_mult

            # BULLISH FVG: Upward gap - low[i] is ABOVE high[i-2]
            # Gap zone is between candle 1's high and candle 3's low
            if low[i] > high[i - 2]:
                gap_lo = float(high[i - 2])  # Bottom of gap: candle 1 high
                gap_hi = float(low[i])        # Top of gap: candle 3 low
                gap_size = gap_hi - gap_lo

                # Validate minimum gap size using ATR (primary filter)
                if atr_at_i > 0 and gap_size < min_gap_size_atr:
                    continue

                # Fallback validation (absolute and percentage) if ATR unavailable
                if gap_size < min_gap_points:
                    continue
                mid_price = (gap_lo + gap_hi) / 2.0
                gap_pct = (gap_size / mid_price) * 100.0
                if gap_pct < min_gap_pct:
                    continue

                out.append({"i": i, "side": "long", "lo": gap_lo, "hi": gap_hi, "gap_size": gap_size})

            # BEARISH FVG: Downward gap - high[i] is BELOW low[i-2]
            # Gap zone is between candle 3's high and candle 1's low
            if high[i] < low[i - 2]:
                gap_lo = float(high[i])       # Bottom of gap: candle 3 high
                gap_hi = float(low[i - 2])    # Top of gap: candle 1 low
                gap_size = gap_hi - gap_lo

                # Validate minimum gap size using ATR (primary filter)
                if atr_at_i > 0 and gap_size < min_gap_size_atr:
                    continue

                # Fallback validation (absolute and percentage) if ATR unavailable
                if gap_size < min_gap_points:
                    continue
                mid_price = (gap_lo + gap_hi) / 2.0
                gap_pct = (gap_size / mid_price) * 100.0
                if gap_pct < min_gap_pct:
                    continue

                out.append({"i": i, "side": "short", "lo": gap_lo, "hi": gap_hi, "gap_size": gap_size})
        except Exception:
            continue
    return out


def _overlaps_interval(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    return not (hi1 < lo2 or hi2 < lo1)


def _candle_enters_fvg(candle_high: float, candle_low: float, gap_lo: float, gap_hi: float) -> bool:
    """Check if candle's wick touches/enters the FVG gap zone."""
    return _overlaps_interval(candle_low, candle_high, gap_lo, gap_hi)


def monitor_fvg_and_detect_entry(
    fvgs: List[Dict],
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    or_levels: Optional[Tuple[float, float]] = None,
) -> Optional[Dict]:
    """
    IMPROVED FVG Entry Logic - Two-phase confirmation system:

    1. Find the most recent FVG (always override previous)
    2. PHASE 1 - Touch Detection: Wait for candle to ENTER the FVG zone (wick touch)
    3. PHASE 2 - Confirmation: Next candle must confirm rejection with:
       - Close in the rejection direction
       - Candle color matches trade direction (bearish for short, bullish for long)
    4. Strategy types:
       - CONTINUATION: Rejection in FVG direction (bullish FVG → long, bearish FVG → short)
       - INVERSION: Rejection opposite to FVG (bullish FVG → short, bearish FVG → long)
    5. SL placed BEYOND the FVG gap with buffer (not at gap boundary)

    CRITICAL: Polling Interval Safety
    - This function scans ALL closed candles after FVG formation
    - Even if bot checks every 60s, it analyzes every 1m candle that closed in between
    - Entry price is the CLOSE of confirmation candle (market entry on close)
    - Prevents false entries by requiring proper candle color confirmation

    Returns dict with entry details or None
    """
    if not fvgs or len(c) < 5:
        return None

    debug_fvg = _env_bool("FVG_DEBUG_DETECTION", False)

    # Always use the MOST RECENT FVG (last in list)
    fvg = fvgs[-1]

    fvg_side = fvg["side"]
    i0 = int(fvg["i"])
    gap_lo = float(fvg["lo"])
    gap_hi = float(fvg["hi"])
    gap_size = float(fvg.get("gap_size", gap_hi - gap_lo))

    # SL buffer configuration (percentage beyond gap)
    sl_buffer_pct = float(os.getenv("FVG_SL_BUFFER_PCT", "0.1") or 0.1) / 100.0  # 0.1% default

    if debug_fvg:
        tg(f"🔍 Monitoring FVG at bar {i0}: side={fvg_side}, gap=[{gap_lo:.2f}, {gap_hi:.2f}], size=${gap_size:.2f}")

    # Monitor candles AFTER FVG formation
    min_delay = int(os.getenv("FVG_MIN_CONFIRM_BARS", "1") or 1)
    scan_start = i0 + min_delay

    if scan_start >= len(c):
        return None  # FVG too recent

    # Track if we found a touch (phase 1)
    touch_found = False
    touch_idx = -1

    # Scan all candles after FVG formation
    for t in range(scan_start, len(c)):
        candle_open = float(o[t])
        candle_high = float(h[t])
        candle_low = float(l[t])
        candle_close = float(c[t])

        # Determine candle color (true = bullish/green, false = bearish/red)
        is_bullish_candle = candle_close >= candle_open

        # PHASE 1: Look for initial touch/entry into FVG zone
        if not touch_found:
            if _candle_enters_fvg(candle_high, candle_low, gap_lo, gap_hi):
                touch_found = True
                touch_idx = t
                if debug_fvg:
                    candle_color = "🟢GREEN" if is_bullish_candle else "🔴RED"
                    tg(f"🎯 PHASE 1: Candle {t} touched FVG: H:{candle_high:.2f} L:{candle_low:.2f} C:{candle_close:.2f} [{candle_color}]")
            continue  # Keep scanning

        # PHASE 2: We have a touch, now look for confirmation on NEXT candle
        # This is the confirmation candle (t > touch_idx)
        if t == touch_idx:
            continue  # Skip the touch candle itself, wait for next candle

        if debug_fvg:
            candle_color = "🟢GREEN" if is_bullish_candle else "🔴RED"
            tg(f"🔍 PHASE 2: Checking confirmation candle {t}: O:{candle_open:.2f} H:{candle_high:.2f} L:{candle_low:.2f} C:{candle_close:.2f} [{candle_color}]")

        # Check rejection direction and candle color
        entry_signal = None

        if fvg_side == "long":
            # BULLISH FVG - check for rejections

            # 1. CONTINUATION: Rejection upward (close above gap, bullish candle) → LONG
            if candle_close > gap_hi and is_bullish_candle:
                sl = gap_lo - (gap_lo * sl_buffer_pct)  # SL below gap with buffer
                entry_signal = {
                    "trade_side": "long",
                    "strategy_type": "continuation",
                    "entry": candle_close,
                    "sl": sl,
                }
                if debug_fvg:
                    tg(f"✅ CONTINUATION LONG: Bullish FVG + bullish candle closes above gap ({candle_close:.2f} > {gap_hi:.2f})")

            # 2. INVERSION: Rejection downward (close below gap, bearish candle) → SHORT
            elif candle_close < gap_lo and not is_bullish_candle:
                sl = gap_hi + (gap_hi * sl_buffer_pct)  # SL above gap with buffer
                entry_signal = {
                    "trade_side": "short",
                    "strategy_type": "inversion",
                    "entry": candle_close,
                    "sl": sl,
                }
                if debug_fvg:
                    tg(f"✅ INVERSION SHORT: Bullish FVG + bearish candle closes below gap ({candle_close:.2f} < {gap_lo:.2f})")

        else:  # fvg_side == "short"
            # BEARISH FVG - check for rejections

            # 1. CONTINUATION: Rejection downward (close below gap, bearish candle) → SHORT
            if candle_close < gap_lo and not is_bullish_candle:
                sl = gap_hi + (gap_hi * sl_buffer_pct)  # SL above gap with buffer
                entry_signal = {
                    "trade_side": "short",
                    "strategy_type": "continuation",
                    "entry": candle_close,
                    "sl": sl,
                }
                if debug_fvg:
                    tg(f"✅ CONTINUATION SHORT: Bearish FVG + bearish candle closes below gap ({candle_close:.2f} < {gap_lo:.2f})")

            # 2. INVERSION: Rejection upward (close above gap, bullish candle) → LONG
            elif candle_close > gap_hi and is_bullish_candle:
                sl = gap_lo - (gap_lo * sl_buffer_pct)  # SL below gap with buffer
                entry_signal = {
                    "trade_side": "long",
                    "strategy_type": "inversion",
                    "entry": candle_close,
                    "sl": sl,
                }
                if debug_fvg:
                    tg(f"✅ INVERSION LONG: Bearish FVG + bullish candle closes above gap ({candle_close:.2f} > {gap_hi:.2f})")

        # If no confirmation yet, reset touch and keep scanning for new touch
        if entry_signal is None:
            if debug_fvg:
                tg(f"❌ No confirmation: candle color or close direction doesn't match. Resetting touch detection.")
            touch_found = False
            touch_idx = -1
            continue

        # We have a valid confirmed entry signal
        if entry_signal:
            rr_cfg = float(os.getenv("FVG_RR", "2.0") or 2.0)
            entry = entry_signal["entry"]
            sl = entry_signal["sl"]

            risk = abs(entry - sl)
            if entry_signal["trade_side"] == "long":
                tp = entry + rr_cfg * risk
            else:
                tp = entry - rr_cfg * risk

            # Log detailed entry confirmation
            if debug_fvg:
                touch_candle_o = float(o[touch_idx])
                touch_candle_h = float(h[touch_idx])
                touch_candle_l = float(l[touch_idx])
                touch_candle_c = float(c[touch_idx])
                touch_is_bullish = touch_candle_c >= touch_candle_o
                touch_color = "🟢GREEN" if touch_is_bullish else "🔴RED"

                confirm_is_bullish = candle_close >= candle_open
                confirm_color = "🟢GREEN" if confirm_is_bullish else "🔴RED"

                strategy_emoji = "🔄" if entry_signal["strategy_type"] == "continuation" else "🔀"
                trade_direction = "🟢LONG" if entry_signal["trade_side"] == "long" else "🔴SHORT"

                tg(
                    f"\n{'='*50}\n"
                    f"✅ ENTRY SIGNAL CONFIRMED\n"
                    f"{'='*50}\n"
                    f"📊 FVG Context:\n"
                    f"  • FVG Type: {fvg_side.upper()}\n"
                    f"  • FVG Bar: #{i0}\n"
                    f"  • Gap Zone: ${gap_lo:.2f} - ${gap_hi:.2f}\n"
                    f"  • Gap Size: ${gap_size:.2f}\n"
                    f"\n🎯 Two-Phase Confirmation:\n"
                    f"  PHASE 1 - Touch (Bar #{touch_idx}):\n"
                    f"    • Candle: {touch_color}\n"
                    f"    • O:{touch_candle_o:.2f} H:{touch_candle_h:.2f} L:{touch_candle_l:.2f} C:{touch_candle_c:.2f}\n"
                    f"    • Action: Wick entered FVG zone\n"
                    f"\n  PHASE 2 - Confirmation (Bar #{t}):\n"
                    f"    • Candle: {confirm_color}\n"
                    f"    • O:{candle_open:.2f} H:{candle_high:.2f} L:{candle_low:.2f} C:{candle_close:.2f}\n"
                    f"    • Action: Closed {'above' if entry_signal['trade_side'] == 'long' else 'below'} FVG gap\n"
                    f"    • Candle color matches trade direction ✓\n"
                    f"\n{strategy_emoji} Strategy: {entry_signal['strategy_type'].upper()}\n"
                    f"  • {fvg_side.upper()} FVG rejected {'upward' if entry_signal['trade_side'] == 'long' else 'downward'}\n"
                    f"  • Trade: {trade_direction}\n"
                    f"\n💰 Trade Parameters:\n"
                    f"  • Entry: ${entry:.2f}\n"
                    f"  • Stop Loss: ${sl:.2f}\n"
                    f"  • Take Profit: ${tp:.2f}\n"
                    f"  • Risk: ${risk:.2f}\n"
                    f"  • Reward: ${abs(tp - entry):.2f}\n"
                    f"  • R:R Ratio: {rr_cfg:.1f}:1\n"
                    f"{'='*50}\n"
                )

            return {
                "side": entry_signal["trade_side"],
                "fvg_side": fvg_side,
                "strategy_type": entry_signal["strategy_type"],
                "fvg_i": i0,
                "touch_i": touch_idx,
                "confirm_i": t,
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
                "rr": float(rr_cfg),
                "pattern": f"fvg_{entry_signal['strategy_type']}",
                "gap_size": float(gap_size),
                "gap_lo": gap_lo,
                "gap_hi": gap_hi,
                "touch_candle": {
                    "o": float(o[touch_idx]),
                    "h": float(h[touch_idx]),
                    "l": float(l[touch_idx]),
                    "c": float(c[touch_idx]),
                    "is_bullish": touch_is_bullish,
                },
                "confirm_candle": {
                    "o": float(candle_open),
                    "h": float(candle_high),
                    "l": float(candle_low),
                    "c": float(candle_close),
                    "is_bullish": confirm_is_bullish,
                },
            }

    return None


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
        # Track last logged FVG to detect overrides
        self.last_logged_fvg_idx = None

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

        CRITICAL: Only considers CLOSED candles for rejection confirmation (excludes last/forming candle)

        Returns a payload dict with diagnostics for logging.
        """
        ohlcv = fetch_candles(ex, symbol, timeframe=self.fvg_timeframe, limit=limit)
        arr = np.asarray(ohlcv, dtype=float)
        ts, o, h, l, c, v = arr.T

        # CRITICAL: Exclude the last candle (currently forming) from analysis
        # Only work with CLOSED candles to prevent premature entries
        if len(ts) > 1:
            ts, o, h, l, c, v = ts[:-1], o[:-1], h[:-1], l[:-1], c[:-1], v[:-1]
        else:
            # Not enough data
            return {
                "symbol": symbol,
                "now": int(time.time()),
                "or_ready": False,
                "signal": None,
            }

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
        # NEW: Include close prices for ATR calculation
        fvgs: List[Dict] = []
        debug_mode = _env_bool("FVG_DEBUG_DETECTION", False)

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
                c_scan = c[scan_start:scan_end]
                in_scan = detect_fvgs(h_scan, l_scan, c_scan, lookback=len(h_scan))
                for f in in_scan:
                    fvg_idx = int(f["i"]) + int(scan_start)
                    fvgs.append({
                        "i": fvg_idx,  # remap to full-series index
                        "side": f["side"],
                        "lo": f["lo"],
                        "hi": f["hi"],
                        "gap_size": f.get("gap_size", f["hi"] - f["lo"]),
                    })

                # Log FVG detection and overrides
                if fvgs and debug_mode:
                    latest_fvg = fvgs[-1]
                    fvg_idx = latest_fvg["i"]

                    # Check if this is a new FVG (different from last logged)
                    if self.last_logged_fvg_idx is None:
                        # First FVG detected
                        fvg_ts = datetime.fromtimestamp(ts[fvg_idx] / 1000, tz=timezone.utc).strftime("%H:%M:%S")
                        side_emoji = "📈" if latest_fvg["side"] == "long" else "📉"
                        tg(
                            f"{side_emoji} FVG DETECTED on {symbol}\n"
                            f"  Type: {latest_fvg['side'].upper()}\n"
                            f"  Time: {fvg_ts} UTC (bar #{fvg_idx})\n"
                            f"  Gap: ${latest_fvg['lo']:.2f} - ${latest_fvg['hi']:.2f}\n"
                            f"  Size: ${latest_fvg['gap_size']:.2f}\n"
                            f"  Status: Monitoring for rejection..."
                        )
                        self.last_logged_fvg_idx = fvg_idx
                    elif fvg_idx != self.last_logged_fvg_idx:
                        # New FVG overrides previous one
                        fvg_ts = datetime.fromtimestamp(ts[fvg_idx] / 1000, tz=timezone.utc).strftime("%H:%M:%S")
                        side_emoji = "📈" if latest_fvg["side"] == "long" else "📉"
                        tg(
                            f"🔄 FVG OVERRIDE on {symbol}\n"
                            f"  Previous FVG (bar #{self.last_logged_fvg_idx}) replaced\n"
                            f"  New Type: {latest_fvg['side'].upper()}\n"
                            f"  Time: {fvg_ts} UTC (bar #{fvg_idx})\n"
                            f"  Gap: ${latest_fvg['lo']:.2f} - ${latest_fvg['hi']:.2f}\n"
                            f"  Size: ${latest_fvg['gap_size']:.2f}\n"
                            f"  Status: Monitoring for rejection..."
                        )
                        self.last_logged_fvg_idx = fvg_idx

        # Build signal using new monitoring logic
        # NEW: No restriction on OR boundary for trading (removed allow_outside_or check)
        signal = None
        if or_ready and fvgs:
            cand = monitor_fvg_and_detect_entry(
                fvgs, o, h, l, c, v,
                or_levels=(orh, orl),
            )
            # Allow signals anytime after OR is ready (removed post_first_candle_i restriction)
            if cand is not None:
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
            "o": o, "h": h, "l": l, "c": c, "v": v,
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
            fvg_side = sig.get('fvg_side', side)
            strategy_type = sig.get('strategy_type', 'continuation')

            # Calculate key metrics
            risk_usd = abs(entry - sl)
            reward_usd = abs(tp - entry)

            # Show entry position relative to OR (informational only, not a restriction)
            entry_position = ""
            if or_high is not None and or_low is not None:
                if entry > or_high:
                    entry_position = f"📍 Entry: ${entry:.2f} (${entry - or_high:.2f} above OR)"
                elif entry < or_low:
                    entry_position = f"📍 Entry: ${entry:.2f} (${or_low - entry:.2f} below OR)"
                else:
                    entry_position = f"📍 Entry: ${entry:.2f} (inside OR range)"
            else:
                entry_position = f"📍 Entry: ${entry:.2f}"

            # Strategy type indicator
            strategy_emoji = "🔄" if strategy_type == "continuation" else "🔀"
            strategy_label = f"{strategy_emoji} {strategy_type.upper()}"

            br = f"Entry: ${entry:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f}\nRisk: ${risk_usd:.2f} | Reward: ${reward_usd:.2f} | RR: {sig['rr']:.1f}:1"

            # Human-readable bar details (UTC) with safe bounds (no raw indices)
            ts_arr = payload.get("ts")
            o_arr = payload.get("o")
            h_arr = payload.get("h")
            l_arr = payload.get("l")
            c_arr = payload.get("c")
            v_arr = payload.get("v")
            if ts_arr is None: ts_arr = []
            if o_arr is None: o_arr = []
            if h_arr is None: h_arr = []
            if l_arr is None: l_arr = []
            if c_arr is None: c_arr = []
            if v_arr is None: v_arr = []

            def iso(ms_val):
                try:
                    return datetime.fromtimestamp(float(ms_val)/1000.0, tz=timezone.utc).strftime("%H:%M:%S")
                except Exception:
                    return "n/a"

            def bar_line(i: int, include_vol: bool = True) -> str:
                try:
                    if isinstance(i, (int, np.integer)) and 0 <= i < len(ts_arr):
                        o_val = float(o_arr[i])
                        h_val = float(h_arr[i])
                        l_val = float(l_arr[i])
                        c_val = float(c_arr[i])
                        # Determine color based on close vs open
                        color = "🟢" if c_val >= o_val else "🔴"
                        bar_str = f"{color} {iso(ts_arr[i])} | O:{o_val:.2f} H:{h_val:.2f} L:{l_val:.2f} C:{c_val:.2f}"
                        if include_vol and i < len(v_arr):
                            bar_str += f" V:{int(v_arr[i])}"
                        return bar_str
                    return "unavailable"
                except Exception:
                    return "unavailable"

            i_f = int(sig.get("fvg_i", -1))
            i_t = int(sig.get("touch_i", -1))
            i_c = int(sig.get("confirm_i", -1))

            # Calculate FVG age (bars between formation and confirmation)
            fvg_age = i_c - i_f if (i_f >= 0 and i_c >= 0) else 0

            # Show candles leading up to entry (including FVG formation, touch, and confirmation)
            # This gives complete context of the two-phase price action
            recent_candles = []
            if i_c >= 0:
                for j in range(max(0, i_c - 4), i_c + 1):
                    label = ""
                    if j == i_f:
                        label = " ← FVG FORMED"
                    elif j == i_f - 1:
                        label = " (middle candle)"
                    elif j == i_f - 2:
                        label = " (first candle)"
                    elif j == i_t:
                        label = " ← TOUCH (Phase 1)"
                    elif j == i_c:
                        label = " ← CONFIRMATION (Phase 2)"

                    recent_candles.append(f"  {j - i_c + 5}. {bar_line(j)}{label}")

            candles_block = "\n".join(recent_candles) if recent_candles else "No candle data"

            # FVG detection explanation with strategy type
            fvg_explanation = ""
            if fvg_side == "long":
                base_explanation = (
                    f"BULLISH FVG: Gap UP detected\n"
                    f"  • Gap zone: ${gap_lo:.2f} to ${gap_hi:.2f}\n"
                )
                if strategy_type == "continuation":
                    fvg_explanation = base_explanation + f"  • CONTINUATION: Price entered gap, closed ABOVE ${gap_hi:.2f} → LONG"
                else:
                    fvg_explanation = base_explanation + f"  • INVERSION: Price entered gap, closed BELOW ${gap_lo:.2f} → SHORT"
            else:
                base_explanation = (
                    f"BEARISH FVG: Gap DOWN detected\n"
                    f"  • Gap zone: ${gap_lo:.2f} to ${gap_hi:.2f}\n"
                )
                if strategy_type == "continuation":
                    fvg_explanation = base_explanation + f"  • CONTINUATION: Price entered gap, closed BELOW ${gap_lo:.2f} → SHORT"
                else:
                    fvg_explanation = base_explanation + f"  • INVERSION: Price entered gap, closed ABOVE ${gap_hi:.2f} → LONG"

            tg(
                f"🎯 NY-Open FVG {sym} [{side.upper()}]\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 {or_info}\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📈 {fvg_explanation}\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"{entry_position}\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ {strategy_label} CONFIRMED\n"
                f"{br}\n"
                f"⏱️ FVG Age: {fvg_age} bars\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Last 5 Candles:\n"
                f"{candles_block}"
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