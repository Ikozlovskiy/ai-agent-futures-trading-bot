"""
Multi-Confluence Scalper (MCS)
24/7 scalping strategy with 4-layer confluence model:
  Layer 1: Market Structure (HTF trend filter)
  Layer 2: Price Action Patterns (FVG, S/D, Trendline, Double Touch)
  Layer 3: Confirmation Filters (Volume, ATR, Spread)
  Layer 4: Multi-Timeframe Alignment
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from utils import tg
from datahub import fetch_candles, ema, atr


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Calculate EMA for the given array."""
    if arr is None or len(arr) < span + 5:
        return np.array([])
    alpha = 2 / (span + 1.0)
    out = np.zeros_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


def _rsi(arr: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI."""
    if len(arr) < period + 1:
        return np.full_like(arr, 50.0)

    delta = np.diff(arr)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    avg_gain = np.zeros(len(arr))
    avg_loss = np.zeros(len(arr))

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(arr)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR."""
    if len(h) < period + 1:
        return np.zeros_like(h)

    tr = np.maximum(h[1:] - l[1:], 
                    np.maximum(np.abs(h[1:] - c[:-1]), 
                              np.abs(l[1:] - c[:-1])))

    atr_vals = np.zeros(len(h))
    atr_vals[period] = np.mean(tr[:period])

    for i in range(period + 1, len(h)):
        atr_vals[i] = (atr_vals[i-1] * (period - 1) + tr[i-1]) / period

    return atr_vals


class MultiConfluenceScalper:
    """
    Professional 24/7 scalping strategy with multi-layer confluence.
    """

    def __init__(self):
        # Timeframes
        self.htf = os.getenv("SCALP_HTF_TIMEFRAME", "15m")
        self.mtf = os.getenv("SCALP_MTF_TIMEFRAME", "5m")
        self.ltf = os.getenv("SCALP_LTF_TIMEFRAME", "1m")

        # Layer 1: Market Structure
        self.trend_ema_fast = int(os.getenv("SCALP_TREND_EMA_FAST", "50") or 50)
        self.trend_ema_slow = int(os.getenv("SCALP_TREND_EMA_SLOW", "200") or 200)
        self.bias_candle_count = int(os.getenv("SCALP_BIAS_CANDLE_COUNT", "5") or 5)

        # Layer 2: Pattern Enablement
        self.enable_fvg = _env_bool("SCALP_ENABLE_FVG", True)
        self.enable_sd = _env_bool("SCALP_ENABLE_SD_ZONES", True)
        self.enable_trendline = _env_bool("SCALP_ENABLE_TRENDLINE", True)
        self.enable_double = _env_bool("SCALP_ENABLE_DOUBLE_TOUCH", True)

        # Layer 3: Confirmation Filters
        self.vol_mult = float(os.getenv("SCALP_VOL_MULT", "1.3") or 1.3)
        self.vol_lookback = int(os.getenv("SCALP_VOL_LOOKBACK", "20") or 20)
        # Lower default for scalping since 1m ATR is naturally much smaller than 15m ATR
        # Set to 0 to disable this check entirely
        self.atr_ratio_min = float(os.getenv("SCALP_ATR_RATIO_MIN", "0.15") or 0.15)
        self.spread_max_pct = float(os.getenv("SCALP_SPREAD_MAX_PCT", "0.05") or 0.05)

        # Layer 4: Multi-Timeframe
        self.rsi_period = 14
        self.rsi_ob_threshold = int(os.getenv("SCALP_RSI_OVERBOUGHT", "75") or 75)
        self.rsi_os_threshold = int(os.getenv("SCALP_RSI_OVERSOLD", "25") or 25)

        # Risk Management
        self.min_sl_pct = float(os.getenv("SCALP_MIN_SL_PCT", "0.3") or 0.3)
        self.max_sl_pct = float(os.getenv("SCALP_MAX_SL_PCT", "0.8") or 0.8)

        # FVG Pattern Management (prevent immediate re-entry on same pattern)
        self.fvg_max_age_minutes = int(os.getenv("FVG_MAX_AGE_MINUTES", "120") or 120)
        self.fvg_pattern_cache_size = int(os.getenv("FVG_PATTERN_CACHE_SIZE", "50") or 50)
        self.used_fvg_indices = {}  # {symbol: [list of used FVG formation indices]}

        # Time-of-Day Weights
        self.parse_time_weights()

    def parse_time_weights(self):
        """Parse time-of-day priority hours from config."""
        high = os.getenv("SCALP_HIGH_PRIORITY_HOURS", "8-17")
        med = os.getenv("SCALP_MED_PRIORITY_HOURS", "0-8,17-20")
        low = os.getenv("SCALP_LOW_PRIORITY_HOURS", "20-24")

        def parse_ranges(s: str) -> List[Tuple[int, int]]:
            ranges = []
            for part in s.split(","):
                if "-" in part:
                    start, end = part.split("-")
                    ranges.append((int(start), int(end)))
            return ranges

        self.high_hours = parse_ranges(high)
        self.med_hours = parse_ranges(med)
        self.low_hours = parse_ranges(low)

    def get_time_weight(self) -> float:
        """Get confidence multiplier based on current UTC hour."""
        hour = datetime.now(timezone.utc).hour

        for start, end in self.high_hours:
            if start <= hour < end:
                return 1.0

        for start, end in self.med_hours:
            if start <= hour < end:
                return 0.8

        for start, end in self.low_hours:
            if start <= hour < end:
                return 0.6

        return 0.8  # Default medium confidence

    # ========== LAYER 1: MARKET STRUCTURE ==========

    def check_htf_trend(self, ex, symbol: str) -> Optional[str]:
        """
        Check 15m trend using EMA50/200.
        Returns: 'long', 'short', or 'neutral'
        """
        try:
            ohlcv = fetch_candles(ex, symbol, timeframe=self.htf, limit=250)
            arr = np.asarray(ohlcv, dtype=float)
            _, _, _, _, c, _ = arr.T

            ema50 = _ema(c, self.trend_ema_fast)
            ema200 = _ema(c, self.trend_ema_slow)

            if len(ema50) < 10 or len(ema200) < 10:
                return "neutral"

            current_close = float(c[-1])
            current_ema50 = float(ema50[-1])
            current_ema200 = float(ema200[-1])

            if current_close > current_ema50 > current_ema200:
                return "long"  # FIX: Changed from "bullish"
            elif current_close < current_ema50 < current_ema200:
                return "short"  # FIX: Changed from "bearish"
            else:
                return "neutral"
        except Exception as e:
            tg(f"⚠️ HTF trend check error for {symbol}: {e}")
            return "neutral"

    def check_mtf_bias(self, ex, symbol: str) -> Optional[str]:
        """
        Check 5m bias using candle color consistency.
        Returns: 'long', 'short', or 'neutral'
        """
        try:
            ohlcv = fetch_candles(ex, symbol, timeframe=self.mtf, limit=20)
            arr = np.asarray(ohlcv, dtype=float)
            _, o, _, _, c, _ = arr.T

            recent_candles = min(self.bias_candle_count, len(c))
            green_count = sum(1 for i in range(-recent_candles, 0) if c[i] > o[i])
            red_count = recent_candles - green_count

            if green_count >= recent_candles * 0.6:
                return "long"  # FIX: Changed from "bullish"
            elif red_count >= recent_candles * 0.6:
                return "short"  # FIX: Changed from "bearish"
            else:
                return "neutral"
        except Exception:
            return "neutral"

    # ========== LAYER 2: PATTERN DETECTION ==========

    def detect_fvg(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, 
                   lookback: int = 50, symbol: str = "") -> List[Dict]:
        """
        Detect Fair Value Gaps and check if LAST CANDLE is touching them.

        NEW APPROACH:
        1. Detect FVG formations (3-candle gaps)
        2. Check if the LAST CLOSED candle (-2) is touching any valid FVG
        3. Return pattern only if last candle touches + confirms direction
        4. Volume will be checked on the LAST candle, not the historical FVG
        5. Filter out recently used FVGs and stale patterns (age limit)
        """
        from datahub import atr as calc_atr

        fvgs = []
        n = len(h)
        start = max(2, n - lookback)

        # ATR-based minimum gap size filter
        min_atr_mult = float(os.getenv("MIN_FVG_ATR_MULTIPLIER", "0.3") or 0.3)
        atr_period = int(os.getenv("FVG_ATR_PERIOD", "14") or 14)

        # Calculate ATR for the entire series
        atr_vals = calc_atr(h, l, c, period=atr_period)

        # Fallback filters
        min_gap_pct = float(os.getenv("FVG_MIN_GAP_PCT", "0.03") or 0.03)
        min_gap_points = float(os.getenv("FVG_MIN_GAP_POINTS", "0.01") or 0.01)

        # Get used FVG indices for this symbol (prevent re-trading same pattern)
        used_indices = self.used_fvg_indices.get(symbol, []) if hasattr(self, 'used_fvg_indices') else []

        # First pass: detect all FVG formations
        detected_fvgs = []
        for i in range(start, n - 2):  # Stop before last 2 candles to ensure we have current candle
            try:
                # FILTER 1: Skip if this FVG was already used recently
                if i in used_indices:
                    continue

                # FILTER 2: Check FVG age (don't trade stale patterns)
                # Calculate age in candles (1m timeframe = 1 candle per minute)
                fvg_age_candles = (n - 2) - i  # Age from current closed candle to FVG formation
                if hasattr(self, 'fvg_max_age_minutes') and fvg_age_candles > self.fvg_max_age_minutes:
                    continue  # FVG too old

                # Get ATR at formation candle
                atr_at_i = float(atr_vals[i]) if i < len(atr_vals) else 0.0
                min_gap_size_atr = atr_at_i * min_atr_mult

                # BULLISH FVG: Upward gap - low[i] is ABOVE high[i-2]
                if l[i] > h[i - 2]:
                    gap_lo = float(h[i - 2])  # Bottom of gap: candle 1 high
                    gap_hi = float(l[i])      # Top of gap: candle 3 low
                    gap_size = gap_hi - gap_lo

                    # Validate minimum gap size using ATR (primary filter)
                    if atr_at_i > 0 and gap_size < min_gap_size_atr:
                        continue

                    # Fallback validation
                    if gap_size < min_gap_points:
                        continue
                    mid_price = (gap_lo + gap_hi) / 2.0
                    gap_pct = (gap_size / mid_price) * 100.0
                    if gap_pct < min_gap_pct:
                        continue

                    detected_fvgs.append({
                        "i": i,
                        "fvg_side": "long",  # FVG direction
                        "lo": gap_lo,
                        "hi": gap_hi,
                        "gap_size": gap_size
                    })

                # BEARISH FVG: Downward gap - high[i] is BELOW low[i-2]
                if h[i] < l[i - 2]:
                    gap_lo = float(h[i])      # Bottom of gap: candle 3 high
                    gap_hi = float(l[i - 2])  # Top of gap: candle 1 low
                    gap_size = gap_hi - gap_lo

                    # Validate minimum gap size using ATR
                    if atr_at_i > 0 and gap_size < min_gap_size_atr:
                        continue

                    # Fallback validation
                    if gap_size < min_gap_points:
                        continue
                    mid_price = (gap_lo + gap_hi) / 2.0
                    gap_pct = (gap_size / mid_price) * 100.0
                    if gap_pct < min_gap_pct:
                        continue

                    detected_fvgs.append({
                        "i": i,
                        "fvg_side": "short",
                        "lo": gap_lo,
                        "hi": gap_hi,
                        "gap_size": gap_size
                    })
            except Exception:
                continue

        if not detected_fvgs:
            return []

        # NEW APPROACH: Check if LAST CLOSED candle is touching any FVG
        # Use index -2 (last closed candle), -1 is the current forming candle
        last_candle_idx = len(c) - 2
        if last_candle_idx < 0:
            return []

        last_high = float(h[last_candle_idx])
        last_low = float(l[last_candle_idx])
        last_close = float(c[last_candle_idx])

        # Check each detected FVG to see if last candle is touching it
        for fvg in reversed(detected_fvgs):  # Check most recent FVGs first
            fvg_side = fvg["fvg_side"]
            gap_lo = fvg["lo"]
            gap_hi = fvg["hi"]
            gap_size = fvg["gap_size"]

            # Check if last candle touches the FVG zone (wick enters gap)
            candle_touches = not (last_high < gap_lo or last_low > gap_hi)

            if not candle_touches:
                continue  # Last candle doesn't touch this FVG, try next one

            # Last candle touches the FVG! Now determine entry direction based on close
            entry_signal = None

            if fvg_side == "long":
                # BULLISH FVG - two possibilities:
                # 1. CONTINUATION: Close above gap → LONG
                if last_close > gap_hi:
                    entry_signal = {
                        "trade_side": "long",
                        "strategy_type": "continuation",
                    }
                # 2. INVERSION: Close below gap → SHORT
                elif last_close < gap_lo:
                    entry_signal = {
                        "trade_side": "short",
                        "strategy_type": "inversion",
                    }

            else:  # fvg_side == "short"
                # BEARISH FVG - two possibilities:
                # 1. CONTINUATION: Close below gap → SHORT
                if last_close < gap_lo:
                    entry_signal = {
                        "trade_side": "short",
                        "strategy_type": "continuation",
                    }
                # 2. INVERSION: Close above gap → LONG
                elif last_close > gap_hi:
                    entry_signal = {
                        "trade_side": "long",
                        "strategy_type": "inversion",
                    }

            # If we have a valid entry signal, return it
            if entry_signal:
                fvgs.append({
                    "i": last_candle_idx,  # CRITICAL: Use LAST candle index for volume check
                    "fvg_i": fvg["i"],  # Original FVG formation index
                    "side": entry_signal["trade_side"],
                    "fvg_side": fvg_side,
                    "strategy_type": entry_signal["strategy_type"],
                    "lo": gap_lo,
                    "hi": gap_hi,
                    "gap_size": gap_size,
                    "pattern": f"fvg_{entry_signal['strategy_type']}"
                })
                break  # Take first valid FVG that last candle is touching

        return fvgs

    def detect_sd_zones(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, 
                        v: np.ndarray, lookback: int = 30) -> List[Dict]:
        """
        Detect Supply/Demand zones on 5m.
        Fresh zones where price left with strong momentum and hasn't returned.
        """
        zones = []
        n = len(c)
        start = max(5, n - lookback)

        for i in range(start, n - 3):
            # Demand zone: Strong bullish move after consolidation
            # RELAXED: 60% body ratio (was 70%)
            if c[i] > o[i] and (c[i] - o[i]) / max(1e-9, h[i] - l[i]) > 0.6:
                # Check volume expansion (RELAXED: 1.2x instead of 1.3x)
                if i >= 10:
                    avg_vol = np.mean(v[max(0, i-10):i])
                    if v[i] > avg_vol * 1.2:
                        # Define zone as the base before the move
                        zone_high = float(o[i])
                        zone_low = float(l[i-1])

                        # Check if zone is "fresh" (not retested)
                        retested = any(l[j] <= zone_high for j in range(i+1, min(i+10, n)))
                        if not retested:
                            zones.append({
                                "i": i,
                                "side": "long",
                                "lo": zone_low,
                                "hi": zone_high,
                                "pattern": "demand_zone"
                            })

            # Supply zone: Strong bearish move after consolidation
            # RELAXED: 60% body ratio (was 70%)
            if c[i] < o[i] and (o[i] - c[i]) / max(1e-9, h[i] - l[i]) > 0.6:
                if i >= 10:
                    avg_vol = np.mean(v[max(0, i-10):i])
                    if v[i] > avg_vol * 1.2:
                        zone_low = float(o[i])
                        zone_high = float(h[i-1])

                        retested = any(h[j] >= zone_low for j in range(i+1, min(i+10, n)))
                        if not retested:
                            zones.append({
                                "i": i,
                                "side": "short",
                                "lo": zone_low,
                                "hi": zone_high,
                                "pattern": "supply_zone"
                            })

        return zones

    def detect_trendline_break(self, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray,
                               lookback: int = 30) -> List[Dict]:
        """
        Detect trendline breaks on 5m.
        Identify uptrend/downtrend lines and detect clean breaks with volume.
        """
        breaks = []
        n = len(c)
        start = max(10, n - lookback)

        # Minimum touches required for valid trendline
        min_touches = 3

        for i in range(start, n - 3):
            # Look for uptrend line (connecting higher lows)
            lows = []
            for j in range(max(0, i - 20), i):
                # Identify swing lows (local minima)
                if j > 0 and j < len(l) - 1:
                    if l[j] < l[j-1] and l[j] < l[j+1]:
                        lows.append((j, l[j]))

            # Need at least 3 swing lows for trendline
            if len(lows) >= min_touches:
                lows_sorted = sorted(lows, key=lambda x: x[0])
                # Check if they form higher lows (uptrend)
                is_uptrend = all(lows_sorted[k][1] < lows_sorted[k+1][1] 
                                for k in range(len(lows_sorted)-1))

                if is_uptrend:
                    # Check for break below trendline with volume
                    last_low = lows_sorted[-1][1]
                    # RELAXED: 0.15% buffer (was 0.2%)
                    if l[i] < last_low * 0.9985:
                        # Check volume expansion (RELAXED: 1.2x was 1.3x)
                        if i >= 10:
                            avg_vol = np.mean(v[max(0, i-10):i])
                            if v[i] > avg_vol * 1.2:
                                breaks.append({
                                    "i": i,
                                    "side": "short",  # Break of uptrend = short signal
                                    "lo": float(l[i]),
                                    "hi": float(last_low),
                                    "pattern": "trendline_break"
                                })

            # Look for downtrend line (connecting lower highs)
            highs = []
            for j in range(max(0, i - 20), i):
                # Identify swing highs (local maxima)
                if j > 0 and j < len(h) - 1:
                    if h[j] > h[j-1] and h[j] > h[j+1]:
                        highs.append((j, h[j]))

            if len(highs) >= min_touches:
                highs_sorted = sorted(highs, key=lambda x: x[0])
                # Check if they form lower highs (downtrend)
                is_downtrend = all(highs_sorted[k][1] > highs_sorted[k+1][1]
                                  for k in range(len(highs_sorted)-1))

                if is_downtrend:
                    # Check for break above trendline with volume
                    last_high = highs_sorted[-1][1]
                    # RELAXED: 0.15% buffer (was 0.2%)
                    if h[i] > last_high * 1.0015:
                        if i >= 10:
                            avg_vol = np.mean(v[max(0, i-10):i])
                            if v[i] > avg_vol * 1.2:
                                breaks.append({
                                    "i": i,
                                    "side": "long",  # Break of downtrend = long signal
                                    "lo": float(last_high),
                                    "hi": float(h[i]),
                                    "pattern": "trendline_break"
                                })

        return breaks

    def detect_double_touch(self, h: np.ndarray, l: np.ndarray, lookback: int = 20) -> List[Dict]:
        """
        Detect double bottom/top patterns on 1m.
        Two equal lows/highs within lookback period.
        """
        patterns = []
        n = len(l)
        start = max(5, n - lookback)

        for i in range(start, n - 2):
            # Double bottom
            for j in range(i + 2, min(i + lookback, n)):
                # RELAXED: 0.3% tolerance (was 0.2%)
                if abs(l[i] - l[j]) / l[i] < 0.003:
                    # Check if there's a higher low between them (RELAXED: 0.25% was 0.3%)
                    if all(l[k] >= min(l[i], l[j]) * 1.0025 for k in range(i+1, j)):
                        patterns.append({
                            "i": j,
                            "side": "long",
                            "lo": float(min(l[i], l[j])),
                            "hi": float(max(h[i], h[j])),
                            "pattern": "double_bottom"
                        })
                        break

            # Double top
            for j in range(i + 2, min(i + lookback, n)):
                # RELAXED: 0.3% tolerance (was 0.2%)
                if abs(h[i] - h[j]) / h[i] < 0.003:
                    # RELAXED: 0.25% buffer (was 0.3%)
                    if all(h[k] <= max(h[i], h[j]) * 0.9975 for k in range(i+1, j)):
                        patterns.append({
                            "i": j,
                            "side": "short",
                            "lo": float(min(l[i], l[j])),
                            "hi": float(max(h[i], h[j])),
                            "pattern": "double_top"
                        })
                        break

        return patterns

    # ========== LAYER 3: CONFIRMATION FILTERS ==========

    def check_volume_expansion(self, v: np.ndarray, trigger_i: int) -> bool:
        """Check if trigger candle has volume expansion."""
        debug = _env_bool("SCALP_DEBUG", False)

        # Allow disabling this check entirely by setting vol_mult to 0
        if self.vol_mult <= 0:
            if debug:
                tg(f"  ↳ Volume check: DISABLED (vol_mult=0) - PASS")
            return True

        # Validate we have enough historical data
        if trigger_i < self.vol_lookback:
            if debug:
                tg(f"  ↳ Volume check: Not enough history (trigger_i={trigger_i} < lookback={self.vol_lookback}) - PASS")
            return True  # Not enough data, pass

        # Ensure trigger_i is within bounds
        if trigger_i >= len(v):
            if debug:
                tg(f"  ↳ Volume check: trigger_i ({trigger_i}) out of bounds (len={len(v)}) - PASS")
            return True

        trigger_vol = float(v[trigger_i])
        lookback_start = max(0, trigger_i - self.vol_lookback)
        avg_vol = float(np.mean(v[lookback_start:trigger_i]))

        if avg_vol < 1e-9:
            if debug:
                tg(f"  ↳ Volume check: avg_vol too small - PASS")
            return True

        vol_ratio = trigger_vol / avg_vol
        is_expanded = trigger_vol >= avg_vol * self.vol_mult

        if debug:
            tg(f"  ↳ Volume check: trigger={trigger_vol:.0f} avg={avg_vol:.0f} ratio={vol_ratio:.2f}x (need {self.vol_mult:.2f}x) - {'✓ PASS' if is_expanded else '✗ FAIL'}")

        return is_expanded

    def check_atr_ratio(self, ex, symbol: str) -> bool:
        """Check if LTF ATR is sufficient vs HTF ATR (market is moving)."""
        debug = _env_bool("SCALP_DEBUG", False)

        # Allow disabling this check entirely by setting threshold to 0
        if self.atr_ratio_min <= 0:
            if debug:
                tg(f"  ↳ ATR ratio check: DISABLED (threshold=0) - PASS")
            return True

        try:
            # Get LTF ATR
            ltf_ohlcv = fetch_candles(ex, symbol, timeframe=self.ltf, limit=50)
            ltf_arr = np.asarray(ltf_ohlcv, dtype=float)
            _, _, ltf_h, ltf_l, ltf_c, _ = ltf_arr.T
            ltf_atr = _atr(ltf_h, ltf_l, ltf_c, 14)

            # Get HTF ATR
            htf_ohlcv = fetch_candles(ex, symbol, timeframe=self.htf, limit=50)
            htf_arr = np.asarray(htf_ohlcv, dtype=float)
            _, _, htf_h, htf_l, htf_c, _ = htf_arr.T
            htf_atr = _atr(htf_h, htf_l, htf_c, 14)

            if len(ltf_atr) < 5 or len(htf_atr) < 5:
                if debug:
                    tg(f"  ↳ ATR ratio check: Not enough data - PASS")
                return True

            current_ltf_atr = float(ltf_atr[-1])
            current_htf_atr = float(htf_atr[-1])

            if current_htf_atr < 1e-9:
                if debug:
                    tg(f"  ↳ ATR ratio check: HTF ATR too small - PASS")
                return True

            ratio = current_ltf_atr / current_htf_atr
            is_moving = ratio >= self.atr_ratio_min

            if debug:
                tg(f"  ↳ ATR ratio check: LTF={current_ltf_atr:.4f} ({self.ltf}) HTF={current_htf_atr:.4f} ({self.htf}) ratio={ratio:.2f} (need {self.atr_ratio_min:.2f}) - {'✓ PASS' if is_moving else '✗ FAIL'}")

            return is_moving

        except Exception as e:
            if debug:
                tg(f"  ↳ ATR ratio check: Error ({e}) - PASS")
            return True  # Pass if error

    def check_spread(self, ex, symbol: str) -> bool:
        """Check if bid-ask spread is acceptable."""
        try:
            ticker = ex.fetch_ticker(symbol)
            bid = float(ticker.get("bid") or 0)
            ask = float(ticker.get("ask") or 0)

            if bid <= 0 or ask <= 0:
                return True  # No spread data, pass

            spread_pct = abs(ask - bid) / bid * 100
            return spread_pct <= self.spread_max_pct

        except Exception:
            return True

    # ========== LAYER 4: MULTI-TIMEFRAME ALIGNMENT ==========

    def check_mtf_rsi(self, ex, symbol: str, side: str) -> bool:
        """Check 15m RSI is not overbought/oversold."""
        try:
            ohlcv = fetch_candles(ex, symbol, timeframe=self.htf, limit=50)
            arr = np.asarray(ohlcv, dtype=float)
            _, _, _, _, c, _ = arr.T

            rsi_vals = _rsi(c, self.rsi_period)
            if len(rsi_vals) < 5:
                return True

            current_rsi = float(rsi_vals[-1])

            if side == "long" and current_rsi > self.rsi_ob_threshold:
                return False  # Overbought, don't long
            if side == "short" and current_rsi < self.rsi_os_threshold:
                return False  # Oversold, don't short

            return True

        except Exception:
            return True

    def check_entry_candle_quality(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, 
                                   c: np.ndarray, trigger_i: int, side: str) -> bool:
        """Check if entry candle is strong (body > 40%, closes in top/bottom 40%)."""
        try:
            candle_range = float(h[trigger_i] - l[trigger_i])
            body_size = abs(float(c[trigger_i] - o[trigger_i]))

            if candle_range < 1e-9:
                return False

            # IMPROVED: 40% body ratio (stronger quality filter)
            body_ratio = body_size / candle_range
            if body_ratio < 0.40:
                return False  # Weak candle

            # SCALPER FIX: Check if closes in top/bottom 40% (more relaxed)
            if side == "long":
                close_position = (c[trigger_i] - l[trigger_i]) / candle_range
                return close_position >= 0.60  # Top 40%
            else:
                close_position = (h[trigger_i] - c[trigger_i]) / candle_range
                return close_position >= 0.60  # Bottom 40%

        except Exception:
            return True

    # ========== MAIN ANALYSIS ==========

    def analyze_symbol(self, ex, symbol: str) -> Optional[Dict]:
        """
        Main analysis function: runs all 4 layers and returns signal if all pass.
        Returns dict with signal details or None.
        """
        try:
            debug = _env_bool("SCALP_DEBUG", False)

            if debug:
                tg(f"🔎 ANALYZING {symbol}...")

            # Get time weight
            time_weight = self.get_time_weight()

            # Layer 1: Market Structure
            htf_trend = self.check_htf_trend(ex, symbol)
            mtf_bias = self.check_mtf_bias(ex, symbol)

            if debug:
                tg(f"🔍 {symbol} | L1: HTF={htf_trend} MTF={mtf_bias} Time={time_weight:.0%}")

            # RELAXED: Allow neutral HTF if MTF has strong bias
            # For scalping, we prioritize MTF bias and allow HTF to be neutral
            if mtf_bias == "neutral":
                if debug:
                    tg(f"❌ {symbol} | L1 FAIL: MTF bias is neutral (need directional bias)")
                return None  # Must have at least MTF bias

            # SCALPER FIX: Only check HTF if it has a directional opinion
            # If HTF is neutral, we rely purely on MTF bias for direction
            if htf_trend != "neutral" and htf_trend != mtf_bias:
                # HTF has opinion that conflicts with MTF - be cautious
                if debug:
                    tg(f"⚠️ {symbol} | L1: HTF ({htf_trend}) conflicts with MTF ({mtf_bias}) - allowing for scalping")
                # For scalping, we still allow this but note the conflict
                # The aggressive MTF bias takes priority

            # Use MTF bias as primary direction (more responsive for scalping)
            trade_direction = mtf_bias

            # Fetch LTF data for pattern detection (increased limit for volume analysis)
            ltf_ohlcv = fetch_candles(ex, symbol, timeframe=self.ltf, limit=150)
            ltf_arr = np.asarray(ltf_ohlcv, dtype=float)
            ts, o, h, l, c, v = ltf_arr.T

            # Fetch MTF data for some patterns
            mtf_ohlcv = fetch_candles(ex, symbol, timeframe=self.mtf, limit=50)
            mtf_arr = np.asarray(mtf_ohlcv, dtype=float)
            mtf_ts, mtf_o, mtf_h, mtf_l, mtf_c, mtf_v = mtf_arr.T

            # Layer 2: Detect ALL enabled patterns
            all_patterns = []

            if self.enable_fvg:
                fvgs = self.detect_fvg(o, h, l, c, lookback=100, symbol=symbol)
                all_patterns.extend(fvgs)
                if debug:
                    if fvgs:
                        strategy_type = fvgs[0].get("strategy_type", "unknown") if fvgs else "unknown"
                        entry_idx = fvgs[0].get("i", -1)
                        fvg_formation_idx = fvgs[0].get("fvg_i", -1)
                        tg(f"  ↳ FVG {strategy_type}: Last candle (idx={entry_idx}) touching FVG formed at idx={fvg_formation_idx}")
                    else:
                        tg(f"  ↳ No FVG: Last candle not touching any valid gaps")

            if self.enable_sd:
                sd_zones = self.detect_sd_zones(mtf_o, mtf_h, mtf_l, mtf_c, mtf_v, lookback=30)
                all_patterns.extend(sd_zones)
                if debug:
                    if sd_zones:
                        tg(f"  ↳ Found {len(sd_zones)} S/D zones")
                    else:
                        tg(f"  ↳ No S/D zones detected")

            if self.enable_trendline:
                trendlines = self.detect_trendline_break(mtf_h, mtf_l, mtf_c, mtf_v, lookback=30)
                all_patterns.extend(trendlines)
                if debug:
                    if trendlines:
                        tg(f"  ↳ Found {len(trendlines)} trendline breaks")
                    else:
                        tg(f"  ↳ No trendline breaks detected")

            if self.enable_double:
                doubles = self.detect_double_touch(h, l, lookback=20)
                all_patterns.extend(doubles)
                if debug:
                    if doubles:
                        tg(f"  ↳ Found {len(doubles)} double touch patterns")
                    else:
                        tg(f"  ↳ No double touch patterns detected")

            if not all_patterns:
                if debug:
                    tg(f"❌ {symbol} | L2 FAIL: No patterns detected (FVG:{self.enable_fvg}, S/D:{self.enable_sd}, TL:{self.enable_trendline}, DT:{self.enable_double})")
                return None  # No patterns detected

            # Filter patterns by trade direction
            valid_patterns = [p for p in all_patterns if p["side"] == trade_direction]
            if not valid_patterns:
                if debug:
                    tg(f"❌ {symbol} | L2 FAIL: {len(all_patterns)} patterns found but none match {trade_direction}")
                return None

            if debug:
                tg(f"✅ {symbol} | L2 PASS: {len(valid_patterns)}/{len(all_patterns)} patterns match {trade_direction}")

            # Pick most recent pattern
            pattern = sorted(valid_patterns, key=lambda x: x["i"])[-1]

            # Layer 3: Confirmation Filters
            trigger_i = pattern["i"]

            if debug:
                tg(f"🔍 {symbol} | L3: Checking confirmations (trigger candle index: {trigger_i}/{len(v)-1})")

            # Volume check
            vol_ok = self.check_volume_expansion(v, trigger_i)
            if not vol_ok:
                if debug:
                    tg(f"❌ {symbol} | L3 FAIL: Insufficient volume expansion")
                return None

            # ATR ratio check
            atr_ok = self.check_atr_ratio(ex, symbol)
            if not atr_ok:
                if debug:
                    tg(f"❌ {symbol} | L3 FAIL: ATR ratio too low (market not moving)")
                return None

            # Spread check
            spread_ok = self.check_spread(ex, symbol)
            if not spread_ok:
                if debug:
                    tg(f"❌ {symbol} | L3 FAIL: Spread too wide")
                return None

            if debug:
                tg(f"✅ {symbol} | L3 PASS: Vol✓ ATR✓ Spread✓")

            # Layer 4: Multi-Timeframe Alignment
            rsi_ok = self.check_mtf_rsi(ex, symbol, pattern["side"])
            if not rsi_ok:
                if debug:
                    tg(f"❌ {symbol} | L4 FAIL: RSI overbought/oversold")
                return None

            candle_ok = self.check_entry_candle_quality(o, h, l, c, trigger_i, pattern["side"])
            if not candle_ok:
                if debug:
                    tg(f"❌ {symbol} | L4 FAIL: Entry candle quality insufficient")
                return None

            if debug:
                tg(f"✅ {symbol} | L4 PASS: RSI✓ Candle✓")

            # Calculate entry, SL, TP
            entry = float(c[-1])

            # Calculate SL based on pattern type
            is_fvg = "fvg" in pattern.get("pattern", "")

            if is_fvg:
                # FVG-SPECIFIC SL/TP LOGIC (matches NY Open FVG strategy)
                gap_lo = float(pattern["lo"])
                gap_hi = float(pattern["hi"])
                gap_size = float(pattern.get("gap_size", gap_hi - gap_lo))
                strategy_type = pattern.get("strategy_type", "continuation")

                # SL placement based on FVG type and strategy
                sl_buffer_pct = 0.1  # 0.1% buffer beyond gap boundary for safety

                if pattern["side"] == "long":
                    # Long entry - SL below the FVG gap low
                    sl = gap_lo * (1 - sl_buffer_pct / 100)
                    if debug:
                        tg(f"  ↳ FVG LONG: Entry=${entry:.2f}, Gap=${gap_lo:.2f}-${gap_hi:.2f}, SL=${sl:.2f} (below gap)")
                else:
                    # Short entry - SL above the FVG gap high
                    sl = gap_hi * (1 + sl_buffer_pct / 100)
                    if debug:
                        tg(f"  ↳ FVG SHORT: Entry=${entry:.2f}, Gap=${gap_lo:.2f}-${gap_hi:.2f}, SL=${sl:.2f} (above gap)")

                # TP calculation based on gap size (matches NY Open FVG)
                # Use FVG_RR from config (default 1.5 from .env.scalper)
                fvg_rr = float(os.getenv("FVG_RR", "1.5") or 1.5)

                # Risk is entry to SL distance
                risk = abs(entry - sl)

                # For FVG, we use multiple TPs but scale from the FVG_RR base
                tp1_rr = fvg_rr  # Use base FVG_RR for TP1
                tp2_rr = fvg_rr * 1.67  # ~2.5x for TP2
                tp3_rr = fvg_rr * 2.33  # ~3.5x for TP3

                if pattern["side"] == "long":
                    tp1 = entry + risk * tp1_rr
                    tp2 = entry + risk * tp2_rr
                    tp3 = entry + risk * tp3_rr
                else:
                    tp1 = entry - risk * tp1_rr
                    tp2 = entry - risk * tp2_rr
                    tp3 = entry - risk * tp3_rr

                if debug:
                    tg(f"  ↳ FVG Risk/Reward: Risk=${risk:.2f} ({risk/entry*100:.2f}%), TP1=${tp1:.2f} (RR={tp1_rr:.1f}), TP2=${tp2:.2f}, TP3=${tp3:.2f}")

            else:
                # NON-FVG PATTERNS: use pattern boundaries with buffer
                if pattern["side"] == "long":
                    sl = pattern["lo"] * 0.999  # Just below pattern low
                else:
                    sl = pattern["hi"] * 1.001  # Just above pattern high

                # Validate SL distance for non-FVG patterns
                sl_distance_pct = abs(entry - sl) / entry * 100
                if sl_distance_pct < self.min_sl_pct:
                    sl = entry * (1 - self.min_sl_pct / 100) if pattern["side"] == "long" else entry * (1 + self.min_sl_pct / 100)
                elif sl_distance_pct > self.max_sl_pct:
                    if debug:
                        tg(f"❌ {symbol} | Rejected: SL too far ({sl_distance_pct:.2f}% > {self.max_sl_pct}%)")
                    return None  # SL too far, skip

                # Calculate TPs using configured RR ratios
                tp1_rr = float(os.getenv("SCALP_TP1_RR", "1.5") or 1.5)
                tp2_rr = float(os.getenv("SCALP_TP2_RR", "2.5") or 2.5)
                tp3_rr = float(os.getenv("SCALP_TP3_RR", "3.5") or 3.5)

                risk = abs(entry - sl)

                if pattern["side"] == "long":
                    tp1 = entry + risk * tp1_rr
                    tp2 = entry + risk * tp2_rr
                    tp3 = entry + risk * tp3_rr
                else:
                    tp1 = entry - risk * tp1_rr
                    tp2 = entry - risk * tp2_rr
                    tp3 = entry - risk * tp3_rr

            # Final validation: ensure SL is not too tight (min 0.2% for any pattern type)
            final_sl_distance_pct = abs(entry - sl) / entry * 100
            if final_sl_distance_pct < 0.2:
                if debug:
                    tg(f"❌ {symbol} | Rejected: SL too tight ({final_sl_distance_pct:.3f}% < 0.2%)")
                return None

            # Apply time weight to confidence
            base_confidence = 0.75
            final_confidence = base_confidence * time_weight

            # Add FVG-specific details to signal if this is an FVG pattern
            fvg_details = {}
            if "fvg" in pattern.get("pattern", ""):
                fvg_details = {
                    "gap_lo": pattern.get("lo", 0),
                    "gap_hi": pattern.get("hi", 0),
                    "gap_size": pattern.get("gap_size", 0),
                    "fvg_side": pattern.get("fvg_side", "unknown"),
                    "strategy_type": pattern.get("strategy_type", "unknown"),
                    "last_candle_high": float(h[trigger_i]) if trigger_i < len(h) else 0,
                    "last_candle_low": float(l[trigger_i]) if trigger_i < len(l) else 0,
                    "last_candle_close": float(c[trigger_i]) if trigger_i < len(c) else entry,
                }

            signal = {
                "symbol": symbol,
                "side": pattern["side"],
                "entry": entry,
                "sl": float(sl),
                "tp1": float(tp1),
                "tp2": float(tp2),
                "tp3": float(tp3),
                "pattern": pattern["pattern"],
                "htf_trend": htf_trend,
                "mtf_bias": mtf_bias,
                "confidence": final_confidence,
                "time_weight": time_weight,
                "trigger_i": trigger_i,
                "fvg_details": fvg_details,  # Add FVG context
                "fvg_formation_i": pattern.get("fvg_i", -1),  # Track FVG formation index for caching
            }

            if debug:
                tg(f"🎯 {symbol} | SIGNAL GENERATED | {pattern['side'].upper()} | {pattern['pattern']}")

            return signal

        except Exception as e:
            tg(f"⚠️ Scalper analysis error for {symbol}: {e}")
            import traceback
            if _env_bool("SCALP_DEBUG", False):
                tg(f"🔍 Traceback: {traceback.format_exc()}")
            return None

    def mark_fvg_used(self, symbol: str, fvg_formation_i: int):
        """Mark an FVG pattern as used to prevent re-trading."""
        if symbol not in self.used_fvg_indices:
            self.used_fvg_indices[symbol] = []

        # Add to used list
        self.used_fvg_indices[symbol].append(fvg_formation_i)

        # Keep only recent N indices (prevent memory bloat)
        if len(self.used_fvg_indices[symbol]) > self.fvg_pattern_cache_size:
            self.used_fvg_indices[symbol] = self.used_fvg_indices[symbol][-self.fvg_pattern_cache_size:]

    def log_signal(self, signal: Dict):
        """Log signal to Telegram with full details including FVG information."""
        sym = signal["symbol"]
        side = signal["side"].upper()
        pattern = signal["pattern"].replace("_", " ").upper()
        entry = signal["entry"]
        sl = signal["sl"]
        tp1 = signal["tp1"]
        tp2 = signal["tp2"]
        tp3 = signal["tp3"]
        conf = signal["confidence"]
        time_weight = signal["time_weight"]

        risk = abs(entry - sl)
        reward1 = abs(tp1 - entry)

        # Build FVG details section for FVG patterns
        fvg_details = ""
        strategy_indicator = ""
        if "fvg" in pattern.lower():
            # Extract FVG info from signal (added by analyze_symbol)
            fvg_info = signal.get("fvg_details", {})

            if "continuation" in pattern.lower():
                strategy_indicator = " 🔄 CONTINUATION"
                entry_logic = "Price entered FVG gap and closed THROUGH it in FVG direction"
            elif "inversion" in pattern.lower():
                strategy_indicator = " 🔀 INVERSION"
                entry_logic = "Price entered FVG gap and closed OPPOSITE to FVG direction (rejection)"
            else:
                entry_logic = "FVG entry"

            # FVG gap boundaries
            gap_lo = fvg_info.get("gap_lo", 0)
            gap_hi = fvg_info.get("gap_hi", 0)
            gap_size = fvg_info.get("gap_size", 0)
            fvg_side = fvg_info.get("fvg_side", "unknown")

            # Last candle details
            last_high = fvg_info.get("last_candle_high", 0)
            last_low = fvg_info.get("last_candle_low", 0)
            last_close = fvg_info.get("last_candle_close", entry)

            if gap_lo and gap_hi:
                gap_pct = (gap_size / ((gap_lo + gap_hi) / 2)) * 100
                fvg_direction_emoji = "📈" if fvg_side == "long" else "📉"

                # Show which gap boundary is used for SL
                if side == "LONG":
                    sl_reference = f"SL below gap low (${gap_lo:.2f})"
                else:
                    sl_reference = f"SL above gap high (${gap_hi:.2f})"

                fvg_details = (
                    f"\n━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🎯 <b>FVG ENTRY DETAILS</b>\n"
                    f"{fvg_direction_emoji} FVG Type: {fvg_side.upper()} GAP {strategy_indicator}\n"
                    f"📏 Gap: ${gap_lo:.2f} - ${gap_hi:.2f} (${gap_size:.2f} / {gap_pct:.2f}%)\n"
                    f"🕯️ Trigger Candle: H=${last_high:.2f} L=${last_low:.2f} C=${last_close:.2f}\n"
                    f"🛡️ {sl_reference}\n"
                    f"💡 Logic: {entry_logic}"
                )

        # Convert long/short to bullish/bearish for display
        htf_display = signal['htf_trend'].replace('long', 'BULLISH').replace('short', 'BEARISH').upper()
        mtf_display = signal['mtf_bias'].replace('long', 'BULLISH').replace('short', 'BEARISH').upper()

        # Trend alignment indicator
        if signal['htf_trend'] == signal['mtf_bias']:
            trend_alignment = "✅ ALIGNED"
        elif signal['htf_trend'] == "neutral":
            trend_alignment = "⚠️ HTF NEUTRAL"
        else:
            trend_alignment = "⚠️ DIVERGENT"

        tg(
            f"🎯 <b>SCALP SIGNAL</b> {sym} [{side}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Pattern: {pattern}{strategy_indicator if not fvg_details else ''}\n"
            f"🕐 Session: {time_weight:.0%} confidence\n"
            f"📈 HTF (15m): {htf_display}\n"
            f"📊 MTF (5m): {mtf_display} {trend_alignment}"
            f"{fvg_details}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Entry: ${entry:.2f}\n"
            f"🛑 Stop Loss: ${sl:.2f} ({-risk/entry*100:.2f}%)\n"
            f"🎯 TP1: ${tp1:.2f} (+{reward1/entry*100:.2f}%) [50%]\n"
            f"🎯 TP2: ${tp2:.2f} [30%]\n"
            f"🎯 TP3: ${tp3:.2f} [20%]\n"
            f"🎲 Confidence: {conf:.0%}"
        )
