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
        self.atr_ratio_min = float(os.getenv("SCALP_ATR_RATIO_MIN", "0.5") or 0.5)
        self.spread_max_pct = float(os.getenv("SCALP_SPREAD_MAX_PCT", "0.05") or 0.05)

        # Layer 4: Multi-Timeframe
        self.rsi_period = 14
        self.rsi_ob_threshold = 75
        self.rsi_os_threshold = 25

        # Risk Management
        self.min_sl_pct = float(os.getenv("SCALP_MIN_SL_PCT", "0.3") or 0.3)
        self.max_sl_pct = float(os.getenv("SCALP_MAX_SL_PCT", "0.8") or 0.8)

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
        Returns: 'bullish', 'bearish', or 'neutral'
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
                return "bullish"
            elif current_close < current_ema50 < current_ema200:
                return "bearish"
            else:
                return "neutral"
        except Exception as e:
            tg(f"⚠️ HTF trend check error for {symbol}: {e}")
            return "neutral"

    def check_mtf_bias(self, ex, symbol: str) -> Optional[str]:
        """
        Check 5m bias using candle color consistency.
        Returns: 'bullish', 'bearish', or 'neutral'
        """
        try:
            ohlcv = fetch_candles(ex, symbol, timeframe=self.mtf, limit=20)
            arr = np.asarray(ohlcv, dtype=float)
            _, o, _, _, c, _ = arr.T

            recent_candles = min(self.bias_candle_count, len(c))
            green_count = sum(1 for i in range(-recent_candles, 0) if c[i] > o[i])
            red_count = recent_candles - green_count

            if green_count >= recent_candles * 0.6:
                return "bullish"
            elif red_count >= recent_candles * 0.6:
                return "bearish"
            else:
                return "neutral"
        except Exception:
            return "neutral"

    # ========== LAYER 2: PATTERN DETECTION ==========

    def detect_fvg(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, 
                   lookback: int = 50) -> List[Dict]:
        """
        Detect Fair Value Gaps on 1m with NY Open style ATR-based filtering.

        Uses same logic as NY Open FVG strategy:
        1. Detect FVG formation (3-candle pattern)
        2. Filter by ATR-based minimum gap size
        3. Monitor subsequent candles for entry (continuation or inversion)
        4. SL at gap boundary, TP based on RR
        """
        from datahub import atr as calc_atr

        fvgs = []
        n = len(h)
        start = max(2, n - lookback)

        # ATR-based minimum gap size filter (NY Open style)
        min_atr_mult = float(os.getenv("MIN_FVG_ATR_MULTIPLIER", "0.3") or 0.3)
        atr_period = int(os.getenv("FVG_ATR_PERIOD", "14") or 14)

        # Calculate ATR for the entire series
        atr_vals = calc_atr(h, l, c, period=atr_period)

        # Fallback filters
        min_gap_pct = float(os.getenv("FVG_MIN_GAP_PCT", "0.03") or 0.03)
        min_gap_points = float(os.getenv("FVG_MIN_GAP_POINTS", "0.01") or 0.01)

        # First pass: detect all FVG formations
        detected_fvgs = []
        for i in range(start, n):
            try:
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

        # Second pass: Monitor the MOST RECENT FVG for entry (continuation or inversion)
        # This matches NY Open logic of always using the latest FVG
        fvg = detected_fvgs[-1]
        fvg_side = fvg["fvg_side"]
        i0 = fvg["i"]
        gap_lo = fvg["lo"]
        gap_hi = fvg["hi"]
        gap_size = fvg["gap_size"]

        # Monitor candles after FVG formation for entry
        min_delay = 1  # Wait at least 1 candle after FVG formation
        scan_start = i0 + min_delay

        if scan_start >= len(c):
            return []  # FVG too recent, no entry yet

        # Scan candles after FVG for entry signal
        for t in range(scan_start, len(c)):
            candle_high = float(h[t])
            candle_low = float(l[t])
            candle_close = float(c[t])

            # Check if candle enters FVG zone (wick touch)
            candle_enters = not (candle_high < gap_lo or candle_low > gap_hi)

            if not candle_enters:
                continue

            # Determine entry based on close direction
            entry_signal = None

            if fvg_side == "long":
                # BULLISH FVG - two possibilities:

                # 1. CONTINUATION: Close above gap → LONG
                if candle_close > gap_hi:
                    entry_signal = {
                        "trade_side": "long",
                        "strategy_type": "continuation",
                    }

                # 2. INVERSION: Close below gap → SHORT
                elif candle_close < gap_lo:
                    entry_signal = {
                        "trade_side": "short",
                        "strategy_type": "inversion",
                    }

            else:  # fvg_side == "short"
                # BEARISH FVG - two possibilities:

                # 1. CONTINUATION: Close below gap → SHORT
                if candle_close < gap_lo:
                    entry_signal = {
                        "trade_side": "short",
                        "strategy_type": "continuation",
                    }

                # 2. INVERSION: Close above gap → LONG
                elif candle_close > gap_hi:
                    entry_signal = {
                        "trade_side": "long",
                        "strategy_type": "inversion",
                    }

            # If we have entry, return it
            if entry_signal:
                fvgs.append({
                    "i": t,  # Entry candle index
                    "fvg_i": i0,  # FVG formation index
                    "side": entry_signal["trade_side"],
                    "fvg_side": fvg_side,
                    "strategy_type": entry_signal["strategy_type"],
                    "lo": gap_lo,
                    "hi": gap_hi,
                    "gap_size": gap_size,
                    "pattern": f"fvg_{entry_signal['strategy_type']}"
                })
                break  # Take first valid entry

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
        if trigger_i < self.vol_lookback:
            return True  # Not enough data, pass

        trigger_vol = float(v[trigger_i])
        avg_vol = float(np.mean(v[max(0, trigger_i - self.vol_lookback):trigger_i]))

        if avg_vol < 1e-9:
            return True

        return trigger_vol >= avg_vol * self.vol_mult

    def check_atr_ratio(self, ex, symbol: str) -> bool:
        """Check if LTF ATR is sufficient vs HTF ATR (market is moving)."""
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
                return True

            current_ltf_atr = float(ltf_atr[-1])
            current_htf_atr = float(htf_atr[-1])

            if current_htf_atr < 1e-9:
                return True

            ratio = current_ltf_atr / current_htf_atr
            return ratio >= self.atr_ratio_min

        except Exception:
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

            # RELAXED: 35% body ratio (was 40%)
            body_ratio = body_size / candle_range
            if body_ratio < 0.35:
                return False  # Weak candle

            # RELAXED: Check if closes in top/bottom 35% (was 40%)
            if side == "long":
                close_position = (c[trigger_i] - l[trigger_i]) / candle_range
                return close_position >= 0.65  # Top 35%
            else:
                close_position = (h[trigger_i] - c[trigger_i]) / candle_range
                return close_position >= 0.65  # Bottom 35%

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

            # Get time weight
            time_weight = self.get_time_weight()

            # Layer 1: Market Structure
            htf_trend = self.check_htf_trend(ex, symbol)
            mtf_bias = self.check_mtf_bias(ex, symbol)

            if debug:
                tg(f"🔍 {symbol} | L1: HTF={htf_trend} MTF={mtf_bias} Time={time_weight:.0%}")

            # RELAXED: Allow neutral HTF if MTF has strong bias
            # This helps scalping strategy be more active while still having direction
            if mtf_bias == "neutral":
                if debug:
                    tg(f"❌ {symbol} | L1 FAIL: MTF bias is neutral")
                return None  # Must have at least MTF bias

            # Determine trading direction
            if htf_trend != "neutral" and htf_trend != mtf_bias:
                if debug:
                    tg(f"❌ {symbol} | L1 FAIL: HTF conflicts with MTF (HTF={htf_trend}, MTF={mtf_bias})")
                return None  # If HTF has opinion, it must agree with MTF

            # Use MTF bias as primary direction (more responsive for scalping)
            trade_direction = mtf_bias

            # Fetch LTF data for pattern detection
            ltf_ohlcv = fetch_candles(ex, symbol, timeframe=self.ltf, limit=100)
            ltf_arr = np.asarray(ltf_ohlcv, dtype=float)
            ts, o, h, l, c, v = ltf_arr.T

            # Fetch MTF data for some patterns
            mtf_ohlcv = fetch_candles(ex, symbol, timeframe=self.mtf, limit=50)
            mtf_arr = np.asarray(mtf_ohlcv, dtype=float)
            mtf_ts, mtf_o, mtf_h, mtf_l, mtf_c, mtf_v = mtf_arr.T

            # Layer 2: Detect ALL enabled patterns
            all_patterns = []

            if self.enable_fvg:
                fvgs = self.detect_fvg(o, h, l, c, lookback=50)
                all_patterns.extend(fvgs)
                if debug and fvgs:
                    strategy_type = fvgs[0].get("strategy_type", "unknown") if fvgs else "unknown"
                    tg(f"  ↳ Found {len(fvgs)} FVG patterns ({strategy_type})")

            if self.enable_sd:
                sd_zones = self.detect_sd_zones(mtf_o, mtf_h, mtf_l, mtf_c, mtf_v, lookback=30)
                all_patterns.extend(sd_zones)
                if debug and sd_zones:
                    tg(f"  ↳ Found {len(sd_zones)} S/D zones")

            if self.enable_trendline:
                trendlines = self.detect_trendline_break(mtf_h, mtf_l, mtf_c, mtf_v, lookback=30)
                all_patterns.extend(trendlines)
                if debug and trendlines:
                    tg(f"  ↳ Found {len(trendlines)} trendline breaks")

            if self.enable_double:
                doubles = self.detect_double_touch(h, l, lookback=20)
                all_patterns.extend(doubles)
                if debug and doubles:
                    tg(f"  ↳ Found {len(doubles)} double touch patterns")

            if not all_patterns:
                if debug:
                    tg(f"❌ {symbol} | L2 FAIL: No patterns detected")
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
                # NY Open FVG style: SL at gap boundary
                if pattern["side"] == "long":
                    sl = float(pattern["lo"])  # SL below gap
                else:
                    sl = float(pattern["hi"])  # SL above gap
            else:
                # Other patterns: use pattern boundaries with buffer
                if pattern["side"] == "long":
                    sl = pattern["lo"] * 0.999  # Just below pattern low
                else:
                    sl = pattern["hi"] * 1.001  # Just above pattern high

            # Validate SL distance
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

            # Apply time weight to confidence
            base_confidence = 0.75
            final_confidence = base_confidence * time_weight

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

    def log_signal(self, signal: Dict):
        """Log signal to Telegram with full details."""
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

        # Add strategy type indicator for FVG patterns
        strategy_indicator = ""
        if "fvg" in pattern.lower():
            if "continuation" in pattern.lower():
                strategy_indicator = " 🔄 CONTINUATION"
            elif "inversion" in pattern.lower():
                strategy_indicator = " 🔀 INVERSION"

        tg(
            f"🎯 <b>SCALP SIGNAL</b> {sym} [{side}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Pattern: {pattern}{strategy_indicator}\n"
            f"🕐 Time Weight: {time_weight:.0%}\n"
            f"📈 Trend: {signal['htf_trend'].upper()} (15m) | {signal['mtf_bias'].upper()} (5m)\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Entry: ${entry:.2f}\n"
            f"SL: ${sl:.2f} (-{(risk/entry*100):.2f}%)\n"
            f"TP1: ${tp1:.2f} (+{(reward1/entry*100):.2f}%) [50%]\n"
            f"TP2: ${tp2:.2f} [30%]\n"
            f"TP3: ${tp3:.2f} [20%]\n"
            f"Confidence: {conf:.0%}"
        )
