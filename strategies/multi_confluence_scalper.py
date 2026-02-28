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

    def detect_fvg(self, h: np.ndarray, l: np.ndarray, c: np.ndarray, 
                   lookback: int = 50) -> List[Dict]:
        """
        Detect Fair Value Gaps on 1m with quality filter.
        FVG gap size must be reasonable relative to ATR (not too small = noise, not too large = likely fills).
        """
        fvgs = []
        n = len(h)
        start = max(2, n - lookback)

        # Calculate ATR for quality filter
        atr_vals = _atr(h, l, c, 14)

        # FVG quality thresholds (relative to ATR)
        min_gap_atr_ratio = float(os.getenv("SCALP_FVG_MIN_GAP_ATR", "0.3") or 0.3)
        max_gap_atr_ratio = float(os.getenv("SCALP_FVG_MAX_GAP_ATR", "2.0") or 2.0)

        for i in range(start, n):
            try:
                current_atr = float(atr_vals[i]) if len(atr_vals) > i else 0
                if current_atr < 1e-9:
                    continue  # Can't validate quality without ATR

                # Bullish FVG
                if l[i] > h[i - 2]:
                    gap_size = float(l[i] - h[i - 2])
                    gap_atr_ratio = gap_size / current_atr

                    # Quality filter: gap must be meaningful but not extreme
                    if min_gap_atr_ratio <= gap_atr_ratio <= max_gap_atr_ratio:
                        fvgs.append({
                            "i": i,
                            "side": "long",
                            "lo": float(h[i - 2]),
                            "hi": float(l[i]),
                            "pattern": "fvg",
                            "gap_quality": gap_atr_ratio
                        })

                # Bearish FVG
                if h[i] < l[i - 2]:
                    gap_size = float(l[i - 2] - h[i])
                    gap_atr_ratio = gap_size / current_atr

                    if min_gap_atr_ratio <= gap_atr_ratio <= max_gap_atr_ratio:
                        fvgs.append({
                            "i": i,
                            "side": "short",
                            "lo": float(h[i]),
                            "hi": float(l[i - 2]),
                            "pattern": "fvg",
                            "gap_quality": gap_atr_ratio
                        })
            except Exception:
                continue

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
            if c[i] > o[i] and (c[i] - o[i]) / max(1e-9, h[i] - l[i]) > 0.7:
                # Check volume expansion
                if i >= 10:
                    avg_vol = np.mean(v[max(0, i-10):i])
                    if v[i] > avg_vol * 1.5:
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
            if c[i] < o[i] and (o[i] - c[i]) / max(1e-9, h[i] - l[i]) > 0.7:
                if i >= 10:
                    avg_vol = np.mean(v[max(0, i-10):i])
                    if v[i] > avg_vol * 1.5:
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
                    if l[i] < last_low * 0.998:  # Break below with 0.2% buffer
                        # Check volume expansion
                        if i >= 10:
                            avg_vol = np.mean(v[max(0, i-10):i])
                            if v[i] > avg_vol * 1.3:
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
                    if h[i] > last_high * 1.002:  # Break above with 0.2% buffer
                        if i >= 10:
                            avg_vol = np.mean(v[max(0, i-10):i])
                            if v[i] > avg_vol * 1.3:
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
                if abs(l[i] - l[j]) / l[i] < 0.002:  # Within 0.2%
                    # Check if there's a higher low between them
                    if all(l[k] >= min(l[i], l[j]) * 1.003 for k in range(i+1, j)):
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
                if abs(h[i] - h[j]) / h[i] < 0.002:
                    if all(h[k] <= max(h[i], h[j]) * 0.997 for k in range(i+1, j)):
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
        """Check if entry candle is strong (body > 50%, closes in extreme)."""
        try:
            candle_range = float(h[trigger_i] - l[trigger_i])
            body_size = abs(float(c[trigger_i] - o[trigger_i]))

            if candle_range < 1e-9:
                return False

            body_ratio = body_size / candle_range
            if body_ratio < 0.5:
                return False  # Weak candle

            # Check if closes in extreme 30%
            if side == "long":
                close_position = (c[trigger_i] - l[trigger_i]) / candle_range
                return close_position >= 0.7  # Top 30%
            else:
                close_position = (h[trigger_i] - c[trigger_i]) / candle_range
                return close_position >= 0.7  # Bottom 30%

        except Exception:
            return True

    # ========== MAIN ANALYSIS ==========

    def analyze_symbol(self, ex, symbol: str) -> Optional[Dict]:
        """
        Main analysis function: runs all 4 layers and returns signal if all pass.
        Returns dict with signal details or None.
        """
        try:
            # Get time weight
            time_weight = self.get_time_weight()

            # Layer 1: Market Structure
            htf_trend = self.check_htf_trend(ex, symbol)
            mtf_bias = self.check_mtf_bias(ex, symbol)

            if htf_trend == "neutral" or mtf_bias == "neutral":
                return None  # No clear trend

            if htf_trend != mtf_bias:
                return None  # Conflicting trends

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
                fvgs = self.detect_fvg(h, l, c, lookback=50)
                all_patterns.extend(fvgs)

            if self.enable_sd:
                sd_zones = self.detect_sd_zones(mtf_o, mtf_h, mtf_l, mtf_c, mtf_v, lookback=30)
                all_patterns.extend(sd_zones)

            if self.enable_trendline:
                trendlines = self.detect_trendline_break(mtf_h, mtf_l, mtf_c, mtf_v, lookback=30)
                all_patterns.extend(trendlines)

            if self.enable_double:
                doubles = self.detect_double_touch(h, l, lookback=20)
                all_patterns.extend(doubles)

            if not all_patterns:
                return None  # No patterns detected

            # Filter patterns by trend alignment
            valid_patterns = [p for p in all_patterns if p["side"] == htf_trend]
            if not valid_patterns:
                return None

            # Pick most recent pattern
            pattern = sorted(valid_patterns, key=lambda x: x["i"])[-1]

            # Layer 3: Confirmation Filters
            trigger_i = pattern["i"]

            # Volume check
            if not self.check_volume_expansion(v, trigger_i):
                return None

            # ATR ratio check
            if not self.check_atr_ratio(ex, symbol):
                return None

            # Spread check
            if not self.check_spread(ex, symbol):
                return None

            # Layer 4: Multi-Timeframe Alignment
            if not self.check_mtf_rsi(ex, symbol, pattern["side"]):
                return None

            if not self.check_entry_candle_quality(o, h, l, c, trigger_i, pattern["side"]):
                return None

            # Calculate entry, SL, TP
            entry = float(c[-1])

            # Calculate SL based on pattern
            if pattern["side"] == "long":
                sl = pattern["lo"] * 0.999  # Just below pattern low
            else:
                sl = pattern["hi"] * 1.001  # Just above pattern high

            # Validate SL distance
            sl_distance_pct = abs(entry - sl) / entry * 100
            if sl_distance_pct < self.min_sl_pct:
                sl = entry * (1 - self.min_sl_pct / 100) if pattern["side"] == "long" else entry * (1 + self.min_sl_pct / 100)
            elif sl_distance_pct > self.max_sl_pct:
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

            return signal

        except Exception as e:
            tg(f"⚠️ Scalper analysis error for {symbol}: {e}")
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

        tg(
            f"🎯 <b>SCALP SIGNAL</b> {sym} [{side}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Pattern: {pattern}\n"
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
