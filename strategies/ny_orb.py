import os
import time
from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timezone

from utils import tg
from datahub import fetch_candles, opening_range, utc_anchor_for_session


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


class NyOrbInspector:
    """
    NY Opening Range Breakout (ORB) Strategy:
    1. Define OR: 14:30-14:45 UTC (configurable)
    2. Wait for a 5m candle close outside OR High/Low
    3. Confirm EMA9 slope in breakout direction
    4. Enter market order
    5. SL at OR midpoint, TP using RR ratio
    """

    def __init__(self,
                 or_start_hhmm_utc: str = None,
                 or_minutes: int = None,
                 or_timeframe: str = None,
                 signal_timeframe: str = None,
                 check_interval: int = None):
        self.or_start_hhmm_utc = or_start_hhmm_utc or os.getenv("ORB_START_UTC", "14:30")
        self.or_minutes = int(or_minutes or int(os.getenv("ORB_MINUTES", "15") or 15))
        self.or_timeframe = or_timeframe or os.getenv("ORB_TIMEFRAME", "15m")
        self.signal_timeframe = signal_timeframe or os.getenv("ORB_SIGNAL_TIMEFRAME", "5m")
        self.check_interval = int(check_interval or int(os.getenv("ORB_CHECK_INTERVAL", "60") or 60))
        self.ema_period = int(os.getenv("ORB_EMA_PERIOD", "9") or 9)
        self.ema_slope_bars = int(os.getenv("ORB_EMA_SLOPE_BARS", "3") or 3)

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        try:
            if timeframe.endswith('m'):
                return int(timeframe[:-1])
            elif timeframe.endswith('h'):
                return int(timeframe[:-1]) * 60
            elif timeframe.endswith('d'):
                return int(timeframe[:-1]) * 1440
            else:
                return 5
        except Exception:
            return 5

    def analyze_symbol(self, ex, symbol: str, limit: int = 500) -> Optional[Dict]:
        """
        Fetch 5m candles, compute OR, detect breakout with EMA9 confirmation.
        Returns payload dict with diagnostics.
        """
        ohlcv = fetch_candles(ex, symbol, timeframe=self.signal_timeframe, limit=limit)
        arr = np.asarray(ohlcv, dtype=float)
        ts, o, h, l, c, v = arr.T

        # Calculate EMA9
        ema9 = _ema(c, self.ema_period)
        if len(ema9) < len(c):
            ema9 = np.full_like(c, np.nan)

        # OR computation via UTC anchor
        start_epoch = utc_anchor_for_session(self.or_start_hhmm_utc)
        or_res = opening_range(ts, h, l, start_epoch, self.or_minutes)
        orh, orl = (float("nan"), float("nan"))
        or_ready = False
        or_open_i = None
        or_close_i = None
        post_first_candle_i = None

        if or_res:
            orh, orl = or_res
            or_open_ms = float(start_epoch) * 1000.0
            or_close_ms = float(start_epoch + 60 * self.or_minutes) * 1000.0

            # Index of first bar at or after OR open
            or_open_i = next((i for i in range(len(ts)) if ts[i] >= or_open_ms), None)
            # Index of first bar at or after OR close
            or_close_i = next((i for i in range(len(ts)) if ts[i] >= or_close_ms), None)

            # Ready when we have at least one complete candle after OR
            if or_open_i is not None and or_close_i is not None and (or_close_i - or_open_i) >= 1:
                if or_close_i < len(ts) - 1:
                    post_first_candle_i = or_close_i
                    or_ready = True

        # Detect breakout signal
        signal = None
        if or_ready and post_first_candle_i is not None and not np.isnan(orh) and not np.isnan(orl):
            # OR midpoint for SL
            or_mid = (orh + orl) / 2.0

            # Check from first candle after OR to current
            for i in range(post_first_candle_i, len(ts)):
                # Skip if not enough data for EMA slope check
                if i < self.ema_slope_bars:
                    continue

                # Bullish breakout: close above OR high
                if float(c[i]) > orh:
                    # Check EMA9 slope (should be rising)
                    ema_slope_up = all(
                        ema9[i - j] > ema9[i - j - 1] 
                        for j in range(self.ema_slope_bars) 
                        if i - j - 1 >= 0 and not np.isnan(ema9[i - j]) and not np.isnan(ema9[i - j - 1])
                    )

                    if ema_slope_up:
                        # Instead of immediate entry, we place a pending order at the high of this breakout candle
                        breakout_high = float(h[i])
                        order_buffer_pct = float(os.getenv("ORB_ORDER_BUFFER_PCT", "0.0005") or 0.0005)  # 0.05%
                        pending_trigger = breakout_high * (1.0 + order_buffer_pct)

                        # SL/TP will be calculated from pending_trigger when filled
                        sl = or_mid
                        risk = max(1e-9, pending_trigger - sl)
                        rr_cfg = float(os.getenv("ORB_RR", "2.0") or 2.0)
                        tp_final = pending_trigger + rr_cfg * risk

                        # Calculate 3-level TP ladder
                        tp_splits_raw = os.getenv("ORB_TP_SPLITS", "0.33,0.66,1.0")
                        tp_splits = [float(x.strip()) for x in tp_splits_raw.split(",")]

                        tp_distance = tp_final - pending_trigger
                        tp_levels = [pending_trigger + (tp_distance * split) for split in tp_splits]

                        signal = {
                            "side": "long",
                            "breakout_i": i,
                            "breakout_close": float(c[i]),
                            "breakout_high": breakout_high,
                            "pending_trigger": pending_trigger,
                            "invalidation_level": orh,  # If price goes below ORH, cancel order
                            "sl": sl,
                            "tp": tp_final,  # Keep for backward compatibility
                            "tp1": float(tp_levels[0]),
                            "tp2": float(tp_levels[1]),
                            "tp3": float(tp_levels[2]),
                            "rr": rr_cfg,
                            "pattern": "orb_long_pending",
                            "or_high": orh,
                            "or_low": orl,
                        }
                        break

                # Bearish breakout: close below OR low
                elif float(c[i]) < orl:
                    # Check EMA9 slope (should be falling)
                    ema_slope_dn = all(
                        ema9[i - j] < ema9[i - j - 1] 
                        for j in range(self.ema_slope_bars) 
                        if i - j - 1 >= 0 and not np.isnan(ema9[i - j]) and not np.isnan(ema9[i - j - 1])
                    )

                    if ema_slope_dn:
                        # Instead of immediate entry, we place a pending order at the low of this breakout candle
                        breakout_low = float(l[i])
                        order_buffer_pct = float(os.getenv("ORB_ORDER_BUFFER_PCT", "0.0005") or 0.0005)  # 0.05%
                        pending_trigger = breakout_low * (1.0 - order_buffer_pct)

                        # SL/TP will be calculated from pending_trigger when filled
                        sl = or_mid
                        risk = max(1e-9, sl - pending_trigger)
                        rr_cfg = float(os.getenv("ORB_RR", "2.0") or 2.0)
                        tp_final = pending_trigger - rr_cfg * risk

                        # Calculate 3-level TP ladder
                        tp_splits_raw = os.getenv("ORB_TP_SPLITS", "0.33,0.66,1.0")
                        tp_splits = [float(x.strip()) for x in tp_splits_raw.split(",")]

                        tp_distance = pending_trigger - tp_final
                        tp_levels = [pending_trigger - (tp_distance * split) for split in tp_splits]

                        signal = {
                            "side": "short",
                            "breakout_i": i,
                            "breakout_close": float(c[i]),
                            "breakout_low": breakout_low,
                            "pending_trigger": pending_trigger,
                            "invalidation_level": orl,  # If price goes above ORL, cancel order
                            "sl": sl,
                            "tp": tp_final,  # Keep for backward compatibility
                            "tp1": float(tp_levels[0]),
                            "tp2": float(tp_levels[1]),
                            "tp3": float(tp_levels[2]),
                            "rr": rr_cfg,
                            "pattern": "orb_short_pending",
                            "or_high": orh,
                            "or_low": orl,
                        }
                        break

        payload = {
            "symbol": symbol,
            "now": int(time.time()),
            "or_ready": bool(or_ready),
            "or_high": float(orh) if or_res else None,
            "or_low": float(orl) if or_res else None,
            "or_mid": float((orh + orl) / 2.0) if or_res else None,
            "or_open_i": int(or_open_i) if or_open_i is not None else None,
            "or_close_i": int(or_close_i) if or_close_i is not None else None,
            "first_after_or_i": int(post_first_candle_i) if post_first_candle_i is not None else None,
            "last_close": float(c[-1]) if len(c) else None,
            "signal": signal,
            "last_ts": int(ts[-1]) if len(ts) else None,
            "ts": ts,
            "o": o, "h": h, "l": l, "c": c,
            "ema9": ema9,
        }
        return payload

    def log_payload(self, payload: Dict, debug: bool = True):
        """Send compact log to Telegram."""
        sym = payload["symbol"]
        or_info = "OR: pending"

        if payload["or_ready"]:
            or_info = f"ORH={payload['or_high']:.6f} ORL={payload['or_low']:.6f} MID={payload['or_mid']:.6f}"

        sig = payload.get("signal")

        # In production mode, only show OR updates and heartbeat without signal details
        if not debug and not sig:
            tg(f"📊 NY-ORB {sym} | {or_info} | Monitoring")
            return

        if sig:
            def iso(ms_val):
                try:
                    return datetime.fromtimestamp(float(ms_val)/1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                except Exception:
                    return "n/a"

            def bar_line(i: int) -> str:
                try:
                    ts_arr = payload.get("ts", [])
                    o_arr = payload.get("o", [])
                    h_arr = payload.get("h", [])
                    l_arr = payload.get("l", [])
                    c_arr = payload.get("c", [])
                    if isinstance(i, (int, np.integer)) and 0 <= i < len(ts_arr):
                        return f"{iso(ts_arr[i])} | O={float(o_arr[i]):.6f} H={float(h_arr[i]):.6f} L={float(l_arr[i]):.6f} C={float(c_arr[i]):.6f}"
                    return "unavailable"
                except Exception:
                    return "unavailable"

            i_b = int(sig.get("breakout_i", -1))
            breakout_bar = bar_line(i_b)

            # Show pending order details if present
            pending_trigger = sig.get("pending_trigger")
            invalidation = sig.get("invalidation_level")
            if pending_trigger and invalidation:
                tp_info = f"TP1={sig.get('tp1', sig['tp']):.6f} TP2={sig.get('tp2', sig['tp']):.6f} TP3={sig.get('tp3', sig['tp']):.6f}"
                tg(
                    f"📊 NY-ORB {sym}\n"
                    f"{or_info}\n"
                    f"🎯 PENDING ORDER [{sig['side'].upper()}] {sig['pattern']} (RR={sig['rr']})\n"
                    f"Trigger={pending_trigger:.6f} SL={sig['sl']:.6f}\n"
                    f"{tp_info}\n"
                    f"Invalidation={'below' if sig['side']=='long' else 'above'} {invalidation:.6f}\n"
                    f"Breakout bar: {breakout_bar}"
                )
            else:
                # Fallback for immediate entry (shouldn't happen with new logic, but handle gracefully)
                entry_or_trigger = sig.get('entry') or sig.get('pending_trigger')
                if entry_or_trigger:
                    tg(
                        f"📊 NY-ORB {sym}\n"
                        f"{or_info}\n"
                        f"✅ BREAKOUT [{sig['side'].upper()}] {sig['pattern']} (RR={sig['rr']})\n"
                        f"Entry/Trigger={entry_or_trigger:.6f} SL={sig['sl']:.6f} TP={sig.get('tp', sig.get('tp3', 'n/a'))}\n"
                        f"Breakout bar: {breakout_bar}"
                    )
                else:
                    tg(
                        f"📊 NY-ORB {sym}\n"
                        f"{or_info}\n"
                        f"✅ BREAKOUT [{sig['side'].upper()}] {sig['pattern']}\n"
                        f"Breakout bar: {breakout_bar}"
                    )
        else:
            tg(f"📊 NY-ORB {sym} | {or_info} | No breakout yet")
