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
    NY Opening Range Breakout (ORB) Strategy with Retest:
    1. Define OR: 14:30-14:45 UTC (configurable)
    2. Initial Breakout: N candles close outside OR High/Low with EMA9 confirmation
    3. Retest: Price pulls back into breakout zone
    4. Continuation: M candles close back outside in same direction
    5. Entry: Market order after retest continuation
    6. Invalidation: If opposite OR side breaks, reset and look for new direction
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
        self.signal_timeframe = signal_timeframe or os.getenv("ORB_SIGNAL_TIMEFRAME", "1m")
        self.check_interval = int(check_interval or int(os.getenv("ORB_CHECK_INTERVAL", "60") or 60))
        self.ema_period = int(os.getenv("ORB_EMA_PERIOD", "9") or 9)
        self.ema_slope_bars = int(os.getenv("ORB_EMA_SLOPE_BARS", "3") or 3)
        self.confirm_candles = int(os.getenv("ORB_CONFIRM_CANDLES", "2") or 2)

        # Retest configuration
        self.require_retest = _env_bool("ORB_REQUIRE_RETEST", True)
        self.retest_timeout_min = int(os.getenv("ORB_RETEST_TIMEOUT_MIN", "45") or 45)
        self.retest_depth = os.getenv("ORB_RETEST_DEPTH", "zone")  # 'zone' or 'or'
        self.retest_confirm_candles = int(os.getenv("ORB_RETEST_CONFIRM_CANDLES", "2") or 2)
        self.reset_on_opposite_break = _env_bool("ORB_RESET_ON_OPPOSITE", True)

        # Volume and body ratio filters
        self.vol_lookback = 10
        self.vol_mult = float(os.getenv("ORB_VOL_MULT", "1.2") or 1.2)
        self.min_body_ratio = float(os.getenv("ORB_MIN_BODY_RATIO", "0.5") or 0.5)

        # State tracking per symbol: {symbol: state_dict}
        self.state = {}

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
        Fetch 1m candles, compute OR, detect breakout with optional retest requirement.
        State machine: idle -> breakout -> waiting_retest -> retest_confirmed -> entry
        Returns payload dict with diagnostics and state info.
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

            or_open_i = next((i for i in range(len(ts)) if ts[i] >= or_open_ms), None)
            or_close_i = next((i for i in range(len(ts)) if ts[i] >= or_close_ms), None)

            if or_open_i is not None and or_close_i is not None and (or_close_i - or_open_i) >= 1:
                if or_close_i < len(ts) - 1:
                    post_first_candle_i = or_close_i
                    or_ready = True

        # Initialize or retrieve state for this symbol
        if symbol not in self.state:
            self.state[symbol] = {
                "phase": "idle",  # idle, breakout_detected, waiting_retest, entry_ready
                "side": None,
                "breakout_ts": None,
                "breakout_i": None,
                "breakout_zone_high": None,  # For long: high of breakout candle
                "breakout_zone_low": None,   # For short: low of breakout candle
                "retest_i": None,
            }

        state = self.state[symbol]
        signal = None
        state_changed = False

        # Reset state on new day or if OR not ready
        if not or_ready:
            if state["phase"] != "idle":
                state.update({
                    "phase": "idle",
                    "side": None,
                    "breakout_ts": None,
                    "breakout_i": None,
                    "breakout_zone_high": None,
                    "breakout_zone_low": None,
                    "retest_i": None,
                })
                state_changed = True

        if or_ready and post_first_candle_i is not None and not np.isnan(orh) and not np.isnan(orl):
            or_mid = (orh + orl) / 2.0
            avg_vol = np.mean(v[-self.vol_lookback-1:-1]) if len(v) > self.vol_lookback else 0
            current_i = len(ts) - 1
            current_ts = ts[current_i]

            # Check for opposite side invalidation
            if self.reset_on_opposite_break and state["phase"] in ("breakout_detected", "waiting_retest"):
                if state["side"] == "long" and float(c[current_i]) < orl:
                    # Price broke below OR low - invalidate long setup
                    state.update({
                        "phase": "idle",
                        "side": None,
                        "breakout_ts": None,
                        "breakout_i": None,
                        "breakout_zone_high": None,
                        "breakout_zone_low": None,
                        "retest_i": None,
                    })
                    state_changed = True
                elif state["side"] == "short" and float(c[current_i]) > orh:
                    # Price broke above OR high - invalidate short setup
                    state.update({
                        "phase": "idle",
                        "side": None,
                        "breakout_ts": None,
                        "breakout_i": None,
                        "breakout_zone_high": None,
                        "breakout_zone_low": None,
                        "retest_i": None,
                    })
                    state_changed = True

            # Check for retest timeout
            if state["phase"] == "waiting_retest" and state["breakout_ts"] is not None:
                time_elapsed_min = (current_ts - state["breakout_ts"]) / 60000.0  # ts in ms
                if time_elapsed_min > self.retest_timeout_min:
                    # Timeout - reset state
                    state.update({
                        "phase": "idle",
                        "side": None,
                        "breakout_ts": None,
                        "breakout_i": None,
                        "breakout_zone_high": None,
                        "breakout_zone_low": None,
                        "retest_i": None,
                    })
                    state_changed = True

            # State machine logic
            if state["phase"] == "idle":
                # Look for initial breakout
                for i in range(post_first_candle_i, len(ts)):
                    if i < max(self.ema_slope_bars, self.confirm_candles - 1):
                        continue

                    # Check for bullish breakout
                    if float(c[i]) > orh:
                        if i - (self.confirm_candles - 1) < post_first_candle_i:
                            continue

                        all_above_orh = all(float(c[i - j]) > orh for j in range(self.confirm_candles))
                        all_green = all(float(c[i - j]) > float(o[i - j]) for j in range(self.confirm_candles))

                        if not (all_above_orh and all_green):
                            continue

                        candle_range = max(1e-9, h[i] - l[i])
                        body_size = abs(c[i] - o[i])
                        body_ratio = body_size / candle_range
                        vol_expansion = v[i] > (avg_vol * self.vol_mult) if avg_vol > 0 else True

                        ema_slope_up = all(
                            ema9[i - j] > ema9[i - j - 1]
                            for j in range(self.ema_slope_bars)
                            if i - j - 1 >= 0 and not np.isnan(ema9[i - j]) and not np.isnan(ema9[i - j - 1])
                        )

                        if ema_slope_up and body_ratio >= self.min_body_ratio and vol_expansion:
                            # Breakout detected!
                            state["phase"] = "waiting_retest" if self.require_retest else "entry_ready"
                            state["side"] = "long"
                            state["breakout_ts"] = float(ts[i])
                            state["breakout_i"] = i
                            state["breakout_zone_high"] = float(h[i])
                            state["breakout_zone_low"] = float(l[i])
                            state_changed = True
                            break

                    # Check for bearish breakout
                    elif float(c[i]) < orl:
                        if i - (self.confirm_candles - 1) < post_first_candle_i:
                            continue

                        all_below_orl = all(float(c[i - j]) < orl for j in range(self.confirm_candles))
                        all_red = all(float(c[i - j]) < float(o[i - j]) for j in range(self.confirm_candles))

                        if not (all_below_orl and all_red):
                            continue

                        candle_range = max(1e-9, h[i] - l[i])
                        body_size = abs(c[i] - o[i])
                        body_ratio = body_size / candle_range
                        vol_expansion = v[i] > (avg_vol * self.vol_mult) if avg_vol > 0 else True

                        ema_slope_dn = all(
                            ema9[i - j] < ema9[i - j - 1]
                            for j in range(self.ema_slope_bars)
                            if i - j - 1 >= 0 and not np.isnan(ema9[i - j]) and not np.isnan(ema9[i - j - 1])
                        )

                        if ema_slope_dn and body_ratio >= self.min_body_ratio and vol_expansion:
                            # Breakout detected!
                            state["phase"] = "waiting_retest" if self.require_retest else "entry_ready"
                            state["side"] = "short"
                            state["breakout_ts"] = float(ts[i])
                            state["breakout_i"] = i
                            state["breakout_zone_high"] = float(h[i])
                            state["breakout_zone_low"] = float(l[i])
                            state_changed = True
                            break

            elif state["phase"] == "waiting_retest":
                # Look for retest: price pulls back into breakout zone, then continues
                if state["side"] == "long":
                    # Define retest zone
                    if self.retest_depth == "or":
                        retest_zone_low = orh
                        retest_zone_high = state["breakout_zone_high"]
                    else:  # "zone" (default)
                        retest_zone_low = state["breakout_zone_low"]
                        retest_zone_high = state["breakout_zone_high"]

                    # Check if price has retested (pulled back into zone)
                    retest_occurred = False
                    retest_i = None
                    for i in range(state["breakout_i"] + 1, len(ts)):
                        if float(l[i]) <= retest_zone_high and float(h[i]) >= retest_zone_low:
                            retest_occurred = True
                            retest_i = i
                            break

                    if retest_occurred and retest_i is not None:
                        # Now look for continuation: M candles close above breakout zone high
                        for i in range(retest_i + 1, len(ts)):
                            if i - (self.retest_confirm_candles - 1) < retest_i + 1:
                                continue

                            all_above_zone = all(
                                float(c[i - j]) > state["breakout_zone_high"]
                                for j in range(self.retest_confirm_candles)
                            )
                            all_green = all(
                                float(c[i - j]) > float(o[i - j])
                                for j in range(self.retest_confirm_candles)
                            )

                            if all_above_zone and all_green:
                                # Retest confirmed! Ready for entry
                                state["phase"] = "entry_ready"
                                state["retest_i"] = retest_i
                                state_changed = True
                                break

                elif state["side"] == "short":
                    # Define retest zone
                    if self.retest_depth == "or":
                        retest_zone_high = orl
                        retest_zone_low = state["breakout_zone_low"]
                    else:  # "zone" (default)
                        retest_zone_high = state["breakout_zone_high"]
                        retest_zone_low = state["breakout_zone_low"]

                    # Check if price has retested (pulled back into zone)
                    retest_occurred = False
                    retest_i = None
                    for i in range(state["breakout_i"] + 1, len(ts)):
                        if float(h[i]) >= retest_zone_low and float(l[i]) <= retest_zone_high:
                            retest_occurred = True
                            retest_i = i
                            break

                    if retest_occurred and retest_i is not None:
                        # Now look for continuation: M candles close below breakout zone low
                        for i in range(retest_i + 1, len(ts)):
                            if i - (self.retest_confirm_candles - 1) < retest_i + 1:
                                continue

                            all_below_zone = all(
                                float(c[i - j]) < state["breakout_zone_low"]
                                for j in range(self.retest_confirm_candles)
                            )
                            all_red = all(
                                float(c[i - j]) < float(o[i - j])
                                for j in range(self.retest_confirm_candles)
                            )

                            if all_below_zone and all_red:
                                # Retest confirmed! Ready for entry
                                state["phase"] = "entry_ready"
                                state["retest_i"] = retest_i
                                state_changed = True
                                break

            # Generate signal if entry is ready
            if state["phase"] == "entry_ready":
                rr_cfg = float(os.getenv("ORB_RR", "2.0") or 2.0)
                entry = float(c[-1])
                sl = or_mid

                if state["side"] == "long":
                    risk = max(1e-9, entry - sl)
                    tp_final = entry + rr_cfg * risk

                    signal = {
                        "side": "long",
                        "breakout_i": state["breakout_i"],
                        "retest_i": state.get("retest_i"),
                        "entry": entry,
                        "sl": sl,
                        "tp": tp_final,
                        "rr": rr_cfg,
                        "pattern": "orb_long_retest" if self.require_retest else "orb_long_direct",
                        "or_high": orh,
                        "or_low": orl,
                        "breakout_zone_high": state["breakout_zone_high"],
                        "breakout_zone_low": state["breakout_zone_low"],
                    }
                else:  # short
                    risk = max(1e-9, sl - entry)
                    tp_final = entry - rr_cfg * risk

                    signal = {
                        "side": "short",
                        "breakout_i": state["breakout_i"],
                        "retest_i": state.get("retest_i"),
                        "entry": entry,
                        "sl": sl,
                        "tp": tp_final,
                        "rr": rr_cfg,
                        "pattern": "orb_short_retest" if self.require_retest else "orb_short_direct",
                        "or_high": orh,
                        "or_low": orl,
                        "breakout_zone_high": state["breakout_zone_high"],
                        "breakout_zone_low": state["breakout_zone_low"],
                    }

                # Reset state after generating signal (one trade per session)
                state.update({
                    "phase": "idle",
                    "side": None,
                    "breakout_ts": None,
                    "breakout_i": None,
                    "breakout_zone_high": None,
                    "breakout_zone_low": None,
                    "retest_i": None,
                })

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
            "state": dict(state),  # Include current state in payload
            "state_changed": state_changed,
            "last_ts": int(ts[-1]) if len(ts) else None,
            "ts": ts,
            "o": o, "h": h, "l": l, "c": c,
            "ema9": ema9,
        }
        return payload

    def log_payload(self, payload: Dict, debug: bool = True):
        """Send comprehensive log to Telegram with state machine tracking."""
        sym = payload["symbol"]
        or_high = payload.get("or_high")
        or_low = payload.get("or_low")
        or_mid = payload.get("or_mid")

        or_info = "OR: pending"
        if payload["or_ready"] and or_high and or_low and or_mid:
            or_range = or_high - or_low
            or_info = f"ORH=${or_high:.2f} | ORL=${or_low:.2f} | MID=${or_mid:.2f} (range: ${or_range:.2f})"

        state = payload.get("state", {})
        state_changed = payload.get("state_changed", False)
        sig = payload.get("signal")

        # Always log state changes (even in production mode)
        if state_changed and not sig:
            phase = state.get("phase", "idle")
            side = state.get("side", "none")

            if phase == "waiting_retest":
                breakout_zone_high = state.get("breakout_zone_high")
                breakout_zone_low = state.get("breakout_zone_low")
                tg(
                    f"🔍 NY-ORB {sym} | {or_info}\n"
                    f"📈 BREAKOUT DETECTED [{side.upper()}]\n"
                    f"⏳ Waiting for retest into ${breakout_zone_low:.2f}-${breakout_zone_high:.2f}\n"
                    f"⏱️ Timeout: {self.retest_timeout_min} min"
                )
                return
            elif phase == "idle" and state.get("breakout_ts"):
                # State was reset (timeout or opposite break)
                tg(f"⚠️ NY-ORB {sym} | {or_info}\n❌ Setup invalidated - resetting")
                return

        # In production mode, only show OR updates and heartbeat without detailed state
        if not debug and not sig and not state_changed:
            phase = state.get("phase", "idle")
            if phase == "waiting_retest":
                side = state.get("side", "")
                tg(f"📊 NY-ORB {sym} | {or_info} | Waiting for {side} retest")
            else:
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
                        return f"{iso(ts_arr[i])} | O={float(o_arr[i]):.2f} H={float(h_arr[i]):.2f} L={float(l_arr[i]):.2f} C={float(c_arr[i]):.2f}"
                    return "unavailable"
                except Exception:
                    return "unavailable"

            entry = sig.get("entry")
            sl = sig.get("sl")
            tp = sig.get("tp")
            side = sig.get("side")
            pattern = sig.get("pattern", "")

            risk_usd = abs(entry - sl)
            reward_usd = abs(tp - entry)

            i_b = int(sig.get("breakout_i", -1))
            i_r = sig.get("retest_i")

            breakout_bar = bar_line(i_b)
            retest_info = ""
            if i_r is not None and isinstance(i_r, (int, np.integer)):
                retest_bar = bar_line(int(i_r))
                retest_info = f"\nRetest bar:     {retest_bar}"

            breakout_zone_high = sig.get("breakout_zone_high")
            breakout_zone_low = sig.get("breakout_zone_low")
            zone_info = ""
            if breakout_zone_high and breakout_zone_low:
                zone_size = breakout_zone_high - breakout_zone_low
                zone_info = f"\nBreakout Zone:  ${breakout_zone_low:.2f}-${breakout_zone_high:.2f} (${zone_size:.2f})"

            tg(
                f"🎯 NY-ORB {sym} [{side.upper()}]\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 {or_info}\n"
                f"{zone_info}\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ {pattern.upper().replace('_', ' ')} CONFIRMED\n"
                f"Entry: ${entry:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f}\n"
                f"Risk: ${risk_usd:.2f} | Reward: ${reward_usd:.2f} | RR: {sig['rr']:.1f}:1\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Breakout bar:   {breakout_bar}"
                f"{retest_info}"
            )
        else:
            # Monitoring log with state info
            phase = state.get("phase", "idle")
            last_close = payload.get("last_close")

            status_info = ""
            if phase == "waiting_retest":
                side = state.get("side", "")
                breakout_zone_high = state.get("breakout_zone_high")
                breakout_zone_low = state.get("breakout_zone_low")
                status_info = f" | Waiting {side} retest (${breakout_zone_low:.2f}-${breakout_zone_high:.2f})"
            elif last_close and or_high and or_low:
                if last_close > or_high:
                    status_info = f" | Price: ${last_close:.2f} (${last_close - or_high:.2f} above OR)"
                elif last_close < or_low:
                    status_info = f" | Price: ${last_close:.2f} (${or_low - last_close:.2f} below OR)"
                else:
                    status_info = f" | Price: ${last_close:.2f} (inside OR)"

            tg(f"📊 NY-ORB {sym} | {or_info}{status_info}")
