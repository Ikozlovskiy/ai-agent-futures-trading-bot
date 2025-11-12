import ccxt
import os
import time
import calendar
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from utils import tg

def build_exchange():
    ex = ccxt.binanceusdm({
        "apiKey": os.getenv("BINANCE_API_KEY", ""),
        "secret": os.getenv("BINANCE_API_SECRET", ""),
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    ex.load_markets()
    return ex

def fetch_candles(ex, symbol: str, timeframe: str, limit: int=300) -> List[List[float]]:
    """Returns list of [ts, o, h, l, c, v]."""
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

def last_price(ex, symbol: str) -> float:
    t = ex.fetch_ticker(symbol)
    return float(t.get("last") or t.get("close") or t["info"].get("lastPrice"))

# ====== Indicators & helpers ======

def ema(arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=period, adjust=False).mean().values

def rsi(closes: np.ndarray, period: int=14) -> np.ndarray:
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta>0, delta, 0.0)
    loss = np.where(delta<0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
    rs = np.divide(avg_gain, np.where(avg_loss==0, np.nan, avg_loss))
    return 100 - (100/(1+rs))

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int=14) -> np.ndarray:
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close)
    ])
    return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values

def vwap(ts: np.ndarray, closes: np.ndarray, volume: np.ndarray, anchor_epoch: int) -> np.ndarray:
    """Anchored VWAP from anchor_epoch (UTC)."""
    mask = ts >= anchor_epoch * 1000  # ts in ms
    pv = (closes[mask] * volume[mask]).cumsum()
    vv = (volume[mask]).cumsum()
    avwap = pv / np.where(vv == 0, np.nan, vv)
    out = np.full_like(closes, np.nan, dtype=float)
    out[np.where(mask)[0]] = avwap
    return out

def keltner_width(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int=20, atr_mult: float=2.0) -> np.ndarray:
    ema_mid = ema(close, period)
    atr_vals = atr(high, low, close, period)
    upper = ema_mid + atr_mult * atr_vals
    lower = ema_mid - atr_mult * atr_vals
    return (upper - lower) / np.where(ema_mid==0, np.nan, ema_mid)

def swings(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, left: int=2, right: int=2) -> Tuple[List[int], List[int]]:
    sh, sl = [], []
    n = len(closes)
    for i in range(left, n-right):
        hwin = highs[i-left:i+right+1]
        lwin = lows[i-left:i+right+1]
        if highs[i] == max(hwin): sh.append(i)
        if lows[i]  == min(lwin): sl.append(i)
    return sh, sl

def simple_zones(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, pad: float=0.0015):
    sh, sl = swings(closes, highs, lows)
    levels_high = [highs[i] for i in sh[-10:]]
    levels_low  = [lows[i]  for i in sl[-10:]]
    zones = {"supply": [], "demand": []}
    for L in sorted(levels_high):
        zones["supply"].append((L*(1- pad), L*(1+ pad)))
    for L in sorted(levels_low):
        zones["demand"].append((L*(1- pad), L*(1+ pad)))
    return zones

def structure_bias(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> str:
    sh, sl = swings(closes, highs, lows)
    if len(sh)<1 or len(sl)<1:
        return "neutral"
    last_sh = highs[sh[-1]]
    last_sl = lows[sl[-1]]
    if closes[-1] > last_sh: return "bullish"
    if closes[-1] < last_sl: return "bearish"
    return "neutral"

def utc_anchor_for_session(start_hhmm: str) -> int:
    hh, mm = [int(x) for x in start_hhmm.split(":")]
    now = time.gmtime()
    # Build a UTC tuple and convert using calendar.timegm for correct UTC epoch
    return int(calendar.timegm((now.tm_year, now.tm_mon, now.tm_mday, hh, mm, 0, 0, 0, 0)))

def opening_range(ts: np.ndarray, highs: np.ndarray, lows: np.ndarray, start_epoch: int, minutes: int) -> Optional[Tuple[float, float]]:
    start_ms = start_epoch * 1000
    end_ms = start_ms + minutes * 60 * 1000
    mask = (ts >= start_ms) & (ts < end_ms)
    if mask.sum() < 1: 
        return None
    return float(np.max(highs[mask])), float(np.min(lows[mask]))

# ====== Cached state ======

class MarketState:
    """In-memory state shared across passes (no DB). Supports mixed Combo A/B per symbol."""
    def __init__(self, symbols: List[str], htf: str, itf: str, ltf: str, atr_period: int):
        self.symbols = symbols
        self.htf, self.itf, self.ltf = htf, itf, ltf
        self.atr_period = atr_period
        self.htf_maps: Dict[str, Dict] = {}
        self.itf_setups: Dict[str, Dict] = {}
        self.last_candles: Dict[Tuple[str,str], List[List[float]]] = {}
        self.session_start = os.getenv("SESSION_START_UTC", "08:00")
        self.orb_minutes = int(os.getenv("ORB_MINUTES", "15"))

    # ----- Per-symbol helpers for A -----
    def _compute_htf_A(self, ex, sym: str) -> Dict:
        ohlcv = fetch_candles(ex, sym, self.htf, limit=300)
        self.last_candles[(sym, self.htf)] = ohlcv
        arr = np.asarray(ohlcv, dtype=float)
        _, o, h, l, c, _ = arr.T
        return {"zones": simple_zones(c, h, l), "bias": structure_bias(c, h, l)}

    def _compute_itf_A(self, ex, sym: str, htf_map: Dict) -> Dict:
        TOL = 0.0018
        ohlcv = fetch_candles(ex, sym, self.itf, limit=300)
        self.last_candles[(sym, self.itf)] = ohlcv
        arr = np.asarray(ohlcv, dtype=float)
        ts, o, h, l, c, v = arr.T
        atr_vals = atr(h, l, c, period=self.atr_period)
        atr_last = float(atr_vals[-2]) if len(atr_vals) >= 2 else 0.0
        last_c  = float(c[-2])

        # Base ITF snapshot with arrays and core metrics for regime checks
        itf = {
            "ts": ts, "o": o, "h": h, "l": l, "c": c, "v": v,
            "atr": atr_last,
            "price": last_c,
        }

        z = htf_map.get("zones", {})
        bias = htf_map.get("bias", "neutral")
        chosen_zone = None
        side = None
        if bias == "bullish":
            for lo, hi in z.get("demand", []):
                if lo <= last_c <= hi*(1+TOL):
                    chosen_zone = (float(lo), float(hi))
                    side = "long"
                    break
        elif bias == "bearish":
            for lo, hi in z.get("supply", []):
                if lo*(1-TOL) <= last_c <= hi:
                    chosen_zone = (float(lo), float(hi))
                    side = "short"
                    break

        # Compute simplified break/retest characteristics vs zone bounds
        if chosen_zone:
            itf["zone"] = chosen_zone
            itf["side"] = side
            lo, hi = chosen_zone
            if side == "long":
                broke_level = last_c > hi
                break_close_dist_atr = max(0.0, (last_c - hi)) / max(atr_last, 1e-9)
                retest_dist_atr = abs(float(l[-2]) - hi) / max(atr_last, 1e-9)
                itf["broke_level"] = bool(broke_level)
                itf["break_close_dist_atr"] = float(break_close_dist_atr)
                itf["retested"] = True
                itf["retest_dist_atr"] = float(retest_dist_atr)
            else:
                broke_level = last_c < lo
                break_close_dist_atr = max(0.0, (lo - last_c)) / max(atr_last, 1e-9)
                retest_dist_atr = abs(float(h[-2]) - lo) / max(atr_last, 1e-9)
                itf["broke_level"] = bool(broke_level)
                itf["break_close_dist_atr"] = float(break_close_dist_atr)
                itf["retested"] = True
                itf["retest_dist_atr"] = float(retest_dist_atr)

        return itf

    # ----- Per-symbol helpers for B -----
    def _compute_htf_B(self, ex, sym: str) -> Dict:
        anchor = utc_anchor_for_session(os.getenv("SESSION_START_UTC", "08:00"))
        ohlcv = fetch_candles(ex, sym, self.htf, limit=500)
        self.last_candles[(sym, self.htf)] = ohlcv
        arr = np.asarray(ohlcv, dtype=float)
        ts, o, h, l, c, v = arr.T
        return {
            "vwap": vwap(ts, c, v, anchor),
            "keltner_w": keltner_width(h, l, c, period=20, atr_mult=2.0)
        }

    def _compute_itf_B(self, ex, sym: str) -> Dict:
        ohlcv = fetch_candles(ex, sym, self.itf, limit=200)
        self.last_candles[(sym, self.itf)] = ohlcv
        arr = np.asarray(ohlcv, dtype=float)
        ts, o, h, l, c, v = arr.T
        atr_vals = atr(h, l, c, period=self.atr_period)
        atr_now = float(atr_vals[-2]) if len(atr_vals) >= 2 else 0.0

        # Base ITF snapshot with arrays for regime/strategy checks
        itf = {"ts": ts, "o": o, "h": h, "l": l, "c": c, "v": v, "atr": atr_now, "price": float(c[-2])}

        window = 20
        if len(c) >= window + 2:
            last_h = float(np.max(h[-window:]))
            last_l = float(np.min(l[-window:]))
            range_pct = (last_h - last_l) / max(c[-2], 1e-9)
            atr_mean = float(pd.Series(atr_vals[-(window+1):-1]).mean()) if len(atr_vals) >= window + 1 else atr_now
            contracting = atr_now < atr_mean
            small_box = range_pct < 0.006
            if contracting and small_box:
                itf.update({"side": "both", "box": (last_l, last_h)})

        return itf

    # ----- Mixed refresh (A or B per symbol) -----
    def refresh_htf_mixed(self, ex, combo_map: Dict[str, str]):
        for sym in self.symbols:
            combo = combo_map.get(sym, "A").upper()
            if combo == "A":
                self.htf_maps[sym] = self._compute_htf_A(ex, sym)
            else:
                self.htf_maps[sym] = self._compute_htf_B(ex, sym)

    def refresh_itf_mixed(self, ex, combo_map: Dict[str, str]):
        for sym in self.symbols:
            combo = combo_map.get(sym, "A").upper()
            if combo == "A":
                self.itf_setups[sym] = self._compute_itf_A(ex, sym, self.htf_maps.get(sym, {}))
            else:
                self.itf_setups[sym] = self._compute_itf_B(ex, sym)

    def build_ltf_snapshot(self, ex, sym: str) -> Dict:
        ohlcv = fetch_candles(ex, sym, self.ltf, limit=300)
        self.last_candles[(sym, self.ltf)] = ohlcv
        arr = np.asarray(ohlcv, dtype=float)
        ts, o, h, l, c, v = arr.T
        ema9  = pd.Series(c).ewm(span=9, adjust=False).mean().values
        rsi14 = rsi(c, 14)
        atr_vals = atr(h, l, c, period=self.atr_period)
        atr_ltf = float(atr_vals[-2]) if len(atr_vals) >= 2 else 0.0
        return {"ts": ts, "o": o, "h": h, "l": l, "c": c, "v": v, "ema9": ema9, "rsi14": rsi14, "atr_ltf": atr_ltf}
