import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np

import policy


def _base_ltf(n: int, side: str, seed: int = 1) -> Dict:
    """Up or down trending LTF with EMA9, volumes and timestamps."""
    up = (side == "long")
    rng = np.random.default_rng(seed)
    base = 10000.0
    drift = 3.0 if up else -3.0
    idx = np.arange(n, dtype=float)
    noise = rng.normal(0, 0.8, size=n)
    c = base + drift * idx + noise
    tr_base = 10.0
    o = c.copy()
    h = np.maximum(o, c) + tr_base * 0.5
    l = np.minimum(o, c) - tr_base * 0.5
    v = np.full(n, 100.0)
    ema9 = policy._ema(c, 9)
    ts_start = int(time.time()) - n * 60
    ts = (np.arange(n) * 60 + ts_start).astype(int)
    return {"ts": ts, "o": o, "h": h, "l": l, "c": c, "ema9": ema9, "v": v}


def _itf_from(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray) -> Dict:
    tr = policy._true_range(o, h, l, c)
    atr = policy._atr_from_tr(tr, 14)
    return {
        "ts": np.arange(len(c), dtype=int),
        "o": o.copy(), "h": h.copy(), "l": l.copy(), "c": c.copy(), "v": v.copy(),
        "atr": float(atr), "price": float(c[-2]), "funding": 0.0
    }


def _htf_zone_around(price: float, atr: float, side: str) -> Tuple[Dict, Tuple[float, float]]:
    z_half = max(0.6 * max(atr, 1e-9), 1e-6)
    zone = (price - z_half, price + z_half)
    htf_map = {"zones": {"demand": [zone], "supply": [zone]}, "bias": "bullish" if side == "long" else "bearish"}
    return htf_map, zone


def build_for_EMA9(n: int, side: str) -> Tuple[Dict, Dict, Dict]:
    ltf = _base_ltf(n, side, seed=10)
    up = (side == "long")
    k = n - 2
    # Shape -2 bar: strong body, close near EMA9, volume and TR expansion vs median
    ema9 = ltf["ema9"]
    c = ltf["c"]; o = ltf["o"]; h = ltf["h"]; l = ltf["l"]; v = ltf["v"]
    target = float(ema9[k])
    if up:
        c[k] = target * 1.0012
        o[k] = c[k] - 7.0
        h[k] = c[k] + 5.0
        l[k] = o[k] - 2.0
    else:
        c[k] = target * 0.9988
        o[k] = c[k] + 7.0
        h[k] = o[k] + 2.0
        l[k] = c[k] - 5.0
    v[k] = 160.0
    itf = _itf_from(ltf["o"], ltf["h"], ltf["l"], ltf["c"], ltf["v"])
    htf = {}
    return ltf, itf, htf


def build_for_A(n: int, side: str) -> Tuple[Dict, Dict, Dict]:
    ltf = _base_ltf(n, side, seed=20)
    up = (side == "long")
    # LTF sweep and reclaim on -2
    k = n - 2
    ema9 = ltf["ema9"]; c = ltf["c"]; o = ltf["o"]; h = ltf["h"]; l = ltf["l"]; v = ltf["v"]
    if up:
        l[k] = min(l[k], l[k - 1] - 2.0)      # sweep a new low vs previous
        o[k] = c[k - 1] - 6.0
        c[k] = float(ema9[k]) + 4.0           # close > ema9
        h[k] = max(h[k], c[k] + 2.0)          # small top wick
    else:
        h[k] = max(h[k], h[k - 1] + 2.0)
        o[k] = c[k - 1] + 6.0
        c[k] = float(ema9[k]) - 4.0
        l[k] = min(l[k], c[k] - 2.0)
    v[k] = 140.0
    # ITF trending, zone around price, mark break/retest OK
    itf = _itf_from(ltf["o"], ltf["h"], ltf["l"], ltf["c"], ltf["v"])
    price = itf["price"]; atr = itf["atr"]
    htf, zone = _htf_zone_around(price, atr, side)
    itf.update({
        "zone": zone, "side": side,
        "broke_level": True, "break_close_dist_atr": 0.9,
        "retested": True, "retest_dist_atr": 0.2
    })
    return ltf, itf, htf


def build_for_C1(n: int, side: str) -> Tuple[Dict, Dict, Dict]:
    ltf = _base_ltf(n, side, seed=30)
    up = (side == "long")
    # Add gentle oscillations to form multiple fractal swings (especially lows for uptrend)
    c = ltf["c"]; o = ltf["o"]; h = ltf["h"]; l = ltf["l"]; v = ltf["v"]; ema9 = ltf["ema9"]
    idx = np.arange(n, dtype=float)
    wave = 1.5 * np.sin(idx / 6.0)
    c += wave; o[:] = c - 2.0; h[:] = c + 5.0; l[:] = c - 5.0
    # Ensure several swing lows
    for j in range(n - 40, n - 8, 6):
        l[j] = min(l[j], l[j - 1] - 3.0, l[j + 1] - 3.0)
    # Shape -2 bar near EMA9 and above last swing for long; inverse for short
    k = n - 2
    if up:
        o[k] = c[k - 1] - 6.0
        c[k] = float(ema9[k]) + 3.0
    else:
        o[k] = c[k - 1] + 6.0
        c[k] = float(ema9[k]) - 3.0
    h[k] = max(h[k], c[k] + 4.0); l[k] = min(l[k], o[k] - 2.0)
    v[k] = 160.0
    itf = _itf_from(ltf["o"], ltf["h"], ltf["l"], ltf["c"], ltf["v"])
    htf = {}
    return ltf, itf, htf


def build_for_C2(n: int, side: str) -> Tuple[Dict, Dict, Dict]:
    # LTF base (we'll make last-2 a big expansion bar)
    ltf = _base_ltf(n, side, seed=40)
    up = (side == "long")
    c = ltf["c"]; o = ltf["o"]; h = ltf["h"]; l = ltf["l"]; v = ltf["v"]
    # ITF: tight box in the last 36 bars (excluding the last 2)
    itf = _itf_from(o, h, l, c, v)
    box_n = 36
    s0 = n - (box_n + 2)
    mid = float(np.mean(itf["c"][s0:n - 2]))
    itf["c"][s0:n - 2] = mid
    itf["o"][s0:n - 2] = mid
    itf["h"][s0:n - 2] = mid + 0.8  # small box height
    itf["l"][s0:n - 2] = mid - 0.8
    itf["v"][s0:n - 2] = np.full(box_n, 95.0)
    # Recompute ATR after box shaping
    trI = policy._true_range(itf["o"], itf["h"], itf["l"], itf["c"])
    atrI = policy._atr_from_tr(trI, 14)
    itf["atr"] = float(atrI)
    itf["price"] = float(itf["c"][-2])
    # LTF breakout: make c[-2] far above ITF box hi (+ buffer) with big TR and volume
    hi = float(np.max(itf["h"][-box_n:]))
    buf = 0.6  # align with default-ish buffer mult
    k = n - 2
    if up:
        o[k] = hi + buf * atrI - 4.0
        c[k] = hi + buf * atrI + 6.0
        h[k] = c[k] + 2.0
        l[k] = o[k] - 2.0
    else:
        o[k] = hi - buf * atrI + 4.0
        c[k] = hi - buf * atrI - 6.0
        h[k] = o[k] + 2.0
        l[k] = c[k] - 2.0
    v[k] = 180.0
    htf = {}
    return ltf, itf, htf


def build_for_BR(n: int, side: str) -> Tuple[Dict, Dict, Dict]:
    # Build LTF with a clear break above last swing high and immediate retest
    ltf = _base_ltf(n, side, seed=50)
    up = (side == "long")
    c = ltf["c"]; o = ltf["o"]; h = ltf["h"]; l = ltf["l"]; v = ltf["v"]
    # Create a prominent swing high at t0, then a break at i, and a retest at k=i+1
    t0 = n - 12
    h[t0] = max(h[t0], h[t0 - 1] + 6.0, h[t0 + 1] + 6.0)  # local max
    # Level to break
    level = h[t0]
    buf = 0.0012  # default BR buffer
    i = n - 10
    o[i] = level * (1.0 + buf) - 1.0
    c[i] = level * (1.0 + buf) + 2.0
    h[i] = c[i] + 1.5
    l[i] = o[i] - 1.5
    # Retest at k
    k = i + 1
    o[k] = level
    l[k] = level * (1.0 - 0.0008)  # within retest tolerance
    c[k] = o[k] + 6.0             # strong bullish body
    h[k] = c[k] + 2.0
    v[k] = 160.0
    # Ensure EMA50/200 uptrend by nudging earlier section upwards
    ltf["c"][: n - 50] += np.linspace(0, 40, n - 50)
    ltf["o"][: n - 50] = ltf["c"][: n - 50] - 2.0
    ltf["h"][: n - 50] = ltf["c"][: n - 50] + 5.0
    ltf["l"][: n - 50] = ltf["c"][: n - 50] - 5.0
    # ITF needed only for ATR
    itf = _itf_from(ltf["o"], ltf["h"], ltf["l"], ltf["c"], ltf["v"])
    htf = {}
    return ltf, itf, htf


def build_for_combo(code: str, n: int, side: str) -> Tuple[Dict, Dict, Dict]:
    c = code.upper()
    if c == "EMA9":
        return build_for_EMA9(n, side)
    if c == "A":
        return build_for_A(n, side)
    if c == "C1":
        return build_for_C1(n, side)
    if c == "C2":
        return build_for_C2(n, side)
    if c == "BR":
        return build_for_BR(n, side)
    # Default: EMA9
    return build_for_EMA9(n, side)


def main():
    parser = argparse.ArgumentParser(description="Local tester for policy.decide (synthetic)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--combo", default="ALL", help="A | C1 | C2 | BR | EMA9 | ALL | composite like 'EMA9+BR'")
    parser.add_argument("--side", default="long", choices=["long", "short"])
    parser.add_argument("--size", type=float, default=100.0)
    parser.add_argument("--sl-mult", type=float, default=1.5)
    parser.add_argument("--tp-mult", type=float, default=3.0)
    parser.add_argument("--length", type=int, default=260)
    args = parser.parse_args()

    # Relax thresholds for synthetic data
    os.environ.setdefault("ATR_SOURCE", "ITF")
    os.environ.setdefault("REGIME_ATR_MULT", "0.5")
    os.environ.setdefault("C1_MIN_SWINGS", "3")
    os.environ.setdefault("C1_PULLBACK_PROX", "0.006")
    os.environ.setdefault("C2_EXP_TR_MULT", "1.0")
    os.environ.setdefault("C2_VOL_MULT", "1.0")
    os.environ.setdefault("C2_BOX_ATR_MAX", "1.2")
    os.environ.setdefault("EMA9_PROX_TOL", "0.006")
    os.environ.setdefault("EMA9_TR_MULT", "0.95")
    os.environ.setdefault("EMA9_VOL_MULT", "1.02")

    combo = args.combo.strip().upper()
    if combo == "ALL":
        combos = ["A", "C1", "C2", "BR", "EMA9"]
        print("=== Local Decision Test (ALL) ===")
        for code in combos:
            ltf, itf, htf = build_for_combo(code, args.length, args.side)
            d = policy.decide(
                symbol=args.symbol, combo=code,
                htf_map=htf, itf_setup=itf, ltf_snapshot=ltf,
                size_usdt=args.size, sl_mult=args.sl_mult, tp_mult=args.tp_mult,
            )
            if d is None:
                print(f"[{code}] -> None")
            else:
                print(f"[{code}] -> {d.side} sl={d.sl:.6f} tp={d.tp:.6f} conf={d.confidence:.2f} reason={d.reason}")
        return

    # Single or composite like 'EMA9+BR'
    if "+" in combo:
        parts = [p.strip() for p in combo.split("+") if p.strip()]
        print(f"=== Local Decision Test ({combo}) ===")
        for code in parts:
            ltf, itf, htf = build_for_combo(code, args.length, args.side)
            d = policy.decide(
                symbol=args.symbol, combo=code,
                htf_map=htf, itf_setup=itf, ltf_snapshot=ltf,
                size_usdt=args.size, sl_mult=args.sl_mult, tp_mult=args.tp_mult,
            )
            if d is None:
                print(f"[{code}] -> None")
            else:
                print(f"[{code}] -> {d.side} sl={d.sl:.6f} tp={d.tp:.6f} conf={d.confidence:.2f} reason={d.reason}")
        return

    # Single
    ltf, itf, htf = build_for_combo(combo, args.length, args.side)
    d = policy.decide(
        symbol=args.symbol, combo=combo,
        htf_map=htf, itf_setup=itf, ltf_snapshot=ltf,
        size_usdt=args.size, sl_mult=args.sl_mult, tp_mult=args.tp_mult,
    )
    print("=== Local Decision Test ===")
    if d is None:
        print("Decision: None")
    else:
        print(f"Decision: {d.side} sl={d.sl:.6f} tp={d.tp:.6f} conf={d.confidence:.2f} reason={d.reason}")


if __name__ == "__main__":
    main()
