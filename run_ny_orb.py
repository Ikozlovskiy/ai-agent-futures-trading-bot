import os
import sys
import time
from typing import List, Dict
from datetime import datetime, timezone, timedelta

# Load .env.orb if it exists (for wife's account)
from dotenv import load_dotenv
if os.path.exists(".env.orb"):
    load_dotenv(".env.orb", override=True)
    print("✅ Loaded .env.orb configuration")
else:
    load_dotenv()  # Fallback to standard .env
    print("⚠️ .env.orb not found, using default .env")

from utils import tg, parse_map_env, get_per_symbol_value
from datahub import build_exchange
from strategies.ny_orb import NyOrbInspector
from executor import execute, poll_positions_and_report, has_open_position
from models import Decision


def _countdown(or_hhmm_utc: str) -> str:
    """Calculate countdown to next OR session."""
    try:
        hh, mm = [int(p) for p in or_hhmm_utc.split(":")]
        now = datetime.now(timezone.utc)
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if now >= target:
            target = target + timedelta(days=1)
        delta = target - now
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except Exception:
        return "n/a"


def _symbols_from_env() -> List[str]:
    """Get symbols from ORB-specific or fallback to SYMBOLS env."""
    raw = os.getenv("ORB_SYMBOLS") or os.getenv("NY_OPEN_SYMBOLS") or os.getenv("SYMBOLS")
    if not raw:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    out = []
    for s in str(raw).split(","):
        s = s.strip()
        if s:
            out.append(s)
    if not out:
        out = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    return out


def _weekday_mon_fri() -> bool:
    """Check if today is Monday-Friday."""
    return time.gmtime().tm_wday < 5


def main():
    ex = build_exchange()
    symbols = _symbols_from_env()

    # ORB-specific parameters
    or_start_utc = os.getenv("ORB_START_UTC", "14:30")
    or_minutes = int(os.getenv("ORB_MINUTES", "15") or 15)
    or_timeframe = os.getenv("ORB_TIMEFRAME", "15m")
    signal_timeframe = os.getenv("ORB_SIGNAL_TIMEFRAME", "5m")
    check_interval = int(os.getenv("ORB_CHECK_INTERVAL", "60") or 60)

    insp = NyOrbInspector(
        or_start_hhmm_utc=or_start_utc,
        or_minutes=or_minutes,
        or_timeframe=or_timeframe,
        signal_timeframe=signal_timeframe,
        check_interval=check_interval,
    )

    # Disable dynamic re-arms for ORB strategy
    os.environ["DYN_ROI_STAGES"] = ""

    # Sizing per symbol
    size_default = float(os.getenv("RISK_NOTIONAL_USDT", "50") or 50.0)
    size_map_raw = parse_map_env("RISK_NOTIONAL_MAP")
    size_map = {k: float(v) for k, v in size_map_raw.items()}
    symbol_size = {s: get_per_symbol_value(s, size_map, size_default) for s in symbols}

    # Daily counters (UTC reset at midnight)
    max_trades = int(os.getenv("ORB_MAX_TRADES", "2") or 2)
    trades_today: Dict[str, int] = {s: 0 for s in symbols}
    last_reset_day = time.gmtime().tm_yday

    # Logging keepalive per symbol
    keepalive_secs = int(os.getenv("ORB_LOG_KEEPALIVE_SEC", "300") or 300)
    debug_mode = (os.getenv("ORB_DEBUG", "false").lower() == "true")

    if debug_mode:
        tg(f"🤖 NY-ORB Strategy started | Symbols={symbols} | OR_START_UTC={or_start_utc} | OR={or_timeframe}/{or_minutes}min | Signal_TF={signal_timeframe} | CheckInterval={check_interval}s | RR={os.getenv('ORB_RR','2.0')} | MaxTrades/day={max_trades}")

    # Track last confirmed breakout to avoid spam
    last_confirm_key: Dict[str, int] = {}
    last_log_ts: Dict[str, float] = {}
    last_hourly_ping = 0.0
    last_poll = 0.0

    trade_enabled = (os.getenv("ORB_TRADE", "true").lower() == "true")

    while True:
        t0 = time.time()

        # UTC daily reset
        cur_day = time.gmtime().tm_yday
        if cur_day != last_reset_day:
            trades_today = {s: 0 for s in symbols}
            last_reset_day = cur_day
            tg("🗓️ NY-ORB: daily counters reset (UTC midnight)")

        # Hourly countdown ping
        if time.time() - last_hourly_ping >= 3600.0:
            eta = _countdown(or_start_utc)
            tg(f"⏳ NY-ORB | Next session starts in {eta} (UTC {or_start_utc})")
            last_hourly_ping = time.time()

        for sym in symbols:
            try:
                payload = insp.analyze_symbol(ex, sym, limit=500)
                sig = payload.get("signal")

                # Keepalive logs
                now_ts = time.time()
                should_keepalive = (now_ts - last_log_ts.get(sym, 0.0)) >= keepalive_secs

                if sig:
                    key = f"{sym}:{sig['side']}"
                    prev_idx = last_confirm_key.get(key, -1)
                    # New breakout
                    if int(sig["breakout_i"]) != int(prev_idx):
                        if debug_mode:
                            insp.log_payload(payload)
                        last_confirm_key[key] = int(sig["breakout_i"])
                        last_log_ts[sym] = now_ts
                    else:
                        # Same breakout, periodic heartbeat
                        if should_keepalive:
                            insp.log_payload({**payload, "signal": None}, debug=debug_mode)
                            last_log_ts[sym] = now_ts
                else:
                    # No signal, keepalive
                    if should_keepalive:
                        insp.log_payload(payload, debug=debug_mode)
                        last_log_ts[sym] = now_ts

                # Trading logic
                if trade_enabled and sig and payload.get("or_ready", False):
                    # Only weekdays
                    if not _weekday_mon_fri():
                        continue
                    # Respect daily max trades
                    if trades_today.get(sym, 0) >= max_trades:
                        continue
                    # Ensure breakout happens after OR period
                    faoi = payload.get("first_after_or_i")
                    try:
                        if faoi is None or int(sig.get("breakout_i", -1)) < int(faoi):
                            continue
                    except Exception:
                        continue
                    # Skip if position already open
                    if has_open_position(ex, sym):
                        continue

                    # Build Decision
                    reason = {
                        "strategy": "NY_ORB",
                        "pattern": str(sig.get("pattern")),
                        "rr": str(sig.get("rr")),
                    }
                    decision = Decision(
                        symbol=sym,
                        side=sig["side"],
                        entry_type="market",
                        size_usdt=float(symbol_size.get(sym, 0.0)),
                        sl=float(sig["sl"]),
                        tp=float(sig["tp"]),
                        confidence=0.62,
                        reason=reason,
                        valid_until=time.time() + 120.0
                    )
                    execute(ex, decision)
                    trades_today[sym] = trades_today.get(sym, 0) + 1
                    tg(f"📝 NY-ORB trade placed for {sym}. Today count={trades_today[sym]}/{max_trades}")

            except Exception as e:
                tg(f"⚠️ NY-ORB error {sym}: {e}")
                if debug_mode:
                    import traceback
                    tg(f"Stack trace: {traceback.format_exc()}")

        # Poll positions periodically
        if time.time() - last_poll >= 10.0:
            try:
                poll_positions_and_report(ex)
            except Exception as pe:
                if debug_mode:
                    tg(f"⚠️ NY-ORB poll error: {pe}")
            last_poll = time.time()

        # Sleep until next check
        elapsed = time.time() - t0
        sleep_for = max(5.0, check_interval - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
