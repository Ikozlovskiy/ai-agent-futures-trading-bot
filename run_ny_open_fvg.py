import os
import time
from typing import List, Dict

from utils import tg, parse_map_env, get_per_symbol_value
from datahub import build_exchange
from strategies.ny_open_fvg import NyOpenFVGInspector
from executor import execute, poll_positions_and_report, has_open_position
from models import Decision

def _countdown(or_hhmm_utc: str) -> str:
    try:
        from datetime import datetime, timezone, timedelta
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
    raw = os.getenv("NY_OPEN_SYMBOLS") or os.getenv("SYMBOLS")
    if not raw:
        # Default to the three requested symbols
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    out = []
    for s in str(raw).split(","):
        s = s.strip()
        if s:
            out.append(s)
    if not out:
        out = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    return out


def main():
    ex = build_exchange()
    symbols = _symbols_from_env()

    # Configurable parameters via .env
    or_start_utc = os.getenv("OR_START_UTC", "14:30")  # keep UTC; timezone/session integration added later
    or_minutes = int(os.getenv("OR_MINUTES", "15") or 15)
    or_timeframe = os.getenv("OR_TIMEFRAME", "15m")
    fvg_timeframe = os.getenv("FVG_TIMEFRAME", "5m")
    check_interval = int(os.getenv("FVG_CHECK_INTERVAL", "300") or 300)  # 5 minutes in seconds
    allow_outside = (os.getenv("FVG_ALLOW_OUTSIDE", "true").lower() == "true")
    require_break_within = os.getenv("REQUIRE_BREAK_WITHIN")
    require_break_within = int(require_break_within) if (require_break_within and require_break_within.isdigit()) else None

    insp = NyOpenFVGInspector(
        or_start_hhmm_utc=or_start_utc,
        or_minutes=or_minutes,
        or_timeframe=or_timeframe,
        fvg_timeframe=fvg_timeframe,
        allow_outside_or=allow_outside,
        require_break_within=require_break_within,
    )

    # Disable dynamic re-arms for FVG strategy (incompatible with LADDER mode)
    # Note: BRACKET_MODE can be set in .env (ATR, ROE, FIXED_PCT, or LADDER)
    # For LADDER mode, FVG entry price is used with LADDER_TP_PCT and LADDER_SL_PCT
    os.environ["DYN_ROI_STAGES"] = ""   # Disable dynamic ladder for FVG strategy

    # Sizing per symbol
    size_default = float(os.getenv("RISK_NOTIONAL_USDT", "50") or 50.0)
    size_map_raw = parse_map_env("RISK_NOTIONAL_MAP")
    size_map = {k: float(v) for k, v in size_map_raw.items()}
    symbol_size = {s: get_per_symbol_value(s, size_map, size_default) for s in symbols}

    # Daily counters (UTC) with reset at midnight
    max_trades = int(os.getenv("NY_OPEN_MAX_TRADES", "2") or 2)
    trades_today: Dict[str, int] = {s: 0 for s in symbols}
    last_reset_day = time.gmtime().tm_yday

    # Logging keepalive (per symbol)
    keepalive_secs = int(os.getenv("NY_OPEN_LOG_KEEPALIVE_SEC", "300") or 300)

    debug_mode = (os.getenv("NY_OPEN_DEBUG", "false").lower() == "true")

    if debug_mode:
        tg(f"🤖 NY-Open FVG inspector started | Symbols={symbols} | OR_START_UTC={or_start_utc} | OR={or_timeframe}/{or_minutes}min | FVG_TF={fvg_timeframe} | CheckInterval={check_interval}s | allow_outside={allow_outside} | req_break_within={require_break_within} | RR={os.getenv('FVG_RR','2.0')} | MaxTrades/day={max_trades}")

    # Track last confirmed index per symbol/side to reduce spam
    last_confirm_key: Dict[str, int] = {}
    last_log_ts: Dict[str, float] = {}
    last_hourly_ping = 0.0
    last_poll = 0.0

    def _weekday_mon_fri() -> bool:
        # 0=Mon, ... 6=Sun
        return time.gmtime().tm_wday < 5

    trade_enabled = (os.getenv("NY_OPEN_TRADE", "true").lower() == "true")

    while True:
        t0 = time.time()

        # UTC daily reset of counters
        cur_day = time.gmtime().tm_yday
        if cur_day != last_reset_day:
            trades_today = {s: 0 for s in symbols}
            last_reset_day = cur_day
            tg("🗓️ NY-Open FVG: daily counters reset (UTC midnight)")

        # Hourly countdown ping
        if time.time() - last_hourly_ping >= 3600.0:
            eta = _countdown(or_start_utc)
            tg(f"⏳ NY-Open FVG | Next session starts in {eta} (UTC {or_start_utc})")
            last_hourly_ping = time.time()

        for sym in symbols:
            try:
                payload = insp.analyze_symbol(ex, sym, limit=500)
                sig = payload.get("signal")

                # Robust per-symbol keepalive logs (no fragile modulo window)
                now_ts = time.time()
                should_keepalive = (now_ts - last_log_ts.get(sym, 0.0)) >= keepalive_secs

                if sig:
                    key = f"{sym}:{sig['side']}"
                    prev_idx = last_confirm_key.get(key, -1)
                    # New confirmation -> log immediately (only in debug mode for FVG details)
                    if int(sig["confirm_i"]) != int(prev_idx):
                        if debug_mode:
                            insp.log_payload(payload)
                        last_confirm_key[key] = int(sig["confirm_i"])
                        last_log_ts[sym] = now_ts
                    else:
                        # Same confirmation; still emit a heartbeat periodically (show in production)
                        if should_keepalive:
                            insp.log_payload({**payload, "signal": None}, debug=debug_mode)
                            last_log_ts[sym] = now_ts
                else:
                    # No signal yet; still log on keepalive (show in production)
                    if should_keepalive:
                        insp.log_payload(payload, debug=debug_mode)
                        last_log_ts[sym] = now_ts

                # Trading branch
                if trade_enabled and sig and payload.get("or_ready", False):
                    # Only weekdays
                    if not _weekday_mon_fri():
                        continue
                    # Respect daily max trades
                    if trades_today.get(sym, 0) >= max_trades:
                        continue
                    # Ensure touch happens after OR period closes
                    faoi = payload.get("first_after_or_i")
                    try:
                        if faoi is None or int(sig.get("touch_i", -1)) < int(faoi):
                            continue
                    except Exception:
                        continue
                    # Skip if there is already an open position
                    if has_open_position(ex, sym):
                        continue

                    # Build Decision from signal (market entry, RR-based brackets)
                    reason = {
                        "strategy": "NY_OPEN_FVG",
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
                        confidence=0.60,
                        reason=reason,
                        valid_until=time.time() + 120.0
                    )
                    execute(ex, decision)
                    trades_today[sym] = trades_today.get(sym, 0) + 1
                    # Trade placement log is always sent (not controlled by debug mode)
                    tg(f"📝 NY-Open FVG trade placed for {sym}. Today count={trades_today[sym]}/{max_trades}")

            except Exception as e:
                # Always log symbol-specific errors to diagnose issues
                tg(f"⚠️ NY-Open FVG error {sym}: {e}")
                if debug_mode:
                    import traceback
                    tg(f"Stack trace: {traceback.format_exc()}")

        # Poll positions periodically to handle exits and PnL logs
        if time.time() - last_poll >= 10.0:
            try:
                poll_positions_and_report(ex)
            except Exception as pe:
                if debug_mode:
                    tg(f"⚠️ NY-Open FVG poll error: {pe}")
            last_poll = time.time()

        # Pace based on configured check interval (e.g., 5 minutes)
        elapsed = time.time() - t0
        sleep_for = max(5.0, check_interval - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
