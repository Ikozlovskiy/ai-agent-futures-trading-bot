"""
Multi-Confluence Scalper Runner
24/7 execution with time-weighted confidence and dynamic position management.
"""

import os
import time
from datetime import datetime, timezone

from utils import tg
from datahub import build_exchange
from strategies.multi_confluence_scalper import MultiConfluenceScalper
from executor import execute, poll_positions_and_report, has_open_position
from models import Decision


def main():
    ex = build_exchange()

    # Configuration
    symbols_raw = os.getenv("SCALP_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT")
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

    check_interval = int(os.getenv("SCALP_CHECK_INTERVAL", "10") or 10)
    risk_pct = float(os.getenv("SCALP_RISK_PCT", "0.5") or 0.5)
    max_trades_per_day = int(os.getenv("SCALP_MAX_TRADES_PER_DAY", "10") or 10)
    daily_loss_cap = float(os.getenv("SCALP_DAILY_LOSS_CAP_USDT", "100") or 100)
    min_loss_cooldown = int(os.getenv("SCALP_MIN_LOSS_COOLDOWN_SEC", "600") or 600)

    # Sizing
    risk_notional = float(os.getenv("RISK_NOTIONAL_USDT", "1000") or 1000)

    # Initialize strategy
    scalper = MultiConfluenceScalper()

    # State tracking
    trades_today = {s: 0 for s in symbols}
    last_reset_day = time.gmtime().tm_yday
    last_trade_time = {}
    last_trade_result = {}  # 'win' or 'loss'
    daily_pnl = 0.0

    tg(
        f"🚀 <b>Multi-Confluence Scalper Started</b>\n"
        f"Symbols: {', '.join(symbols)}\n"
        f"Check Interval: {check_interval}s\n"
        f"Risk: {risk_pct}% per trade\n"
        f"Max Trades/Day: {max_trades_per_day}\n"
        f"Daily Loss Cap: ${daily_loss_cap}"
    )

    last_poll = 0.0
    last_hourly_log = 0.0

    while True:
        try:
            now = time.time()

            # Daily reset
            cur_day = time.gmtime().tm_yday
            if cur_day != last_reset_day:
                trades_today = {s: 0 for s in symbols}
                daily_pnl = 0.0
                last_reset_day = cur_day
                tg("🗓️ Scalper: Daily counters reset (UTC midnight)")

            # Hourly status log
            if now - last_hourly_log >= 3600:
                hour_utc = datetime.now(timezone.utc).hour
                time_weight = scalper.get_time_weight()
                tg(
                    f"⏰ Scalper Status | UTC Hour: {hour_utc:02d}:00 | "
                    f"Time Weight: {time_weight:.0%} | "
                    f"Trades Today: {sum(trades_today.values())}/{max_trades_per_day}"
                )
                last_hourly_log = now

            # Check if we can trade (only 1 position at a time)
            has_position = any(has_open_position(ex, s) for s in symbols)

            if not has_position:
                # Check daily limits
                if sum(trades_today.values()) >= max_trades_per_day:
                    time.sleep(check_interval)
                    continue

                if daily_pnl <= -abs(daily_loss_cap):
                    time.sleep(check_interval)
                    continue

                # Scan for signals
                for sym in symbols:
                    # Skip if recent loss (cooldown)
                    if last_trade_result.get(sym) == 'loss':
                        time_since_loss = now - last_trade_time.get(sym, 0)
                        if time_since_loss < min_loss_cooldown:
                            continue

                    # Analyze symbol
                    signal = scalper.analyze_symbol(ex, sym)

                    if signal:
                        # Log signal
                        scalper.log_signal(signal)

                        # Build decision
                        decision = Decision(
                            symbol=sym,
                            side=signal["side"],
                            entry_type="market",
                            size_usdt=risk_notional,
                            sl=signal["sl"],
                            tp=signal["tp3"],  # Use TP3 as final TP for executor
                            confidence=signal["confidence"],
                            reason={
                                "strategy": "MultiConfluenceScalper",
                                "pattern": signal["pattern"],
                                "htf_trend": signal["htf_trend"],
                                "time_weight": signal["time_weight"],
                            },
                            valid_until=now + 60.0
                        )

                        # Execute trade
                        execute(ex, decision)

                        # Update state
                        trades_today[sym] = trades_today.get(sym, 0) + 1
                        last_trade_time[sym] = now

                        # Break after taking one position (single position rule)
                        break

            # Poll positions
            if now - last_poll >= 5.0:
                poll_positions_and_report(ex)
                last_poll = now

            # Sleep
            time.sleep(check_interval)

        except KeyboardInterrupt:
            tg("👋 Scalper stopped by user")
            break
        except Exception as e:
            tg(f"⚠️ Scalper main loop error: {e}")
            import traceback
            tg(f"🔍 Traceback: {traceback.format_exc()}")
            time.sleep(30)


if __name__ == "__main__":
    main()
