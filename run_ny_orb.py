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

    # Track pending orders: {symbol: {"order_id": str, "side": str, "signal": dict, "placed_at": float}}
    pending_orders: Dict[str, Dict] = {}

    # Track ORB positions for ladder management: {symbol: {"entry": float, "side": str, "qty": float, "tp1": float, "tp2": float, "tp3": float, "tp1_hit": bool, "tp2_hit": bool, "current_sl": float}}
    orb_positions: Dict[str, Dict] = {}

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

        # Check pending orders for invalidation or fills
        if pending_orders:
            for sym in list(pending_orders.keys()):
                try:
                    pending_info = pending_orders[sym]
                    order_id = pending_info["order_id"]
                    side = pending_info["side"]
                    sig = pending_info["signal"]
                    invalidation_level = float(sig["invalidation_level"])

                    # Fetch current price
                    ticker = ex.fetch_ticker(sym)
                    current_price = float(ticker["last"])

                    # Check for invalidation
                    is_invalidated = False
                    if side == "long" and current_price < invalidation_level:
                        is_invalidated = True
                    elif side == "short" and current_price > invalidation_level:
                        is_invalidated = True

                    if is_invalidated:
                        # Cancel the pending order
                        try:
                            ex.cancel_order(order_id, sym)
                            tg(f"❌ NY-ORB pending order cancelled for {sym}\n"
                               f"Reason: Price returned {'below' if side=='long' else 'above'} OR {invalidation_level:.6f}\n"
                               f"Current price: {current_price:.6f}")
                        except Exception as cancel_err:
                            tg(f"⚠️ NY-ORB failed to cancel order for {sym}: {cancel_err}")

                        # Remove from tracking
                        del pending_orders[sym]
                        continue

                    # Check if order was filled (became a position)
                    if has_open_position(ex, sym):
                        # Order was filled, setup ladder tracking and place initial SL
                        try:
                            # Fetch the position to get actual entry price
                            positions = ex.fetch_positions([sym])
                            pos = next((p for p in positions if float(p.get("contracts", 0)) != 0), None)

                            if pos:
                                entry_price = float(pos["entryPrice"])
                                position_side = pos["side"]
                                qty = abs(float(pos["contracts"]))

                                # Get TP levels from signal
                                tp1 = float(sig.get("tp1", sig["tp"]))
                                tp2 = float(sig.get("tp2", sig["tp"]))
                                tp3 = float(sig.get("tp3", sig["tp"]))
                                initial_sl = float(sig["sl"])

                                # Place initial SL order
                                sl_side = "sell" if position_side == "long" else "buy"
                                sl_order = ex.create_order(
                                    symbol=sym,
                                    type="stop_market",
                                    side=sl_side,
                                    amount=qty,
                                    price=None,
                                    params={
                                        "stopPrice": initial_sl,
                                        "reduceOnly": True,
                                    }
                                )

                                # Setup ladder tracking
                                orb_positions[sym] = {
                                    "entry": entry_price,
                                    "side": position_side,
                                    "qty": qty,
                                    "tp1": tp1,
                                    "tp2": tp2,
                                    "tp3": tp3,
                                    "tp1_hit": False,
                                    "tp2_hit": False,
                                    "current_sl": initial_sl,
                                    "sl_order_id": sl_order["id"],
                                }

                                trades_today[sym] = trades_today.get(sym, 0) + 1

                                # Get TP quantity splits
                                qty_splits_raw = os.getenv("ORB_TP_QTY_PCT", "33,33,100")
                                qty_splits = [float(x.strip()) for x in qty_splits_raw.split(",")]

                                tg(f"✅ NY-ORB order FILLED for {sym}\n"
                                   f"Entry: {entry_price:.6f} | SL: {initial_sl:.6f}\n"
                                   f"TP1: {tp1:.6f} ({qty_splits[0]}%)\n"
                                   f"TP2: {tp2:.6f} ({qty_splits[1]-qty_splits[0]}%)\n"
                                   f"TP3: {tp3:.6f} ({qty_splits[2]-qty_splits[1]}%)\n"
                                   f"Today count: {trades_today[sym]}/{max_trades}")

                        except Exception as bracket_err:
                            tg(f"⚠️ NY-ORB bracket setup error for {sym}: {bracket_err}")

                        # Remove from pending tracking
                        del pending_orders[sym]
                        continue

                    # Check if order was cancelled externally or expired
                    try:
                        order_status = ex.fetch_order(order_id, sym)
                        if order_status["status"] in ["canceled", "cancelled", "expired", "rejected"]:
                            tg(f"ℹ️ NY-ORB pending order {order_status['status']} for {sym}")
                            del pending_orders[sym]
                    except Exception:
                        # Order might not exist anymore
                        pass

                except Exception as e:
                    if debug_mode:
                        tg(f"⚠️ NY-ORB pending order monitoring error for {sym}: {e}")

        # Monitor ORB ladder positions for progressive TP/SL
        if orb_positions:
            qty_splits_raw = os.getenv("ORB_TP_QTY_PCT", "33,33,100")
            qty_splits = [float(x.strip()) for x in qty_splits_raw.split(",")]

            for sym in list(orb_positions.keys()):
                try:
                    pos_info = orb_positions[sym]

                    # Check if position still exists
                    if not has_open_position(ex, sym):
                        tg(f"ℹ️ NY-ORB position closed for {sym}")
                        del orb_positions[sym]
                        continue

                    # Fetch current position and price
                    positions = ex.fetch_positions([sym])
                    pos = next((p for p in positions if float(p.get("contracts", 0)) != 0), None)
                    if not pos:
                        del orb_positions[sym]
                        continue

                    ticker = ex.fetch_ticker(sym)
                    current_price = float(ticker["last"])
                    current_qty = abs(float(pos["contracts"]))

                    entry = pos_info["entry"]
                    side = pos_info["side"]
                    original_qty = pos_info["qty"]
                    tp1 = pos_info["tp1"]
                    tp2 = pos_info["tp2"]
                    tp3 = pos_info["tp3"]
                    tp1_hit = pos_info["tp1_hit"]
                    tp2_hit = pos_info["tp2_hit"]
                    current_sl = pos_info["current_sl"]

                    # TP1 logic
                    if not tp1_hit:
                        tp1_reached = (current_price >= tp1) if side == "long" else (current_price <= tp1)
                        if tp1_reached:
                            # Close first portion (qty_splits[0]%)
                            close_qty = original_qty * (qty_splits[0] / 100.0)
                            close_qty = float(ex.amount_to_precision(sym, close_qty))

                            close_side = "sell" if side == "long" else "buy"
                            ex.create_order(
                                symbol=sym,
                                type="market",
                                side=close_side,
                                amount=close_qty,
                                params={"reduceOnly": True}
                            )

                            # Move SL to breakeven (entry price)
                            new_sl = entry
                            try:
                                # Cancel old SL order
                                ex.cancel_order(pos_info["sl_order_id"], sym)
                            except Exception:
                                pass

                            # Place new SL at breakeven
                            remaining_qty = current_qty - close_qty
                            sl_order = ex.create_order(
                                symbol=sym,
                                type="stop_market",
                                side=close_side,
                                amount=remaining_qty,
                                price=None,
                                params={
                                    "stopPrice": new_sl,
                                    "reduceOnly": True,
                                }
                            )

                            pos_info["tp1_hit"] = True
                            pos_info["current_sl"] = new_sl
                            pos_info["sl_order_id"] = sl_order["id"]

                            tg(f"🎯 NY-ORB TP1 HIT for {sym}\n"
                               f"Closed {qty_splits[0]}% at {current_price:.6f}\n"
                               f"SL moved to BREAKEVEN: {new_sl:.6f}")

                    # TP2 logic
                    elif not tp2_hit:
                        tp2_reached = (current_price >= tp2) if side == "long" else (current_price <= tp2)
                        if tp2_reached:
                            # Close second portion (qty_splits[1] - qty_splits[0])%
                            close_pct = qty_splits[1] - qty_splits[0]
                            close_qty = original_qty * (close_pct / 100.0)
                            close_qty = float(ex.amount_to_precision(sym, close_qty))

                            close_side = "sell" if side == "long" else "buy"
                            ex.create_order(
                                symbol=sym,
                                type="market",
                                side=close_side,
                                amount=close_qty,
                                params={"reduceOnly": True}
                            )

                            # Move SL to TP1 price
                            new_sl = tp1
                            try:
                                # Cancel old SL order
                                ex.cancel_order(pos_info["sl_order_id"], sym)
                            except Exception:
                                pass

                            # Place new SL at TP1
                            remaining_qty = current_qty - close_qty
                            sl_order = ex.create_order(
                                symbol=sym,
                                type="stop_market",
                                side=close_side,
                                amount=remaining_qty,
                                price=None,
                                params={
                                    "stopPrice": new_sl,
                                    "reduceOnly": True,
                                }
                            )

                            pos_info["tp2_hit"] = True
                            pos_info["current_sl"] = new_sl
                            pos_info["sl_order_id"] = sl_order["id"]

                            tg(f"🎯 NY-ORB TP2 HIT for {sym}\n"
                               f"Closed {close_pct:.0f}% at {current_price:.6f}\n"
                               f"SL moved to TP1: {new_sl:.6f}")

                    # TP3 logic
                    else:
                        tp3_reached = (current_price >= tp3) if side == "long" else (current_price <= tp3)
                        if tp3_reached:
                            # Close remaining position (100% of what's left)
                            close_side = "sell" if side == "long" else "buy"
                            ex.create_order(
                                symbol=sym,
                                type="market",
                                side=close_side,
                                amount=current_qty,
                                params={"reduceOnly": True}
                            )

                            tg(f"🎯 NY-ORB TP3 HIT for {sym}\n"
                               f"Position fully closed at {current_price:.6f}")

                            # Remove from tracking
                            del orb_positions[sym]

                except Exception as e:
                    if debug_mode:
                        tg(f"⚠️ NY-ORB ladder monitoring error for {sym}: {e}")
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
