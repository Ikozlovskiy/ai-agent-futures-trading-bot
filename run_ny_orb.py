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
            if debug_mode:
                tg(f"🔍 Checking {len(pending_orders)} pending orders")

            for sym in list(pending_orders.keys()):
                try:
                    pending_info = pending_orders[sym]
                    order_id = pending_info["order_id"]
                    side = pending_info["side"]
                    sig = pending_info["signal"]
                    invalidation_level = float(sig["invalidation_level"])

                    if debug_mode:
                        tg(f"🔍 Checking pending order for {sym}: order_id={order_id}, side={side}")

                    # Fetch current price
                    ticker = ex.fetch_ticker(sym)
                    current_price = float(ticker["last"])

                    if debug_mode:
                        tg(f"🔍 {sym} current price: {current_price}, invalidation: {invalidation_level}")

                    # Check for invalidation
                    is_invalidated = False
                    if side == "long" and current_price < invalidation_level:
                        is_invalidated = True
                    elif side == "short" and current_price > invalidation_level:
                        is_invalidated = True

                    if debug_mode:
                        tg(f"🔍 {sym} is_invalidated: {is_invalidated}")

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
                    has_pos = has_open_position(ex, sym)
                    if debug_mode:
                        tg(f"🔍 {sym} has_open_position: {has_pos}")

                    if has_pos:
                        tg(f"✅ {sym} pending order FILLED - setting up ladder")
                        # Order was filled, setup ladder tracking and place initial SL
                        try:
                            # Fetch the position to get actual entry price
                            positions = ex.fetch_positions([sym])
                            tg(f"🔍 Fetched {len(positions)} positions for {sym}")
                            pos = next((p for p in positions if float(p.get("contracts", 0)) != 0), None)

                            if pos:
                                tg(f"🔍 Found position: entry={pos.get('entryPrice')}, contracts={pos.get('contracts')}, side={pos.get('side')}")
                                entry_price = float(pos["entryPrice"])
                                position_side = pos["side"]
                                qty = abs(float(pos["contracts"]))

                                # Get TP levels from signal
                                tp1 = float(sig.get("tp1", sig["tp"]))
                                tp2 = float(sig.get("tp2", sig["tp"]))
                                tp3 = float(sig.get("tp3", sig["tp"]))
                                initial_sl = float(sig["sl"])

                                # Get TP quantity splits
                                qty_splits_raw = os.getenv("ORB_TP_QTY_PCT", "33,33,100")
                                qty_splits = [float(x.strip()) for x in qty_splits_raw.split(",")]

                                # Calculate individual TP quantities
                                tp1_qty = qty * (qty_splits[0] / 100.0)
                                tp2_qty = qty * ((qty_splits[1] - qty_splits[0]) / 100.0)
                                tp3_qty = qty * ((qty_splits[2] - qty_splits[1]) / 100.0)

                                # Round quantities to exchange precision
                                tp1_qty = float(ex.amount_to_precision(sym, tp1_qty))
                                tp2_qty = float(ex.amount_to_precision(sym, tp2_qty))
                                tp3_qty = float(ex.amount_to_precision(sym, tp3_qty))

                                # Round TP prices to exchange tick size
                                tp1 = float(ex.price_to_precision(sym, tp1))
                                tp2 = float(ex.price_to_precision(sym, tp2))
                                tp3 = float(ex.price_to_precision(sym, tp3))
                                initial_sl = float(ex.price_to_precision(sym, initial_sl))

                                tp_side = "sell" if position_side == "long" else "buy"
                                sl_side = tp_side

                                # Place TP orders with error handling
                                tp1_order = None
                                tp2_order = None
                                tp3_order = None

                                try:
                                    tg(f"📝 Placing TP1 order: {tp1_qty} @ {tp1}")
                                    tp1_order = ex.create_order(
                                        symbol=sym,
                                        type="limit",
                                        side=tp_side,
                                        amount=tp1_qty,
                                        price=tp1,
                                        params={"reduceOnly": True}
                                    )
                                    tg(f"✅ TP1 order placed: ID={tp1_order['id']}")
                                except Exception as e:
                                    tg(f"❌ Failed to place TP1 order for {sym}: {e}")
                                    if debug_mode:
                                        import traceback
                                        tg(f"TP1 trace: {traceback.format_exc()}")

                                try:
                                    tg(f"📝 Placing TP2 order: {tp2_qty} @ {tp2}")
                                    tp2_order = ex.create_order(
                                        symbol=sym,
                                        type="limit",
                                        side=tp_side,
                                        amount=tp2_qty,
                                        price=tp2,
                                        params={"reduceOnly": True}
                                    )
                                    tg(f"✅ TP2 order placed: ID={tp2_order['id']}")
                                except Exception as e:
                                    tg(f"❌ Failed to place TP2 order for {sym}: {e}")
                                    if debug_mode:
                                        import traceback
                                        tg(f"TP2 trace: {traceback.format_exc()}")

                                try:
                                    tg(f"📝 Placing TP3 order: {tp3_qty} @ {tp3}")
                                    tp3_order = ex.create_order(
                                        symbol=sym,
                                        type="limit",
                                        side=tp_side,
                                        amount=tp3_qty,
                                        price=tp3,
                                        params={"reduceOnly": True}
                                    )
                                    tg(f"✅ TP3 order placed: ID={tp3_order['id']}")
                                except Exception as e:
                                    tg(f"❌ Failed to place TP3 order for {sym}: {e}")
                                    if debug_mode:
                                        import traceback
                                        tg(f"TP3 trace: {traceback.format_exc()}")

                                # Place initial SL order (for full position)
                                sl_order = None
                                try:
                                    tg(f"📝 Placing SL order: {qty} @ {initial_sl}")
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
                                    tg(f"✅ SL order placed: ID={sl_order['id']}")
                                except Exception as e:
                                    tg(f"❌ Failed to place SL order for {sym}: {e}")
                                    if debug_mode:
                                        import traceback
                                        tg(f"SL trace: {traceback.format_exc()}")

                                # Setup ladder tracking (only if orders were placed successfully)
                                if tp1_order and tp2_order and tp3_order and sl_order:
                                    orb_positions[sym] = {
                                        "entry": entry_price,
                                        "side": position_side,
                                        "qty": qty,
                                        "tp1": tp1,
                                        "tp2": tp2,
                                        "tp3": tp3,
                                        "tp1_qty": tp1_qty,
                                        "tp2_qty": tp2_qty,
                                        "tp3_qty": tp3_qty,
                                        "tp1_hit": False,
                                        "tp2_hit": False,
                                        "tp3_hit": False,
                                        "current_sl": initial_sl,
                                        "sl_order_id": sl_order["id"],
                                        "tp1_order_id": tp1_order["id"],
                                        "tp2_order_id": tp2_order["id"],
                                        "tp3_order_id": tp3_order["id"],
                                    }

                                    tg(f"✅ NY-ORB order FILLED for {sym}\n"
                                       f"Entry: {entry_price:.6f} | SL: {initial_sl:.6f}\n"
                                       f"TP1: {tp1:.6f} ({qty_splits[0]}% = {tp1_qty})\n"
                                       f"TP2: {tp2:.6f} ({qty_splits[1]-qty_splits[0]}% = {tp2_qty})\n"
                                       f"TP3: {tp3:.6f} ({qty_splits[2]-qty_splits[1]}% = {tp3_qty})\n"
                                       f"Today count: {trades_today[sym]}/{max_trades}")
                                else:
                                    tg(f"⚠️ NY-ORB: Some orders failed to place for {sym}. Position will not be tracked.")

                                trades_today[sym] = trades_today.get(sym, 0) + 1
                            else:
                                tg(f"⚠️ {sym} has_open_position=True but no position found in fetch_positions!")

                        except Exception as bracket_err:
                            tg(f"⚠️ NY-ORB bracket setup error for {sym}: {bracket_err}")
                            if debug_mode:
                                import traceback
                                tg(f"Bracket trace: {traceback.format_exc()}")

                        # Remove from pending tracking
                        tg(f"🗑️ Removing {sym} from pending_orders tracking")
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

        # Monitor ORB ladder positions by checking TP order status
        if orb_positions:
            for sym in list(orb_positions.keys()):
                try:
                    pos_info = orb_positions[sym]

                    # Check if position still exists
                    if not has_open_position(ex, sym):
                        tg(f"ℹ️ NY-ORB position closed for {sym}")
                        del orb_positions[sym]
                        continue

                    entry = pos_info["entry"]
                    side = pos_info["side"]
                    tp1 = pos_info["tp1"]
                    tp2 = pos_info["tp2"]
                    tp1_hit = pos_info["tp1_hit"]
                    tp2_hit = pos_info["tp2_hit"]
                    tp3_hit = pos_info.get("tp3_hit", False)

                    close_side = "sell" if side == "long" else "buy"

                    # Check TP1 order status
                    if not tp1_hit:
                        try:
                            tp1_order = ex.fetch_order(pos_info["tp1_order_id"], sym)
                            if tp1_order["status"] == "closed" or tp1_order["filled"] >= pos_info["tp1_qty"] * 0.95:
                                # TP1 was filled, move SL to breakeven
                                new_sl = entry

                                # Cancel old SL order
                                try:
                                    ex.cancel_order(pos_info["sl_order_id"], sym)
                                except Exception:
                                    pass

                                # Get current position size
                                positions = ex.fetch_positions([sym])
                                pos = next((p for p in positions if float(p.get("contracts", 0)) != 0), None)
                                if pos:
                                    remaining_qty = abs(float(pos["contracts"]))

                                    # Place new SL at breakeven for remaining position
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

                                    fill_price = float(tp1_order.get("average") or tp1_order.get("price") or tp1)
                                    tg(f"🎯 NY-ORB TP1 HIT for {sym}\n"
                                       f"Closed {pos_info['tp1_qty']} at {fill_price:.6f}\n"
                                       f"SL moved to BREAKEVEN: {new_sl:.6f}")
                        except Exception as e:
                            if debug_mode:
                                tg(f"⚠️ NY-ORB TP1 check error for {sym}: {e}")

                    # Check TP2 order status
                    elif not tp2_hit:
                        try:
                            tp2_order = ex.fetch_order(pos_info["tp2_order_id"], sym)
                            if tp2_order["status"] == "closed" or tp2_order["filled"] >= pos_info["tp2_qty"] * 0.95:
                                # TP2 was filled, move SL to TP1
                                new_sl = tp1

                                # Cancel old SL order
                                try:
                                    ex.cancel_order(pos_info["sl_order_id"], sym)
                                except Exception:
                                    pass

                                # Get current position size
                                positions = ex.fetch_positions([sym])
                                pos = next((p for p in positions if float(p.get("contracts", 0)) != 0), None)
                                if pos:
                                    remaining_qty = abs(float(pos["contracts"]))

                                    # Place new SL at TP1 for remaining position
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

                                    fill_price = float(tp2_order.get("average") or tp2_order.get("price") or tp2)
                                    tg(f"🎯 NY-ORB TP2 HIT for {sym}\n"
                                       f"Closed {pos_info['tp2_qty']} at {fill_price:.6f}\n"
                                       f"SL moved to TP1: {new_sl:.6f}")
                        except Exception as e:
                            if debug_mode:
                                tg(f"⚠️ NY-ORB TP2 check error for {sym}: {e}")

                    # Check TP3 order status
                    elif not tp3_hit:
                        try:
                            tp3_order = ex.fetch_order(pos_info["tp3_order_id"], sym)
                            if tp3_order["status"] == "closed" or tp3_order["filled"] >= pos_info["tp3_qty"] * 0.95:
                                # TP3 was filled, position should be fully closed
                                fill_price = float(tp3_order.get("average") or tp3_order.get("price") or pos_info["tp3"])
                                tg(f"🎯 NY-ORB TP3 HIT for {sym}\n"
                                   f"Position fully closed at {fill_price:.6f}")

                                # Remove from tracking
                                del orb_positions[sym]
                        except Exception as e:
                            if debug_mode:
                                tg(f"⚠️ NY-ORB TP3 check error for {sym}: {e}")

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
