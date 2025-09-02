import os, time
from typing import List
from utils import next_close_epoch, env_bool, parse_map_env, get_per_symbol_value, tg
from datahub import build_exchange, MarketState
from policy import decide
from executor import execute, poll_positions_and_report, has_open_position

def main():
    ex = build_exchange()

    symbols: List[str] = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",") if s.strip()]
    HTF = os.getenv("HTF", "1h")
    ITF = os.getenv("ITF", "15m")
    LTF = os.getenv("LTF", "1m")

    STRAT_DEFAULT = os.getenv("STRATEGY_COMBO", "A").upper()
    STRAT_MAP = {k: v.upper() for k, v in parse_map_env("STRATEGY_MAP").items()}

    size_default = float(os.getenv("RISK_NOTIONAL_USDT", "5"))
    size_map_raw = parse_map_env("RISK_NOTIONAL_MAP")
    size_map = {k: float(v) for k, v in size_map_raw.items()}

    atr_period = int(os.getenv("ATR_PERIOD", "14"))
    sl_mult = float(os.getenv("ATR_SL_MULT", "1.0"))
    tp_mult = float(os.getenv("ATR_TP_MULT", "2.0"))
    log_signals_only = env_bool("LOG_SIGNALS_ONLY", False)

    ms = MarketState(symbols, HTF, ITF, LTF, atr_period)

    due = {
        "htf": next_close_epoch(HTF),
        "itf": next_close_epoch(ITF),
        "ltf": next_close_epoch(LTF),
        "poll": int(time.time()) + 10
    }
    tg(f"🤖 Agent started. Symbols={symbols}  TFs={HTF}/{ITF}/{LTF}\n"
       f"Combos={ {s: STRAT_MAP.get(s, STRAT_DEFAULT) for s in symbols} }\n"
       f"RiskUSDT={ {s: get_per_symbol_value(s, size_map, size_default) for s in symbols} }")

    while True:
        now = int(time.time())
        sleep_for = min(due.values()) - now + 2
        if sleep_for > 0:
            time.sleep(sleep_for)

        if int(time.time()) >= due["htf"]:
            try:
                ms.refresh_htf_mixed(ex, {s: STRAT_MAP.get(s, STRAT_DEFAULT) for s in symbols})
            except Exception as e:
                tg(f"⚠️ HTF refresh error: {e}")
            due["htf"] = next_close_epoch(HTF)

        if int(time.time()) >= due["itf"]:
            try:
                ms.refresh_itf_mixed(ex, {s: STRAT_MAP.get(s, STRAT_DEFAULT) for s in symbols})
            except Exception as e:
                tg(f"⚠️ ITF refresh error: {e}")
            due["itf"] = next_close_epoch(ITF)

        if int(time.time()) >= due["ltf"]:
            for sym in symbols:
                try:
                    ltf_snapshot = ms.build_ltf_snapshot(ex, sym)
                    htf_map = ms.htf_maps.get(sym)
                    itf_setup = ms.itf_setups.get(sym)
                    combo = STRAT_MAP.get(sym, STRAT_DEFAULT)
                    size_usdt = get_per_symbol_value(sym, size_map, size_default)
                    decision = decide(sym, combo, htf_map, itf_setup, ltf_snapshot, size_usdt, sl_mult, tp_mult)
                    if decision:
                        if log_signals_only:
                            from pprint import pformat
                            tg(f"📣 <b>SIGNAL</b> (DRY) {sym} {decision.side.upper()} conf={decision.confidence}\n{pformat(decision.reason)}")
                        else:
                            if has_open_position(ex, sym):
                                tg(f"⏸️ Skip {sym}: position already open.")
                                continue
                            execute(ex, decision)
                except Exception as e:
                    tg(f"⚠️ LTF decision error {sym}: {e}")
            due["ltf"] = next_close_epoch(LTF)

        if int(time.time()) >= due["poll"]:
            try:
                poll_positions_and_report(ex)
            except Exception as e:
                tg(f"⚠️ poll error: {e}")
            due["poll"] = int(time.time()) + 10

        if int(time.time()) >= due.get("heartbeat", 0):
            open_syms = list(ms.itf_setups.keys())
            tg(f"🫀 Heartbeat: bot alive. Open positions={list(open_syms)}")
            due["heartbeat"] = int(time.time()) + 3600

if __name__ == "__main__":
    main()
