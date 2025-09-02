import os, time, math, requests
from dotenv import load_dotenv

load_dotenv()

TF_SEC = {"1m":60, "3m":180, "5m":300, "15m":900, "1h":3600, "4h":14400, "1d":86400}

def tf_seconds(tf: str) -> int:
    unit = tf[-1]
    mult = {"m":60,"h":3600,"d":86400}[unit]
    return int(tf[:-1]) * mult

def next_close_epoch(tf: str) -> int:
    sec = tf_seconds(tf)
    now = time.time()
    return int(math.ceil(now / sec) * sec)

def env_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return v.lower() in ("1","true","yes","y","on")

def tg(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat  = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat): 
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass

def parse_map_env(name: str) -> dict:
    """Parse 'A:B,C:D' or 'SYM:A,SYM2:B' into dict. Trims spaces."""
    raw = os.getenv(name, "")
    out = {}
    if not raw:
        return out
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = [x.strip() for x in p.split(":", 1)]
        out[k] = v
    return out

def get_per_symbol_value(symbol: str, map_env: dict, default_value):
    return map_env.get(symbol, default_value)
