from dataclasses import dataclass
from typing import Dict, Literal

Side = Literal["long", "short"]
EntryType = Literal["market", "limit"]

@dataclass
class Decision:
    symbol: str
    side: Side
    entry_type: EntryType
    size_usdt: float
    sl: float
    tp: float
    confidence: float
    reason: Dict[str, str]
    valid_until: float  # epoch seconds

@dataclass
class PositionMemo:
    symbol: str
    side: Side
    qty: float
    entry_price: float
    opened_at: float
