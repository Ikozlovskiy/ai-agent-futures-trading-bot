#!/usr/bin/env python3
"""
Backtesting script for ORB and NY Open FVG strategies.
Fetches historical data from Binance and simulates strategy execution.

Usage:
  python backtest.py --strategy orb --period 2025-01 [--expand]
  python backtest.py --strategy fvg --period 2024 [--expand]
  python backtest.py --strategy orb --start 2024-12-01 --end 2025-01-15 [--expand]
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
from dotenv import load_dotenv
import calendar

# Import strategy classes
from strategies.ny_orb import NyOrbInspector
from strategies.ny_open_fvg import NyOpenFVGInspector, detect_fvgs, find_touch_and_confirm
from datahub import build_exchange, fetch_candles
from utils import parse_map_env, get_per_symbol_value


class BacktestResult:
    """Container for backtest results"""
    def __init__(self):
        self.trades: List[Dict] = []
        self.total_pnl: float = 0.0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_trades: int = 0
        self.largest_win: float = 0.0
        self.largest_loss: float = 0.0
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0
        self.max_consecutive_wins: int = 0
        self.max_consecutive_losses: int = 0
        self.max_drawdown: float = 0.0
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0

    def add_trade(self, trade: Dict):
        """Add a trade and update statistics"""
        self.trades.append(trade)
        self.total_trades += 1
        pnl = trade['pnl']
        self.total_pnl += pnl
        self.current_equity += pnl

        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
            if pnl > self.largest_win:
                self.largest_win = pnl
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
            if pnl < self.largest_loss:
                self.largest_loss = pnl

        # Track drawdown
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        else:
            drawdown = self.peak_equity - self.current_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def average_win(self) -> float:
        if self.winning_trades == 0:
            return 0.0
        return sum(t['pnl'] for t in self.trades if t['pnl'] > 0) / self.winning_trades

    @property
    def average_loss(self) -> float:
        if self.losing_trades == 0:
            return 0.0
        return sum(t['pnl'] for t in self.trades if t['pnl'] < 0) / self.losing_trades

    @property
    def profit_factor(self) -> float:
        total_wins = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        total_losses = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        if total_losses == 0:
            return float('inf') if total_wins > 0 else 0.0
        return total_wins / total_losses


def parse_period(period_str: str) -> Tuple[datetime, datetime]:
    """
    Parse period string into start and end dates.
    Formats supported:
    - '2025-01' -> January 2025
    - '2025' -> Full year 2025
    - 'last_month' -> Previous calendar month
    - 'last_year' -> Previous calendar year
    """
    now = datetime.now(timezone.utc)

    if period_str == 'last_month':
        # Get first day of current month, then go back one month
        first_of_month = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        end = first_of_month - timedelta(days=1)
        start = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
        # Add time to cover full day
        end = end.replace(hour=23, minute=59, second=59)
        return start, end

    elif period_str == 'last_year':
        year = now.year - 1
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        return start, end

    elif len(period_str) == 4:  # Year only: '2025'
        year = int(period_str)
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        return start, end

    elif len(period_str) == 7:  # Year-Month: '2025-01'
        year, month = map(int, period_str.split('-'))
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        # Get last day of month
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        end = end.replace(hour=23, minute=59, second=59)
        return start, end

    else:
        raise ValueError(f"Invalid period format: {period_str}")


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format"""
    return datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)


def calculate_or_for_date(candles: np.ndarray, target_date: datetime.date, 
                          or_start_hhmm: str, or_minutes: int) -> Optional[Tuple[float, float, int, int]]:
    """
    Calculate Opening Range for a specific historical date.
    Returns: (or_high, or_low, or_start_idx, or_end_idx) or None
    """
    # Parse OR start time (e.g., "14:30")
    hh, mm = map(int, or_start_hhmm.split(':'))

    # Create OR window for the target date
    or_start = datetime(target_date.year, target_date.month, target_date.day, 
                       hh, mm, 0, tzinfo=timezone.utc)
    or_end = or_start + timedelta(minutes=or_minutes)

    or_start_ms = int(or_start.timestamp() * 1000)
    or_end_ms = int(or_end.timestamp() * 1000)

    # Find candles within OR window
    or_candles = []
    or_start_idx = None
    or_end_idx = None

    for idx, candle in enumerate(candles):
        ts = candle[0]
        if or_start_ms <= ts < or_end_ms:
            if or_start_idx is None:
                or_start_idx = idx
            or_candles.append(candle)
            or_end_idx = idx

    if len(or_candles) == 0:
        return None

    # Calculate OR high and low
    highs = [c[2] for c in or_candles]
    lows = [c[3] for c in or_candles]
    or_high = float(max(highs))
    or_low = float(min(lows))

    return or_high, or_low, or_start_idx, or_end_idx


def fetch_historical_data(ex, symbol: str, timeframe: str, start: datetime, end: datetime) -> np.ndarray:
    """
    Fetch historical OHLCV data from Binance for the given period.
    Returns numpy array with columns: [ts, o, h, l, c, v]
    """
    print(f"  Fetching {symbol} {timeframe} data from {start.date()} to {end.date()}...")

    # Convert to milliseconds
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_candles = []
    current_start = start_ms

    # Binance has a limit of 1500 candles per request
    limit = 1500

    # Get the correct underlying exchange based on symbol
    if hasattr(ex, '_get_exchange'):
        # BinanceFuturesWrapper - get underlying exchange
        underlying_ex = ex._get_exchange(symbol)
    else:
        # Direct exchange object
        underlying_ex = ex

    while current_start < end_ms:
        try:
            ohlcv = underlying_ex.fetch_ohlcv(symbol, timeframe=timeframe, since=current_start, limit=limit)
            if not ohlcv:
                break

            # Filter candles within our range
            filtered = [c for c in ohlcv if start_ms <= c[0] <= end_ms]
            all_candles.extend(filtered)

            # Move to next batch
            if len(ohlcv) < limit:
                break
            current_start = ohlcv[-1][0] + 1

        except Exception as e:
            print(f"  Error fetching data: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            break

    if not all_candles:
        return np.array([])

    # Remove duplicates and sort
    unique_candles = {c[0]: c for c in all_candles}
    sorted_candles = sorted(unique_candles.values(), key=lambda x: x[0])

    print(f"  Fetched {len(sorted_candles)} candles")
    return np.array(sorted_candles, dtype=float)


def simulate_trade(entry: float, sl: float, tp: float, side: str, size_usdt: float, 
                   candles: np.ndarray, entry_idx: int, fee_rate: float = 0.0005) -> Dict:
    """
    Simulate a trade from entry until SL or TP is hit.
    Returns trade result dict with PnL, exit price, exit reason, etc.
    """
    # Calculate quantity
    qty = size_usdt / entry

    # Entry fee
    entry_fee = size_usdt * fee_rate

    # Check each subsequent candle for SL/TP hit
    for i in range(entry_idx + 1, len(candles)):
        ts, o, h, l, c, v = candles[i]

        if side == "long":
            # Check if SL hit (low touches or breaks SL)
            if l <= sl:
                exit_price = sl
                exit_reason = "SL"
                pnl = (exit_price - entry) * qty
                exit_fee = abs(exit_price * qty) * fee_rate
                net_pnl = pnl - entry_fee - exit_fee
                return {
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'exit_ts': int(ts),
                    'exit_idx': i,
                    'pnl': net_pnl,
                    'gross_pnl': pnl,
                    'fees': entry_fee + exit_fee,
                    'bars_held': i - entry_idx
                }
            # Check if TP hit (high touches or breaks TP)
            elif h >= tp:
                exit_price = tp
                exit_reason = "TP"
                pnl = (exit_price - entry) * qty
                exit_fee = abs(exit_price * qty) * fee_rate
                net_pnl = pnl - entry_fee - exit_fee
                return {
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'exit_ts': int(ts),
                    'exit_idx': i,
                    'pnl': net_pnl,
                    'gross_pnl': pnl,
                    'fees': entry_fee + exit_fee,
                    'bars_held': i - entry_idx
                }
        else:  # short
            # Check if SL hit (high touches or breaks SL)
            if h >= sl:
                exit_price = sl
                exit_reason = "SL"
                pnl = (entry - exit_price) * qty
                exit_fee = abs(exit_price * qty) * fee_rate
                net_pnl = pnl - entry_fee - exit_fee
                return {
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'exit_ts': int(ts),
                    'exit_idx': i,
                    'pnl': net_pnl,
                    'gross_pnl': pnl,
                    'fees': entry_fee + exit_fee,
                    'bars_held': i - entry_idx
                }
            # Check if TP hit (low touches or breaks TP)
            elif l <= tp:
                exit_price = tp
                exit_reason = "TP"
                pnl = (entry - exit_price) * qty
                exit_fee = abs(exit_price * qty) * fee_rate
                net_pnl = pnl - entry_fee - exit_fee
                return {
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'exit_ts': int(ts),
                    'exit_idx': i,
                    'pnl': net_pnl,
                    'gross_pnl': pnl,
                    'fees': entry_fee + exit_fee,
                    'bars_held': i - entry_idx
                }

    # Trade never hit SL or TP (end of data)
    exit_price = candles[-1][4]  # Last close
    if side == "long":
        pnl = (exit_price - entry) * qty
    else:
        pnl = (entry - exit_price) * qty
    exit_fee = abs(exit_price * qty) * fee_rate
    net_pnl = pnl - entry_fee - exit_fee

    return {
        'exit_price': exit_price,
        'exit_reason': 'EOD',  # End of data
        'exit_ts': int(candles[-1][0]),
        'exit_idx': len(candles) - 1,
        'pnl': net_pnl,
        'gross_pnl': pnl,
        'fees': entry_fee + exit_fee,
        'bars_held': len(candles) - 1 - entry_idx
    }


def backtest_orb(ex, symbols: List[str], start: datetime, end: datetime, 
                 risk_notional: float, risk_map: Dict[str, float]) -> BacktestResult:
    """Backtest ORB strategy by manually calculating OR and detecting breakouts"""
    print("\n=== ORB Strategy Backtest ===")
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Symbols: {', '.join(symbols)}")

    result = BacktestResult()

    # Get OR parameters from config
    or_start_hhmm = os.getenv("ORB_START_UTC", "14:30")
    or_minutes = int(os.getenv("ORB_MINUTES", "15") or 15)
    signal_timeframe = os.getenv("ORB_SIGNAL_TIMEFRAME", "5m")
    rr_ratio = float(os.getenv("ORB_RR", "2.0") or 2.0)
    max_trades_per_day = int(os.getenv("ORB_MAX_TRADES", "1") or 1)

    for symbol in symbols:
        print(f"\n--- Backtesting {symbol} ---")
        size_usdt = get_per_symbol_value(symbol, risk_map, risk_notional)
        print(f"Position size: {size_usdt} USDT")

        # Fetch data
        candles = fetch_historical_data(ex, symbol, signal_timeframe, start, end)
        if len(candles) == 0:
            print(f"  No data available for {symbol}")
            continue

        print(f"  Processing {len(candles)} candles...")

        # Group by day
        from collections import defaultdict
        days_data = defaultdict(list)
        for idx, candle in enumerate(candles):
            ts = candle[0]
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            day_key = dt.date()
            days_data[day_key].append((idx, candle))

        print(f"  Found {len(days_data)} trading days")
        daily_trade_count = defaultdict(int)

        # Process each day
        for day_key in sorted(days_data.keys()):
            if daily_trade_count[day_key] >= max_trades_per_day:
                continue

            # Calculate OR for this specific day
            or_result = calculate_or_for_date(candles, day_key, or_start_hhmm, or_minutes)
            if not or_result:
                continue

            or_high, or_low, or_start_idx, or_end_idx = or_result
            or_mid = (or_high + or_low) / 2.0

            # Look for breakout after OR ends
            breakout_found = False
            for idx, candle in days_data[day_key]:
                if idx <= or_end_idx:
                    continue  # Still in OR period

                ts, o, h, l, c, v = candle

                # Bullish breakout
                if c > or_high and not breakout_found:
                    entry_price = or_high * 1.0005  # Small buffer for pending order
                    sl = or_mid
                    risk = entry_price - sl
                    tp = entry_price + (rr_ratio * risk)
                    side = "long"
                    breakout_found = True

                    print(f"  ✓ LONG breakout on {day_key} @ {entry_price:.2f} (OR: {or_low:.2f}-{or_high:.2f})")

                    trade_result = simulate_trade(
                        entry=entry_price, sl=sl, tp=tp, side=side,
                        size_usdt=size_usdt, candles=candles,
                        entry_idx=idx, fee_rate=0.0005
                    )

                    trade = {
                        'symbol': symbol,
                        'entry_ts': int(ts),
                        'entry_date': day_key.strftime('%Y-%m-%d'),
                        'entry_price': entry_price,
                        'sl': sl, 'tp': tp, 'side': side,
                        'size_usdt': size_usdt,
                        'pattern': 'orb_long',
                        **trade_result
                    }
                    result.add_trade(trade)
                    daily_trade_count[day_key] += 1
                    print(f"    → {trade_result['exit_reason']} @ {trade_result['exit_price']:.2f} | P&L: {trade_result['pnl']:+.2f} USDT")
                    break

                # Bearish breakout
                elif c < or_low and not breakout_found:
                    entry_price = or_low * 0.9995
                    sl = or_mid
                    risk = sl - entry_price
                    tp = entry_price - (rr_ratio * risk)
                    side = "short"
                    breakout_found = True

                    print(f"  ✓ SHORT breakout on {day_key} @ {entry_price:.2f} (OR: {or_low:.2f}-{or_high:.2f})")

                    trade_result = simulate_trade(
                        entry=entry_price, sl=sl, tp=tp, side=side,
                        size_usdt=size_usdt, candles=candles,
                        entry_idx=idx, fee_rate=0.0005
                    )

                    trade = {
                        'symbol': symbol,
                        'entry_ts': int(ts),
                        'entry_date': day_key.strftime('%Y-%m-%d'),
                        'entry_price': entry_price,
                        'sl': sl, 'tp': tp, 'side': side,
                        'size_usdt': size_usdt,
                        'pattern': 'orb_short',
                        **trade_result
                    }
                    result.add_trade(trade)
                    daily_trade_count[day_key] += 1
                    print(f"    → {trade_result['exit_reason']} @ {trade_result['exit_price']:.2f} | P&L: {trade_result['pnl']:+.2f} USDT")
                    break

        print(f"  Total trades for {symbol}: {len([t for t in result.trades if t['symbol'] == symbol])}")

    return result


def backtest_fvg(ex, symbols: List[str], start: datetime, end: datetime,
                 risk_notional: float, risk_map: Dict[str, float]) -> BacktestResult:
    """Backtest NY Open FVG strategy by replaying historical data"""
    print("\n=== NY Open FVG Strategy Backtest ===")
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Symbols: {', '.join(symbols)}")

    result = BacktestResult()
    inspector = NyOpenFVGInspector()

    for symbol in symbols:
        print(f"\n--- Backtesting {symbol} ---")
        size_usdt = get_per_symbol_value(symbol, risk_map, risk_notional)
        print(f"Position size: {size_usdt} USDT")

        # Fetch data for FVG timeframe
        candles = fetch_historical_data(ex, symbol, inspector.fvg_timeframe, start, end)
        if len(candles) == 0:
            print(f"  No data available for {symbol}")
            continue

        print(f"  Processing {len(candles)} candles...")

        # Group candles by trading day
        from collections import defaultdict
        days_data = defaultdict(list)

        for idx, candle in enumerate(candles):
            ts = candle[0]
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            day_key = dt.date()
            days_data[day_key].append((idx, candle))

        print(f"  Found {len(days_data)} trading days")

        # Track trades per day
        max_trades_per_day = int(os.getenv("NY_OPEN_MAX_TRADES", "1") or 1)
        daily_trade_count = defaultdict(int)

        # Process each day
        for day_idx, day_key in enumerate(sorted(days_data.keys())):
            if daily_trade_count[day_key] >= max_trades_per_day:
                continue

            # Get historical context (include previous days for lookback)
            lookback_days = min(10, day_idx)
            hist_days = sorted(days_data.keys())[max(0, day_idx - lookback_days):day_idx + 1]

            historical = []
            for d in hist_days:
                historical.extend([c[1] for c in days_data[d]])

            if len(historical) < 50:
                continue

            # Create mock exchange that returns our historical data
            class HistoricalExchange:
                def __init__(self, data):
                    self.historical_data = data

                def fetch_ohlcv(self, sym, timeframe, limit):
                    return self.historical_data[-limit:] if len(self.historical_data) >= limit else self.historical_data

            mock_ex = HistoricalExchange(historical)

            # Run strategy analysis on this historical data
            try:
                payload = inspector.analyze_symbol(mock_ex, symbol, limit=len(historical))
                signal = payload.get('signal')
                or_ready = payload.get('or_ready')

                # DEBUG: Show status for first few days
                if day_idx < 5:
                    or_high = payload.get('or_high')
                    or_low = payload.get('or_low')
                    fvgs_found = payload.get('fvgs_found', 0)
                    print(f"  DEBUG {day_key}: OR ready={or_ready}, OR_H={or_high}, OR_L={or_low}, FVGs={fvgs_found}, signal={signal is not None}")

                if signal and or_ready:
                    if daily_trade_count[day_key] >= max_trades_per_day:
                        continue

                    entry_price = signal['entry']
                    sl = signal['sl']
                    tp = signal['tp']
                    side = signal['side']

                    print(f"  ✓ Signal on {day_key}: {side.upper()} @ {entry_price:.2f} (SL: {sl:.2f}, TP: {tp:.2f})")

                    # Find actual entry index
                    day_candles = days_data[day_key]
                    entry_idx = day_candles[0][0] if day_candles else 0

                    # Simulate the trade
                    trade_result = simulate_trade(
                        entry=entry_price,
                        sl=sl,
                        tp=tp,
                        side=side,
                        size_usdt=size_usdt,
                        candles=candles,
                        entry_idx=entry_idx,
                        fee_rate=0.0005
                    )

                    # Record trade
                    trade = {
                        'symbol': symbol,
                        'entry_ts': int(candles[entry_idx][0]),
                        'entry_date': day_key.strftime('%Y-%m-%d'),
                        'entry_price': entry_price,
                        'sl': sl,
                        'tp': tp,
                        'side': side,
                        'size_usdt': size_usdt,
                        'pattern': signal.get('pattern', 'fvg_rejection'),
                        **trade_result
                    }

                    result.add_trade(trade)
                    daily_trade_count[day_key] += 1

                    print(f"    → {trade_result['exit_reason']} @ {trade_result['exit_price']:.2f} | P&L: {trade_result['pnl']:+.2f} USDT")
                elif day_idx < 5:
                    if not or_ready:
                        print(f"  DEBUG {day_key}: OR not ready yet")
                    elif not signal:
                        print(f"  DEBUG {day_key}: OR ready but no FVG rejection signal")

            except Exception as e:
                if day_idx < 5:
                    print(f"  ERROR on {day_key}: {e}")
                    import traceback
                    traceback.print_exc()

        print(f"  Total trades for {symbol}: {len([t for t in result.trades if t['symbol'] == symbol])}")

    return result


def print_summary(result: BacktestResult):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)
    print(f"Total Trades:        {result.total_trades}")
    print(f"Winning Trades:      {result.winning_trades}")
    print(f"Losing Trades:       {result.losing_trades}")
    print(f"Win Rate:            {result.win_rate:.2f}%")
    print(f"")
    print(f"Total P&L:           {result.total_pnl:+.2f} USDT")
    print(f"Average Win:         {result.average_win:+.2f} USDT")
    print(f"Average Loss:        {result.average_loss:+.2f} USDT")
    print(f"Largest Win:         {result.largest_win:+.2f} USDT")
    print(f"Largest Loss:        {result.largest_loss:+.2f} USDT")
    print(f"Profit Factor:       {result.profit_factor:.2f}")
    print(f"")
    print(f"Max Consecutive Wins:   {result.max_consecutive_wins}")
    print(f"Max Consecutive Losses: {result.max_consecutive_losses}")
    print(f"Max Drawdown:           {result.max_drawdown:.2f} USDT")
    print("="*60)


def print_detailed_trades(result: BacktestResult):
    """Print detailed trade-by-trade log"""
    print("\n" + "="*80)
    print("DETAILED TRADE LOG")
    print("="*80)

    for i, trade in enumerate(result.trades, 1):
        exit_date = datetime.fromtimestamp(trade['exit_ts'] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nTrade #{i}")
        print(f"  Symbol:       {trade['symbol']}")
        print(f"  Side:         {trade['side'].upper()}")
        print(f"  Pattern:      {trade['pattern']}")
        print(f"  Entry:        {trade['entry_date']} @ {trade['entry_price']:.6f}")
        print(f"  Exit:         {exit_date} @ {trade['exit_price']:.6f}")
        print(f"  Exit Reason:  {trade['exit_reason']}")
        print(f"  SL:           {trade['sl']:.6f}")
        print(f"  TP:           {trade['tp']:.6f}")
        print(f"  Size:         {trade['size_usdt']:.2f} USDT")
        print(f"  Bars Held:    {trade['bars_held']}")
        print(f"  Gross P&L:    {trade['gross_pnl']:+.2f} USDT")
        print(f"  Fees:         -{trade['fees']:.2f} USDT")
        print(f"  Net P&L:      {trade['pnl']:+.2f} USDT")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--strategy', required=True, choices=['orb', 'fvg'],
                       help='Strategy to backtest (orb or fvg)')
    parser.add_argument('--period', help='Period to backtest (e.g., 2025-01, 2025, last_month, last_year)')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD) - alternative to --period')
    parser.add_argument('--end', help='End date (YYYY-MM-DD) - alternative to --period')
    parser.add_argument('--expand', action='store_true', 
                       help='Show detailed trade-by-trade log')

    args = parser.parse_args()

    # Load appropriate .env file
    if args.strategy == 'orb':
        env_file = '.env.orb'
        print(f"Loading configuration from {env_file}")
        load_dotenv(env_file, override=True)
        symbols_key = "ORB_SYMBOLS"
    else:
        env_file = '.env'
        print(f"Loading configuration from {env_file}")
        load_dotenv(env_file, override=True)
        symbols_key = "NY_OPEN_SYMBOLS"

    # Parse time period
    if args.start and args.end:
        start_date = parse_date(args.start)
        end_date = parse_date(args.end)
    elif args.period:
        start_date, end_date = parse_period(args.period)
    else:
        print("Error: Must specify either --period or both --start and --end")
        sys.exit(1)

    print(f"Backtest period: {start_date.date()} to {end_date.date()}")

    # Get symbols
    symbols_str = os.getenv(symbols_key, "BTC/USDT,ETH/USDT,SOL/USDT")
    symbols = [s.strip() for s in symbols_str.split(',')]

    # Get position sizing
    risk_notional = float(os.getenv("RISK_NOTIONAL_USDT", "300") or 300)
    risk_map = parse_map_env("RISK_NOTIONAL_MAP")
    risk_map = {k: float(v) for k, v in risk_map.items()}

    # Build exchange connection
    print("Connecting to Binance...")
    ex = build_exchange()

    # Run backtest
    if args.strategy == 'orb':
        result = backtest_orb(ex, symbols, start_date, end_date, risk_notional, risk_map)
    else:
        result = backtest_fvg(ex, symbols, start_date, end_date, risk_notional, risk_map)

    # Print results
    print_summary(result)

    if args.expand:
        print_detailed_trades(result)


if __name__ == "__main__":
    main()
