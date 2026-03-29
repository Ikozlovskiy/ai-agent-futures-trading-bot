# FVG Strategy Flows Documentation

## NY Open FVG Strategy Flow

### Configuration
- **Timeframe**: 1m candles for FVG detection
- **Check Interval**: 60 seconds
- **OR Period**: 13:30-13:45 UTC (15 minutes)
  - **Note**: All times are in UTC, not local time
  - Ukraine (EET/EEST): UTC+2 winter, UTC+3 summer
  - 13:30 UTC = 15:30 Kyiv (winter) or 16:30 Kyiv (summer)
  - Exchange data and strategy logic both use UTC
- **ATR Filter**: 0.5x ATR minimum gap size
- **RR Ratio**: 2.1:1

### Complete Trade Flow Example

#### Scenario: Bullish FVG Continuation LONG

**Timeline:**

**13:30:00 UTC** - Opening Range Starts
- Bot fetches last 500 x 1m candles
- Calculates OR period: 13:30-13:45 UTC
- OR not yet ready (needs to complete)

**13:31:00** - Bot checks (60s interval)
- OR still forming
- No trading signals yet
- Logs: "📊 NY-Open FVG ETH/USDT | OR: pending | Monitoring active"

**13:45:00** - OR Period Closes
- OR Complete: ORH=$3100, ORL=$3080 (range: $20)
- OR becomes "ready" for FVG detection
- Starts scanning for FVGs in window: 13:30-14:45 UTC (60min after OR)

**13:46:00** - Bot checks
- Fetches candles, **excludes last forming candle** (only uses CLOSED candles)
- Scans 13:30-13:45 range for FVG formation
- **Detects Bullish FVG:**
  - Candle 1 (13:42): High=$3095
  - Candle 2 (13:43): Middle candle
  - Candle 3 (13:44): Low=$3098 (gap: $3095-$3098, size=$3, ATR=$8)
  - Gap size $3 > 0.5*ATR ($4) ✅ Passes ATR filter
  - FVG stored: {i=13:44 index, side="long", lo=3095, hi=3098}
- **Monitors candles after 13:44 for entry:**
  - 13:45: C=$3102 (no wick in gap yet)
  - No entry signal yet
- Logs: "📊 NY-Open FVG ETH/USDT | OR ready | FVGs=1 | Last[long]: $3095-$3098 | Waiting for entry"

**13:47:00** - Bot checks (60s later)
- Fetches candles, excludes last forming candle
- Uses SAME FVG from 13:44 (most recent)
- **Monitors candles after FVG:**
  - 13:45: Close=$3102, High=$3105, Low=$3100 (no wick in gap)
  - 13:46: Close=$3099, High=$3102, **Low=$3096** ← **WICK ENTERS GAP ($3095-$3098)**
    - Candle enters FVG zone: Low $3096 overlaps gap $3095-$3098 ✅
    - Close $3099 > gap_hi $3098 ✅ **CONTINUATION LONG**
    - **ENTRY TRIGGERED**: Entry=$3099, SL=$3095, TP=$3099+2.1*(3099-3095)=$3107.40
  - Returns signal immediately (takes first valid entry)

- **Trade Execution:**
  - Checks: Weekday ✅, Daily max trades ✅, No open position ✅
  - Creates Decision: side=LONG, entry_type=market, size=$200, sl=$3095, tp=$3107.40
  - Executes market buy at $3099
  - Logs: "🎯 NY-Open FVG ETH/USDT [LONG] | 🔄 CONTINUATION | Entry: $3099 | SL: $3095 | TP: $3107.40"

**13:48:00** - Bot checks
- Has open position, skips new signals
- Monitors position for exit

**Trade Outcome:**
- TP hit at $3107.40 → +$8.40 gain (+0.27% on $3099)
- Risk was $4 (entry-SL), reward was $8.40
- Actual RR achieved: 2.1:1 ✅

---

#### Scenario: Bearish FVG Inversion LONG

**Timeline:**

**13:50:00** - New FVG detected
- **Bearish FVG formed:**
  - Candle 1 (13:47): Low=$3080
  - Candle 2 (13:48): Middle candle  
  - Candle 3 (13:49): High=$3077 (gap: $3077-$3080, size=$3)
  - Gap size $3 < 0.5*ATR ($4) ❌ **REJECTED by ATR filter**
  - FVG too small, ignored

**13:52:00** - Larger Bearish FVG detected
- **Bearish FVG formed:**
  - Candle 1 (13:49): Low=$3105
  - Candle 2 (13:50): Middle candle
  - Candle 3 (13:51): High=$3099 (gap: $3099-$3105, size=$6)
  - Gap size $6 > 0.5*ATR ($8 * 0.5 = $4) ✅ Passes ATR filter
  - FVG stored: {i=13:51 index, side="short", lo=3099, hi=3105}

**13:53:00** - Bot checks
- **Monitors candles after FVG:**
  - 13:52: Close=$3096, High=$3100, Low=$3095 (no wick in gap)
  - No entry yet

**13:54:00** - Bot checks
- **Monitors candles after FVG:**
  - 13:52: (no entry)
  - 13:53: Close=$3107, High=$3108, **Low=$3102** ← **WICK ENTERS GAP ($3099-$3105)**
    - Candle enters FVG zone: Low $3102 overlaps gap $3099-$3105 ✅
    - Close $3107 > gap_hi $3105 ✅ **INVERSION LONG** (price rejected down, closed above)
    - **ENTRY TRIGGERED**: Entry=$3107, SL=$3099, TP=$3107+2.1*(3107-3099)=$3123.80

- **Trade Execution:**
  - Checks: Weekday ✅, Daily max trades (1/2) ✅, No open position ✅
  - Executes market buy at $3107
  - Logs: "🎯 NY-Open FVG ETH/USDT [LONG] | 🔀 INVERSION | Entry: $3107 | SL: $3099 | TP: $3123.80"

---

### Critical Behavior Notes:

1. **Polling Safety:**
   - Bot checks every 60s but analyzes ALL closed 1m candles
   - Loop `for t in range(scan_start, len(c))` scans every candle after FVG
   - Even if check interval > candle timeframe, no entries are missed

2. **Entry Timing:**
   - Entry price = CLOSE of confirmation candle (market entry)
   - Not the wick touch price (we confirm AFTER candle closes)
   - This is correct - we trade confirmations, not predictions

3. **FVG Override:**
   - Always uses MOST RECENT FVG (last in list)
   - New FVG automatically overrides previous one
   - Only one active FVG monitored at a time

4. **Closed Candle Only:**
   - Excludes last forming candle from analysis
   - Prevents premature entries on incomplete candles
   - All signals are on CONFIRMED (closed) price action

---

## Multi-Confluence Scalper Flow

### Configuration
- **Timeframe**: 1m candles for FVG detection
- **Check Interval**: 10 seconds (more aggressive)
- **HTF**: 15m (trend filter)
- **MTF**: 5m (bias check)
- **ATR Filter**: 0.3x ATR minimum gap size (more permissive)
- **TPs**: 1.5:1, 2.5:1, 3.5:1 (50%, 30%, 20% closes)

### Complete Trade Flow Example

#### Scenario: FVG Continuation SHORT with Multi-Layer Confluence

**Timeline:**

**08:00:00 UTC** - Bot starts (London session - high priority)
- Time weight: 1.0x (high priority hour)
- Fetches HTF (15m), MTF (5m), LTF (1m) data
- No position yet

**08:00:10** - Bot checks (10s interval)
- **Layer 1: Market Structure**
  - HTF (15m): EMA50=$3100 < EMA200=$3150, Price=$3080 < EMA50 → "bearish" ✅
  - MTF (5m): Last 5 candles: 4 red, 1 green → 80% red → "bearish" ✅
  - Trade direction: BEARISH (aligned) ✅

- **Layer 2: FVG Detection (1m candles)**
  - Fetches last 100 x 1m candles
  - **Bearish FVG detected:**
    - Candle 1 (07:57): Low=$3085
    - Candle 2 (07:58): Middle candle
    - Candle 3 (07:59): High=$3080 (gap: $3080-$3085, size=$5)
    - ATR=$12, min gap=$12*0.3=$3.6
    - Gap $5 > $3.6 ✅ Passes ATR filter
    - FVG: {i=07:59 index, fvg_side="short", lo=3080, hi=3085}

  - **Monitors candles after FVG (from 08:00 onward):**
    - 08:00: Close=$3082, High=$3084, Low=$3081 (wick in gap $3080-$3085)
      - Wick enters FVG: High $3084 overlaps gap ✅
      - Close $3082 inside gap → No signal yet
    - No entry yet (close must be outside gap)

  - Logs: "❌ SOL/USDT | L2 FAIL: FVG found but candle closed inside gap"

**08:00:20** - Bot checks (10s later)
- **Layer 1:** Still bearish ✅
- **Layer 2: FVG Monitoring**
  - Uses SAME FVG from 07:59
  - **New candle analyzed:**
    - 08:00: (previous - no entry)
    - 08:01: Close=$3078, High=$3083, Low=$3077 (wick in gap)
      - Wick enters FVG: High $3083 overlaps gap $3080-$3085 ✅
      - Close $3078 < gap_lo $3080 ✅ **CONTINUATION SHORT**
      - Pattern: {i=08:01, side="short", strategy_type="continuation", lo=3080, hi=3085}

  - Logs: "✅ SOL/USDT | L2 PASS: 1 FVG pattern (continuation)"

- **Layer 4: Multi-Timeframe Alignment**
  - RSI (15m): 45 (not oversold <25) ✅
  - Entry candle quality (08:01): Body=($3083-$3078)/$6=83% > 35% ✅, closes in bottom 83% ✅
  - Logs: "✅ SOL/USDT | L4 PASS: RSI✓ Candle✓"

- **Calculate Entry:**
  - Entry: $3078 (current close)
  - SL: $3085 (gap_hi - FVG boundary) ← **NY Open style SL**
  - Risk: $7
  - TP1: $3078 - 1.5*$7 = $3067.50 (50% close)
  - TP2: $3078 - 2.5*$7 = $3060.50 (30% close)
  - TP3: $3078 - 3.5*$7 = $3053.50 (20% close)

  - SL distance: $7/$3078 = 0.23% < 0.8% max ✅

- **Trade Execution:**
  - Confidence: 0.75 * 1.0 (time weight) = 0.75
  - No open position ✅, daily trades 0/10 ✅
  - Executes market SHORT at $3078
  - Logs: "🎯 SCALP SIGNAL SOL/USDT [SHORT] | 📊 Pattern: FVG CONTINUATION 🔄 | Entry: $3078 | SL: $3085 | TP1: $3067.50 [50%]"

**08:01:30** - Position monitoring
- Price drops to $3067.50
- TP1 hit → Close 50% of position → +$10.50 profit on 50% ($5.25 net)
- SL moves to breakeven ($3078) for remaining 50%

**08:02:00** - Continued monitoring
- Price drops to $3060.50
- TP2 hit → Close 30% more → +$5.25 profit on 30% ($1.58 net)
- Total profit so far: $6.83
- 20% position remains with SL at breakeven

**Trade Outcome:**
- TP2 hit, remaining 20% stopped at breakeven
- Total profit: $6.83 on $200 position
- ROI: ~3.4% (with 20x leverage → ~68% ROE)

---

### Critical Behavior Notes:

1. **10s Check Interval with 1m Candles:**
   - Checks 6x per candle (every 10s)
   - First 5 checks see forming candle (excluded from analysis)
   - 6th check (at :60s) sees newly closed candle
   - Scans ALL closed candles in FVG monitoring loop
   - **No entries missed** even with sub-candle polling

2. **Layer Filtering:**
   - Layer 1 (HTF/MTF) filters direction ← Must pass first
   - Layer 2 (FVG) detects pattern ← Core signal
   - Layer 3 (Volume/Spread) **DISABLED** in current config
   - Layer 4 (RSI/Quality) final filter ← Prevents extremes

3. **FVG Detection:**
   - Same logic as NY Open (ATR-based, continuation/inversion)
   - More permissive filter (0.3x ATR vs 0.5x)
   - SL at gap boundary (NY Open style)
   - Multiple TPs for risk management

4. **Time-of-Day Weight:**
   - High priority (08:00-17:00 UTC): 1.0x confidence
   - Medium (00:00-08:00, 17:00-20:00): 0.8x confidence
   - Low (20:00-00:00): 0.6x confidence
   - Affects final confidence score only

---

## Common Issues & Edge Cases

### 1. **Missed Wicks Between Polls?**
❌ **NO** - Both strategies scan ALL closed candles in monitoring loop
- Even if bot checks every 60s, it analyzes every 1m candle that closed
- Loop: `for t in range(scan_start, len(c))` ensures complete scan

### 2. **Entry on Forming Candle?**
❌ **NO** - Both strategies explicitly exclude last forming candle
- `ts, o, h, l, c, v = ts[:-1], o[:-1], h[:-1], l[:-1], c[:-1], v[:-1]`
- Only CLOSED candles are analyzed

### 3. **Entry Price vs Wick Touch Price?**
✅ **CORRECT** - Entry = CLOSE of confirmation candle (market entry)
- Not trying to catch exact wick touch
- Trading confirmation AFTER candle closes
- More reliable than trying to catch intra-candle moves

### 4. **Multiple FVGs Active?**
❌ **NO** - Both strategies use MOST RECENT FVG only
- `fvg = fvgs[-1]` (NY Open)
- `detected_fvgs[-1]` (Scalper)
- New FVG automatically overrides previous

### 5. **Scalper: Layer 3 Disabled - Is This Safe?**
✅ **YES** - Layer 3 (volume/spread) adds quality but reduces frequency
- Current config prioritizes more signals with Layer 1+2+4
- Can re-enable Layer 3 if too many false signals occur

### 6. **Both Continuation AND Inversion Triggered?**
❌ **NO** - Mutually exclusive conditions
- Continuation: Close OUTSIDE gap in FVG direction
- Inversion: Close OUTSIDE gap in OPPOSITE direction
- Candle can only close in one location
- First valid signal is taken (break loop after entry)

---

## Recommendations

### NY Open FVG:
- ✅ Implementation is correct
- ✅ Polling interval (60s) is safe with 1m candles
- ✅ Entry logic (continuation/inversion) is sound
- ✅ SL/TP at gap boundaries with RR is optimal
- ⚠️ Consider adding max position time limit (e.g., 4 hours)

### Multi-Confluence Scalper:
- ✅ Implementation is correct
- ✅ 10s polling is safe with 1m candles
- ✅ Layer 1+2+4 filtering is sufficient
- ✅ Multiple TPs with partial closes reduces risk
- ⚠️ Consider re-enabling Layer 3 (volume) if signals are too noisy
- ⚠️ Monitor ATR filter (0.3x) - may be too permissive in low volatility

---

**Last Updated**: 2026-03-29
**Strategies Version**: NY Open FVG v2.0, Multi-Confluence Scalper v2.0
