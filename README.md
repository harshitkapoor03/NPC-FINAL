

---

```markdown
# SmartTrendVolatilityMaker Strategy for Hummingbot

A custom market-making strategy for [Hummingbot](https://hummingbot.org), built from scratch using trend analysis,
volatility-based order sizing, fractal retracement detection, and inventory risk management.

---

## 📈 What It Does

This strategy dynamically places limit buy and sell orders based on:

- 📊 **Trend detection** via MACD, RSI, and Bollinger Bands
- 🌪️ **Volatility analysis** using NATR and BB width
- 🔄 **Fibonacci-based retracement detection** using fractals + trend alignment
- ⚖️ **Inventory balancing**: adjusts aggressiveness based on SOL/FDUSD holdings
- 🚫 **Risk controls**: throttles trading during tight spreads or flat volatility

---

## 🛠️ Technical Highlights

| Component | Description |
|----------|-------------|
| **Trend Detection** | Weighted score using MACD, RSI, BB position, and MACD crossovers |
| **Volatility Filter** | Uses NATR and BB Width to determine high/low market volatility |
| **Fractal Fib Retracement** | Identifies local high/low fractals and checks price proximity to Fib levels (0.382, 0.5, 0.618) |
| **Inventory Management** | Adjusts or disables buy/sell orders based on portfolio SOL% |
| **Order Pricing Logic** | Applies adaptive spread based on trend, volatility, and retracement |
| **Order Skipping** | Avoids placing orders during extreme imbalance or insufficient spread |

---

## 📂 Files

- `updated_submission_npc.py` → The custom strategy script
- `conf_submissionupdated_1.yml` → The config file used for strategy parameters

---

## ▶️ How to Run

### Prerequisites
- Hummingbot installed (see: [Install Hummingbot](https://docs.hummingbot.org/installation/))
- Python 3.8+ environment
- Pandas TA installed:
  ```bash
  pip install pandas_ta
  ```

### 1. Copy Files
Place both files in your `hummingbot/scripts` directory:
```
hummingbot/
├── scripts/
│   ├── updated_submission_npc.py
├── conf/
│   ├── conf_submissionupdated_1.yml
```

### 2. Rename Script (Optional)
If needed, rename the script to match your config:
```bash
mv updated_submission_npc.py tope3.py
```

Or update the `script_file_name:` field in your `.yml` config accordingly.

---

### 3. Launch Strategy

From the Hummingbot CLI:

```bash
script start conf_submissionupdated_1.yml
```

---

## 📋 Parameters in Config

```yaml
exchange: binance_paper_trade
trading_pair: SOL-FDUSD
order_amount: 0.1
order_refresh_time: 45
candles_interval: 1m
candles_max_records: 1000
```

---

## 🧠 Strategy Logic In Brief

- **Trend** is detected using MACD crossovers, BB positioning, and RSI scoring.
- **Volatility** determines how aggressive the spread and order size should be.
- **Retracements** are confirmed only when:
  - A fractal high/low is recent (within 20 candles)
  - It’s in the same trend direction as the current one
  - The price is near a Fib level (0.382, 0.5, or 0.618)
- **Orders** are placed only if:
  - Spread is wide enough (> 0.2%)
  - Volatility isn't dead-flat
  - Inventory allocation is within bounds

---

## ✅ Why This Project Matters

This strategy reflects strong understanding of:

- Real-world market dynamics
- Signal combination and scoring
- Risk and portfolio exposure management
- Pythonic, modular code structure for live trading

---

## 📮 Contact

Feel free to reach out if you'd like help deploying it, understanding the logic, or expanding it for a live exchange!

```
