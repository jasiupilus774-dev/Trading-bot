import os
import math
import ccxt
import pandas as pd
import ta
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# =========================
# CONFIG
# =========================
PAIRS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
DAYS = 180

INITIAL_CASH = 1000.0
RISK_PER_TRADE = 0.01

FEE_RATE = 0.001
SLIPPAGE = 0.0002

EMA_TREND = 200
RSI_LEN = 14
BB_LEN = 20
BB_STD = 2
ATR_LEN = 14

SL_ATR = 1.0
TP_ATR = 2.0

# =========================
@dataclass
class Trade:
    side: str
    entry: float
    exit: float
    qty: float
    pnl: float

# =========================
def fetch_data(pair):
    ex = ccxt.binance({"enableRateLimit": True})
    since = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)
    rows = ex.fetch_ohlcv(pair, timeframe=TIMEFRAME, since=since)
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def add_indicators(df):
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=EMA_TREND).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=RSI_LEN).rsi()

    bb = ta.volatility.BollingerBands(df["close"], window=BB_LEN, window_dev=BB_STD)
    df["bb_low"] = bb.bollinger_lband()
    df["bb_high"] = bb.bollinger_hband()

    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=ATR_LEN
    ).average_true_range()

    return df

# =========================
def run_backtest(df, pair):
    cash = INITIAL_CASH
    peak = INITIAL_CASH
    maxDD = 0

    in_pos = False
    side = None
    entry = qty = sl = tp = None

    trades = []
    wins = losses = 0
    gross_profit = gross_loss = 0

    for i in range(2, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]

        if any(pd.isna(prev[x]) for x in ["ema200","rsi","bb_low","bb_high","atr"]):
            continue

        # EXIT
        if in_pos:
            exit_price = None

            if side == "LONG":
                if curr["low"] <= sl:
                    exit_price = sl
                elif curr["high"] >= tp:
                    exit_price = tp

            elif side == "SHORT":
                if curr["high"] >= sl:
                    exit_price = sl
                elif curr["low"] <= tp:
                    exit_price = tp

            if exit_price:
                real_exit = exit_price * (1 - SLIPPAGE if side=="LONG" else 1 + SLIPPAGE)

                if side == "LONG":
                    gross = (real_exit - entry) * qty
                else:
                    gross = (entry - real_exit) * qty

                fees = (entry * qty + real_exit * qty) * FEE_RATE
                pnl = gross - fees

                cash += pnl
                peak = max(peak, cash)
                dd = (peak - cash)/peak
                maxDD = max(maxDD, dd)

                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    losses += 1
                    gross_loss += pnl

                trades.append(Trade(side, entry, real_exit, qty, pnl))
                in_pos = False

        # ENTRY
        if not in_pos:
            trend_up = prev["close"] > prev["ema200"]
            trend_down = prev["close"] < prev["ema200"]

            long_signal = (
                trend_up and
                prev["rsi"] < 30 and
                prev["close"] <= prev["bb_low"]
            )

            short_signal = (
                trend_down and
                prev["rsi"] > 70 and
                prev["close"] >= prev["bb_high"]
            )

            if long_signal or short_signal:
                side = "LONG" if long_signal else "SHORT"
                entry = curr["open"] * (1 + SLIPPAGE if side=="LONG" else 1 - SLIPPAGE)

                atr = prev["atr"]
                sl_dist = atr * SL_ATR
                tp_dist = atr * TP_ATR

                if side == "LONG":
                    sl = entry - sl_dist
                    tp = entry + tp_dist
                else:
                    sl = entry + sl_dist
                    tp = entry - tp_dist

                risk = cash * RISK_PER_TRADE
                qty = risk / sl_dist
                in_pos = True

    total_trades = wins + losses
    win_rate = (wins/total_trades*100) if total_trades else 0
    pf = (gross_profit/abs(gross_loss)) if gross_loss!=0 else 0
    expectancy = (cash-INITIAL_CASH)/total_trades if total_trades else 0

    print(f"\nPODSUMOWANIE {pair}")
    print(f"Trades: {total_trades}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Net PnL: {cash-INITIAL_CASH:.2f} USDT")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Expectancy: {expectancy:.4f}")
    print(f"MaxDD: {maxDD*100:.2f}%")

    return trades

# =========================
def main():
    print("\n--- BACKTEST LONG + SHORT ---")
    for pair in PAIRS:
        df = fetch_data(pair)
        df = add_indicators(df)
        run_backtest(df, pair)
    print("\n--- KONIEC ---")

if __name__ == "__main__":
    main()
