import os
import math
from datetime import datetime, timezone
import pandas as pd
import ccxt
import ta

# =========================
# USTAWIENIA BACKTESTU
# =========================
PAIRS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"

# Okres w dniach do pobrania (1h -> ~24 świece/dzień)
DAYS = int(os.getenv("BT_DAYS", "180"))  # np. 180 dni
LIMIT_PER_CALL = 1000  # Binance zwykle 1000 świec na call

TRADE_AMOUNT_USDT = float(os.getenv("TRADE_AMOUNT_USDT", "50"))

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_BUY = float(os.getenv("RSI_BUY", "45"))   # testowo lub docelowo 35
RSI_SELL = float(os.getenv("RSI_SELL", "55")) # testowo lub docelowo 65

MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))

FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))         # 0.1%
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0002"))  # 0.02%

# =========================
# DANE
# =========================
def timeframe_to_ms(tf: str) -> int:
    # prosty mapper dla popularnych TF
    m = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
         "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
         "1d": 86400}
    if tf not in m:
        raise ValueError(f"Nieobsługiwany timeframe: {tf}")
    return m[tf] * 1000

def fetch_ohlcv_full(exchange, symbol, timeframe, since_ms, until_ms):
    all_rows = []
    tf_ms = timeframe_to_ms(timeframe)
    since = since_ms

    while since < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=LIMIT_PER_CALL)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        # zabezpieczenie przed zapętleniem
        if last_ts == since:
            break
        since = last_ts + tf_ms

        # jeśli już za daleko, kończ
        if last_ts >= until_ms - tf_ms:
            break

    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=RSI_PERIOD).rsi()

    macd = ta.trend.MACD(
        df["close"],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    return df

def signal_at_row(df: pd.DataFrame, i: int) -> str:
    # sygnał liczony na zamknięciu świecy i (używa i-1 i i)
    if i < 2:
        return "HOLD"
    last = df.iloc[i]
    prev = df.iloc[i-1]

    rsi = last["rsi"]
    cross_up = prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]
    cross_down = prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]

    if pd.isna(rsi) or pd.isna(prev["macd"]) or pd.isna(prev["macd_signal"]) or pd.isna(last["macd"]) or pd.isna(last["macd_signal"]):
        return "HOLD"

    if rsi < RSI_BUY and cross_up:
        return "BUY"
    if rsi > RSI_SELL and cross_down:
        return "SELL"
    return "HOLD"


# =========================
# BACKTEST
# =========================
def backtest_symbol(df: pd.DataFrame, symbol: str) -> dict:
    """
    Założenia:
    - 1 pozycja max (long only), BUY otwiera, SELL zamyka
    - transakcja wykonana na OPEN następnej świecy po sygnale
    - kwota stała w USDT na wejście
    - fee liczone na wejściu i wyjściu
    - slippage jako % ceny
    """
    in_pos = False
    entry_price = None
    qty = 0.0

    trades = []
    equity = [0.0]  # krzywa PnL skumulowanego
    cum_pnl = 0.0

    # iterujemy po świecach i generujemy sygnał na close świecy i,
    # a wykonujemy na open świecy i+1 -> dlatego stop na len-2
    for i in range(2, len(df)-1):
        sig = signal_at_row(df, i)
        next_open = float(df.iloc[i+1]["open"])
        ts = int(df.iloc[i+1]["timestamp"])

        if sig == "BUY" and not in_pos:
            fill = next_open * (1 + SLIPPAGE_RATE)
            qty = TRADE_AMOUNT_USDT / fill
            fee_in = (fill * qty) * FEE_RATE

            in_pos = True
            entry_price = fill

            trades.append({
                "ts": ts,
                "symbol": symbol,
                "side": "BUY",
                "price": fill,
                "qty": qty,
                "fee": fee_in,
                "pnl": 0.0
            })

        elif sig == "SELL" and in_pos:
            fill = next_open * (1 - SLIPPAGE_RATE)
            fee_out = (fill * qty) * FEE_RATE

            gross = (fill - entry_price) * qty
            pnl = gross - (fee_out + (entry_price * qty * FEE_RATE))  # fee in + fee out

            cum_pnl += pnl
            equity.append(cum_pnl)

            trades.append({
                "ts": ts,
                "symbol": symbol,
                "side": "SELL",
                "price": fill,
                "qty": qty,
                "fee": fee_out,
                "pnl": pnl
            })

            in_pos = False
            entry_price = None
            qty = 0.0

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {
            "symbol": symbol,
            "trades": 0,
            "round_trips": 0,
            "net_pnl": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "expectancy": None,
            "max_drawdown": None,
            "trades_df": trades_df
        }

    # round-trips = liczba SELL
    sells = trades_df[trades_df["side"] == "SELL"].copy()
    round_trips = len(sells)

    net_pnl = float(sells["pnl"].sum())
    wins = sells[sells["pnl"] > 0]
    losses = sells[sells["pnl"] < 0]

    win_rate = (len(wins) / round_trips) if round_trips else None
    gross_profit = float(wins["pnl"].sum())
    gross_loss = float(losses["pnl"].sum())  # ujemne
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else (math.inf if gross_profit > 0 else None)

    avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
    avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0  # ujemne
    expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) if win_rate is not None else None

    # max drawdown na krzywej equity
    eq = pd.Series(equity)
    roll_max = eq.cummax()
    dd = eq - roll_max
    max_dd = float(dd.min())  # ujemne

    # dodaj czytelny czas
    def ts_to_str(x):
        return datetime.fromtimestamp(int(x)/1000, tz=timezone.utc).isoformat()

    trades_df["time_utc"] = trades_df["ts"].apply(ts_to_str)

    return {
        "symbol": symbol,
        "trades": len(trades_df),
        "round_trips": round_trips,
        "net_pnl": net_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_dd,
        "trades_df": trades_df
    }

def main():
    exchange = ccxt.binance({"enableRateLimit": True})
    exchange.load_markets()

    now_ms = exchange.milliseconds()
    since_ms = now_ms - DAYS * 24 * 60 * 60 * 1000

    print(f"\nBACKTEST: timeframe={TIMEFRAME} days={DAYS}")
    print(f"Params: RSI({RSI_PERIOD}) BUY<{RSI_BUY} SELL>{RSI_SELL} | MACD {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}")
    print(f"Costs: fee={FEE_RATE} slippage={SLIPPAGE_RATE} | trade_usdt={TRADE_AMOUNT_USDT}\n")

    results = []
    for sym in PAIRS:
        df = fetch_ohlcv_full(exchange, sym, TIMEFRAME, since_ms, now_ms)
        if df.empty or len(df) < 100:
            print(f"{sym}: za mało danych ({len(df)})")
            continue
        df = add_indicators(df)
        res = backtest_symbol(df, sym)
        results.append(res)

        print(f"{sym}: round_trips={res['round_trips']} net_pnl={res['net_pnl']:.4f} "
              f"win_rate={None if res['win_rate'] is None else round(res['win_rate']*100,2)}% "
              f"PF={res['profit_factor']} "
              f"exp={res['expectancy']} "
              f"maxDD={res['max_drawdown']:.4f}")

        # zapis transakcji do CSV (osobno dla symbolu)
        out_csv = f"trades_{sym.replace('/','_')}.csv"
        res["trades_df"].to_csv(out_csv, index=False)
        print(f"  -> zapisano {out_csv}")

    # podsumowanie łączne
    if results:
        total_pnl = sum(r["net_pnl"] for r in results)
        total_round_trips = sum(r["round_trips"] for r in results)
        print(f"\nTOTAL: round_trips={total_round_trips} net_pnl={total_pnl:.4f}\n")

if __name__ == "__main__":
    main()
