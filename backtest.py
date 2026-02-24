# backtest.py
# Trend Following LONG-only (SPOT): EMA200 + Breakout + ATR SL/TP
# Zapisuje transakcje do trades_<PAIR>.csv + podsumowanie łączne
#
# Wymagania: pip install ccxt pandas ta
#
# ENV (opcjonalnie):
#   PAIRS="BTC/USDT,ETH/USDT"
#   TIMEFRAME="1h"
#   DAYS="180"
#   TRADE_USDT="50"
#   EMA_LEN="200"
#   BREAKOUT_LOOKBACK="20"
#   ATR_LEN="14"
#   SL_ATR="1.0"
#   TP_ATR="1.5"
#   FEE_RATE="0.001"
#   SLIPPAGE_RATE="0.0002"

import os
import time
import csv
from datetime import datetime, timezone

import ccxt
import pandas as pd
import ta

# =========================
# CONFIG
# =========================
PAIRS = [p.strip() for p in os.getenv("PAIRS", "BTC/USDT,ETH/USDT").split(",") if p.strip()]
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
DAYS = int(os.getenv("DAYS", "180"))
TRADE_USDT = float(os.getenv("TRADE_USDT", os.getenv("TRADE_AMOUNT_USDT", "50")))

EMA_LEN = int(os.getenv("EMA_LEN", "200"))
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "20"))

ATR_LEN = int(os.getenv("ATR_LEN", "14"))
SL_ATR = float(os.getenv("SL_ATR", "1.0"))
TP_ATR = float(os.getenv("TP_ATR", "1.5"))

FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))            # 0.1%
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0002")) # 0.02%

MAX_OHLCV_LIMIT = 1000


# =========================
# HELPERS
# =========================
def timeframe_to_ms(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60_000
    if tf.endswith("h"):
        return int(tf[:-1]) * 3_600_000
    if tf.endswith("d"):
        return int(tf[:-1]) * 86_400_000
    raise ValueError(f"Nieobsługiwany timeframe: {tf}")


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def pair_to_filename(pair: str) -> str:
    return pair.replace("/", "_")


# =========================
# EXCHANGE (public market data)
# =========================
def connect_exchange() -> ccxt.Exchange:
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })


def fetch_ohlcv_all(ex: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    tf_ms = timeframe_to_ms(timeframe)
    data = []
    since = since_ms

    max_iters = 5000
    it = 0

    while True:
        it += 1
        if it > max_iters:
            break

        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=MAX_OHLCV_LIMIT)
        if not batch:
            break

        data.extend(batch)
        last_ts = batch[-1][0]
        since = last_ts + tf_ms

        if since >= now_ms() - tf_ms:
            break

        time.sleep(ex.rateLimit / 1000)

    if not data:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


# =========================
# INDICATORS
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=EMA_LEN).ema_indicator()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_LEN)
    df["atr"] = atr.average_true_range()

    # breakout level: max high z poprzednich N świec (bez bieżącej)
    df["hh"] = df["high"].shift(1).rolling(BREAKOUT_LOOKBACK).max()

    return df


# =========================
# BACKTEST
# =========================
def save_trades_csv(pair: str, trades: list) -> str:
    filename = f"trades_{pair_to_filename(pair)}.csv"
    fields = ["ts", "pair", "side", "price", "qty", "sl", "tp", "fee", "gross", "pnl", "equity", "note"]

    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in trades:
            # ensure all keys
            for k in fields:
                row.setdefault(k, "")
            w.writerow(row)

    return filename


def backtest_pair(df: pd.DataFrame, pair: str) -> dict:
    if df.empty or len(df) < (EMA_LEN + ATR_LEN + BREAKOUT_LOOKBACK + 5):
        return {
            "pair": pair,
            "round_trips": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "maxDD": 0.0,
            "trades": [],
        }

    df = add_indicators(df).dropna().reset_index(drop=True)
    if df.empty or len(df) < 5:
        return {
            "pair": pair,
            "round_trips": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "maxDD": 0.0,
            "trades": [],
        }

    in_pos = False
    entry_price = 0.0
    qty = 0.0
    sl = 0.0
    tp = 0.0
    entry_fee = 0.0
    entry_ts = None

    trades = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    net_pnl = 0.0
    round_trips = 0

    for i in range(len(df)):
        row = df.iloc[i]
        ts = int(row["ts"])
        close = float(row["close"])
        ema200 = float(row["ema200"])
        atr = float(row["atr"])
        hh = float(row["hh"])

        # Filtr trendu: tylko LONG gdy close > EMA200
        uptrend = close > ema200

        # Wejście: breakout
        buy_sig = uptrend and (close > hh)

        # Jeśli w pozycji: sprawdzamy SL/TP na podstawie close (konserwatywnie)
        if in_pos:
            # exit reason
            reason = None
            exit_price_raw = close

            if close <= sl:
                reason = "SL"
                exit_price_raw = sl  # konserwatywnie egzekucja na SL
            elif close >= tp:
                reason = "TP"
                exit_price_raw = tp  # konserwatywnie egzekucja na TP

            if reason is not None:
                # sprzedajemy gorzej (slippage w dół)
                exit_price = exit_price_raw * (1.0 - SLIPPAGE_RATE)
                exit_value = exit_price * qty
                exit_fee = exit_value * FEE_RATE

                gross = (exit_price - entry_price) * qty
                fees = entry_fee + exit_fee
                pnl = gross - fees

                net_pnl += pnl
                equity += pnl

                peak = max(peak, equity)
                dd = equity - peak
                max_dd = min(max_dd, dd)

                if pnl >= 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    losses += 1
                    gross_loss += pnl  # ujemne

                round_trips += 1

                trades.append({
                    "ts": iso_utc(ts),
                    "pair": pair,
                    "side": f"SELL_{reason}",
                    "price": exit_price,
                    "qty": qty,
                    "sl": sl,
                    "tp": tp,
                    "fee": exit_fee,
                    "gross": gross,
                    "pnl": pnl,
                    "equity": equity,
                    "note": f"entry={iso_utc(entry_ts)}",
                })

                # reset
                in_pos = False
                entry_price = 0.0
                qty = 0.0
                sl = 0.0
                tp = 0.0
                entry_fee = 0.0
                entry_ts = None

                continue  # idziemy do następnej świecy

        # Entry tylko jeśli nie jesteśmy w pozycji
        if (not in_pos) and buy_sig:
            # kupujemy gorzej (slippage w górę)
            entry_price = close * (1.0 + SLIPPAGE_RATE)
            qty = TRADE_USDT / entry_price

            # SL/TP o ATR (od entry)
            sl = entry_price - (atr * SL_ATR)
            tp = entry_price + (atr * TP_ATR)

            entry_value = entry_price * qty
            entry_fee = entry_value * FEE_RATE
            entry_ts = ts

            trades.append({
                "ts": iso_utc(ts),
                "pair": pair,
                "side": "BUY",
                "price": entry_price,
                "qty": qty,
                "sl": sl,
                "tp": tp,
                "fee": entry_fee,
                "gross": "",
                "pnl": "",
                "equity": equity,
                "note": f"EMA{EMA_LEN} breakout{BREAKOUT_LOOKBACK} ATR{ATR_LEN}",
            })

            in_pos = True

    total_trades = wins + losses
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0.0
    expectancy = net_pnl / total_trades if total_trades > 0 else 0.0

    return {
        "pair": pair,
        "round_trips": round_trips,
        "wins": wins,
        "losses": losses,
        "net_pnl": float(net_pnl),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),  # ujemne
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "maxDD": float(max_dd),
        "trades": trades,
    }


def main():
    print(f"BACKTEST: timeframe={TIMEFRAME} days={DAYS}")
    print(f"Strategy: LONG-only EMA{EMA_LEN} + Breakout({BREAKOUT_LOOKBACK}) + ATR SL/TP")
    print(f"ATR: len={ATR_LEN} SL_ATR={SL_ATR} TP_ATR={TP_ATR}")
    print(f"Costs: fee={FEE_RATE} slippage={SLIPPAGE_RATE} | trade_usdt={TRADE_USDT}\n")

    ex = connect_exchange()
    since_ms = now_ms() - DAYS * 86_400_000

    results = []

    for pair in PAIRS:
        try:
            df = fetch_ohlcv_all(ex, pair, TIMEFRAME, since_ms)
            res = backtest_pair(df, pair)

            out = save_trades_csv(pair, res["trades"])
            trades_n = res["wins"] + res["losses"]
            win_rate = (res["wins"] / trades_n * 100) if trades_n > 0 else 0.0

            print(
                f"{pair}: round_trips={res['round_trips']} trades={trades_n} "
                f"win_rate={win_rate:.2f}% net_pnl={res['net_pnl']:.4f} "
                f"PF={res['profit_factor']:.3f} exp={res['expectancy']:.4f} maxDD={res['maxDD']:.4f}"
            )
            print(f"-> zapisano {out}\n")

            results.append(res)
        except Exception as e:
            print(f"❌ {pair}: {e}\n")

    # =========================
    # PODSUMOWANIE ŁĄCZNE
    # =========================
    if results:
        total_pnl = sum(r.get("net_pnl", 0) for r in results)
        total_round_trips = sum(r.get("round_trips", 0) for r in results)

        total_wins = sum(r.get("wins", 0) for r in results)
        total_losses = sum(r.get("losses", 0) for r in results)
        total_trades = total_wins + total_losses

        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        gross_profit = sum(r.get("gross_profit", 0) for r in results)
        gross_loss = sum(r.get("gross_loss", 0) for r in results)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

        expectancy = total_pnl / total_trades if total_trades > 0 else 0

        print("\n==============================")
        print("BACKTEST SUMMARY")
        print("==============================")
        print(f"Round trips: {total_round_trips}")
        print(f"Trades:      {total_trades}")
        print(f"Win rate:    {win_rate:.2f}%")
        print(f"Net PnL:     {total_pnl:.4f} USDT")
        print(f"ProfitFact:  {profit_factor:.3f}")
        print(f"Expectancy:  {expectancy:.4f} USDT/trade")
        print("==============================\n")
    else:
        print("Brak wyników (results puste).")


if __name__ == "__main__":
    main()
