# backtest.py
# Backtest RSI + MACD (spot) dla wielu par, z kosztami (fee + slippage),
# zapisuje szczegółowe transakcje do trades_<PAIR>.csv i drukuje statystyki per para + łączne.
#
# Wymagania: pip install ccxt pandas ta
#
# Uruchomienie lokalnie:
#   python backtest.py
#
# Zmienne ENV (opcjonalne):
#   PAIRS="BTC/USDT,ETH/USDT"
#   TIMEFRAME="1h"
#   DAYS="180"
#   TRADE_USDT="50"
#   RSI_PERIOD="14"
#   RSI_BUY="45"
#   RSI_SELL="55"
#   MACD_FAST="12" MACD_SLOW="26" MACD_SIGNAL="9"
#   FEE_RATE="0.001"
#   SLIPPAGE_RATE="0.0002"

import os
import time
import math
import csv
from datetime import datetime, timezone

import ccxt
import pandas as pd
import ta


# =========================
# CONFIG (ENV -> default)
# =========================
PAIRS = [p.strip() for p in os.getenv("PAIRS", "BTC/USDT,ETH/USDT").split(",") if p.strip()]
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
DAYS = int(os.getenv("DAYS", "180"))
TRADE_USDT = float(os.getenv("TRADE_USDT", os.getenv("TRADE_AMOUNT_USDT", "50")))

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_BUY = float(os.getenv("RSI_BUY", "45"))
RSI_SELL = float(os.getenv("RSI_SELL", "55"))

MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))

FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))            # 0.1%
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0002")) # 0.02%

# bezpieczeństwo: na testy nie ustawiaj ogromnych wartości
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
# EXCHANGE (public data)
# =========================
def connect_exchange() -> ccxt.Exchange:
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    return ex


def fetch_ohlcv_all(ex: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """
    Pobiera OHLCV od since_ms do teraz, paginując.
    Binance limit 1000 na fetch; ccxt obsługuje to.
    """
    tf_ms = timeframe_to_ms(timeframe)
    data = []
    since = since_ms

    # zabezpieczenie przed pętlą
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
        # następne okno (unikamy duplikatów)
        since = last_ts + tf_ms

        # jeśli jesteśmy blisko teraz, kończymy
        if since >= now_ms() - tf_ms:
            break

        # rate limit
        time.sleep(ex.rateLimit / 1000)

    if not data:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=RSI_PERIOD).rsi()

    macd = ta.trend.MACD(
        df["close"],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL,
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    return df


# =========================
# STRATEGY + BACKTEST
# =========================
def macd_cross_up(prev_macd, prev_sig, macd_val, sig_val) -> bool:
    return (prev_macd < prev_sig) and (macd_val > sig_val)


def macd_cross_down(prev_macd, prev_sig, macd_val, sig_val) -> bool:
    return (prev_macd > prev_sig) and (macd_val < sig_val)


def backtest_pair(df: pd.DataFrame, pair: str) -> dict:
    """
    Long-only: BUY gdy RSI < RSI_BUY i MACD cross up
              SELL gdy RSI > RSI_SELL i MACD cross down
    Wejście/wyjście na close świecy, z poślizgiem i prowizją.
    Stała wielkość pozycji: TRADE_USDT / entry_price.
    """
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

    df = add_indicators(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < 5:
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
    entry_price = None
    qty = 0.0
    entry_fees = 0.0
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

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]

        price = float(cur["close"])
        rsi = float(cur["rsi"])
        prev_macd = float(prev["macd"])
        prev_sig = float(prev["macd_signal"])
        macd_val = float(cur["macd"])
        sig_val = float(cur["macd_signal"])

        buy_sig = (rsi < RSI_BUY) and macd_cross_up(prev_macd, prev_sig, macd_val, sig_val)
        sell_sig = (rsi > RSI_SELL) and macd_cross_down(prev_macd, prev_sig, macd_val, sig_val)

        ts = int(cur["ts"])

        # ENTRY
        if (not in_pos) and buy_sig:
            # kupujemy gorzej (slippage w górę)
            entry_price = price * (1.0 + SLIPPAGE_RATE)
            qty = TRADE_USDT / entry_price
            # fee od wartości transakcji
            entry_value = entry_price * qty
            entry_fees = entry_value * FEE_RATE
            entry_ts = ts

            trades.append({
                "ts": iso_utc(ts),
                "pair": pair,
                "side": "BUY",
                "price": entry_price,
                "qty": qty,
                "rsi": rsi,
                "macd": macd_val,
                "macd_signal": sig_val,
                "fee": entry_fees,
                "pnl": "",
                "equity": equity,
                "note": "RSI+MACD",
            })
            in_pos = True
            continue

        # EXIT
        if in_pos and sell_sig:
            # sprzedajemy gorzej (slippage w dół)
            exit_price = price * (1.0 - SLIPPAGE_RATE)
            exit_value = exit_price * qty
            exit_fee = exit_value * FEE_RATE

            gross = (exit_price - entry_price) * qty
            fees = entry_fees + exit_fee
            pnl = gross - fees

            net_pnl += pnl
            equity += pnl

            # drawdown
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
                "side": "SELL",
                "price": exit_price,
                "qty": qty,
                "rsi": rsi,
                "macd": macd_val,
                "macd_signal": sig_val,
                "fee": exit_fee,
                "pnl": pnl,
                "equity": equity,
                "note": f"EXIT | entry={iso_utc(entry_ts)}",
            })

            # reset
            in_pos = False
            entry_price = None
            qty = 0.0
            entry_fees = 0.0
            entry_ts = None

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


def save_trades_csv(pair: str, trades: list) -> str:
    filename = f"trades_{pair_to_filename(pair)}.csv"
    if not trades:
        # zapis pustego nagłówka też jest ok
        fields = ["ts", "pair", "side", "price", "qty", "rsi", "macd", "macd_signal", "fee", "pnl", "equity", "note"]
        with open(filename, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
        return filename

    fields = list(trades[0].keys())
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in trades:
            w.writerow(row)
    return filename


# =========================
# MAIN
# =========================
def main():
    print(f"BACKTEST: timeframe={TIMEFRAME} days={DAYS}")
    print(f"Params: RSI({RSI_PERIOD}) BUY<{RSI_BUY} SELL>{RSI_SELL} | MACD {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}")
    print(f"Costs: fee={FEE_RATE} slippage={SLIPPAGE_RATE} | trade_usdt={TRADE_USDT}\n")

    ex = connect_exchange()

    tf_ms = timeframe_to_ms(TIMEFRAME)
    since_ms = now_ms() - DAYS * 86_400_000

    results = []

    for pair in PAIRS:
        try:
            df = fetch_ohlcv_all(ex, pair, TIMEFRAME, since_ms)
            res = backtest_pair(df, pair)

            # zapis transakcji
            out = save_trades_csv(pair, res["trades"])
            print(f"{pair}: round_trips={res['round_trips']} net_pnl={res['net_pnl']:.4f} "
                  f"win_rate={(res['wins']/(res['wins']+res['losses'])*100) if (res['wins']+res['losses'])>0 else 0:.2f}% "
                  f"PF={res['profit_factor']:.3f} exp={res['expectancy']:.4f} maxDD={res['maxDD']:.4f}")
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
        gross_loss = sum(r.get("gross_loss", 0) for r in results)  # ujemne
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
