import os
import math
import csv
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
import ta


# =========================
# KONFIGURACJA (A)
# =========================
PAIRS = ["BTC/USDT", "ETH/USDT"]

TIMEFRAME = os.getenv("TIMEFRAME", "1h")
DAYS = int(os.getenv("DAYS", "180"))

TRADE_USDT = float(os.getenv("TRADE_USDT", "50"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))          # 0.1%
SLIPPAGE = float(os.getenv("SLIPPAGE_RATE", "0.0002"))    # 0.02%

# Trend
EMA_LEN = int(os.getenv("EMA_LEN", "200"))

# Breakout (Donchian)
BREAKOUT_LEN = int(os.getenv("BREAKOUT_LEN", "20"))

# ATR SL/TP
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
SL_ATR = float(os.getenv("SL_ATR", "1.0"))    # SL = 1.0 * ATR
TP_ATR = float(os.getenv("TP_ATR", "1.5"))    # TP = 1.5 * ATR

# CSV output
OUT_DIR = os.getenv("OUT_DIR", ".")  # na Railway zostaw "."
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# BINANCE (SPOT TESTNET do świec / dane)
# =========================
def connect_binance():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Brak BINANCE_API_KEY / BINANCE_API_SECRET w Variables")

    ex = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    # Testnet (dla spójności z tym co już masz na Railway)
    ex.set_sandbox_mode(True)
    return ex


def fetch_ohlcv(ex, pair, timeframe, since_ms):
    """
    Pobiera OHLCV od since_ms do teraz.
    """
    all_rows = []
    limit = 1000
    now_ms = ex.milliseconds()

    while True:
        rows = ex.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=limit)
        if not rows:
            break
        all_rows.extend(rows)

        last_ts = rows[-1][0]
        # zabezpieczenie przed pętlą
        if last_ts == since_ms:
            break

        since_ms = last_ts + 1
        if since_ms >= now_ms:
            break

        # jak przyszło mniej niż limit, to koniec
        if len(rows) < limit:
            break

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMA200 trend
    df["ema"] = ta.trend.EMAIndicator(df["close"], window=EMA_LEN).ema_indicator()

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=ATR_LEN
    ).average_true_range()

    # Donchian channel (rolling high/low)
    df["donch_hi"] = df["high"].rolling(BREAKOUT_LEN).max()
    df["donch_lo"] = df["low"].rolling(BREAKOUT_LEN).min()

    return df


def costs_for_trade(entry_price, exit_price, qty):
    """
    koszty: fee od nominału wejścia + wyjścia
    """
    notional_in = entry_price * qty
    notional_out = exit_price * qty
    fees = (notional_in + notional_out) * FEE_RATE
    return fees


def backtest_pair(df: pd.DataFrame, pair: str):
    """
    Strategia A: trend + breakout + ATR SL/TP.
    LONG gdy close > EMA i wybicie donch_hi
    SHORT gdy close < EMA i wybicie donch_lo
    """
    # Upewnij się, że mamy dane do wskaźników
    df = df.copy()
    df = add_indicators(df)

    # start dopiero gdy wszystko policzone
    warmup = max(EMA_LEN, ATR_LEN, BREAKOUT_LEN) + 2
    if len(df) <= warmup:
        raise RuntimeError(f"Za mało świec dla {pair}. Potrzebuję > {warmup}")

    in_pos = False
    side = None  # "LONG" / "SHORT"
    entry = None
    sl = None
    tp = None
    qty = 0.0

    equity = 0.0
    peak_equity = 0.0
    max_dd = 0.0

    trades = []
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        close = float(row["close"])
        ema = float(row["ema"])
        atr = float(row["atr"])
        donch_hi_prev = float(prev["donch_hi"])
        donch_lo_prev = float(prev["donch_lo"])

        # sygnały wejścia (breakout na zamknięciu świecy)
        long_signal = (close > ema) and (close > donch_hi_prev)
        short_signal = (close < ema) and (close < donch_lo_prev)

        # jeśli nie w pozycji — otwieramy
        if not in_pos:
            if long_signal and atr > 0:
                side = "LONG"
                entry = close * (1 + SLIPPAGE)
                qty = TRADE_USDT / entry
                sl = entry - SL_ATR * atr
                tp = entry + TP_ATR * atr

                in_pos = True
                trades.append({
                    "ts_entry": row["ts"].isoformat(),
                    "pair": pair,
                    "side": side,
                    "entry": entry,
                    "qty": qty,
                    "sl": sl,
                    "tp": tp,
                    "ts_exit": None,
                    "exit": None,
                    "reason": None,
                    "gross": None,
                    "fees": None,
                    "pnl": None,
                    "equity": None,
                })

            elif short_signal and atr > 0:
                side = "SHORT"
                entry = close * (1 - SLIPPAGE)  # short entry (konserwatywnie)
                qty = TRADE_USDT / entry
                sl = entry + SL_ATR * atr
                tp = entry - TP_ATR * atr

                in_pos = True
                trades.append({
                    "ts_entry": row["ts"].isoformat(),
                    "pair": pair,
                    "side": side,
                    "entry": entry,
                    "qty": qty,
                    "sl": sl,
                    "tp": tp,
                    "ts_exit": None,
                    "exit": None,
                    "reason": None,
                    "gross": None,
                    "fees": None,
                    "pnl": None,
                    "equity": None,
                })

            continue

        # jeśli w pozycji — sprawdzamy SL/TP
        assert trades, "Brak trade, a in_pos=True"
        t = trades[-1]

        exit_price = None
        reason = None

        # Zakładamy wykonanie na close (prosto, konserwatywnie + slippage)
        if side == "LONG":
            # SL/TP
            if close <= sl:
                exit_price = close * (1 - SLIPPAGE)
                reason = "SL"
            elif close >= tp:
                exit_price = close * (1 - SLIPPAGE)
                reason = "TP"

        elif side == "SHORT":
            if close >= sl:
                exit_price = close * (1 + SLIPPAGE)
                reason = "SL"
            elif close <= tp:
                exit_price = close * (1 + SLIPPAGE)
                reason = "TP"

        if exit_price is None:
            continue

        # rozliczenie
        if side == "LONG":
            gross = (exit_price - entry) * qty
        else:
            gross = (entry - exit_price) * qty

        fees = costs_for_trade(entry, exit_price, qty)
        pnl = gross - fees

        equity += pnl
        peak_equity = max(peak_equity, equity)
        dd = equity - peak_equity  # <= 0
        max_dd = min(max_dd, dd)

        if pnl >= 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += pnl  # ujemne

        # zapis trade
        t["ts_exit"] = row["ts"].isoformat()
        t["exit"] = exit_price
        t["reason"] = reason
        t["gross"] = gross
        t["fees"] = fees
        t["pnl"] = pnl
        t["equity"] = equity

        # reset pozycji
        in_pos = False
        side = None
        entry = sl = tp = None
        qty = 0.0

    # metryki
    round_trips = wins + losses
    net_pnl = equity
    win_rate = (wins / round_trips * 100.0) if round_trips else 0.0

    # Profit Factor: suma zysków / suma strat (wartość dodatnia)
    pf = (gross_profit / abs(gross_loss)) if gross_loss != 0 else 0.0
    expectancy = (net_pnl / round_trips) if round_trips else 0.0

    return {
        "pair": pair,
        "round_trips": round_trips,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "profit_factor": pf,
        "expectancy": expectancy,
        "max_dd": max_dd,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "trades": trades,
    }


def write_trades_csv(pair: str, trades: list):
    safe_pair = pair.replace("/", "_")
    path = os.path.join(OUT_DIR, f"trades_{safe_pair}.csv")
    fields = [
        "ts_entry", "ts_exit", "pair", "side", "entry", "exit", "qty",
        "sl", "tp", "reason", "gross", "fees", "pnl", "equity"
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in trades:
            row = {k: t.get(k) for k in fields}
            w.writerow(row)
    print(f"-> zapisano {os.path.basename(path)}")


def main():
    ex = connect_binance()

    # od kiedy pobierać
    since_dt = datetime.now(timezone.utc) - timedelta(days=DAYS)
    since_ms = int(since_dt.timestamp() * 1000)

    print(f"BACKTEST: timeframe={TIMEFRAME} days={DAYS}")
    print("Strategy: A (LONG+SHORT) EMA200 + Breakout(Donchian) + ATR SL/TP")
    print(f"Params: EMA={EMA_LEN} Donch={BREAKOUT_LEN} ATR={ATR_LEN} SL_ATR={SL_ATR} TP_ATR={TP_ATR}")
    print(f"Costs: fee={FEE_RATE} slippage={SLIPPAGE} | trade_usdt={TRADE_USDT}\n")

    results = []

    for pair in PAIRS:
        df = fetch_ohlcv(ex, pair, TIMEFRAME, since_ms)
        r = backtest_pair(df, pair)

        results.append({
            "pair": r["pair"],
            "round_trips": r["round_trips"],
            "wins": r["wins"],
            "losses": r["losses"],
            "win_rate": r["win_rate"],
            "net_pnl": r["net_pnl"],
            "profit_factor": r["profit_factor"],
            "expectancy": r["expectancy"],
            "max_dd": r["max_dd"],
            "gross_profit": r["gross_profit"],
            "gross_loss": r["gross_loss"],
        })

        print(
            f'{pair}: round_trips={r["round_trips"]} '
            f'win_rate={r["win_rate"]:.2f}% '
            f'net_pnl={r["net_pnl"]:.4f} '
            f'PF={r["profit_factor"]:.3f} '
            f'exp={r["expectancy"]:.4f} '
            f'maxDD={r["max_dd"]:.4f}'
        )
        write_trades_csv(pair, r["trades"])
        print()

    # =========================
    # PODSUMOWANIE ŁĄCZNE
    # =========================
    if results:
        total_pnl = sum(x["net_pnl"] for x in results)
        total_round_trips = sum(x["round_trips"] for x in results)
        total_wins = sum(x["wins"] for x in results)
        total_losses = sum(x["losses"] for x in results)
        total_trades = total_wins + total_losses

        win_rate = (total_wins / total_trades * 100.0) if total_trades else 0.0

        gross_profit = sum(x["gross_profit"] for x in results)
        gross_loss = sum(x["gross_loss"] for x in results)  # ujemne
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else 0.0
        expectancy = (total_pnl / total_trades) if total_trades else 0.0

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


if __name__ == "__main__":
    main()
