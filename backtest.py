import os
import csv
import math
import pandas as pd
import ta
import ccxt
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# =========================
# CONFIGURATION (SPOT LONG-only)
# =========================
PAIRS = ["BTC/USDT", "ETH/USDT"]

TIMEFRAME = os.getenv("TIMEFRAME", "1h")   # "15m" albo "1h"
DAYS = int(os.getenv("DAYS", "180"))

INITIAL_CASH = float(os.getenv("INITIAL_CASH", "1000.0"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # 1% ryzyka na trade

FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))          # 0.1%
SLIPPAGE = float(os.getenv("SLIPPAGE_RATE", "0.0002"))    # 0.02%

# Strategia: EMA200 + Donchian + RSI + ATR SL + trailing
EMA_LEN = int(os.getenv("EMA_LEN", "200"))
DONCH_LEN = int(os.getenv("DONCH_LEN", "10"))
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.5"))

# Filtry RSI (LONG)
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_LONG_MIN = float(os.getenv("RSI_LONG_MIN", "35"))
RSI_LONG_MAX = float(os.getenv("RSI_LONG_MAX", "65"))

# CSV output (opcjonalnie)
TRADES_CSV = os.getenv("TRADES_CSV", "backtest_trades.csv")


@dataclass
class Trade:
    ts_entry: str
    ts_exit: str
    pair: str
    side: str
    entry: float
    exit: float
    qty: float
    pnl: float
    reason: str


# =========================
# UTILS
# =========================
def tf_to_ms(tf: str) -> int:
    unit = tf[-1]
    n = int(tf[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 60 * 60_000
    if unit == "d":
        return n * 24 * 60 * 60_000
    raise ValueError(f"Unknown timeframe: {tf}")


def ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def connect_binance_public():
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


def fetch_ohlcv_history(ex, pair: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """
    Pobiera OHLCV od since_ms do teraz, z paginacją (ważne).
    """
    all_rows = []
    limit = 1000
    now_ms = ex.milliseconds()
    tf_ms = tf_to_ms(timeframe)

    cur = since_ms
    while True:
        batch = ex.fetch_ohlcv(pair, timeframe=timeframe, since=cur, limit=limit)
        if not batch:
            break

        all_rows.extend(batch)
        last_ts = batch[-1][0]

        # przesuwamy o jedną świecę do przodu
        cur = last_ts + tf_ms

        if cur >= now_ms:
            break

        # jeśli dostaliśmy mniej niż limit, kończymy
        if len(batch) < limit:
            break

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


# =========================
# INDICATORS
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=EMA_LEN).ema_indicator()

    # Donchian liczymy i przesuwamy o 1 świecę, żeby nie używać bieżącej
    df["donch_high"] = df["high"].rolling(DONCH_LEN).max().shift(1)
    df["donch_low"] = df["low"].rolling(DONCH_LEN).min().shift(1)

    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=ATR_LEN
    ).average_true_range()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=RSI_LEN).rsi()
    return df


# =========================
# BACKTEST ENGINE (LONG-only)
# =========================
def run_backtest(df: pd.DataFrame, pair: str):
    cash = INITIAL_CASH
    peak_cash = INITIAL_CASH
    max_drawdown = 0.0

    in_pos = False
    entry = None
    qty = 0.0
    sl = None
    entry_ts = None

    trades: list[Trade] = []

    # Warmup
    warmup = max(EMA_LEN, DONCH_LEN, ATR_LEN, RSI_LEN) + 5
    if len(df) <= warmup:
        return trades, cash, max_drawdown

    for i in range(warmup + 1, len(df)):
        prev = df.iloc[i - 1]  # sygnał na zamknięciu prev
        cur = df.iloc[i]       # wejście/wyjście na open cur (model)

        # --- EXIT (jeśli w pozycji)
        if in_pos:
            exit_price = None
            reason = None

            # SL intrabar: jeśli low <= sl, wyjście po sl (z poślizgiem)
            if cur["low"] <= sl:
                exit_raw = float(sl)
                exit_price = exit_raw * (1 - SLIPPAGE)
                reason = "SL"
            else:
                # trailing SL: podciągamy SL do prev.donch_low (konserwatywnie)
                if not pd.isna(prev["donch_low"]):
                    sl = max(sl, float(prev["donch_low"]))

            if reason:
                gross = (exit_price - entry) * qty
                fees = (entry * qty + exit_price * qty) * FEE_RATE
                pnl = gross - fees

                cash += pnl
                peak_cash = max(peak_cash, cash)
                dd = (peak_cash - cash) / peak_cash if peak_cash > 0 else 0.0
                max_drawdown = max(max_drawdown, dd)

                trades.append(
                    Trade(
                        ts_entry=entry_ts.isoformat(),
                        ts_exit=cur["ts"].isoformat(),
                        pair=pair,
                        side="LONG",
                        entry=float(entry),
                        exit=float(exit_price),
                        qty=float(qty),
                        pnl=float(pnl),
                        reason=reason,
                    )
                )

                in_pos = False
                entry = None
                qty = 0.0
                sl = None
                entry_ts = None

        # --- ENTRY (tylko jeśli flat)
        if not in_pos:
            if pd.isna(prev["ema200"]) or pd.isna(prev["donch_high"]) or pd.isna(prev["atr"]) or pd.isna(prev["rsi"]):
                continue

            # Trend + RSI pullback + breakout
            is_long = (
                prev["close"] > prev["ema200"]
                and RSI_LONG_MIN <= prev["rsi"] <= RSI_LONG_MAX
                and prev["close"] >= prev["donch_high"]
            )

            if is_long:
                entry_raw = float(cur["open"])
                entry = entry_raw * (1 + SLIPPAGE)

                sl_dist = float(prev["atr"]) * SL_ATR_MULT
                if sl_dist <= 0:
                    continue

                # risk-based sizing (bez lewara): ryzyko 1% kapitału
                risk_usdt = cash * RISK_PER_TRADE
                qty = risk_usdt / sl_dist

                # ograniczenie do posiadanego cash
                max_qty = cash / entry
                qty = min(qty, max_qty)

                # zabezpieczenie: jeśli qty jest praktycznie 0
                if qty <= 0:
                    continue

                sl = entry - sl_dist
                entry_ts = cur["ts"]
                in_pos = True

    return trades, cash, max_drawdown


def write_trades(trades: list[Trade], path: str):
    ensure_dir_for_file(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_entry", "ts_exit", "pair", "side", "entry", "exit", "qty", "pnl", "reason"])
        for t in trades:
            w.writerow([t.ts_entry, t.ts_exit, t.pair, t.side, t.entry, t.exit, t.qty, t.pnl, t.reason])


# =========================
# RUN + REPORT
# =========================
if __name__ == "__main__":
    print(f"--- START BACKTESTU (SPOT LONG-only) | {DAYS} dni | TF: {TIMEFRAME} ---")
    print(f"Params: EMA={EMA_LEN} Donch={DONCH_LEN} ATR={ATR_LEN} SL_ATR_MULT={SL_ATR_MULT} RSI({RSI_LEN})=[{RSI_LONG_MIN},{RSI_LONG_MAX}]")
    print(f"Costs: fee={FEE_RATE} slippage={SLIPPAGE} | initial_cash={INITIAL_CASH} | risk_per_trade={RISK_PER_TRADE}\n")

    ex = connect_binance_public()
    all_trades: list[Trade] = []

    for pair in PAIRS:
        since_dt = datetime.now(timezone.utc) - timedelta(days=DAYS)
        since_ms = int(since_dt.timestamp() * 1000)

        df = fetch_ohlcv_history(ex, pair, TIMEFRAME, since_ms)
        df = add_indicators(df)

        trades, final_cash, mdd = run_backtest(df, pair)
        all_trades.extend(trades)

        total_pnl = final_cash - INITIAL_CASH
        roi = (total_pnl / INITIAL_CASH) * 100.0

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = sum(t.pnl for t in losses)  # ujemne lub 0
        pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else 0.0

        expectancy = (total_pnl / len(trades)) if trades else 0.0

        print(f"PODSUMOWANIE DLA {pair}:")
        print(f"  Kapitał końcowy: {final_cash:.2f} USDT ({roi:+.2f}%)")
        print(f"  Max Drawdown:    {mdd*100:.2f}%")
        print(f"  Liczba tradów:   {len(trades)}")
        print(f"  Win Rate:        {win_rate:.2f}%")
        print(f"  Profit Factor:   {pf:.2f}")
        print(f"  Expectancy:      {expectancy:.4f} USDT/trade\n")

    # Zapis wszystkich trade’ów do jednego CSV (opcjonalnie)
    if TRADES_CSV:
        write_trades(all_trades, TRADES_CSV)
        print(f"Zapisano wszystkie transakcje do: {TRADES_CSV}")

    print("\n--- KONIEC ---")
