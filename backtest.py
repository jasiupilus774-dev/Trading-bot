import os
import math
import pandas as pd
import ta
import ccxt
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

# =========================
# STRATEGY A (SPOT LONG-only)
# Mean Reversion w trendzie (EMA200 filter)
# =========================

# --- Settings ---
PAIRS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
DAYS = 180

INITIAL_CASH = 1000.0
RISK_PER_TRADE = 0.01        # 1% kapitału ryzykowane na trade (na SL)
FEE_RATE = 0.001             # 0.1%
SLIPPAGE = 0.0002            # 0.02%

# Trend filter
EMA_TREND_LEN = 200

# Mean reversion tools
EMA_MEAN_LEN = 20
RSI_LEN = 14
BB_LEN = 20
BB_STD = 2.0

# Volatility
ATR_LEN = 14

# Risk logic
SL_ATR_MULT = 1.5            # SL = 1.5 * ATR
TP_ATR_MULT = 1.0            # TP = 1.0 * ATR (często trafia przy MR)
EXIT_ON_EMA20 = True         # dodatkowe wyjście: zamknij jak wróci powyżej EMA20

# Entry thresholds
RSI_ENTRY_MAX = 35.0         # wejście tylko gdy RSI <= 35
MIN_BARS = 250               # żeby EMA200/ATR/BB miały sens


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    fees: float
    reason: str
    rsi: float
    atr: float
    ema200: float


# =========================
# Data
# =========================
def fetch_ohlcv(pair: str) -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True})
    since = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)
    rows = ex.fetch_ohlcv(pair, timeframe=TIMEFRAME, since=since)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=EMA_TREND_LEN).ema_indicator()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=EMA_MEAN_LEN).ema_indicator()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=RSI_LEN).rsi()

    bb = ta.volatility.BollingerBands(close=df["close"], window=BB_LEN, window_dev=BB_STD)
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_high"] = bb.bollinger_hband()

    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_LEN)
    df["atr"] = atr.average_true_range()

    return df


# =========================
# Backtest core
# =========================
def compute_max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = (peak - equity_curve) / peak
    return float(dd.max()) if len(dd) else 0.0


def run_backtest(df: pd.DataFrame, pair: str):
    cash = INITIAL_CASH
    equity = []
    trades = []

    in_pos = False
    entry = qty = sl = tp = None
    entry_ts = None

    # iterujemy od 1 żeby móc używać prev bar
    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        # equity mark-to-market
        equity.append(cash if not in_pos else cash)  # w tej wersji equity = cash (bez MTM); DD liczymy po zamknięciach

        # pomijamy, jeśli wskaźniki jeszcze nie gotowe
        if any(pd.isna(prev[x]) for x in ["ema200", "ema20", "rsi", "bb_low", "atr"]):
            continue

        # =========================
        # EXIT LOGIC (intrabar)
        # =========================
        if in_pos:
            exit_price = None
            reason = None

            # konserwatywnie: jeśli w tej samej świecy dotknęło i SL i TP → liczymy SL
            hit_sl = curr["low"] <= sl
            hit_tp = curr["high"] >= tp

            if hit_sl:
                exit_price = sl
                reason = "SL"
            elif hit_tp:
                exit_price = tp
                reason = "TP"
            elif EXIT_ON_EMA20 and curr["close"] >= curr["ema20"]:
                exit_price = curr["close"]
                reason = "EMA20"

            if reason is not None:
                # wyjście ze slippage
                real_exit = exit_price * (1 - SLIPPAGE)

                gross = (real_exit - entry) * qty
                fees = (entry * qty + real_exit * qty) * FEE_RATE
                pnl = gross - fees
                cash += pnl

                trades.append(
                    Trade(
                        ts_entry=entry_ts.isoformat(),
                        ts_exit=curr["ts"].isoformat(),
                        pair=pair,
                        side="LONG",
                        entry=float(entry),
                        exit=float(real_exit),
                        qty=float(qty),
                        pnl=float(pnl),
                        fees=float(fees),
                        reason=reason,
                        rsi=float(prev["rsi"]),
                        atr=float(prev["atr"]),
                        ema200=float(prev["ema200"]),
                    )
                )

                in_pos = False
                entry = qty = sl = tp = None
                entry_ts = None

        # =========================
        # ENTRY LOGIC
        # =========================
        if not in_pos:
            # Trend UP: cena powyżej EMA200
            trend_up = prev["close"] > prev["ema200"]

            # Mean reversion trigger:
            # 1) RSI niskie
            # 2) cena poniżej dolnego BB (przeciągnięcie)
            mr_trigger = (prev["rsi"] <= RSI_ENTRY_MAX) and (prev["close"] <= prev["bb_low"])

            if trend_up and mr_trigger:
                # wejdź na open następnej świecy (curr.open) + slippage
                entry = float(curr["open"]) * (1 + SLIPPAGE)

                # SL/TP na ATR z poprzedniej świecy (żeby nie patrzeć w przyszłość)
                atr_val = float(prev["atr"])
                if atr_val <= 0:
                    continue

                sl_dist = atr_val * SL_ATR_MULT
                tp_dist = atr_val * TP_ATR_MULT

                sl = entry - sl_dist
                tp = entry + tp_dist

                # risk-based position sizing
                risk_usdt = cash * RISK_PER_TRADE
                # qty tak, aby strata do SL = risk_usdt
                qty = risk_usdt / sl_dist

                # SPOT: bez lewara → wartość pozycji nie może przekroczyć cash
                max_qty = cash / entry
                qty = min(qty, max_qty)

                # jeżeli qty za małe (np. przez mały cash), pomiń
                if qty <= 0 or math.isclose(qty, 0.0):
                    entry = qty = sl = tp = None
                    continue

                in_pos = True
                entry_ts = curr["ts"]

    # equity curve po zamknięciach
    if not equity:
        equity = [INITIAL_CASH]

    equity_series = pd.Series(equity)
    mdd = compute_max_drawdown(equity_series)

    return trades, cash, mdd


def summarize(pair: str, trades, final_cash: float, mdd: float):
    total_pnl = final_cash - INITIAL_CASH
    roi = (total_pnl / INITIAL_CASH) * 100

    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = n - wins
    win_rate = (wins / n * 100) if n > 0 else 0.0

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = sum(t.pnl for t in trades if t.pnl < 0)
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else 0.0

    expectancy = (total_pnl / n) if n > 0 else 0.0

    print(f"\nPODSUMOWANIE DLA {pair}:")
    print(f"  Kapitał końcowy: {final_cash:.2f} USDT ({roi:+.2f}%)")
    print(f"  Max Drawdown:    {mdd*100:.2f}%")
    print(f"  Liczba tradów:   {n}")
    print(f"  Win Rate:        {win_rate:.2f}%")
    print(f"  Profit Factor:   {profit_factor:.2f}")
    print(f"  Expectancy:      {expectancy:.4f} USDT/trade")

    return {
        "pair": pair,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_pnl": total_pnl,
        "roi": roi,
        "pf": profit_factor,
        "expectancy": expectancy,
        "mdd": mdd,
        "final_cash": final_cash,
    }


def save_trades_csv(all_trades):
    if not all_trades:
        print("\nBrak transakcji — nie zapisuję CSV.")
        return None

    df = pd.DataFrame([asdict(t) for t in all_trades])
    path = os.path.join(OUTPUT_DIR, "backtest_trades.csv")
    df.to_csv(path, index=False)
    print(f"\nZapisano wszystkie transakcje do: {path}")
    return path


def main():
    print(f"\n--- START BACKTESTU (SPOT LONG-only) | {DAYS} dni | TF: {TIMEFRAME} ---")
    print("Strategia A: Mean Reversion w trendzie (EMA200) + BB(20,2) + RSI")
    print(f"Params: EMA200={EMA_TREND_LEN} EMA20={EMA_MEAN_LEN} RSI<={RSI_ENTRY_MAX} BB={BB_LEN},{BB_STD}")
    print(f"Risk: SL={SL_ATR_MULT}*ATR TP={TP_ATR_MULT}*ATR | fee={FEE_RATE} slippage={SLIPPAGE} | risk_per_trade={RISK_PER_TRADE}\n")

    results = []
    all_trades = []

    for pair in PAIRS:
        df = fetch_ohlcv(pair)
        df = add_indicators(df)

        if len(df) < MIN_BARS:
            print(f"{pair}: za mało danych ({len(df)} barów).")
            continue

        trades, final_cash, mdd = run_backtest(df, pair)
        all_trades.extend(trades)
        results.append(summarize(pair, trades, final_cash, mdd))

    # ===== TOTAL SUMMARY =====
    if results:
        total_net = sum(r["net_pnl"] for r in results)
        total_trades = sum(r["trades"] for r in results)
        total_wins = sum(r["wins"] for r in results)
        total_losses = sum(r["losses"] for r in results)

        total_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

        gross_profit = sum(max(0.0, t.pnl) for t in all_trades)
        gross_loss = sum(min(0.0, t.pnl) for t in all_trades)
        total_pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else 0.0

        total_expectancy = (total_net / total_trades) if total_trades > 0 else 0.0

        print("\n==============================")
        print("BACKTEST SUMMARY (TOTAL)")
        print("==============================")
        print(f"Trades:       {total_trades}")
        print(f"Win rate:     {total_win_rate:.2f}%")
        print(f"Net PnL:      {total_net:.4f} USDT")
        print(f"ProfitFactor: {total_pf:.3f}")
        print(f"Expectancy:   {total_expectancy:.4f} USDT/trade")
        print("==============================\n")

    save_trades_csv(all_trades)
    print("--- KONIEC ---\n")


if __name__ == "__main__":
    main()
