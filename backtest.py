import os
import math
import csv
import ccxt
import pandas as pd
import ta
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# =========================
# CONFIGURATION
# =========================
PAIRS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
DAYS = int(os.getenv("DAYS", "180"))

# Portfolio / risk
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "1000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # 1% risk per trade

# Costs
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))            # 0.1%
SLIPPAGE = float(os.getenv("SLIPPAGE", "0.0002"))           # 0.02%

# Strategy A2: EMA200 trend + "pullback below EMA20 by ATR" + ATR SL/TP
EMA_200 = int(os.getenv("EMA_200", "200"))
EMA_20 = int(os.getenv("EMA_20", "20"))
ATR_LEN = int(os.getenv("ATR_LEN", "14"))

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))        # SL distance = 1.0 * ATR
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))        # TP distance = 1.5 * ATR

RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_ENTRY_MAX = float(os.getenv("RSI_ENTRY_MAX", "40.0"))   # pullback RSI threshold (was 35)
DIP_ATR_MULT = float(os.getenv("DIP_ATR_MULT", "0.8"))      # entry requires close < ema20 - 0.8*ATR

# Data safety
MIN_BARS = int(os.getenv("MIN_BARS", "250"))

# Outputs
OUT_DIR = os.getenv("OUT_DIR", ".")
TRADES_CSV = os.path.join(OUT_DIR, "backtest_trades.csv")

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
# DATA FETCH
# =========================
def fetch_ohlcv(pair: str) -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True})
    since = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)
    rows = ex.fetch_ohlcv(pair, timeframe=TIMEFRAME, since=since, limit=2000)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=EMA_200).ema_indicator()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=EMA_20).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_LEN).average_true_range()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=RSI_LEN).rsi()
    return df

# =========================
# UTILS: CSV
# =========================
def save_trades_csv(trades: list[Trade]) -> None:
    os.makedirs(os.path.dirname(TRADES_CSV) or ".", exist_ok=True)
    fields = ["ts_entry", "ts_exit", "pair", "side", "entry", "exit", "qty", "pnl", "reason"]
    with open(TRADES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in trades:
            w.writerow({
                "ts_entry": t.ts_entry,
                "ts_exit": t.ts_exit,
                "pair": t.pair,
                "side": t.side,
                "entry": f"{t.entry:.6f}",
                "exit": f"{t.exit:.6f}",
                "qty": f"{t.qty:.8f}",
                "pnl": f"{t.pnl:.6f}",
                "reason": t.reason,
            })

# =========================
# BACKTEST ENGINE (SPOT LONG-only)
# =========================
def run_backtest(df: pd.DataFrame, pair: str):
    cash = INITIAL_CASH
    peak_cash = INITIAL_CASH
    max_drawdown = 0.0

    in_pos = False
    entry = None
    qty = 0.0
    sl = None
    tp = None
    entry_ts = None

    trades: list[Trade] = []
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    round_trips = 0

    # need enough bars for indicators
    if len(df) < MIN_BARS:
        return {
            "pair": pair,
            "round_trips": 0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "maxDD": 0.0,
        }, trades

    for i in range(2, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        # skip if indicators missing
        if any(pd.isna(prev[x]) for x in ["ema200", "ema20", "atr", "rsi"]):
            continue

        # =========================
        # EXIT logic
        # =========================
        if in_pos:
            exit_price = None
            reason = None

            # check intrabar SL/TP
            if curr["low"] <= sl:
                exit_price = sl
                reason = "SL"
            elif curr["high"] >= tp:
                exit_price = tp
                reason = "TP"

            if reason:
                # apply slippage
                real_exit = exit_price * (1 - SLIPPAGE)
                # pnl
                gross = (real_exit - entry) * qty
                fees = (entry * qty + real_exit * qty) * FEE_RATE
                net_pnl = gross - fees

                cash += net_pnl
                peak_cash = max(peak_cash, cash)
                dd = (peak_cash - cash) / peak_cash if peak_cash > 0 else 0
                max_drawdown = max(max_drawdown, dd)

                if net_pnl > 0:
                    wins += 1
                    gross_profit += net_pnl
                else:
                    losses += 1
                    gross_loss += net_pnl

                trades.append(Trade(
                    ts_entry=entry_ts.isoformat(),
                    ts_exit=curr["ts"].isoformat(),
                    pair=pair,
                    side="LONG",
                    entry=float(entry),
                    exit=float(real_exit),
                    qty=float(qty),
                    pnl=float(net_pnl),
                    reason=reason
                ))
                round_trips += 1
                in_pos = False
                entry = None
                qty = 0.0
                sl = None
                tp = None
                entry_ts = None

        # =========================
        # ENTRY logic (A2)
        # =========================
        if not in_pos:
            # Trend filter: only long above EMA200
            trend_ok = prev["close"] > prev["ema200"]

            # Pullback trigger:
            # - RSI below threshold (oversold-ish in uptrend)
            # - price below EMA20 by DIP_ATR_MULT*ATR
            mr_trigger = (prev["rsi"] <= RSI_ENTRY_MAX) and (prev["close"] < (prev["ema20"] - DIP_ATR_MULT * prev["atr"]))

            if trend_ok and mr_trigger:
                # enter at next candle open + slippage
                entry = float(curr["open"]) * (1 + SLIPPAGE)

                # SL/TP distances
                sl_dist = float(prev["atr"]) * SL_ATR_MULT
                tp_dist = float(prev["atr"]) * TP_ATR_MULT
                if sl_dist <= 0 or tp_dist <= 0:
                    continue

                sl = entry - sl_dist
                tp = entry + tp_dist

                # position sizing by risk (no leverage, limited by cash)
                risk_usd = cash * RISK_PER_TRADE
                qty = risk_usd / sl_dist  # qty so that SL hit ~= risk_usd

                # cap by available cash (spot)
                max_qty = cash / entry
                qty = min(qty, max_qty)

                # reject micro orders
                if qty <= 0:
                    continue

                entry_ts = curr["ts"]
                in_pos = True

    # Summary
    net_pnl_total = cash - INITIAL_CASH
    trades_count = wins + losses
    maxDD_pct = max_drawdown * 100.0

    result = {
        "pair": pair,
        "round_trips": round_trips,
        "trades": trades_count,
        "wins": wins,
        "losses": losses,
        "net_pnl": net_pnl_total,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "maxDD": maxDD_pct,
    }
    return result, trades

# =========================
# MAIN
# =========================
def main():
    print(f"\n--- START BACKTESTU (SPOT LONG-only) | {DAYS} dni | TF: {TIMEFRAME} ---")
    print(f"Strategy: A2 (EMA200 trend + pullback below EMA20 by {DIP_ATR_MULT}*ATR + ATR SL/TP)")
    print(f"Params: EMA200={EMA_200} EMA20={EMA_20} ATR={ATR_LEN} SL_ATR={SL_ATR_MULT} TP_ATR={TP_ATR_MULT} RSI({RSI_LEN})<= {RSI_ENTRY_MAX}")
    print(f"Costs: fee={FEE_RATE} slippage={SLIPPAGE} | initial_cash={INITIAL_CASH} | risk_per_trade={RISK_PER_TRADE}\n")

    all_trades: list[Trade] = []
    results = []

    for pair in PAIRS:
        try:
            print(f"Pobieranie danych: {pair} ...")
            df = fetch_ohlcv(pair)
            df = add_indicators(df)

            res, trades = run_backtest(df, pair)
            results.append(res)
            all_trades.extend(trades)

            trades_n = res["trades"]
            win_rate = (res["wins"] / trades_n * 100) if trades_n > 0 else 0.0
            pf = (res["gross_profit"] / abs(res["gross_loss"])) if res["gross_loss"] != 0 else 0.0
            expectancy = (res["net_pnl"] / trades_n) if trades_n > 0 else 0.0

            print(f"\nPODSUMOWANIE DLA {pair}:")
            print(f"  Round trips:   {res['round_trips']}")
            print(f"  Trades:        {trades_n}")
            print(f"  Win rate:      {win_rate:.2f}%")
            print(f"  Net PnL:       {res['net_pnl']:.4f} USDT")
            print(f"  Profit Factor: {pf:.3f}")
            print(f"  Expectancy:    {expectancy:.4f} USDT/trade")
            print(f"  Max Drawdown:  {res['maxDD']:.2f}%")

        except Exception as e:
            print(f"ERROR {pair}: {e}")

    # Save trades
    if all_trades:
        save_trades_csv(all_trades)
        print(f"\nZapisano wszystkie transakcje do: {TRADES_CSV}")
    else:
        print("\nBrak transakcji — nic nie zapisano do CSV.")

    # Global summary
    if results:
        total_pnl = sum(r["net_pnl"] for r in results)
        total_round_trips = sum(r["round_trips"] for r in results)
        total_wins = sum(r["wins"] for r in results)
        total_losses = sum(r["losses"] for r in results)
        total_trades = total_wins + total_losses

        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
        gross_profit = sum(r["gross_profit"] for r in results)
        gross_loss = sum(r["gross_loss"] for r in results)
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else 0.0
        expectancy = (total_pnl / total_trades) if total_trades > 0 else 0.0
        maxDD = max(r["maxDD"] for r in results) if results else 0.0

        print("\n==============================")
        print("BACKTEST SUMMARY (TOTAL)")
        print("==============================")
        print(f"Round trips: {total_round_trips}")
        print(f"Trades:      {total_trades}")
        print(f"Win rate:    {win_rate:.2f}%")
        print(f"Net PnL:     {total_pnl:.4f} USDT")
        print(f"ProfitFact:  {profit_factor:.3f}")
        print(f"Expectancy:  {expectancy:.4f} USDT/trade")
        print(f"MaxDD (max): {maxDD:.2f}%")
        print("==============================\n")

    print("--- KONIEC ---\n")

if __name__ == "__main__":
    main()
