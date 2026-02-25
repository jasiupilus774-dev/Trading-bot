import os
import math
import csv
import pandas as pd
import ta
import ccxt
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# =========================
# CONFIG (PUBLIC / no keys)
# =========================
PAIRS = ["BTC/USDT"]
TIMEFRAME = "1h"
DAYS = 720

INITIAL_CASH = 1000.0
RISK_PER_TRADE = 0.01          # 1% equity risk per trade
FEE_RATE = 0.001               # 0.1% per side (spot fee model)
SLIPPAGE = 0.0002              # 2 bps

# Strategy: EMA200 trend + Donchian breakout + ATR squeeze filter + ATR SL/TP
EMA_LEN = 200
DONCH_LEN = 10
ATR_LEN = 14

# SL/TP using ATR
SL_ATR_MULT = 1.0
TP_ATR_MULT = 2.0

# Volatility compression filter (ATR% of price)
ATR_PCT_LEN = 50               # lookback for "low vol" regime
ATR_PCT_Q = 0.40               # 0.25 = bottom quartile (more strict). Try 0.40 if too few trades.

# If you want long-only / short-only:
ALLOW_LONG = True
ALLOW_SHORT = False

# Output
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

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
    equity_after: float

# =========================
# DATA & INDICATORS
# =========================
def fetch_ohlcv_public(pair: str) -> pd.DataFrame:
    print(f"Pobieranie danych (PUBLIC) dla {pair}...")
    ex = ccxt.binance({"enableRateLimit": True})
    since = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)

    rows = ex.fetch_ohlcv(pair, timeframe=TIMEFRAME, since=since)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=EMA_LEN).ema_indicator()

    # Donchian channel (use previous completed bars -> shift(1))
    df["donch_high"] = df["high"].rolling(DONCH_LEN).max().shift(1)
    df["donch_low"]  = df["low"].rolling(DONCH_LEN).min().shift(1)

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=ATR_LEN
    ).average_true_range()

    # ATR% (volatility regime)
    df["atr_pct"] = (df["atr"] / df["close"]) * 100.0
    df["atr_pct_thr"] = df["atr_pct"].rolling(ATR_PCT_LEN).quantile(ATR_PCT_Q).shift(1)

    return df

# =========================
# BACKTEST ENGINE
# =========================
def run_backtest(df: pd.DataFrame, pair: str):
    cash = INITIAL_CASH
    peak_cash = INITIAL_CASH
    max_dd = 0.0

    in_pos = False
    side = None          # "LONG" or "SHORT"
    entry = None
    qty = 0.0
    sl = None
    tp = None
    entry_ts = None

    trades: list[Trade] = []

    def apply_fees(entry_px, exit_px, q):
        # fee charged on notional both sides
        return (entry_px * q + exit_px * q) * FEE_RATE

    def update_dd(current_cash):
        nonlocal peak_cash, max_dd
        peak_cash = max(peak_cash, current_cash)
        dd = (peak_cash - current_cash) / peak_cash if peak_cash > 0 else 0.0
        max_dd = max(max_dd, dd)

    for i in range(2, len(df)):
        curr = df.iloc[i]      # current bar (we enter at open of curr)
        prev = df.iloc[i - 1]  # signal bar (closed)

        # skip until indicators ready
        if pd.isna(prev["ema200"]) or pd.isna(prev["donch_high"]) or pd.isna(prev["donch_low"]) or pd.isna(prev["atr"]) or pd.isna(prev["atr_pct_thr"]):
            continue

        # =========================
        # 1) EXIT LOGIC (intrabar SL/TP)
        # =========================
        if in_pos:
            exit_px = None
            reason = None

            if side == "LONG":
                # SL hit?
                if curr["low"] <= sl:
                    exit_px = sl
                    reason = "SL"
                # TP hit?
                elif curr["high"] >= tp:
                    exit_px = tp
                    reason = "TP"
            else:  # SHORT
                if curr["high"] >= sl:
                    exit_px = sl
                    reason = "SL"
                elif curr["low"] <= tp:
                    exit_px = tp
                    reason = "TP"

            if reason:
                # apply slippage on exit
                real_exit = exit_px * (1 - SLIPPAGE) if side == "LONG" else exit_px * (1 + SLIPPAGE)

                gross = (real_exit - entry) * qty if side == "LONG" else (entry - real_exit) * qty
                fees = apply_fees(entry, real_exit, qty)
                pnl = gross - fees

                cash += pnl
                update_dd(cash)

                trades.append(
                    Trade(
                        ts_entry=entry_ts.isoformat(),
                        ts_exit=curr["ts"].isoformat(),
                        pair=pair,
                        side=side,
                        entry=float(entry),
                        exit=float(real_exit),
                        qty=float(qty),
                        pnl=float(pnl),
                        reason=reason,
                        equity_after=float(cash),
                    )
                )

                in_pos = False
                side, entry, qty, sl, tp, entry_ts = None, None, 0.0, None, None, None

        # =========================
        # 2) ENTRY LOGIC (only if flat)
        # =========================
        if not in_pos:
            # Volatility compression filter:
            # take breakouts when ATR% is in "low" regime vs history
            low_vol = prev["atr_pct"] <= prev["atr_pct_thr"]

            # Trend filter:
            trend_up = prev["close"] > prev["ema200"]
            trend_dn = prev["close"] < prev["ema200"]

            # Breakout conditions (on prev close vs Donchian)
            long_sig = trend_up and low_vol and (prev["close"] >= prev["donch_high"])
            short_sig = trend_dn and low_vol and (prev["close"] <= prev["donch_low"])

            if (long_sig and ALLOW_LONG) or (short_sig and ALLOW_SHORT):
                side = "LONG" if long_sig else "SHORT"

                # enter at curr open (+/- slippage)
                entry = curr["open"] * (1 + SLIPPAGE) if side == "LONG" else curr["open"] * (1 - SLIPPAGE)

                # Risk-based sizing using ATR distance
                sl_dist = prev["atr"] * SL_ATR_MULT
                tp_dist = prev["atr"] * TP_ATR_MULT
                if sl_dist <= 0 or tp_dist <= 0:
                    continue

                risk_usd = cash * RISK_PER_TRADE
                qty = risk_usd / sl_dist

                # spot constraint: no leverage
                max_qty = cash / entry
                qty = min(qty, max_qty)

                # ignore microscopic positions
                if qty * entry < 10:  # <10 USDT notional -> skip
                    side, entry, qty = None, None, 0.0
                    continue

                if side == "LONG":
                    sl = entry - sl_dist
                    tp = entry + tp_dist
                else:
                    sl = entry + sl_dist
                    tp = entry - tp_dist

                entry_ts = curr["ts"]
                in_pos = True

    # If still in position at end -> close at last close (mark-to-market)
    if in_pos:
        last = df.iloc[-1]
        mkt_exit = last["close"]
        real_exit = mkt_exit * (1 - SLIPPAGE) if side == "LONG" else mkt_exit * (1 + SLIPPAGE)
        gross = (real_exit - entry) * qty if side == "LONG" else (entry - real_exit) * qty
        fees = (entry * qty + real_exit * qty) * FEE_RATE
        pnl = gross - fees
        cash += pnl
        update_dd(cash)
        trades.append(
            Trade(
                ts_entry=entry_ts.isoformat(),
                ts_exit=last["ts"].isoformat(),
                pair=pair,
                side=side,
                entry=float(entry),
                exit=float(real_exit),
                qty=float(qty),
                pnl=float(pnl),
                reason="EOD",
                equity_after=float(cash),
            )
        )

    return trades, cash, max_dd

# =========================
# REPORTING
# =========================
def profit_factor(trades: list[Trade]) -> float:
    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = sum(t.pnl for t in trades if t.pnl < 0)
    return (gp / abs(gl)) if gl < 0 else 0.0

def expectancy(trades: list[Trade]) -> float:
    return (sum(t.pnl for t in trades) / len(trades)) if trades else 0.0

def save_trades_csv(trades: list[Trade], path: str):
    fields = ["ts_entry", "ts_exit", "pair", "side", "entry", "exit", "qty", "pnl", "reason", "equity_after"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in trades:
            w.writerow({
                "ts_entry": t.ts_entry,
                "ts_exit": t.ts_exit,
                "pair": t.pair,
                "side": t.side,
                "entry": f"{t.entry:.8f}",
                "exit": f"{t.exit:.8f}",
                "qty": f"{t.qty:.8f}",
                "pnl": f"{t.pnl:.8f}",
                "reason": t.reason,
                "equity_after": f"{t.equity_after:.8f}",
            })

def main():
    print(f"\n--- START BACKTESTU (PUBLIC) | {DAYS} dni | TF: {TIMEFRAME} ---")
    print(f"Strategy: EMA{EMA_LEN} trend + Donchian({DONCH_LEN}) breakout + ATR% squeeze(q={ATR_PCT_Q},len={ATR_PCT_LEN}) + ATR SL/TP")
    print(f"Params: ATR={ATR_LEN} SL_ATR={SL_ATR_MULT} TP_ATR={TP_ATR_MULT} | fee={FEE_RATE} slippage={SLIPPAGE} | risk={RISK_PER_TRADE} | long={ALLOW_LONG} short={ALLOW_SHORT}\n")

    all_trades: list[Trade] = []
    results = []

    for pair in PAIRS:
        df = fetch_ohlcv_public(pair)
        df = add_indicators(df)

        trades, final_cash, mdd = run_backtest(df, pair)
        all_trades.extend(trades)

        pnl = final_cash - INITIAL_CASH
        roi = (pnl / INITIAL_CASH) * 100.0
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl < 0)
        n = len(trades)
        wr = (wins / n * 100.0) if n else 0.0
        pf = profit_factor(trades)
        exp = expectancy(trades)

        print(f"PODSUMOWANIE {pair}:")
        print(f"  Trades:        {n}")
        print(f"  Win rate:      {wr:.2f}%")
        print(f"  Net PnL:       {pnl:.2f} USDT ({roi:+.2f}%)")
        print(f"  Profit Factor: {pf:.3f}")
        print(f"  Expectancy:    {exp:.4f} USDT/trade")
        print(f"  MaxDD:         {mdd*100:.2f}%")

        out_path = os.path.join(OUT_DIR, f"trades_{pair.replace('/','_')}.csv")
        save_trades_csv(trades, out_path)
        print(f"  -> zapisano: {out_path}\n")

        results.append({
            "pair": pair,
            "trades": n,
            "wins": wins,
            "losses": losses,
            "net_pnl": pnl,
            "gross_profit": sum(t.pnl for t in trades if t.pnl > 0),
            "gross_loss": sum(t.pnl for t in trades if t.pnl < 0),
            "maxdd": mdd,
        })

    # ===== Global summary
    total_pnl = sum(r["net_pnl"] for r in results)
    total_trades = sum(r["trades"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    total_losses = sum(r["losses"] for r in results)
    total_wr = (total_wins / total_trades * 100.0) if total_trades else 0.0

    gp = sum(r["gross_profit"] for r in results)
    gl = sum(r["gross_loss"] for r in results)
    total_pf = (gp / abs(gl)) if gl < 0 else 0.0
    total_exp = (total_pnl / total_trades) if total_trades else 0.0
    maxdd_any = max((r["maxdd"] for r in results), default=0.0)

    all_path = os.path.join(OUT_DIR, "backtest_trades_all.csv")
    save_trades_csv(all_trades, all_path)

    print("=================================")
    print("BACKTEST SUMMARY (TOTAL)")
    print("=================================")
    print(f"Trades:        {total_trades}")
    print(f"Win rate:      {total_wr:.2f}%")
    print(f"Net PnL:       {total_pnl:.2f} USDT")
    print(f"Profit Factor: {total_pf:.3f}")
    print(f"Expectancy:    {total_exp:.4f} USDT/trade")
    print(f"MaxDD (max):   {maxdd_any*100:.2f}%")
    print(f"-> zapisano:   {all_path}")
    print("=================================\n")
    print("--- KONIEC ---\n")

if __name__ == "__main__":
    main()
