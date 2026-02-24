import os
import time
import logging
import json
import csv
from datetime import datetime

import ccxt
import pandas as pd
import ta

# =========================
# KONFIGURACJA
# =========================
PAIRS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
TRADE_AMOUNT_USDT = float(os.getenv("TRADE_AMOUNT_USDT", "50"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "3600"))  # sekundy

# RSI — pullbacki w trendzie
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_BUY = float(os.getenv("RSI_BUY", "40"))
RSI_SELL = float(os.getenv("RSI_SELL", "60"))

# MACD
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))

# EMA filtr trendu
EMA_FAST = int(os.getenv("EMA_FAST", "50"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "200"))

# Risk/Reward
RR = float(os.getenv("RR", "1.5"))
SL_PCT = float(os.getenv("SL_PCT", "0.02"))
TP_PCT = float(os.getenv("TP_PCT", str(SL_PCT * RR)))

# Koszty symulacji
FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))          # 0.1%
SLIPPAGE = float(os.getenv("SLIPPAGE_RATE", "0.0002"))    # 0.02%

# Tryby:
# MODE=paper   -> paper trading (CSV/state)
# MODE=live    -> real orders on Binance testnet (spot)
# MODE=backtest-> jednorazowy backtest
MODE = os.getenv("MODE", "paper").lower()
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"  # ma sens głównie w MODE=live

BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "180"))

# Dane na dysku (Railway Volume: ustaw DATA_DIR=/data)
DATA_DIR = os.getenv("DATA_DIR", ".").rstrip("/")
STATE_PATH = f"{DATA_DIR}/bot_state.json"
TRADES_CSV = f"{DATA_DIR}/trades.csv"

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# =========================
# STATE
# =========================
def _default_pos():
    return {"in_pos": False, "entry": None, "qty": 0.0, "sl": None, "tp": None}

def load_state(pairs):
    data = {}
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

    for p in pairs:
        data.setdefault(p, _default_pos())
    return data

def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)

def append_trade(row):
    log.info(f"📊 TRADE EVENT: {row}")

# =========================
# BINANCE (CCXT)
# =========================
def connect_binance():
    key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_API_SECRET")
    if not key or not secret:
        raise RuntimeError("❌ Brak BINANCE_API_KEY lub BINANCE_API_SECRET")

    ex = ccxt.binance({
        "apiKey": key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    ex.set_sandbox_mode(True)  # TESTNET

    bal = ex.fetch_balance()
    usdt = bal["free"].get("USDT", 0)
    log.info("✅ Połączono z Binance SPOT TESTNET")
    log.info(f"💰 TESTNET USDT balance: {usdt}")
    return ex

# =========================
# MARKET DATA
# =========================
def get_candles(ex, pair, limit=250):
    ohlcv = ex.fetch_ohlcv(pair, TIMEFRAME, limit=limit)
    return pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=EMA_FAST).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=EMA_SLOW).ema_indicator()

    return df

# =========================
# STRATEGIA
# =========================
def get_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi = float(last["rsi"])
    price = float(last["close"])  # sygnały i zarządzanie na close świecy

    uptrend = price > float(last["ema_slow"]) and float(last["ema_fast"]) > float(last["ema_slow"])
    downtrend = price < float(last["ema_slow"]) and float(last["ema_fast"]) < float(last["ema_slow"])
    trend = "UP" if uptrend else "DOWN" if downtrend else "FLAT"

    cross_up = float(prev["macd"]) < float(prev["macd_signal"]) and float(last["macd"]) > float(last["macd_signal"])
    cross_down = float(prev["macd"]) > float(prev["macd_signal"]) and float(last["macd"]) < float(last["macd_signal"])

    if uptrend and rsi < RSI_BUY and cross_up:
        return "BUY", rsi, trend, price
    if downtrend and rsi > RSI_SELL and cross_down:
        return "SELL", rsi, trend, price
    return "HOLD", rsi, trend, price

# =========================
# PAPER (symulacja)
# =========================
def paper_buy(state, pair, price, rsi, trend):
    entry = price * (1 + SLIPPAGE)
    qty = TRADE_AMOUNT_USDT / entry
    sl = entry * (1 - SL_PCT)
    tp = entry * (1 + TP_PCT)

    state[pair] = {"in_pos": True, "entry": entry, "qty": qty, "sl": sl, "tp": tp}
    save_state(state)

    append_trade({
        "ts": datetime.utcnow().isoformat(),
        "pair": pair,
        "side": "BUY",
        "price": entry,
        "qty": qty,
        "sl": sl,
        "tp": tp,
        "rsi": rsi,
        "trend": trend,
        "mode": "paper"
    })

    log.info(f"🧾 PAPER BUY {pair} @ {entry:.2f} | SL:{sl:.2f} TP:{tp:.2f} | RSI:{rsi:.1f} Trend:{trend}")

def paper_sell(state, pair, price, rsi, trend, reason):
    pos = state[pair]
    exit_price = price * (1 - SLIPPAGE)

    qty = float(pos["qty"])
    entry = float(pos["entry"])

    gross = (exit_price - entry) * qty
    fees = (entry * qty + exit_price * qty) * FEE_RATE
    pnl = gross - fees

    append_trade({
        "ts": datetime.utcnow().isoformat(),
        "pair": pair,
        "side": f"SELL_{reason}",
        "price": exit_price,
        "qty": qty,
        "entry": entry,
        "sl": pos.get("sl"),
        "tp": pos.get("tp"),
        "gross": gross,
        "fees": fees,
        "pnl": pnl,
        "rsi": rsi,
        "trend": trend,
        "mode": "paper"
    })

    log.info(f"🧾 PAPER SELL {pair} @ {exit_price:.2f} | PnL:{pnl:.4f} | {reason}")

    state[pair] = _default_pos()
    save_state(state)

# =========================
# LIVE (prawdziwe zlecenia testnet)
# =========================
def live_buy(ex, pair, price, rsi, trend):
    if DRY_RUN:
        log.info(f"🟡 DRY_RUN: LIVE BUY {pair} pominięty")
        return None

    entry = price  # na testnecie i tak będzie po rynku; slippage liczymy w zarządzaniu/metrykach
    qty = TRADE_AMOUNT_USDT / entry
    qty = float(ex.amount_to_precision(pair, qty))

    order = ex.create_market_buy_order(pair, qty)
    log.info(f"🟢 LIVE BUY {pair} qty={qty} | order_id={order.get('id')}")
    return order

def live_sell(ex, pair):
    if DRY_RUN:
        log.info(f"🟡 DRY_RUN: LIVE SELL {pair} pominięty")
        return None

    base = pair.split("/")[0]
    bal = ex.fetch_balance()["free"].get(base, 0)
    if not bal or bal <= 0:
        log.info(f"⚠️ LIVE: brak {base} do sprzedaży")
        return None

    qty = float(ex.amount_to_precision(pair, bal))
    order = ex.create_market_sell_order(pair, qty)
    log.info(f"🔴 LIVE SELL {pair} qty={qty} | order_id={order.get('id')}")
    return order

# =========================
# BACKTEST
# =========================
def backtest_pair(df: pd.DataFrame, pair: str):
    in_pos = False
    entry = qty = sl = tp = None
    trades = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for i in range(2, len(df)):
        window = df.iloc[:i+1]
        sig, rsi, trend, price = get_signal(window)

        if in_pos:
            if price <= sl:
                exit_price = price * (1 - SLIPPAGE)
                gross = (exit_price - entry) * qty
                fees = (entry * qty + exit_price * qty) * FEE_RATE
                pnl = gross - fees
                equity += pnl
                trades.append((df.iloc[i]["ts"], "SELL_SL", entry, exit_price, qty, pnl))
                in_pos = False
            elif price >= tp:
                exit_price = price * (1 - SLIPPAGE)
                gross = (exit_price - entry) * qty
                fees = (entry * qty + exit_price * qty) * FEE_RATE
                pnl = gross - fees
                equity += pnl
                trades.append((df.iloc[i]["ts"], "SELL_TP", entry, exit_price, qty, pnl))
                in_pos = False
            elif sig == "SELL":
                exit_price = price * (1 - SLIPPAGE)
                gross = (exit_price - entry) * qty
                fees = (entry * qty + exit_price * qty) * FEE_RATE
                pnl = gross - fees
                equity += pnl
                trades.append((df.iloc[i]["ts"], "SELL_SIGNAL", entry, exit_price, qty, pnl))
                in_pos = False
        else:
            if sig == "BUY":
                entry = price * (1 + SLIPPAGE)
                qty = TRADE_AMOUNT_USDT / entry
                sl = entry * (1 - SL_PCT)
                tp = entry * (1 + TP_PCT)
                in_pos = True
                trades.append((df.iloc[i]["ts"], "BUY", entry, None, qty, 0.0))

        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    pnl_list = [t[5] for t in trades if str(t[1]).startswith("SELL")]
    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p < 0]

    round_trips = len(pnl_list)
    win_rate = (len(wins) / round_trips * 100) if round_trips else 0.0
    gross_profit = sum(wins)
    gross_loss = -sum(losses) if losses else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    expectancy = (sum(pnl_list) / round_trips) if round_trips else 0.0
    net_pnl = sum(pnl_list)

    out = f"{DATA_DIR}/trades_{pair.replace('/', '_')}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "action", "entry", "exit", "qty", "pnl"])
        for row in trades:
            w.writerow(row)

    return {
        "pair": pair,
        "round_trips": round_trips,
        "net_pnl": net_pnl,
        "win_rate": win_rate,
        "pf": pf,
        "expectancy": expectancy,
        "max_dd": max_dd,
        "csv": out
    }

def run_backtest():
    log.info(f"BACKTEST: timeframe={TIMEFRAME} days={BACKTEST_DAYS}")
    log.info(f"Params: RSI({RSI_PERIOD}) BUY<{RSI_BUY} SELL>{RSI_SELL} | MACD {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}")
    log.info(f"Trend: EMA{EMA_FAST}/EMA{EMA_SLOW} | SL={SL_PCT*100:.2f}% TP={TP_PCT*100:.2f}%")
    log.info(f"Costs: fee={FEE_RATE} slippage={SLIPPAGE} | trade_usdt={TRADE_AMOUNT_USDT}")

    ex = connect_binance()

    total_pnl = 0.0
    total_round_trips = 0

    for pair in PAIRS:
        limit = min(1500, BACKTEST_DAYS * 24 + 250)
        df = get_candles(ex, pair, limit=limit)
        df = calculate_indicators(df).dropna().reset_index(drop=True)

        res = backtest_pair(df, pair)
        total_pnl += res["net_pnl"]
        total_round_trips += res["round_trips"]

        log.info(
            f"{pair}: round_trips={res['round_trips']} net_pnl={res['net_pnl']:.4f} "
            f"win_rate={res['win_rate']:.2f}% PF={res['pf']:.4f} exp={res['expectancy']:.4f} maxDD={res['max_dd']:.4f}"
        )
        log.info(f"-> zapisano {res['csv']}")

    log.info(f"TOTAL: round_trips={total_round_trips} net_pnl={total_pnl:.4f}")

# =========================
# PĘTLA GŁÓWNA (paper/live)
# =========================
def run_loop():
    log.info(f"🤖 Bot startuje | MODE={MODE} | timeframe={TIMEFRAME}")
    ex = connect_binance()
    state = load_state(PAIRS)

    while True:
        log.info(f"🔍 {datetime.now().strftime('%H:%M:%S')} — sprawdzam sygnały")

        for pair in PAIRS:
            try:
                df = calculate_indicators(get_candles(ex, pair, limit=250)).dropna()
                signal, rsi, trend, price = get_signal(df)
                pos = state[pair]

                log.info(f"{pair} | close: {price:.2f} | RSI:{rsi:.1f} | Trend:{trend} | → {signal}")

                if pos["in_pos"]:
                    if price <= float(pos["sl"]):
                        if MODE == "paper":
                            paper_sell(state, pair, price, rsi, trend, reason="SL")
                        else:
                            live_sell(ex, pair)
                            state[pair] = _default_pos()
                            save_state(state)

                    elif price >= float(pos["tp"]):
                        if MODE == "paper":
                            paper_sell(state, pair, price, rsi, trend, reason="TP")
                        else:
                            live_sell(ex, pair)
                            state[pair] = _default_pos()
                            save_state(state)

                    elif signal == "SELL":
                        if MODE == "paper":
                            paper_sell(state, pair, price, rsi, trend, reason="SIGNAL")
                        else:
                            live_sell(ex, pair)
                            state[pair] = _default_pos()
                            save_state(state)

                else:
                    if signal == "BUY":
                        # wejście
                        if MODE == "paper":
                            paper_buy(state, pair, price, rsi, trend)
                        else:
                            live_buy(ex, pair, price, rsi, trend)
                            # zapisujemy SL/TP do state nawet w live, żeby bot wiedział kiedy wyjść
                            entry = price * (1 + SLIPPAGE)
                            qty = TRADE_AMOUNT_USDT / entry
                            sl = entry * (1 - SL_PCT)
                            tp = entry * (1 + TP_PCT)
                            state[pair] = {"in_pos": True, "entry": entry, "qty": qty, "sl": sl, "tp": tp}
                            save_state(state)

            except Exception as e:
                log.error(f"❌ {pair}: {e}")

        log.info("⏰ Następne sprawdzenie za 60 min")
        time.sleep(CHECK_INTERVAL)

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    if MODE == "backtest":
        run_backtest()
    elif MODE in ("paper", "live"):
        run_loop()
    else:
        raise RuntimeError("Ustaw MODE=paper albo MODE=live albo MODE=backtest")
