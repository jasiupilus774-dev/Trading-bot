import os
import time
import logging
from datetime import datetime
import json
import csv

import ccxt
import pandas as pd
import ta


# =========================
# PLIKI / SYMULACJA
# =========================
STATE_PATH = "bot_state.json"
TRADES_CSV = "trades.csv"

FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))        # 0.1%
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0002"))  # 0.02%

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"


def load_state(pairs):
    """Wczytuje stan pozycji (żeby restart Railway nie powodował dubli)."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            data = json.load(f)
        # upewnij się, że są wszystkie pary
        for p in pairs:
            data.setdefault(p, {"in_pos": False, "entry": None, "qty": 0.0, "last_action_ts": None})
        return data
    return {p: {"in_pos": False, "entry": None, "qty": 0.0, "last_action_ts": None} for p in pairs}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def append_trade(row: dict):
    """Dopisuje transakcje paper do CSV."""
    file_exists = os.path.exists(TRADES_CSV)
    fieldnames = [
        "ts", "pair", "side", "price", "qty", "entry", "gross", "fees", "pnl", "rsi"
    ]
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        # dopisz brakujące pola jako None
        for k in fieldnames:
            row.setdefault(k, None)
        w.writerow(row)


# =========================
# KONFIGURACJA STRATEGII
# =========================
PAIRS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
TRADE_AMOUNT_USDT = 50
CHECK_INTERVAL = 3600  # 1h

RSI_PERIOD = 14
RSI_BUY = 35
RSI_SELL = 55

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# =========================
# BINANCE TESTNET (CCXT)
# =========================
def connect_binance():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        raise RuntimeError("❌ Brak BINANCE_API_KEY lub BINANCE_API_SECRET")

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    # 🔴 KLUCZOWE: TESTNET
    exchange.set_sandbox_mode(True)

    # test połączenia
    balance = exchange.fetch_balance()
    usdt = balance["free"].get("USDT", 0)

    log.info("✅ Połączono z Binance SPOT TESTNET")
    log.info(f"💰 TESTNET USDT balance: {usdt}")
    log.info(f"🧪 DRY_RUN = {DRY_RUN}")
    log.info(f"⚙️ fee={FEE_RATE} slippage={SLIPPAGE_RATE}")

    return exchange


# =========================
# MARKET DATA / INDIKATORY
# =========================
def get_candles(exchange, pair):
    ohlcv = exchange.fetch_ohlcv(pair, TIMEFRAME, limit=150)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df


def calculate_indicators(df):
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


def get_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi = last["rsi"]

    cross_up = prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]
    cross_down = prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]

    if rsi < RSI_BUY and cross_up:
        return "BUY"
    if rsi > RSI_SELL and cross_down:
        return "SELL"
    return "HOLD"


# =========================
# PAPER TRADING
# =========================
def paper_buy(state, pair, price, rsi):
    entry_price = price * (1 + SLIPPAGE_RATE)
    qty = TRADE_AMOUNT_USDT / entry_price

    state[pair]["in_pos"] = True
    state[pair]["entry"] = entry_price
    state[pair]["qty"] = qty
    state[pair]["last_action_ts"] = datetime.utcnow().isoformat()

    append_trade({
        "ts": state[pair]["last_action_ts"],
        "pair": pair,
        "side": "BUY_PAPER",
        "price": entry_price,
        "qty": qty,
        "rsi": float(rsi),
    })

    log.info(f"🧾 PAPER BUY {pair} @ {entry_price:.2f} qty={qty:.6f}")
    save_state(state)


def paper_sell(state, pair, price, rsi):
    exit_price = price * (1 - SLIPPAGE_RATE)
    qty = float(state[pair]["qty"])
    entry = float(state[pair]["entry"])

    gross = (exit_price - entry) * qty
    fees = (entry * qty + exit_price * qty) * FEE_RATE
    pnl = gross - fees

    append_trade({
        "ts": datetime.utcnow().isoformat(),
        "pair": pair,
        "side": "SELL_PAPER",
        "price": exit_price,
        "qty": qty,
        "entry": entry,
        "gross": gross,
        "fees": fees,
        "pnl": pnl,
        "rsi": float(rsi),
    })

    log.info(f"🧾 PAPER SELL {pair} @ {exit_price:.2f} pnl={pnl:.4f} (fees={fees:.4f})")

    state[pair]["in_pos"] = False
    state[pair]["entry"] = None
    state[pair]["qty"] = 0.0
    state[pair]["last_action_ts"] = datetime.utcnow().isoformat()
    save_state(state)


# =========================
# MAIN LOOP
# =========================
def run_bot():
    log.info("🤖 Bot startuje")
    exchange = connect_binance()
    state = load_state(PAIRS)

    while True:
        log.info(f"🔍 {datetime.now().strftime('%H:%M:%S')} — sprawdzam sygnały")

        for pair in PAIRS:
            try:
                df = calculate_indicators(get_candles(exchange, pair))
                signal = get_signal(df)

                price = exchange.fetch_ticker(pair)["last"]
                rsi = df.iloc[-1]["rsi"]

                log.info(f"{pair} | Cena: {price:.2f} | RSI: {rsi:.1f} | Sygnał: {signal}")

                # ===== PAPER TRADING =====
                pos = state.get(pair, {"in_pos": False, "entry": None, "qty": 0.0, "last_action_ts": None})
                state[pair] = pos

                if signal == "BUY" and not pos["in_pos"]:
                    paper_buy(state, pair, price, rsi)

                elif signal == "SELL" and pos["in_pos"]:
                    paper_sell(state, pair, price, rsi)

            except Exception as e:
                log.error(f"❌ {pair}: {e}")

        log.info("⏰ Następne sprawdzenie za 60 min")
        time.sleep(CHECK_INTERVAL)


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    run_bot()
