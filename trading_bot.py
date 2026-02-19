import os
import time
import logging
from datetime import datetime

import ccxt
import pandas as pd
import ta

# =========================
# KONFIGURACJA
# =========================
PAIRS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
TRADE_AMOUNT_USDT = 50

RSI_PERIOD = 14
RSI_BUY = 35
RSI_SELL = 65

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

CHECK_INTERVAL = 3600  # 1h

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

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
        "options": {
            "defaultType": "spot",
        },
    })

    # 🔴 KLUCZOWE — TESTNET
    exchange.set_sandbox_mode(True)

    # test połączenia
    balance = exchange.fetch_balance()
    usdt = balance["free"].get("USDT", 0)

    log.info("✅ Połączono z Binance SPOT TESTNET")
    log.info(f"💰 TESTNET USDT balance: {usdt}")
    log.info(f"🧪 DRY_RUN = {DRY_RUN}")

    return exchange

# =========================
# MARKET DATA
# =========================
def get_candles(exchange, pair):
    ohlcv = exchange.fetch_ohlcv(pair, TIMEFRAME, limit=100)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    return df

def calculate_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(
        df["close"], window=RSI_PERIOD
    ).rsi()

    macd = ta.trend.MACD(
        df["close"],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    return df

# =========================
# STRATEGIA
# =========================
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
# BALANCE / ORDERS
# =========================
def get_balance(exchange, currency="USDT"):
    return exchange.fetch_balance()["free"].get(currency, 0)

def buy(exchange, pair):
    if DRY_RUN:
        log.info(f"🟡 DRY_RUN: BUY {pair} pominięty")
        return None

    price = exchange.fetch_ticker(pair)["last"]
    amount = exchange.amount_to_precision(pair, TRADE_AMOUNT_USDT / price)

    log.info(f"🟢 BUY {pair} | Cena: {price:.2f} | Ilość: {amount}")
    return exchange.create_market_buy_order(pair, amount)

def sell(exchange, pair):
    if DRY_RUN:
        log.info(f"🟡 DRY_RUN: SELL {pair} pominięty")
        return None

    base = pair.split("/")[0]
    balance = exchange.fetch_balance()["free"].get(base, 0)

    if balance <= 0:
        log.info(f"⚠️ Brak pozycji {pair}")
        return None

    price = exchange.fetch_ticker(pair)["last"]
    amount = exchange.amount_to_precision(pair, balance)

    log.info(f"🔴 SELL {pair} | Cena: {price:.2f} | Ilość: {amount}")
    return exchange.create_market_sell_order(pair, amount)

# =========================
# MAIN LOOP
# =========================
def run_bot():
    log.info("🤖 Bot startuje")
    exchange = connect_binance()

    while True:
        log.info(f"🔍 {datetime.now().strftime('%H:%M:%S')} — sprawdzam sygnały")

        for pair in PAIRS:
            try:
                df = get_candles(exchange, pair)
                df = calculate_indicators(df)

                signal = get_signal(df)
                price = exchange.fetch_ticker(pair)["last"]
                rsi = df.iloc[-1]["rsi"]

                log.info(
                    f"{pair} | Cena: {price:.2f} | RSI: {rsi:.1f} | Sygnał: {signal}"
                )

                if signal == "BUY":
                    if get_balance(exchange) >= TRADE_AMOUNT_USDT:
                        buy(exchange, pair)
                    else:
                        log.warning("⚠️ Za mało USDT")

                elif signal == "SELL":
                    sell(exchange, pair)

            except Exception as e:
                log.error(f"❌ {pair}: {e}")

        log.info("⏰ Następne sprawdzenie za 60 min")
        time.sleep(CHECK_INTERVAL)

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    run_bot()
