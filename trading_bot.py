"""
Crypto Trading Bot — BTC/USDT & ETH/USDT
Strategia: RSI + MACD | Giełda: Binance
"""

import os, time, logging
from datetime import datetime
import ccxt, pandas as pd, ta

# ===== KONFIGURACJA =====
PAIRS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
TRADE_AMOUNT_USDT = 50   # zmień na swoją kwotę
RSI_PERIOD, RSI_BUY, RSI_SELL = 14, 35, 65
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
CHECK_INTERVAL = 3600    # co 1 godzinę

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

def connect_binance():
    exchange = ccxt.binance({
        "apiKey": os.getenv("BINANCE_API_KEY"),
        "secret": os.getenv("BINANCE_API_SECRET"),
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    log.info("✅ Połączono z Binance")
    return exchange

def get_candles(exchange, pair):
    ohlcv = exchange.fetch_ohlcv(pair, TIMEFRAME, limit=100)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    return df

def calculate_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=RSI_PERIOD).rsi()
    macd = ta.trend.MACD(df["close"], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    return df

def get_signal(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    rsi = last["rsi"]
    cross_up = prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]
    cross_down = prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]
    if rsi < RSI_BUY and cross_up:   return "BUY"
    if rsi > RSI_SELL and cross_down: return "SELL"
    return "HOLD"

def get_balance(exchange, currency="USDT"):
    return exchange.fetch_balance()["free"].get(currency, 0)

def buy(exchange, pair):
    price = exchange.fetch_ticker(pair)["last"]
    amount = exchange.amount_to_precision(pair, TRADE_AMOUNT_USDT / price)
    log.info(f"🟢 KUPNO {pair} | Cena: {price:.2f} | Ilość: {amount}")
    return exchange.create_market_buy_order(pair, amount)

def sell(exchange, pair):
    base = pair.split("/")[0]
    position = exchange.fetch_balance()["free"].get(base, 0)
    if position <= 0:
        return log.info(f"⚠️ Brak pozycji {pair}")
    price = exchange.fetch_ticker(pair)["last"]
    amount = exchange.amount_to_precision(pair, position)
    log.info(f"🔴 SPRZEDAŻ {pair} | Cena: {price:.2f} | Ilość: {amount}")
    return exchange.create_market_sell_order(pair, amount)

def run_bot():
    log.info("🤖 Bot startuje!")
    exchange = connect_binance()
    while True:
        log.info(f"🔍 {datetime.now().strftime('%H:%M:%S')} — Sprawdzam sygnały...")
        for pair in PAIRS:
            try:
                df = calculate_indicators(get_candles(exchange, pair))
                signal = get_signal(df)
                rsi = df.iloc[-1]["rsi"]
                price = exchange.fetch_ticker(pair)["last"]
                log.info(f"{pair} | Cena: {price:.2f} | RSI: {rsi:.1f} | → {signal}")
                if signal == "BUY":
                    if get_balance(exchange) >= TRADE_AMOUNT_USDT: buy(exchange, pair)
                    else: log.warning("Za mało USDT!")
                elif signal == "SELL":
                    sell(exchange, pair)
            except Exception as e:
                log.error(f"❌ {pair}: {e}")
        log.info(f"⏰ Następne sprawdzenie za 60 min...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run_bot()
