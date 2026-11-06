"""
Binance USDT-M Perpetual SHORT signal scanner -> Discord bot

Conditions (all must be true at the same time):
  1. Symbol is a USDT-M PERPETUAL, in the top-N 24h gainers, 24h change > MIN_24H_GAIN
  2. 15m Supertrend is RED (downtrend)
  3. 15m EMA12 < EMA48

A signal fires once, at the moment all conditions become true.
It does NOT place trades. Signals only.
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

# ----------------------------- CONFIG ---------------------------------
BINANCE = "https://fapi.binance.com"

MIN_24H_GAIN = float(os.getenv("MIN_24H_GAIN", 8.0))   # % 24h gain threshold
TOP_N = int(os.getenv("TOP_N", 30))                     # only look at top N gainers
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
EMA_FAST = 12
EMA_SLOW = 48
ST_PERIOD = 10                                          # Supertrend ATR length
ST_MULT = 3.0                                           # Supertrend multiplier
KLINE_LIMIT = 300

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 20))     # seconds between scans
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", 60))       # min minutes between signals per symbol
# True  = evaluate the still-forming candle (instant signals, may flicker/repaint)
# False = only evaluate closed candles (slower but stable)
USE_LIVE_CANDLE = os.getenv("USE_LIVE_CANDLE", "true").lower() == "true"
# If True, symbols already matching on startup are ignored until they re-trigger
SKIP_STARTUP_MATCHES = True

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "")
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bot")
session = requests.Session()
session.headers.update({"User-Agent": "short-signal-bot/1.0"})


# ------------------------- Binance helpers ----------------------------
def get_json(path, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(BINANCE + path, params=params, timeout=10)
            if r.status_code in (418, 429):
                wait = int(r.headers.get("Retry-After", 30))
                log.warning("Rate limited, sleeping %ss", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning("Request %s failed (%s) attempt %d", path, e, attempt + 1)
            time.sleep(2 * (attempt + 1))
    return None


def load_perp_symbols():
    info = get_json("/fapi/v1/exchangeInfo")
    if not info:
        return None
    return {
        s["symbol"]
        for s in info["symbols"]
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
    }


def top_gainers(perp_symbols):
    tickers = get_json("/fapi/v1/ticker/24hr")
    if not tickers:
        return []
    rows = [
        {
            "symbol": t["symbol"],
            "change": float(t["priceChangePercent"]),
            "price": float(t["lastPrice"]),
        }
        for t in tickers
        if t["symbol"] in perp_symbols
    ]
    rows.sort(key=lambda x: x["change"], reverse=True)
    rows = rows[:TOP_N]
    return [r for r in rows if r["change"] > MIN_24H_GAIN]


def get_klines(symbol):
    data = get_json(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": TIMEFRAME, "limit": KLINE_LIMIT},
    )
    if not data or len(data) < EMA_SLOW + 5:
        return None
    df = pd.DataFrame(
        data,
        columns=["open_time", "open", "high", "low", "close", "volume",
                 "close_time", "qv", "trades", "tbv", "tqv", "ignore"],
    )
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    if not USE_LIVE_CANDLE:
        df = df.iloc[:-1]  # drop forming candle
    return df.reset_index(drop=True)


# --------------------------- Indicators -------------------------------
def supertrend(df, period=ST_PERIOD, mult=ST_MULT):
    """Returns (line, direction). direction: -1 = red/downtrend, 1 = green/uptrend."""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values  # Wilder RMA

    hl2 = (high + low) / 2
    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr

    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    direction = np.ones(n, dtype=int)
    line = np.zeros(n)
    final_upper[0], final_lower[0] = basic_upper[0], basic_lower[0]
    line[0] = final_lower[0]

    for i in range(1, n):
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if close[i] > final_upper[i - 1]:
            direction[i] = 1
        elif close[i] < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return line, direction


def evaluate(symbol):
    df = get_klines(symbol)
    if df is None:
        return None
    ema_f = df["close"].ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
    ema_s = df["close"].ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]
    line, direction = supertrend(df)
    return {
        "ok": bool(direction[-1] == -1 and ema_f < ema_s),
        "st_line": float(line[-1]),
        "ema_f": float(ema_f),
        "ema_s": float(ema_s),
        "close": float(df["close"].iloc[-1]),
    }


# ----------------------------- Discord --------------------------------
def send_discord(content=None, embed=None):
    if not DISCORD_TOKEN or not DISCORD_CHANNEL_ID:
        log.error("DISCORD_BOT_TOKEN / DISCORD_CHANNEL_ID not set")
        return
    url = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages"
    headers = {"Authorization": f"Bot {DISCORD_TOKEN}", "Content-Type": "application/json"}
    payload = {}
    if content:
        payload["content"] = content
    if embed:
        payload["embeds"] = [embed]
    for _ in range(3):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            if r.status_code == 429:
                time.sleep(float(r.json().get("retry_after", 2)))
                continue
            if r.status_code >= 400:
                log.error("Discord error %s: %s", r.status_code, r.text)
            return
        except requests.RequestException as e:
            log.warning("Discord send failed: %s", e)
            time.sleep(2)


def fmt(p):
    return f"{p:.6f}".rstrip("0").rstrip(".") if p < 1 else f"{p:,.4f}".rstrip("0").rstrip(".")


def send_signal(sym, change, res):
    embed = {
        "title": f"🔴 SHORT SIGNAL: {sym}",
        "color": 0xE74C3C,
        "description": (
            f"**24h change:** +{change:.2f}%\n"
            f"**Price:** {fmt(res['close'])}\n"
            f"**{TIMEFRAME} Supertrend:** RED (level {fmt(res['st_line'])})\n"
            f"**EMA{EMA_FAST}:** {fmt(res['ema_f'])}  <  **EMA{EMA_SLOW}:** {fmt(res['ema_s'])}\n\n"
            f"[Open on Binance](https://www.binance.com/en/futures/{sym})"
        ),
        "footer": {"text": "Signal only - not financial advice. Manage your own risk."},
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }
    send_discord(embed=embed)
    log.info("SIGNAL sent for %s", sym)


# ------------------------------ Main ----------------------------------
def main():
    perp = load_perp_symbols()
    while not perp:
        log.error("Could not load exchange info, retrying in 15s")
        time.sleep(15)
        perp = load_perp_symbols()
    perp_loaded_at = time.time()

    active = set()          # symbols currently satisfying ALL conditions
    last_signal = {}        # symbol -> unix time of last signal
    first_scan = True

    send_discord(content=f"✅ Short scanner online. Gainers > {MIN_24H_GAIN}% | "
                         f"{TIMEFRAME} Supertrend red + EMA{EMA_FAST}<EMA{EMA_SLOW}")
    log.info("Started. Scanning every %ss", SCAN_INTERVAL)

    while True:
        loop_start = time.time()
        try:
            if time.time() - perp_loaded_at > 3600:
                new = load_perp_symbols()
                if new:
                    perp, perp_loaded_at = new, time.time()

            gainers = top_gainers(perp)
            meta = {g["symbol"]: g for g in gainers}

            with ThreadPoolExecutor(max_workers=8) as ex:
                results = dict(zip(meta.keys(), ex.map(evaluate, meta.keys())))

            now_ok = {s for s, r in results.items() if r and r["ok"]}
            new_hits = now_ok - active

            if first_scan and SKIP_STARTUP_MATCHES:
                if new_hits:
                    log.info("Startup matches ignored: %s", ", ".join(sorted(new_hits)))
                new_hits = set()

            for sym in sorted(new_hits):
                if time.time() - last_signal.get(sym, 0) < COOLDOWN_MIN * 60:
                    continue
                send_signal(sym, meta[sym]["change"], results[sym])
                last_signal[sym] = time.time()

            active = now_ok
            first_scan = False
            log.info("Scanned %d gainers, %d matching", len(gainers), len(now_ok))
        except Exception:
            log.exception("Scan loop error")

        time.sleep(max(1, SCAN_INTERVAL - (time.time() - loop_start)))


if __name__ == "__main__":
    main()
