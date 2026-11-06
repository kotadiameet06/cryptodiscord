"""
Perpetual-futures SHORT signal scanner -> Discord bot
Exchanges (set EXCHANGE env var): bybit (default) | okx | binance

Conditions (all must be true at the same time):
  1. USDT perpetual contract, in the top-N 24h gainers, 24h change > MIN_24H_GAIN
  2. 15m Supertrend is RED (downtrend)
  3. 15m EMA12 < EMA48

A signal fires once, at the moment all conditions become true.
It does NOT place trades. Signals only. Uses public market data, no API keys.
"""

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import requests

# ----------------------------- CONFIG ---------------------------------
EXCHANGE = os.getenv("EXCHANGE", "bybit").lower()      # bybit | okx | binance
BASES = {
    "bybit": "https://api.bybit.com",
    "okx": "https://www.okx.com",
    "binance": "https://fapi.binance.com",
}

MIN_24H_GAIN = float(os.getenv("MIN_24H_GAIN", 8.0))   # % 24h gain threshold
TOP_N = int(os.getenv("TOP_N", 30))                     # only look at top N gainers
TF_LABEL = "15m"
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

if EXCHANGE not in BASES:
    sys.exit(f"Unknown EXCHANGE '{EXCHANGE}'. Use bybit, okx or binance.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bot")
session = requests.Session()
session.headers.update({"User-Agent": "short-signal-bot/2.0"})

_last_alert = 0.0


def alert(text):
    """Rate-limited Discord warning so failures are never silent."""
    global _last_alert
    if time.time() - _last_alert > 600:
        _last_alert = time.time()
        send_discord(content=text)


# --------------------------- HTTP helper ------------------------------
def get_json(path, params=None, retries=3):
    url = BASES[EXCHANGE] + path
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=10)
            if r.status_code in (418, 429):
                wait = int(float(r.headers.get("Retry-After", 30)))
                log.warning("Rate limited by %s, sleeping %ss", EXCHANGE, wait)
                if wait >= 60:
                    alert(f"⚠️ {EXCHANGE} rate-limited this server's IP. "
                          f"Pausing about {wait // 60} min.")
                time.sleep(wait)
                continue
            if r.status_code in (403, 451):
                log.error("HTTP %s from %s: region blocked or rate limited "
                          "(use a non-US server region)", r.status_code, EXCHANGE)
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            log.warning("Request %s failed (%s) attempt %d", path, e, attempt + 1)
            time.sleep(2 * (attempt + 1))
    return None


# ------------------------- Exchange adapters --------------------------
# Each adapter provides:
#   symbols() -> set of native ids (USDT perpetuals), or None on failure
#   gainers(perp) -> list of {id, name, change, price}, sorted by gain desc
#   klines(id) -> list of [open, high, low, close] oldest -> newest (last = forming)
#   url(id) -> link to the trading page

# ---- Bybit (v5, linear USDT perpetuals) ----
def bybit_symbols():
    out, cursor = set(), None
    for _ in range(10):
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        d = get_json("/v5/market/instruments-info", params)
        if not d or d.get("retCode") != 0:
            return None
        res = d["result"]
        for s in res["list"]:
            if (s.get("contractType") == "LinearPerpetual"
                    and s.get("quoteCoin") == "USDT"
                    and s.get("status") == "Trading"):
                out.add(s["symbol"])
        cursor = res.get("nextPageCursor")
        if not cursor:
            break
    return out or None


def bybit_gainers(perp):
    d = get_json("/v5/market/tickers", {"category": "linear"})
    if not d or d.get("retCode") != 0:
        return []
    rows = []
    for t in d["result"]["list"]:
        if t["symbol"] in perp:
            rows.append({"id": t["symbol"], "name": t["symbol"],
                         "change": float(t["price24hPcnt"]) * 100,
                         "price": float(t["lastPrice"])})
    return rows


def bybit_klines(sym):
    d = get_json("/v5/market/kline", {"category": "linear", "symbol": sym,
                                      "interval": "15", "limit": KLINE_LIMIT})
    if not d or d.get("retCode") != 0:
        return None
    rows = d["result"]["list"][::-1]            # API is newest-first
    return [[r[1], r[2], r[3], r[4]] for r in rows]


# ---- OKX (v5, USDT-margined swaps) ----
def okx_symbols():
    d = get_json("/api/v5/public/instruments", {"instType": "SWAP"})
    if not d or d.get("code") != "0":
        return None
    return {i["instId"] for i in d["data"]
            if i["instId"].endswith("-USDT-SWAP") and i.get("state") == "live"} or None


def okx_gainers(perp):
    d = get_json("/api/v5/market/tickers", {"instType": "SWAP"})
    if not d or d.get("code") != "0":
        return []
    rows = []
    for t in d["data"]:
        if t["instId"] in perp:
            last, op = float(t["last"]), float(t["open24h"])
            if op <= 0:
                continue
            rows.append({"id": t["instId"],
                         "name": t["instId"].replace("-SWAP", "").replace("-", ""),
                         "change": (last / op - 1) * 100, "price": last})
    return rows


def okx_klines(inst):
    d = get_json("/api/v5/market/candles", {"instId": inst, "bar": "15m",
                                            "limit": KLINE_LIMIT})
    if not d or d.get("code") != "0":
        return None
    rows = d["data"][::-1]                      # API is newest-first
    return [[r[1], r[2], r[3], r[4]] for r in rows]


# ---- Binance (USD-M futures) ----
def binance_symbols():
    info = get_json("/fapi/v1/exchangeInfo")
    if not info:
        return None
    return {s["symbol"] for s in info["symbols"]
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"}


def binance_gainers(perp):
    d = get_json("/fapi/v1/ticker/24hr")
    if not d:
        return []
    return [{"id": t["symbol"], "name": t["symbol"],
             "change": float(t["priceChangePercent"]),
             "price": float(t["lastPrice"])}
            for t in d if t["symbol"] in perp]


def binance_klines(sym):
    d = get_json("/fapi/v1/klines", {"symbol": sym, "interval": "15m",
                                     "limit": KLINE_LIMIT})
    if not d:
        return None
    return [[k[1], k[2], k[3], k[4]] for k in d]


ADAPTERS = {
    "bybit": (bybit_symbols, bybit_gainers, bybit_klines,
              lambda i: f"https://www.bybit.com/trade/usdt/{i}"),
    "okx": (okx_symbols, okx_gainers, okx_klines,
            lambda i: f"https://www.okx.com/trade-swap/{i.lower()}"),
    "binance": (binance_symbols, binance_gainers, binance_klines,
                lambda i: f"https://www.binance.com/en/futures/{i}"),
}
load_perp_symbols, _gainers, _klines, trade_url = ADAPTERS[EXCHANGE]


def top_gainers(perp):
    rows = _gainers(perp)
    rows.sort(key=lambda x: x["change"], reverse=True)
    return [r for r in rows[:TOP_N] if r["change"] > MIN_24H_GAIN]


def get_klines(sym):
    rows = _klines(sym)
    if not rows:
        return None
    if not USE_LIVE_CANDLE:
        rows = rows[:-1]                        # drop forming candle
    if len(rows) < EMA_SLOW + 5:
        return None
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"]).astype(float)
    return df


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


def evaluate(sym):
    df = get_klines(sym)
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


def send_signal(g, res):
    embed = {
        "title": f"🔴 SHORT SIGNAL: {g['name']}",
        "color": 0xE74C3C,
        "description": (
            f"**24h change:** +{g['change']:.2f}%\n"
            f"**Price:** {fmt(res['close'])}\n"
            f"**{TF_LABEL} Supertrend:** RED (level {fmt(res['st_line'])})\n"
            f"**EMA{EMA_FAST}:** {fmt(res['ema_f'])}  <  **EMA{EMA_SLOW}:** {fmt(res['ema_s'])}\n\n"
            f"[Open on {EXCHANGE.capitalize()}]({trade_url(g['id'])})"
        ),
        "footer": {"text": f"Data: {EXCHANGE}. Signal only - not financial advice."},
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
    }
    send_discord(embed=embed)
    log.info("SIGNAL sent for %s", g["name"])


# ------------------- Keep-alive web server (Render) -------------------
class _Ping(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"bot running")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def start_web_server():
    port = os.getenv("PORT")
    if not port:
        return
    server = HTTPServer(("0.0.0.0", int(port)), _Ping)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("Health server listening on port %s", port)


# ------------------------------ Main ----------------------------------
def main():
    start_web_server()
    perp = load_perp_symbols()
    while not perp:
        log.error("Could not load %s contract list, retrying in 15s", EXCHANGE)
        time.sleep(15)
        perp = load_perp_symbols()
    perp_loaded_at = time.time()

    active = set()          # ids currently satisfying ALL conditions
    last_signal = {}        # id -> unix time of last signal
    first_scan = True

    send_discord(content=f"✅ Short scanner online ({EXCHANGE}). Gainers > {MIN_24H_GAIN}% | "
                         f"{TF_LABEL} Supertrend red + EMA{EMA_FAST}<EMA{EMA_SLOW}")
    log.info("Started on %s. Scanning every %ss", EXCHANGE, SCAN_INTERVAL)

    while True:
        loop_start = time.time()
        try:
            if time.time() - perp_loaded_at > 3600:
                new = load_perp_symbols()
                if new:
                    perp, perp_loaded_at = new, time.time()

            gainers = top_gainers(perp)
            meta = {g["id"]: g for g in gainers}

            with ThreadPoolExecutor(max_workers=8) as ex:
                results = dict(zip(meta.keys(), ex.map(evaluate, meta.keys())))

            now_ok = {s for s, r in results.items() if r and r["ok"]}
            new_hits = now_ok - active

            if first_scan and SKIP_STARTUP_MATCHES:
                if new_hits:
                    log.info("Startup matches ignored: %s", ", ".join(sorted(new_hits)))
                new_hits = set()

            for sid in sorted(new_hits):
                if time.time() - last_signal.get(sid, 0) < COOLDOWN_MIN * 60:
                    continue
                send_signal(meta[sid], results[sid])
                last_signal[sid] = time.time()

            active = now_ok
            first_scan = False
            log.info("Scanned %d gainers, %d matching", len(gainers), len(now_ok))
        except Exception:
            log.exception("Scan loop error")

        time.sleep(max(1, SCAN_INTERVAL - (time.time() - loop_start)))


if __name__ == "__main__":
    main()
