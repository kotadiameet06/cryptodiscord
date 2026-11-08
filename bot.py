"""
Perpetual-futures SHORT signal scanner -> Discord bot
Exchanges (set EXCHANGE env var): bybit (default) | okx | binance

Strategy:
  - Watch the top-N 24h gainers among USDT perpetuals (no minimum % gain)
  - Alert when a 5m candle CLOSES below the EMA288 while the 5m Supertrend is RED
  - The alert fires once, on the first closed candle where both are true
    (it does not matter which of the two happened first)

Signals only. It does NOT place trades. Uses public market data, no API keys.
"""

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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

TOP_N = int(os.getenv("TOP_N", 20))                     # watch the top N 24h gainers
_mg = os.getenv("MIN_24H_GAIN", "").strip()
MIN_24H_GAIN = float(_mg) if _mg else None              # optional; empty = no minimum

TF_LABEL = "5m"
TF_MS = 5 * 60 * 1000
EMA_LEN = int(os.getenv("EMA_LEN", 288))                # EMA length on 5m candles
ST_PERIOD = int(os.getenv("ST_PERIOD", 12))             # Supertrend ATR length
ST_MULT = float(os.getenv("ST_MULT", 3.0))              # Supertrend multiplier
HISTORY = int(os.getenv("HISTORY_CANDLES", 900))        # candles loaded per coin (EMA288 needs a long warm-up)
UPDATE_CANDLES = 10                                     # small refresh once a new candle closes
CLOSE_DELAY_S = 2                                       # wait a moment after the candle boundary

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))     # seconds between scans
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", 30))       # min minutes between signals per coin
# Debug: comma-separated coins to explain in the logs, e.g. DEBUG_SYMBOLS=IDOLUSDT
DEBUG_SYMBOLS = {x.strip().upper() for x in os.getenv("DEBUG_SYMBOLS", "").split(",") if x.strip()}

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
session.headers.update({"User-Agent": "short-signal-bot/3.0"})

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
#   gainers(perp) -> list of {id, name, change, price}
#   klines(id, limit) -> [[ts_ms, open, high, low, close], ...] oldest -> newest (5m candles,
#                        the last one is the still-forming candle)
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
    return [{"id": t["symbol"], "name": t["symbol"],
             "change": float(t["price24hPcnt"]) * 100,
             "price": float(t["lastPrice"])}
            for t in d["result"]["list"] if t["symbol"] in perp]


def bybit_klines(sym, limit):
    d = get_json("/v5/market/kline", {"category": "linear", "symbol": sym,
                                      "interval": "5", "limit": min(limit, 1000)})
    if not d or d.get("retCode") != 0:
        return None
    rows = d["result"]["list"][::-1]            # API is newest-first
    return [[int(r[0]), r[1], r[2], r[3], r[4]] for r in rows]


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


def okx_klines(inst, limit):
    out, after = [], None
    while len(out) < limit:
        want = min(300, limit - len(out))
        params = {"instId": inst, "bar": "5m", "limit": want}
        if after:
            params["after"] = after             # records older than this timestamp
        d = get_json("/api/v5/market/candles", params)
        if not d or d.get("code") != "0":
            break
        data = d["data"]                        # newest-first
        if not data:
            break
        out.extend([[int(r[0]), r[1], r[2], r[3], r[4]] for r in data])
        after = data[-1][0]
        if len(data) < want:
            break
    return out[::-1] or None


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


def binance_klines(sym, limit):
    d = get_json("/fapi/v1/klines", {"symbol": sym, "interval": "5m",
                                     "limit": min(limit, 1500)})
    if not d:
        return None
    return [[int(k[0]), k[1], k[2], k[3], k[4]] for k in d]


ADAPTERS = {
    "bybit": (bybit_symbols, bybit_gainers, bybit_klines,
              lambda i: f"https://www.bybit.com/trade/usdt/{i}"),
    "okx": (okx_symbols, okx_gainers, okx_klines,
            lambda i: f"https://www.okx.com/trade-swap/{i.lower()}"),
    "binance": (binance_symbols, binance_gainers, binance_klines,
                lambda i: f"https://www.binance.com/en/futures/{i}"),
}
load_perp_symbols, _gainers, _klines, trade_url = ADAPTERS[EXCHANGE]

_last_rank = []


def top_gainers(perp):
    """Top-N 24h gainers (optionally above MIN_24H_GAIN)."""
    global _last_rank
    rows = _gainers(perp)
    rows.sort(key=lambda x: x["change"], reverse=True)
    _last_rank = rows
    rows = rows[:TOP_N]
    if MIN_24H_GAIN is not None:
        rows = [r for r in rows if r["change"] > MIN_24H_GAIN]
    return rows


# ------------------------- Candle cache -------------------------------
# History is downloaded once per coin; afterwards only a few recent candles are
# fetched, and only after a new 5m candle should have closed. This keeps the
# request count tiny and gives the EMA288 a long warm-up (matches charts better).
cache = {}      # id -> {open_ts_ms: (open, high, low, close)}


def _to_map(rows):
    return {r[0]: tuple(float(x) for x in r[1:5]) for r in rows}


def update_cache(sid):
    """True = new data to evaluate (first load or a new candle), False = nothing new,
    None = fetch failed."""
    now_ms = int(time.time() * 1000)
    c = cache.get(sid)
    if c is None or now_ms - max(c) > (UPDATE_CANDLES - 2) * TF_MS:
        rows = _klines(sid, HISTORY)            # first load, or cache too stale
        if not rows:
            return None
        cache[sid] = _to_map(rows)
        return True
    last_open = max(c)
    if now_ms < last_open + TF_MS + CLOSE_DELAY_S * 1000:
        return False                            # current candle has not closed yet
    rows = _klines(sid, UPDATE_CANDLES)
    if not rows:
        return None
    c.update(_to_map(rows))
    extra = len(c) - (HISTORY + 100)
    if extra > 0:
        for ts in sorted(c)[:extra]:
            del c[ts]
    return max(c) != last_open


# --------------------------- Indicators -------------------------------
def supertrend(high, low, close, period=ST_PERIOD, mult=ST_MULT):
    """Returns (line, direction). direction: -1 = red/downtrend, 1 = green/uptrend."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values  # Wilder RMA

    hl2 = (high + low) / 2
    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr

    n = len(close)
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


def evaluate(sid):
    """Evaluate CLOSED candles only (the newest, still-forming candle is ignored)."""
    c = cache[sid]
    ts = sorted(c)[:-1]
    if len(ts) < EMA_LEN + 2:
        return {"status": "too_few", "candles": len(ts)}
    arr = np.array([c[t] for t in ts])
    high, low, close = arr[:, 1], arr[:, 2], arr[:, 3]
    ema = pd.Series(close).ewm(span=EMA_LEN, adjust=False).mean().values
    line, direction = supertrend(high, low, close)
    cond = (close < ema) & (direction == -1)
    return {
        "status": "ok",
        "closed_ts": int(ts[-1]),
        "close": float(close[-1]),
        "ema": float(ema[-1]),
        "st_line": float(line[-1]),
        "below": bool(close[-1] < ema[-1]),
        "red": bool(direction[-1] == -1),
        "now": bool(cond[-1]),
        "prev": bool(cond[-2]),
        "signal": bool(cond[-1] and not cond[-2]),   # first candle where both are true
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


def send_signal(g, rank, res):
    closed_at = datetime.fromtimestamp((res["closed_ts"] + TF_MS) / 1000, tz=timezone.utc)
    embed = {
        "title": f"🔴 SHORT SIGNAL: {g['name']}",
        "color": 0xE74C3C,
        "description": (
            f"**{TF_LABEL} candle closed below EMA{EMA_LEN}** with Supertrend RED\n\n"
            f"**Close:** {fmt(res['close'])}\n"
            f"**EMA{EMA_LEN}:** {fmt(res['ema'])}\n"
            f"**Supertrend({ST_PERIOD},{ST_MULT:g}):** RED (level {fmt(res['st_line'])})\n"
            f"**24h change:** {g['change']:+.2f}% (gainer rank #{rank})\n"
            f"**Candle closed:** {closed_at.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"[Open on {EXCHANGE.capitalize()}]({trade_url(g['id'])})"
        ),
        "footer": {"text": f"Data: {EXCHANGE}. Signal only - not financial advice."},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    send_discord(embed=embed)
    log.info("SIGNAL sent for %s", g["name"])


# ------------------------------ Debug ---------------------------------
def debug_report(perp, info):
    """Explain in the logs why chosen coins did or did not trigger."""
    for name in sorted(DEBUG_SYMBOLS):
        sid = name[:-4] + "-USDT-SWAP" if EXCHANGE == "okx" and name.endswith("USDT") else name
        if sid not in perp:
            log.info("DEBUG %s: not a listed USDT perpetual on %s", name, EXCHANGE)
            continue
        rank = next((i + 1 for i, r in enumerate(_last_rank) if r["id"] == sid), None)
        if rank is None:
            log.info("DEBUG %s: no ticker data this scan", name)
            continue
        head = (f"DEBUG {name}: 24h {_last_rank[rank - 1]['change']:+.1f}% "
                f"(rank {rank}/{len(_last_rank)} on {EXCHANGE})")
        if rank > TOP_N:
            log.info("%s -> NOT scanned (outside top %d)", head, TOP_N)
            continue
        r = info.get(sid)
        if not r:
            log.info("%s -> waiting for candle data", head)
        elif r["status"] == "too_few":
            log.info("%s -> only %d closed candles, need %d for EMA%d",
                     head, r["candles"], EMA_LEN + 2, EMA_LEN)
        else:
            log.info("%s | last closed close<EMA%d=%s | ST red=%s | now=%s prev=%s | SIGNAL=%s",
                     head, EMA_LEN, r["below"], r["red"], r["now"], r["prev"], r["signal"])


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
def safe_update(sid):
    try:
        return update_cache(sid)
    except Exception:
        log.exception("Candle update failed for %s", sid)
        return None


def main():
    start_web_server()
    perp = load_perp_symbols()
    while not perp:
        log.error("Could not load %s contract list, retrying in 15s", EXCHANGE)
        time.sleep(15)
        perp = load_perp_symbols()
    perp_loaded_at = time.time()

    info = {}               # id -> latest evaluation
    last_signal = {}        # id -> unix time of last signal
    last_in_top = {}        # id -> unix time it was last in the top N

    min_txt = f" above {MIN_24H_GAIN}%" if MIN_24H_GAIN is not None else ""
    send_discord(content=f"✅ Short scanner online ({EXCHANGE}). Top {TOP_N} 24h gainers{min_txt} | "
                         f"alert when a {TF_LABEL} candle closes below EMA{EMA_LEN} "
                         f"with Supertrend({ST_PERIOD},{ST_MULT:g}) red")
    log.info("Started on %s. Scanning every %ss", EXCHANGE, SCAN_INTERVAL)

    while True:
        loop_start = time.time()
        try:
            if time.time() - perp_loaded_at > 3600:
                new = load_perp_symbols()
                if new:
                    perp, perp_loaded_at = new, time.time()

            gainers = top_gainers(perp)
            ids = [g["id"] for g in gainers]
            rank_of = {g["id"]: i + 1 for i, g in enumerate(gainers)}
            meta = {g["id"]: g for g in gainers}

            with ThreadPoolExecutor(max_workers=6) as ex:
                updated = dict(zip(ids, ex.map(safe_update, ids)))

            evaluated = 0
            for sid in ids:
                last_in_top[sid] = time.time()
                if not updated.get(sid):
                    continue
                evaluated += 1
                res = evaluate(sid)
                info[sid] = res
                if (res.get("signal")
                        and time.time() - last_signal.get(sid, 0) >= COOLDOWN_MIN * 60):
                    send_signal(meta[sid], rank_of[sid], res)
                    last_signal[sid] = time.time()

            for sid in list(cache):                 # forget coins that left the top N for 1h
                if sid not in meta and time.time() - last_in_top.get(sid, 0) > 3600:
                    cache.pop(sid, None)
                    info.pop(sid, None)

            active = sorted(meta[s]["name"] for s in ids if info.get(s, {}).get("now"))
            log.info("Scanned top %d gainers | evaluated %d new candles | cached %d | "
                     "below EMA%d + red now: %s", len(ids), evaluated, len(cache), EMA_LEN,
                     ", ".join(active) or "none")
            if DEBUG_SYMBOLS:
                debug_report(perp, info)
        except Exception:
            log.exception("Scan loop error")

        time.sleep(max(1, SCAN_INTERVAL - (time.time() - loop_start)))


if __name__ == "__main__":
    main()
