"""
USDT-perpetual 7-DAY-HIGH scanner -> Discord bot
Exchanges (set EXCHANGE env var): bybit (default) | okx | binance

Strategy:
  - Universe: top-N 24h gainers among USDT perpetuals whose 24h change is above MIN_24H_GAIN (5%)
  - Alert when the 5m candle makes a NEW HIGH above the highest high of the previous ~7 days
    (live candle by default, so the alert comes the moment the high is made)

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

import requests

# ----------------------------- CONFIG ---------------------------------
EXCHANGE = os.getenv("EXCHANGE", "bybit").lower()      # bybit | okx | binance
BASES = {
    "bybit": "https://api.bybit.com",
    "okx": "https://www.okx.com",
    "binance": "https://fapi.binance.com",
}

TOP_N = int(os.getenv("TOP_N", 20))                     # watch the top N 24h gainers...
MIN_24H_GAIN = float(os.getenv("MIN_24H_GAIN", 5.0))    # ...with 24h change above this %
LOOKBACK_DAYS = float(os.getenv("LOOKBACK_DAYS", 7))    # "7-day high"

# True  = check the still-forming 5m candle (alert the moment a new 7d high prints)
# False = check only closed 5m candles (alert up to 5 min later, fewer false alarms)
USE_LIVE_CANDLE = os.getenv("USE_LIVE_CANDLE", "true").lower() == "true"
# True = skip coins with less than LOOKBACK_DAYS of history (a new listing has no real 7d high)
REQUIRE_FULL_WINDOW = os.getenv("REQUIRE_FULL_WINDOW", "true").lower() == "true"

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))     # seconds between scans
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", 30))       # min minutes between alerts per coin
# Debug: comma-separated coins to explain in the logs, e.g. DEBUG_SYMBOLS=IDOLUSDT
DEBUG_SYMBOLS = {x.strip().upper() for x in os.getenv("DEBUG_SYMBOLS", "").split(",") if x.strip()}

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "")
# ----------------------------------------------------------------------

TF5_MS = 5 * 60 * 1000
HOUR_MS = 60 * 60 * 1000
WINDOW_MS = int(LOOKBACK_DAYS * 24 * HOUR_MS)
HOURLY_LIMIT = int(LOOKBACK_DAYS * 24) + 12             # 1h candles used for the long baseline
FIVE_LIMIT = 40                                         # recent 5m candles (~3.3h)

if EXCHANGE not in BASES:
    sys.exit(f"Unknown EXCHANGE '{EXCHANGE}'. Use bybit, okx or binance.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bot")
session = requests.Session()
session.headers.update({"User-Agent": "high-scanner-bot/4.0"})

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
#   klines(id, tf, limit) -> [[ts_ms, open, high, low, close], ...] oldest -> newest
#                            tf is "5m" or "1h"; the last candle is the still-forming one
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


def bybit_klines(sym, tf, limit):
    d = get_json("/v5/market/kline", {"category": "linear", "symbol": sym,
                                      "interval": {"5m": "5", "1h": "60"}[tf],
                                      "limit": min(limit, 1000)})
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


def okx_klines(inst, tf, limit):
    d = get_json("/api/v5/market/candles", {"instId": inst,
                                            "bar": {"5m": "5m", "1h": "1H"}[tf],
                                            "limit": min(limit, 300)})
    if not d or d.get("code") != "0" or not d["data"]:
        return None
    rows = d["data"][::-1]                      # API is newest-first
    return [[int(r[0]), r[1], r[2], r[3], r[4]] for r in rows]


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


def binance_klines(sym, tf, limit):
    d = get_json("/fapi/v1/klines", {"symbol": sym, "interval": tf,
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
    """Top-N 24h gainers whose 24h change is above MIN_24H_GAIN."""
    global _last_rank
    rows = _gainers(perp)
    rows.sort(key=lambda x: x["change"], reverse=True)
    _last_rank = rows
    return [r for r in rows[:TOP_N] if r["change"] > MIN_24H_GAIN]


# ------------------------- Candle caches ------------------------------
# Baseline = highest high of the past ~7 days. To keep requests tiny it is built from:
#   - completed 1h candles (one request per coin per hour), plus
#   - recent 5m candles covering the part of the current hour not yet in a 1h candle.
HOURLY = {}     # id -> {open_ts_ms: high}
FIVE = {}       # id -> {open_ts_ms: high}


def _highs(rows):
    return {r[0]: float(r[2]) for r in rows}


def refresh_hourly(sid):
    now_ms = int(time.time() * 1000)
    c = HOURLY.get(sid)
    if c is None or now_ms >= max(c) + HOUR_MS + 5000:      # a new hour has completed
        rows = _klines(sid, "1h", HOURLY_LIMIT)
        if not rows:
            return c is not None
        HOURLY[sid] = _highs(rows)
    return True


def refresh_five(sid):
    now_ms = int(time.time() * 1000)
    c = FIVE.get(sid)
    if c is None or now_ms - max(c) > 10 * TF5_MS:          # first load or stale
        rows = _klines(sid, "5m", FIVE_LIMIT)
        if not rows:
            return False
        FIVE[sid] = _highs(rows)
        return True
    if not USE_LIVE_CANDLE and now_ms < max(c) + TF5_MS + 2000:
        return True                                         # no new closed candle yet
    rows = _klines(sid, "5m", 12)
    if not rows:
        return False
    c.update(_highs(rows))
    extra = len(c) - (FIVE_LIMIT + 20)
    if extra > 0:
        for ts in sorted(c)[:extra]:
            del c[ts]
    return True


def prior_high(sid, ts_c):
    """Highest high in the LOOKBACK window strictly before the 5m candle that opened at ts_c."""
    hourly, five = HOURLY[sid], FIVE[sid]
    done = {t: h for t, h in hourly.items()
            if t + HOUR_MS <= ts_c and t + HOUR_MS > ts_c - WINDOW_MS}
    if not done:
        return None, "no_hourly_data"
    if REQUIRE_FULL_WINDOW and min(hourly) > ts_c - WINDOW_MS + HOUR_MS:
        return None, "short_history"
    last_end = max(done) + HOUR_MS
    if min(five) > last_end:
        return None, "gap_in_data"                          # never alert on incomplete data
    best = max(done.values())
    for t, h in five.items():
        if last_end <= t < ts_c:
            best = max(best, h)
    return best, "ok"


def check(sid):
    if not (refresh_hourly(sid) and refresh_five(sid)):
        return None
    five = FIVE[sid]
    ts = sorted(five)
    if USE_LIVE_CANDLE:
        ts_c = ts[-1]                                       # forming candle
    else:
        if len(ts) < 2:
            return None
        ts_c = ts[-2]                                       # last closed candle
    prev_high, status = prior_high(sid, ts_c)
    if status != "ok":
        return {"status": status}
    cur_high = five[ts_c]
    return {
        "status": "ok",
        "candle_ts": int(ts_c),
        "cur_high": cur_high,
        "prev_high": prev_high,
        "breakout_pct": (cur_high / prev_high - 1) * 100,
        "signal": bool(cur_high > prev_high),
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
    candle_at = datetime.fromtimestamp(res["candle_ts"] / 1000, tz=timezone.utc)
    days = f"{LOOKBACK_DAYS:g}"
    embed = {
        "title": f"📈 {days}-DAY HIGH: {g['name']}",
        "color": 0xF39C12,
        "description": (
            f"**New {days}-day high on the 5m chart**\n\n"
            f"**New high:** {fmt(res['cur_high'])}\n"
            f"**Previous {days}d high:** {fmt(res['prev_high'])} "
            f"(+{res['breakout_pct']:.2f}% above)\n"
            f"**Price now:** {fmt(g['price'])}\n"
            f"**24h change:** {g['change']:+.2f}% (gainer rank #{rank})\n"
            f"**5m candle opened:** {candle_at.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
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
        change = _last_rank[rank - 1]["change"]
        head = f"DEBUG {name}: 24h {change:+.1f}% (rank {rank}/{len(_last_rank)} on {EXCHANGE})"
        if rank > TOP_N or change <= MIN_24H_GAIN:
            log.info("%s -> NOT scanned (needs top %d and 24h > %.1f%%)", head, TOP_N, MIN_24H_GAIN)
            continue
        r = info.get(sid)
        if not r:
            log.info("%s -> waiting for candle data", head)
        elif r["status"] != "ok":
            log.info("%s -> skipped: %s", head, r["status"])
        else:
            log.info("%s | candle high %s vs previous %sd high %s (%+.2f%%) | SIGNAL=%s",
                     head, fmt(r["cur_high"]), f"{LOOKBACK_DAYS:g}", fmt(r["prev_high"]),
                     r["breakout_pct"], r["signal"])


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
def safe_check(sid):
    try:
        return check(sid)
    except Exception:
        log.exception("Check failed for %s", sid)
        return None


def main():
    start_web_server()
    perp = load_perp_symbols()
    while not perp:
        log.error("Could not load %s contract list, retrying in 15s", EXCHANGE)
        time.sleep(15)
        perp = load_perp_symbols()
    perp_loaded_at = time.time()

    info = {}               # id -> latest check result
    last_signal = {}        # id -> unix time of last alert
    last_in_list = {}       # id -> unix time it was last a candidate

    send_discord(content=f"✅ 7-day-high scanner online ({EXCHANGE}). Top {TOP_N} gainers with "
                         f"24h change > {MIN_24H_GAIN:g}% | alert when a coin makes a new "
                         f"{LOOKBACK_DAYS:g}-day high on 5m "
                         f"({'live candle' if USE_LIVE_CANDLE else 'closed candles'})")
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
                results = dict(zip(ids, ex.map(safe_check, ids)))

            hits = []
            for sid in ids:
                last_in_list[sid] = time.time()
                res = results.get(sid)
                if res is None:
                    continue
                info[sid] = res
                if (res.get("signal")
                        and time.time() - last_signal.get(sid, 0) >= COOLDOWN_MIN * 60):
                    send_signal(meta[sid], rank_of[sid], res)
                    last_signal[sid] = time.time()
                    hits.append(meta[sid]["name"])

            for store in (HOURLY, FIVE):            # forget coins that left the list for 2h
                for sid in list(store):
                    if sid not in meta and time.time() - last_in_list.get(sid, 0) > 7200:
                        store.pop(sid, None)
                        info.pop(sid, None)

            log.info("Scanned %d candidate gainers | alerts this scan: %s",
                     len(ids), ", ".join(hits) or "none")
            if DEBUG_SYMBOLS:
                debug_report(perp, info)
        except Exception:
            log.exception("Scan loop error")

        time.sleep(max(1, SCAN_INTERVAL - (time.time() - loop_start)))


if __name__ == "__main__":
    main()
