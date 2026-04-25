"""
╔══════════════════════════════════════════════════════════════════════╗
║           STOCK ALERT DISCORD BOT  —  single file                   ║
║                                                                      ║
║  Slash commands:                                                     ║
║    /alert_add       create a price/rsi/volume alert                  ║
║    /alert_list      list your active alerts                          ║
║    /alert_delete    delete an alert by ID                            ║
║    /alert_pause     pause without deleting                           ║
║    /alert_resume    resume a paused alert                            ║
║    /price           live price card (price, RSI, volume, change)     ║
║    /history         recent trigger history                           ║
║    /watchlist_create  create a named watchlist                       ║
║    /watchlist_add     add a ticker to a watchlist                    ║
║    /watchlist_remove  remove a ticker from a watchlist               ║
║    /watchlist_delete  delete an entire watchlist                     ║
║    /watchlist_list    list all your watchlists                       ║
║    /watchlist        full analysis report for a watchlist            ║
║                                                                      ║
║  Background:                                                         ║
║    • Alert price-checker loop (every 5 min)                         ║
║    • FastAPI webhook  POST /webhook/alert  (external triggers)       ║
║                                                                      ║
║  Storage:                                                            ║
║    • Alerts   → Postgres (alerts + trigger_history tables)           ║
║    • Watchlists → watchlists.json  (no DB dependency for this)       ║
╚══════════════════════════════════════════════════════════════════════╝

ENV VARS REQUIRED:
  DISCORD_TOKEN      — bot token from Discord Developer Portal
  DATABASE_URL       — Postgres connection string (Railway provides this)
  ALERT_CHANNEL_ID   — fallback channel ID for alert pings
  WEBHOOK_SECRET     — secret header value for POST /webhook/alert
  PORT               — HTTP port (default 8000, Railway sets this)
"""

# ── stdlib ──────────────────────────────────────────────────────────────────────
import os
import sys
import json
import math
import time
import random
import asyncio
import logging
import statistics
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# ── third-party ─────────────────────────────────────────────────────────────────
import discord
from discord import app_commands
from discord.ext import tasks
import uvicorn
from fastapi import FastAPI, Request, HTTPException
import psycopg2
import psycopg2.extras
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bot")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS / CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

WATCHLIST_FILE    = Path(os.environ.get("WATCHLIST_FILE", "watchlists.json"))
ALERT_CHANNEL_ID  = int(os.environ.get("ALERT_CHANNEL_ID", 0))
CHECK_INTERVAL    = int(os.environ.get("CHECK_INTERVAL_MIN", 5))
MC_SIMULATIONS    = int(os.environ.get("MC_SIMULATIONS", 500))   # Monte Carlo runs
MC_DAYS           = int(os.environ.get("MC_DAYS", 252))          # 1 trading year

VALID_CONDITIONS = [
    "price_change_pct",   # % change from previous close (negative = drop)
    "rsi_below",          # RSI drops below threshold
    "rsi_above",          # RSI rises above threshold
    "volume_spike",       # volume >= N × 20-day average volume
    "price_below",        # absolute price falls below threshold
    "price_above",        # absolute price rises above threshold
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def db():
    """Return a new Postgres connection with RealDictCursor."""
    return psycopg2.connect(
        os.environ["DATABASE_URL"],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def db_init():
    """Create tables if they don't exist (idempotent)."""
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS "pgcrypto";

                CREATE TABLE IF NOT EXISTS alerts (
                    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
                    ticker              TEXT        NOT NULL,
                    condition_type      TEXT        NOT NULL,
                    threshold           NUMERIC     NOT NULL,
                    description         TEXT        DEFAULT '',
                    discord_user_id     TEXT        NOT NULL,
                    discord_channel_id  TEXT        NOT NULL,
                    active              BOOLEAN     NOT NULL DEFAULT true,
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE TABLE IF NOT EXISTS trigger_history (
                    id             BIGSERIAL   PRIMARY KEY,
                    alert_id       UUID        REFERENCES alerts(id) ON DELETE SET NULL,
                    ticker         TEXT        NOT NULL,
                    condition_type TEXT        NOT NULL,
                    threshold      NUMERIC     NOT NULL,
                    reason         TEXT,
                    triggered_at   TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(active);
                CREATE INDEX IF NOT EXISTS idx_alerts_user   ON alerts(discord_user_id);
                CREATE INDEX IF NOT EXISTS idx_history_alert ON trigger_history(alert_id);
                CREATE INDEX IF NOT EXISTS idx_history_time  ON trigger_history(triggered_at DESC);
            """)
        conn.commit()
        log.info("DB tables ready.")
    finally:
        conn.close()

# ═══════════════════════════════════════════════════════════════════════════════
# WATCHLIST JSON STORAGE
# ═══════════════════════════════════════════════════════════════════════════════
#
# File structure:
# {
#   "<discord_user_id>": {
#     "<watchlist_name>": {
#       "created_at": "ISO-string",
#       "tickers": ["AAPL", "MSFT", ...]
#     },
#     ...
#   },
#   ...
# }

def wl_load() -> dict:
    if WATCHLIST_FILE.exists():
        try:
            return json.loads(WATCHLIST_FILE.read_text())
        except json.JSONDecodeError:
            log.warning("watchlists.json corrupted – starting fresh")
    return {}


def wl_save(data: dict):
    WATCHLIST_FILE.write_text(json.dumps(data, indent=2))


def wl_get_user(data: dict, user_id: str) -> dict:
    return data.setdefault(str(user_id), {})


def wl_get(user_id: str, name: str) -> Optional[dict]:
    data = wl_load()
    return wl_get_user(data, user_id).get(name.lower())


def wl_names(user_id: str) -> list[str]:
    data = wl_load()
    return list(wl_get_user(data, user_id).keys())

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET DATA  (yfinance – no API key required)
# ═══════════════════════════════════════════════════════════════════════════════

def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    ag, al = sum(gains[:period]) / period, sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    return round(100.0 if al == 0 else 100 - 100 / (1 + ag / al), 2)


def _macd(closes: list[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (macd_line, signal_line, histogram)."""
    def ema(vals, n):
        if len(vals) < n:
            return None
        k, v = 2 / (n + 1), vals[0]
        for x in vals[1:]:
            v = x * k + v * (1 - k)
        return round(v, 4)
    if len(closes) < 35:
        return None, None, None
    e12 = ema(closes, 12)
    e26 = ema(closes, 26)
    if e12 is None or e26 is None:
        return None, None, None
    macd_line = round(e12 - e26, 4)
    signal    = ema([closes[i] - closes[i - 1] for i in range(1, len(closes))][-9:], 9)
    hist      = round(macd_line - signal, 4) if signal is not None else None
    return macd_line, signal, hist


def _bollinger(closes: list[float], period: int = 20) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (upper, middle, lower) Bollinger Bands."""
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    mid    = statistics.mean(window)
    std    = statistics.stdev(window)
    return round(mid + 2 * std, 4), round(mid, 4), round(mid - 2 * std, 4)


def _atr(highs, lows, closes, period: int = 14) -> Optional[float]:
    """Average True Range."""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return round(statistics.mean(trs[-period:]), 4)


def _support_resistance(closes: list[float], n: int = 5) -> tuple[float, float]:
    """Naive S/R: rolling min/max over last 20 days."""
    window = closes[-20:] if len(closes) >= 20 else closes
    return round(min(window), 2), round(max(window), 2)


def fetch_ticker(ticker: str) -> dict:
    """
    Full data fetch for one ticker.
    Returns a rich dict used by both /price, /watchlist, and the alert checker.
    """
    t    = yf.Ticker(ticker)
    hist = t.history(period="60d", interval="1d")

    if hist.empty or len(hist) < 2:
        raise ValueError(f"No data for {ticker}")

    closes   = [float(x) for x in hist["Close"].tolist()]
    highs    = [float(x) for x in hist["High"].tolist()]
    lows     = [float(x) for x in hist["Low"].tolist()]
    volumes  = [float(x) for x in hist["Volume"].tolist()]

    price      = closes[-1]
    prev_close = closes[-2]
    volume     = volumes[-1]
    avg_vol    = statistics.mean(volumes[-20:]) if len(volumes) >= 20 else statistics.mean(volumes)
    pct_change = round((price - prev_close) / prev_close * 100, 4)

    rsi        = _rsi(closes)
    macd_l, macd_s, macd_h = _macd(closes)
    bb_up, bb_mid, bb_low  = _bollinger(closes)
    atr        = _atr(highs, lows, closes)
    support, resistance = _support_resistance(closes)

    # 52-week high/low (use full available history if <252 bars)
    hist_yr = t.history(period="1y", interval="1d")
    if not hist_yr.empty:
        yr_closes = hist_yr["Close"].tolist()
        wk52_low  = round(min(yr_closes), 2)
        wk52_high = round(max(yr_closes), 2)
    else:
        wk52_low, wk52_high = round(min(closes), 2), round(max(closes), 2)

    return {
        "ticker":     ticker.upper(),
        "price":      round(price, 4),
        "prev_close": round(prev_close, 4),
        "pct_change": pct_change,
        "volume":     volume,
        "avg_vol":    round(avg_vol, 0),
        "rsi":        rsi,
        "macd_line":  macd_l,
        "macd_signal":macd_s,
        "macd_hist":  macd_h,
        "bb_upper":   bb_up,
        "bb_mid":     bb_mid,
        "bb_lower":   bb_low,
        "atr":        atr,
        "support":    support,
        "resistance": resistance,
        "wk52_low":   wk52_low,
        "wk52_high":  wk52_high,
        "closes":     closes,          # raw list for Monte Carlo
        "fetched_at": datetime.utcnow().isoformat(),
    }


def fetch_all(tickers: list[str]) -> dict[str, dict]:
    """Batch fetch, skipping failures."""
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = fetch_ticker(ticker)
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"fetch_all skipped {ticker}: {e}")
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo(closes: list[float], days: int = MC_DAYS, sims: int = MC_SIMULATIONS) -> dict:
    """
    Geometric Brownian Motion Monte Carlo.
    Returns distribution stats + price target scenarios.
    """
    if len(closes) < 10:
        return {}

    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    mu          = statistics.mean(log_returns)
    sigma       = statistics.stdev(log_returns)
    current     = closes[-1]

    final_prices = []
    for _ in range(sims):
        price = current
        for _ in range(days):
            price *= math.exp(mu + sigma * random.gauss(0, 1))
        final_prices.append(price)

    final_prices.sort()

    def pct(p):
        return round(final_prices[int(p * sims / 100)], 2)

    mean_final = round(statistics.mean(final_prices), 2)
    expected_return = round((mean_final - current) / current * 100, 2)

    return {
        "current":           round(current, 2),
        "mean_target":       mean_final,
        "p10":               pct(10),    # bear case
        "p25":               pct(25),    # mild bear
        "p50":               pct(50),    # median
        "p75":               pct(75),    # mild bull
        "p90":               pct(90),    # bull case
        "expected_return":   expected_return,
        "prob_gain":         round(sum(1 for p in final_prices if p > current) / sims * 100, 1),
        "sigma_annual":      round(sigma * math.sqrt(252) * 100, 2),   # annualised vol %
        "days":              days,
        "simulations":       sims,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# RULE EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(alert: dict, data: dict) -> tuple[bool, str]:
    cond  = alert["condition_type"]
    thr   = float(alert["threshold"])
    tkr   = alert["ticker"]

    try:
        if cond == "price_change_pct":
            pct = data.get("pct_change", 0.0)
            return pct <= thr, f"{tkr} changed {pct:+.2f}% (trigger ≤ {thr:+.2f}%)"
        elif cond == "rsi_below":
            rsi = data.get("rsi")
            return (rsi is not None and rsi < thr), f"{tkr} RSI {rsi} (trigger < {thr})"
        elif cond == "rsi_above":
            rsi = data.get("rsi")
            return (rsi is not None and rsi > thr), f"{tkr} RSI {rsi} (trigger > {thr})"
        elif cond == "volume_spike":
            vol, avg = data.get("volume", 0), data.get("avg_vol", 1) or 1
            ratio = vol / avg
            return ratio >= thr, f"{tkr} volume {ratio:.1f}× avg (trigger ≥ {thr}×)"
        elif cond == "price_below":
            p = data.get("price", 0)
            return p <= thr, f"{tkr} price ${p:.2f} (trigger ≤ ${thr})"
        elif cond == "price_above":
            p = data.get("price", 0)
            return p >= thr, f"{tkr} price ${p:.2f} (trigger ≥ ${thr})"
        else:
            return False, f"Unknown condition: {cond}"
    except Exception as e:
        log.error(f"evaluate error alert={alert.get('id')}: {e}")
        return False, f"Evaluation error: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI WEBHOOK
# ═══════════════════════════════════════════════════════════════════════════════

api = FastAPI(title="Stock Alert Webhook")

@api.get("/health")
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@api.post("/webhook/alert")
async def webhook_alert(request: Request):
    """
    External systems (TradingView, etc.) POST here to fire a Discord alert.
    Header:  X-Webhook-Secret: <WEBHOOK_SECRET env var>
    Body:    { "channel_id": 123456, "message": "..." }
    """
    if request.headers.get("X-Webhook-Secret", "") != os.environ.get("WEBHOOK_SECRET", "changeme"):
        raise HTTPException(status_code=401, detail="Invalid secret")
    body       = await request.json()
    channel_id = int(body.get("channel_id") or ALERT_CHANNEL_ID)
    message    = str(body.get("message", "")).strip()
    if not message:
        raise HTTPException(status_code=400, detail="message required")
    channel = bot.get_channel(channel_id)
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    embed = discord.Embed(title="📡 External Alert", description=message,
                          color=0x00ff88, timestamp=datetime.utcnow())
    await channel.send(embed=embed)
    return {"ok": True}

# ═══════════════════════════════════════════════════════════════════════════════
# DISCORD BOT
# ═══════════════════════════════════════════════════════════════════════════════

intents = discord.Intents.default()
bot     = discord.Client(intents=intents)
tree    = app_commands.CommandTree(bot)

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS: shared embed builders
# ───────────────────────────────────────────────────────────────────────────────

def _trend_arrow(pct: float) -> str:
    if pct > 1:   return "▲▲"
    if pct > 0:   return "▲"
    if pct < -1:  return "▼▼"
    if pct < 0:   return "▼"
    return "━"

def _rsi_label(rsi: Optional[float]) -> str:
    if rsi is None: return "N/A"
    if rsi < 30:  return f"{rsi:.1f} 🟥 Oversold"
    if rsi > 70:  return f"{rsi:.1f} 🟨 Overbought"
    return f"{rsi:.1f} 🟩 Neutral"

def _bb_position(price: float, upper: Optional[float], lower: Optional[float]) -> str:
    if upper is None or lower is None: return "N/A"
    if price >= upper: return "Above upper band"
    if price <= lower: return "Below lower band"
    pct = (price - lower) / (upper - lower) * 100
    return f"{pct:.0f}% of band"

def _fmt(v, prefix="$", decimals=2) -> str:
    if v is None: return "N/A"
    return f"{prefix}{v:,.{decimals}f}"

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: /alert_add
# ═══════════════════════════════════════════════════════════════════════════════

@tree.command(name="alert_add", description="Create a stock alert rule")
@app_commands.describe(
    ticker      = "Ticker symbol, e.g. SPY",
    condition   = "What to watch for",
    threshold   = "Trigger value — negative for drops (e.g. -2 = drops 2%)",
    description = "Optional note about this alert",
)
@app_commands.choices(condition=[
    app_commands.Choice(name="Price drops X%",         value="price_change_pct"),
    app_commands.Choice(name="RSI below threshold",    value="rsi_below"),
    app_commands.Choice(name="RSI above threshold",    value="rsi_above"),
    app_commands.Choice(name="Volume spike (N× avg)",  value="volume_spike"),
    app_commands.Choice(name="Price falls below $",    value="price_below"),
    app_commands.Choice(name="Price rises above $",    value="price_above"),
])
async def alert_add(interaction: discord.Interaction,
                    ticker: str,
                    condition: app_commands.Choice[str],
                    threshold: float,
                    description: str = ""):
    await interaction.response.defer(ephemeral=True)
    ticker = ticker.upper().strip()
    conn   = db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO alerts
                  (ticker, condition_type, threshold, description,
                   discord_channel_id, discord_user_id, active, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,true,now())
                RETURNING id
            """, (ticker, condition.value, threshold, description,
                  str(interaction.channel_id), str(interaction.user.id)))
            conn.commit()
            aid = cur.fetchone()["id"]
    finally:
        conn.close()

    e = discord.Embed(title="✅ Alert Created", color=0x00c853, timestamp=datetime.utcnow())
    e.add_field(name="Ticker",    value=f"`{ticker}`",     inline=True)
    e.add_field(name="Condition", value=condition.name,    inline=True)
    e.add_field(name="Threshold", value=str(threshold),    inline=True)
    if description:
        e.add_field(name="Note", value=description, inline=False)
    e.set_footer(text=f"ID: {aid}")
    await interaction.followup.send(embed=e, ephemeral=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: /alert_list
# ═══════════════════════════════════════════════════════════════════════════════

@tree.command(name="alert_list", description="List your alerts")
@app_commands.describe(show_all="Include paused/inactive alerts")
async def alert_list(interaction: discord.Interaction, show_all: bool = False):
    await interaction.response.defer(ephemeral=True)
    conn = db()
    try:
        with conn.cursor() as cur:
            if show_all:
                cur.execute("SELECT * FROM alerts WHERE discord_user_id=%s ORDER BY created_at DESC",
                            (str(interaction.user.id),))
            else:
                cur.execute("SELECT * FROM alerts WHERE discord_user_id=%s AND active=true ORDER BY created_at DESC",
                            (str(interaction.user.id),))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        await interaction.followup.send("No alerts found.", ephemeral=True)
        return

    e = discord.Embed(title="📋 Your Alerts", color=0x0288d1, timestamp=datetime.utcnow())
    for r in rows[:10]:
        icon = "🟢" if r["active"] else "🔴"
        val  = f"Condition: `{r['condition_type']}`  Threshold: `{r['threshold']}`"
        if r["description"]:
            val += f"\n_{r['description']}_"
        val += f"\nID: `{r['id']}`"
        e.add_field(name=f"{icon} `{r['ticker']}`", value=val, inline=False)
    if len(rows) > 10:
        e.set_footer(text=f"Showing 10 of {len(rows)}")
    await interaction.followup.send(embed=e, ephemeral=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: /alert_delete
# ═══════════════════════════════════════════════════════════════════════════════

@tree.command(name="alert_delete", description="Delete an alert by ID")
@app_commands.describe(alert_id="Alert ID from /alert_list")
async def alert_delete(interaction: discord.Interaction, alert_id: str):
    await interaction.response.defer(ephemeral=True)
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM alerts WHERE id=%s AND discord_user_id=%s RETURNING ticker",
                        (alert_id, str(interaction.user.id)))
            conn.commit()
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        await interaction.followup.send("❌ Alert not found.", ephemeral=True)
        return
    await interaction.followup.send(f"🗑️ Deleted alert for **{row['ticker']}** (`{alert_id}`)", ephemeral=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: /alert_pause  &  /alert_resume
# ═══════════════════════════════════════════════════════════════════════════════

async def _set_active(interaction: discord.Interaction, alert_id: str, active: bool):
    await interaction.response.defer(ephemeral=True)
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE alerts SET active=%s WHERE id=%s AND discord_user_id=%s RETURNING ticker",
                        (active, alert_id, str(interaction.user.id)))
            conn.commit()
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        await interaction.followup.send("❌ Alert not found.", ephemeral=True)
        return
    verb = "▶️ Resumed" if active else "⏸️ Paused"
    await interaction.followup.send(f"{verb} alert for **{row['ticker']}** (`{alert_id}`)", ephemeral=True)

@tree.command(name="alert_pause",  description="Pause an alert without deleting it")
@app_commands.describe(alert_id="Alert ID")
async def alert_pause(interaction: discord.Interaction, alert_id: str):
    await _set_active(interaction, alert_id, False)

@tree.command(name="alert_resume", description="Resume a paused alert")
@app_commands.describe(alert_id="Alert ID")
async def alert_resume(interaction: discord.Interaction, alert_id: str):
    await _set_active(interaction, alert_id, True)

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: /price
# ═══════════════════════════════════════════════════════════════════════════════

@tree.command(name="price", description="Full price card for a ticker")
@app_commands.describe(ticker="Ticker symbol, e.g. AAPL")
async def price_cmd(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker = ticker.upper().strip()
    try:
        d = fetch_ticker(ticker)
    except Exception as e:
        await interaction.followup.send(f"❌ Could not fetch `{ticker}`: {e}")
        return

    arrow = _trend_arrow(d["pct_change"])
    color = 0x00c853 if d["pct_change"] >= 0 else 0xf44336

    e = discord.Embed(title=f"{arrow}  {ticker}", color=color, timestamp=datetime.utcnow())
    e.add_field(name="Price",          value=f"**${d['price']:,.4f}**",                      inline=True)
    e.add_field(name="Day Change",     value=f"{d['pct_change']:+.2f}%",                      inline=True)
    e.add_field(name="Prev Close",     value=f"${d['prev_close']:,.4f}",                      inline=True)
    e.add_field(name="Volume",         value=f"{d['volume']:,.0f}",                           inline=True)
    e.add_field(name="Avg Vol (20d)",  value=f"{d['avg_vol']:,.0f}",                          inline=True)
    e.add_field(name="Vol Ratio",      value=f"{d['volume']/d['avg_vol']:.2f}×",              inline=True)
    e.add_field(name="RSI (14)",       value=_rsi_label(d["rsi"]),                            inline=True)
    e.add_field(name="ATR (14)",       value=_fmt(d["atr"]),                                  inline=True)
    e.add_field(name="Bollinger",      value=_bb_position(d["price"], d["bb_upper"], d["bb_lower"]), inline=True)
    e.add_field(name="BB Upper",       value=_fmt(d["bb_upper"]),                             inline=True)
    e.add_field(name="BB Lower",       value=_fmt(d["bb_lower"]),                             inline=True)
    e.add_field(name="MACD Line",      value=str(d["macd_line"]) if d["macd_line"] else "N/A", inline=True)
    e.add_field(name="52W Low",        value=_fmt(d["wk52_low"]),                             inline=True)
    e.add_field(name="52W High",       value=_fmt(d["wk52_high"]),                            inline=True)
    e.add_field(name="Support",        value=_fmt(d["support"]),                              inline=True)
    e.add_field(name="Resistance",     value=_fmt(d["resistance"]),                           inline=True)
    await interaction.followup.send(embed=e)

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: /history
# ═══════════════════════════════════════════════════════════════════════════════

@tree.command(name="history", description="Recent alert trigger history")
@app_commands.describe(ticker="Filter by ticker (leave blank for all)")
async def history_cmd(interaction: discord.Interaction, ticker: str = ""):
    await interaction.response.defer(ephemeral=True)
    conn = db()
    try:
        with conn.cursor() as cur:
            if ticker:
                cur.execute("""
                    SELECT th.* FROM trigger_history th
                    JOIN alerts a ON th.alert_id = a.id
                    WHERE a.discord_user_id=%s AND th.ticker=%s
                    ORDER BY th.triggered_at DESC LIMIT 10
                """, (str(interaction.user.id), ticker.upper()))
            else:
                cur.execute("""
                    SELECT th.* FROM trigger_history th
                    JOIN alerts a ON th.alert_id = a.id
                    WHERE a.discord_user_id=%s
                    ORDER BY th.triggered_at DESC LIMIT 10
                """, (str(interaction.user.id),))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        await interaction.followup.send("No trigger history found.", ephemeral=True)
        return

    e = discord.Embed(title="📜 Trigger History", color=0xff9800, timestamp=datetime.utcnow())
    for r in rows:
        ts = r["triggered_at"].strftime("%m/%d %H:%M UTC")
        e.add_field(name=f"`{r['ticker']}` — {ts}", value=r["reason"] or "—", inline=False)
    await interaction.followup.send(embed=e, ephemeral=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COMMANDS: /watchlist_create  /watchlist_add  /watchlist_remove
#            /watchlist_delete  /watchlist_list
# ═══════════════════════════════════════════════════════════════════════════════

@tree.command(name="watchlist_create", description="Create a new watchlist")
@app_commands.describe(name="Name for your watchlist, e.g. 'tech' or 'dividend'")
async def watchlist_create(interaction: discord.Interaction, name: str):
    await interaction.response.defer(ephemeral=True)
    uid  = str(interaction.user.id)
    key  = name.lower().strip()
    data = wl_load()
    user = wl_get_user(data, uid)
    if key in user:
        await interaction.followup.send(f"❌ Watchlist **{key}** already exists.", ephemeral=True)
        return
    user[key] = {"created_at": datetime.utcnow().isoformat(), "tickers": []}
    wl_save(data)
    await interaction.followup.send(f"✅ Created watchlist **{key}**.", ephemeral=True)


@tree.command(name="watchlist_add", description="Add tickers to a watchlist")
@app_commands.describe(name="Watchlist name", tickers="Space-separated tickers, e.g. AAPL MSFT GOOG")
async def watchlist_add(interaction: discord.Interaction, name: str, tickers: str):
    await interaction.response.defer(ephemeral=True)
    uid   = str(interaction.user.id)
    key   = name.lower().strip()
    new   = [t.upper().strip() for t in tickers.split()]
    data  = wl_load()
    user  = wl_get_user(data, uid)
    if key not in user:
        await interaction.followup.send(f"❌ Watchlist **{key}** not found. Create it first with `/watchlist_create`.", ephemeral=True)
        return
    existing = set(user[key]["tickers"])
    added    = [t for t in new if t not in existing]
    user[key]["tickers"].extend(added)
    wl_save(data)
    if added:
        await interaction.followup.send(f"➕ Added **{', '.join(added)}** to **{key}**.", ephemeral=True)
    else:
        await interaction.followup.send(f"All tickers already in **{key}**.", ephemeral=True)


@tree.command(name="watchlist_remove", description="Remove a ticker from a watchlist")
@app_commands.describe(name="Watchlist name", ticker="Ticker to remove")
async def watchlist_remove(interaction: discord.Interaction, name: str, ticker: str):
    await interaction.response.defer(ephemeral=True)
    uid  = str(interaction.user.id)
    key  = name.lower().strip()
    tkr  = ticker.upper().strip()
    data = wl_load()
    user = wl_get_user(data, uid)
    if key not in user:
        await interaction.followup.send(f"❌ Watchlist **{key}** not found.", ephemeral=True)
        return
    before = len(user[key]["tickers"])
    user[key]["tickers"] = [t for t in user[key]["tickers"] if t != tkr]
    wl_save(data)
    if len(user[key]["tickers"]) < before:
        await interaction.followup.send(f"➖ Removed **{tkr}** from **{key}**.", ephemeral=True)
    else:
        await interaction.followup.send(f"**{tkr}** was not in **{key}**.", ephemeral=True)


@tree.command(name="watchlist_delete", description="Delete an entire watchlist")
@app_commands.describe(name="Watchlist name to delete")
async def watchlist_delete(interaction: discord.Interaction, name: str):
    await interaction.response.defer(ephemeral=True)
    uid  = str(interaction.user.id)
    key  = name.lower().strip()
    data = wl_load()
    user = wl_get_user(data, uid)
    if key not in user:
        await interaction.followup.send(f"❌ Watchlist **{key}** not found.", ephemeral=True)
        return
    del user[key]
    wl_save(data)
    await interaction.followup.send(f"🗑️ Deleted watchlist **{key}**.", ephemeral=True)


@tree.command(name="watchlist_list", description="List all your watchlists")
async def watchlist_list(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    uid   = str(interaction.user.id)
    names = wl_names(uid)
    if not names:
        await interaction.followup.send("You have no watchlists. Create one with `/watchlist_create`.", ephemeral=True)
        return
    data = wl_load()
    user = data[uid]
    e    = discord.Embed(title="📁 Your Watchlists", color=0x7c4dff, timestamp=datetime.utcnow())
    for n in names:
        wl     = user[n]
        tcount = len(wl["tickers"])
        tlist  = ", ".join(wl["tickers"][:8]) + ("…" if tcount > 8 else "")
        e.add_field(name=f"**{n}**  ({tcount} tickers)",
                    value=tlist or "_empty_", inline=False)
    await interaction.followup.send(embed=e, ephemeral=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: /watchlist  — the big one
# ═══════════════════════════════════════════════════════════════════════════════

@tree.command(name="watchlist", description="Full analysis report for a watchlist")
@app_commands.describe(
    name        = "Watchlist name",
    show_mc     = "Include Monte Carlo price targets (takes a few extra seconds)",
)
async def watchlist_cmd(interaction: discord.Interaction, name: str, show_mc: bool = True):
    await interaction.response.defer()       # public — everyone sees this

    uid = str(interaction.user.id)
    key = name.lower().strip()
    wl  = wl_get(uid, key)

    if wl is None:
        await interaction.followup.send(f"❌ Watchlist **{key}** not found.", ephemeral=True)
        return

    tickers = wl["tickers"]
    if not tickers:
        await interaction.followup.send(f"Watchlist **{key}** is empty.", ephemeral=True)
        return

    await interaction.followup.send(f"⏳ Fetching data for **{key}** ({len(tickers)} tickers)…")

    # ── Fetch all data ─────────────────────────────────────────────────────────
    market = fetch_all(tickers)
    failed = [t for t in tickers if t not in market]

    # ── Summary embed ──────────────────────────────────────────────────────────
    gainers  = sorted([d for d in market.values() if d["pct_change"] >= 0], key=lambda x: -x["pct_change"])
    losers   = sorted([d for d in market.values() if d["pct_change"] < 0],  key=lambda x:  x["pct_change"])
    avg_chg  = statistics.mean([d["pct_change"] for d in market.values()]) if market else 0
    oversold = [d["ticker"] for d in market.values() if d["rsi"] and d["rsi"] < 30]
    obought  = [d["ticker"] for d in market.values() if d["rsi"] and d["rsi"] > 70]

    summary = discord.Embed(
        title       = f"📊  Watchlist: **{key}**",
        description = (
            f"**{len(market)}** tickers fetched  •  "
            f"Avg change: **{avg_chg:+.2f}%**\n"
            f"Gainers: {len(gainers)}  •  Losers: {len(losers)}"
            + (f"  •  ⚠️ Failed: {', '.join(failed)}" if failed else "")
        ),
        color       = 0x00c853 if avg_chg >= 0 else 0xf44336,
        timestamp   = datetime.utcnow(),
    )
    if oversold:
        summary.add_field(name="🟥 Oversold (RSI<30)",    value=" · ".join(oversold), inline=False)
    if obought:
        summary.add_field(name="🟨 Overbought (RSI>70)",  value=" · ".join(obought),  inline=False)
    await interaction.channel.send(embed=summary)

    # ── Per-ticker detailed embeds ─────────────────────────────────────────────
    for tkr in tickers:
        if tkr not in market:
            continue
        d     = market[tkr]
        arrow = _trend_arrow(d["pct_change"])
        color = 0x00c853 if d["pct_change"] >= 0 else 0xf44336

        e = discord.Embed(title=f"{arrow}  {tkr}", color=color)

        # Price block
        e.add_field(name="Price",         value=f"**${d['price']:,.4f}**",          inline=True)
        e.add_field(name="Day Δ",         value=f"{d['pct_change']:+.2f}%",          inline=True)
        e.add_field(name="Volume Ratio",  value=f"{d['volume']/d['avg_vol']:.2f}×",  inline=True)

        # Indicators
        e.add_field(name="RSI (14)",      value=_rsi_label(d["rsi"]),                inline=True)
        e.add_field(name="ATR (14)",      value=_fmt(d["atr"]),                      inline=True)
        e.add_field(name="Bollinger",     value=_bb_position(d["price"], d["bb_upper"], d["bb_lower"]), inline=True)

        # Levels
        e.add_field(name="Support",       value=_fmt(d["support"]),                  inline=True)
        e.add_field(name="Resistance",    value=_fmt(d["resistance"]),               inline=True)
        e.add_field(name="52W Range",     value=f"${d['wk52_low']} – ${d['wk52_high']}", inline=True)

        # MACD
        if d["macd_line"] is not None:
            macd_dir = "▲ Bullish" if (d["macd_line"] or 0) > (d["macd_signal"] or 0) else "▼ Bearish"
            e.add_field(name="MACD",      value=f"{d['macd_line']} ({macd_dir})",    inline=True)

        # Monte Carlo
        if show_mc and d.get("closes"):
            mc = monte_carlo(d["closes"])
            if mc:
                mc_text = (
                    f"Bear (P10): **${mc['p10']:,.2f}**\n"
                    f"Median:      **${mc['p50']:,.2f}**\n"
                    f"Bull (P90): **${mc['p90']:,.2f}**\n"
                    f"Prob. gain: **{mc['prob_gain']}%**\n"
                    f"Ann. vol:   **{mc['sigma_annual']}%**"
                )
                e.add_field(name=f"🎲 Monte Carlo ({mc['days']}d / {mc['simulations']} sims)",
                            value=mc_text, inline=False)

        await interaction.channel.send(embed=e)
        await asyncio.sleep(0.4)   # avoid rate-limit

    await interaction.channel.send(f"✅ **{key}** analysis complete.")

# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND ALERT CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

def _is_market_open() -> bool:
    """Rough NYSE hours check (Mon–Fri, 09:30–16:00 ET, UTC-4 approx)."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    et_min = ((now.hour - 4) % 24) * 60 + now.minute
    return 9 * 60 + 30 <= et_min <= 16 * 60


@tasks.loop(minutes=CHECK_INTERVAL)
async def alert_loop():
    if not _is_market_open():
        log.debug("Market closed – skip check.")
        return

    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM alerts WHERE active=true")
            alerts = cur.fetchall()

        if not alerts:
            return

        tickers = list({a["ticker"] for a in alerts})
        market  = fetch_all(tickers)

        for alert in alerts:
            data = market.get(alert["ticker"])
            if data is None:
                continue

            triggered, reason = evaluate(dict(alert), data)
            if not triggered:
                continue

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trigger_history
                      (alert_id, ticker, condition_type, threshold, reason, triggered_at)
                    VALUES (%s,%s,%s,%s,%s,now())
                """, (alert["id"], alert["ticker"], alert["condition_type"],
                      alert["threshold"], reason))
            conn.commit()

            channel_id = int(alert["discord_channel_id"] or ALERT_CHANNEL_ID)
            channel    = bot.get_channel(channel_id)
            if channel is None:
                continue

            d     = data
            arrow = _trend_arrow(d["pct_change"])
            color = 0xf44336 if d["pct_change"] < 0 else 0xff9800

            e = discord.Embed(
                title       = f"🚨  Alert: {alert['ticker']}",
                description = reason,
                color       = color,
                timestamp   = datetime.utcnow(),
            )
            e.add_field(name="Price",      value=f"${d['price']:.4f}",           inline=True)
            e.add_field(name="Day Δ",      value=f"{arrow} {d['pct_change']:+.2f}%", inline=True)
            if d["rsi"]:
                e.add_field(name="RSI",    value=f"{d['rsi']:.1f}",              inline=True)
            e.set_footer(text=f"Alert ID: {alert['id']}")

            mention = f"<@{alert['discord_user_id']}>"
            await channel.send(content=mention, embed=e)
            log.info(f"Alert fired: {reason}")

    except Exception as exc:
        log.error(f"alert_loop error: {exc}", exc_info=True)
    finally:
        conn.close()

# ═══════════════════════════════════════════════════════════════════════════════
# BOT STARTUP (RAILWAY SAFE)
# ═══════════════════════════════════════════════════════════════════════════════

@bot.event
async def on_ready():
    await tree.sync()
    if not alert_loop.is_running():
        alert_loop.start()
    log.info(f"Ready: {bot.user} | Slash commands synced")


async def run_api():
    """Run FastAPI inside Railway container safely."""
    config = uvicorn.Config(
        api,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    db_init()

    async with bot:
        bot_task = asyncio.create_task(bot.start(os.environ["DISCORD_TOKEN"]))
        api_task = asyncio.create_task(run_api())

        await asyncio.gather(bot_task, api_task)


if __name__ == "__main__":
    asyncio.run(main())
if __name__ == "__main__":
    asyncio.run(main())
