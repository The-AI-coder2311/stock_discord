import os
import asyncio
import json
import math
import random
import statistics
import logging
import io
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import tasks

import uvicorn
from fastapi import FastAPI, Request, HTTPException

import psycopg2
import psycopg2.extras
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ───────────────────────── CONFIG ─────────────────────────

TOKEN          = os.environ["DISCORD_TOKEN"]
PORT           = int(os.environ.get("PORT", 8000))
ALERT_CHANNEL_ID = int(os.environ.get("ALERT_CHANNEL_ID", 0))
WATCHLIST_FILE = Path("watchlists.json")
PORTFOLIO_FILE = Path("portfolios.json")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

# ───────────────────────── DISCORD SETUP ─────────────────────────

intents = discord.Intents.default()
client  = discord.Client(intents=intents)
tree    = app_commands.CommandTree(client)

# ───────────────────────── FASTAPI ─────────────────────────

api = FastAPI()

@api.get("/health")
def health():
    return {"ok": True}

# ───────────────────────── DB ─────────────────────────

def db():
    return psycopg2.connect(
        os.environ["DATABASE_URL"],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )

def db_init():
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                ticker TEXT,
                condition_type TEXT,
                threshold NUMERIC,
                discord_user_id TEXT,
                discord_channel_id TEXT,
                active BOOLEAN DEFAULT true,
                created_at TIMESTAMPTZ DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS trigger_history (
                id BIGSERIAL PRIMARY KEY,
                alert_id UUID REFERENCES alerts(id) ON DELETE SET NULL,
                ticker TEXT,
                reason TEXT,
                triggered_at TIMESTAMPTZ DEFAULT now()
            );
            """)
        conn.commit()
    finally:
        conn.close()

# ───────────────────────── WATCHLIST ─────────────────────────

def wl_load() -> dict:
    if WATCHLIST_FILE.exists():
        return json.loads(WATCHLIST_FILE.read_text())
    return {}

def wl_save(data: dict):
    WATCHLIST_FILE.write_text(json.dumps(data, indent=2))

def wl_get(user_id: str) -> list[str]:
    return wl_load().get(user_id, [])

def wl_add(user_id: str, ticker: str) -> bool:
    data   = wl_load()
    tickers = data.setdefault(user_id, [])
    ticker = ticker.upper()
    if ticker in tickers:
        return False
    tickers.append(ticker)
    wl_save(data)
    return True

def wl_remove(user_id: str, ticker: str) -> bool:
    data   = wl_load()
    tickers = data.get(user_id, [])
    ticker = ticker.upper()
    if ticker not in tickers:
        return False
    tickers.remove(ticker)
    wl_save(data)
    return True

# ───────────────────────── PORTFOLIO ─────────────────────────
# Stored as: { user_id: { ticker: { shares: float, avg_cost: float } } }

def pf_load() -> dict:
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text())
    return {}

def pf_save(data: dict):
    PORTFOLIO_FILE.write_text(json.dumps(data, indent=2))

def pf_get(user_id: str) -> dict:
    return pf_load().get(user_id, {})

def pf_add(user_id: str, ticker: str, shares: float, cost: float):
    data   = pf_load()
    user   = data.setdefault(user_id, {})
    ticker = ticker.upper()
    if ticker in user:
        existing      = user[ticker]
        total_shares  = existing["shares"] + shares
        existing["avg_cost"] = (
            (existing["avg_cost"] * existing["shares"]) + (cost * shares)
        ) / total_shares
        existing["shares"] = total_shares
    else:
        user[ticker] = {"shares": shares, "avg_cost": cost}
    pf_save(data)

def pf_remove(user_id: str, ticker: str) -> bool:
    data   = pf_load()
    user   = data.get(user_id, {})
    ticker = ticker.upper()
    if ticker not in user:
        return False
    del user[ticker]
    pf_save(data)
    return True

# ───────────────────────── INDICATORS ─────────────────────────

def sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return round(sum(closes[-period:]) / period, 4)

def ema(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    k   = 2 / (period + 1)
    val = sum(closes[:period]) / period
    for price in closes[period:]:
        val = price * k + val * (1 - k)
    return round(val, 4)

def macd(closes: list[float], fast=12, slow=26, signal=9) -> dict | None:
    e_fast = ema(closes, fast)
    e_slow = ema(closes, slow)
    if e_fast is None or e_slow is None:
        return None
    macd_line = e_fast - e_slow
    if len(closes) < slow + signal:
        return {"macd": round(macd_line, 4), "signal": None, "hist": None}
    macd_history = []
    for i in range(slow - 1, len(closes)):
        ef = ema(closes[: i + 1], fast)
        es = ema(closes[: i + 1], slow)
        if ef and es:
            macd_history.append(ef - es)
    signal_line = ema(macd_history, signal)
    histogram   = round(macd_line - signal_line, 4) if signal_line else None
    return {
        "macd":   round(macd_line, 4),
        "signal": round(signal_line, 4) if signal_line else None,
        "hist":   histogram,
    }

def bollinger(closes: list[float], period=20, num_std=2) -> dict | None:
    if len(closes) < period:
        return None
    window = closes[-period:]
    mid    = sum(window) / period
    std    = statistics.stdev(window)
    return {
        "upper": round(mid + num_std * std, 4),
        "mid":   round(mid, 4),
        "lower": round(mid - num_std * std, 4),
    }

def rsi(closes: list[float], period=14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

# ───────────────────────── DATA FETCH ─────────────────────────

def fetch(ticker: str) -> dict:
    t       = yf.Ticker(ticker.upper())
    h       = t.history(period="60d")
    closes  = h["Close"].tolist()
    volumes = h["Volume"].tolist()

    price    = closes[-1]
    prev     = closes[-2]
    pct      = ((price - prev) / prev) * 100
    avg_vol  = sum(volumes[-20:]) / 20
    vol_spike = (volumes[-1] / avg_vol) if avg_vol > 0 else 1.0

    return {
        "ticker":     ticker.upper(),
        "price":      price,
        "pct":        pct,
        "rsi":        rsi(closes),
        "macd":       macd(closes),
        "bollinger":  bollinger(closes),
        "sma20":      sma(closes, 20),
        "sma50":      sma(closes, 50),
        "ema12":      ema(closes, 12),
        "volume":     volumes[-1],
        "avg_volume": avg_vol,
        "vol_spike":  vol_spike,
        "closes":     closes,
    }

# ───────────────────────── MONTE CARLO ─────────────────────────

def monte_carlo(closes: list[float], days: int = 252, sims: int = 2000) -> dict:
    """
    Vectorised GBM Monte Carlo using numpy.
    Returns percentile stats AND the full paths array for charting.
    """
    log_returns = np.diff(np.log(closes))
    mu          = float(log_returns.mean())
    sigma       = float(log_returns.std())
    start       = closes[-1]

    rng    = np.random.default_rng()
    shocks = rng.normal(mu, sigma, size=(sims, days))

    # Build paths: shape (sims, days+1)
    paths         = np.empty((sims, days + 1))
    paths[:, 0]   = start
    paths[:, 1:]  = start * np.exp(np.cumsum(shocks, axis=1))

    final     = paths[:, -1]
    loss_prob = float(np.mean(final < start) * 100)

    return {
        "paths":     paths,
        "start":     start,
        "p10":       round(float(np.percentile(final, 10)), 2),
        "p25":       round(float(np.percentile(final, 25)), 2),
        "p50":       round(float(np.percentile(final, 50)), 2),
        "p75":       round(float(np.percentile(final, 75)), 2),
        "p90":       round(float(np.percentile(final, 90)), 2),
        "loss_prob": round(loss_prob, 1),
    }


def generate_mc_chart(mc: dict, ticker: str) -> io.BytesIO:
    """
    Renders a fan-chart (P10–P90 band) for the Monte Carlo result
    and returns a PNG BytesIO buffer ready to send to Discord.
    """
    paths     = mc["paths"]            # (sims, days+1)
    start     = mc["start"]
    days_x    = np.arange(paths.shape[1])

    p10  = np.percentile(paths, 10, axis=0)
    p25  = np.percentile(paths, 25, axis=0)
    p50  = np.percentile(paths, 50, axis=0)
    p75  = np.percentile(paths, 75, axis=0)
    p90  = np.percentile(paths, 90, axis=0)
    mean = paths.mean(axis=0)

    # ── Palette ──
    bg      = "#0d1117"
    purple  = "#b47fff"
    mean_c  = "#ddb4ff"
    cur_c   = "#ff7864"

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=bg)
    ax.set_facecolor(bg)

    # Shaded confidence bands
    ax.fill_between(days_x, p10, p90, color=purple, alpha=0.13, linewidth=0)
    ax.fill_between(days_x, p25, p75, color=purple, alpha=0.22, linewidth=0)

    # Lines
    ax.plot(days_x, p50,  color=purple, linewidth=1.4, alpha=0.85, label="Median")
    ax.plot(days_x, mean, color=mean_c, linewidth=2.0,              label="Mean")
    ax.axhline(start, color=cur_c, linewidth=1.4, linestyle="--",   label="Current")

    # Spine / tick styling
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#888", labelsize=9)
    ax.set_xlabel("Days",      color="#888", fontsize=10)
    ax.set_ylabel("Price ($)", color="#888", fontsize=10)
    ax.grid(color="#ffffff", alpha=0.04, linewidth=0.5)

    # Legend
    legend_handles = [
        mpatches.Patch(color=mean_c,              label="Mean"),
        mpatches.Patch(color=purple, alpha=0.35,  label="P10–P90 band"),
        mpatches.Patch(color=purple, alpha=0.55,  label="P25–P75 band"),
        mpatches.Patch(color=cur_c,               label="Current price"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        framealpha=0,
        labelcolor="#aaaaaa",
        fontsize=9,
    )

    ax.set_title(
        f"{ticker} — Monte Carlo ({paths.shape[0]:,} sims, 1 yr)",
        color="#cccccc", fontsize=11, pad=10,
    )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    return buf

# ───────────────────────── ALERT HELPERS ─────────────────────────

def db_add_alert(ticker: str, condition_type: str, threshold: float,
                 user_id: str, channel_id: str) -> str:
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO alerts (ticker, condition_type, threshold,
                   discord_user_id, discord_channel_id)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (ticker, condition_type, threshold, user_id, channel_id),
            )
            row = cur.fetchone()
        conn.commit()
        return str(row["id"])
    finally:
        conn.close()

def db_remove_alert(alert_id: str, user_id: str) -> bool:
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alerts SET active=false WHERE id=%s AND discord_user_id=%s",
                (alert_id, user_id),
            )
            affected = cur.rowcount
        conn.commit()
        return affected > 0
    finally:
        conn.close()

def db_list_alerts(user_id: str) -> list:
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM alerts WHERE discord_user_id=%s AND active=true ORDER BY created_at DESC",
                (user_id,),
            )
            return cur.fetchall()
    finally:
        conn.close()

def db_log_trigger(alert_id, ticker: str, reason: str):
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO trigger_history (alert_id, ticker, reason) VALUES (%s, %s, %s)",
                (alert_id, ticker, reason),
            )
        conn.commit()
    finally:
        conn.close()

def db_recent_triggers(limit=10) -> list:
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM trigger_history ORDER BY triggered_at DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()

# ───────────────────────── ALERT LOOP ─────────────────────────

CONDITION_CHECKS = {
    "price_above":  lambda d, t: (d["price"] > t,  f"price ${d['price']:.2f} > ${t}"),
    "price_below":  lambda d, t: (d["price"] < t,  f"price ${d['price']:.2f} < ${t}"),
    "pct_drop":     lambda d, t: (d["pct"] < -t,   f"dropped {d['pct']:.2f}% (threshold -{t}%)"),
    "pct_gain":     lambda d, t: (d["pct"] > t,    f"gained {d['pct']:.2f}% (threshold +{t}%)"),
    "rsi_above":    lambda d, t: (d["rsi"] is not None and d["rsi"] > t,
                                  f"RSI {d['rsi']} > {t} (overbought)"),
    "rsi_below":    lambda d, t: (d["rsi"] is not None and d["rsi"] < t,
                                  f"RSI {d['rsi']} < {t} (oversold)"),
    "volume_spike": lambda d, t: (d["vol_spike"] > t,
                                  f"volume spike {d['vol_spike']:.1f}x avg (threshold {t}x)"),
}

@tasks.loop(minutes=5)
async def alert_loop():
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM alerts WHERE active=true")
            alerts = cur.fetchall()
    finally:
        conn.close()

    if not alerts:
        return

    tickers = list(set(a["ticker"] for a in alerts))
    data    = {}
    for t in tickers:
        try:
            data[t] = fetch(t)
        except Exception as e:
            log.warning(f"Failed to fetch {t}: {e}")

    for a in alerts:
        d     = data.get(a["ticker"])
        if not d:
            continue
        check = CONDITION_CHECKS.get(a["condition_type"])
        if not check:
            continue

        triggered, reason = check(d, float(a["threshold"]))
        if not triggered:
            continue

        channel = client.get_channel(int(a["discord_channel_id"]))
        if channel:
            await channel.send(
                f"🚨 **{a['ticker']}** alert triggered for <@{a['discord_user_id']}>: {reason}"
            )

        db_log_trigger(a["id"], a["ticker"], reason)

        conn = db()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE alerts SET active=false WHERE id=%s", (a["id"],))
            conn.commit()
        finally:
            conn.close()

# ───────────────────────── DAILY DIGEST ─────────────────────────

@tasks.loop(hours=24)
async def daily_digest():
    channel = client.get_channel(ALERT_CHANNEL_ID)
    if not channel:
        return

    all_watchlists = wl_load()
    all_tickers    = list(set(t for tickers in all_watchlists.values() for t in tickers))
    if not all_tickers:
        return

    lines = ["📊 **Daily Market Digest**\n"]
    for ticker in sorted(all_tickers):
        try:
            d     = fetch(ticker)
            arrow = "📈" if d["pct"] >= 0 else "📉"
            lines.append(
                f"{arrow} **{ticker}** ${d['price']:.2f} ({d['pct']:+.2f}%) | "
                f"RSI {d['rsi']} | Vol spike {d['vol_spike']:.1f}x"
            )
        except Exception:
            lines.append(f"⚠️ {ticker}: fetch error")

    await channel.send("\n".join(lines))

# ───────────────────────── SLASH COMMANDS ─────────────────────────

# ── /ping ──

@tree.command(name="ping", description="Check bot latency")
async def ping(i: discord.Interaction):
    await i.response.send_message(
        f"🏓 Pong! Latency: {round(client.latency * 1000)}ms", ephemeral=True
    )

# ── /price ──

@tree.command(name="price", description="Get current price and basic indicators for a ticker")
async def price(i: discord.Interaction, ticker: str):
    await i.response.defer()
    try:
        d = fetch(ticker)
    except Exception as e:
        await i.followup.send(f"❌ Could not fetch `{ticker.upper()}`: {e}")
        return

    arrow = "📈" if d["pct"] >= 0 else "📉"
    bb    = d["bollinger"] or {}
    m     = d["macd"]      or {}

    msg = (
        f"{arrow} **{d['ticker']}** — ${d['price']:.2f} ({d['pct']:+.2f}%)\n"
        f"```\n"
        f"RSI(14)   : {d['rsi']}\n"
        f"SMA20     : {d['sma20']}\n"
        f"SMA50     : {d['sma50']}\n"
        f"EMA12     : {d['ema12']}\n"
        f"MACD      : {m.get('macd')}  Signal: {m.get('signal')}  Hist: {m.get('hist')}\n"
        f"BB Upper  : {bb.get('upper')}\n"
        f"BB Mid    : {bb.get('mid')}\n"
        f"BB Lower  : {bb.get('lower')}\n"
        f"Vol Spike : {d['vol_spike']:.2f}x avg\n"
        f"```"
    )
    await i.followup.send(msg)

# ── /simulate ──

@tree.command(name="simulate", description="Run a Monte Carlo price simulation (1 year, 2000 paths)")
async def simulate(i: discord.Interaction, ticker: str):
    await i.response.defer()
    try:
        d  = fetch(ticker)
        mc = monte_carlo(d["closes"])
    except Exception as e:
        await i.followup.send(f"❌ Simulation failed for `{ticker.upper()}`: {e}")
        return

    start = d["price"]
    text  = (
        f"🎲 **{d['ticker']}** Monte Carlo (1 yr, 2,000 sims) — start: **${start:.2f}**\n"
        f"```\n"
        f"P10  (bear): ${mc['p10']}\n"
        f"P25        : ${mc['p25']}\n"
        f"P50 (base) : ${mc['p50']}\n"
        f"P75        : ${mc['p75']}\n"
        f"P90  (bull): ${mc['p90']}\n"
        f"Loss prob  : {mc['loss_prob']}%\n"
        f"```"
    )

    chart_buf = generate_mc_chart(mc, d["ticker"])
    await i.followup.send(
        text,
        file=discord.File(chart_buf, filename=f"{d['ticker']}_montecarlo.png"),
    )

# ── /watchlist ──

wl_group = app_commands.Group(name="watchlist", description="Manage your watchlist")

@wl_group.command(name="add", description="Add a ticker to your watchlist")
async def wl_add_cmd(i: discord.Interaction, ticker: str):
    uid   = str(i.user.id)
    added = wl_add(uid, ticker)
    if added:
        await i.response.send_message(f"✅ Added **{ticker.upper()}** to your watchlist.", ephemeral=True)
    else:
        await i.response.send_message(f"⚠️ **{ticker.upper()}** is already in your watchlist.", ephemeral=True)

@wl_group.command(name="remove", description="Remove a ticker from your watchlist")
async def wl_remove_cmd(i: discord.Interaction, ticker: str):
    uid     = str(i.user.id)
    removed = wl_remove(uid, ticker)
    if removed:
        await i.response.send_message(f"🗑️ Removed **{ticker.upper()}** from your watchlist.", ephemeral=True)
    else:
        await i.response.send_message(f"⚠️ **{ticker.upper()}** wasn't in your watchlist.", ephemeral=True)

@wl_group.command(name="show", description="Show your watchlist with live prices")
async def wl_show_cmd(i: discord.Interaction):
    await i.response.defer(ephemeral=True)
    uid     = str(i.user.id)
    tickers = wl_get(uid)
    if not tickers:
        await i.followup.send("Your watchlist is empty. Use `/watchlist add <ticker>` to start.")
        return

    lines = [f"📋 **{i.user.display_name}'s Watchlist**\n```"]
    for t in tickers:
        try:
            d     = fetch(t)
            arrow = "▲" if d["pct"] >= 0 else "▼"
            lines.append(f"{t:<8} ${d['price']:>10.2f}  {arrow} {d['pct']:+.2f}%  RSI {d['rsi']}")
        except Exception:
            lines.append(f"{t:<8} — fetch error")
    lines.append("```")
    await i.followup.send("\n".join(lines))

tree.add_command(wl_group)

# ── /alert ──

alert_group = app_commands.Group(name="alert", description="Manage price alerts")

CONDITION_CHOICES = [
    app_commands.Choice(name="Price above",    value="price_above"),
    app_commands.Choice(name="Price below",    value="price_below"),
    app_commands.Choice(name="% Drop ≥",       value="pct_drop"),
    app_commands.Choice(name="% Gain ≥",       value="pct_gain"),
    app_commands.Choice(name="RSI above",      value="rsi_above"),
    app_commands.Choice(name="RSI below",      value="rsi_below"),
    app_commands.Choice(name="Volume spike ≥", value="volume_spike"),
]

@alert_group.command(name="add", description="Create a new alert")
@app_commands.choices(condition=CONDITION_CHOICES)
async def alert_add(i: discord.Interaction, ticker: str,
                    condition: app_commands.Choice[str], threshold: float):
    uid      = str(i.user.id)
    cid      = str(i.channel_id)
    alert_id = db_add_alert(ticker.upper(), condition.value, threshold, uid, cid)
    await i.response.send_message(
        f"✅ Alert created! `{ticker.upper()}` — {condition.name} **{threshold}**\nID: `{alert_id}`",
        ephemeral=True,
    )

@alert_group.command(name="remove", description="Remove an alert by ID")
async def alert_remove(i: discord.Interaction, alert_id: str):
    uid     = str(i.user.id)
    removed = db_remove_alert(alert_id, uid)
    if removed:
        await i.response.send_message(f"🗑️ Alert `{alert_id}` removed.", ephemeral=True)
    else:
        await i.response.send_message(f"⚠️ Alert not found or not yours.", ephemeral=True)

@alert_group.command(name="list", description="List your active alerts")
async def alert_list(i: discord.Interaction):
    uid    = str(i.user.id)
    alerts = db_list_alerts(uid)
    if not alerts:
        await i.response.send_message("You have no active alerts.", ephemeral=True)
        return

    lines = ["📢 **Your Active Alerts**\n```"]
    for a in alerts:
        lines.append(
            f"{str(a['id'])[:8]}…  {a['ticker']:<6}  {a['condition_type']:<14}  @ {a['threshold']}"
        )
    lines.append("```")
    await i.response.send_message("\n".join(lines), ephemeral=True)

tree.add_command(alert_group)

# ── /portfolio ──

pf_group = app_commands.Group(name="portfolio", description="Track your stock portfolio")

@pf_group.command(name="add", description="Add a position to your portfolio")
async def pf_add_cmd(i: discord.Interaction, ticker: str, shares: float, avg_cost: float):
    pf_add(str(i.user.id), ticker, shares, avg_cost)
    await i.response.send_message(
        f"✅ Added **{shares} shares of {ticker.upper()}** at avg cost ${avg_cost:.2f}.",
        ephemeral=True,
    )

@pf_group.command(name="remove", description="Remove a position from your portfolio")
async def pf_remove_cmd(i: discord.Interaction, ticker: str):
    removed = pf_remove(str(i.user.id), ticker)
    if removed:
        await i.response.send_message(f"🗑️ Removed **{ticker.upper()}** from your portfolio.", ephemeral=True)
    else:
        await i.response.send_message(f"⚠️ **{ticker.upper()}** not found in your portfolio.", ephemeral=True)

@pf_group.command(name="show", description="Show your portfolio with P&L")
async def pf_show_cmd(i: discord.Interaction):
    await i.response.defer(ephemeral=True)
    uid       = str(i.user.id)
    positions = pf_get(uid)
    if not positions:
        await i.followup.send("Your portfolio is empty. Use `/portfolio add` to get started.")
        return

    lines       = [f"💼 **{i.user.display_name}'s Portfolio**\n```"]
    total_value = 0.0
    total_cost  = 0.0

    for ticker, pos in positions.items():
        try:
            d       = fetch(ticker)
            cur_val = d["price"] * pos["shares"]
            cost_val = pos["avg_cost"] * pos["shares"]
            pnl     = cur_val - cost_val
            pnl_pct = (pnl / cost_val) * 100 if cost_val > 0 else 0
            total_value += cur_val
            total_cost  += cost_val
            sign = "+" if pnl >= 0 else ""
            lines.append(
                f"{ticker:<6}  {pos['shares']:>8.2f} sh  "
                f"${d['price']:>8.2f}  "
                f"P&L: {sign}${pnl:>8.2f} ({sign}{pnl_pct:.1f}%)"
            )
        except Exception:
            lines.append(f"{ticker:<6}  — fetch error")

    total_pnl     = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    sign = "+" if total_pnl >= 0 else ""
    lines.append(
        f"\n{'TOTAL':<6}  {'':>8}    ${total_value:>8.2f}  "
        f"P&L: {sign}${total_pnl:.2f} ({sign}{total_pnl_pct:.1f}%)"
    )
    lines.append("```")
    await i.followup.send("\n".join(lines))

tree.add_command(pf_group)

# ── /leaderboard ──

@tree.command(name="leaderboard", description="Show top movers across all watched tickers today")
async def leaderboard(i: discord.Interaction):
    await i.response.defer()
    all_watchlists = wl_load()
    all_tickers    = list(set(t for tickers in all_watchlists.values() for t in tickers))
    if not all_tickers:
        await i.followup.send("No tickers are being watched yet.")
        return

    results = []
    for ticker in all_tickers:
        try:
            d = fetch(ticker)
            results.append((ticker, d["price"], d["pct"]))
        except Exception:
            pass

    results.sort(key=lambda x: x[2], reverse=True)

    lines = ["🏆 **Top Movers Today**\n```"]
    for rank, (ticker, px, pct) in enumerate(results, 1):
        arrow = "▲" if pct >= 0 else "▼"
        lines.append(f"#{rank:<3} {ticker:<8} ${px:>10.2f}  {arrow} {pct:+.2f}%")
    lines.append("```")
    await i.followup.send("\n".join(lines))

# ── /logs ──

@tree.command(name="logs", description="Show recent alert trigger history")
async def logs_cmd(i: discord.Interaction):
    rows = db_recent_triggers(limit=10)
    if not rows:
        await i.response.send_message("No alerts have triggered yet.", ephemeral=True)
        return

    lines = ["📜 **Recent Alert Triggers**\n```"]
    for row in rows:
        ts = row["triggered_at"].strftime("%m-%d %H:%M")
        lines.append(f"{ts}  {row['ticker']:<6}  {row['reason']}")
    lines.append("```")
    await i.response.send_message("\n".join(lines), ephemeral=True)

# ───────────────────────── SYNC ─────────────────────────

DEV_GUILD_ID = int(os.environ["DEV_GUILD_ID"]) if os.environ.get("DEV_GUILD_ID") else None

async def sync_commands():
    try:
        if DEV_GUILD_ID:
            guild  = discord.Object(id=DEV_GUILD_ID)
            tree.copy_global_to(guild=guild)
            synced = await tree.sync(guild=guild)
            log.info(f"[DEV] Synced {len(synced)} commands to guild {DEV_GUILD_ID}")
        else:
            synced = await tree.sync()
            log.info(f"[PROD] Synced {len(synced)} commands globally")
        log.info(f"Registered commands: {[c.name for c in synced]}")
    except discord.HTTPException as e:
        log.error(f"Sync failed (HTTP {e.status}): {e.text}")
    except Exception as e:
        log.error(f"Sync failed: {e}")

# ───────────────────────── EVENTS ─────────────────────────

@client.event
async def on_ready():
    log.info(f"Logged in as {client.user} (id: {client.user.id})")
    await sync_commands()
    if not alert_loop.is_running():
        alert_loop.start()
    if ALERT_CHANNEL_ID and not daily_digest.is_running():
        daily_digest.start()

# ───────────────────────── FASTAPI RUNNER ─────────────────────────

async def run_api():
    config = uvicorn.Config(api, host="0.0.0.0", port=PORT)
    server = uvicorn.Server(config)
    await server.serve()

# ───────────────────────── MAIN ─────────────────────────

async def main():
    db_init()

    registered = {c.name for c in tree.get_commands()}
    expected   = {"ping", "price", "simulate", "watchlist", "alert", "portfolio",
                  "leaderboard", "logs"}
    missing    = expected - registered
    if missing:
        raise RuntimeError(f"Commands not registered with tree: {missing}")
    log.info(f"Pre-flight OK — {len(registered)} commands ready to sync")

    await asyncio.gather(
        client.start(TOKEN),
        run_api(),
    )

if __name__ == "__main__":
    asyncio.run(main())
