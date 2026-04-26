import os
import asyncio
import json
import math
import random
import statistics
import logging
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

# ───────────────────────── CONFIG ─────────────────────────

TOKEN = os.environ["DISCORD_TOKEN"]
PORT = int(os.environ.get("PORT", 8000))
ALERT_CHANNEL_ID = int(os.environ.get("ALERT_CHANNEL_ID", 0))
WATCHLIST_FILE = Path("watchlists.json")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

# ───────────────────────── DISCORD SETUP ─────────────────────────

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

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
                alert_id UUID,
                ticker TEXT,
                reason TEXT,
                triggered_at TIMESTAMPTZ DEFAULT now()
            );
            """)
        conn.commit()
    finally:
        conn.close()

# ───────────────────────── WATCHLIST ─────────────────────────

def wl_load():
    if WATCHLIST_FILE.exists():
        return json.loads(WATCHLIST_FILE.read_text())
    return {}

def wl_save(data):
    WATCHLIST_FILE.write_text(json.dumps(data, indent=2))

# ───────────────────────── INDICATORS ─────────────────────────

def rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def fetch(ticker):
    t = yf.Ticker(ticker)
    h = t.history(period="60d")
    closes = h["Close"].tolist()

    price = closes[-1]
    prev = closes[-2]
    pct = ((price - prev) / prev) * 100

    return {
        "ticker": ticker,
        "price": price,
        "pct": pct,
        "rsi": rsi(closes),
        "closes": closes
    }

# ───────────────────────── MONTE CARLO ─────────────────────────

def monte_carlo(closes, days=252, sims=200):
    log_returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
    ]
    mu = statistics.mean(log_returns)
    sigma = statistics.stdev(log_returns)
    start = closes[-1]

    results = []

    for _ in range(sims):
        price = start
        for _ in range(days):
            price *= math.exp(mu + sigma * random.gauss(0, 1))
        results.append(price)

    results.sort()
    return {
        "p10": results[int(0.1 * sims)],
        "p50": results[int(0.5 * sims)],
        "p90": results[int(0.9 * sims)],
    }

# ───────────────────────── ALERT LOOP ─────────────────────────

@tasks.loop(minutes=5)
async def alert_loop():
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM alerts WHERE active=true")
            alerts = cur.fetchall()

        tickers = list(set(a["ticker"] for a in alerts))
        data = {t: fetch(t) for t in tickers}

        for a in alerts:
            d = data.get(a["ticker"])
            if not d:
                continue

            trigger = False
            reason = ""

            if a["condition_type"] == "price_above":
                trigger = d["price"] > float(a["threshold"])
                reason = "price above"

            if trigger:
                channel = client.get_channel(int(a["discord_channel_id"]))
                if channel:
                    await channel.send(f"🚨 {a['ticker']} triggered: {reason}")

    finally:
        conn.close()

# ───────────────────────── COMMANDS ─────────────────────────

@tree.command(name="ping")
async def ping(i): 
    await i.response.send_message("pong", ephemeral=True)

@tree.command(name="price")
async def price(i, ticker: str):
    d = fetch(ticker.upper())
    await i.response.send_message(
        f"{ticker}: ${d['price']:.2f} ({d['pct']:.2f}%)"
    )

# ───────────────────────── SYNC ─────────────────────────

async def sync():
    try:
        synced = await tree.sync()
        log.info(f"Synced {len(synced)} commands")
    except Exception as e:
        log.error(e)

# ───────────────────────── EVENTS ─────────────────────────

@client.event
async def on_ready():
    log.info(f"Logged in as {client.user}")
    await sync()
    if not alert_loop.is_running():
        alert_loop.start()

# ───────────────────────── FASTAPI RUNNER ─────────────────────────

async def run_api():
    config = uvicorn.Config(api, host="0.0.0.0", port=PORT)
    server = uvicorn.Server(config)
    await server.serve()

# ───────────────────────── MAIN ─────────────────────────

async def main():
    db_init()

    await asyncio.gather(
        client.start(TOKEN),
        run_api()
    )

if __name__ == "__main__":
    asyncio.run(main())
