import os
import asyncio
import logging
from datetime import datetime

import discord
from discord import app_commands
from fastapi import FastAPI
import uvicorn

# ───────────────────────── CONFIG ─────────────────────────

TOKEN = os.environ["DISCORD_TOKEN"]
PORT = int(os.environ.get("PORT", 8000))

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
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


async def run_api():
    config = uvicorn.Config(api, host="0.0.0.0", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()

# ───────────────────────── SLASH COMMAND TEST ─────────────────────────

@tree.command(name="ping", description="test bot latency")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("pong", ephemeral=True)

# ───────────────────────── SAFE SYNC ─────────────────────────

async def sync_commands():
    try:
        synced = await tree.sync()
        log.info(f"Synced {len(synced)} commands globally")
    except Exception as e:
        log.error(f"Command sync failed: {e}")

# ───────────────────────── BOT EVENTS ─────────────────────────

@client.event
async def on_ready():
    log.info(f"Logged in as {client.user}")

    # IMPORTANT: sync once bot is fully ready
    await sync_commands()

# ───────────────────────── MAIN RUNNER ─────────────────────────

async def main():
    api_task = asyncio.create_task(run_api())

    async with client:
        bot_task = asyncio.create_task(client.start(TOKEN))

        await asyncio.gather(bot_task, api_task)


if __name__ == "__main__":
    asyncio.run(main())
