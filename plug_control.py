import asyncio
import os
from dotenv import load_dotenv
from tapo import ApiClient

load_dotenv('/home/nvidia/.env')

EMAIL = os.getenv("TAPO_EMAIL")
PASSWORD = os.getenv("TAPO_PASSWORD")
IP = os.getenv("TAPO_PLUG_IP")

async def plug_on():
    client = ApiClient(EMAIL, PASSWORD)
    device = await client.p100(IP)
    await device.on()
    print("Plug turned ON ✅")

async def plug_off():
    client = ApiClient(EMAIL, PASSWORD)
    device = await client.p100(IP)
    await device.off()
    print("Plug turned OFF ✅")

def turn_on_plug():
    asyncio.run(plug_on())

def turn_off_plug():
    asyncio.run(plug_off())
