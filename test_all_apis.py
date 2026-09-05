import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import time
import asyncio
from dotenv import load_dotenv
load_dotenv("/Users/gyanvardhan/Downloads/riaAIVoice-main/.env")

import httpx
from livekit import api

async def test_livekit():
    print("--- 1. Testing LiveKit Cloud Server Credentials ---")
    url = os.getenv("LIVEKIT_URL")
    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")
    try:
        lk_api = api.LiveKitAPI(url=url, api_key=key, api_secret=secret)
        rooms = await lk_api.room.list_rooms(api.ListRoomsRequest())
        print(f"✅ LiveKit Cloud Connected! Active rooms count: {len(rooms.rooms)}")
        await lk_api.aclose()
    except Exception as e:
        print(f"❌ LiveKit Connection Error: {e}")

async def test_groq_llm():
    print("\n--- 2. Benchmarking Groq Models Latency ---")
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("⚠️ GROQ_API_KEY not found in .env")
        return

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"}
            )
            if r.status_code == 200:
                models = [m["id"] for m in r.json().get("data", [])]
                
                test_candidates = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "allam-2-7b", "qwen/qwen3.6-27b"]
                for candidate in test_candidates:
                    if candidate in models:
                        t0 = time.perf_counter()
                        cr = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                            json={
                                "model": candidate,
                                "messages": [
                                    {"role": "system", "content": "You are Ria from Roxy Automobiles. Output 1 short Telugu sentence under 10 words. No thinking tags."},
                                    {"role": "user", "content": "హలో ఎవరండీ?"}
                                ],
                                "temperature": 0.1,
                                "max_tokens": 50
                            }
                        )
                        dt = (time.perf_counter() - t0) * 1000
                        if cr.status_code == 200:
                            content = cr.json()['choices'][0]['message']['content']
                            print(f"⚡ '{candidate}' -> {dt:.0f}ms:")
                            print(f"   \"{content.strip()}\"\n")
                        else:
                            print(f"❌ '{candidate}' Error ({dt:.0f}ms): {cr.status_code} {cr.text}\n")
            else:
                print(f"❌ Failed to fetch Groq models: {r.status_code} {r.text}")
        except Exception as e:
            print(f"❌ Groq Exception: {e}")

async def test_sarvam():
    print("\n--- 3. Testing Sarvam AI (STT & TTS: bulbul:v3 with speaker 'kavya') ---")
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        print("⚠️ SARVAM_API_KEY not found in .env")
        return

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            t0 = time.perf_counter()
            r = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={"api-subscription-key": key, "Content-Type": "application/json"},
                json={
                    "inputs": ["నమస్కారం, నా పేరు రియా."],
                    "target_language_code": "te-IN",
                    "speaker": "kavya",
                    "model": "bulbul:v3"
                }
            )
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                audio_len = len(r.json().get("audios", [""])[0])
                print(f"✅ Sarvam TTS ({dt:.0f}ms): Audio payload {audio_len} chars")
            else:
                print(f"❌ Sarvam TTS Error: {r.status_code} {r.text}")
        except Exception as e:
            print(f"❌ Sarvam TTS Exception: {e}")

async def main():
    await test_livekit()
    await test_groq_llm()
    await test_sarvam()

if __name__ == "__main__":
    asyncio.run(main())
