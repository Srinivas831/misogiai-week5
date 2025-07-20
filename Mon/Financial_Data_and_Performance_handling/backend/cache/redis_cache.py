import redis.asyncio as redis_async
import os
from dotenv import load_dotenv
import json

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis = redis_async.from_url(REDIS_URL, decode_responses=True)

async def get_cached_answer(query: str):
    print("query",query)
    print("get_cached_answer")
    result = await redis.get(query)
    if result:
        return json.loads(result)
    return None

async def set_cached_answer(query: str, answer: str, ttl_seconds: int = 3600):
    print("query",query)
    print("set_cached_answer")
    await redis.set(query, json.dumps({"answer": answer}), ex=ttl_seconds)