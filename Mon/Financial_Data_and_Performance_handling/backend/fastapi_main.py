from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_engine import get_answer
from cache.redis_cache import get_cached_answer, set_cached_answer
import asyncio
import time
import logging
from typing import Dict, Optional

# Import the document API router
from api.document_api import router as document_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Data RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the document API router
app.include_router(document_router)

class QueryInput(BaseModel):
    question: str
    api_key: Optional[str] = None

# Simple rate limiting implementation
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, list] = {}
    
    async def check_rate_limit(self, api_key: str = "default"):
        current_time = time.time()
        # Initialize if this is a new API key
        if api_key not in self.request_counts:
            self.request_counts[api_key] = []
        
        # Remove requests older than 1 minute
        self.request_counts[api_key] = [t for t in self.request_counts[api_key] 
                                       if current_time - t < 60]
        
        # Check if rate limit exceeded
        if len(self.request_counts[api_key]) >= self.requests_per_minute:
            return False
        
        # Add current request timestamp
        self.request_counts[api_key].append(current_time)
        return True

rate_limiter = RateLimiter(requests_per_minute=100)  # 100 requests per minute as per requirements

async def check_rate_limit(data: QueryInput):
    api_key = data.api_key if data.api_key else "default"
    if not await rate_limiter.check_rate_limit(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return data

@app.post("/ask")
async def ask_query(data: QueryInput = Depends(check_rate_limit)):
    start_time = time.time()
    question = data.question.strip().lower()
    
    try:
        # 1. Try Redis
        cached = await get_cached_answer(question)
        if cached:
            logger.info(f"Cache hit for: {question[:30]}...")
            response_time = time.time() - start_time
            return {"answer": cached["answer"], "cached": True, "response_time_seconds": response_time}

        # 2. Use RAG to generate
        logger.info(f"Cache miss, generating answer for: {question[:30]}...")
        answer = await asyncio.to_thread(get_answer, question)

        # 3. Cache it with appropriate TTL
        await set_cached_answer(question, answer)  # TTL is handled in the cache module

        response_time = time.time() - start_time
        logger.info(f"Response generated in {response_time:.2f}s")
        
        # Check if response time is within requirements
        if response_time > 2:
            logger.warning(f"Response time exceeded 2s: {response_time:.2f}s")
            
        return {"answer": answer, "cached": False, "response_time_seconds": response_time}
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
