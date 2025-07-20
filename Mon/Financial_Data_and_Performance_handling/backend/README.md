# Financial Data RAG System

A RAG (Retrieval-Augmented Generation) system for financial data analysis with Redis caching for high-performance production use.

## Features

- Document ingestion pipeline for financial reports and statements
- Vector database storage with Pinecone
- Semantic retrieval of relevant financial information
- OpenAI-powered answer generation
- Redis caching with different TTLs for real-time vs historical data
- Rate limiting per API key
- Async API with FastAPI
- Handles 100+ concurrent requests with <2s response time
- Background job queue with Celery for document processing

## Setup

### Prerequisites

- Python 3.9+
- Redis server
- Pinecone account
- OpenAI API key

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file based on `.env.example` with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   REDIS_URL=redis://localhost:6379
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=your_pinecone_index_name
   ```

## Usage

### Ingesting Documents

Place your financial PDF documents in the `data` directory, then run:

```
python main.py
```

This will:
1. Load and split the PDF documents
2. Embed the chunks
3. Store them in Pinecone

### Running the API

Start the FastAPI server:

```
uvicorn fastapi_main:app --reload
```

The API will be available at http://localhost:8000

### Running the Celery Worker

Start the Celery worker for background processing:

```
python celery_worker.py
```

Or directly with Celery:

```
celery -A queue.celery_config worker --loglevel=info
```

### API Endpoints

#### RAG Query Endpoints

- `POST /ask`: Ask a question about financial data
  - Request body: `{"question": "What was Company X's revenue in 2022?", "api_key": "optional_api_key"}`
  - Response: `{"answer": "...", "cached": true/false, "response_time_seconds": 0.123}`

- `GET /health`: Health check endpoint
  - Response: `{"status": "healthy"}`

#### Document Processing Endpoints

- `POST /documents/upload`: Upload a document
  - Form data: `file` (file), `process_now` (boolean)
  - Response: `{"status": "success", "file_path": "...", "task_id": "..."}`

- `POST /documents/batch-process`: Process multiple documents
  - Request body: `{"file_paths": ["path/to/doc1.pdf", "path/to/doc2.pdf"]}`
  - Response: `{"status": "success", "task_id": "..."}`

- `POST /documents/refresh-embeddings`: Refresh all embeddings
  - Request body: `{"directory": "data"}`
  - Response: `{"status": "success", "task_id": "..."}`

- `GET /documents/task/{task_id}`: Check task status
  - Response: `{"status": "completed|in_progress|failed", "task_id": "...", "result": {...}}`

## Architecture

- **FastAPI**: Async API framework
- **Redis**: Caching layer with TTL (1h for real-time, 24h for historical data)
- **Pinecone**: Vector database for embeddings
- **LangChain**: Orchestration of the RAG pipeline
- **OpenAI**: LLM for answer generation
- **Celery**: Background job queue for document processing

## Performance Considerations

- Redis caching aims for >70% cache hit ratio
- Rate limiting is set to 100 requests per minute per API key
- Response time is monitored and logged if it exceeds 2 seconds
- Background processing with Celery for computationally intensive tasks
- Error handling for Redis connection issues ensures the system degrades gracefully 