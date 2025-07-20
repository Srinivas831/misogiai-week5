from celery import Celery
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Redis URL from environment or use default
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Create Celery instance
# We're using Redis as both the message broker and result backend
# The broker is where Celery sends and receives messages
# The result backend is where task results are stored
celery_app = Celery(
    'financial_rag',  # Name of the Celery app
    broker=REDIS_URL,  # Use Redis as the message broker
    backend=REDIS_URL,  # Use Redis as the result backend
    include=[
        'queue.tasks'  # Include the tasks module where we define our tasks
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task serialization format
    task_serializer='json',
    # Result serialization format
    result_serializer='json',
    # Content types accepted
    accept_content=['json'],
    # Enable UTC timezone for consistency
    enable_utc=True,
    # Task result expiration time (in seconds)
    result_expires=3600,
    # Maximum number of tasks a worker can execute before it's replaced with a new one
    worker_max_tasks_per_child=1000,
    # Concurrency: number of worker processes/threads
    worker_concurrency=4,
    # Task time limit in seconds
    task_time_limit=600,  # 10 minutes
    # Task soft time limit in seconds (warning before hard limit)
    task_soft_time_limit=300,  # 5 minutes
)

# If this file is executed directly, print some information
if __name__ == '__main__':
    print(f"Celery app '{celery_app.main}' configured with broker: {celery_app.conf.broker_url}") 