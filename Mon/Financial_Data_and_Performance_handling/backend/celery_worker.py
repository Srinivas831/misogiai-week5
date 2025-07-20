"""
Celery worker script for background tasks.

This script starts a Celery worker that processes background tasks
like document processing, embedding generation, etc.

Usage:
    python celery_worker.py
"""

from queue.celery_config import celery_app

if __name__ == '__main__':
    # Start the Celery worker
    celery_app.worker_main(['worker', '--loglevel=info']) 