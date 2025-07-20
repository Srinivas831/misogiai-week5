"""
Example script to demonstrate using the Celery queue for document processing.

This script shows how to:
1. Process a single document
2. Process multiple documents in a batch
3. Check the status of a task
4. Refresh all embeddings
"""

import os
import sys
import time

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from queue.tasks import process_document, process_document_batch, refresh_embeddings
from celery.result import AsyncResult

def process_single_document(file_path):
    """Process a single document and wait for the result."""
    print(f"Processing document: {file_path}")
    
    # Start the task
    task = process_document.delay(file_path)
    print(f"Task started with ID: {task.id}")
    
    # Wait for the task to complete
    print("Waiting for task to complete...")
    result = task.get()  # This will block until the task completes
    
    print(f"Task completed with status: {result['status']}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    
    if result['status'] == 'success':
        print(f"Processed {result['chunk_count']} chunks")
    else:
        print(f"Error: {result['message']}")
    
    return result

def process_multiple_documents(file_paths):
    """Process multiple documents in a batch."""
    print(f"Processing {len(file_paths)} documents in batch")
    
    # Start the batch task
    task = process_document_batch.delay(file_paths)
    print(f"Batch task started with ID: {task.id}")
    
    # Wait for the task to complete
    print("Waiting for batch task to complete...")
    result = task.get()  # This will block until the task completes
    
    print(f"Batch task completed with status: {result['status']}")
    print(f"Total processing time: {result['processing_time']:.2f} seconds")
    print(f"Successful: {result['successful_documents']}, Failed: {result['failed_documents']}")
    
    return result

def check_task_status(task_id):
    """Check the status of a task by ID."""
    print(f"Checking status of task: {task_id}")
    
    # Get the task result
    task_result = AsyncResult(task_id)
    
    if task_result.ready():
        if task_result.successful():
            print(f"Task completed successfully")
            print(f"Result: {task_result.result}")
        else:
            print(f"Task failed: {task_result.result}")
    else:
        print(f"Task is still in progress")
    
    return task_result

def refresh_all_embeddings(directory="data"):
    """Refresh all embeddings in the specified directory."""
    print(f"Refreshing all embeddings in directory: {directory}")
    
    # Start the refresh task
    task = refresh_embeddings.delay(directory)
    print(f"Refresh task started with ID: {task.id}")
    
    # Wait for the task to complete
    print("Waiting for refresh task to complete...")
    result = task.get()  # This will block until the task completes
    
    print(f"Refresh task completed with status: {result['status']}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    
    if result['status'] == 'success':
        print(f"Processed {result['chunk_count']} chunks")
    else:
        print(f"Error: {result['message']}")
    
    return result

if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # Check if there are any PDF files in the data directory
    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                if f.lower().endswith('.pdf')]
    
    if pdf_files:
        # Process a single document
        print("\n=== Processing Single Document ===")
        result = process_single_document(pdf_files[0])
        
        if len(pdf_files) > 1:
            # Process multiple documents
            print("\n=== Processing Multiple Documents ===")
            batch_result = process_multiple_documents(pdf_files[:3])  # Process up to 3 documents
        
        # Refresh all embeddings
        print("\n=== Refreshing All Embeddings ===")
        refresh_result = refresh_all_embeddings(data_dir)
    else:
        print(f"No PDF files found in {data_dir}")
        print("Please add some PDF files to the data directory and run this script again.") 