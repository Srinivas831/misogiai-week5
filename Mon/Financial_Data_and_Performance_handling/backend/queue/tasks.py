from queue.celery_config import celery_app
import logging
import os
import time
from loaders.pdf_loader import load_pdf_files_and_split
from vectorstore.store import embed_and_store
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@celery_app.task(name="process_document")
def process_document(file_path: str) -> dict:
    """
    Process a single document: load, split, embed, and store in the vector database.
    
    Args:
        file_path: Path to the document to process
        
    Returns:
        dict: Processing results with status and metadata
    """
    start_time = time.time()
    logger.info(f"Starting to process document: {file_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {
                "status": "error",
                "message": f"File not found: {file_path}",
                "file_path": file_path,
                "processing_time": 0
            }
        
        # 1. Load and split the document
        logger.info(f"Loading and splitting document: {file_path}")
        document_chunks = load_pdf_files_and_split(file_path)
        
        # 2. Embed and store the chunks
        logger.info(f"Embedding and storing {len(document_chunks)} chunks from: {file_path}")
        embed_and_store(document_chunks)
        
        processing_time = time.time() - start_time
        logger.info(f"Document processed successfully: {file_path} in {processing_time:.2f}s")
        
        return {
            "status": "success",
            "message": "Document processed successfully",
            "file_path": file_path,
            "chunk_count": len(document_chunks),
            "processing_time": processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing document {file_path}: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing document: {str(e)}",
            "file_path": file_path,
            "processing_time": processing_time
        }

@celery_app.task(name="process_document_batch")
def process_document_batch(file_paths: List[str]) -> dict:
    """
    Process a batch of documents in parallel using Celery's task system.
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        dict: Processing results with status and metadata
    """
    start_time = time.time()
    logger.info(f"Starting to process batch of {len(file_paths)} documents")
    
    # Create a list of tasks to process each document
    tasks = [process_document.delay(file_path) for file_path in file_paths]
    
    # Wait for all tasks to complete and collect results
    results = [task.get() for task in tasks]
    
    # Count successes and failures
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "error"]
    
    processing_time = time.time() - start_time
    logger.info(f"Batch processing completed in {processing_time:.2f}s. "
                f"Successes: {len(successes)}, Failures: {len(failures)}")
    
    return {
        "status": "completed",
        "total_documents": len(file_paths),
        "successful_documents": len(successes),
        "failed_documents": len(failures),
        "processing_time": processing_time,
        "results": results
    }

@celery_app.task(name="refresh_embeddings")
def refresh_embeddings(directory: str = "data") -> dict:
    """
    Refresh all embeddings by reprocessing all documents in the data directory.
    
    Args:
        directory: Directory containing documents to refresh
        
    Returns:
        dict: Processing results
    """
    start_time = time.time()
    logger.info(f"Starting to refresh all embeddings from directory: {directory}")
    
    try:
        # 1. Load and split all documents in the directory
        all_chunks = load_pdf_files_and_split(directory)
        
        # 2. Embed and store all chunks
        embed_and_store(all_chunks)
        
        processing_time = time.time() - start_time
        logger.info(f"Embeddings refreshed successfully in {processing_time:.2f}s")
        
        return {
            "status": "success",
            "message": "Embeddings refreshed successfully",
            "directory": directory,
            "chunk_count": len(all_chunks),
            "processing_time": processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error refreshing embeddings: {str(e)}")
        return {
            "status": "error",
            "message": f"Error refreshing embeddings: {str(e)}",
            "directory": directory,
            "processing_time": processing_time
        } 