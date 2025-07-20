from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import logging
import uuid
from queue.tasks import process_document, process_document_batch, refresh_embeddings
from celery.result import AsyncResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/documents", tags=["Document Processing"])

# Directory to save uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    process_now: bool = Form(False)
):
    """
    Upload a document and optionally process it immediately.
    
    Args:
        file: The file to upload
        process_now: Whether to process the document immediately
        
    Returns:
        JSON response with upload status and task ID if processing
    """
    try:
        # Create a unique filename to avoid collisions
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File uploaded successfully: {file_path}")
        
        # Process the document if requested
        if process_now:
            # Start a Celery task to process the document
            task = process_document.delay(file_path)
            return {
                "status": "success",
                "message": "File uploaded and processing started",
                "file_path": file_path,
                "original_filename": file.filename,
                "task_id": task.id
            }
        else:
            return {
                "status": "success",
                "message": "File uploaded successfully",
                "file_path": file_path,
                "original_filename": file.filename
            }
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.post("/batch-process")
async def batch_process_documents(file_paths: List[str]):
    """
    Process a batch of documents in the background.
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        JSON response with task ID
    """
    try:
        # Start a Celery task to process the documents
        task = process_document_batch.delay(file_paths)
        
        return {
            "status": "success",
            "message": f"Started processing {len(file_paths)} documents",
            "task_id": task.id
        }
        
    except Exception as e:
        logger.error(f"Error starting batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting batch processing: {str(e)}")

@router.post("/refresh-embeddings")
async def refresh_all_embeddings(directory: str = "data"):
    """
    Refresh all embeddings by reprocessing all documents.
    
    Args:
        directory: Directory containing documents to refresh
        
    Returns:
        JSON response with task ID
    """
    try:
        # Start a Celery task to refresh embeddings
        task = refresh_embeddings.delay(directory)
        
        return {
            "status": "success",
            "message": "Started refreshing embeddings",
            "task_id": task.id
        }
        
    except Exception as e:
        logger.error(f"Error starting embedding refresh: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting embedding refresh: {str(e)}")

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a task.
    
    Args:
        task_id: ID of the task to check
        
    Returns:
        JSON response with task status
    """
    try:
        # Get the task result
        task_result = AsyncResult(task_id)
        
        # Check if the task exists
        if not task_result:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        # Get the task status
        if task_result.ready():
            if task_result.successful():
                return {
                    "status": "completed",
                    "task_id": task_id,
                    "result": task_result.result
                }
            else:
                return {
                    "status": "failed",
                    "task_id": task_id,
                    "error": str(task_result.result)
                }
        else:
            return {
                "status": "in_progress",
                "task_id": task_id
            }
            
    except Exception as e:
        logger.error(f"Error checking task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking task status: {str(e)}") 