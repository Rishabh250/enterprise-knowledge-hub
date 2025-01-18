import logging
import os
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from src.ingestion.google_drive import GoogleDriveLoader
from src.ingestion.document_loader import DocumentLoader
from src.vectorstore.vector_store import VectorStoreManager
from src.models.tracking import IngestionTracker, DocumentTrack

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tracker
tracker = IngestionTracker()

router = APIRouter()

class DriveFileRequest(BaseModel):
    """Request model for Drive file ingestion"""
    folder_id: str

class DriveIngestionResponse(BaseModel):
    """Response model for Drive ingestion endpoints"""
    status: str
    message: str
    files_processed: List[str]

@router.post("/ingest/drive/files", response_model=DriveIngestionResponse)
async def ingest_drive_files(request: DriveFileRequest = Body(...)) -> DriveIngestionResponse:
    """
    Ingest specific files from Google Drive using their IDs
    """
    drive_loader = GoogleDriveLoader()
    doc_loader = DocumentLoader()
    vector_store_manager = VectorStoreManager()
    processed_files = []
    documents = []
    folder_id = request.folder_id

    # Get list of files in folder
    try:
        files = drive_loader.list_files_in_folder(folder_id)
    except Exception as e:
        logger.error("Error listing files in folder %s: %s", folder_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to list files in folder")

    # Process each file
    for file in files:
        file_id = file.get('id')
        file_name = file.get('name')
        downloaded_paths = [] 

        try:
            # Check if file is already vectorized
            if tracker.is_file_vectorized(file_id):
                logger.info("File %s already vectorized, skipping", file_name)
                continue

            # Download and process file
            downloaded_path = drive_loader.download_single_file(file_id)
            downloaded_paths.append(downloaded_path)

            # Track initial download
            tracker.track(DocumentTrack(
                file_path=downloaded_path,
                file_id=file_id,
                status='downloaded',
                timestamp=datetime.now()
            ))

            # Load and process document
            docs = doc_loader.load_document(downloaded_path)
            documents.extend(docs)
            processed_files.append(downloaded_path)

            # Update tracking status
            tracker.update_vectorization_status(downloaded_path, success=True)

            # Delete file after vectorization
            os.remove(downloaded_path)
        except Exception as e:
            error_msg = str(e)
            logger.error("Error processing %s: %s", file_name, error_msg)
            # Only update status for files that were downloaded
            for path in downloaded_paths:
                tracker.update_vectorization_status(path, success=False, error=error_msg)

    if not processed_files:
        message = "No new files to process" if tracker.get_vectorized_file_ids() else "No files were processed successfully"
        raise HTTPException(status_code=400, detail=message)

    # Update vector store only if we have new documents
    if documents:
        try:
            vectorstore_path = os.getenv("VECTORSTORE_PATH")
            vector_store_manager.create_vectorstore(documents, vectorstore_path)
        except Exception as e:
            logger.error("Error updating vector store: %s", str(e))
            raise HTTPException(status_code=500, detail="Failed to update vector store")

    return DriveIngestionResponse(
        status="success",
        message="Successfully processed %d files" % len(processed_files),
        files_processed=processed_files
    )
