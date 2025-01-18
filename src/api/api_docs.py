from typing import Dict, List
from pydantic import BaseModel, Field


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""
    query: str = Field(..., description="The query to search for in the knowledge base")


class QueryResponse(BaseModel):
    """Response model for the query endpoint."""
    response: str = Field(..., description="The AI-generated response")
    sources: List[Dict] = Field(..., description="List of source documents used")


class DriveFileRequest(BaseModel):
    """Request model for Drive file ingestion."""
    folder_id: str = Field(..., description="Google Drive folder ID to process")


class DriveIngestionResponse(BaseModel):
    """Response model for Drive ingestion endpoints."""
    status: str = Field(..., description="Status of the ingestion process")
    message: str = Field(..., description="Detailed status message")
    files_processed: List[str] = Field(..., description="List of processed file paths")


class TrackingStatusResponse(BaseModel):
    """Response model for the tracking status endpoint."""
    status: str = Field(..., description="Overall tracking status")
    total_files: int = Field(..., description="Total number of files tracked")
    vectorized_files: int = Field(..., description="Number of successfully vectorized files")
    failed_files: int = Field(..., description="Number of failed files")
    vectorized_paths: List[str] = Field(..., description="Paths of vectorized files")
    failed_paths: List[str] = Field(..., description="Paths of failed files")


# API Routes Documentation
API_ROUTES = {
    "Query Knowledge Base": {
        "path": "/query",
        "method": "POST",
        "description": "Query the knowledge base using Retrieval Augmented Generation (RAG).",
        "request_model": QueryRequest,
        "response_model": QueryResponse,
        "example_request": {
            "query": "What is the company's vacation policy?"
        },
        "example_response": {
            "response": "According to the policy...",
            "sources": [{"source": "HR-Policy-2024.pdf", "page": 5}]
        }
    },
    "Ingest Drive Files": {
        "path": "/ingest/drive/files",
        "method": "POST",
        "description": "Ingest and vectorize files from a Google Drive folder.",
        "request_model": DriveFileRequest,
        "response_model": DriveIngestionResponse,
        "example_request": {
            "folder_id": "1234567890abcdef"
        },
        "example_response": {
            "status": "success",
            "message": "Successfully processed 3 files",
            "files_processed": [
                "doc1.pdf",
                "doc2.docx",
                "subfolder/doc3.txt"
            ]
        }
    },
    "Force Reingest Drive Files": {
        "path": "/ingest/drive/files/force",
        "method": "POST",
        "description": "Force re-ingestion of files from Google Drive, ignoring previous vectorization status.",
        "request_model": DriveFileRequest,
        "response_model": DriveIngestionResponse,
        "example_request": {
            "folder_id": "1234567890abcdef"
        },
        "example_response": {
            "status": "success",
            "message": "Successfully processed 3 files",
            "files_processed": [
                "doc1.pdf",
                "doc2.docx",
                "doc3.txt"
            ]
        }
    },
    "Get Tracking Status": {
        "path": "/tracking/status",
        "method": "GET",
        "description": "Get document processing and vectorization status summary.",
        "response_model": TrackingStatusResponse,
        "example_response": {
            "status": "success",
            "total_files": 5,
            "vectorized_files": 3,
            "failed_files": 1,
            "vectorized_paths": [
                "doc1.pdf",
                "doc2.docx",
                "doc3.txt"
            ],
            "failed_paths": [
                "error.xlsx"
            ]
        }
    }
}

# Code Examples
CODE_EXAMPLES = {
    "curl": {
        "Query": """
curl -X POST "http://localhost:8000/query" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "What is the company\'s vacation policy?"}'
""",
        "Ingest": """
curl -X POST "http://localhost:8000/ingest/drive/files" \\
     -H "Content-Type: application/json" \\
     -d '{"folder_id": "1234567890abcdef"}'
""",
        "Force Reingest": """
curl -X POST "http://localhost:8000/ingest/drive/files/force" \\
     -H "Content-Type: application/json" \\
     -d '{"folder_id": "1234567890abcdef"}'
""",
        "Get Status": """
curl "http://localhost:8000/tracking/status"
"""
    },
    "python": """
import requests
from typing import Dict, Optional

class KnowledgeHubClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def query_knowledge_base(self, query: str) -> Dict:
        \"\"\"Query the knowledge base.\"\"\"
        response = requests.post(
            f"{self.base_url}/query",
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()

    def ingest_drive_files(self, folder_id: str, force: bool = False) -> Dict:
        \"\"\"Ingest files from Google Drive folder.\"\"\"
        endpoint = f"{self.base_url}/ingest/drive/files"
        if force:
            endpoint += "/force"
        response = requests.post(
            endpoint,
            json={"folder_id": folder_id}
        )
        response.raise_for_status()
        return response.json()

    def get_tracking_status(self) -> Dict:
        \"\"\"Get document processing status.\"\"\"
        response = requests.get(f"{self.base_url}/tracking/status")
        response.raise_for_status()
        return response.json()
"""
}

def get_api_documentation() -> Dict:
    """Get complete API documentation with examples."""
    routes = {
        key: {
            **value,
            "request_model": value["request_model"].schema() if "request_model" in value else None,
            "response_model": value["response_model"].schema() if "response_model" in value else None,
        }
        for key, value in API_ROUTES.items()
    }
    return {
        "title": "Enterprise Knowledge Hub API",
        "version": "1.0.0",
        "description": "API for document ingestion and querying using RAG.",
        "routes": routes,
        "code_examples": CODE_EXAMPLES,
    }


def get_route_details(route_name: str) -> Dict:
    """Get detailed documentation for a specific route."""
    if route_name not in API_ROUTES:
        raise ValueError(f"Route '{route_name}' not found in documentation")
    return API_ROUTES[route_name]
