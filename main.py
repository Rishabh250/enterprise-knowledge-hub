import logging
import os
from typing import Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.api.drive_routes import router as drive_router
from src.rag.rag_pipeline import get_rag_pipeline, get_doc_retriever
from src.utils.constants import APP_CONFIG, CORS_CONFIG, LANGUAGE_MAP
from src.utils.helpers import log_message

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnterpriseKnowledgeHub")

# FastAPI app configuration
app = FastAPI(**APP_CONFIG)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    **CORS_CONFIG
)

# Include routers
app.include_router(drive_router, prefix="/api/v1")

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(
        ...,
        example="What is the company's policy on remote work?",
        description="The query to be processed",
        min_length=1
    )
    language: str = Field(
        default="en",
        example="en",
        description="The language in which the response is desired"
    )

    class Config:
        """Configuration for the QueryRequest model."""
        json_schema_extra = {
            "example": {
                "query": "What is the company's policy on remote work?",
                "language": "en"
            }
        }

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint that returns a welcome message."""
    return {"message": "Welcome to the Enterprise Knowledge Hub API"}

@app.post("/api/v1/query")
async def process_query(query_body: QueryRequest) -> Dict[str, Any]:
    """Process a query request and return the response."""
    try:
        log_message(f"Received query: {query_body}")

        if not query_body.query.strip():
            raise ValueError("Query cannot be empty")

        language = query_body.language.lower()
        full_language = LANGUAGE_MAP.get(language, language)

        query = query_body.query

        log_message(f"Processing query: {query}")
        rag_pipeline = get_rag_pipeline()
        process_retrieval = get_doc_retriever()

        retrieved_docs = process_retrieval({
            "question": query,
            "language": full_language
        })

        context = retrieved_docs.get("context", "")
        sources = retrieved_docs.get("sources", [])

        chain_response = rag_pipeline.invoke({
            "question": query,
            "language": full_language,
            "context": context
            })

        return {
            "response": chain_response,
            "language": full_language,
            "sources": sources
        }

    except ValueError as ve:
        log_message(f"Validation error: {ve}", level="error")
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        log_message(f"Query processing failed: {str(e)}", level="error")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your query"
        ) from e

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
