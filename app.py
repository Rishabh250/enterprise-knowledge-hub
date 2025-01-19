import logging
import os
from functools import lru_cache
from typing import Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
import uvicorn
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from src.api.drive_routes import router as drive_router
from src.utils.constants import PROMPT_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnterpriseKnowledgeHub")

# FastAPI app configuration
app = FastAPI(
    title="Enterprise Knowledge Hub",
    description="API for document ingestion and querying using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(drive_router, prefix="/api/v1")

def get_env_var(var_name: str) -> str:
    """Helper function to get environment variables with error handling"""
    value = os.getenv(var_name)
    if not value:
        logger.error(f"{var_name} environment variable not set")
        raise ValueError(f"{var_name} environment variable not set")
    return value

@lru_cache()
def get_llm() -> OllamaLLM:
    """Initialize and return the LLM model."""
    return OllamaLLM(model=get_env_var("MODEL"))

@lru_cache()
def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize and return the embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@lru_cache()
def get_vectorstore() -> FAISS:
    """Initialize and return the vector store."""
    vectorstore_path = get_env_var("VECTORSTORE_PATH")

    if not os.path.exists(vectorstore_path):
        logger.error(f"Vector store not found at {vectorstore_path}")
        raise FileNotFoundError(f"Vector store not found at {vectorstore_path}")

    return FAISS.load_local(
        vectorstore_path,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

@lru_cache()
def get_rag_pipeline():
    """Initialize and return the RAG pipeline."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question", "language"])
    llm = get_llm()
    output_parser = StrOutputParser()

    # Create a parallel runnable for input preparation
    input_preparation = RunnableParallel(
        question=lambda x: x["question"],
        context=lambda x: retriever.get_relevant_documents(x["question"]),
        language=lambda x: x["language"]
    )

    # Chain the components
    rag_chain = input_preparation | prompt | llm | output_parser

    return rag_chain

# Language mapping with type hints
LANGUAGE_MAP: Dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "hi": "Hindi"
}

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
        logger.info("Received query: %s", query_body)

        if not query_body.query.strip():
            raise ValueError("Query cannot be empty")

        language = query_body.language.lower()
        full_language = LANGUAGE_MAP.get(language, language)

        # Remove the direct retriever usage since it's now handled in the pipeline
        rag_chain = get_rag_pipeline()
        result = rag_chain.invoke({
            "question": query_body.query,
            "language": full_language
        })

        # Get the sources from the context step of the pipeline
        context_docs = rag_chain.steps[0].invoke({
            "question": query_body.query,
            "language": full_language
        })["context"]

        return {
            "response": result,
            "sources": [doc.metadata for doc in context_docs],
            "language": full_language
        }

    except ValueError as ve:
        logger.error("Validation error: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        logger.exception("Query processing failed")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your query"
        ) from e

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
