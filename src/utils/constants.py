"""Constants for the project"""
from typing import Dict
APP_CONFIG = {
    "title": "Enterprise Knowledge Hub",
    "description": "API for document ingestion and querying using RAG",
    "version": "1.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc"
}

CORS_CONFIG = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"]
}

LANGUAGE_MAP: Dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "hi": "Hindi"
}

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RETRIEVER_K = 3

PROMPT_TEMPLATE = """You are an intelligent assistant helping with enterprise knowledge management,
specializing in retrieving and explaining organizational knowledge.

Instructions:
- **Response Structure**:
  1. Provide a brief outline to summarize your approach to the query.
  2. Start with a clear and structured **overview** of the topic.
  3. If applicable, use **tables** to present information in a structured format.
  4. Prioritize based on:
    1. Direct relevance to the query
    2. Recency and timeliness of information 
    3. Specificity and depth of details
    4. Strategic importance to the organization

- **Content Guidelines**:
  - Simplify complex topics into manageable sections.
  - For processes or workflows, use **numbered steps** and include prerequisites.
  - Define **technical terms** (if any) for better understanding.

Context:
{context}

Question:
{question}

Make sure to respond in {language} language.

Helpful Answer:"""
