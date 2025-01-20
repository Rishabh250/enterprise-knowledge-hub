"""
This module contains the RAG pipeline for the application.
"""

import os
from functools import lru_cache
from typing import List, Dict, Any

from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM

from src.utils.constants import (
    PROMPT_TEMPLATE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RETRIEVER_K
)
from src.utils.env import get_env_var
from src.utils.helpers import log_message

@lru_cache()
def get_llm() -> OllamaLLM:
    """Initialize and return the LLM model."""
    model_name = get_env_var("MODEL")
    if not model_name:
        raise ValueError("MODEL environment variable is not set.")
    return OllamaLLM(model=model_name)

@lru_cache()
def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize and return the embedding model."""
    if not DEFAULT_EMBEDDING_MODEL:
        raise ValueError("Default embedding model is not configured.")
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)

@lru_cache()
def get_vectorstore() -> FAISS:
    """Initialize and return the vector store."""
    vectorstore_path = get_env_var("VECTORSTORE_PATH")

    if not vectorstore_path or not os.path.exists(vectorstore_path):
        log_message(f"Vector store not found at {vectorstore_path}", "error")
        raise FileNotFoundError(f"Vector store not found at {vectorstore_path}")

    return FAISS.load_local(
        vectorstore_path,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

def get_ensemble_retriever(docs: List[str]) -> EnsembleRetriever:
    """Create an ensemble retriever combining semantic and keyword search."""
    vectorstore = get_vectorstore()

    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": DEFAULT_RETRIEVER_K,
            "score_threshold": 0.7,
        }
    )

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = DEFAULT_RETRIEVER_K

    return EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

@lru_cache()
def get_doc_retriever():
    """Initialize and return the document retriever."""
    llm = get_llm()
    vectorstore = get_vectorstore()

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": DEFAULT_RETRIEVER_K,
                "score_threshold": 0.1,
            }
        ),
        llm=llm
    )

    def rerank_documents(docs: List[Any]) -> List[Any]:
        """Rerank documents based on relevance to query."""
        return sorted(docs, key=lambda doc: getattr(doc, "score", 0), reverse=True)

    def process_retrieval(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process document retrieval with multiple strategies."""
        query = input_dict.get("question")
        if not query:
            raise ValueError("Input dictionary must contain a 'question' key.")

        docs = multi_query_retriever.get_relevant_documents(query)

        reranked_docs = rerank_documents(docs)

        if not reranked_docs:
            log_message(f"No relevant documents found for query: {query}", level="warning")
            return {
                "context": "No relevant documents were found for your query.",
                "sources": []
            }

        return {
            "context": "\n\n".join(doc.page_content for doc in reranked_docs[:DEFAULT_RETRIEVER_K]),
            "sources": [doc.metadata for doc in reranked_docs[:DEFAULT_RETRIEVER_K]],
        }

    return process_retrieval

@lru_cache()
def get_rag_pipeline():
    """Initialize and return the advanced RAG pipeline."""
    llm = get_llm()

    template = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question", "language"]
    )

    output_parser = StrOutputParser()

    runnable_parallel = RunnableParallel({
        "question": RunnablePassthrough(),
        "language": lambda x: x.get("language", "en"),
        "context": lambda x: x.get("context", ""),
    })

    rag_chain = (
        runnable_parallel
        | template
        | llm
        | output_parser
    )

    return rag_chain
