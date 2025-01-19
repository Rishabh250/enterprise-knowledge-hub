import os
import warnings
from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH")

class VectorStoreManager:
    """Manager for creating and loading FAISS vector stores"""
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize vector store manager with embedding model"""
        self._embedding_model = embedding_model_name
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    @property
    @lru_cache(maxsize=1)
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy load and cache embeddings model"""
        if not self._embeddings:
            self._embeddings = HuggingFaceEmbeddings(model_name=self._embedding_model)
        return self._embeddings

    def create_vectorstore(self, documents: List[Document], save_path: str) -> FAISS:
        """Create and save a FAISS vector store from documents"""
        if not documents:
            raise ValueError("No documents provided to create vector store")

        vectorstore = FAISS.from_documents(documents, self.embeddings)
        vectorstore.save_local(save_path)
        return vectorstore

    def load_vectorstore(self, load_path: str, trust_source: bool = False) -> FAISS:
        """
        Load a FAISS vector store from disk
        
        Args:
            load_path: Path to the vector store
            trust_source: If True, allows deserialization of the vector store.
                        WARNING: Only set to True if you trust the source.
        
        Returns:
            FAISS vector store instance
            
        Raises:
            SecurityError: If trust_source is False
            RuntimeError: If loading fails
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Vector store not found at {load_path}")

        if not trust_source:
            warnings.warn(
                "Loading vector stores requires deserializing pickle files, which can be unsafe. "
                "If you trust the source of this vector store (e.g., you created it), "
                "set trust_source=True. Never set trust_source=True with files from untrusted sources.",
                UserWarning
            )
            raise SecurityError("Refusing to load vector store without explicit trust_source=True")

        try:
            return FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load vector store from {load_path}. Error: {str(e)}")

class SecurityError(Exception):
    """Raised when attempting unsafe operations without explicit permission"""
    pass
