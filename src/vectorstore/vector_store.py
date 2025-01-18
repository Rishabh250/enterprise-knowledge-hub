from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv
import warnings

load_dotenv()

# Move print statement here for debugging during module import
print("VECTORSTORE_PATH:", os.getenv("VECTORSTORE_PATH"))

class VectorStoreManager:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def create_vectorstore(self, documents: List[Document], save_path: str):
        """Create and save a FAISS vector store from documents"""
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        vectorstore.save_local(save_path)
        return vectorstore

    def load_vectorstore(self, load_path: str, trust_source: bool = False):
        """
        Load a FAISS vector store from disk
        
        Args:
            load_path: Path to the vector store
            trust_source: If True, allows deserialization of the vector store.
                        WARNING: Only set to True if you trust the source of the vector store.
                        Setting this to True with untrusted sources could lead to code execution vulnerabilities.
        """
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
            raise RuntimeError(f"Failed to load vector store from {load_path}. Have you ingested any documents yet? Error: {str(e)}")

class SecurityError(Exception):
    """Raised when attempting unsafe operations without explicit permission"""
    pass 