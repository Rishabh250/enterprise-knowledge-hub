import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.ingestion.document_loader import DocumentLoader
from src.vectorstore.vector_store import VectorStoreManager
from dotenv import load_dotenv

def ingest_documents(docs_dir: str):
    """
    Ingest documents from a directory and create a vector store
    
    Args:
        docs_dir: Directory containing documents to ingest
    """
    load_dotenv()
    
    # Get vectorstore path from environment
    vectorstore_path = os.getenv("VECTORSTORE_PATH")
    if not vectorstore_path:
        raise ValueError("VECTORSTORE_PATH not set in .env file")

    # Initialize document loader and vector store manager
    loader = DocumentLoader()
    vector_store_manager = VectorStoreManager()
    
    # Process all documents in the directory
    documents = []
    for root, _, files in os.walk(docs_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                print(f"Processing {file_path}")
                docs = loader.load_document(file_path)
                documents.extend(docs)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    if documents:
        print(f"Creating vector store with {len(documents)} documents")
        vector_store_manager.create_vectorstore(documents, vectorstore_path)
        print(f"Vector store created successfully at {vectorstore_path}")
    else:
        print("No documents were processed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument("--docs-dir", required=True, help="Directory containing documents to ingest")
    args = parser.parse_args()
    
    ingest_documents(args.docs_dir) 