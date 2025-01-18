import os
import argparse
from pathlib import Path
from src.ingestion.document_loader import DocumentLoader
from src.vectorstore.vector_store import VectorStoreManager
from dotenv import load_dotenv

def ingest_documents(docs_dir: str, vectorstore_path: str):
    """
    Ingest documents from a directory and create a vector store
    
    Args:
        docs_dir: Directory containing documents to ingest
        vectorstore_path: Path where to save the vector store
    """
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
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument(
        "--docs-dir",
        type=str,
        required=True,
        help="Directory containing documents to ingest"
    )
    parser.add_argument(
        "--vectorstore-path",
        type=str,
        help="Path where to save the vector store"
    )
    
    args = parser.parse_args()
    load_dotenv()
    
    # Use provided vectorstore path or get from environment
    vectorstore_path = args.vectorstore_path or os.getenv("VECTORSTORE_PATH")
    
    # Create vectorstore directory if it doesn't exist
    Path(vectorstore_path).parent.mkdir(parents=True, exist_ok=True)
    
    ingest_documents(args.docs_dir, vectorstore_path) 