import os
from pathlib import Path
from dotenv import load_dotenv

def setup_directories():
    """Create necessary directories for the project"""
    load_dotenv()
    
    # Get vectorstore path from environment
    vectorstore_path = os.getenv("VECTORSTORE_PATH")
    if not vectorstore_path:
        raise ValueError("VECTORSTORE_PATH not set in .env file")
    
    # Create directories
    directories = [
        "data/documents",  # For storing input documents
        vectorstore_path,  # For storing FAISS index
        "logs"            # For logging
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

if __name__ == "__main__":
    setup_directories() 