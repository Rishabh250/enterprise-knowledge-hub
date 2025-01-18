import os
from typing import List, Optional
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .google_drive import GoogleDriveLoader

class DocumentLoader:
    """Class for loading and splitting documents"""
    def __init__(self, google_drive_credentials: str = None):
        """
        Initialize document loader
        
        Args:
            google_drive_credentials: Path to Google Drive service account credentials
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self._init_google_drive(google_drive_credentials)

    def _init_google_drive(self, credentials_path: Optional[str] = None) -> None:
        """Initialize Google Drive loader if credentials are available"""
        try:
            self.google_drive_loader = GoogleDriveLoader(credentials_path)
            self.google_drive_enabled = True
        except FileNotFoundError:
            print("Google Drive integration not configured. Skipping initialization.")
            self.google_drive_enabled = False
        except (IOError, OSError) as e:
            print(f"Error initializing Google Drive: {str(e)}")
            self.google_drive_enabled = False

    def load_document(self, file_path: str, is_google_drive: bool = False) -> List:
        """
        Load and split a document based on its file type
        
        Args:
            file_path: Local file path or Google Drive file ID
            is_google_drive: Whether the file is from Google Drive
        """
        if is_google_drive:
            if not self.google_drive_enabled:
                raise ValueError("Google Drive integration is not configured")
            try:
                file_path = self.google_drive_loader.download_file(file_path)
            except (IOError, OSError) as e:
                raise IOError(f"Error downloading from Google Drive: {str(e)}") from e

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
        elif file_extension == '.docx':
            try:
                loader = Docx2txtLoader(file_path)
            except (ImportError, IOError):
                loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        documents = loader.load()
        return self.text_splitter.split_documents(documents)
