import os
import io
from pathlib import Path
from typing import List, Dict, Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import logging

from dotenv import load_dotenv

load_dotenv()

GOOGLE_DRIVE_FOLDER = os.getenv("GOOGLE_DRIVE_FOLDER")
GOOGLE_DRIVE_CREDENTIALS_PATH = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleDriveLoader:
    """Handles authentication and file downloading from Google Drive using service account"""

    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    DOWNLOAD_DIR = GOOGLE_DRIVE_FOLDER

    EXPORT_FORMATS = {
        'application/vnd.google-apps.document': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # DOCX
        'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # XLSX
        'application/vnd.google-apps.presentation': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',  # PPTX
    }

    def __init__(self, credentials_path: Optional[str] = None) -> None:
        """
        Initialize Google Drive loader

        Args:
            credentials_path: Path to service account JSON file from Google Cloud Console

        Raises:
            FileNotFoundError: If credentials file is not found
        """
        self.credentials_path = credentials_path or GOOGLE_DRIVE_CREDENTIALS_PATH
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                "Service account credentials not found at %s. "
                "Please follow the setup instructions in the README to configure Google Drive integration." % 
                self.credentials_path
            )

        # Ensure download directory exists
        Path(self.DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

    def authenticate(self):
        """
        Handle Google Drive authentication using service account

        Returns:
            Google Drive service object

        Raises:
            Exception: If authentication fails
        """
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=self.SCOPES
            )
            return build('drive', 'v3', credentials=credentials)
        except Exception as e:
            raise Exception("Authentication failed: %s" % str(e))

    def download_file(self, file_id: str, save_path: Optional[str] = None) -> str:
        """
        Download a file from Google Drive

        Args:
            file_id: Google Drive file ID
            save_path: Path to save the downloaded file (optional)

        Returns:
            str: Path to the downloaded file

        Raises:
            Exception: If download or save fails
        """
        service = self.authenticate()

        try:
            # Get file metadata
            file_metadata = service.files().get(fileId=file_id).execute()
            file_name = file_metadata.get('name')
            mime_type = file_metadata.get('mimeType')

            # Handle different file types
            if mime_type.startswith('application/vnd.google-apps.'):
                # Handle Google Workspace files
                if mime_type == 'application/vnd.google-apps.document':
                    request = service.files().export_media(
                        fileId=file_id,
                        mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    )
                    file_name += '.docx'
                elif mime_type == 'application/vnd.google-apps.spreadsheet':
                    request = service.files().export_media(
                        fileId=file_id,
                        mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                    file_name += '.xlsx'
                elif mime_type == 'application/vnd.google-apps.presentation':
                    request = service.files().export_media(
                        fileId=file_id,
                        mimeType='application/vnd.openxmlformats-officedocument.presentationml.presentation'
                    )
                    file_name += '.pptx'
                else:
                    logger.warning("Unsupported Google Workspace file type: %s. Skipping file: %s", mime_type, file_name)
                    raise Exception("Unsupported Google Workspace file type: %s" % mime_type)
            else:
                # Handle regular files
                request = service.files().get_media(fileId=file_id)

            file_handle = io.BytesIO()
            downloader = MediaIoBaseDownload(file_handle, request)

            # Download the file
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.info("Download %d%% completed", int(status.progress() * 100))

            # Save the file
            if not save_path:
                save_path = os.path.join(self.DOWNLOAD_DIR, file_name)
                # Handle duplicate filenames
                counter = 1
                while os.path.exists(save_path):
                    base_name, ext = os.path.splitext(file_name)
                    save_path = os.path.join(self.DOWNLOAD_DIR, f"{base_name}_{counter}{ext}")
                    counter += 1

            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(file_handle.getvalue())

            logger.info("Saved file to: %s", save_path)
            return save_path

        except Exception as e:
            raise Exception("Error downloading file from Google Drive: %s" % str(e))

    def list_files_in_folder(self, folder_id: str) -> List[Dict[str, str]]:
        """
        List all files in a specified Google Drive folder, including files in subfolders.

        Args:
            folder_id: Google Drive folder ID

        Returns:
            List[Dict[str, str]]: List of file metadata dictionaries containing 'id', 'name', and 'mimeType'

        Raises:
            Exception: If listing files fails
        """
        service = self.authenticate()
        all_files = []

        def list_files_recursive(folder_id: str):
            try:
                query = f"'{folder_id}' in parents and trashed=false"
                results = service.files().list(
                    q=query,
                    fields="files(id, name, mimeType)",
                    pageSize=1000
                ).execute()
                files = results.get('files', [])
                for file in files:
                    if file['mimeType'] == 'application/vnd.google-apps.folder':
                        list_files_recursive(file['id'])
                    else:
                        all_files.append(file)
            except Exception as e:
                raise Exception("Error listing files: %s" % str(e))

        list_files_recursive(folder_id)
        return all_files

    def download_single_file(self, file_id: str, save_dir: Optional[str] = None) -> str:
        """
        Download a single file from Google Drive.

        Args:
            file_id: Google Drive file ID
            save_dir: Directory to save the downloaded file (optional)

        Returns:
            str: Path to the downloaded file

        Raises:
            Exception: If file download fails
        """
        save_dir = self.DOWNLOAD_DIR if save_dir is None else save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Get file metadata
        service = self.authenticate()
        try:
            file = service.files().get(fileId=file_id, fields="id, name, mimeType").execute()
            file_name = file.get('name')
            mime_type = file.get('mimeType')

            # Map of common MIME types to file extensions
            mime_to_extension = {
                'application/vnd.google-apps.document': '.docx',
                'application/vnd.google-apps.spreadsheet': '.xlsx',
                'application/vnd.google-apps.presentation': '.pptx',
                'application/pdf': '.pdf',
                'text/plain': '.txt',
                'image/jpeg': '.jpg',
                'image/png': '.png',
            }

            # Determine file extension based on MIME type
            extension = os.path.splitext(file_name)[1]  # Check if the name already has an extension
            if not extension:
                extension = mime_to_extension.get(mime_type, '')  # Default to no extension if unknown

            file_name_with_extension = file_name + extension
            save_path = os.path.join(save_dir, file_name_with_extension)

            # Download the file
            self.download_file(file_id, save_path)
            logger.info("Downloaded file: %s", save_path)
            return save_path

        except Exception as e:
            logger.error("Error downloading file (ID: %s): %s", file_id, str(e))
            raise Exception("Error downloading file from Google Drive: %s" % str(e))
