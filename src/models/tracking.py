from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel

class DocumentTrack(BaseModel):
    """Track document ingestion status"""
    file_path: str
    file_id: Optional[str] = None  # For Google Drive files
    status: str  # 'success', 'failed', or 'vectorized'
    timestamp: datetime
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    vectorized: bool = False  # Whether the document is stored in vector DB
    vectorized_at: Optional[datetime] = None

class IngestionTracker:
    """Manage document ingestion tracking"""

    def __init__(self, tracking_file: str = "data/tracking/ingestion_log.jsonl"):
        self.tracking_file = tracking_file
        self._ensure_tracking_dir()

    def _ensure_tracking_dir(self):
        """Ensure tracking directory exists"""
        from pathlib import Path
        Path(self.tracking_file).parent.mkdir(parents=True, exist_ok=True)

    def track(self, document: DocumentTrack):
        """Add tracking entry"""
        with open(self.tracking_file, 'a') as f:
            f.write(document.json() + '\n')

    def update_vectorization_status(self, file_path: str, success: bool = True, error: str = None):
        """Update vectorization status for a document"""
        from pathlib import Path
        import tempfile
        import shutil
        
        if not Path(self.tracking_file).exists():
            return
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        updated = False
        
        try:
            with open(self.tracking_file, 'r') as f:
                for line in f:
                    if line.strip():
                        track = DocumentTrack.parse_raw(line)
                        if track.file_path == file_path:
                            track.vectorized = success
                            track.vectorized_at = datetime.now()
                            if error:
                                track.error = error
                            updated = True
                        temp_file.write(track.json() + '\n')
            
            temp_file.close()
            # Replace original file with updated one
            shutil.move(temp_file.name, self.tracking_file)
            
            if not updated:
                # If file wasn't found, add new entry
                self.track(DocumentTrack(
                    file_path=file_path,
                    status='vectorized' if success else 'failed',
                    timestamp=datetime.now(),
                    vectorized=success,
                    vectorized_at=datetime.now() if success else None,
                    error=error
                ))
        finally:
            # Cleanup temp file if something went wrong
            if Path(temp_file.name).exists():
                Path(temp_file.name).unlink()
    
    def get_history(self) -> List[DocumentTrack]:
        """Get ingestion history"""
        from pathlib import Path
        if not Path(self.tracking_file).exists():
            return []
        
        tracks = []
        with open(self.tracking_file) as f:
            for line in f:
                if line.strip():
                    tracks.append(DocumentTrack.parse_raw(line))
        return tracks
    
    def get_vectorized_files(self) -> List[str]:
        """Get list of files that have been vectorized"""
        return [
            track.file_path 
            for track in self.get_history() 
            if track.vectorized
        ]
    
    def get_failed_files(self) -> List[str]:
        """Get list of files that failed vectorization"""
        return [
            track.file_path 
            for track in self.get_history() 
            if not track.vectorized and track.status == 'failed'
        ]
    
    def is_file_vectorized(self, file_id: str) -> bool:
        """
        Check if a file has already been vectorized
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            bool: True if file is already vectorized, False otherwise
        """
        for track in self.get_history():
            if track.file_id == file_id and track.vectorized:
                return True
        return False
    
    def get_vectorized_file_ids(self) -> List[str]:
        """Get list of file IDs that have been vectorized"""
        return [
            track.file_id 
            for track in self.get_history() 
            if track.vectorized and track.file_id is not None
        ] 