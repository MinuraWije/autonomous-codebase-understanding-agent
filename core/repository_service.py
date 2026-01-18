"""Repository service for centralized repository operations."""
from typing import Optional, Dict
from indexing.metadata_store import get_metadata_store
from core.exceptions import RepositoryNotFoundError
from core.constants import ERROR_REPO_NOT_FOUND


class RepositoryService:
    """Service for repository operations."""
    
    def __init__(self):
        """Initialize the repository service."""
        self.metadata_store = get_metadata_store()
    
    def get_repository(self, repo_id: str) -> Dict:
        """
        Get repository metadata.
        
        Args:
            repo_id: Repository ID
        
        Returns:
            Repository metadata
        
        Raises:
            RepositoryNotFoundError: If repository not found
        """
        repo = self.metadata_store.get_repo_metadata(repo_id)
        if not repo:
            raise RepositoryNotFoundError(ERROR_REPO_NOT_FOUND.format(repo_id=repo_id))
        return repo
    
    def validate_repository_exists(self, repo_id: str) -> None:
        """
        Validate that a repository exists.
        
        Args:
            repo_id: Repository ID
        
        Raises:
            RepositoryNotFoundError: If repository not found
        """
        if not self.metadata_store.get_repo_metadata(repo_id):
            raise RepositoryNotFoundError(ERROR_REPO_NOT_FOUND.format(repo_id=repo_id))
    
    def list_repositories(self) -> list:
        """
        List all repositories.
        
        Returns:
            List of repository summaries
        """
        return self.metadata_store.list_repos()
    
    def get_repository_status(self, repo_id: str) -> Dict:
        """
        Get repository indexing status.
        
        Args:
            repo_id: Repository ID
        
        Returns:
            Status dictionary
        """
        repo = self.metadata_store.get_repo_metadata(repo_id)
        if not repo:
            return {'status': 'not_found'}
        
        return {
            'status': 'completed',
            'repo_id': repo['repo_id'],
            'url': repo['url'],
            'indexed_at': repo['indexed_at'],
            'stats': repo['stats']
        }
