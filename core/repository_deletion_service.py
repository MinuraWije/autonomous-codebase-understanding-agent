"""Service for repository deletion operations."""
from indexing.metadata_store import get_metadata_store
from indexing.vector_store import get_vector_store
from core.repository_service import RepositoryService
from core.exceptions import RepositoryNotFoundError
from core.constants import MESSAGE_REPO_DELETED


class RepositoryDeletionService:
    """Service for repository deletion."""
    
    def __init__(self):
        """Initialize the deletion service."""
        self.repository_service = RepositoryService()
        self.metadata_store = get_metadata_store()
    
    def delete_repository(self, repo_id: str) -> str:
        """
        Delete a repository and all its indexed data.
        
        Args:
            repo_id: Repository ID
        
        Returns:
            Success message
        
        Raises:
            RepositoryNotFoundError: If repository not found
        """
        # Validate repository exists
        self.repository_service.validate_repository_exists(repo_id)
        
        # Delete from metadata store
        self.metadata_store.delete_repo(repo_id)
        
        # Delete from vector store
        vector_store = get_vector_store()
        vector_store.delete_collection(repo_id)
        
        return MESSAGE_REPO_DELETED.format(repo_id=repo_id)
