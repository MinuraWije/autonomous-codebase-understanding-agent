"""Indexing service for repository indexing operations."""
from typing import Optional
from indexing.pipeline import index_repository, get_indexing_status
from indexing.loader import generate_repo_id
from core.exceptions import IndexingError
from core.constants import ERROR_INDEXING_FAILED


class IndexingService:
    """Service for indexing operations."""
    
    def start_indexing(
        self,
        github_url: Optional[str] = None,
        local_path: Optional[str] = None,
        branch: str = "main"
    ) -> str:
        """
        Start indexing a repository.
        
        Args:
            github_url: GitHub repository URL
            local_path: Local repository path
            branch: Branch to clone
        
        Returns:
            Repository ID
        
        Raises:
            ValueError: If neither github_url nor local_path provided
        """
        if not github_url and not local_path:
            raise ValueError("Either github_url or local_path must be provided")
        
        return generate_repo_id(github_url or local_path)
    
    def index_repository_task(
        self,
        github_url: Optional[str] = None,
        local_path: Optional[str] = None,
        branch: str = "main"
    ) -> None:
        """
        Background task to index a repository.
        
        Args:
            github_url: GitHub repository URL
            local_path: Local repository path
            branch: Branch to clone
        """
        try:
            index_repository(
                github_url=github_url,
                local_path=local_path,
                branch=branch
            )
            print(f"Indexing completed for {github_url or local_path}")
        except Exception as e:
            error_msg = ERROR_INDEXING_FAILED.format(error=str(e))
            print(error_msg)
            raise IndexingError(error_msg) from e
    
    def get_indexing_status(self, repo_id: str) -> dict:
        """
        Get indexing status for a repository.
        
        Args:
            repo_id: Repository ID
        
        Returns:
            Status dictionary
        """
        return get_indexing_status(repo_id)
