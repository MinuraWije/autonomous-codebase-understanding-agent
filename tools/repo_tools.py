"""Repository inspection tools."""
from typing import List, Dict
from indexing.metadata_store import get_metadata_store
from core.constants import KEY_FILE_PATTERNS


def get_repo_summary(repo_id: str) -> Dict:
    """
    Get a summary of the repository.
    
    Args:
        repo_id: Repository ID
    
    Returns:
        Repository summary
    """
    metadata_store = get_metadata_store()
    repo = metadata_store.get_repo_metadata(repo_id)
    
    if not repo:
        raise ValueError(f"Repository not found: {repo_id}")
    
    return {
        'repo_id': repo['repo_id'],
        'url': repo['url'],
        'local_path': repo['local_path'],
        'commit_hash': repo['commit_hash'],
        'indexed_at': repo['indexed_at'],
        'stats': repo['stats']
    }


def list_all_repos() -> List[Dict]:
    """
    List all indexed repositories.
    
    Returns:
        List of repository summaries
    """
    metadata_store = get_metadata_store()
    return metadata_store.list_repos()


def get_key_files(repo_id: str, top_n: int = 10) -> List[str]:
    """
    Identify key files in the repository (entry points, main modules).
    
    Args:
        repo_id: Repository ID
        top_n: Number of key files to return
    
    Returns:
        List of file paths
    """
    metadata_store = get_metadata_store()
    repo = metadata_store.get_repo_metadata(repo_id)
    
    if not repo:
        raise ValueError(f"Repository not found: {repo_id}")
    
    from pathlib import Path
    repo_path = Path(repo['local_path'])
    
    key_files = []
    for pattern in KEY_FILE_PATTERNS:
        matches = list(repo_path.rglob(pattern))
        for match in matches[:2]:  # Limit per pattern
            try:
                rel_path = match.relative_to(repo_path)
                key_files.append(str(rel_path))
            except ValueError:
                pass
    
    return key_files[:top_n]
