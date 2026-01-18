"""File operation tools for reading code files."""
from pathlib import Path
from typing import Optional, List
from indexing.metadata_store import get_metadata_store


def open_file(repo_id: str, file_path: str) -> str:
    """
    Open and read a complete file.
    
    Args:
        repo_id: Repository ID
        file_path: Path to file (relative to repo root)
    
    Returns:
        File contents
    """
    metadata_store = get_metadata_store()
    repo = metadata_store.get_repo_metadata(repo_id)
    
    if not repo:
        raise ValueError(f"Repository not found: {repo_id}")
    
    full_path = Path(repo['local_path']) / file_path
    
    if not full_path.exists():
        raise ValueError(f"File not found: {file_path}")
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


def open_span(repo_id: str, file_path: str, start_line: int, end_line: int) -> str:
    """
    Open and read a specific line range from a file.
    
    Args:
        repo_id: Repository ID
        file_path: Path to file (relative to repo root)
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (inclusive)
    
    Returns:
        File contents for the specified range
    """
    content = open_file(repo_id, file_path)
    lines = content.split('\n')
    
    # Adjust for 1-indexed lines
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    
    return '\n'.join(lines[start_idx:end_idx])


def list_files(repo_id: str, pattern: Optional[str] = None) -> List[str]:
    """
    List files in a repository, optionally filtered by pattern.
    
    Args:
        repo_id: Repository ID
        pattern: Optional pattern to match (e.g., "*.py", "auth")
    
    Returns:
        List of file paths
    """
    metadata_store = get_metadata_store()
    repo = metadata_store.get_repo_metadata(repo_id)
    
    if not repo:
        raise ValueError(f"Repository not found: {repo_id}")
    
    repo_path = Path(repo['local_path'])
    
    if pattern:
        # Use glob pattern
        if '*' in pattern:
            files = list(repo_path.rglob(pattern))
        else:
            # Simple substring match
            files = [f for f in repo_path.rglob('*') if pattern in str(f)]
    else:
        files = list(repo_path.rglob('*'))
    
    # Filter to only files and make relative
    file_paths = []
    for f in files:
        if f.is_file():
            try:
                rel_path = f.relative_to(repo_path)
                file_paths.append(str(rel_path))
            except ValueError:
                pass
    
    return sorted(file_paths)


def get_file_structure(repo_id: str, max_depth: int = 3) -> dict:
    """
    Get the directory structure of the repository.
    
    Args:
        repo_id: Repository ID
        max_depth: Maximum depth to traverse
    
    Returns:
        Dictionary representing file structure
    """
    metadata_store = get_metadata_store()
    repo = metadata_store.get_repo_metadata(repo_id)
    
    if not repo:
        raise ValueError(f"Repository not found: {repo_id}")
    
    repo_path = Path(repo['local_path'])
    
    def build_tree(path: Path, depth: int = 0) -> dict:
        if depth >= max_depth:
            return {}
        
        tree = {}
        try:
            for item in sorted(path.iterdir()):
                # Skip hidden files and common ignore patterns
                if item.name.startswith('.') or item.name in ['node_modules', '__pycache__', 'venv']:
                    continue
                
                if item.is_dir():
                    tree[item.name] = build_tree(item, depth + 1)
                else:
                    tree[item.name] = None
        except PermissionError:
            pass
        
        return tree
    
    return build_tree(repo_path)
