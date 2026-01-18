"""Repository loading and file filtering."""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from git import Repo
from app.config import settings


@dataclass
class RepoMetadata:
    """Metadata about a loaded repository."""
    repo_id: str
    url: Optional[str]
    local_path: str
    commit_hash: str
    stats: Dict[str, int]


# File extensions to ignore
BINARY_EXTENSIONS = {
    '.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', '.bin', '.dat',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.zip', '.tar', '.gz', '.rar', '.7z',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico',
    '.mp3', '.mp4', '.avi', '.mov', '.wav',
    '.ttf', '.woff', '.woff2', '.eot',
}

# Directories to ignore
IGNORE_DIRS = {
    'node_modules', '.git', '__pycache__', 'venv', 'env', '.venv', '.env',
    'dist', 'build', '.idea', '.vscode', '.pytest_cache', '.mypy_cache',
    'coverage', '.coverage', 'htmlcov', '.tox', '.eggs', '*.egg-info',
    'target', 'bin', 'obj', '.gradle', '.mvn',
}

# Lock files and build artifacts to ignore
IGNORE_FILES = {
    'package-lock.json', 'yarn.lock', 'poetry.lock', 'Pipfile.lock',
    'Gemfile.lock', 'composer.lock', 'cargo.lock',
    '.DS_Store', 'Thumbs.db', '.gitignore', '.dockerignore',
}

# Language extensions we want to index
SOURCE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', '.c', '.cpp',
    '.h', '.hpp', '.cs', '.php', '.rb', '.swift', '.kt', '.scala', '.sh',
    '.bash', '.sql', '.md', '.txt', '.yaml', '.yml', '.json', '.xml', '.html',
    '.css', '.scss', '.less', '.vue', '.svelte',
}


def generate_repo_id(url_or_path: str) -> str:
    """Generate a unique repo ID from URL or path."""
    return hashlib.md5(url_or_path.encode()).hexdigest()[:12]


def clone_repo(github_url: str, branch: str = "main") -> RepoMetadata:
    """
    Clone a GitHub repository.
    
    Args:
        github_url: GitHub repository URL
        branch: Branch to clone (default: main)
    
    Returns:
        RepoMetadata with repository information
    """
    repo_id = generate_repo_id(github_url)
    target_dir = settings.repos_dir / repo_id
    
    # Remove existing directory if it exists
    if target_dir.exists():
        import shutil
        shutil.rmtree(target_dir)
    
    # Clone the repository
    repo = Repo.clone_from(github_url, target_dir, branch=branch, depth=1)
    commit_hash = repo.head.commit.hexsha
    
    # Get file statistics
    stats = get_repo_stats(target_dir)
    
    return RepoMetadata(
        repo_id=repo_id,
        url=github_url,
        local_path=str(target_dir),
        commit_hash=commit_hash,
        stats=stats
    )


def load_local_repo(path: str) -> RepoMetadata:
    """
    Load a local repository.
    
    Args:
        path: Path to local repository
    
    Returns:
        RepoMetadata with repository information
    """
    repo_path = Path(path).resolve()
    if not repo_path.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    repo_id = generate_repo_id(str(repo_path))
    
    # Try to get git info
    commit_hash = "unknown"
    try:
        repo = Repo(repo_path)
        commit_hash = repo.head.commit.hexsha
    except Exception:
        pass  # Not a git repo or no commits
    
    stats = get_repo_stats(repo_path)
    
    return RepoMetadata(
        repo_id=repo_id,
        url=None,
        local_path=str(repo_path),
        commit_hash=commit_hash,
        stats=stats
    )


def filter_files(repo_path: Path) -> List[Path]:
    """
    Get list of valid source files from repository.
    
    Args:
        repo_path: Path to repository
    
    Returns:
        List of file paths to index
    """
    valid_files = []
    
    for root, dirs, files in os.walk(repo_path):
        # Remove ignored directories from traversal
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
        
        for file in files:
            # Skip ignored files
            if file in IGNORE_FILES or file.startswith('.'):
                continue
            
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            
            # Skip binary files
            if ext in BINARY_EXTENSIONS:
                continue
            
            # Include source files
            if ext in SOURCE_EXTENSIONS:
                valid_files.append(file_path)
            # Also include files without extension if they're likely source files
            elif not ext and is_text_file(file_path):
                valid_files.append(file_path)
    
    return valid_files


def is_text_file(file_path: Path, sample_size: int = 512) -> bool:
    """Check if a file is likely a text file by reading a sample."""
    try:
        with open(file_path, 'rb') as f:
            sample = f.read(sample_size)
        
        # Check for null bytes (indicates binary)
        if b'\x00' in sample:
            return False
        
        # Try to decode as UTF-8
        try:
            sample.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False
    except Exception:
        return False


def get_repo_stats(repo_path: Path) -> Dict[str, int]:
    """
    Get statistics about the repository.
    
    Args:
        repo_path: Path to repository
    
    Returns:
        Dictionary with file counts by language
    """
    stats = {
        'total_files': 0,
        'by_language': {}
    }
    
    files = filter_files(repo_path)
    stats['total_files'] = len(files)
    
    for file in files:
        ext = file.suffix.lower() or 'no_extension'
        stats['by_language'][ext] = stats['by_language'].get(ext, 0) + 1
    
    return stats


def get_language_from_extension(file_path: Path) -> str:
    """Map file extension to language name."""
    ext_to_lang = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sh': 'bash',
        '.bash': 'bash',
        '.sql': 'sql',
        '.md': 'markdown',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
    }
    return ext_to_lang.get(file_path.suffix.lower(), 'unknown')
