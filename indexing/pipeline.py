"""Main indexing pipeline that orchestrates the indexing process."""
from pathlib import Path
from typing import Optional
from indexing.loader import clone_repo, load_local_repo, filter_files, get_language_from_extension, RepoMetadata
from indexing.chunking import chunk_file
from indexing.vector_store import get_vector_store
from indexing.metadata_store import get_metadata_store


def index_repository(
    github_url: Optional[str] = None,
    local_path: Optional[str] = None,
    branch: str = "main"
) -> RepoMetadata:
    """
    Index a repository (GitHub or local).
    
    Args:
        github_url: GitHub repository URL
        local_path: Local repository path
        branch: Branch to clone (for GitHub repos)
    
    Returns:
        Repository metadata
    """
    if github_url:
        print(f"Cloning repository: {github_url}")
        repo_metadata = clone_repo(github_url, branch)
    elif local_path:
        print(f"Loading local repository: {local_path}")
        repo_metadata = load_local_repo(local_path)
    else:
        raise ValueError("Either github_url or local_path must be provided")
    
    print(f"Repository ID: {repo_metadata.repo_id}")
    print(f"Total files: {repo_metadata.stats['total_files']}")
    print(f"File types: {repo_metadata.stats['by_language']}")
    
    # Save repo metadata
    metadata_store = get_metadata_store()
    metadata_store.save_repo_metadata(repo_metadata)
    
    # Get files to index
    repo_path = Path(repo_metadata.local_path)
    files = filter_files(repo_path)
    
    print(f"Indexing {len(files)} files...")
    
    # Chunk all files
    all_chunks = []
    for i, file_path in enumerate(files):
        if (i + 1) % 10 == 0:
            print(f"Processing file {i+1}/{len(files)}")
        
        language = get_language_from_extension(file_path)
        
        # Make file path relative to repo root
        relative_path = file_path.relative_to(repo_path)
        
        chunks = chunk_file(file_path, repo_metadata.repo_id, language)
        
        # Update chunk file paths to be relative
        for chunk in chunks:
            chunk.file_path = str(relative_path)
        
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks")
    
    # Save chunks to metadata store
    print("Saving chunks to metadata store...")
    metadata_store.save_chunks(all_chunks)
    
    # Add chunks to vector store
    print("Adding chunks to vector store...")
    vector_store = get_vector_store()
    vector_store.add_chunks(all_chunks)
    
    print("Indexing complete!")
    return repo_metadata


def get_indexing_status(repo_id: str) -> dict:
    """
    Get the indexing status of a repository.
    
    Args:
        repo_id: Repository ID
    
    Returns:
        Status dictionary
    """
    metadata_store = get_metadata_store()
    repo = metadata_store.get_repo_metadata(repo_id)
    
    if not repo:
        return {'status': 'not_found'}
    
    return {
        'status': 'completed',
        'repo_id': repo['repo_id'],
        'url': repo['url'],
        'indexed_at': repo['indexed_at'],
        'stats': repo['stats']
    }
