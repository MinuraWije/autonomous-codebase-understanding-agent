"""Tests for the indexing pipeline."""
import pytest
from pathlib import Path
from indexing.chunking import chunk_by_size, count_tokens, CodeChunk
from indexing.loader import filter_files, generate_repo_id, get_language_from_extension


def test_generate_repo_id():
    """Test repo ID generation."""
    url = "https://github.com/test/repo"
    repo_id = generate_repo_id(url)
    
    assert repo_id is not None
    assert len(repo_id) == 12
    assert repo_id == generate_repo_id(url)  # Deterministic


def test_get_language_from_extension():
    """Test language detection from file extension."""
    assert get_language_from_extension(Path("test.py")) == "python"
    assert get_language_from_extension(Path("test.js")) == "javascript"
    assert get_language_from_extension(Path("test.ts")) == "typescript"
    assert get_language_from_extension(Path("test.java")) == "java"
    assert get_language_from_extension(Path("test.go")) == "go"


def test_count_tokens():
    """Test token counting."""
    text = "This is a test"
    tokens = count_tokens(text)
    assert tokens > 0
    assert tokens < 10  # Should be around 4-5


def test_chunk_by_size():
    """Test size-based chunking."""
    content = "\n".join([f"line {i}" for i in range(100)])
    file_path = Path("test.py")
    repo_id = "test123"
    language = "python"
    
    chunks = chunk_by_size(content, file_path, repo_id, language)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, CodeChunk) for chunk in chunks)
    assert all(chunk.repo_id == repo_id for chunk in chunks)
    assert all(chunk.language == language for chunk in chunks)


def test_chunk_line_numbers():
    """Test that chunks preserve line numbers correctly."""
    content = "\n".join([f"line {i}" for i in range(20)])
    file_path = Path("test.py")
    repo_id = "test123"
    language = "python"
    
    chunks = chunk_by_size(content, file_path, repo_id, language)
    
    # First chunk should start at line 1
    assert chunks[0].start_line == 1
    
    # Each chunk should have valid line ranges
    for chunk in chunks:
        assert chunk.start_line > 0
        assert chunk.end_line >= chunk.start_line


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
