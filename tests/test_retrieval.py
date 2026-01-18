"""Tests for retrieval tools."""
import pytest
from tools.retrieval_tools import extract_keywords, merge_and_rerank


def test_extract_keywords():
    """Test keyword extraction from questions."""
    question = "Where is authentication handled in the AuthService?"
    keywords = extract_keywords(question)
    
    assert len(keywords) > 0
    assert 'authentication' in keywords or 'authservice' in keywords.lower()
    
    # Should not include common words
    assert 'where' not in keywords
    assert 'the' not in keywords


def test_extract_keywords_camelcase():
    """Test extraction of camelCase identifiers."""
    question = "How does UserManager handle authentication?"
    keywords = extract_keywords(question)
    
    assert any('usermanager' in k.lower() for k in keywords)


def test_merge_and_rerank():
    """Test merging and reranking of search results."""
    vector_results = [
        {
            'chunk_id': 'chunk1',
            'score': 0.9,
            'file_path': 'auth.py',
            'start_line': 10,
            'end_line': 20,
            'text': 'auth code'
        },
        {
            'chunk_id': 'chunk2',
            'score': 0.7,
            'file_path': 'user.py',
            'start_line': 5,
            'end_line': 15,
            'text': 'user code'
        }
    ]
    
    lexical_results = [
        {
            'chunk_id': 'chunk1',  # Same as vector result
            'score': 5.0,
            'file_path': 'auth.py',
            'start_line': 10,
            'end_line': 20,
            'text': 'auth code'
        },
        {
            'chunk_id': 'chunk3',
            'score': 3.0,
            'file_path': 'middleware.py',
            'start_line': 1,
            'end_line': 10,
            'text': 'middleware code'
        }
    ]
    
    merged = merge_and_rerank(vector_results, lexical_results, k=10)
    
    # Should have 3 unique chunks (chunk1 appears in both)
    assert len(merged) == 3
    
    # chunk1 should be first (appears in both)
    assert merged[0]['chunk_id'] == 'chunk1'
    
    # All chunks should have combined_score
    assert all('combined_score' in chunk for chunk in merged)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
