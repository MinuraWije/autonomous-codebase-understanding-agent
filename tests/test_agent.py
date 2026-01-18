"""Tests for the agent."""
import pytest
from agent.prompts import extract_citations, extract_json_from_response


def test_extract_citations():
    """Test citation extraction from answer text."""
    answer = """
    Authentication is handled in [auth/middleware.py:10-25] and 
    also uses [utils/crypto.py:5-15] for password hashing.
    """
    
    citations = extract_citations(answer)
    
    assert len(citations) == 2
    assert citations[0]['file_path'] == 'auth/middleware.py'
    assert citations[0]['start_line'] == 10
    assert citations[0]['end_line'] == 25
    assert citations[1]['file_path'] == 'utils/crypto.py'


def test_extract_json_from_response():
    """Test JSON extraction from LLM responses."""
    # Test with clean JSON
    response = '{"key": "value", "number": 42}'
    result = extract_json_from_response(response)
    assert result == {"key": "value", "number": 42}
    
    # Test with markdown code block
    response = '```json\n{"key": "value"}\n```'
    result = extract_json_from_response(response)
    assert result == {"key": "value"}
    
    # Test with surrounding text
    response = 'Here is the JSON: {"key": "value"} and more text'
    result = extract_json_from_response(response)
    assert result == {"key": "value"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
