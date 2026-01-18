"""Evaluation metrics for the codebase understanding agent."""
from typing import List, Dict
import re


def calculate_groundedness(answer: str, citations: List[Dict], retrieved_chunks: List[Dict]) -> float:
    """
    Calculate what percentage of citations are valid.
    
    Args:
        answer: The generated answer
        citations: List of citations from the answer
        retrieved_chunks: Chunks that were retrieved
    
    Returns:
        Groundedness score (0-1)
    """
    if not citations:
        # No citations means we can't verify groundedness
        return 0.0
    
    valid_citations = 0
    
    # Create a map of file:line ranges from retrieved chunks
    retrieved_spans = set()
    for chunk in retrieved_chunks:
        file_path = chunk.get('file_path', chunk.get('metadata', {}).get('file_path', ''))
        start = chunk.get('start_line', chunk.get('metadata', {}).get('start_line', 0))
        end = chunk.get('end_line', chunk.get('metadata', {}).get('end_line', 0))
        retrieved_spans.add((file_path, start, end))
    
    # Check each citation
    for citation in citations:
        cite_file = citation['file_path']
        cite_start = citation['start_line']
        cite_end = citation['end_line']
        
        # Check if this citation overlaps with any retrieved chunk
        for ret_file, ret_start, ret_end in retrieved_spans:
            if cite_file == ret_file:
                # Check for overlap
                if not (cite_end < ret_start or cite_start > ret_end):
                    valid_citations += 1
                    break
    
    return valid_citations / len(citations) if citations else 0.0


def calculate_retrieval_hit_rate(retrieved_chunks: List[Dict], expected_files: List[str]) -> float:
    """
    Calculate if expected files were retrieved.
    
    Args:
        retrieved_chunks: Chunks that were retrieved
        expected_files: List of expected file patterns
    
    Returns:
        Hit rate (0-1)
    """
    if not expected_files:
        return 1.0  # No expectations means success
    
    retrieved_files = set()
    for chunk in retrieved_chunks:
        file_path = chunk.get('file_path', chunk.get('metadata', {}).get('file_path', ''))
        retrieved_files.add(file_path.lower())
    
    hits = 0
    for expected in expected_files:
        expected_lower = expected.lower()
        # Check if any retrieved file contains the expected pattern
        if any(expected_lower in f for f in retrieved_files):
            hits += 1
    
    return hits / len(expected_files) if expected_files else 1.0


def calculate_hallucination_rate(answer: str, retrieved_chunks: List[Dict]) -> float:
    """
    Estimate hallucination rate by checking if answer content appears in chunks.
    This is a rough heuristic.
    
    Args:
        answer: The generated answer
        retrieved_chunks: Chunks that were retrieved
    
    Returns:
        Estimated hallucination rate (0-1, lower is better)
    """
    # Extract key technical terms from answer (simplified)
    answer_lower = answer.lower()
    
    # Remove common words
    stop_words = {
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their'
    }
    
    # Extract words from answer
    answer_words = re.findall(r'\b[a-z_][a-z0-9_]*\b', answer_lower)
    answer_words = [w for w in answer_words if w not in stop_words and len(w) > 3]
    
    if not answer_words:
        return 0.0
    
    # Combine all chunk text
    all_chunk_text = ' '.join([
        chunk.get('text', chunk.get('chunk_text', '')).lower()
        for chunk in retrieved_chunks
    ])
    
    # Count how many answer words appear in chunks
    found_words = sum(1 for word in answer_words if word in all_chunk_text)
    
    # Hallucination rate is inverse of found words ratio
    found_ratio = found_words / len(answer_words) if answer_words else 1.0
    hallucination_rate = 1.0 - found_ratio
    
    return hallucination_rate


def has_citation_format(answer: str) -> bool:
    """Check if answer has proper citation format."""
    pattern = r'\[[^\]]+:\d+-\d+\]'
    return bool(re.search(pattern, answer))


def count_citations(answer: str) -> int:
    """Count the number of citations in an answer."""
    pattern = r'\[[^\]]+:\d+-\d+\]'
    return len(re.findall(pattern, answer))


def calculate_metrics(result: Dict, expected_files: List[str] = None) -> Dict:
    """
    Calculate all metrics for a result.
    
    Args:
        result: Agent result with answer, citations, retrieved_chunks
        expected_files: Expected files for this question
    
    Returns:
        Dictionary of metrics
    """
    answer = result.get('final_answer', result.get('draft_answer', ''))
    citations = result.get('citations', [])
    retrieved_chunks = result.get('retrieved_chunks', [])
    
    metrics = {
        'has_citations': has_citation_format(answer),
        'citation_count': count_citations(answer),
        'groundedness': calculate_groundedness(answer, citations, retrieved_chunks),
        'hallucination_rate': calculate_hallucination_rate(answer, retrieved_chunks),
        'chunks_retrieved': len(retrieved_chunks),
    }
    
    if expected_files:
        metrics['retrieval_hit_rate'] = calculate_retrieval_hit_rate(retrieved_chunks, expected_files)
    
    return metrics
