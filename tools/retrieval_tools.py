"""Retrieval tools for searching code."""
import re
from typing import List, Dict, Optional
from indexing.vector_store import get_vector_store
from indexing.metadata_store import get_metadata_store


def vector_search(question: str, repo_id: str, k: int = 10) -> List[Dict]:
    """
    Perform semantic vector search.
    
    Args:
        question: Search query
        repo_id: Repository ID
        k: Number of results
    
    Returns:
        List of matching chunks
    """
    vector_store = get_vector_store()
    return vector_store.search(question, repo_id, n_results=k)


def lexical_search(keyword: str, repo_id: str, k: int = 10) -> List[Dict]:
    """
    Perform lexical/keyword search.
    
    Args:
        keyword: Search keyword
        repo_id: Repository ID
        k: Number of results
    
    Returns:
        List of matching chunks
    """
    metadata_store = get_metadata_store()
    return metadata_store.search_chunks_lexical(repo_id, keyword, limit=k)


def extract_keywords(question: str) -> List[str]:
    """
    Extract likely keywords from a question.
    
    Args:
        question: Question text
    
    Returns:
        List of keywords
    """
    # Remove common question words
    stop_words = {
        'how', 'what', 'where', 'when', 'why', 'who', 'which', 'is', 'are', 'the',
        'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
        'does', 'do', 'did', 'can', 'could', 'would', 'should', 'will', 'be'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', question.lower())
    
    # Filter and return
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Also look for camelCase or PascalCase words in original question
    camel_words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', question)
    keywords.extend([w.lower() for w in camel_words])
    
    # Look for snake_case
    snake_words = re.findall(r'\b[a-z]+_[a-z_]+\b', question)
    keywords.extend(snake_words)
    
    return list(set(keywords))[:5]  # Return top 5 unique keywords


def hybrid_search(question: str, repo_id: str, k: int = 12) -> List[Dict]:
    """
    Perform hybrid search combining vector and lexical search.
    
    Args:
        question: Search query
        repo_id: Repository ID
        k: Number of results to return
    
    Returns:
        List of deduplicated and ranked chunks
    """
    # Vector search (semantic)
    vector_results = vector_search(question, repo_id, k=k)
    
    # Lexical search (keywords)
    keywords = extract_keywords(question)
    lexical_results = []
    
    for keyword in keywords[:3]:  # Use top 3 keywords
        results = lexical_search(keyword, repo_id, k=k//2)
        lexical_results.extend(results)
    
    # Merge and deduplicate
    merged = merge_and_rerank(vector_results, lexical_results, k=k)
    
    return merged


def merge_and_rerank(
    vector_results: List[Dict],
    lexical_results: List[Dict],
    k: int = 12
) -> List[Dict]:
    """
    Merge vector and lexical search results, deduplicate, and rerank.
    
    Args:
        vector_results: Results from vector search
        lexical_results: Results from lexical search
        k: Number of results to return
    
    Returns:
        Merged and ranked results
    """
    # Create a map of chunk_id to result
    chunk_map = {}
    
    # Add vector results with weight
    for i, result in enumerate(vector_results):
        chunk_id = result['chunk_id']
        score = result.get('score', 0.5)
        
        # Boost score based on rank
        rank_boost = (len(vector_results) - i) / len(vector_results) * 0.3
        
        chunk_map[chunk_id] = {
            **result,
            'combined_score': score + rank_boost,
            'sources': ['vector']
        }
    
    # Add lexical results
    for i, result in enumerate(lexical_results):
        chunk_id = result['chunk_id']
        score = result.get('score', 0.3)
        
        # Normalize lexical score (it's typically much higher)
        normalized_score = min(score / 10.0, 1.0)
        
        # Boost score based on rank
        rank_boost = (len(lexical_results) - i) / len(lexical_results) * 0.2
        
        if chunk_id in chunk_map:
            # Chunk found in both - boost its score significantly
            chunk_map[chunk_id]['combined_score'] += normalized_score + rank_boost + 0.3
            chunk_map[chunk_id]['sources'].append('lexical')
        else:
            chunk_map[chunk_id] = {
                **result,
                'combined_score': normalized_score + rank_boost,
                'sources': ['lexical']
            }
    
    # Sort by combined score
    results = list(chunk_map.values())
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Deduplicate by file span (keep highest scoring if overlapping)
    deduplicated = deduplicate_by_file_span(results)
    
    return deduplicated[:k]


def deduplicate_by_file_span(results: List[Dict], max_chunks: int = 12) -> List[Dict]:
    """
    Deduplicate chunks that overlap in the same file.
    
    Args:
        results: List of search results
        max_chunks: Maximum chunks to return
    
    Returns:
        Deduplicated results
    """
    kept = []
    file_spans = {}  # file_path -> list of (start, end, idx)
    
    for result in results:
        file_path = result.get('file_path', result.get('metadata', {}).get('file_path', ''))
        start = result.get('start_line', result.get('metadata', {}).get('start_line', 0))
        end = result.get('end_line', result.get('metadata', {}).get('end_line', 0))
        
        if not file_path:
            kept.append(result)
            continue
        
        # Check for overlap
        if file_path not in file_spans:
            file_spans[file_path] = []
        
        overlaps = False
        for existing_start, existing_end, existing_idx in file_spans[file_path]:
            # Check if spans overlap significantly (>50%)
            overlap_start = max(start, existing_start)
            overlap_end = min(end, existing_end)
            overlap_size = max(0, overlap_end - overlap_start)
            
            span_size = end - start
            if span_size > 0 and overlap_size / span_size > 0.5:
                overlaps = True
                break
        
        if not overlaps:
            file_spans[file_path].append((start, end, len(kept)))
            kept.append(result)
        
        if len(kept) >= max_chunks:
            break
    
    return kept
