"""Retrieval tools for searching code."""
import re
from typing import List, Dict, Optional
from indexing.vector_store import get_vector_store
from indexing.metadata_store import get_metadata_store
from core.constants import (
    STOP_WORDS, 
    VECTOR_SEARCH_WEIGHT, 
    LEXICAL_SEARCH_WEIGHT, 
    RANK_BOOST_FACTOR, 
    OVERLAP_THRESHOLD,
    QUERY_EXPANSIONS,
    MULTI_TERM_MATCH_BOOST,
    TEST_FILE_PENALTY,
    DOC_FILE_PENALTY,
    PATH_DEPTH_BOOST,
    TEST_FILE_PATTERNS,
    DOC_FILE_PATTERNS
)


def expand_query_for_vector_search(question: str) -> str:
    """
    Expand query with synonyms for better vector search.
    Adds related terms to help semantic search find more relevant results.
    
    Args:
        question: Original question
    
    Returns:
        Expanded query string
    """
    # Extract base keywords
    keywords = extract_keywords(question, expand=False)
    
    if not keywords:
        return question
    
    # Get expansions for top keywords
    expanded_terms = expand_query_terms(keywords[:3])  # Expand top 3 keywords
    
    # Build expanded query: original + key expansions
    expanded_parts = [question]
    
    # Add most relevant expansions (limit to avoid query bloat)
    for term in expanded_terms[:5]:
        if term not in question.lower():
            expanded_parts.append(term)
    
    return ' '.join(expanded_parts[:3])  # Limit to 3 parts to keep query focused


def vector_search(question: str, repo_id: str, k: int = 10, use_expansion: bool = True) -> List[Dict]:
    """
    Perform semantic vector search with optional query expansion.
    
    Args:
        question: Search query
        repo_id: Repository ID
        k: Number of results
        use_expansion: Whether to expand query with synonyms
    
    Returns:
        List of matching chunks
    """
    vector_store = get_vector_store()
    
    # Use expanded query for better semantic matching
    search_query = expand_query_for_vector_search(question) if use_expansion else question
    
    return vector_store.search(search_query, repo_id, n_results=k)


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


def expand_query_terms(terms: List[str]) -> List[str]:
    """
    Expand query terms with synonyms and related terms.
    
    Args:
        terms: List of base terms
    
    Returns:
        Expanded list of terms including synonyms
    """
    expanded = set(terms)
    
    for term in terms:
        term_lower = term.lower()
        # Check for exact match in expansions
        if term_lower in QUERY_EXPANSIONS:
            expanded.update(QUERY_EXPANSIONS[term_lower])
        # Check for partial matches (e.g., "auth" in "authentication")
        else:
            for key, synonyms in QUERY_EXPANSIONS.items():
                if key in term_lower or term_lower in key:
                    expanded.update(synonyms)
                    expanded.add(key)
    
    return list(expanded)


def extract_keywords(question: str, expand: bool = True) -> List[str]:
    """
    Extract likely keywords from a question with optional expansion.
    Extracts technical terms, function names, class names, and common keywords.
    
    Args:
        question: Question text
        expand: Whether to expand terms with synonyms
    
    Returns:
        List of keywords (expanded if requested)
    """
    # Extract basic words
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', question.lower())
    
    # Filter stop words and short words
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    # Extract camelCase or PascalCase (likely function/class names)
    camel_words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', question)
    # Split camelCase into components (e.g., "getUserData" -> ["get", "user", "data"])
    for camel_word in camel_words:
        # Split camelCase
        split_words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', camel_word)
        keywords.extend([w.lower() for w in split_words if len(w) > 2])
        keywords.append(camel_word.lower())  # Also keep the full word
    
    # Extract snake_case (likely variable/function names)
    snake_words = re.findall(r'\b[a-z]+_[a-z_]+\b', question)
    # Split snake_case into components
    for snake_word in snake_words:
        parts = snake_word.split('_')
        keywords.extend([p for p in parts if len(p) > 2])
        keywords.append(snake_word)  # Also keep the full word
    
    # Extract technical patterns (e.g., "API", "HTTP", "JSON", "SQL")
    tech_patterns = re.findall(r'\b[A-Z]{2,}\b', question)
    keywords.extend([p.lower() for p in tech_patterns])
    
    # Remove duplicates while preserving order
    unique_keywords = []
    seen = set()
    for kw in keywords:
        if kw not in seen:
            unique_keywords.append(kw)
            seen.add(kw)
    
    # Expand with synonyms if requested
    if expand and unique_keywords:
        expanded = expand_query_terms(unique_keywords)
        # Prioritize original keywords, then add expansions
        result = unique_keywords + [e for e in expanded if e not in seen]
        return result[:8]  # Return top 8 (increased from 5 to accommodate expansions)
    
    return unique_keywords[:5]  # Return top 5 if not expanding


def hybrid_search(question: str, repo_id: str, k: int = 12) -> List[Dict]:
    """
    Perform hybrid search combining vector and lexical search with query expansion.
    
    Args:
        question: Search query
        repo_id: Repository ID
        k: Number of results to return
    
    Returns:
        List of deduplicated and ranked chunks
    """
    # Vector search (semantic) - uses original question for semantic understanding
    vector_results = vector_search(question, repo_id, k=k)
    
    # Lexical search (keywords) - uses expanded keywords for better coverage
    keywords = extract_keywords(question, expand=True)
    lexical_results = []
    
    # Use top 4 keywords (increased from 3 to leverage expansions)
    # Prioritize original keywords first, then expansions
    for keyword in keywords[:4]:
        results = lexical_search(keyword, repo_id, k=k//2)
        lexical_results.extend(results)
    
    # Extract base keywords (non-expanded) for multi-term matching
    base_keywords = extract_keywords(question, expand=False)
    
    # Merge and deduplicate with reranking
    merged = merge_and_rerank(
        vector_results, 
        lexical_results, 
        k=k,
        query_keywords=base_keywords,
        original_question=question
    )
    
    return merged


def _is_test_file(file_path: str) -> bool:
    """Check if file is a test file."""
    file_path_lower = file_path.lower()
    return any(pattern in file_path_lower for pattern in TEST_FILE_PATTERNS)


def _is_doc_file(file_path: str) -> bool:
    """Check if file is a documentation file."""
    file_path_lower = file_path.lower()
    return any(pattern in file_path_lower for pattern in DOC_FILE_PATTERNS)


def _calculate_path_depth(file_path: str) -> int:
    """
    Calculate path depth (number of directory separators).
    Root level files have depth 0.
    """
    # Normalize path separators
    normalized = file_path.replace('\\', '/')
    # Count separators, excluding leading/trailing
    parts = [p for p in normalized.split('/') if p]
    return max(0, len(parts) - 1)  # Subtract 1 for filename


def _count_keyword_matches(chunk_text: str, keywords: List[str]) -> int:
    """
    Count how many keywords appear in the chunk text.
    
    Args:
        chunk_text: Text content of the chunk
        keywords: List of keywords to search for
    
    Returns:
        Number of unique keywords found
    """
    if not keywords or not chunk_text:
        return 0
    
    chunk_lower = chunk_text.lower()
    matches = 0
    
    for keyword in keywords:
        # Check for whole word matches
        keyword_lower = keyword.lower()
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(keyword_lower) + r'\b'
        if re.search(pattern, chunk_lower):
            matches += 1
    
    return matches


def merge_and_rerank(
    vector_results: List[Dict],
    lexical_results: List[Dict],
    k: int = 12,
    query_keywords: Optional[List[str]] = None,
    original_question: Optional[str] = None
) -> List[Dict]:
    """
    Merge vector and lexical search results, deduplicate, and rerank with advanced scoring.
    
    Args:
        vector_results: Results from vector search
        lexical_results: Results from lexical search
        k: Number of results to return
        query_keywords: Base keywords from query (for multi-term matching)
        original_question: Original question text (for context)
    
    Returns:
        Merged and ranked results
    """
    # Create a map of chunk_id to result
    chunk_map = {}
    
    # Add vector results with weight
    for i, result in enumerate(vector_results):
        chunk_id = result['chunk_id']
        score = result.get('score', 0.5) * VECTOR_SEARCH_WEIGHT
        
        # Boost score based on rank
        rank_boost = (len(vector_results) - i) / len(vector_results) * RANK_BOOST_FACTOR
        
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
        normalized_score = min(score / 10.0, 1.0) * LEXICAL_SEARCH_WEIGHT
        
        # Boost score based on rank
        rank_boost = (len(lexical_results) - i) / len(lexical_results) * (RANK_BOOST_FACTOR * 0.67)
        
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
    
    # Apply advanced reranking factors
    results = list(chunk_map.values())
    
    for result in results:
        file_path = result.get('file_path', result.get('metadata', {}).get('file_path', ''))
        chunk_text = result.get('text', result.get('chunk_text', ''))
        
        # Multi-term matching boost
        if query_keywords and len(query_keywords) > 1:
            matches = _count_keyword_matches(chunk_text, query_keywords)
            if matches > 1:
                # Boost for each additional keyword match beyond the first
                additional_matches = matches - 1
                result['combined_score'] += additional_matches * MULTI_TERM_MATCH_BOOST
        
        # File type penalties (only apply if we have keywords suggesting implementation search)
        if query_keywords and file_path:
            # Check if question seems to be about implementation (not testing)
            is_implementation_query = (
                original_question and 
                not any(word in original_question.lower() for word in ['test', 'spec', 'example', 'sample'])
            )
            
            if is_implementation_query:
                if _is_test_file(file_path):
                    result['combined_score'] += TEST_FILE_PENALTY
                elif _is_doc_file(file_path):
                    result['combined_score'] += DOC_FILE_PENALTY
        
        # Path depth boost (prefer files closer to root, up to 3 levels)
        if file_path:
            depth = _calculate_path_depth(file_path)
            if depth <= 3:
                # Boost for shallower files (depth 0 gets max boost, depth 3 gets min)
                depth_boost = (3 - depth) * PATH_DEPTH_BOOST
                result['combined_score'] += depth_boost
    
    # Sort by combined score
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
            # Check if spans overlap significantly
            overlap_start = max(start, existing_start)
            overlap_end = min(end, existing_end)
            overlap_size = max(0, overlap_end - overlap_start)
            
            span_size = end - start
            if span_size > 0 and overlap_size / span_size > OVERLAP_THRESHOLD:
                overlaps = True
                break
        
        if not overlaps:
            file_spans[file_path].append((start, end, len(kept)))
            kept.append(result)
        
        if len(kept) >= max_chunks:
            break
    
    return kept
