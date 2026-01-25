"""Context window optimization for managing chunk prioritization and truncation."""
from typing import List, Dict, Optional, Tuple
from indexing.chunking import count_tokens
from core.constants import (
    DEFAULT_MAX_CITATIONS,
    KEY_FILE_PATTERNS,
    TEST_FILE_PATTERNS,
    DOC_FILE_PATTERNS,
    DEFAULT_CONTEXT_WINDOW,
    RESERVE_TOKENS_FOR_PROMPT,
    RESERVE_TOKENS_FOR_RESPONSE,
    MIN_CHUNK_TOKENS_AFTER_TRUNCATION
)


def optimize_chunks_for_context(
    chunks: List[Dict],
    max_context_tokens: int = DEFAULT_CONTEXT_WINDOW,
    reserve_prompt_tokens: int = RESERVE_TOKENS_FOR_PROMPT,
    reserve_response_tokens: int = RESERVE_TOKENS_FOR_RESPONSE,
    question: Optional[str] = None
) -> List[Dict]:
    """
    Optimize chunks for context window by prioritizing and truncating intelligently.
    
    Args:
        chunks: List of chunk dictionaries with 'chunk_text' or 'text', scores, and metadata
        max_context_tokens: Maximum context window size in tokens
        reserve_prompt_tokens: Tokens to reserve for prompt template
        reserve_response_tokens: Tokens to reserve for LLM response
        question: Optional question text for relevance-based prioritization
    
    Returns:
        Optimized list of chunks (prioritized and truncated if needed)
    """
    if not chunks:
        return chunks
    
    # Calculate available tokens for chunks
    available_tokens = max_context_tokens - reserve_prompt_tokens - reserve_response_tokens
    
    # Step 1: Prioritize chunks
    prioritized_chunks = _prioritize_chunks(chunks, question)
    
    # Step 2: Calculate total tokens needed
    total_tokens = sum(_get_chunk_token_count(chunk) for chunk in prioritized_chunks)
    
    # Step 3: If within limit, return as-is (clean up internal fields)
    if total_tokens <= available_tokens:
        return _clean_chunks(prioritized_chunks)
    
    # Step 4: Select and truncate chunks to fit
    optimized_chunks = _select_and_truncate_chunks(
        prioritized_chunks,
        available_tokens,
        question
    )
    
    return _clean_chunks(optimized_chunks)


def _prioritize_chunks(chunks: List[Dict], question: Optional[str] = None) -> List[Dict]:
    """
    Prioritize chunks based on relevance scores and importance factors.
    
    Priority factors:
    1. Combined score from retrieval (if available)
    2. Whether chunk appears in both vector and lexical results
    3. File importance (key files like main.py, app.py)
    4. Symbol importance (functions/classes vs plain code)
    5. File type (prefer implementation over tests/docs for implementation queries)
    
    Args:
        chunks: List of chunks to prioritize
        question: Optional question for context-aware prioritization
    
    Returns:
        Prioritized list of chunks
    """
    # Calculate priority score for each chunk
    prioritized = []
    
    for chunk in chunks:
        priority_score = _calculate_priority_score(chunk, question)
        prioritized.append({
            **chunk,
            '_priority_score': priority_score
        })
    
    # Sort by priority score (highest first)
    prioritized.sort(key=lambda x: x['_priority_score'], reverse=True)
    
    return prioritized


def _calculate_priority_score(chunk: Dict, question: Optional[str] = None) -> float:
    """
    Calculate priority score for a chunk.
    
    Args:
        chunk: Chunk dictionary
        question: Optional question for context
    
    Returns:
        Priority score (higher is better)
    """
    score = 0.0
    
    # Base score from retrieval (if available)
    combined_score = chunk.get('combined_score', 0.0)
    if combined_score > 0:
        score += combined_score * 10.0  # Scale up retrieval scores
    
    # Boost for chunks found in both vector and lexical search
    sources = chunk.get('sources', [])
    if isinstance(sources, list) and len(sources) > 1:
        score += 2.0  # Significant boost for multi-source matches
    
    # Boost for key files
    file_path = chunk.get('file_path', chunk.get('metadata', {}).get('file_path', ''))
    if file_path:
        filename = file_path.split('/')[-1].split('\\')[-1].lower()
        if any(pattern.lower() in filename for pattern in KEY_FILE_PATTERNS):
            score += 1.5
    
    # Boost for chunks with symbols (functions/classes)
    symbol_name = chunk.get('symbol_name', chunk.get('metadata', {}).get('symbol_name'))
    if symbol_name:
        score += 1.0
    
    # Context-aware adjustments based on question
    if question:
        question_lower = question.lower()
        
        # Penalize test files for implementation queries
        if any(word in question_lower for word in ['how', 'where', 'what', 'implement']):
            if _is_test_file(file_path):
                score -= 0.5
            elif _is_doc_file(file_path):
                score -= 0.3
        
        # Boost chunks that match question keywords
        chunk_text = chunk.get('text', chunk.get('chunk_text', '')).lower()
        question_words = set(word for word in question_lower.split() if len(word) > 3)
        chunk_words = set(word for word in chunk_text.split() if len(word) > 3)
        matching_words = question_words.intersection(chunk_words)
        if matching_words:
            score += len(matching_words) * 0.2
    
    # Boost for chunks with higher relevance scores from vector search
    vector_score = chunk.get('score', 0.0)
    if 'vector' in sources:
        score += vector_score * 5.0
    
    return score


def _is_test_file(file_path: str) -> bool:
    """Check if file is a test file."""
    if not file_path:
        return False
    file_path_lower = file_path.lower()
    return any(pattern in file_path_lower for pattern in TEST_FILE_PATTERNS)


def _is_doc_file(file_path: str) -> bool:
    """Check if file is a documentation file."""
    if not file_path:
        return False
    file_path_lower = file_path.lower()
    return any(pattern in file_path_lower for pattern in DOC_FILE_PATTERNS)


def _select_and_truncate_chunks(
    prioritized_chunks: List[Dict],
    available_tokens: int,
    question: Optional[str] = None
) -> List[Dict]:
    """
    Select chunks that fit within token limit and truncate if needed.
    
    Strategy:
    1. Keep top-priority chunks fully
    2. For remaining chunks, intelligently truncate to fit
    3. Preserve important parts: function signatures, class definitions, key logic
    
    Args:
        prioritized_chunks: Prioritized chunks
        available_tokens: Available token budget
        question: Optional question for context
    
    Returns:
        Selected and truncated chunks
    """
    selected = []
    used_tokens = 0
    
    for chunk in prioritized_chunks:
        chunk_tokens = _get_chunk_token_count(chunk)
        
        # If chunk fits fully, add it
        if used_tokens + chunk_tokens <= available_tokens:
            selected.append(chunk)
            used_tokens += chunk_tokens
        else:
            # Try to fit a truncated version
            remaining_tokens = available_tokens - used_tokens
            
            # Reserve minimum tokens for essential chunks
            if remaining_tokens < MIN_CHUNK_TOKENS_AFTER_TRUNCATION:
                # Not enough space for even a minimal chunk
                break
            
            # Truncate chunk intelligently
            truncated_chunk = _truncate_chunk_intelligently(chunk, remaining_tokens, question)
            if truncated_chunk:
                selected.append(truncated_chunk)
                used_tokens += _get_chunk_token_count(truncated_chunk)
            
            # Stop if we've used most of the budget
            if used_tokens >= available_tokens * 0.95:
                break
    
    return selected


def _truncate_chunk_intelligently(
    chunk: Dict,
    max_tokens: int,
    question: Optional[str] = None
) -> Optional[Dict]:
    """
    Intelligently truncate a chunk while preserving important parts.
    
    Preserves:
    - Function/class signatures
    - Docstrings
    - Key logic sections
    - Imports (if at start)
    
    Args:
        chunk: Chunk to truncate
        max_tokens: Maximum tokens for truncated chunk
        question: Optional question for context
    
    Returns:
        Truncated chunk or None if too small
    """
    chunk_text = chunk.get('text', chunk.get('chunk_text', ''))
    if not chunk_text:
        return None
    
    # If chunk is already small enough, return as-is
    if count_tokens(chunk_text) <= max_tokens:
        return chunk
    
    lines = chunk_text.split('\n')
    
    # Strategy: Keep beginning (signatures, imports, docstrings) and end (return/logic)
    # This preserves function structure while truncating middle sections
    
    # Find important sections
    important_lines = _identify_important_lines(lines, question)
    
    # Build truncated version
    truncated_lines = []
    used_tokens = 0
    added_lines = set()
    
    # First pass: Add important lines
    for line_idx in important_lines:
        if line_idx < len(lines):
            line = lines[line_idx]
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens <= max_tokens:
                truncated_lines.append((line_idx, line))
                added_lines.add(line_idx)
                used_tokens += line_tokens
    
    # Second pass: Fill remaining space with context around important lines
    # Add lines before and after important sections
    important_indices = set(important_lines)
    for line_idx in sorted(important_indices):
        # Add a few lines before
        for i in range(max(0, line_idx - 2), line_idx):
            if i not in added_lines and i < len(lines):
                line_text = lines[i]
                line_tokens = count_tokens(line_text)
                if used_tokens + line_tokens <= max_tokens:
                    truncated_lines.append((i, line_text))
                    added_lines.add(i)
                    used_tokens += line_tokens
        
        # Add a few lines after
        for i in range(line_idx + 1, min(len(lines), line_idx + 3)):
            if i not in added_lines:
                line_text = lines[i]
                line_tokens = count_tokens(line_text)
                if used_tokens + line_tokens <= max_tokens:
                    truncated_lines.append((i, line_text))
                    added_lines.add(i)
                    used_tokens += line_tokens
    
    # Third pass: Fill with remaining lines in order if space allows
    for i, line in enumerate(lines):
        if i not in added_lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens <= max_tokens:
                truncated_lines.append((i, line))
                used_tokens += line_tokens
            else:
                break
    
    # Sort by line index and build truncated text
    truncated_lines.sort(key=lambda x: x[0])
    truncated_text = '\n'.join(line for _, line in truncated_lines)
    
    # Add truncation indicator if we removed content
    if len(truncated_lines) < len(lines):
        truncated_text += "\n# ... [truncated for context window] ..."
    
    # Create truncated chunk
    truncated_chunk = {
        **chunk,
        'text': truncated_text,
        'chunk_text': truncated_text,
        '_truncated': True,
        '_original_token_count': count_tokens(chunk_text),
        '_truncated_token_count': count_tokens(truncated_text)
    }
    
    return truncated_chunk


def _identify_important_lines(lines: List[str], question: Optional[str] = None) -> List[int]:
    """
    Identify important lines in a chunk (signatures, docstrings, key logic).
    
    Args:
        lines: List of code lines
        question: Optional question for context
    
    Returns:
        List of line indices (0-based) that are important
    """
    important = []
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Function/class definitions
        if any(line_stripped.startswith(keyword) for keyword in ['def ', 'class ', 'async def ']):
            important.append(i)
        
        # Docstrings
        if '"""' in line_stripped or "'''" in line_stripped:
            important.append(i)
        
        # Return statements (key logic)
        if line_stripped.startswith('return '):
            important.append(i)
        
        # Import statements (usually at top)
        if line_stripped.startswith('import ') or line_stripped.startswith('from '):
            if i < 20:  # Only top imports
                important.append(i)
        
        # Decorators
        if line_stripped.startswith('@'):
            important.append(i)
        
        # Question keyword matches (if question provided)
        if question:
            question_lower = question.lower()
            line_lower = line_stripped.lower()
            question_words = set(word for word in question_lower.split() if len(word) > 3)
            line_words = set(word for word in line_lower.split() if len(word) > 3)
            if question_words.intersection(line_words):
                important.append(i)
    
    # Always include first and last lines for context
    if lines:
        if 0 not in important:
            important.append(0)
        if len(lines) - 1 not in important:
            important.append(len(lines) - 1)
    
    return sorted(set(important))


def _get_chunk_token_count(chunk: Dict) -> int:
    """
    Get token count for a chunk.
    
    Args:
        chunk: Chunk dictionary
    
    Returns:
        Token count
    """
    text = chunk.get('text', chunk.get('chunk_text', ''))
    if not text:
        return 0
    return count_tokens(text)


def _clean_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Remove internal priority score field but keep useful metadata like _truncated.
    
    Args:
        chunks: List of chunks with internal fields
    
    Returns:
        Cleaned chunks without _priority_score but keeping _truncated
    """
    cleaned = []
    for chunk in chunks:
        cleaned_chunk = {k: v for k, v in chunk.items() if k != '_priority_score'}
        cleaned.append(cleaned_chunk)
    return cleaned


def estimate_prompt_tokens(question: str, chunks: List[Dict]) -> int:
    """
    Estimate total tokens needed for a prompt.
    
    Args:
        question: Question text
        chunks: List of chunks
    
    Returns:
        Estimated token count
    """
    question_tokens = count_tokens(question)
    chunk_tokens = sum(_get_chunk_token_count(chunk) for chunk in chunks)
    
    # Add overhead for prompt template (headers, formatting, etc.)
    template_overhead = 300
    
    return question_tokens + chunk_tokens + template_overhead
