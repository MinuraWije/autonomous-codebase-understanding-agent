"""Query variation generation for multi-query retrieval."""
from typing import List, Dict, Optional, Set
from core.llm_service import LLMServiceFactory
from core.error_handler import safe_execute
import json


def generate_query_variations(
    question: str,
    num_variations: int = 5,
    use_llm: bool = True
) -> List[str]:
    """
    Generate multiple query variations from the original question.
    
    Creates diverse queries that explore the question from different angles:
    - Different phrasings and synonyms
    - Different levels of specificity
    - Different aspects of the question
    - Technical vs. conceptual perspectives
    
    Args:
        question: Original question
        num_variations: Number of variations to generate (3-5 recommended)
        use_llm: Whether to use LLM for generation (fallback to rule-based if False)
    
    Returns:
        List of query variations (includes original question)
    """
    if not question or not question.strip():
        return [question]
    
    # Always include original question
    variations = [question]
    
    if use_llm:
        llm_variations = _generate_llm_variations(question, num_variations - 1)
        if llm_variations:
            variations.extend(llm_variations)
    
    # Fallback or supplement with rule-based variations
    rule_variations = _generate_rule_based_variations(question, num_variations - len(variations))
    variations.extend(rule_variations)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        v_lower = v.lower().strip()
        if v_lower and v_lower not in seen:
            seen.add(v_lower)
            unique_variations.append(v)
    
    # Limit to requested number
    return unique_variations[:num_variations]


def _generate_llm_variations(question: str, num_variations: int) -> List[str]:
    """
    Generate query variations using LLM.
    
    Args:
        question: Original question
        num_variations: Number of variations to generate
    
    Returns:
        List of query variations
    """
    if num_variations <= 0:
        return []
    
    prompt = f"""Generate {num_variations} diverse search query variations for this question about code.

Original Question: {question}

Create queries that:
1. Use different phrasings and synonyms
2. Explore different aspects or angles
3. Vary specificity (some more general, some more specific)
4. Use technical terminology vs. plain language
5. Focus on different components (functions, classes, patterns, etc.)

OUTPUT FORMAT: Output ONLY valid JSON array of strings:
["query variation 1", "query variation 2", ...]

EXAMPLES:

Question: "Where is user authentication handled?"
[
  "user authentication implementation",
  "login handler code",
  "JWT token validation logic",
  "session management setup",
  "authentication middleware"
]

Question: "How does error handling work?"
[
  "error handling implementation",
  "exception catching and processing",
  "error response formatting",
  "try catch blocks usage",
  "error logging and reporting"
]

Now generate {num_variations} variations for the question above. Output ONLY valid JSON array:"""
    
    try:
        llm_service = LLMServiceFactory.create_planner_service()
        response = safe_execute(
            lambda: llm_service.invoke_text(prompt),
            default_return="[]"
        )
        
        # Parse JSON response
        variations = _extract_json_array(response)
        
        # Validate and clean variations
        cleaned = []
        for v in variations:
            if isinstance(v, str) and v.strip() and len(v.strip()) > 5:
                cleaned.append(v.strip())
        
        return cleaned[:num_variations]
    except Exception:
        return []


def _extract_json_array(text: str) -> List[str]:
    """Extract JSON array from text response."""
    import re
    
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON array
    try:
        # Try parsing directly
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    
    # Try to find array pattern
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to extract quoted strings
    quoted_strings = re.findall(r'"([^"]+)"', text)
    if quoted_strings:
        return quoted_strings
    
    return []


def _generate_rule_based_variations(question: str, num_variations: int) -> List[str]:
    """
    Generate query variations using rule-based heuristics.
    
    Args:
        question: Original question
        num_variations: Number of variations to generate
    
    Returns:
        List of query variations
    """
    if num_variations <= 0:
        return []
    
    variations = []
    question_lower = question.lower()
    
    # Strategy 1: Add implementation-focused terms
    if any(word in question_lower for word in ['how', 'where', 'what']):
        # Add "implementation" or "code" if not present
        if 'implementation' not in question_lower and 'code' not in question_lower:
            variations.append(f"{question} implementation")
            variations.append(f"{question} code")
    
    # Strategy 2: Extract key terms and create focused queries
    key_terms = _extract_key_terms(question)
    if len(key_terms) >= 2:
        # Create queries focusing on different term combinations
        for i, term in enumerate(key_terms[:3]):  # Use top 3 terms
            if term not in question_lower:
                variations.append(f"{term} {question}")
    
    # Strategy 3: Add technical context
    tech_contexts = ['function', 'class', 'module', 'handler', 'service']
    for context in tech_contexts:
        if context not in question_lower and len(variations) < num_variations:
            variations.append(f"{question} {context}")
    
    # Strategy 4: Simplify question (remove question words)
    simplified = _simplify_question(question)
    if simplified and simplified.lower() != question_lower:
        variations.append(simplified)
    
    # Strategy 5: Add action verbs
    action_verbs = ['find', 'locate', 'search', 'get', 'retrieve']
    for verb in action_verbs:
        if verb not in question_lower and len(variations) < num_variations:
            # Only add if question doesn't already have an action verb
            if not any(v in question_lower for v in ['find', 'locate', 'search', 'get', 'retrieve', 'how', 'where']):
                variations.append(f"{verb} {question}")
    
    # Remove duplicates and limit
    seen = set()
    unique_variations = []
    for v in variations:
        v_lower = v.lower().strip()
        if v_lower and v_lower not in seen and v_lower != question.lower():
            seen.add(v_lower)
            unique_variations.append(v)
    
    return unique_variations[:num_variations]


def _extract_key_terms(question: str) -> List[str]:
    """Extract key technical terms from question."""
    import re
    
    # Remove question words
    question_clean = re.sub(r'\b(how|what|where|when|why|which|who|is|are|does|do|can|could|would|should)\b', '', question.lower())
    
    # Extract technical terms (camelCase, PascalCase, snake_case, or technical words)
    terms = []
    
    # CamelCase/PascalCase
    camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', question)
    terms.extend([t.lower() for t in camel_case])
    
    # Technical acronyms (2-4 uppercase letters)
    acronyms = re.findall(r'\b[A-Z]{2,4}\b', question)
    terms.extend([a.lower() for a in acronyms])
    
    # Multi-word technical phrases (3+ chars per word)
    phrases = re.findall(r'\b[a-z]{3,}(?:\s+[a-z]{3,})+\b', question_clean)
    terms.extend(phrases[:3])  # Limit phrases
    
    # Single important words (4+ chars, not common words)
    common_words = {'this', 'that', 'with', 'from', 'into', 'over', 'under', 'after', 'before'}
    words = re.findall(r'\b[a-z]{4,}\b', question_clean)
    terms.extend([w for w in words if w not in common_words][:5])
    
    return list(set(terms))[:8]  # Return top 8 unique terms


def _simplify_question(question: str) -> str:
    """Simplify question by removing question words and making it a search query."""
    import re
    
    # Remove question words at start
    simplified = re.sub(r'^(how|what|where|when|why|which|who)\s+', '', question, flags=re.IGNORECASE)
    
    # Remove trailing question mark
    simplified = simplified.rstrip('?')
    
    # Remove "is" or "are" at start if present
    simplified = re.sub(r'^(is|are)\s+', '', simplified, flags=re.IGNORECASE)
    
    return simplified.strip()


def rewrite_queries_based_on_results(
    original_queries: List[str],
    retrieved_chunks: List[Dict],
    question: str,
    max_new_queries: int = 3
) -> List[str]:
    """
    Rewrite queries based on previous retrieval results.
    
    Analyzes what was found and generates new queries to fill gaps or explore
    related areas that weren't well covered.
    
    Args:
        original_queries: Queries that were already used
        retrieved_chunks: Chunks retrieved from previous queries
        question: Original question
        max_new_queries: Maximum number of new queries to generate
    
    Returns:
        List of rewritten/new queries
    """
    if not retrieved_chunks or max_new_queries <= 0:
        return []
    
    # Analyze what was found
    found_files = set()
    found_symbols = set()
    found_keywords = set()
    
    for chunk in retrieved_chunks:
        file_path = chunk.get('file_path', chunk.get('metadata', {}).get('file_path', ''))
        if file_path:
            found_files.add(file_path.split('/')[-1].split('\\')[-1])
        
        symbol = chunk.get('symbol_name', chunk.get('metadata', {}).get('symbol_name'))
        if symbol:
            found_symbols.add(symbol)
        
        # Extract keywords from chunk text
        chunk_text = chunk.get('text', chunk.get('chunk_text', ''))
        if chunk_text:
            keywords = _extract_key_terms(chunk_text[:500])  # First 500 chars
            found_keywords.update(keywords[:5])
    
    # Generate queries that explore gaps
    new_queries = []
    
    # Strategy 1: Query for related files/modules not yet found
    if found_files:
        # Query for related modules
        for file in list(found_files)[:2]:  # Use top 2 files
            file_base = file.split('.')[0]  # Remove extension
            if file_base and file_base not in question.lower():
                new_queries.append(f"{question} {file_base} related")
    
    # Strategy 2: Query for related symbols/functions
    if found_symbols and len(new_queries) < max_new_queries:
        for symbol in list(found_symbols)[:2]:  # Use top 2 symbols
            if symbol.lower() not in question.lower():
                new_queries.append(f"{question} {symbol}")
    
    # Strategy 3: Query for related concepts from found keywords
    if found_keywords and len(new_queries) < max_new_queries:
        for keyword in list(found_keywords)[:3]:
            if keyword not in question.lower() and len(keyword) > 3:
                new_queries.append(f"{question} {keyword}")
    
    # Strategy 4: Use LLM to generate gap-filling queries
    if len(new_queries) < max_new_queries:
        llm_queries = _generate_gap_filling_queries(
            question,
            original_queries,
            found_files,
            found_symbols,
            max_new_queries - len(new_queries)
        )
        new_queries.extend(llm_queries)
    
    # Remove duplicates and limit
    seen = set()
    unique_queries = []
    for q in new_queries:
        q_lower = q.lower().strip()
        if q_lower and q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)
    
    return unique_queries[:max_new_queries]


def _generate_gap_filling_queries(
    question: str,
    original_queries: List[str],
    found_files: Set[str],
    found_symbols: Set[str],
    num_queries: int
) -> List[str]:
    """Generate queries to fill gaps using LLM."""
    if num_queries <= 0:
        return []
    
    found_files_str = ', '.join(list(found_files)[:5]) if found_files else "none"
    found_symbols_str = ', '.join(list(found_symbols)[:5]) if found_symbols else "none"
    
    prompt = f"""Generate {num_queries} new search queries to find additional relevant code.

Original Question: {question}
Queries Already Used: {', '.join(original_queries[:3])}
Files Found: {found_files_str}
Symbols Found: {found_symbols_str}

Create queries that:
1. Explore related areas not yet covered
2. Use different terminology or synonyms
3. Focus on complementary aspects
4. Search for related functions, classes, or patterns

OUTPUT FORMAT: Output ONLY valid JSON array:
["new query 1", "new query 2", ...]

Generate {num_queries} queries:"""
    
    try:
        llm_service = LLMServiceFactory.create_planner_service()
        response = safe_execute(
            lambda: llm_service.invoke_text(prompt),
            default_return="[]"
        )
        
        queries = _extract_json_array(response)
        return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:num_queries]
    except Exception:
        return []
