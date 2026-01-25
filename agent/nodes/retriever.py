"""Retriever node for the agent."""
from agent.state import AgentState
from tools.retrieval_tools import hybrid_search, merge_and_rerank
from core.constants import (
    DEFAULT_MAX_CHUNKS_PER_QUERY, 
    DEFAULT_MAX_CITATIONS,
    DEFAULT_QUERY_VARIATIONS
)
from core.query_variation import (
    generate_query_variations,
    rewrite_queries_based_on_results
)
from app.config import settings


def retriever_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant code chunks using multi-query retrieval.
    
    Uses query variations and adaptive rewriting for better coverage.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with retrieved chunks
    """
    retrieval_iteration = state.get('retrieval_iteration', 0) + 1
    
    # Determine base queries based on iteration
    base_queries = _get_queries_for_iteration(state, retrieval_iteration)
    
    # Generate query variations for multi-query retrieval
    all_queries = []
    for base_query in base_queries:
        variations = generate_query_variations(
            base_query,
            num_variations=getattr(settings, 'query_variations', DEFAULT_QUERY_VARIATIONS),
            use_llm=True
        )
        all_queries.extend(variations)
    
    # For follow-up iterations, also use query rewriting based on previous results
    if retrieval_iteration > 1:
        existing_chunks = state.get('retrieved_chunks', [])
        if existing_chunks:
            rewritten_queries = rewrite_queries_based_on_results(
                base_queries,
                existing_chunks,
                state['question'],
                max_new_queries=3
            )
            all_queries.extend(rewritten_queries)
    
    # Remove duplicates while preserving order
    seen_queries = set()
    unique_queries = []
    for q in all_queries:
        q_lower = q.lower().strip()
        if q_lower and q_lower not in seen_queries:
            seen_queries.add(q_lower)
            unique_queries.append(q)
    
    # Retrieve chunks for all query variations
    all_chunks = _retrieve_with_multi_query(
        unique_queries,
        state['repo_id'],
        state.get('retrieved_chunks', []),
        state['question']
    )
    
    # Combine with existing chunks and limit
    combined_chunks = state.get('retrieved_chunks', []) + all_chunks
    combined_chunks = combined_chunks[:DEFAULT_MAX_CITATIONS]
    
    reasoning_trace = state.get('reasoning_trace', [])
    reasoning_trace.append(
        f"Iteration {retrieval_iteration}: Used {len(unique_queries)} query variations, "
        f"retrieved {len(all_chunks)} new chunks ({len(combined_chunks)} total)"
    )
    
    return {
        **state,
        'retrieved_chunks': combined_chunks,
        'retrieval_iteration': retrieval_iteration,
        'reasoning_trace': reasoning_trace
    }


def _get_queries_for_iteration(state: AgentState, iteration: int) -> list:
    """Get queries for the current retrieval iteration."""
    if iteration == 1:
        # First iteration: use plan queries
        return state['plan']['search_queries']
    else:
        # Follow-up iterations: use verifier queries
        verification = state.get('verification_result', {})
        queries = verification.get('follow_up_queries', [])
        
        if not queries:
            # Fallback to original question
            queries = [state['question']]
        
        return queries


def _retrieve_with_multi_query(
    queries: list,
    repo_id: str,
    existing_chunks: list,
    original_question: str = None
) -> list:
    """
    Retrieve chunks using multi-query retrieval with merging and reranking.
    
    Retrieves for each query variation, then merges and reranks results
    to get the best chunks overall.
    
    Args:
        queries: List of query variations
        repo_id: Repository ID
        existing_chunks: Already retrieved chunks (to avoid duplicates)
        original_question: Original question for reranking context
    
    Returns:
        List of new, deduplicated chunks
    """
    existing_chunk_ids = {
        chunk['chunk_id'] 
        for chunk in existing_chunks
    }
    
    # Collect results from all queries
    all_results = []
    query_results_map = {}  # Track which query found which chunks
    
    for query in queries:
        chunks = hybrid_search(query, repo_id, k=DEFAULT_MAX_CHUNKS_PER_QUERY)
        
        # Track source query for each chunk
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            if chunk_id not in existing_chunk_ids:
                # Add query as metadata for tracking
                if 'sources' not in chunk:
                    chunk['sources'] = []
                if 'query_sources' not in chunk:
                    chunk['query_sources'] = []
                chunk['query_sources'].append(query)
                
                if chunk_id not in query_results_map:
                    query_results_map[chunk_id] = chunk
                    all_results.append(chunk)
                else:
                    # Chunk found by multiple queries - boost its score
                    existing_chunk = query_results_map[chunk_id]
                    existing_chunk['query_sources'].append(query)
                    # Boost score for multi-query matches
                    existing_score = existing_chunk.get('combined_score', 0.0)
                    existing_chunk['combined_score'] = existing_score + 0.2
    
    # If we have results from multiple queries, merge and rerank
    if len(queries) > 1 and all_results:
        # Since hybrid_search already returns merged and ranked results,
        # we just need to deduplicate and rerank across all queries
        # Boost chunks found by multiple queries
        chunk_scores = {}
        for chunk in all_results:
            chunk_id = chunk['chunk_id']
            base_score = chunk.get('combined_score', chunk.get('score', 0.5))
            query_count = len(chunk.get('query_sources', []))
            
            # Boost score for chunks found by multiple queries (indicates high relevance)
            if query_count > 1:
                base_score += (query_count - 1) * 0.3
            
            # Additional boost if original question keywords match chunk content
            if original_question:
                chunk_text = chunk.get('text', chunk.get('chunk_text', '')).lower()
                question_lower = original_question.lower()
                question_words = set(word for word in question_lower.split() if len(word) > 3)
                chunk_words = set(word for word in chunk_text.split() if len(word) > 3)
                matching_words = question_words.intersection(chunk_words)
                if matching_words:
                    base_score += len(matching_words) * 0.1
            
            chunk_scores[chunk_id] = base_score
            chunk['combined_score'] = base_score
        
        # Sort by combined score
        all_results.sort(key=lambda x: chunk_scores.get(x['chunk_id'], 0), reverse=True)
        
        # Limit results
        return all_results[:DEFAULT_MAX_CITATIONS * 2]
    
    # If single query or no merging needed, return as-is
    return all_results


def _retrieve_new_chunks(
    queries: list,
    repo_id: str,
    existing_chunks: list
) -> list:
    """
    Legacy retrieval method (kept for backward compatibility).
    
    Use _retrieve_with_multi_query for new implementations.
    """
    existing_chunk_ids = {
        chunk['chunk_id'] 
        for chunk in existing_chunks
    }
    
    all_chunks = []
    for query in queries:
        chunks = hybrid_search(query, repo_id, k=DEFAULT_MAX_CHUNKS_PER_QUERY)
        
        for chunk in chunks:
            if chunk['chunk_id'] not in existing_chunk_ids:
                all_chunks.append(chunk)
                existing_chunk_ids.add(chunk['chunk_id'])
    
    return all_chunks
