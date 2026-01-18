"""Retriever node for the agent."""
from agent.state import AgentState
from tools.retrieval_tools import hybrid_search
from core.constants import DEFAULT_MAX_CHUNKS_PER_QUERY, DEFAULT_MAX_CITATIONS


def retriever_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant code chunks.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with retrieved chunks
    """
    retrieval_iteration = state.get('retrieval_iteration', 0) + 1
    
    # Determine queries based on iteration
    queries = _get_queries_for_iteration(state, retrieval_iteration)
    
    # Retrieve chunks for each query
    all_chunks = _retrieve_new_chunks(
        queries,
        state['repo_id'],
        state.get('retrieved_chunks', [])
    )
    
    # Combine with existing chunks and limit
    combined_chunks = state.get('retrieved_chunks', []) + all_chunks
    combined_chunks = combined_chunks[:DEFAULT_MAX_CITATIONS]
    
    reasoning_trace = state.get('reasoning_trace', [])
    reasoning_trace.append(
        f"Iteration {retrieval_iteration}: Retrieved {len(all_chunks)} new chunks "
        f"({len(combined_chunks)} total)"
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


def _retrieve_new_chunks(
    queries: list,
    repo_id: str,
    existing_chunks: list
) -> list:
    """Retrieve new chunks for queries, avoiding duplicates."""
    all_chunks = []
    existing_chunk_ids = {
        chunk['chunk_id'] 
        for chunk in existing_chunks
    }
    
    for query in queries:
        chunks = hybrid_search(query, repo_id, k=DEFAULT_MAX_CHUNKS_PER_QUERY)
        
        # Add new chunks (avoid duplicates from previous iterations)
        for chunk in chunks:
            if chunk['chunk_id'] not in existing_chunk_ids:
                all_chunks.append(chunk)
                existing_chunk_ids.add(chunk['chunk_id'])
    
    return all_chunks
