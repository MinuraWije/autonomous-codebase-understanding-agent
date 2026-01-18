"""Retriever node for the agent."""
from agent.state import AgentState
from tools.retrieval_tools import hybrid_search


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
    if retrieval_iteration == 1:
        # First iteration: use plan queries
        queries = state['plan']['search_queries']
    else:
        # Follow-up iterations: use verifier queries
        verification = state.get('verification_result', {})
        queries = verification.get('follow_up_queries', [])
        
        if not queries:
            # Fallback to original question
            queries = [state['question']]
    
    # Retrieve chunks for each query
    all_chunks = []
    existing_chunk_ids = {
        chunk['chunk_id'] 
        for chunk in state.get('retrieved_chunks', [])
    }
    
    for query in queries:
        chunks = hybrid_search(query, state['repo_id'], k=8)
        
        # Add new chunks (avoid duplicates from previous iterations)
        for chunk in chunks:
            if chunk['chunk_id'] not in existing_chunk_ids:
                all_chunks.append(chunk)
                existing_chunk_ids.add(chunk['chunk_id'])
    
    # Combine with existing chunks
    combined_chunks = state.get('retrieved_chunks', []) + all_chunks
    
    # Limit total chunks
    combined_chunks = combined_chunks[:15]
    
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
