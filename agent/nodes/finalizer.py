"""Finalizer node for the agent."""
from agent.state import AgentState
from tools.file_tools import open_span


def finalizer_node(state: AgentState) -> AgentState:
    """
    Finalize the answer with formatted citations.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with final answer
    """
    draft_answer = state.get('draft_answer', '')
    citations = state.get('citations', [])
    
    # Enhance citations with actual text snippets
    enhanced_citations = []
    for citation in citations:
        try:
            # Try to get the actual code snippet
            snippet = open_span(
                state['repo_id'],
                citation['file_path'],
                citation['start_line'],
                citation['end_line']
            )
            
            # Limit snippet length
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            
            enhanced_citations.append({
                **citation,
                'text_snippet': snippet
            })
        except Exception:
            # If we can't get the snippet, keep the citation without it
            enhanced_citations.append({
                **citation,
                'text_snippet': '[Code snippet unavailable]'
            })
    
    # Format final answer with citation summary
    final_answer = draft_answer
    
    if enhanced_citations:
        final_answer += "\n\n### References:\n"
        for i, citation in enumerate(enhanced_citations, 1):
            final_answer += f"\n{i}. `{citation['file_path']}` (lines {citation['start_line']}-{citation['end_line']})"
    
    reasoning_trace = state.get('reasoning_trace', [])
    reasoning_trace.append("Finalized answer with enhanced citations")
    
    return {
        **state,
        'final_answer': final_answer,
        'citations': enhanced_citations,
        'reasoning_trace': reasoning_trace
    }
