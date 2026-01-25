"""Finalizer node for the agent."""
from agent.state import AgentState
from core.citation_service import CitationService


def finalizer_node(state: AgentState) -> AgentState:
    """
    Finalize the answer with post-processing: summary, code snippets, and formatted citations.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with final answer
    """
    draft_answer = state.get('draft_answer', '')
    citations = state.get('citations', [])
    
    citation_service = CitationService()
    
    # Enhance citations with actual text snippets
    enhanced_citations = citation_service.enhance_citations(
        citations,
        state['repo_id']
    )
    
    # Post-process answer with summary, code snippets, and formatted citations
    final_answer = citation_service.post_process_answer(
        draft_answer,
        enhanced_citations,
        include_summary=True,
        include_code_snippets=True
    )
    
    reasoning_trace = state.get('reasoning_trace', [])
    reasoning_trace.append(
        f"Finalized answer with {len(enhanced_citations)} citations "
        f"(summary, code snippets, and formatted references)"
    )
    
    return {
        **state,
        'final_answer': final_answer,
        'citations': enhanced_citations,
        'reasoning_trace': reasoning_trace
    }
