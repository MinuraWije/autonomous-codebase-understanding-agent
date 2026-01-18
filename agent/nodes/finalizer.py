"""Finalizer node for the agent."""
from agent.state import AgentState
from core.citation_service import CitationService


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
    
    citation_service = CitationService()
    
    # Enhance citations with actual text snippets
    enhanced_citations = citation_service.enhance_citations(
        citations,
        state['repo_id']
    )
    
    # Format final answer with citation summary
    reference_section = citation_service.format_citations_for_answer(
        enhanced_citations
    )
    final_answer = draft_answer + reference_section
    
    reasoning_trace = state.get('reasoning_trace', [])
    reasoning_trace.append("Finalized answer with enhanced citations")
    
    return {
        **state,
        'final_answer': final_answer,
        'citations': enhanced_citations,
        'reasoning_trace': reasoning_trace
    }
