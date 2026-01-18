"""Synthesizer node for the agent."""
from agent.state import AgentState
from agent.prompts import get_synthesizer_prompt, extract_citations
from core.llm_service import LLMServiceFactory
from core.citation_service import CitationService
from core.error_handler import safe_execute


def synthesizer_node(state: AgentState) -> AgentState:
    """
    Synthesize an answer from retrieved chunks.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with draft answer and citations
    """
    chunks = state['retrieved_chunks']
    
    if not chunks:
        return {
            **state,
            'draft_answer': "No relevant code was found to answer this question.",
            'citations': []
        }
    
    llm_service = LLMServiceFactory.create_synthesizer_service()
    citation_service = CitationService()
    prompt = get_synthesizer_prompt(state['question'], chunks)
    
    draft_answer = safe_execute(
        lambda: llm_service.invoke_text(prompt),
        default_return="Error generating answer"
    )
    
    # Extract citations with fallback to context-based extraction
    citations = citation_service.extract_citations_from_answer(
        draft_answer,
        retrieved_chunks=chunks
    )
    
    reasoning_trace = state.get('reasoning_trace', [])
    reasoning_trace.append(f"Generated answer with {len(citations)} citations")
    
    return {
        **state,
        'draft_answer': draft_answer,
        'citations': citations,
        'reasoning_trace': reasoning_trace
    }
