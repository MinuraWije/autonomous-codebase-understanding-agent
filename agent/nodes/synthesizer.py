"""Synthesizer node for the agent."""
from agent.state import AgentState
from agent.prompts import get_synthesizer_prompt, extract_citations
from core.llm_service import LLMServiceFactory
from core.citation_service import CitationService
from core.error_handler import safe_execute
from core.context_optimizer import optimize_chunks_for_context
from app.config import settings


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
    
    # Optimize chunks for context window (prioritize and truncate if needed)
    question = state['question']
    optimized_chunks = optimize_chunks_for_context(
        chunks,
        max_context_tokens=settings.context_window_size,
        reserve_prompt_tokens=settings.reserve_prompt_tokens,
        reserve_response_tokens=settings.reserve_response_tokens,
        question=question
    )
    
    # Track optimization in reasoning trace
    reasoning_trace = state.get('reasoning_trace', [])
    if len(optimized_chunks) < len(chunks):
        truncated_count = sum(1 for c in optimized_chunks if c.get('_truncated', False))
        reasoning_trace.append(
            f"Context optimization: {len(optimized_chunks)}/{len(chunks)} chunks selected, "
            f"{truncated_count} truncated to fit context window"
        )
    else:
        reasoning_trace.append(f"Context optimization: All {len(chunks)} chunks fit within context window")
    
    llm_service = LLMServiceFactory.create_synthesizer_service()
    citation_service = CitationService()
    prompt = get_synthesizer_prompt(question, optimized_chunks)
    
    draft_answer = safe_execute(
        lambda: llm_service.invoke_text(prompt),
        default_return="Error generating answer"
    )
    
    # Extract citations with fallback to context-based extraction
    # Use original chunks for citation extraction to preserve full context
    citations = citation_service.extract_citations_from_answer(
        draft_answer,
        retrieved_chunks=chunks  # Use original chunks for citation matching
    )
    
    reasoning_trace.append(f"Generated answer with {len(citations)} citations")
    
    return {
        **state,
        'draft_answer': draft_answer,
        'citations': citations,
        'reasoning_trace': reasoning_trace
    }
