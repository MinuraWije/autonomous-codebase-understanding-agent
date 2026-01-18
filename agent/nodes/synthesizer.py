"""Synthesizer node for the agent."""
from agent.llm_wrapper import HuggingFaceChatLLM
from agent.state import AgentState
from agent.prompts import get_synthesizer_prompt, extract_citations
from app.config import settings


def synthesizer_node(state: AgentState) -> AgentState:
    """
    Synthesize an answer from retrieved chunks.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with draft answer and citations
    """
    llm = HuggingFaceChatLLM(
        model=settings.synthesizer_model,
        huggingface_api_key=settings.huggingface_api_key,
        temperature=0
    )
    
    chunks = state['retrieved_chunks']
    
    if not chunks:
        return {
            **state,
            'draft_answer': "No relevant code was found to answer this question.",
            'citations': []
        }
    
    prompt = get_synthesizer_prompt(state['question'], chunks)
    
    try:
        response = llm.invoke(prompt)
        draft_answer = response.content
        
        # Extract citations
        citations = extract_citations(draft_answer)
        
    except Exception as e:
        print(f"Synthesizer error: {e}")
        draft_answer = f"Error generating answer: {e}"
        citations = []
    
    reasoning_trace = state.get('reasoning_trace', [])
    reasoning_trace.append(f"Generated answer with {len(citations)} citations")
    
    return {
        **state,
        'draft_answer': draft_answer,
        'citations': citations,
        'reasoning_trace': reasoning_trace
    }
