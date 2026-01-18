"""Verifier node for the agent."""
from agent.state import AgentState
from agent.prompts import get_verifier_prompt, extract_json_from_response
from core.llm_service import LLMServiceFactory
from core.error_handler import safe_execute


def verifier_node(state: AgentState) -> AgentState:
    """
    Verify if the answer is grounded in retrieved code.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with verification result
    """
    llm_service = LLMServiceFactory.create_verifier_service()
    prompt = get_verifier_prompt(
        state['question'],
        state['draft_answer'],
        state['retrieved_chunks']
    )
    
    verification = safe_execute(
        _verify_answer,
        llm_service,
        prompt,
        default_return=_create_fallback_verification()
    )
    
    reasoning_trace = state.get('reasoning_trace', [])
    reasoning_trace.append(
        f"Verification: grounded={verification['is_grounded']}, "
        f"unsupported_claims={len(verification['unsupported_claims'])}"
    )
    
    return {
        **state,
        'verification_result': verification,
        'reasoning_trace': reasoning_trace
    }


def _verify_answer(llm_service, prompt: str) -> dict:
    """Verify answer using the LLM service."""
    response_text = llm_service.invoke_text(prompt)
    verification = extract_json_from_response(response_text)
    
    # Validate verification has required fields
    if 'is_grounded' not in verification:
        verification['is_grounded'] = True  # Default to accepting
    if 'unsupported_claims' not in verification:
        verification['unsupported_claims'] = []
    if 'missing_information' not in verification:
        verification['missing_information'] = []
    if 'follow_up_queries' not in verification:
        verification['follow_up_queries'] = []
    
    return verification


def _create_fallback_verification() -> dict:
    """Create a fallback verification result."""
    return {
        'is_grounded': True,
        'unsupported_claims': [],
        'missing_information': [],
        'follow_up_queries': []
    }
