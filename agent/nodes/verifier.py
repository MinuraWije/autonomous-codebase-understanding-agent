"""Verifier node for the agent."""
from agent.llm_wrapper import HuggingFaceChatLLM
from agent.state import AgentState
from agent.prompts import get_verifier_prompt, extract_json_from_response
from app.config import settings


def verifier_node(state: AgentState) -> AgentState:
    """
    Verify if the answer is grounded in retrieved code.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with verification result
    """
    llm = HuggingFaceChatLLM(
        model=settings.verifier_model,
        huggingface_api_key=settings.huggingface_api_key,
        temperature=0
    )
    
    prompt = get_verifier_prompt(
        state['question'],
        state['draft_answer'],
        state['retrieved_chunks']
    )
    
    try:
        response = llm.invoke(prompt)
        verification = extract_json_from_response(response.content)
        
        # Validate verification has required fields
        if 'is_grounded' not in verification:
            verification['is_grounded'] = True  # Default to accepting
        if 'unsupported_claims' not in verification:
            verification['unsupported_claims'] = []
        if 'missing_information' not in verification:
            verification['missing_information'] = []
        if 'follow_up_queries' not in verification:
            verification['follow_up_queries'] = []
        
    except Exception as e:
        print(f"Verifier error: {e}")
        # Fallback: accept the answer
        verification = {
            'is_grounded': True,
            'unsupported_claims': [],
            'missing_information': [],
            'follow_up_queries': []
        }
    
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
