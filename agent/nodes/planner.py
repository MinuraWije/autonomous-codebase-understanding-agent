"""Planner node for the agent."""
from agent.state import AgentState
from agent.prompts import get_planner_prompt, extract_json_from_response
from core.llm_service import LLMServiceFactory
from core.error_handler import safe_execute


def planner_node(state: AgentState) -> AgentState:
    """
    Plan how to answer the question.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with plan
    """
    llm_service = LLMServiceFactory.create_planner_service()
    prompt = get_planner_prompt(state['question'])
    
    plan = safe_execute(
        _generate_plan,
        llm_service,
        prompt,
        state['question'],
        default_return=_create_fallback_plan(state['question'])
    )
    
    reasoning_trace = state.get('reasoning_trace', [])
    reasoning_trace.append(f"Plan: {plan['reasoning']}")
    
    return {
        **state,
        'plan': plan,
        'reasoning_trace': reasoning_trace,
        'retrieval_iteration': 0
    }


def _generate_plan(llm_service, prompt: str, question: str) -> dict:
    """Generate a plan using the LLM service."""
    response_text = llm_service.invoke_text(prompt)
    plan = extract_json_from_response(response_text)
    
    # Validate plan has required fields
    if not plan.get('search_queries'):
        plan['search_queries'] = [question]
    if not plan.get('reasoning'):
        plan['reasoning'] = "Direct search for question keywords"
    if not plan.get('expected_files'):
        plan['expected_files'] = []
    
    return plan


def _create_fallback_plan(question: str) -> dict:
    """Create a fallback plan."""
    return {
        'reasoning': 'Using fallback plan due to parsing error',
        'search_queries': [question],
        'expected_files': []
    }
