"""Planner node for the agent."""
from agent.llm_wrapper import HuggingFaceChatLLM
from agent.state import AgentState
from agent.prompts import get_planner_prompt, extract_json_from_response
from app.config import settings


def planner_node(state: AgentState) -> AgentState:
    """
    Plan how to answer the question.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with plan
    """
    llm = HuggingFaceChatLLM(
        model=settings.planner_model,
        huggingface_api_key=settings.huggingface_api_key,
        temperature=0
    )
    
    prompt = get_planner_prompt(state['question'])
    
    try:
        response = llm.invoke(prompt)
        plan = extract_json_from_response(response.content)
        
        # Validate plan has required fields
        if not plan.get('search_queries'):
            plan['search_queries'] = [state['question']]
        if not plan.get('reasoning'):
            plan['reasoning'] = "Direct search for question keywords"
        if not plan.get('expected_files'):
            plan['expected_files'] = []
        
    except Exception as e:
        print(f"Planner error: {e}")
        # Fallback plan
        plan = {
            'reasoning': 'Using fallback plan due to parsing error',
            'search_queries': [state['question']],
            'expected_files': []
        }
    
    reasoning_trace = state.get('reasoning_trace', [])
    reasoning_trace.append(f"Plan: {plan['reasoning']}")
    
    return {
        **state,
        'plan': plan,
        'reasoning_trace': reasoning_trace,
        'retrieval_iteration': 0
    }
