"""LangGraph workflow for the codebase understanding agent."""
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.planner import planner_node
from agent.nodes.retriever import retriever_node
from agent.nodes.synthesizer import synthesizer_node
from agent.nodes.verifier import verifier_node
from agent.nodes.finalizer import finalizer_node
from app.config import settings
from core.constants import DEFAULT_MAX_RETRIEVAL_ITERATIONS


def should_retrieve_more(state: AgentState) -> str:
    """
    Determine if we should retrieve more evidence.
    
    Args:
        state: Current agent state
    
    Returns:
        Next node name
    """
    verification = state.get('verification_result', {})
    iteration = state.get('retrieval_iteration', 0)
    max_iterations = getattr(settings, 'max_retrieval_iterations', DEFAULT_MAX_RETRIEVAL_ITERATIONS)
    
    # Check if verification failed and we haven't exceeded max iterations
    if not verification.get('is_grounded', True):
        if iteration < max_iterations:
            # Check if verifier provided follow-up queries
            if verification.get('follow_up_queries'):
                return "retrieve_more"
    
    return "finalize"


def create_agent_graph():
    """
    Create the LangGraph workflow for the agent.
    
    Returns:
        Compiled graph
    """
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("finalizer", finalizer_node)
    
    # Define the flow
    workflow.set_entry_point("planner")
    
    # Basic flow
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", "verifier")
    
    # Conditional edge: verify -> retrieve more OR finalize
    workflow.add_conditional_edges(
        "verifier",
        should_retrieve_more,
        {
            "retrieve_more": "retriever",  # Loop back to retriever
            "finalize": "finalizer"
        }
    )
    
    # Final node
    workflow.add_edge("finalizer", END)
    
    # Compile the graph
    return workflow.compile()


def create_simple_agent_graph():
    """
    Create a simpler version without the verification loop.
    Useful for faster responses.
    
    Returns:
        Compiled graph
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("finalizer", finalizer_node)
    
    # Define the flow
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", "finalizer")
    workflow.add_edge("finalizer", END)
    
    return workflow.compile()
