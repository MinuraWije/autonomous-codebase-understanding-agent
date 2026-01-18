"""Agent state definition for LangGraph."""
from typing import TypedDict, List, Optional, Dict, Any


class AgentState(TypedDict, total=False):
    """State for the codebase understanding agent."""
    
    # Input
    question: str
    repo_id: str
    
    # Planning
    plan: Optional[Dict[str, Any]]  # {"reasoning": str, "search_queries": List[str], "expected_files": List[str]}
    
    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]  # [{chunk_id, text, file, lines, score}]
    retrieval_iteration: int
    
    # Synthesis
    draft_answer: Optional[str]
    
    # Verification
    verification_result: Optional[Dict[str, Any]]  # {is_grounded: bool, unsupported_claims: List, missing_information: List, follow_up_queries: List}
    
    # Output
    final_answer: Optional[str]
    citations: List[Dict[str, Any]]  # [{file, start_line, end_line, text_snippet}]
    reasoning_trace: List[str]
