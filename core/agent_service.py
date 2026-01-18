"""Agent service for running agent workflows."""
from typing import Dict, Any
from agent.graph import create_agent_graph, create_simple_agent_graph
from core.exceptions import AgentExecutionError, RepositoryNotFoundError
from core.repository_service import RepositoryService
from core.constants import ERROR_AGENT_EXECUTION


class AgentService:
    """Service for agent operations."""
    
    def __init__(self):
        """Initialize the agent service."""
        self.repository_service = RepositoryService()
    
    def run_agent(
        self,
        question: str,
        repo_id: str,
        use_verification: bool = True
    ) -> Dict[str, Any]:
        """
        Run the agent to answer a question.
        
        Args:
            question: Question to answer
            repo_id: Repository ID
            use_verification: Whether to use verification loop
        
        Returns:
            Agent result dictionary
        
        Raises:
            RepositoryNotFoundError: If repository not found
            AgentExecutionError: If agent execution fails
        """
        # Validate repository exists
        self.repository_service.validate_repository_exists(repo_id)
        
        # Create appropriate graph
        if use_verification:
            agent = create_agent_graph()
        else:
            agent = create_simple_agent_graph()
        
        # Run the agent
        try:
            result = agent.invoke({
                'question': question,
                'repo_id': repo_id,
                'retrieval_iteration': 0,
                'reasoning_trace': []
            })
            return result
        except Exception as e:
            raise AgentExecutionError(
                ERROR_AGENT_EXECUTION.format(error=str(e))
            )
