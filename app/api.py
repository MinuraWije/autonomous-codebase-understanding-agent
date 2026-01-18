"""API routes for the codebase understanding agent."""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import List
from app.schemas import (
    IndexRequest, IndexResponse, QuestionRequest, AnswerResponse,
    RepoStatus, RepoSummary, ArchitectureSummaryResponse, Citation
)
from core.indexing_service import IndexingService
from core.repository_service import RepositoryService
from core.agent_service import AgentService
from core.architecture_service import ArchitectureService
from core.repository_deletion_service import RepositoryDeletionService
from core.exceptions import (
    RepositoryNotFoundError,
    AgentExecutionError,
    IndexingError
)
from core.constants import MESSAGE_INDEXING_STARTED


router = APIRouter()

# Initialize services
indexing_service = IndexingService()
repository_service = RepositoryService()
agent_service = AgentService()
architecture_service = ArchitectureService()
deletion_service = RepositoryDeletionService()


@router.post("/repos/index", response_model=IndexResponse, tags=["Repositories"])
async def index_repo(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Start indexing a repository.
    
    Either github_url or local_path must be provided.
    """
    if not request.github_url and not request.local_path:
        raise HTTPException(400, "Either github_url or local_path must be provided")
    
    # Generate repo_id
    repo_id = indexing_service.start_indexing(
        github_url=request.github_url,
        local_path=request.local_path,
        branch=request.branch
    )
    
    # Start indexing in background
    background_tasks.add_task(
        indexing_service.index_repository_task,
        github_url=request.github_url,
        local_path=request.local_path,
        branch=request.branch
    )
    
    return IndexResponse(
        repo_id=repo_id,
        stats={},
        message=MESSAGE_INDEXING_STARTED.format(repo_id=repo_id)
    )


@router.get("/repos/{repo_id}/status", response_model=RepoStatus, tags=["Repositories"])
async def get_repo_status(repo_id: str):
    """Get the indexing status of a repository."""
    status = indexing_service.get_indexing_status(repo_id)
    return status


@router.get("/repos", response_model=List[RepoSummary], tags=["Repositories"])
async def list_repos():
    """List all indexed repositories."""
    repos = repository_service.list_repositories()
    return repos


@router.post("/chat", response_model=AnswerResponse, tags=["Chat"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question about a repository.
    
    The agent will:
    1. Plan how to search for relevant code
    2. Retrieve relevant code chunks
    3. Synthesize an answer with citations
    4. Optionally verify the answer is grounded in the code
    """
    try:
        result = agent_service.run_agent(
            question=request.question,
            repo_id=request.repo_id,
            use_verification=request.use_verification
        )
        
        # Format response
        citations = [
            Citation(
                file_path=c['file_path'],
                start_line=c['start_line'],
                end_line=c['end_line'],
                text_snippet=c.get('text_snippet', '')
            )
            for c in result.get('citations', [])
        ]
        
        return AnswerResponse(
            answer=result.get('final_answer', result.get('draft_answer', 'No answer generated')),
            citations=citations,
            reasoning_trace=result.get('reasoning_trace')
        )
    except RepositoryNotFoundError as e:
        raise HTTPException(404, str(e))
    except AgentExecutionError as e:
        raise HTTPException(500, str(e))


@router.get("/repos/{repo_id}/summary", response_model=ArchitectureSummaryResponse, tags=["Repositories"])
async def get_architecture_summary(repo_id: str):
    """Generate an architecture summary for the repository."""
    try:
        result = architecture_service.generate_summary(repo_id)
        return ArchitectureSummaryResponse(**result)
    except RepositoryNotFoundError as e:
        raise HTTPException(404, str(e))


@router.delete("/repos/{repo_id}", tags=["Repositories"])
async def delete_repo(repo_id: str):
    """Delete a repository and all its indexed data."""
    try:
        message = deletion_service.delete_repository(repo_id)
        return {"message": message}
    except RepositoryNotFoundError as e:
        raise HTTPException(404, str(e))
