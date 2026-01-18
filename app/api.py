"""API routes for the codebase understanding agent."""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import List
from app.schemas import (
    IndexRequest, IndexResponse, QuestionRequest, AnswerResponse,
    RepoStatus, RepoSummary, ArchitectureSummaryResponse, Citation
)
from indexing.pipeline import index_repository, get_indexing_status
from indexing.metadata_store import get_metadata_store
from agent.graph import create_agent_graph, create_simple_agent_graph
from tools.file_tools import get_file_structure
from tools.repo_tools import get_key_files
from agent.llm_wrapper import HuggingFaceChatLLM
from app.config import settings


router = APIRouter()


# Background task for indexing
def index_repo_task(github_url: str = None, local_path: str = None, branch: str = "main"):
    """Background task to index a repository."""
    try:
        index_repository(github_url=github_url, local_path=local_path, branch=branch)
        print(f"Indexing completed for {github_url or local_path}")
    except Exception as e:
        print(f"Indexing failed: {e}")


@router.post("/repos/index", response_model=IndexResponse, tags=["Repositories"])
async def index_repo(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Start indexing a repository.
    
    Either github_url or local_path must be provided.
    """
    if not request.github_url and not request.local_path:
        raise HTTPException(400, "Either github_url or local_path must be provided")
    
    # Start indexing in background
    background_tasks.add_task(
        index_repo_task,
        github_url=request.github_url,
        local_path=request.local_path,
        branch=request.branch
    )
    
    # Generate repo_id for response
    from indexing.loader import generate_repo_id
    repo_id = generate_repo_id(request.github_url or request.local_path)
    
    return IndexResponse(
        repo_id=repo_id,
        stats={},
        message="Indexing started in background. Check status at /repos/{repo_id}/status"
    )


@router.get("/repos/{repo_id}/status", response_model=RepoStatus, tags=["Repositories"])
async def get_repo_status(repo_id: str):
    """Get the indexing status of a repository."""
    status = get_indexing_status(repo_id)
    return status


@router.get("/repos", response_model=List[RepoSummary], tags=["Repositories"])
async def list_repos():
    """List all indexed repositories."""
    metadata_store = get_metadata_store()
    repos = metadata_store.list_repos()
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
    # Check repo exists
    metadata_store = get_metadata_store()
    repo = metadata_store.get_repo_metadata(request.repo_id)
    
    if not repo:
        raise HTTPException(404, f"Repository not found: {request.repo_id}")
    
    # Create appropriate graph
    if request.use_verification:
        agent = create_agent_graph()
    else:
        agent = create_simple_agent_graph()
    
    # Run the agent
    try:
        result = agent.invoke({
            'question': request.question,
            'repo_id': request.repo_id,
            'retrieval_iteration': 0,
            'reasoning_trace': []
        })
        
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
        
    except Exception as e:
        raise HTTPException(500, f"Error running agent: {str(e)}")


@router.get("/repos/{repo_id}/summary", response_model=ArchitectureSummaryResponse, tags=["Repositories"])
async def get_architecture_summary(repo_id: str):
    """
    Generate an architecture summary for the repository.
    """
    metadata_store = get_metadata_store()
    repo = metadata_store.get_repo_metadata(repo_id)
    
    if not repo:
        raise HTTPException(404, f"Repository not found: {repo_id}")
    
    # Get key files
    try:
        key_files = get_key_files(repo_id, top_n=10)
    except Exception:
        key_files = []
    
    # Get file structure
    try:
        file_structure = get_file_structure(repo_id, max_depth=3)
    except Exception:
        file_structure = {}
    
    # Generate summary using LLM
    stats = repo.get('stats', {})
    by_language = stats.get('by_language', {})
    
    llm = HuggingFaceChatLLM(
        model=settings.planner_model,
        huggingface_api_key=settings.huggingface_api_key,
        temperature=0.3
    )
    
    prompt = f"""Analyze this codebase structure and generate a 2-3 paragraph architecture overview.

Repository Stats:
- Total files: {stats.get('total_files', 0)}
- Languages: {', '.join(f"{k}: {v} files" for k, v in list(by_language.items())[:5])}

Key Files:
{chr(10).join(f"- {f}" for f in key_files[:10])}

Top-level Structure:
{chr(10).join(f"- {k}/" for k in list(file_structure.keys())[:10])}

Focus on:
1. Overall architecture pattern (MVC, microservices, monolith, etc.)
2. Main components and their responsibilities
3. Technology stack
4. Data flow

Provide a clear, concise summary suitable for onboarding a new developer."""
    
    try:
        response = llm.invoke(prompt)
        summary = response.content
    except Exception as e:
        summary = f"Could not generate summary: {e}"
    
    return ArchitectureSummaryResponse(
        summary=summary,
        key_files=key_files,
        file_structure=file_structure
    )


@router.delete("/repos/{repo_id}", tags=["Repositories"])
async def delete_repo(repo_id: str):
    """Delete a repository and all its indexed data."""
    metadata_store = get_metadata_store()
    repo = metadata_store.get_repo_metadata(repo_id)
    
    if not repo:
        raise HTTPException(404, f"Repository not found: {repo_id}")
    
    # Delete from metadata store
    metadata_store.delete_repo(repo_id)
    
    # Delete from vector store
    from indexing.vector_store import get_vector_store
    vector_store = get_vector_store()
    vector_store.delete_collection(repo_id)
    
    return {"message": f"Repository {repo_id} deleted successfully"}
