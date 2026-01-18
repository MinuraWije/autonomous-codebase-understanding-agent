"""Pydantic schemas for API requests and responses."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    """Request to index a repository."""
    github_url: Optional[str] = Field(None, description="GitHub repository URL")
    local_path: Optional[str] = Field(None, description="Local repository path")
    branch: str = Field("main", description="Branch to clone (for GitHub repos)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "github_url": "https://github.com/user/repo",
                "branch": "main"
            }
        }


class IndexResponse(BaseModel):
    """Response after starting indexing."""
    repo_id: str
    stats: Dict[str, Any]
    message: str


class QuestionRequest(BaseModel):
    """Request to ask a question about a repository."""
    repo_id: str = Field(..., description="Repository ID")
    question: str = Field(..., description="Question to ask")
    use_verification: bool = Field(True, description="Use verification loop")
    
    class Config:
        json_schema_extra = {
            "example": {
                "repo_id": "abc123def456",
                "question": "Where is authentication handled?",
                "use_verification": True
            }
        }


class Citation(BaseModel):
    """Code citation."""
    file_path: str
    start_line: int
    end_line: int
    text_snippet: str


class AnswerResponse(BaseModel):
    """Response with answer to question."""
    answer: str
    citations: List[Citation]
    reasoning_trace: Optional[List[str]] = None


class RepoStatus(BaseModel):
    """Repository status."""
    status: str  # "not_found", "indexing", "completed"
    repo_id: Optional[str] = None
    url: Optional[str] = None
    indexed_at: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


class RepoSummary(BaseModel):
    """Repository summary."""
    repo_id: str
    url: Optional[str]
    local_path: str
    commit_hash: str
    indexed_at: str
    stats: Dict[str, Any]


class ArchitectureSummaryResponse(BaseModel):
    """Architecture summary response."""
    summary: str
    key_files: List[str]
    file_structure: Dict[str, Any]
