"""Architecture analysis service."""
from typing import Dict, List
from tools.file_tools import get_file_structure
from tools.repo_tools import get_key_files
from core.repository_service import RepositoryService
from core.llm_service import LLMServiceFactory
from core.error_handler import safe_execute
from core.exceptions import RepositoryNotFoundError


class ArchitectureService:
    """Service for architecture analysis."""
    
    def __init__(self):
        """Initialize the architecture service."""
        self.repository_service = RepositoryService()
    
    def generate_summary(self, repo_id: str) -> Dict[str, any]:
        """
        Generate architecture summary for a repository.
        
        Args:
            repo_id: Repository ID
        
        Returns:
            Dictionary with summary, key_files, and file_structure
        
        Raises:
            RepositoryNotFoundError: If repository not found
        """
        repo = self.repository_service.get_repository(repo_id)
        
        # Get key files and file structure
        key_files = safe_execute(
            get_key_files,
            repo_id,
            top_n=10,
            default_return=[]
        )
        
        file_structure = safe_execute(
            get_file_structure,
            repo_id,
            max_depth=3,
            default_return={}
        )
        
        # Generate summary using LLM
        summary = self._generate_llm_summary(repo, key_files, file_structure)
        
        return {
            'summary': summary,
            'key_files': key_files,
            'file_structure': file_structure
        }
    
    def _generate_llm_summary(
        self,
        repo: Dict,
        key_files: List[str],
        file_structure: Dict
    ) -> str:
        """Generate summary using LLM."""
        stats = repo.get('stats', {})
        by_language = stats.get('by_language', {})
        
        llm_service = LLMServiceFactory.create_summary_service()
        
        prompt = self._build_summary_prompt(stats, by_language, key_files, file_structure)
        
        summary = safe_execute(
            lambda: llm_service.invoke_text(prompt),
            default_return="Could not generate summary"
        )
        
        return summary
    
    def _build_summary_prompt(
        self,
        stats: Dict,
        by_language: Dict,
        key_files: List[str],
        file_structure: Dict
    ) -> str:
        """Build the summary prompt."""
        return f"""Analyze this codebase structure and generate a 2-3 paragraph architecture overview.

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
