"""Citation service for handling citations."""
from typing import List, Dict, Optional
from tools.file_tools import open_span
from agent.prompts import extract_citations
from core.constants import DEFAULT_SNIPPET_LENGTH
from core.exceptions import FileNotFoundError


class CitationService:
    """Service for citation operations."""
    
    def extract_citations_from_answer(self, answer_text: str) -> List[Dict]:
        """
        Extract citations from answer text.
        
        Args:
            answer_text: Answer text containing citations
        
        Returns:
            List of citation dictionaries
        """
        return extract_citations(answer_text)
    
    def enhance_citations(
        self,
        citations: List[Dict],
        repo_id: str
    ) -> List[Dict]:
        """
        Enhance citations with actual code snippets.
        
        Args:
            citations: List of citation dictionaries
            repo_id: Repository ID
        
        Returns:
            List of enhanced citations with text snippets
        """
        enhanced = []
        for citation in citations:
            enhanced_citation = self._enhance_single_citation(citation, repo_id)
            enhanced.append(enhanced_citation)
        return enhanced
    
    def _enhance_single_citation(
        self,
        citation: Dict,
        repo_id: str
    ) -> Dict:
        """
        Enhance a single citation with code snippet.
        
        Args:
            citation: Citation dictionary
            repo_id: Repository ID
        
        Returns:
            Enhanced citation dictionary
        """
        try:
            snippet = open_span(
                repo_id,
                citation['file_path'],
                citation['start_line'],
                citation['end_line']
            )
            
            # Limit snippet length
            if len(snippet) > DEFAULT_SNIPPET_LENGTH:
                snippet = snippet[:DEFAULT_SNIPPET_LENGTH] + "..."
            
            return {
                **citation,
                'text_snippet': snippet
            }
        except Exception:
            # If we can't get the snippet, keep the citation without it
            return {
                **citation,
                'text_snippet': '[Code snippet unavailable]'
            }
    
    def format_citations_for_answer(self, citations: List[Dict]) -> str:
        """
        Format citations as a reference section for the answer.
        
        Args:
            citations: List of citation dictionaries
        
        Returns:
            Formatted reference section
        """
        if not citations:
            return ""
        
        reference_section = "\n\n### References:\n"
        for i, citation in enumerate(citations, 1):
            reference_section += (
                f"\n{i}. `{citation['file_path']}` "
                f"(lines {citation['start_line']}-{citation['end_line']})"
            )
        
        return reference_section
