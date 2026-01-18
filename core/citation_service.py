"""Citation service for handling citations."""
from typing import List, Dict, Optional
from tools.file_tools import open_span
from agent.prompts import extract_citations
from core.constants import DEFAULT_SNIPPET_LENGTH
from core.exceptions import FileNotFoundError


class CitationService:
    """Service for citation operations."""
    
    def extract_citations_from_answer(
        self, 
        answer_text: str,
        retrieved_chunks: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Extract citations from answer text.
        
        Args:
            answer_text: Answer text containing citations
            retrieved_chunks: Optional list of retrieved chunks for fallback extraction
        
        Returns:
            List of citation dictionaries
        """
        citations = extract_citations(answer_text)
        
        # If no citations found but we have retrieved chunks, try to infer citations
        # from file paths mentioned in the answer
        if not citations and retrieved_chunks:
            citations = self._extract_citations_from_context(answer_text, retrieved_chunks)
            if citations:
                print(f"Fallback extraction found {len(citations)} citations from context")
            else:
                print(f"Fallback extraction found 0 citations despite {len(retrieved_chunks)} chunks")
        
        return citations
    
    def _extract_citations_from_context(
        self,
        answer_text: str,
        retrieved_chunks: List[Dict]
    ) -> List[Dict]:
        """
        Fallback: Extract citations by matching file paths mentioned in answer
        with retrieved chunks. Uses multiple strategies for better matching.
        
        Args:
            answer_text: Answer text
            retrieved_chunks: Retrieved code chunks
        
        Returns:
            List of inferred citations
        """
        import re
        import os
        citations = []
        seen_keys = set()
        
        # Strategy 1: Extract full file paths with extensions
        file_pattern = r'([a-zA-Z0-9_/\\\.-]+\.(?:py|js|ts|java|go|rs|cpp|c|h|tsx|jsx|md|txt))'
        mentioned_files = set(re.findall(file_pattern, answer_text))
        
        # Strategy 2: Extract component/class names (PascalCase or UPPER_CASE)
        component_pattern = r'\b([A-Z][a-zA-Z0-9]+(?:[A-Z][a-zA-Z0-9]+)*)\b'
        mentioned_components = set(re.findall(component_pattern, answer_text))
        
        # Strategy 3: Extract file names without paths (e.g., "App.tsx", "main.py")
        filename_pattern = r'\b([a-zA-Z0-9_-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|c|h|md|txt))\b'
        mentioned_filenames = set(re.findall(filename_pattern, answer_text))
        
        # Match mentioned files/components with retrieved chunks
        for chunk in retrieved_chunks:
            file_path = chunk.get('file_path', chunk.get('metadata', {}).get('file_path', ''))
            if not file_path:
                continue
                
            start_line = chunk.get('start_line', chunk.get('metadata', {}).get('start_line', 0))
            end_line = chunk.get('end_line', chunk.get('metadata', {}).get('end_line', 0))
            
            # Get filename and basename for matching
            filename = os.path.basename(file_path)
            basename = os.path.splitext(filename)[0]
            
            # Check if this chunk should be cited
            should_cite = False
            
            # Match 1: Exact file path match
            if file_path in mentioned_files:
                should_cite = True
            # Match 2: Filename match
            elif filename in mentioned_filenames:
                should_cite = True
            # Match 3: Component name matches filename (e.g., "EmergencyButton" matches "EmergencyButton.tsx")
            elif basename in mentioned_components:
                should_cite = True
            # Match 4: Partial path match (e.g., "App.tsx" matches "src/components/App.tsx")
            elif any(mentioned_file in file_path or file_path.endswith(mentioned_file) 
                    for mentioned_file in mentioned_files):
                should_cite = True
            # Match 5: If answer mentions the file but in a different format
            elif any(mentioned_filename == filename for mentioned_filename in mentioned_filenames):
                should_cite = True
            
            # Add citation if matched and not duplicate
            if should_cite:
                key = (file_path, start_line)
                if key not in seen_keys:
                    seen_keys.add(key)
                    citations.append({
                        'file_path': file_path,
                        'start_line': start_line,
                        'end_line': end_line,
                        'text_snippet': ''
                    })
        
        # Strategy 4: If still no citations but we have chunks, use top chunks
        # (since they were used to generate the answer)
        if not citations and retrieved_chunks:
            # Use top 5 chunks as citations
            for chunk in retrieved_chunks[:5]:
                file_path = chunk.get('file_path', chunk.get('metadata', {}).get('file_path', ''))
                if file_path:
                    start_line = chunk.get('start_line', chunk.get('metadata', {}).get('start_line', 0))
                    end_line = chunk.get('end_line', chunk.get('metadata', {}).get('end_line', 0))
                    key = (file_path, start_line)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        citations.append({
                            'file_path': file_path,
                            'start_line': start_line,
                            'end_line': end_line,
                            'text_snippet': ''
                        })
        
        return citations
    
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
