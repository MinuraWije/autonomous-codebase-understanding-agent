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
        
        # Group citations by file
        grouped = self._group_citations_by_file(citations)
        
        reference_section = "\n\n### References:\n"
        
        for file_path, file_citations in grouped.items():
            reference_section += f"\n**{file_path}**\n"
            for citation in file_citations:
                line_range = (
                    f"{citation['start_line']}-{citation['end_line']}"
                    if citation['start_line'] != citation['end_line']
                    else str(citation['start_line'])
                )
                reference_section += f"  - Lines {line_range}"
                if citation.get('text_snippet') and citation['text_snippet'] != '[Code snippet unavailable]':
                    # Show a preview of the code
                    snippet_preview = citation['text_snippet'][:100].replace('\n', ' ')
                    if len(citation['text_snippet']) > 100:
                        snippet_preview += "..."
                    reference_section += f": `{snippet_preview}`"
                reference_section += "\n"
        
        return reference_section
    
    def _group_citations_by_file(self, citations: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group citations by file path.
        
        Args:
            citations: List of citation dictionaries
        
        Returns:
            Dictionary mapping file paths to lists of citations
        """
        grouped = {}
        for citation in citations:
            file_path = citation.get('file_path', 'unknown')
            if file_path not in grouped:
                grouped[file_path] = []
            grouped[file_path].append(citation)
        
        # Sort citations within each file by line number
        for file_path in grouped:
            grouped[file_path].sort(key=lambda c: c.get('start_line', 0))
        
        return grouped
    
    def extract_summary(self, answer_text: str, max_sentences: int = 2) -> tuple:
        """
        Extract a summary from the answer (first 1-2 sentences) and return remaining text.
        
        Args:
            answer_text: Full answer text
            max_sentences: Maximum number of sentences for summary
        
        Returns:
            Tuple of (summary_text, remaining_text_without_summary)
        """
        import re
        
        # Remove common prefixes like "Brief Summary:", "Summary:", etc.
        cleaned_text = re.sub(r'^(Brief\s+)?Summary:\s*', '', answer_text, flags=re.IGNORECASE)
        
        # Remove citations for sentence splitting
        text_without_citations = re.sub(r'\[[^\]]+:\d+(?:-\d+)?\]', '', cleaned_text)
        
        # Split into sentences (preserve punctuation)
        # Use a pattern that captures sentence endings
        sentence_pattern = r'([^.!?]+[.!?]+)'
        sentences = re.findall(sentence_pattern, text_without_citations)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            # Fallback: split by periods
            sentences = re.split(r'\.+', text_without_citations)
            sentences = [s.strip() + '.' for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if len(sentences) <= max_sentences:
            # All text is summary, return it all as summary with empty remaining
            summary = ' '.join(sentences).strip()
            # Remove "Brief Summary:" if it appears
            summary = re.sub(r'^(Brief\s+)?Summary:\s*', '', summary, flags=re.IGNORECASE)
            return summary, ""
        
        # Extract summary sentences
        summary_sentences = sentences[:max_sentences]
        remaining_sentences = sentences[max_sentences:]
        
        summary = ' '.join(summary_sentences).strip()
        # Remove "Brief Summary:" prefix if present
        summary = re.sub(r'^(Brief\s+)?Summary:\s*', '', summary, flags=re.IGNORECASE)
        if not summary.endswith('.') and not summary.endswith('!') and not summary.endswith('?'):
            summary += '.'
        
        # Find where summary ends in original text to preserve citations
        # Build a pattern to match the summary text (allowing for citations)
        summary_words = summary.split()[:10]  # Use first 10 words for matching
        summary_start_pattern = ' '.join(summary_words[:3])  # Use first 3 words
        
        # Find the position after summary in original text
        # Look for the end of the summary by finding where remaining sentences start
        remaining_start = ' '.join(remaining_sentences[0].split()[:5]) if remaining_sentences else ""
        
        # Try to find the split point in original text
        remaining_text = cleaned_text
        
        # Method 1: Find where remaining sentences start
        if remaining_start:
            # Find position of remaining text start
            remaining_pos = cleaned_text.find(remaining_start)
            if remaining_pos > 0:
                remaining_text = cleaned_text[remaining_pos:].strip()
            else:
                # Method 2: Find position after summary sentences
                # Count characters in summary and find similar position
                summary_length = len(summary)
                # Look for a good split point (after summary, before remaining)
                # Find first sentence break after summary length
                for i in range(summary_length, min(summary_length + 200, len(cleaned_text))):
                    if cleaned_text[i] in '.!?' and i + 1 < len(cleaned_text):
                        if cleaned_text[i + 1] in ' \n':
                            remaining_text = cleaned_text[i + 1:].strip()
                            break
        else:
            # If we can't find remaining, just use everything after first paragraph
            # Look for double newline or significant break
            para_break = cleaned_text.find('\n\n')
            if para_break > 0:
                remaining_text = cleaned_text[para_break + 2:].strip()
        
        # If remaining text is too short or same as original, don't split
        if len(remaining_text) < 50 or remaining_text == cleaned_text:
            return summary, ""
        
        return summary, remaining_text
    
    def format_code_snippets(self, citations: List[Dict], max_snippets: int = 3) -> str:
        """
        Format code snippets from citations.
        
        Args:
            citations: List of citation dictionaries with text_snippet
            max_snippets: Maximum number of snippets to show
        
        Returns:
            Formatted code snippets section
        """
        snippets_with_code = [
            c for c in citations
            if c.get('text_snippet') and c['text_snippet'] != '[Code snippet unavailable]'
        ][:max_snippets]
        
        if not snippets_with_code:
            return ""
        
        snippets_section = "\n\n### Code Examples:\n"
        
        for i, citation in enumerate(snippets_with_code, 1):
            file_path = citation.get('file_path', 'unknown')
            line_range = (
                f"{citation['start_line']}-{citation['end_line']}"
                if citation['start_line'] != citation['end_line']
                else str(citation['start_line'])
            )
            
            snippets_section += f"\n**Example {i}: {file_path} (lines {line_range})**\n"
            snippets_section += "```\n"
            snippets_section += citation['text_snippet']
            if not citation['text_snippet'].endswith('\n'):
                snippets_section += "\n"
            snippets_section += "```\n"
        
        return snippets_section
    
    def remove_redundant_citations(self, answer_text: str, citations: List[Dict]) -> tuple:
        """
        Remove redundant citations and clean up answer text.
        
        Args:
            answer_text: Original answer text
            citations: List of citations
        
        Returns:
            Tuple of (cleaned_answer, unique_citations)
        """
        import re
        
        # Extract all citation patterns from answer
        citation_pattern = r'\[([^\]]+?):(\d+)(?:-(\d+))?\]'
        matches = re.findall(citation_pattern, answer_text)
        
        # Create set of unique citations
        seen_citations = set()
        unique_citations = []
        citation_map = {}
        
        for match in matches:
            file_path = match[0].strip()
            start_line = int(match[1])
            key = (file_path, start_line)
            
            if key not in seen_citations:
                seen_citations.add(key)
                # Find matching citation in list
                for citation in citations:
                    if (citation.get('file_path') == file_path and 
                        citation.get('start_line') == start_line):
                        unique_citations.append(citation)
                        citation_map[key] = citation
                        break
        
        # If we have unique citations, use them; otherwise use all citations
        if unique_citations:
            return answer_text, unique_citations
        
        return answer_text, citations
    
    def post_process_answer(
        self,
        answer_text: str,
        citations: List[Dict],
        include_summary: bool = True,
        include_code_snippets: bool = True
    ) -> str:
        """
        Post-process answer with summary, code snippets, and formatted citations.
        
        Args:
            answer_text: Original answer text
            citations: List of citations
            include_summary: Whether to add summary at top
            include_code_snippets: Whether to include code examples
        
        Returns:
            Post-processed answer
        """
        # Remove redundant citations
        cleaned_answer, unique_citations = self.remove_redundant_citations(answer_text, citations)
        
        # Build structured answer
        parts = []
        
        # 1. Summary (if requested) - extract and remove from main answer to avoid duplication
        main_answer = cleaned_answer
        if include_summary:
            summary, remaining_text = self.extract_summary(cleaned_answer)
            if summary and remaining_text:
                # Use remaining text as main answer (summary removed)
                main_answer = remaining_text.strip()
                parts.append(f"**Summary:** {summary}\n")
            elif summary:
                # Everything was summary, don't duplicate
                parts.append(f"**Summary:** {summary}\n")
                main_answer = ""  # No main answer if everything was summary
        
        # 2. Main answer (with citations preserved)
        if main_answer:
            # Format "Detailed Explanation:" on a new line if present
            import re
            # Check if answer starts with "Detailed Explanation:" or similar
            detailed_pattern = r'^(Detailed\s+)?Explanation:\s*'
            if re.match(detailed_pattern, main_answer, re.IGNORECASE):
                # Remove the prefix and add it on a new line
                main_answer = re.sub(detailed_pattern, '', main_answer, flags=re.IGNORECASE).strip()
                parts.append("**Detailed Explanation:**\n" + main_answer)
            else:
                parts.append(main_answer)
        
        # 3. Code snippets (if requested and available)
        if include_code_snippets and unique_citations:
            code_snippets = self.format_code_snippets(unique_citations)
            if code_snippets:
                parts.append(code_snippets)
        
        # 4. References section
        references = self.format_citations_for_answer(unique_citations)
        if references:
            parts.append(references)
        
        return "\n".join(parts)
