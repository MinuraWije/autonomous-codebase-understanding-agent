"""Prompt templates for the agent."""
import json


def get_planner_prompt(question: str) -> str:
    """Get the planner prompt."""
    return f"""You are a code analyst planning how to answer a question about a codebase.

Question: {question}

Your task is to create a search plan. Output a JSON object with:
- "reasoning": Brief explanation of your approach
- "search_queries": List of 2-4 specific search queries to find relevant code
- "expected_files": List of file patterns you expect to find (e.g., "auth.py", "middleware")

Make queries specific and diverse. Good examples:
- "authentication middleware setup"
- "request validation logic"
- "database connection initialization"
- "user login endpoint implementation"

Bad examples (too vague):
- "authentication"
- "code"
- "function"

Output ONLY valid JSON, no other text:
{{
  "reasoning": "your reasoning here",
  "search_queries": ["query1", "query2", "query3"],
  "expected_files": ["file1.py", "file2.js"]
}}"""


def get_synthesizer_prompt(question: str, chunks: list) -> str:
    """Get the synthesizer prompt."""
    
    # Format chunks with citations
    chunks_text = ""
    for i, chunk in enumerate(chunks, 1):
        file_path = chunk.get('file_path', chunk.get('metadata', {}).get('file_path', 'unknown'))
        start_line = chunk.get('start_line', chunk.get('metadata', {}).get('start_line', 0))
        end_line = chunk.get('end_line', chunk.get('metadata', {}).get('end_line', 0))
        text = chunk.get('text', chunk.get('chunk_text', ''))
        symbol = chunk.get('symbol_name', chunk.get('metadata', {}).get('symbol_name', ''))
        
        chunks_text += f"\n--- Chunk {i}: {file_path}:{start_line}-{end_line}"
        if symbol:
            chunks_text += f" (Symbol: {symbol})"
        chunks_text += f" ---\n{text}\n"
    
    return f"""Answer the question using ONLY the provided code chunks below.

Question: {question}

Retrieved Code:
{chunks_text}

CRITICAL RULES:
1. Cite EVERY claim with [file_path:start_line-end_line] format
2. Only make claims supported by the retrieved code
3. If information is not in the chunks, say "Not found in retrieved code"
4. Be specific about file paths and line numbers
5. Do not make assumptions about code you haven't seen

Example citation format: [src/auth/middleware.py:45-67]

Answer:"""


def get_verifier_prompt(question: str, draft_answer: str, chunks: list) -> str:
    """Get the verifier prompt."""
    
    # Format chunks for verification
    chunks_summary = []
    for chunk in chunks:
        file_path = chunk.get('file_path', chunk.get('metadata', {}).get('file_path', 'unknown'))
        start_line = chunk.get('start_line', chunk.get('metadata', {}).get('start_line', 0))
        end_line = chunk.get('end_line', chunk.get('metadata', {}).get('end_line', 0))
        text_preview = chunk.get('text', chunk.get('chunk_text', ''))[:200]
        
        chunks_summary.append(f"- {file_path}:{start_line}-{end_line}: {text_preview}...")
    
    chunks_text = '\n'.join(chunks_summary)
    
    return f"""Verify if the answer is fully supported by the retrieved code chunks.

Question: {question}

Answer to verify:
{draft_answer}

Retrieved Code Chunks:
{chunks_text}

For each claim in the answer:
1. Is it supported by a code chunk?
2. Does the citation match actual content?
3. Are there unsupported claims or hallucinations?

Output ONLY valid JSON:
{{
  "is_grounded": true or false,
  "unsupported_claims": ["claim1", "claim2"],
  "missing_information": ["what additional info would help answer better"],
  "follow_up_queries": ["specific query 1", "specific query 2"]
}}

If the answer is well-supported, set is_grounded to true and leave the lists empty.
If there are gaps, provide specific follow-up queries to fill them.

Output ONLY valid JSON, no other text:"""


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    # Try to find JSON in the response
    import re
    
    # Remove markdown code blocks
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    
    # Try to parse
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    
    # Fallback
    return {}


def extract_citations(answer_text: str) -> list:
    """Extract citations from answer text."""
    import re
    
    pattern = r'\[([^\]]+):(\d+)-(\d+)\]'
    matches = re.findall(pattern, answer_text)
    
    citations = []
    for file_path, start_line, end_line in matches:
        citations.append({
            'file_path': file_path,
            'start_line': int(start_line),
            'end_line': int(end_line),
            'text_snippet': ''  # Will be filled later if needed
        })
    
    return citations
