"""Prompt templates for the agent."""
import json


def get_planner_prompt(question: str) -> str:
    """Get the planner prompt with few-shot examples."""
    return f"""You are a code analyst planning how to answer a question about a codebase.

Question: {question}

Your task is to create a search plan. Follow these steps:
1. Analyze what the question is asking for
2. Identify key concepts, functions, or patterns to search for
3. Create 2-4 diverse, specific search queries
4. Predict which files might contain the answer

OUTPUT FORMAT: Output ONLY valid JSON with these fields:
- "reasoning": Brief explanation (1-2 sentences) of your approach
- "search_queries": List of 2-4 specific search queries
- "expected_files": List of file patterns you expect to find

EXAMPLES:

Example 1 - Good:
Question: "Where is user authentication handled?"
{{
  "reasoning": "Need to find authentication logic, likely in middleware, auth service, or login handlers",
  "search_queries": ["authentication middleware implementation", "user login handler", "JWT token validation", "session management setup"],
  "expected_files": ["auth.py", "middleware.py", "login.py", "security.py"]
}}

Example 2 - Good:
Question: "How does the API handle error responses?"
{{
  "reasoning": "Looking for error handling in API routes, exception handlers, and response formatting",
  "search_queries": ["API error handler implementation", "exception response formatting", "HTTP error status codes", "error middleware"],
  "expected_files": ["api.py", "errors.py", "middleware.py", "handlers.py"]
}}

Example 3 - Bad (too vague):
{{
  "reasoning": "Find code",
  "search_queries": ["authentication", "code", "function"],
  "expected_files": []
}}

GUIDELINES:
- Make queries specific: include action words (handle, validate, process) and context
- Use diverse angles: search for the same concept from different perspectives
- Think about where code lives: main files, services, utilities, middleware
- Avoid single-word queries or generic terms

Now create a plan for the question above. Output ONLY valid JSON, no other text:"""


def get_synthesizer_prompt(question: str, chunks: list) -> str:
    """Get the synthesizer prompt with few-shot examples."""
    
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
    
    return f"""You are a code analyst answering questions about a codebase. Answer using ONLY the provided code chunks.

Question: {question}

Retrieved Code:
{chunks_text}

INSTRUCTIONS:
1. Read all code chunks carefully
2. Identify which chunks are relevant to the question
3. Synthesize an answer that directly addresses the question
4. Structure your answer: brief summary, then detailed explanation with citations
5. Cite EVERY claim immediately after making it using [file_path:start_line-end_line]

CITATION FORMAT:
- Use [file_path:start_line-end_line] format
- Example: [src/auth/middleware.py:45-67]
- For single lines: [src/auth/middleware.py:45]
- Cite immediately after each claim or code reference

EXAMPLE GOOD ANSWER:

Question: "Where is authentication handled?"

Answer:
Authentication is handled in two main locations:

1. The authentication middleware is defined in [src/middleware/auth.py:12-45]. This middleware extracts the JWT token from the Authorization header and validates it using the verify_token function.

2. Token verification occurs in [src/services/auth_service.py:23-56], where the verify_token function decodes the JWT and checks its signature against the SECRET_KEY configured in [src/config/settings.py:18].

The middleware is registered in the main application at [src/app.py:67-70], which applies it to all routes except those in the public routes list.

EXAMPLE BAD ANSWER (DO NOT DO THIS):

Authentication is handled in the auth middleware. The code validates tokens and checks permissions. [Some citation that doesn't match the chunks]

CRITICAL RULES:
1. ONLY use information from the chunks above - do not make assumptions
2. Cite EVERY claim with exact file paths and line numbers from chunk headers
3. If information is missing, say "Not found in retrieved code" rather than guessing
4. Be specific: mention function names, class names, and exact locations
5. Structure: Start with a direct answer, then provide supporting details with citations

Now answer the question above. Use the exact file paths and line numbers from the chunk headers:"""


def get_verifier_prompt(question: str, draft_answer: str, chunks: list) -> str:
    """Get the verifier prompt with structured instructions."""
    
    # Format chunks for verification
    chunks_summary = []
    for chunk in chunks:
        file_path = chunk.get('file_path', chunk.get('metadata', {}).get('file_path', 'unknown'))
        start_line = chunk.get('start_line', chunk.get('metadata', {}).get('start_line', 0))
        end_line = chunk.get('end_line', chunk.get('metadata', {}).get('end_line', 0))
        text_preview = chunk.get('text', chunk.get('chunk_text', ''))[:200]
        
        chunks_summary.append(f"- {file_path}:{start_line}-{end_line}: {text_preview}...")
    
    chunks_text = '\n'.join(chunks_summary)
    
    return f"""You are a code verification expert. Verify if the answer is fully grounded in the provided code chunks.

Question: {question}

Answer to verify:
{draft_answer}

Retrieved Code Chunks:
{chunks_text}

VERIFICATION PROCESS:
1. Extract all claims from the answer (each statement that makes an assertion)
2. For each claim, check if it's supported by a code chunk
3. Verify citations match actual chunk content at those line numbers
4. Identify any unsupported claims or hallucinations
5. Determine if additional information is needed to fully answer the question

CHECKLIST:
- [ ] Every claim has a citation
- [ ] Citations reference actual chunks (file paths and line numbers match)
- [ ] No claims about code not in the chunks
- [ ] Citations point to relevant code (not just any code)
- [ ] Answer directly addresses the question

OUTPUT FORMAT: Output ONLY valid JSON:
{{
  "is_grounded": true or false,
  "unsupported_claims": ["exact claim text that's not supported", "another unsupported claim"],
  "missing_information": ["specific information needed", "what would help answer better"],
  "follow_up_queries": ["specific search query 1", "specific search query 2"]
}}

EXAMPLES:

Example 1 - Well-grounded answer:
{{
  "is_grounded": true,
  "unsupported_claims": [],
  "missing_information": [],
  "follow_up_queries": []
}}

Example 2 - Answer with gaps:
{{
  "is_grounded": false,
  "unsupported_claims": ["Claims middleware is registered in app.py but no chunk shows this"],
  "missing_information": ["How the middleware is registered", "What routes are excluded from auth"],
  "follow_up_queries": ["middleware registration in app.py", "public routes configuration"]
}}

Example 3 - Citation mismatch:
{{
  "is_grounded": false,
  "unsupported_claims": ["Citation [auth.py:45-67] claims to show login function but chunk shows validate_token"],
  "missing_information": ["Actual login function implementation"],
  "follow_up_queries": ["user login function implementation", "login endpoint handler"]
}}

GUIDELINES:
- Be strict: if a claim can't be verified in chunks, mark it as unsupported
- Be specific: quote exact claim text that's problematic
- Generate actionable queries: follow-up queries should be specific search terms
- If answer is well-grounded, set is_grounded to true and leave lists empty

Now verify the answer above. Output ONLY valid JSON, no other text:"""


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
    """
    Extract citations from answer text.
    
    Supports multiple citation formats:
    - [file_path:start_line-end_line] (standard format)
    - [file_path:line] (single line)
    - (file_path:start_line-end_line) (parentheses format)
    - file_path:start_line-end_line (no brackets, with file extension)
    """
    import re
    
    citations = []
    seen_keys = set()
    
    # Pattern 1: [file_path:start_line-end_line] or [file_path:line]
    _extract_with_pattern(
        answer_text,
        r'\[([^\]]+?):(\d+)(?:-(\d+))?\]',
        citations,
        seen_keys
    )
    
    # Pattern 2: (file_path:start_line-end_line) or (file_path:line)
    _extract_with_pattern(
        answer_text,
        r'\(([^)]+?):(\d+)(?:-(\d+))?\)',
        citations,
        seen_keys
    )
    
    # Pattern 3: file_path:start_line-end_line (no brackets, requires file extension)
    # Simplified pattern to reduce complexity
    file_extensions = r'py|js|ts|java|go|rs|cpp|c|h|tsx|jsx|md|txt'
    pattern3 = rf'([a-zA-Z0-9_/\\\.-]+\.(?:{file_extensions})):(\d+)(?:-(\d+))?(?=\s|$|,|\.|;|\))'
    _extract_with_pattern(
        answer_text,
        pattern3,
        citations,
        seen_keys
    )
    
    return citations


def _extract_with_pattern(
    text: str,
    pattern: str,
    citations: list,
    seen_keys: set
) -> None:
    """
    Extract citations using a regex pattern and add to citations list.
    
    Args:
        text: Text to search
        pattern: Regex pattern with groups (file_path, start_line, optional end_line)
        citations: List to append citations to
        seen_keys: Set of (file_path, start_line) tuples to avoid duplicates
    """
    import re
    
    matches = re.findall(pattern, text)
    for match in matches:
        file_path = match[0].strip()
        start_line = int(match[1])
        end_line = int(match[2]) if len(match) > 2 and match[2] else start_line
        
        key = (file_path, start_line)
        if key not in seen_keys:
            seen_keys.add(key)
            citations.append({
                'file_path': file_path,
                'start_line': start_line,
                'end_line': end_line,
                'text_snippet': ''
            })
