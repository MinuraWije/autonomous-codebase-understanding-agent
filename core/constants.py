"""Application-wide constants."""
from typing import Dict, List

# Default configuration values
DEFAULT_MAX_RETRIEVAL_ITERATIONS = 3
DEFAULT_MAX_CHUNKS_PER_QUERY = 12
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_CITATIONS = 15
DEFAULT_SNIPPET_LENGTH = 300

# LLM temperature settings
PLANNER_TEMPERATURE = 0.0
SYNTHESIZER_TEMPERATURE = 0.0
VERIFIER_TEMPERATURE = 0.0
SUMMARY_TEMPERATURE = 0.3

# Search configuration
VECTOR_SEARCH_WEIGHT = 0.7
LEXICAL_SEARCH_WEIGHT = 0.3
RANK_BOOST_FACTOR = 0.3
OVERLAP_THRESHOLD = 0.5

# Key file patterns for repository analysis
KEY_FILE_PATTERNS: List[str] = [
    'main.py', 'app.py', '__init__.py', 'index.js', 'index.ts',
    'server.py', 'server.js', 'api.py', 'routes.py', 'views.py',
    'Main.java', 'Application.java', 'main.go', 'README.md'
]

# Stop words for keyword extraction
STOP_WORDS: set = {
    'how', 'what', 'where', 'when', 'why', 'who', 'which', 'is', 'are', 'the',
    'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
    'does', 'do', 'did', 'can', 'could', 'would', 'should', 'will', 'be'
}

# File patterns to ignore
IGNORED_PATTERNS: List[str] = [
    '.git', '__pycache__', 'node_modules', 'venv', '.env', '.DS_Store'
]

# Error messages
ERROR_REPO_NOT_FOUND = "Repository not found: {repo_id}"
ERROR_FILE_NOT_FOUND = "File not found: {file_path}"
ERROR_INDEXING_FAILED = "Indexing failed: {error}"
ERROR_AGENT_EXECUTION = "Error running agent: {error}"

# Response messages
MESSAGE_INDEXING_STARTED = "Indexing started in background. Check status at /repos/{repo_id}/status"
MESSAGE_REPO_DELETED = "Repository {repo_id} deleted successfully"
