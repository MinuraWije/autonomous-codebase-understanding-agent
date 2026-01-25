"""Application-wide constants."""
from typing import Dict, List

# Default configuration values
DEFAULT_MAX_RETRIEVAL_ITERATIONS = 3
DEFAULT_MAX_CHUNKS_PER_QUERY = 12
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_CITATIONS = 15
DEFAULT_SNIPPET_LENGTH = 300
MIN_CHUNK_SIZE_TOKENS = 50  # Minimum tokens before merging small chunks
MAX_CONTEXT_LINES = 10  # Maximum lines to look back for comments/docstrings

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

# Reranking configuration
MULTI_TERM_MATCH_BOOST = 0.15  # Boost per additional matching keyword
TEST_FILE_PENALTY = -0.2  # Penalty for test files when searching implementation
DOC_FILE_PENALTY = -0.15  # Penalty for documentation files when searching implementation
PATH_DEPTH_BOOST = 0.05  # Boost per level closer to root (max 3 levels)

# Test file patterns
TEST_FILE_PATTERNS: List[str] = [
    'test_', '_test', 'spec_', '_spec', '.test.', '.spec.',
    'tests/', 'test/', '__tests__/', 'specs/', 'spec/'
]

# Documentation file patterns
DOC_FILE_PATTERNS: List[str] = [
    'readme', 'changelog', 'license', 'contributing', 'docs/',
    'documentation/', '.md', '.txt', '.rst'
]

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

# Query expansion synonyms for common technical terms
QUERY_EXPANSIONS: Dict[str, List[str]] = {
    # Authentication & Security
    'auth': ['authentication', 'login', 'session', 'token', 'jwt', 'oauth', 'credential'],
    'authentication': ['auth', 'login', 'session', 'token', 'jwt', 'oauth', 'credential'],
    'login': ['authentication', 'auth', 'session', 'credential', 'signin'],
    'session': ['authentication', 'auth', 'login', 'token', 'cookie'],
    'token': ['jwt', 'authentication', 'auth', 'session', 'bearer'],
    
    # Database & Storage
    'database': ['db', 'datastore', 'storage', 'persistence', 'repository'],
    'db': ['database', 'datastore', 'storage', 'persistence'],
    'query': ['search', 'filter', 'select', 'find', 'retrieve'],
    'storage': ['database', 'db', 'persistence', 'cache'],
    
    # API & HTTP
    'api': ['endpoint', 'route', 'handler', 'controller', 'service'],
    'endpoint': ['api', 'route', 'handler', 'controller'],
    'route': ['endpoint', 'api', 'handler', 'path', 'url'],
    'request': ['http', 'api', 'endpoint', 'call'],
    'response': ['return', 'output', 'result', 'reply'],
    
    # Error Handling
    'error': ['exception', 'failure', 'issue', 'problem', 'bug'],
    'exception': ['error', 'failure', 'throw', 'catch'],
    'validation': ['validate', 'check', 'verify', 'sanitize'],
    
    # Configuration & Setup
    'config': ['configuration', 'settings', 'options', 'parameters'],
    'setup': ['initialize', 'configure', 'install', 'bootstrap'],
    'init': ['initialize', 'setup', 'bootstrap', 'start'],
    
    # Data Processing
    'process': ['handle', 'execute', 'run', 'perform', 'do'],
    'handle': ['process', 'manage', 'deal', 'execute'],
    'transform': ['convert', 'change', 'modify', 'map'],
    
    # Testing
    'test': ['testing', 'spec', 'unit', 'integration', 'assert'],
    'testing': ['test', 'spec', 'unit', 'integration'],
    
    # Common patterns
    'middleware': ['interceptor', 'filter', 'handler', 'processor'],
    'service': ['api', 'handler', 'controller', 'manager'],
    'model': ['schema', 'entity', 'data', 'structure'],
    'view': ['template', 'render', 'display', 'ui'],
    'controller': ['handler', 'endpoint', 'route', 'service'],
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
