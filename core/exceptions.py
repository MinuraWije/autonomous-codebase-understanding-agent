"""Custom exceptions for the application."""


class RepositoryNotFoundError(Exception):
    """Raised when a repository is not found."""
    pass


class FileNotFoundError(Exception):
    """Raised when a file is not found."""
    pass


class IndexingError(Exception):
    """Raised when indexing fails."""
    pass


class AgentExecutionError(Exception):
    """Raised when agent execution fails."""
    pass


class LLMError(Exception):
    """Raised when LLM operations fail."""
    pass
