"""Error handling utilities."""
from typing import Callable, Any, Optional
from functools import wraps
from core.exceptions import (
    RepositoryNotFoundError,
    FileNotFoundError,
    IndexingError,
    AgentExecutionError,
    LLMError
)


def handle_errors(
    default_return: Any = None,
    log_error: bool = True
) -> Callable:
    """
    Decorator for error handling.
    
    Args:
        default_return: Default value to return on error
        log_error: Whether to log errors
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RepositoryNotFoundError as e:
                if log_error:
                    print(f"Repository not found: {e}")
                return default_return
            except FileNotFoundError as e:
                if log_error:
                    print(f"File not found: {e}")
                return default_return
            except (IndexingError, AgentExecutionError, LLMError) as e:
                if log_error:
                    print(f"Error in {func.__name__}: {e}")
                return default_return
            except Exception as e:
                if log_error:
                    print(f"Unexpected error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    exception_type: Optional[type] = None,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default_return: Default value to return on error
        exception_type: Specific exception type to catch
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except exception_type if exception_type else Exception as e:
        print(f"Error executing {func.__name__}: {e}")
        return default_return
