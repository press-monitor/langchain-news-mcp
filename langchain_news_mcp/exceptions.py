"""
Exception classes for LangChain News MCP Integration.
"""


class NewsMCPError(Exception):
    """Base exception for News MCP operations."""
    pass


class NewsAPIError(NewsMCPError):
    """Exception for API-related errors."""
    pass


class NewsConnectionError(NewsMCPError):
    """Exception for connection-related errors."""
    pass


class NewsValidationError(NewsMCPError):
    """Exception for input validation errors."""
    pass


class NewsTimeoutError(NewsMCPError):
    """Exception for timeout errors."""
    pass