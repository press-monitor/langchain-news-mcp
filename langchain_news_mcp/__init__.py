"""
LangChain News MCP Integration

Real-time global news intelligence for LangChain agents and applications.
"""

__version__ = "1.0.0"
__author__ = "News MCP Team"
__email__ = "support@pressmonitor.com"
__license__ = "MIT"

from .tools import (
    NewsTool,
    NewsSearchTool,
    NewsHeadlinesTool,
    NewsBriefsTool,
    NewsFullTextTool,
)

from .retrievers import (
    NewsRetriever,
    NewsVectorRetriever,
    NewsTimeRetriever,
)

from .loaders import (
    NewsLoader,
    NewsBatchLoader,
    NewsStreamLoader,
)

from .client import NewsMCPClient
from .exceptions import NewsMCPError, NewsAPIError, NewsConnectionError

__all__ = [
    # Tools
    "NewsTool",
    "NewsSearchTool",
    "NewsHeadlinesTool",
    "NewsBriefsTool",
    "NewsFullTextTool",

    # Retrievers
    "NewsRetriever",
    "NewsVectorRetriever",
    "NewsTimeRetriever",

    # Loaders
    "NewsLoader",
    "NewsBatchLoader",
    "NewsStreamLoader",

    # Client
    "NewsMCPClient",

    # Exceptions
    "NewsMCPError",
    "NewsAPIError",
    "NewsConnectionError",
]