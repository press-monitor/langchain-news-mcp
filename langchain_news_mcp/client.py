"""
News MCP Client for LangChain Integration

Provides the core client functionality for communicating with News MCP Server.
"""

import os
import asyncio
from typing import Dict, List, Any, Optional, Union
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .exceptions import NewsMCPError, NewsAPIError, NewsConnectionError


class NewsMCPClient:
    """
    Client for News MCP Server with retry logic and connection pooling.
    """

    def __init__(
        self,
        server_url: str = None,
        api_key: str = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: Dict[str, str] = None,
    ):
        """
        Initialize the News MCP client.

        Args:
            server_url: URL of the News MCP Server
            api_key: API key for authentication (optional)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            headers: Additional headers to send with requests
        """
        self.server_url = server_url or os.getenv("NEWS_MCP_SERVER_URL", "http://localhost:3000")
        self.api_key = api_key or os.getenv("NEWS_MCP_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries

        # Prepare headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        if headers:
            self.headers.update(headers)

        # HTTP client configuration
        self.client_config = {
            "timeout": httpx.Timeout(timeout),
            "limits": httpx.Limits(max_keepalive_connections=10, max_connections=20),
            "headers": self.headers,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def call_method(
        self,
        method: str,
        params: Dict[str, Any],
        request_id: str = "1"
    ) -> Dict[str, Any]:
        """
        Call an MCP method with retry logic.

        Args:
            method: MCP method name
            params: Method parameters
            request_id: Request ID for tracking

        Returns:
            MCP response result

        Raises:
            NewsConnectionError: Connection or network errors
            NewsAPIError: API or server errors
            NewsMCPError: MCP protocol errors
        """
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }

        try:
            async with httpx.AsyncClient(**self.client_config) as client:
                response = await client.post(
                    f"{self.server_url}/mcp",
                    json=payload
                )

                # Check HTTP status
                if response.status_code == 404:
                    raise NewsConnectionError(f"News MCP Server not found at {self.server_url}")
                elif response.status_code == 401:
                    raise NewsAPIError("Authentication failed - check your API key")
                elif response.status_code == 429:
                    raise NewsAPIError("Rate limit exceeded")
                elif response.status_code >= 500:
                    raise NewsAPIError(f"Server error: {response.status_code}")

                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except Exception as e:
                    raise NewsMCPError(f"Invalid JSON response: {e}")

                # Check for MCP errors
                if "error" in data:
                    error = data["error"]
                    raise NewsMCPError(f"MCP Error {error.get('code', 'unknown')}: {error.get('message', 'Unknown error')}")

                # Return result
                return data.get("result", {})

        except httpx.ConnectError as e:
            raise NewsConnectionError(f"Cannot connect to News MCP Server: {e}")
        except httpx.TimeoutException as e:
            raise NewsConnectionError(f"Request timeout: {e}")
        except httpx.HTTPError as e:
            raise NewsConnectionError(f"HTTP error: {e}")

    def call_method_sync(
        self,
        method: str,
        params: Dict[str, Any],
        request_id: str = "1"
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for call_method.

        Args:
            method: MCP method name
            params: Method parameters
            request_id: Request ID for tracking

        Returns:
            MCP response result
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.call_method(method, params, request_id)
        )

    async def search_headlines(
        self,
        query: str,
        country_code: str = None,
        lang_code: str = None,
        count: int = 10,
        sort: str = "latest"
    ) -> List[Dict[str, Any]]:
        """
        Search for news headlines.

        Args:
            query: Search query
            country_code: Country filter (ISO code)
            lang_code: Language filter (ISO code)
            count: Number of results
            sort: Sort order ('latest' or 'relevance')

        Returns:
            List of news articles
        """
        params = {
            "query_text": query,
            "count": count,
            "sort": sort
        }

        if country_code:
            params["country_code"] = country_code
        if lang_code:
            params["lang_code"] = lang_code

        result = await self.call_method("news_headlines", params)
        return result.get("items", [])

    async def search_briefs(
        self,
        query: str,
        country_code: str = None,
        lang_code: str = None,
        count: int = 10,
        sort: str = "latest"
    ) -> List[Dict[str, Any]]:
        """
        Search for news briefs with summaries.

        Args:
            query: Search query
            country_code: Country filter (ISO code)
            lang_code: Language filter (ISO code)
            count: Number of results
            sort: Sort order ('latest' or 'relevance')

        Returns:
            List of news articles with summaries
        """
        params = {
            "query_text": query,
            "count": count,
            "sort": sort
        }

        if country_code:
            params["country_code"] = country_code
        if lang_code:
            params["lang_code"] = lang_code

        result = await self.call_method("news_briefs", params)
        return result.get("items", [])

    async def search_fulltext(
        self,
        query: str,
        country_code: str = None,
        lang_code: str = None,
        count: int = 5,
        sort: str = "relevance"
    ) -> List[Dict[str, Any]]:
        """
        Search for full-text news articles.

        Args:
            query: Search query
            country_code: Country filter (ISO code)
            lang_code: Language filter (ISO code)
            count: Number of results
            sort: Sort order ('latest' or 'relevance')

        Returns:
            List of full-text news articles
        """
        params = {
            "query_text": query,
            "count": count,
            "sort": sort
        }

        if country_code:
            params["country_code"] = country_code
        if lang_code:
            params["lang_code"] = lang_code

        result = await self.call_method("news_fulltext", params)
        return result.get("items", [])

    async def search_with_metadata(
        self,
        query: str,
        country_code: str = None,
        lang_code: str = None,
        count: int = 5,
        sort: str = "relevance"
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles with enriched metadata.

        Args:
            query: Search query
            country_code: Country filter (ISO code)
            lang_code: Language filter (ISO code)
            count: Number of results
            sort: Sort order ('latest' or 'relevance')

        Returns:
            List of news articles with metadata
        """
        params = {
            "query_text": query,
            "count": count,
            "sort": sort
        }

        if country_code:
            params["country_code"] = country_code
        if lang_code:
            params["lang_code"] = lang_code

        result = await self.call_method("news_fulltext_metadata", params)
        return result.get("items", [])

    async def health_check(self) -> bool:
        """
        Check if the News MCP Server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(**self.client_config) as client:
                response = await client.get(f"{self.server_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass