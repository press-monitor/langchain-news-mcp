"""
LangChain Tools for News MCP Integration

Provides LangChain-compatible tools for news search and analysis.
"""

from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

from .client import NewsMCPClient
from .exceptions import NewsMCPError


class NewsSearchInput(BaseModel):
    """Input schema for news search tools."""
    query: str = Field(description="Search query for news articles")
    news_type: str = Field(
        default="headlines",
        description="Type of news content: 'headlines', 'briefs', or 'fulltext'"
    )
    country_code: Optional[str] = Field(
        default=None,
        description="Country filter (ISO code like 'US', 'UK', 'DE')"
    )
    lang_code: Optional[str] = Field(
        default=None,
        description="Language filter (ISO code like 'en', 'es', 'fr')"
    )
    count: int = Field(
        default=10,
        description="Number of results to return (1-100)"
    )
    sort: str = Field(
        default="latest",
        description="Sort order: 'latest' for newest first, 'relevance' for most relevant"
    )


class NewsTool(BaseTool):
    """
    Main LangChain tool for news search with full parameter control.
    """

    name: str = "news_search"
    description: str = """
    Search for current news articles from global sources. Provide a clear search query.
    You can specify:
    - news_type: 'headlines' for quick updates, 'briefs' for summaries, 'fulltext' for complete articles
    - country_code: ISO codes like 'US', 'UK', 'DE' for regional focus
    - lang_code: ISO codes like 'en', 'es', 'fr' for language filter
    - count: number of results (1-100)
    - sort: 'latest' for newest, 'relevance' for most relevant

    Always analyze the results thoroughly and extract key insights.
    """
    args_schema: Type[BaseModel] = NewsSearchInput

    server_url: str
    api_key: Optional[str]
    timeout: float
    default_count: int
    default_sort: str

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        default_count: int = 10,
        default_sort: str = "latest",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.server_url = server_url
        self.api_key = api_key
        self.timeout = timeout
        self.default_count = default_count
        self.default_sort = default_sort

    def _get_client(self) -> NewsMCPClient:
        """Get configured News MCP client."""
        return NewsMCPClient(
            server_url=self.server_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

    def _format_results(self, articles: List[Dict[str, Any]], news_type: str) -> str:
        """Format news results for LLM consumption."""
        if not articles:
            return "No news articles found for the given query."

        formatted_results = []
        formatted_results.append(f"Found {len(articles)} news articles:\n")

        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown source')
            url = article.get('url', '')
            timestamp = article.get('ts', 0)

            result_text = f"{i}. **{title}**\n"
            result_text += f"   Source: {source}\n"

            if news_type in ['briefs', 'fulltext']:
                description = article.get('description', '')
                if description:
                    # Truncate description for briefs
                    max_desc_length = 200 if news_type == 'briefs' else 500
                    if len(description) > max_desc_length:
                        description = description[:max_desc_length] + "..."
                    result_text += f"   Summary: {description}\n"

            if news_type == 'fulltext':
                content = article.get('content', '')
                if content:
                    # Provide content preview for fulltext
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    result_text += f"   Content: {content}\n"

            if url:
                result_text += f"   URL: {url}\n"

            # Add metadata if available
            if 'entities' in article and article['entities']:
                entities = article['entities'][:5]  # Show first 5 entities
                result_text += f"   Key entities: {', '.join(entities)}\n"

            result_text += "\n"
            formatted_results.append(result_text)

        return "".join(formatted_results)

    def _run(
        self,
        query: str,
        news_type: str = "headlines",
        country_code: Optional[str] = None,
        lang_code: Optional[str] = None,
        count: int = None,
        sort: str = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute news search synchronously."""
        try:
            client = self._get_client()

            # Use defaults if not specified
            count = count or self.default_count
            sort = sort or self.default_sort

            # Validate inputs
            if count < 1 or count > 100:
                return "Error: count must be between 1 and 100"

            if news_type not in ['headlines', 'briefs', 'fulltext']:
                return "Error: news_type must be 'headlines', 'briefs', or 'fulltext'"

            if sort not in ['latest', 'relevance']:
                return "Error: sort must be 'latest' or 'relevance'"

            # Call appropriate method based on news_type
            if news_type == "headlines":
                result = client.call_method_sync("news_headlines", {
                    "query_text": query,
                    "count": count,
                    "sort": sort,
                    **({"country_code": country_code} if country_code else {}),
                    **({"lang_code": lang_code} if lang_code else {}),
                })
            elif news_type == "briefs":
                result = client.call_method_sync("news_briefs", {
                    "query_text": query,
                    "count": count,
                    "sort": sort,
                    **({"country_code": country_code} if country_code else {}),
                    **({"lang_code": lang_code} if lang_code else {}),
                })
            else:  # fulltext
                result = client.call_method_sync("news_fulltext", {
                    "query_text": query,
                    "count": count,
                    "sort": sort,
                    **({"country_code": country_code} if country_code else {}),
                    **({"lang_code": lang_code} if lang_code else {}),
                })

            articles = result.get("items", [])
            return self._format_results(articles, news_type)

        except NewsMCPError as e:
            return f"News search error: {str(e)}"
        except Exception as e:
            return f"Unexpected error during news search: {str(e)}"

    async def _arun(
        self,
        query: str,
        news_type: str = "headlines",
        country_code: Optional[str] = None,
        lang_code: Optional[str] = None,
        count: int = None,
        sort: str = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute news search asynchronously."""
        try:
            client = self._get_client()

            # Use defaults if not specified
            count = count or self.default_count
            sort = sort or self.default_sort

            # Validate inputs
            if count < 1 or count > 100:
                return "Error: count must be between 1 and 100"

            if news_type not in ['headlines', 'briefs', 'fulltext']:
                return "Error: news_type must be 'headlines', 'briefs', or 'fulltext'"

            if sort not in ['latest', 'relevance']:
                return "Error: sort must be 'latest' or 'relevance'"

            # Call appropriate method based on news_type
            if news_type == "headlines":
                articles = await client.search_headlines(
                    query=query,
                    country_code=country_code,
                    lang_code=lang_code,
                    count=count,
                    sort=sort
                )
            elif news_type == "briefs":
                articles = await client.search_briefs(
                    query=query,
                    country_code=country_code,
                    lang_code=lang_code,
                    count=count,
                    sort=sort
                )
            else:  # fulltext
                articles = await client.search_fulltext(
                    query=query,
                    country_code=country_code,
                    lang_code=lang_code,
                    count=count,
                    sort=sort
                )

            return self._format_results(articles, news_type)

        except NewsMCPError as e:
            return f"News search error: {str(e)}"
        except Exception as e:
            return f"Unexpected error during news search: {str(e)}"


class NewsSearchInput(BaseModel):
    """Simplified input schema for basic news search."""
    query: str = Field(description="Search query for news articles")
    count: int = Field(default=10, description="Number of results (1-50)")


class NewsSearchTool(BaseTool):
    """Simplified news search tool with basic parameters."""

    name: str = "news_search_simple"
    description: str = """
    Search for current news headlines. Provide a search query and optionally specify
    the number of results (1-50). Returns recent news articles with titles, sources, and URLs.
    """
    args_schema: Type[BaseModel] = NewsSearchInput

    server_url: str
    api_key: Optional[str]

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.server_url = server_url
        self.api_key = api_key

    def _run(
        self,
        query: str,
        count: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute simple news search."""
        tool = NewsTool(server_url=self.server_url, api_key=self.api_key)
        return tool._run(query=query, count=min(count, 50), news_type="headlines")

    async def _arun(
        self,
        query: str,
        count: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute simple news search asynchronously."""
        tool = NewsTool(server_url=self.server_url, api_key=self.api_key)
        return await tool._arun(query=query, count=min(count, 50), news_type="headlines")


class NewsHeadlinesTool(NewsTool):
    """Specialized tool for news headlines only."""

    name: str = "news_headlines"
    description: str = """
    Search for news headlines. Fast and efficient for getting quick news updates.
    Returns titles, sources, and URLs without full content.
    """

    def _run(self, query: str, **kwargs) -> str:
        return super()._run(query=query, news_type="headlines", **kwargs)

    async def _arun(self, query: str, **kwargs) -> str:
        return await super()._arun(query=query, news_type="headlines", **kwargs)


class NewsBriefsTool(NewsTool):
    """Specialized tool for news briefs with summaries."""

    name: str = "news_briefs"
    description: str = """
    Search for news briefs with summaries. Provides headlines plus short descriptions
    of each article for better understanding of the news content.
    """

    def _run(self, query: str, **kwargs) -> str:
        return super()._run(query=query, news_type="briefs", **kwargs)

    async def _arun(self, query: str, **kwargs) -> str:
        return await super()._arun(query=query, news_type="briefs", **kwargs)


class NewsFullTextTool(NewsTool):
    """Specialized tool for full-text news articles."""

    name: str = "news_fulltext"
    description: str = """
    Search for full-text news articles. Provides complete article content for
    in-depth analysis and research. Best for detailed investigation of topics.
    """

    def _run(self, query: str, **kwargs) -> str:
        kwargs.setdefault('count', 5)  # Default to fewer results for fulltext
        return super()._run(query=query, news_type="fulltext", **kwargs)

    async def _arun(self, query: str, **kwargs) -> str:
        kwargs.setdefault('count', 5)  # Default to fewer results for fulltext
        return await super()._arun(query=query, news_type="fulltext", **kwargs)