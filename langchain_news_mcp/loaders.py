"""
LangChain Document Loaders for News MCP Integration

Provides LangChain-compatible document loaders for news data.
"""

import asyncio
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TextSplitter

from .client import NewsMCPClient
from .exceptions import NewsMCPError


class NewsLoader(BaseLoader):
    """
    LangChain document loader for news articles.
    Converts news articles into LangChain documents for processing.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
        api_key: Optional[str] = None,
        news_type: str = "briefs",
        timeout: float = 30.0,
    ):
        """
        Initialize the news loader.

        Args:
            server_url: News MCP Server URL
            api_key: API key for authentication
            news_type: Type of news content ('headlines', 'briefs', 'fulltext')
            timeout: Request timeout in seconds
        """
        self.server_url = server_url
        self.api_key = api_key
        self.news_type = news_type
        self.timeout = timeout

    def _get_client(self) -> NewsMCPClient:
        """Get configured News MCP client."""
        return NewsMCPClient(
            server_url=self.server_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

    def _article_to_document(self, article: Dict[str, Any]) -> Document:
        """Convert a news article to a LangChain document."""
        # Determine content based on news type
        if self.news_type == "headlines":
            content = article.get("title", "")
        elif self.news_type == "briefs":
            content = article.get("description", article.get("title", ""))
        elif self.news_type == "fulltext":
            content = article.get("content", article.get("description", article.get("title", "")))
        else:
            content = article.get("title", "")

        # Create comprehensive metadata
        metadata = {
            "source": article.get("source", ""),
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "timestamp": article.get("ts", 0),
            "news_type": self.news_type,
            "id": article.get("id", ""),
        }

        # Add optional metadata
        optional_fields = [
            "country", "language", "entities", "sentiment", "subjects",
            "description", "image", "author", "published_date"
        ]

        for field in optional_fields:
            if field in article and article[field]:
                metadata[field] = article[field]

        return Document(page_content=content, metadata=metadata)

    def load(self) -> List[Document]:
        """
        Load documents. This method should be called after setting search parameters.
        Use load_and_split() for more convenient usage.
        """
        raise NotImplementedError(
            "Use load_and_split() method with query parameter instead"
        )

    def load_and_split(
        self,
        query: str,
        country_code: Optional[str] = None,
        lang_code: Optional[str] = None,
        count: int = 50,
        sort: str = "relevance",
        text_splitter: Optional[TextSplitter] = None,
    ) -> List[Document]:
        """
        Load and optionally split news documents.

        Args:
            query: Search query
            country_code: Country filter
            lang_code: Language filter
            count: Number of articles to fetch
            sort: Sort order
            text_splitter: Text splitter for chunking documents

        Returns:
            List of documents
        """
        try:
            client = self._get_client()

            # Prepare search parameters
            params = {
                "query_text": query,
                "count": count,
                "sort": sort,
            }

            if country_code:
                params["country_code"] = country_code
            if lang_code:
                params["lang_code"] = lang_code

            # Call appropriate method based on news type
            if self.news_type == "headlines":
                result = client.call_method_sync("news_headlines", params)
            elif self.news_type == "briefs":
                result = client.call_method_sync("news_briefs", params)
            elif self.news_type == "fulltext":
                result = client.call_method_sync("news_fulltext", params)
            else:
                raise ValueError(f"Invalid news_type: {self.news_type}")

            articles = result.get("items", [])

            # Convert articles to documents
            documents = [self._article_to_document(article) for article in articles]

            # Apply text splitter if provided
            if text_splitter:
                split_documents = []
                for doc in documents:
                    splits = text_splitter.split_documents([doc])
                    split_documents.extend(splits)
                return split_documents

            return documents

        except NewsMCPError as e:
            raise RuntimeError(f"News loading error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during news loading: {str(e)}")

    async def aload_and_split(
        self,
        query: str,
        country_code: Optional[str] = None,
        lang_code: Optional[str] = None,
        count: int = 50,
        sort: str = "relevance",
        text_splitter: Optional[TextSplitter] = None,
    ) -> List[Document]:
        """
        Asynchronously load and optionally split news documents.

        Args:
            query: Search query
            country_code: Country filter
            lang_code: Language filter
            count: Number of articles to fetch
            sort: Sort order
            text_splitter: Text splitter for chunking documents

        Returns:
            List of documents
        """
        try:
            client = self._get_client()

            # Call appropriate method based on news type
            if self.news_type == "headlines":
                articles = await client.search_headlines(
                    query=query,
                    country_code=country_code,
                    lang_code=lang_code,
                    count=count,
                    sort=sort
                )
            elif self.news_type == "briefs":
                articles = await client.search_briefs(
                    query=query,
                    country_code=country_code,
                    lang_code=lang_code,
                    count=count,
                    sort=sort
                )
            elif self.news_type == "fulltext":
                articles = await client.search_fulltext(
                    query=query,
                    country_code=country_code,
                    lang_code=lang_code,
                    count=count,
                    sort=sort
                )
            else:
                raise ValueError(f"Invalid news_type: {self.news_type}")

            # Convert articles to documents
            documents = [self._article_to_document(article) for article in articles]

            # Apply text splitter if provided
            if text_splitter:
                split_documents = []
                for doc in documents:
                    splits = text_splitter.split_documents([doc])
                    split_documents.extend(splits)
                return split_documents

            return documents

        except NewsMCPError as e:
            raise RuntimeError(f"News loading error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during news loading: {str(e)}")


class NewsBatchLoader(NewsLoader):
    """
    Batch loader for efficiently processing multiple news queries.
    """

    def __init__(
        self,
        batch_size: int = 100,
        concurrent_requests: int = 5,
        **kwargs
    ):
        """
        Initialize the batch news loader.

        Args:
            batch_size: Maximum articles per batch
            concurrent_requests: Number of concurrent requests
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.concurrent_requests = concurrent_requests

    async def aload_batch(
        self,
        queries: List[str],
        country_code: Optional[str] = None,
        lang_code: Optional[str] = None,
        sort: str = "relevance",
        text_splitter: Optional[TextSplitter] = None,
    ) -> List[Document]:
        """
        Load documents for multiple queries in batches.

        Args:
            queries: List of search queries
            country_code: Country filter
            lang_code: Language filter
            sort: Sort order
            text_splitter: Text splitter for chunking

        Returns:
            Combined list of documents from all queries
        """
        # Calculate articles per query to stay within batch size
        articles_per_query = max(1, self.batch_size // len(queries))

        # Create semaphore for concurrent request limiting
        semaphore = asyncio.Semaphore(self.concurrent_requests)

        async def load_single_query(query: str) -> List[Document]:
            async with semaphore:
                return await self.aload_and_split(
                    query=query,
                    country_code=country_code,
                    lang_code=lang_code,
                    count=articles_per_query,
                    sort=sort,
                    text_splitter=text_splitter,
                )

        # Execute all queries concurrently
        tasks = [load_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results and handle exceptions
        all_documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Warning: Query '{queries[i]}' failed: {result}")
            else:
                all_documents.extend(result)

        return all_documents

    def load_batch(
        self,
        queries: List[str],
        country_code: Optional[str] = None,
        lang_code: Optional[str] = None,
        sort: str = "relevance",
        text_splitter: Optional[TextSplitter] = None,
    ) -> List[Document]:
        """
        Synchronous wrapper for batch loading.

        Args:
            queries: List of search queries
            country_code: Country filter
            lang_code: Language filter
            sort: Sort order
            text_splitter: Text splitter for chunking

        Returns:
            Combined list of documents from all queries
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.aload_batch(queries, country_code, lang_code, sort, text_splitter)
        )


class NewsStreamLoader(NewsLoader):
    """
    Streaming loader for real-time news processing.
    """

    def __init__(
        self,
        refresh_interval: int = 300,  # 5 minutes
        **kwargs
    ):
        """
        Initialize the streaming news loader.

        Args:
            refresh_interval: Seconds between refresh cycles
        """
        super().__init__(**kwargs)
        self.refresh_interval = refresh_interval

    async def astream_documents(
        self,
        query: str,
        country_code: Optional[str] = None,
        lang_code: Optional[str] = None,
        count: int = 20,
        text_splitter: Optional[TextSplitter] = None,
    ) -> AsyncIterator[Document]:
        """
        Stream documents with periodic refresh.

        Args:
            query: Search query
            country_code: Country filter
            lang_code: Language filter
            count: Number of articles per refresh
            text_splitter: Text splitter for chunking

        Yields:
            Documents as they become available
        """
        seen_urls = set()

        while True:
            try:
                # Load fresh documents
                documents = await self.aload_and_split(
                    query=query,
                    country_code=country_code,
                    lang_code=lang_code,
                    count=count,
                    sort="latest",  # Always use latest for streaming
                    text_splitter=text_splitter,
                )

                # Yield only new documents
                for doc in documents:
                    url = doc.metadata.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        yield doc

                # Wait before next refresh
                await asyncio.sleep(self.refresh_interval)

            except Exception as e:
                print(f"Streaming error: {e}")
                await asyncio.sleep(self.refresh_interval)

    def stream_documents(
        self,
        query: str,
        country_code: Optional[str] = None,
        lang_code: Optional[str] = None,
        count: int = 20,
        text_splitter: Optional[TextSplitter] = None,
    ) -> Iterator[Document]:
        """
        Synchronous streaming wrapper.

        Args:
            query: Search query
            country_code: Country filter
            lang_code: Language filter
            count: Number of articles per refresh
            text_splitter: Text splitter for chunking

        Yields:
            Documents as they become available
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async_generator = self.astream_documents(
            query, country_code, lang_code, count, text_splitter
        )

        while True:
            try:
                document = loop.run_until_complete(async_generator.__anext__())
                yield document
            except StopAsyncIteration:
                break