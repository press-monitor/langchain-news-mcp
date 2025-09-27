"""
LangChain Retrievers for News MCP Integration

Provides LangChain-compatible retrievers for news document retrieval.
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores.base import VectorStore
from langchain.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun

from .client import NewsMCPClient
from .exceptions import NewsMCPError


class NewsRetriever(BaseRetriever):
    """
    LangChain retriever for news documents.
    Perfect for Retrieval-Augmented Generation (RAG) applications.
    """

    server_url: str
    api_key: Optional[str]
    search_kwargs: Dict[str, Any]
    news_type: str
    content_field: str

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
        api_key: Optional[str] = None,
        news_type: str = "briefs",
        content_field: str = "description",
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the news retriever.

        Args:
            server_url: News MCP Server URL
            api_key: API key for authentication
            news_type: Type of news content ('headlines', 'briefs', 'fulltext')
            content_field: Field to use as document content
            search_kwargs: Additional search parameters
        """
        super().__init__(**kwargs)
        self.server_url = server_url
        self.api_key = api_key
        self.news_type = news_type
        self.content_field = content_field
        self.search_kwargs = search_kwargs or {}

        # Set appropriate content field based on news type
        if news_type == "headlines":
            self.content_field = "title"
        elif news_type == "briefs":
            self.content_field = "description"
        elif news_type == "fulltext":
            self.content_field = "content"

    def _get_client(self) -> NewsMCPClient:
        """Get configured News MCP client."""
        return NewsMCPClient(
            server_url=self.server_url,
            api_key=self.api_key
        )

    def _articles_to_documents(self, articles: List[Dict[str, Any]]) -> List[Document]:
        """Convert news articles to LangChain documents."""
        documents = []

        for article in articles:
            # Get content based on news type
            content = article.get(self.content_field, "")
            if not content and self.content_field == "description":
                # Fallback to title if no description
                content = article.get("title", "")

            # Create metadata
            metadata = {
                "source": article.get("source", ""),
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "timestamp": article.get("ts", 0),
                "news_type": self.news_type,
            }

            # Add additional metadata if available
            if "country" in article:
                metadata["country"] = article["country"]
            if "language" in article:
                metadata["language"] = article["language"]
            if "entities" in article:
                metadata["entities"] = article["entities"]
            if "sentiment" in article:
                metadata["sentiment"] = article["sentiment"]

            # Create document
            document = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(document)

        return documents

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant documents synchronously."""
        try:
            client = self._get_client()

            # Prepare search parameters
            params = {
                "query_text": query,
                "count": self.search_kwargs.get("count", 10),
                "sort": self.search_kwargs.get("sort", "relevance"),
            }

            # Add optional parameters
            if "country_code" in self.search_kwargs:
                params["country_code"] = self.search_kwargs["country_code"]
            if "lang_code" in self.search_kwargs:
                params["lang_code"] = self.search_kwargs["lang_code"]

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
            return self._articles_to_documents(articles)

        except NewsMCPError as e:
            # Return empty list on error to prevent retrieval chain failure
            run_manager.on_retriever_error(e)
            return []
        except Exception as e:
            run_manager.on_retriever_error(e)
            return []

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant documents asynchronously."""
        try:
            client = self._get_client()

            # Prepare search parameters
            count = self.search_kwargs.get("count", 10)
            sort = self.search_kwargs.get("sort", "relevance")
            country_code = self.search_kwargs.get("country_code")
            lang_code = self.search_kwargs.get("lang_code")

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

            return self._articles_to_documents(articles)

        except NewsMCPError as e:
            # Return empty list on error to prevent retrieval chain failure
            await run_manager.on_retriever_error(e)
            return []
        except Exception as e:
            await run_manager.on_retriever_error(e)
            return []


class NewsVectorRetriever(BaseRetriever):
    """
    Vector-based news retriever using embeddings for semantic search.
    """

    server_url: str
    api_key: Optional[str]
    vectorstore: VectorStore
    search_kwargs: Dict[str, Any]
    news_type: str

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
        api_key: Optional[str] = None,
        vectorstore: VectorStore = None,
        news_type: str = "briefs",
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the vector news retriever.

        Args:
            server_url: News MCP Server URL
            api_key: API key for authentication
            vectorstore: Vector store for similarity search
            news_type: Type of news content
            search_kwargs: Search parameters for vector store
        """
        super().__init__(**kwargs)
        self.server_url = server_url
        self.api_key = api_key
        self.vectorstore = vectorstore
        self.news_type = news_type
        self.search_kwargs = search_kwargs or {"k": 10}

        if not vectorstore:
            raise ValueError("vectorstore is required for NewsVectorRetriever")

    async def _populate_vectorstore(self, query: str) -> None:
        """Populate vector store with fresh news data."""
        # Get fresh news data
        news_retriever = NewsRetriever(
            server_url=self.server_url,
            api_key=self.api_key,
            news_type=self.news_type,
            search_kwargs={"count": 50, "sort": "relevance"}
        )

        documents = await news_retriever._aget_relevant_documents(
            query,
            run_manager=AsyncCallbackManagerForRetrieverRun.get_noop_manager()
        )

        if documents:
            # Add documents to vector store
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            await self.vectorstore.aadd_texts(texts, metadatas)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant documents using vector similarity."""
        try:
            # Search vector store
            documents = self.vectorstore.similarity_search(
                query, **self.search_kwargs
            )
            return documents

        except Exception as e:
            run_manager.on_retriever_error(e)
            return []

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant documents using vector similarity asynchronously."""
        try:
            # Optionally populate with fresh data
            if self.search_kwargs.get("refresh_data", False):
                await self._populate_vectorstore(query)

            # Search vector store
            documents = await self.vectorstore.asimilarity_search(
                query, **self.search_kwargs
            )
            return documents

        except Exception as e:
            await run_manager.on_retriever_error(e)
            return []


class NewsTimeRetriever(NewsRetriever):
    """
    Time-based news retriever that focuses on temporal aspects of news.
    """

    time_weight: float
    recency_boost: bool

    def __init__(
        self,
        time_weight: float = 0.5,
        recency_boost: bool = True,
        **kwargs
    ):
        """
        Initialize the time-based news retriever.

        Args:
            time_weight: Weight for temporal relevance (0.0 to 1.0)
            recency_boost: Whether to boost recent articles
        """
        super().__init__(**kwargs)
        self.time_weight = time_weight
        self.recency_boost = recency_boost

        # Force sort by latest for time-based retrieval
        self.search_kwargs["sort"] = "latest"

    def _articles_to_documents(self, articles: List[Dict[str, Any]]) -> List[Document]:
        """Convert articles to documents with time-based scoring."""
        documents = super()._articles_to_documents(articles)

        if self.recency_boost:
            # Add recency score to metadata
            import time
            current_time = time.time()

            for i, document in enumerate(documents):
                article_time = document.metadata.get("timestamp", 0)
                if article_time:
                    # Calculate recency score (newer = higher score)
                    time_diff = current_time - article_time
                    hours_old = time_diff / 3600

                    # Recency score: 1.0 for articles < 1 hour old, declining exponentially
                    recency_score = max(0.1, 1.0 / (1 + hours_old / 24))
                    document.metadata["recency_score"] = recency_score
                    document.metadata["hours_old"] = hours_old

        return documents