#!/usr/bin/env python3
"""
Setup script for LangChain News MCP Integration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from __init__.py
def get_version():
    with open("langchain_news_mcp/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="langchain-news-mcp",
    version=get_version(),
    description="LangChain integration for News MCP Server - Real-time global news intelligence for AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="News MCP Team",
    author_email="support@pressmonitor.com",
    url="https://github.com/yourorg/langchain-news-mcp",
    project_urls={
        "Documentation": "https://www.pressmonitor.com/en/docs/online-news-api",
        "API Schema": "https://api.pressmonitor.com/schemas/newsv1",
        "Homepage": "https://www.pressmonitor.com/",
        "Source": "https://github.com/yourorg/langchain-news-mcp",
        "Tracker": "https://github.com/yourorg/langchain-news-mcp/issues",
    },
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.0.350",
        "httpx>=0.25.0",
        "pydantic>=2.4.0",
        "aiofiles>=23.0.0",
        "tenacity>=8.2.0",
    ],
    extras_require={
        "server": [
            "news-mcp-server[full]>=1.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "all": [
            "news-mcp-server[full]>=1.0.0",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "flake8>=6.0.0",
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content :: News/Diary",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords=[
        "langchain", "news", "mcp", "ai", "agent", "nlp", "media", "intelligence",
        "real-time", "global", "multilingual", "search", "analysis", "llm"
    ],
    entry_points={
        "langchain.tools": [
            "news_tool = langchain_news_mcp.tools:NewsTool",
            "news_search_tool = langchain_news_mcp.tools:NewsSearchTool",
            "news_headlines_tool = langchain_news_mcp.tools:NewsHeadlinesTool",
            "news_briefs_tool = langchain_news_mcp.tools:NewsBriefsTool",
            "news_fulltext_tool = langchain_news_mcp.tools:NewsFullTextTool",
        ],
        "langchain.retrievers": [
            "news_retriever = langchain_news_mcp.retrievers:NewsRetriever",
            "news_vector_retriever = langchain_news_mcp.retrievers:NewsVectorRetriever",
            "news_time_retriever = langchain_news_mcp.retrievers:NewsTimeRetriever",
        ],
        "langchain.document_loaders": [
            "news_loader = langchain_news_mcp.loaders:NewsLoader",
            "news_batch_loader = langchain_news_mcp.loaders:NewsBatchLoader",
            "news_stream_loader = langchain_news_mcp.loaders:NewsStreamLoader",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)