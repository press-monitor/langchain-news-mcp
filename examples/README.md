# LangChain News MCP Integration Examples

LangChain-specific examples for news intelligence and AI agent integration.

## 📁 Examples

### 🤖 Agent Examples
- `research_agent.py` - Autonomous news research agent
- `market_analyst.py` - Market intelligence agent
- `brand_monitor.py` - Brand reputation monitoring agent
- `compliance_agent.py` - Regulatory compliance monitoring

### 🔧 Tool Examples
- `news_tools.py` - Custom LangChain tools
- `retriever_examples.py` - News document retrievers
- `memory_integration.py` - Conversation memory with news context

### 🔗 Chain Examples
- `news_qa_chain.py` - Question-answering with news data
- `summarization_chain.py` - News summarization pipeline
- `analysis_chain.py` - Multi-step news analysis

### 📊 Advanced Use Cases
- `multi_agent_crew.py` - Coordinated multi-agent workflows
- `real_time_monitor.py` - Real-time news monitoring system
- `news_chatbot.py` - Conversational news assistant

### 🌐 Integration Patterns
- `async_patterns.py` - Async/await usage patterns
- `error_handling.py` - Robust error handling
- `caching_strategies.py` - Performance optimization

## 🚀 Quick Start

```bash
# Install dependencies
pip install langchain-news-mcp langchain openai

# Set up environment
export OPENAI_API_KEY="your_openai_key"
export NEWS_MCP_SERVER_URL="http://localhost:3000"

# Run basic example
python research_agent.py "artificial intelligence trends"
```

## 🔧 Basic Integration

```python
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain_news_mcp import NewsTool

# Create news tool
news_tool = NewsTool(server_url="http://localhost:3000")

# Create agent
agent = initialize_agent(
    tools=[news_tool],
    llm=OpenAI(),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Use the agent
result = agent.run("Find recent AI news and summarize key trends")
```