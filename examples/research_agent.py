#!/usr/bin/env python3
"""
LangChain News Research Agent Example

This example demonstrates creating an autonomous research agent that can:
- Search for news articles on any topic
- Analyze trends and patterns
- Generate comprehensive reports
- Remember conversation context
"""

import os
import sys
from typing import List, Dict, Any

from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain_news_mcp import NewsTool


class NewsResearchAgent:
    """
    Autonomous news research agent powered by LangChain and News MCP.
    """

    def __init__(
        self,
        news_server_url: str = "http://localhost:3000",
        news_api_key: str = None,
        openai_api_key: str = None,
        temperature: float = 0.1,
        verbose: bool = True
    ):
        """Initialize the research agent."""
        self.news_server_url = news_server_url
        self.news_api_key = news_api_key
        self.verbose = verbose

        # Initialize the news tool
        self.news_tool = NewsTool(
            server_url=news_server_url,
            api_key=news_api_key,
            timeout=30.0,
            default_count=15,
            default_sort="relevance"
        )

        # Customize tool description for better agent performance
        self.news_tool.description = """
        Use this tool to search for current news articles. Provide a clear search query.
        You can specify:
        - news_type: 'headlines' for quick updates, 'briefs' for summaries, 'fulltext' for complete articles
        - country_code: ISO codes like 'US', 'UK', 'DE' for regional focus
        - count: number of results (1-50 recommended)
        - sort: 'latest' for newest, 'relevance' for most relevant

        Always analyze the results thoroughly and extract key insights.
        """

        # Initialize LLM
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=temperature,
            max_tokens=2000
        )

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        # Create the agent
        self.agent = initialize_agent(
            tools=[self.news_tool],
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=verbose,
            max_iterations=5,
            handle_parsing_errors=True
        )

    def research_topic(self, topic: str, depth: str = "comprehensive") -> str:
        """
        Research a topic with specified depth.

        Args:
            topic: The topic to research
            depth: 'summary', 'detailed', or 'comprehensive'
        """
        prompts = {
            "summary": f"""
            Research the topic "{topic}" and provide a brief summary of the current situation.
            Include:
            1. Key recent developments
            2. Main stakeholders involved
            3. Current status
            """,

            "detailed": f"""
            Conduct detailed research on "{topic}" and provide an analytical report.
            Include:
            1. Recent developments and timeline
            2. Key players and stakeholders
            3. Different perspectives and viewpoints
            4. Potential implications and impact
            5. Expert opinions and analysis
            """,

            "comprehensive": f"""
            Perform comprehensive research on "{topic}" and create a thorough intelligence report.
            Structure your analysis with:

            1. EXECUTIVE SUMMARY
               - Key findings and developments
               - Strategic implications

            2. CURRENT SITUATION
               - Latest news and developments
               - Timeline of recent events

            3. STAKEHOLDER ANALYSIS
               - Key players and their positions
               - Institutional responses

            4. MARKET/SECTOR IMPACT
               - Economic implications
               - Industry effects

            5. FUTURE OUTLOOK
               - Trend analysis
               - Potential scenarios
               - Expert predictions

            6. RECOMMENDATIONS
               - Actionable insights
               - Areas to monitor

            Use multiple news searches to gather comprehensive information from different angles.
            """
        }

        prompt = prompts.get(depth, prompts["comprehensive"])
        return self.agent.run(prompt)

    def compare_topics(self, topic1: str, topic2: str) -> str:
        """Compare two topics based on recent news coverage."""
        prompt = f"""
        Compare recent news coverage of "{topic1}" versus "{topic2}".

        For each topic, analyze:
        1. Volume of coverage
        2. Sentiment and tone
        3. Key themes and narratives
        4. Geographic focus
        5. Stakeholder perspectives

        Then provide:
        - Similarities in coverage
        - Key differences
        - Relative importance/priority
        - Interconnections between topics

        Use separate news searches for each topic to ensure comprehensive comparison.
        """
        return self.agent.run(prompt)

    def monitor_developing_story(self, topic: str) -> str:
        """Monitor a developing story for latest updates."""
        prompt = f"""
        Monitor the developing story about "{topic}" for the latest updates.

        Focus on:
        1. Most recent developments (last 24-48 hours)
        2. Breaking news and updates
        3. Official statements and responses
        4. Expert commentary and analysis
        5. Next expected developments

        Provide a timeline of recent events and highlight what's new or changed.
        """
        return self.agent.run(prompt)

    def analyze_sentiment(self, topic: str) -> str:
        """Analyze sentiment and public opinion on a topic."""
        prompt = f"""
        Analyze the sentiment and public opinion regarding "{topic}" based on recent news coverage.

        Examine:
        1. Overall sentiment (positive, negative, neutral)
        2. Regional variations in coverage
        3. Source bias and perspectives
        4. Public vs. institutional opinions
        5. Sentiment trends over time

        Provide insights into:
        - Why sentiment is trending in a particular direction
        - Key factors influencing public opinion
        - Potential reputation implications
        """
        return self.agent.run(prompt)

    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the conversation history."""
        return self.memory.chat_memory.messages


def main():
    """Main function to demonstrate the research agent."""
    print("🔍 News Research Agent - LangChain Integration")
    print("=" * 60)

    # Get configuration from environment
    news_server_url = os.getenv("NEWS_MCP_SERVER_URL", "http://localhost:3000")
    news_api_key = os.getenv("NEWS_MCP_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    # Initialize the agent
    try:
        agent = NewsResearchAgent(
            news_server_url=news_server_url,
            news_api_key=news_api_key,
            openai_api_key=openai_api_key,
            verbose=True
        )
        print("✅ Research agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        sys.exit(1)

    # Interactive mode or command line argument
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
        print(f"\n🔍 Researching topic: {topic}")
        print("-" * 60)

        try:
            result = agent.research_topic(topic, depth="comprehensive")
            print(result)
        except Exception as e:
            print(f"❌ Research failed: {e}")
    else:
        # Interactive mode
        print("\n💬 Interactive mode - Enter topics to research (type 'quit' to exit)")
        print("Commands:")
        print("  research <topic> - Comprehensive research")
        print("  summary <topic> - Brief summary")
        print("  compare <topic1> vs <topic2> - Compare topics")
        print("  monitor <topic> - Monitor developing story")
        print("  sentiment <topic> - Analyze sentiment")
        print("  history - Show conversation history")
        print()

        while True:
            try:
                user_input = input("🔍 Enter command: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input:
                    continue

                parts = user_input.split()
                command = parts[0].lower()

                if command == "research" and len(parts) > 1:
                    topic = " ".join(parts[1:])
                    result = agent.research_topic(topic, depth="comprehensive")
                    print(f"\n📊 Research Results:\n{result}\n")

                elif command == "summary" and len(parts) > 1:
                    topic = " ".join(parts[1:])
                    result = agent.research_topic(topic, depth="summary")
                    print(f"\n📋 Summary:\n{result}\n")

                elif command == "compare" and len(parts) > 3 and parts[2].lower() == "vs":
                    topic1 = parts[1]
                    topic2 = " ".join(parts[3:])
                    result = agent.compare_topics(topic1, topic2)
                    print(f"\n⚖️ Comparison Results:\n{result}\n")

                elif command == "monitor" and len(parts) > 1:
                    topic = " ".join(parts[1:])
                    result = agent.monitor_developing_story(topic)
                    print(f"\n📡 Monitoring Results:\n{result}\n")

                elif command == "sentiment" and len(parts) > 1:
                    topic = " ".join(parts[1:])
                    result = agent.analyze_sentiment(topic)
                    print(f"\n😊 Sentiment Analysis:\n{result}\n")

                elif command == "history":
                    history = agent.get_conversation_history()
                    print(f"\n💭 Conversation History ({len(history)} messages):")
                    for i, msg in enumerate(history[-6:], 1):  # Show last 6 messages
                        role = "🤖" if msg.type == "ai" else "👤"
                        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        print(f"  {i}. {role} {content}")
                    print()

                else:
                    print("❌ Invalid command. Use: research, summary, compare, monitor, sentiment, or history")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

    print("\n✅ Research session completed!")


if __name__ == "__main__":
    main()