from dotenv import load_dotenv
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

@tool
def research_topic(topic: str) -> str:
    """Research a topic and provide a structured explanation."""

    return f"""
Research the following topic:

{topic}

Provide:
1. A simple overview
2. Important points
3. Real-world applications
4. Challenges or limitations
5. A short conclusion

Keep the explanation clear and structured.
"""

agent = create_agent(
    tools=[research_topic],
    model=llm,
    system_prompt="""
You are ResearchBot.

Use the research tool to provide structured, informative, and easy-to-understand research summaries.

Keep responses:
- factual
- organized
- concise
"""
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Research artificial intelligence"
        }
    ]
})

print(response["messages"][-1].content)