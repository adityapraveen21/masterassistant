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
    temperature=0.7
)

@tool
def generate_joke(topic: str) -> str:
    """Generate a joke about a topic."""

    return f"""
Generate a funny joke about:

{topic}

Keep it short and clean.
"""

agent = create_agent(
    tools=[generate_joke],
    model=llm,
    system_prompt="""
You are JokeBot.

Use the joke generation tool to create funny and harmless jokes.

Keep jokes short and entertaining.
"""
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Tell me a joke about programmers"
        }
    ]
})

print(response["messages"][-1].content)