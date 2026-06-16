from dotenv import load_dotenv
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

# -----------------------------
# LLM
# -----------------------------
llm = ChatOpenAI(
    model="openrouter/owl-alpha",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

# -----------------------------
# Tool
# -----------------------------
@tool
def define_word(word: str) -> str:
    """Define a word and give an example sentence."""

    return f"""
Define the word: {word}

Give:
1. A simple definition
2. One example sentence
"""

agent = create_agent(
    tools=[define_word],
    model=llm,
    system_prompt="""
You are DictionaryBot.
Use the tool to define words clearly and provide example usage.
Keep explanations simple and concise.
"""
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What does discombobulated mean?"
        }
    ]
})

print(response["messages"][-1].content)