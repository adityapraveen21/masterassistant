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
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of text."""

    return f"""
Analyze the sentiment of the following text:

{text}

Classify it as:
- Positive
- Negative
- Neutral

Also briefly explain why.
"""

agent = create_agent(
    tools=[analyze_sentiment],
    model=llm,
    system_prompt="""
You are SentimentBot.

Use the sentiment analysis tool to classify emotions and opinions in text.

Keep responses concise.
"""
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "I absolutely loved this movie. It was amazing."
        }
    ]
})

print(response["messages"][-1].content)