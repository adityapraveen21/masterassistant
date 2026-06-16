from dotenv import load_dotenv
import os
import requests

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
def get_news(topic):
    """Get latest news headlines about a topic."""

    api_key = os.getenv("NEWS_API_KEY")

    url = "https://newsapi.org/v2/everything"

    r = requests.get(url, params={
        "q": topic,
        "apiKey": api_key,
        "pageSize": 5,
        "sortBy": "publishedAt",
        "language": "en"
    })

    data = r.json()

    if "articles" not in data:
        return "Could not fetch news."

    articles = data["articles"]

    if not articles:
        return f"No news found for {topic}."

    output = f"Latest news about {topic}:\n\n"

    for i, article in enumerate(articles, start=1):
        title = article["title"]

        output += f"{i}. {title}\n"

    return output


agent = create_agent(
    tools=[get_news],
    model=llm,
    system_prompt="""
You are NewsBot.
Use the news tool to answer questions about current events and latest headlines.
"""
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What are the latest AI news headlines?"
        }
    ]
})

print(response["messages"][-1].content)