from dotenv import load_dotenv
import os
import requests
import numexpr

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
def calculator(expression: str) -> str:
    """Solve mathematical expressions."""

    result = numexpr.evaluate(expression)

    return str(result)


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    r = requests.get(url, params={
        "q": city,
        "appid": api_key,
        "units": "metric"
    })

    data = r.json()

    if "weather" not in data:
        return "Could not fetch weather."

    temp = data["main"]["temp"]

    return str(temp)


agent = create_agent(
    tools=[calculator, get_weather],
    model=llm,
    system_prompt="""
You are ReActBot.

Reason carefully before using tools.

Use:
- calculator for math
- get_weather for weather information
"""
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": """
What is the current temperature in Chennai in Fahrenheit?
"""
        }
    ]
})

print(response["messages"][-1].content)