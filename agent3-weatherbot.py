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
    temperature = 0
)

@tool
def get_weather(city):
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
        return f"Error: {data.get('message', 'unknown error')}"

    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]

    return f"{city}: {weather}, {temp}°C"


agent = create_agent(
    tools=[get_weather],
    model = llm,
    system_prompt = "You are WeatherBot, a helpful agent that answers weather questions using the weather tool"
)


response = agent.invoke(
    {
        "messages":[
            {
                "role":"user",
                "content":"What is the weather in Dubai"
            }
        ]
    }
)

print(response["messages"][-1].content)