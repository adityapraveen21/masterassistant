from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from langchain.tools import tool
from langchain_openai import ChatOpenAI

import requests
import numexpr
import os
from dotenv import load_dotenv


load_dotenv()


tasks=[]
chat_history = ""


class AgentState(TypedDict):
    user_input: str
    route: str
    response: str
    chat_history: str



llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    r = requests.get(
        url,
        params={
            "q": city,
            "appid": api_key,
            "units": "metric"
        }
    )

    data = r.json()

    if "weather" not in data:
        return f"Error: {data.get('message', 'unknown error')}"

    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]

    return f"{city}: {weather}, {temp}°C"


@tool
def calculator(expression: str) -> str:
    """Useful for solving mathematical expressions."""

    result = numexpr.evaluate(expression)

    return str(result)

@tool
def get_news(topic: str) -> str:
    """Get latest news headlines about a topic."""

    api_key = os.getenv("NEWS_API_KEY")

    url = "https://newsapi.org/v2/everything"

    r = requests.get(
        url,
        params={
            "q": topic,
            "apiKey": api_key,
            "pageSize": 5,
            "sortBy": "publishedAt",
            "language": "en"
        }
    )

    data = r.json()

    if "articles" not in data:
        return "Could not fetch news."

    articles = data["articles"]

    if not articles:
        return f"No news found for {topic}."

    output = f"Latest news about {topic}:\n\n"

    for i, article in enumerate(articles, start=1):
        output += f"{i}. {article['title']}\n"

    return output

@tool
def currency_conv(
    amount: float,
    from_currency: str,
    to_currency: str
) -> str:
    """Convert money from one currency to another."""

    api_key = os.getenv("EXCHANGE_RATE_API_KEY")

    url = (
        f"https://v6.exchangerate-api.com/v6/"
        f"{api_key}/latest/{from_currency.upper()}"
    )

    r = requests.get(url)

    data = r.json()

    if data.get("result") != "success":
        return "Currency conversion failed."

    rates = data.get("conversion_rates", {})

    target_rate = rates.get(to_currency.upper())

    if target_rate is None:
        return f"Unsupported currency: {to_currency}"

    converted_amount = amount * target_rate

    return (
        f"{amount:.2f} {from_currency.upper()} = "
        f"{converted_amount:.2f} {to_currency.upper()}"
    )


@tool
def get_stock_price(symbol: str) -> str:
    """Get the latest stock price for a stock ticker."""

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    url = "https://www.alphavantage.co/query"

    r = requests.get(
        url,
        params={
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.upper(),
            "apikey": api_key
        }
    )

    data = r.json()

    quote = data.get("Global Quote")

    if not quote:
        return f"Could not find stock data for {symbol}"

    price = quote.get("05. price")
    change = quote.get("09. change")
    percent = quote.get("10. change percent")

    return (
        f"Stock: {symbol.upper()}\n"
        f"Price: ${price}\n"
        f"Change: {change}\n"
        f"Percent Change: {percent}"
    )

@tool
def add_task(task: str) -> str:
    """Add a task to the todo list."""

    tasks.append(task)

    return f"Task added: {task}"

@tool
def view_tasks() -> str:
    """View all tasks."""

    if not tasks:
        return "No tasks in the todo list."

    output = ""

    for i, task in enumerate(tasks, start=1):
        output += f"{i}. {task}\n"

    return output


@tool
def remove_task(task_number: int) -> str:
    """Remove a task by its number."""

    if task_number < 1 or task_number > len(tasks):
        return "Invalid task number."

    removed = tasks.pop(task_number - 1)

    return f"Removed task: {removed}"


def router_node(state: AgentState):

    query = state["user_input"]
    history = state["chat_history"]

    prompt = f"""
You are a routing agent.

Conversation History:
{history}

Choose exactly one route from:

weather
calculator
news
currency
stock
todo
general_chat

Return ONLY the route name.

User Query:
{query}
"""

    response = llm.invoke(prompt)

    route = response.content.strip().lower()

    valid_routes = {
        "weather",
        "calculator",
        "news",
        "currency",
        "stock",
        "todo",
        "general_chat"
    }

    if route not in valid_routes:
        route = "general_chat"

    return {
        "route": route
    }


def weather_node(state: AgentState):

    query = state["user_input"]
    history = state["chat_history"]

    city_prompt = f"""
Extract only the city name from this weather query.

Conversation History:
{history}
Query:
{query}

Return only the city name.
"""

    city_response = llm.invoke(city_prompt)

    city = city_response.content.strip()

    weather_info = get_weather.invoke(city)

    return {
        "response": weather_info
    }


def calculator_node(state: AgentState):

    query = state["user_input"]
    history = state["chat_history"]

    expression_prompt = f"""
Extract only the mathematical expression from the query.

Conversation History:
{history}

Query:
{query}

Return only the expression.
"""

    response = llm.invoke(expression_prompt)

    expression = response.content.strip()

    result = calculator.invoke(expression)

    return {
        "response": result
    }


def news_node(state: AgentState):

    query = state["user_input"]
    history = state["chat_history"]

    topic_prompt = f"""
Extract the news topic from the query.

Conversation History:
{history}

Query:
{query}

Return only the topic.
"""

    response = llm.invoke(topic_prompt)

    topic = response.content.strip()

    news_result = get_news.invoke(topic)

    return {
        "response": news_result
    }


def currency_node(state: AgentState):

    query = state["user_input"]
    history = state["chat_history"]

    prompt = f"""
Extract:

AMOUNT
FROM
TO

Return exactly in this format:

AMOUNT: value
FROM: currency
TO: currency

Conversation History:
{history}
Query:
{query}
"""

    response = llm.invoke(prompt)

    amount = None
    from_currency = None
    to_currency = None

    for line in response.content.splitlines():

        if line.startswith("AMOUNT:"):
            amount = float(
                line.replace("AMOUNT:", "").strip()
            )

        elif line.startswith("FROM:"):
            from_currency = (
                line.replace("FROM:", "")
                .strip()
                .upper()
            )

        elif line.startswith("TO:"):
            to_currency = (
                line.replace("TO:", "")
                .strip()
                .upper()
            )

    result = currency_conv.invoke(
        {
            "amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency
        }
    )

    return {
        "response": result
    }

def todo_node(state: AgentState):

    query = state["user_input"]
    history = state["chat_history"]

    action_prompt = f"""
Determine which todo action the user wants.

Choose exactly one:

add
view
remove

Conversation History:
{history}
Query:
{query}

Return only the action name.
"""

    action_response = llm.invoke(action_prompt)

    action = action_response.content.strip().lower()

    if action == "view":

        result = view_tasks.invoke({})

        return {
            "response": result
        }

    elif action == "add":

        task_prompt = f"""
Extract only the task to be added.

Conversation History:
{history}
Query:
{query}

Return only the task text.
"""

        task_response = llm.invoke(task_prompt)

        task = task_response.content.strip()

        result = add_task.invoke(task)

        return {
            "response": result
        }

    elif action == "remove":

        remove_prompt = f"""
Extract only the task number to remove.

Conversation History:
{history}
Query:
{query}

Return only the number.
"""

        remove_response = llm.invoke(remove_prompt)

        task_number = int(remove_response.content.strip())

        result = remove_task.invoke(
    {
        "task_number": task_number
    }
)

        return {
            "response": result
        }

    else:

        return {
            "response": "Could not determine todo action."
        }

def stock_node(state: AgentState):

    query = state["user_input"]
    history = state["chat_history"]
    
    prompt = f"""
Extract only the stock ticker symbol.

Conversation History:
{history}

Examples:

Apple -> AAPL
Microsoft -> MSFT
Tesla -> TSLA
Nvidia -> NVDA
Amazon -> AMZN
Google -> GOOGL
Meta -> META

Query:
{query}

Return only the ticker symbol.
"""

    response = llm.invoke(prompt)

    symbol = response.content.strip().upper()

    stock_info = get_stock_price.invoke(symbol)

    return {
        "response": stock_info
    }

def general_chat_node(state: AgentState):

    query = state["user_input"]
    history = state["chat_history"]

    prompt = f"""
Conversation History:
{history}

User:
{query}
"""

    response = llm.invoke(prompt)

    return {
        "response": response.content
    }


def route_decision(state: AgentState):
    return state["route"]


graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("todo", todo_node)
graph.add_node("weather", weather_node)
graph.add_node("calculator", calculator_node)
graph.add_node("news", news_node)
graph.add_node("currency", currency_node)
graph.add_node("stock", stock_node)
graph.add_node("general_chat", general_chat_node)

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "weather": "weather",
        "calculator": "calculator",
        "news": "news",
        "currency": "currency",
        "stock": "stock",
        "todo": "todo",
        "general_chat": "general_chat"
    }
)

graph.add_edge("weather", END)
graph.add_edge("calculator", END)
graph.add_edge("news", END)
graph.add_edge("currency", END)
graph.add_edge("stock", END)
graph.add_edge("todo", END)
graph.add_edge("general_chat", END)


app = graph.compile()


if __name__ == "__main__":

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        result = app.invoke(
            {
                "user_input": user_input,
                "route": "",
                "response": "",
                "chat_history": chat_history
            }
        )

        assistant_response = result["response"]

        print("\nAssistant:", assistant_response)

        chat_history += (
            f"\nUser: {user_input}\n"
            f"Assistant: {assistant_response}\n"
        )
