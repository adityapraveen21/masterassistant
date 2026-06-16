from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

import requests
import os

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


@tool
def get_stock_price(symbol: str) -> str:
    """
    Get the latest stock price for a stock ticker symbol.

    Examples:
    AAPL
    MSFT
    TSLA
    NVDA
    """

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


agent = create_agent(
    tools=[get_stock_price],
    model=llm,
    system_prompt="""
You are StockBot.

Use the get_stock_price tool whenever the user asks about:

- stock prices
- share prices
- company stock value
- market quotes

Convert company names into ticker symbols when possible.

Examples:

Apple -> AAPL
Microsoft -> MSFT
Tesla -> TSLA
Nvidia -> NVDA
Amazon -> AMZN
Google -> GOOGL
Meta -> META

Always use the tool for stock-price requests.
"""
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is Apple's stock price?"
            }
        ]
    }
)

print(response["messages"][-1].content)