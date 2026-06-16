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
def currency_conv(
    amount: float,
    from_currency: str,
    to_currency: str
) -> str:
    """
    Convert money from one currency to another.

    Example:
    amount=100
    from_currency='USD'
    to_currency='INR'
    """

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


agent = create_agent(
    tools=[currency_conv],
    model=llm,
    system_prompt="""
You are CurrencyBot.

Use the currency_conv tool whenever the user asks
to convert money between currencies.

Examples:
- Convert 100 USD to INR
- Convert 50 EUR to GBP
- Convert 1000 JPY to USD

Always use the tool for currency conversions.
"""
)


response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Convert 100 USD to INR"
            }
        ]
    }
)

print(response["messages"][-1].content)