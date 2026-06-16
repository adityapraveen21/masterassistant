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
def translate(text: str, language: str) -> str:
    """Translate text into a target language."""

    return f"Translate this text to {language}: {text}"

agent = create_agent(
    tools=[translate],
    model=llm,
    system_prompt="You are TranslatorBot. Use the tool to translate text accurately."
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Translate 'My name is Aditya' into Arabic"
        }
    ]
})

print(response["messages"][-1].content)