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
def generate_questions(topic: str) -> str:
    """Generate interview questions for a topic."""

    return f"""
Generate 5 interview questions about:

{topic}

Keep them clear and practical.
"""

agent = create_agent(
    tools=[generate_questions],
    model=llm,
    system_prompt="""
You are InterviewBot.

Use the interview question tool to generate useful interview questions.

Keep questions concise and relevant.
"""
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Generate Python interview questions"
        }
    ]
})

print(response["messages"][-1].content)