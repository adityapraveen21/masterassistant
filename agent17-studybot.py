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
def create_study_plan(goal: str) -> str:
    """Create a study plan for a learning goal."""

    return f"""
Create a clear study plan for the following goal:

{goal}

Include:
- daily tasks
- learning milestones
- revision suggestions
- practical practice

Keep the plan structured and realistic.
"""

agent = create_agent(
    tools=[create_study_plan],
    model=llm,
    system_prompt="""
You are StudyBot.

Use the study planning tool to create effective and realistic study schedules.

Keep plans:
- organized
- practical
- concise
"""
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Create a 7-day study plan for learning Python basics"
        }
    ]
})

print(response["messages"][-1].content)