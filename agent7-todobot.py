from dotenv import load_dotenv
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

# -----------------------------
# LLM
# -----------------------------
llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

# -----------------------------
# Task storage
# -----------------------------
tasks = []

# -----------------------------
# Tools
# -----------------------------
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


agent = create_agent(
    tools=[add_task, view_tasks, remove_task],
    model=llm,
    system_prompt="""
You are TodoBot.
Use the tools to manage the user's todo list.
"""
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "add play football task"
        }
    ]
})

print(response["messages"][-1].content)