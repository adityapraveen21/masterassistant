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
# Tool
# -----------------------------
@tool
def explain_code(code: str) -> str:
    """Explain programming code in simple terms."""

    return f"""
Explain the following code clearly and simply.

Include:
- What the code does
- Line-by-line explanation
- Overall purpose

Code:
{code}
"""


# -----------------------------
# Agent
# -----------------------------
agent = create_agent(
    tools=[explain_code],
    model=llm,
    system_prompt="""
You are CodeExplainerBot.

Use the tool to explain code in a beginner-friendly way.

Keep explanations:
- clear
- simple
- structured
"""
)

# -----------------------------
# Run
# -----------------------------
response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": """
Explain this code:

for i in range(5):
    print(i)
"""
        }
    ]
})

print(response["messages"][-1].content)