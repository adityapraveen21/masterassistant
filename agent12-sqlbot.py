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
def generate_sql(request: str) -> str:
    """Generate SQL queries from natural language requests."""

    return f"""
Convert the following request into a valid SQL query.

Request:
{request}

Only return the SQL query.
"""
    
agent = create_agent(
    tools=[generate_sql],
    model=llm,
    system_prompt="""
You are SQLBot.

Use the SQL generation tool to create accurate SQL queries.

Return only SQL unless the user asks for explanation.
"""
)


response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Get all employees whose salary is greater than 50000"
        }
    ]
})

print(response["messages"][-1].content)