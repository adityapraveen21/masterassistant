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
def draft_mail(topic,tone):
    """Draft an email with the given topic and tone."""
    
    return f"""
Write a {tone} email about {topic}:

The email should include a subject line, greeting, clear body and professional closing.
"""

agent = create_agent(
    tools = [draft_mail],
    model = llm,
    system_prompt=
    """you are a helpful e-mail draft generating agent. you will assist with creating  clear concise mail drafts for the user."""
)

response = agent.invoke(
    {
        "messages":[
            {
                "role":"user",
                "content":"Write a professional mail requesting sick leave"
            }
        ]
    }
)

print(response["messages"][-1].content)
