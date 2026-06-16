from dotenv import load_dotenv

from langchain.tools import tool             #makes functions usable by agents as tools.
from langchain.agents import create_agent   #creates the actual combined usable agent. 
from langchain_openai import ChatOpenAI

import numexpr
import os                                         #pythons operating sys module, lets python interact with environment variables, files/folders etc.

load_dotenv()                                     #
llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

@tool
def calculator(expression: str) -> str:
    """Useful for solving mathematical expressions."""
    
    result = numexpr.evaluate(expression)
    return str(result)

agent = create_agent(                        #actually creating our agent finally, assigning it the tools, llm to use, agent type and verbose.
    tools = [calculator],                              #giving agent access to calculator tool.
    model = llm,                                              #brain of the agent, assigning LLM
    system_prompt="You are CalcBot, a helpful calculator agent."
)

def calculator(expression: str) -> str:
    """Useful for solving mathematical expressions."""
    
    result = numexpr.evaluate(expression)
    return str(result)

agent = create_agent(                        #actually creating our agent finally, assigning it the tools, llm to use, agent type and verbose.
    tools = [calculator],                              #giving agent access to calculator tool.
    model = llm,                                              #brain of the agent, assigning LLM
    system_prompt="You are CalcBot, a helpful calculator agent."
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 50 multiplied by 21 divided by 10?"
            }
        ]
    }
)
print(response["messages"][-1].content)



