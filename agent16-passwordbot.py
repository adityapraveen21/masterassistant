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
def check_password(password: str) -> str:
    """Check password strength."""

    score = 0

    if len(password) >= 8:
        score += 1

    if any(char.isupper() for char in password):
        score += 1

    if any(char.islower() for char in password):
        score += 1

    if any(char.isdigit() for char in password):
        score += 1

    if any(not char.isalnum() for char in password):
        score += 1

    if score <= 2:
        strength = "Weak"

    elif score <= 4:
        strength = "Moderate"

    else:
        strength = "Strong"

    return f"""
Password Strength: {strength}

Score: {score}/5

A strong password should include:
- uppercase letters
- lowercase letters
- numbers
- special characters
- minimum 8 characters
"""

agent = create_agent(
    tools=[check_password],
    model=llm,
    system_prompt="""
You are PasswordBot.

Use the password checking tool to evaluate password strength.

Keep responses:
- concise
- security-focused
- practical
"""
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Check password strength for: Hello123"
        }
    ]
})

print(response["messages"][-1].content)