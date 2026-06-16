from dotenv import load_dotenv
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openrouter/owl-alpha",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

@tool
def review_resume(resume_text: str) -> str:
    """Review a resume and provide improvement suggestions."""

    return f"""
Review the following resume.

Provide:
1. Strengths
2. Weaknesses
3. Suggestions for improvement
4. Overall feedback

Keep feedback concise and practical.

Resume:
{resume_text}
"""

agent = create_agent(
    tools=[review_resume],
    model=llm,
    system_prompt="""
You are ResumeBot.

Use the resume review tool to analyze resumes and provide professional feedback.

Keep responses:
- constructive
- concise
- practical
"""
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": """
Review this resume:

John Doe
Python Developer
Skills: Python, SQL, Excel
Experience: Worked on data analysis projects.
"""
        }
    ]
})

print(response["messages"][-1].content)