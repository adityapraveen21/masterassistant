from dotenv import load_dotenv
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

import PyPDF2

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

@tool
def read_pdf(file_path):
    """Read a PDF file and extract text."""

    try:
        text = ""

        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:
                text += page.extract_text() or ""

        if not text.strip():
            return "Error: No readable text found in PDF."

        return text

    except Exception as e:
        return f"Error reading PDF: {str(e)}"


# -----------------------------
# Agent
# -----------------------------
agent = create_agent(
    tools=[read_pdf],
    model=llm,
    system_prompt="""
You are PDFBot.
Use the tool to read PDF files and explain their contents clearly in simple terms.
Summarize key points after reading.
"""
)

# -----------------------------
# Run example
# -----------------------------
response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Read and explain this PDF: sample.pdf"
        }
    ]
})

print(response["messages"][-1].content)