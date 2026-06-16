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
def summarize(text: str) -> str:
    """Summarize text into 2–3 sentences only."""

    return f"""
You are a summarization system.
Compress the text into 2–3 short sentences ONLY.
Remove all examples, repetition, and filler words.
Keep only core meaning.

Text:
{text}
"""


agent = create_agent(
    tools=[summarize],
    model=llm,
    system_prompt="You are summarizerbot. Use the tool to summarize text clearly and in a short form."
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Artificial intelligence has evolved significantly over the past few decades, transitioning from simple rule-based systems to highly advanced models capable of learning from vast amounts of data. Early AI systems were limited in scope and could only perform narrowly defined tasks, such as basic game playing or logical reasoning within constrained environments. However, with the advent of machine learning and, more specifically, deep learning, AI systems began to improve dramatically in their ability to recognize patterns, process natural language, and make predictions. Today, large language models are capable of generating human-like text, answering complex questions, writing code, and even assisting in creative tasks such as storytelling and design. Despite these advancements, AI systems still face significant limitations, including issues related to reasoning consistency, hallucination of facts, and dependence on large datasets for training. Researchers are actively working on improving the reliability, interpretability, and efficiency of these models. At the same time, ethical concerns such as bias, data privacy, and the potential misuse of AI technologies are becoming increasingly important in discussions about the future of the field. As AI continues to develop, it is expected to play an even greater role in industries such as healthcare, education, finance, and transportation, fundamentally transforming the way humans interact with technology and each other"
        }
    ]
})

print(response["messages"][-1].content)