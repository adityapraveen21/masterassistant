from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

llm = ChatOpenAI(model="openai/gpt-oss-20b:free",base_url="https://openrouter.ai/api/v1",api_key=os.getenv("OPENROUTER_API_KEY"),temperature=0.1)

response = llm.invoke("Say Hi")

print(response.content)