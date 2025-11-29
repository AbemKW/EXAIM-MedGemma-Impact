import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"),
    google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY"),
    streaming=True
)

groq_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("GROQ_MODEL", "qwen/qwen3-32b"),
)