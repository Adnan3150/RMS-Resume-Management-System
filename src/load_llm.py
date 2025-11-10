from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]  

load_dotenv(dotenv_path=ROOT_DIR / ".env")  
api_key=os.getenv("GROQ_API_KEY")
# print("groq_api_key: ", api_key)
def load():
    llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.1,
                groq_api_key=api_key
            )
    return llm