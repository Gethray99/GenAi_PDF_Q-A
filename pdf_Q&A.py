import os 
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_gemini import Gemini

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

pdf_loader = PyPDFLoader("/CV/Gethwan_Suriyawansa.pdf")

documents = pdf_loader.load()

chain = load_qa_chain(llm=Gemini(model="gemini-1.5-flash", api_key=api_key, verbose=True))

