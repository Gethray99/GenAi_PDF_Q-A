import os 
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

pdf_loader = PyPDFLoader("CV/Gethwan_Suriyawansa.pdf")

documents = pdf_loader.load()

chain = load_qa_chain(llm=ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", api_key=api_key, verbose=True), chain_type="stuff")
query = "what is the email of the person?"
response = chain.invoke({
    "input_documents": documents,
    "question": query
})
print(f"Response: {response['output_text']}")