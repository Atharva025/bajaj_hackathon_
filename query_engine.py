# query_engine.py

import os
import fitz  # PyMuPDF
import docx
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from the .env file
load_dotenv()

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Load the components ---
print("Loading all necessary components...")

# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'}
)

# 2. Load the FAISS vector store
# The 'allow_dangerous_deserialization=True' is needed for FAISS with LangChain.
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# 3. Create a retriever from the vector store
retriever = db.as_retriever(search_kwargs={'k': 4}) # Retrieve top 4 relevant chunks

# 4. Initialize the LLM via OpenRouter
openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_model = os.getenv("OPENROUTER_MODEL_NAME", "openai/gpt-4o") # Default model

llm = ChatOpenAI(
    model=openrouter_model,
    openai_api_key=openrouter_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "http://localhost", # Recommended by OpenRouter
    }
)

# 5. Define the prompt template
prompt_template = """
You are an expert insurance claim adjudicator. Your task is to evaluate a claim based ONLY on the provided policy clauses and return a structured JSON response.

**Policy Clauses (Context):**
{context}

**User Query:**
{question}

Based strictly on the context provided, determine the claim's status.
Your response MUST be a JSON object with three keys:
1. "decision": A string, either "Approved" or "Rejected".
2. "amount": An integer representing the approved amount. If rejected, this should be 0.
3. "justification": A string explaining the decision by referencing the specific policy clauses from the context.

JSON Response:
"""

# 6. Create the RAG Chain using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | PromptTemplate.from_template(prompt_template)
    | llm
    | StrOutputParser()
)

print("✅ Components loaded successfully. Ready to receive queries.")

def get_response(query: str):
    """
    Takes a user query, runs it through the RAG chain, and returns the structured JSON response.
    """
    print(f"Received query: {query}")
    response = rag_chain.invoke(query)
    print(f"Generated response: {response}")
    return response