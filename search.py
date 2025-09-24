import os
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Embeddings
hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Persistent Chroma
client = chromadb.PersistentClient(path=CHROMA_DIR)

COLLECTION_NAME = "legal_docs"
try:
    client.get_collection(COLLECTION_NAME)
except:
    client.create_collection(COLLECTION_NAME)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=hf,
    collection_name=COLLECTION_NAME,
    client=client,
)

def get_retrieval_qa(model_name="llama-3-70b-8192"):
    """Return RetrievalQA chain using Groq LLM"""
    llm = ChatGroq(
         model="llama-3.1-8b-instant",
        # model=model_name, # available: "llama-3-8b-8192", "llama-3-70b-8192", "mixtral-8x7b-32768"
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=1024,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def answer_query(query: str):
    qa = get_retrieval_qa()
    result = qa.invoke({"query": query})
    return result
