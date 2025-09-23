import os
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# HuggingFace embeddings (must match the one used in ingest.py)
hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use Chroma persistent client
client = chromadb.PersistentClient(path=CHROMA_DIR)

COLLECTION_NAME = "legal_docs"
try:
    client.get_collection(COLLECTION_NAME)  # just to check existence
except:
    client.create_collection(COLLECTION_NAME)

# Now wrap it with LangChain's Chroma wrapper
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=hf,
    collection_name=COLLECTION_NAME,
    client=client,
)

def get_retrieval_qa(llm_type="openai", openai_api_key=None):
    # choose LLM
    if llm_type == "openai":
        if openai_api_key is None:
            raise ValueError("OpenAI key required")
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    else:
        raise NotImplementedError("Local LLM not yet implemented")

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

def answer_query(query: str, openai_api_key: str):
    qa = get_retrieval_qa(llm_type="openai", openai_api_key=openai_api_key)
    result = qa.run(query)
    return result
