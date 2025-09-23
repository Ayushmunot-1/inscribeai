# ingest.py
import os
from utils import encrypt_bytes, parse_document
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# create sentence-transformers model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Use Chroma client
# client = chromadb.Client(Settings(chroma_db_impl="chromadb.db.duckdb", persist_directory=CHROMA_DIR))
# import chromadb

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
client = chromadb.PersistentClient(path=CHROMA_DIR)

COLLECTION_NAME = "legal_docs"
if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
    collection = client.create_collection(name=COLLECTION_NAME)
else:
    collection = client.get_collection(COLLECTION_NAME)

# splitter config
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def ingest_file(filename: str, file_bytes: bytes, uploader: str = "unknown"):
    # encrypt and store raw file (quick)
    os.makedirs("encrypted_files", exist_ok=True)
    enc = encrypt_bytes(file_bytes)
    filepath = os.path.join("encrypted_files", filename + ".enc")
    with open(filepath, "wb") as f:
        f.write(enc)

    # parse
    text = parse_document(filename, file_bytes)
    if not text or len(text.strip()) == 0:
        # fallback - you can add OCR later
        text = "[NO TEXT EXTRACTED]"

    # chunk
    docs = splitter.split_text(text)
    # create ids & metadata
    doc_ids = []
    metadatas = []
    texts = []
    for i, chunk in enumerate(docs):
        id_ = f"{filename}__chunk_{i}"
        doc_ids.append(id_)
        texts.append(chunk)
        metadatas.append({"source_file": filename, "chunk": i, "uploader": uploader})

    # embeddings (batch)
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # add to chroma
    collection.add(
        ids=doc_ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )
    # client.persist()
    return {"added": len(doc_ids), "file": filename}
