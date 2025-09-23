# utils.py
import os
import io
from cryptography.fernet import Fernet
from dotenv import load_dotenv
load_dotenv()
FERNET_KEY = os.getenv("FERNET_KEY")

if not FERNET_KEY:
    # generate and print one for your .env (only once)
    key = Fernet.generate_key()
    print("Generated FERNET_KEY:", key.decode())
    FERNET_KEY = key.decode()

fernet = Fernet(FERNET_KEY.encode())

def encrypt_bytes(b: bytes) -> bytes:
    return fernet.encrypt(b)

def decrypt_bytes(b: bytes) -> bytes:
    return fernet.decrypt(b)

# Basic file parsers
from pdfminer.high_level import extract_text as pdf_extract_text
import docx
def parse_pdf(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        try:
            text = pdf_extract_text(f)
        except Exception:
            text = ""
    return text

def parse_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        doc = docx.Document(f)
        texts = [p.text for p in doc.paragraphs]
    return "\n".join(texts)

def parse_txt(file_bytes: bytes) -> str:
    return file_bytes.decode(errors='ignore')

def parse_document(filename: str, file_bytes: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return parse_pdf(file_bytes)
    elif name.endswith(".docx"):
        return parse_docx(file_bytes)
    elif name.endswith(".txt"):
        return parse_txt(file_bytes)
    else:
        # fallback try decode
        return file_bytes.decode(errors='ignore')
