# app.py
import streamlit as st
from ingest import ingest_file
from search import answer_query
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Legal RAG - MVP", layout="wide")
st.title("Legal Document RAG — Upload & Query (MVP)")

# Simple password gate (replace with proper auth in Day2)
if "authorized" not in st.session_state:
    st.session_state.authorized = False

if not st.session_state.authorized:
    pwd = st.text_input("Enter password to access (for dev only)", type="password")
    if st.button("Enter"):
        if pwd == "devpass":  # change or implement streamlit-authenticator
            st.session_state.authorized = True
        else:
            st.error("Wrong password")
    st.stop()

st.header("Upload files")
uploaded_files = st.file_uploader("PDF / DOCX / TXT", accept_multiple_files=True)
uploader_name = st.text_input("Uploader name (optional)")
if st.button("Ingest files"):
    if not uploaded_files:
        st.warning("Please choose files")
    else:
        status_area = st.empty()
        for f in uploaded_files:
            b = f.read()
            status_area.text(f"Ingesting {f.name}...")
            res = ingest_file(f.name, b, uploader=uploader_name or "unknown")
            status_area.text(f"Ingested {res['file']}: {res['added']} chunks")
        st.success("Done ingesting")

st.header("Query")
q = st.text_area("Ask a question about your ingested documents")
if st.button("Search"):
    if not q:
        st.warning("Type a question")
    else:
        with st.spinner("Searching..."):
            if not OPENAI_API_KEY:
                st.error("Set OPENAI_API_KEY in .env to run retrieval + LLM")
            else:
                ans = answer_query(q, openai_api_key=OPENAI_API_KEY)
                st.markdown("### Answer")
                st.write(ans)

st.sidebar.info("Development MVP. Day 2 will include encryption, auth, OCR and citation formatting.")
