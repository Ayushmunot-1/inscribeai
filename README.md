# ⚖️ Inscribe.AI : Document Search & Synthesis Tool

> **A highly efficient AI tool for legal professionals to rapidly search, sort, and synthesize knowledge from massive document sets.**
> This project demonstrates an advanced Retrieval-Augmented Generation (RAG) architecture using Groq for real-time, low-latency performance combined together using Langchain.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/LLM%20Engine-Groq-00A98F?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-05E82D?style=for-the-badge&logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-189DFF?style=for-the-badge&logo=chroma&logoColor=white)](https://www.trychroma.com/)

---

## ✨ Core Features & Technical Highlights

This application moves beyond basic RAG by incorporating **agentic principles** and robust data management.

* **🧠 Agentic Retrieval (Multi-Query):** Uses the Groq LLM to decompose complex user questions intelligently (Query Division) into multiple sub-queries, ensuring comprehensive context is retrieved from the VectorDB, leading to more accurate answers.
* **🎯 Contextual Compression:** Implements an `LLMChainExtractor` (a form of Re-ranking) to filter out irrelevant information retrieved by the Multi-Query step, ensuring the final Groq model only sees the most pertinent chunks.
* **📄 This document parsing solution utilizes PyPDF and PDFMiner for efficient text extraction from digital PDFs, and intelligently falls back to Tesseract OCR for handling scanned or handwritten documents.
* **🔒 Persistent & Lifecycle Management:** Data is stored securely and locally in **ChromaDB**. It includes an **7-Day Automatic Buffer** to manage data lifecycle by deleting old documents in a separate, scheduled background process.
* **⚡ Low-Latency Synthesis:** Leverages the **Groq API** (`llama-3.1-8b-instant`) for blazing-fast answer generation.

---

## 🛠️ Project Architecture

The application is structured into two main, independently running processes for maximum resilience:

| File / Component | Purpose | Functionality |
| :--- | :--- | :--- |
| `app.py` | **Frontend** | Streamlit UI for file upload and querying. |
| `ingest.py` | **Ingestion Pipeline** | Handles file reading, encryption, chunking, embedding, and storage. |
| `search.py` | **Agentic RAG Engine** | Contains the Multi-Query Retriever, Contextual Compression, and the Groq LLM chain. |
| `scheduler.py` | **Background Process** | Runs continuously to automatically delete documents older than 7 days. |
| `utils.py` | **Utilities** | Contains file encryption (`Fernet`) and the robust **Hybrid PDF Parser** (`PyMuPDF` + `pytesseract`). |

---

## ⚙️ Setup and Installation

### Prerequisites

1.  **Python 3.10+**
2.  **Tesseract OCR Engine:** Must be installed separately on your operating system to enable the handwritten document feature.
3.  **Poppler (for Windows/Linux):** Required for `PyMuPDF` image rendering if Tesseract is used.

### Steps

1.  **Clone the repository:**
    ```bash
    git clone [YOUR-REPO-URL]
    cd [YOUR-REPO-NAME]
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate # Windows
    # source venv/bin/activate # macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables (`.env` file):**
    Create a file named `.env` in the root directory and add your API key and a secret key:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    FERNET_KEY=your_fernet_key_here
    # Optional: CHROMA_PERSIST_DIR=./chroma_db
    ```

---

## 🚀 How to Run the Application

You must run the frontend and the data lifecycle scheduler in **separate terminals**.

### 1. Start the UI

Run this command in your **first terminal**:
```bash
streamlit run app.py
