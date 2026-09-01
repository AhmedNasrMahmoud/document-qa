# Document Q&A — RAG Pipeline

A retrieval-augmented generation (RAG) application that lets you upload documents and ask questions about them in plain English. Built as part of an FDE skills development project.

## How It Works

1. Upload a `.txt` document via the sidebar
2. The document is chunked, embedded, and stored in a local vector database (ChromaDB)
3. When you ask a question, the app finds the most semantically relevant chunks
4. Those chunks are passed to Claude (Anthropic) as context to generate a grounded answer

## Tech Stack

- **LLM:** Anthropic Claude (claude-sonnet-4-6)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** ChromaDB
- **UI:** Streamlit
- **Language:** Python

## Setup

1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your Anthropic API key:
ANTHROPIC_API_KEY=your-key-here
6. Run the app: `streamlit run app.py`

## Project Structure
document-qa/
- **app.py** # Streamlit UI
- **rag.py** # RAG pipeline (CLI version)
- **ask.py** # Phase 1 — basic API call
- **document.txt** # Sample document
- **.env** # API key (not committed)
- **.gitignore** # Files excluded from git