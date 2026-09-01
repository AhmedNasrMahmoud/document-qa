import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── SETUP ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="centered"
)

# Cache these so they don't reload on every interaction
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_db():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(name="documents")

embedder = load_embedder()
collection = load_db()
client = anthropic.Anthropic()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def store_document(text: str):
    """Chunk, embed, and store a document."""
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    embeddings = embedder.encode(chunks).tolist()
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )
    return len(chunks)

def ask(question: str) -> tuple[str, list[str]]:
    """Search for context, ask Claude, return answer + sources."""
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=3)
    context_chunks = results["documents"][0]
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the user's question using 
only the context provided below. If the answer isn't in the context, say 
"I don't have that information in the documents provided."

CONTEXT:
{context}

QUESTION:
{question}
"""
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text, context_chunks


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("📄 Document Q&A")
st.caption("Upload a document, then ask questions about it.")

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        if st.button("Store Document"):
            with st.spinner("Processing..."):
                n_chunks = store_document(text)
            st.success(f"Stored {n_chunks} chunks!")

# --- Main: Chat Interface ---
st.divider()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Sources used"):
                for i, chunk in enumerate(msg["sources"]):
                    st.caption(f"Chunk {i+1}: {chunk}")

# Chat input
if question := st.chat_input("Ask a question about your document..."):

    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Get and show answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, sources = ask(question)
        st.write(answer)
        with st.expander("📚 Sources used"):
            for i, chunk in enumerate(sources):
                st.caption(f"Chunk {i+1}: {chunk}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })