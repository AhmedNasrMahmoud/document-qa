import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── 1. SETUP ──────────────────────────────────────────────────────────────────

# Load the embedding model (runs locally, no API cost)
embedder = SentenceTransformer("all-MiniLM-L2-v2")

# Create a local ChromaDB database (saves to a folder called chroma_db)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Anthropic client
client = anthropic.Anthropic()


# ── 2. STORE DOCUMENTS ────────────────────────────────────────────────────────

def load_and_store(filepath: str):
    """Read a text file, chunk it, embed it, and store in ChromaDB."""

    with open(filepath, "r") as f:
        text = f.read()

    # Split into chunks (simple approach: split by double newline)
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    print(f"Storing {len(chunks)} chunks from {filepath}...")

    # Convert chunks to vectors
    embeddings = embedder.encode(chunks).tolist()

    # Store in ChromaDB (each chunk needs a unique ID)
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    print("Done. Document stored.\n")


# ── 3. SEARCH ─────────────────────────────────────────────────────────────────

def search(question: str, n_results: int = 3) -> list[str]:
    """Convert question to vector, find the closest matching chunks."""

    question_embedding = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results
    )

    return results["documents"][0]  # list of matching chunks


# ── 4. ANSWER ─────────────────────────────────────────────────────────────────

def ask(question: str) -> str:
    """Search for context, then ask Claude with that context."""

    # Step 1: find relevant chunks
    context_chunks = search(question)
    context = "\n\n".join(context_chunks)

    # Step 2: build the prompt with context injected
    prompt = f"""You are a helpful assistant. Answer the user's question using 
only the context provided below. If the answer isn't in the context, say 
"I don't have that information in the documents provided."

CONTEXT:
{context}

QUESTION:
{question}
"""

    # Step 3: send to Claude
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


# ── 5. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load the document (only need to do this once)
    load_and_store("document.txt")

    # Ask questions
    questions = [
        "How do I get a refund?",
        "How long does express shipping take?",
        "Can I return electronics after 20 days?",
        "What is the capital of France?"  # not in the document — watch what happens
    ]

    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask(q)}")
        print("-" * 60)