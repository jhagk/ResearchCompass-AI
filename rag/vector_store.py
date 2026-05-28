import os
from langchain_community.vectorstores import FAISS
from rag.embedder import get_embedder
from config import FAISS_INDEX_PATH


def get_domain_index_path(domain: str) -> str:
    """
    Returns domain-specific FAISS index path.
    """
    safe_domain = domain.replace(" ", "_").replace("/", "_")[:30]
    return f"{FAISS_INDEX_PATH}_{safe_domain}"


def build_vector_store(chunks: list):
    """
    Build FAISS vector store from document chunks.
    """
    embedder = get_embedder()
    vector_store = FAISS.from_documents(chunks, embedder)
    print(f"Built vector store with {len(chunks)} chunks.")
    return vector_store


def save_vector_store(vector_store, domain: str):
    """
    Save FAISS index to domain-specific path.
    """
    path = get_domain_index_path(domain)
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)
    print(f"Vector store saved to {path}")


def load_vector_store(domain: str = None):
    """
    Load FAISS index from domain-specific path.
    """
    embedder = get_embedder()
    path = get_domain_index_path(domain) if domain else FAISS_INDEX_PATH
    vector_store = FAISS.load_local(
        path,
        embedder,
        allow_dangerous_deserialization=True
    )
    print(f"Vector store loaded from {path}")
    return vector_store


def build_and_save(chunks: list, domain: str):
    """
    Full pipeline: chunks -> build -> save.
    """
    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store, domain)
    return vector_store


def index_exists(domain: str) -> bool:
    """
    Check if FAISS index already exists for this domain.
    """
    path = get_domain_index_path(domain)
    return os.path.exists(os.path.join(path, "index.faiss"))