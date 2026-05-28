import os
import streamlit as st
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# enable LLM caching
set_llm_cache(InMemoryCache())

load_dotenv()


def get_secret(key: str) -> str:
    """
    Works both locally and on Streamlit Cloud.
    Tries Streamlit secrets first, then .env file.
    """
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")


# API Keys
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = get_secret("SEMANTIC_SCHOLAR_API_KEY")

# LLM settings
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS = 2000
TEMPERATURE = 0.3

# RAG settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 5

# ArXiv settings
ARXIV_MAX_RESULTS = 8       # reduced for speed

# FAISS paths
FAISS_INDEX_PATH = "data/faiss_index"
REPORTS_PATH = "data/reports"
CACHE_PATH = "data/cache"

# Agent settings
MAX_TOPICS = 5
RESEARCH_LEVELS = ["B.Tech", "M.Tech", "PhD"]

# create required directories automatically
# needed for Streamlit Cloud deployment
os.makedirs("data/cache", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)
os.makedirs("data/faiss_index", exist_ok=True)