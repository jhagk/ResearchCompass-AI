from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

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
ARXIV_MAX_RESULTS = 20

# FAISS paths
FAISS_INDEX_PATH = "data/faiss_index"
REPORTS_PATH = "data/reports"
CACHE_PATH = "data/cache"

# Agent settings
MAX_TOPICS = 5
RESEARCH_LEVELS = ["B.Tech", "M.Tech", "PhD"]

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")