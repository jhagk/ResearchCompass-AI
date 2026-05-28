from langchain_openai import OpenAIEmbeddings
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import OPENAI_API_KEY, EMBEDDING_MODEL


def get_embedder():
    """
    Initialize and return OpenAI embeddings model.
    """
    embedder = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    return embedder