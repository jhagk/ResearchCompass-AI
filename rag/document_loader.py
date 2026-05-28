from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import CHUNK_SIZE, CHUNK_OVERLAP

def papers_to_documents(papers: list) -> list:
    """
    Convert raw ArXiv paper dicts into LangChain Document objects.
    """
    documents = []

    for paper in papers:
        content = f"""
Title: {paper['title']}

Abstract: {paper['abstract']}

Authors: {', '.join(paper['authors'])}
Published: {paper['published']}
URL: {paper['url']}
        """.strip()

        doc = Document(
            page_content=content,
            metadata={
                "title": paper["title"],
                "url": paper["url"],
                "published": paper["published"],
                "authors": paper["authors"]
            }
        )
        documents.append(doc)

    return documents


def split_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def load_and_split_papers(papers: list) -> list:
    """
    Full pipeline: papers list -> chunked documents ready for embedding.
    """
    documents = papers_to_documents(papers)
    chunks = split_documents(documents)
    return chunks