from tools.arxiv_tool import fetch_arxiv_papers
from rag.document_loader import load_and_split_papers

papers = fetch_arxiv_papers.invoke("machine learning")
chunks = load_and_split_papers(papers)

print(f"\nTotal chunks: {len(chunks)}")

# Print first 5 chunks
for i, chunk in enumerate(chunks[:5]):
    print("\n" + "="*50)
    print(f"Chunk {i+1}")
    print("="*50)

    print(chunk.page_content[:500])  # first 500 chars
    print("\nMetadata:")
    print(chunk.metadata)


