from tools.arxiv_tool import fetch_arxiv_papers
from rag.document_loader import load_and_split_papers
from rag.vector_store import build_and_save, load_vector_store

# Step 1 - fetch and chunk
papers = fetch_arxiv_papers.invoke("machine learning")
chunks = load_and_split_papers(papers)

# Step 2 - build and save
vector_store = build_and_save(chunks)

# Step 3 - reload and test search
vs = load_vector_store()
results = vs.similarity_search("what are open problems in machine learning", k=3)

print(f"Top 3 similar chunks:")
print("---")
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content[:200]}")
    print()