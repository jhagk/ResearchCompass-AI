from rag.embedder import get_embedder

embedder = get_embedder()

sample_text = "gautam kumar jha"
vector = embedder.embed_query(sample_text)

print(f"Embedding model: text-embedding-3-small")
print(f"Vector dimensions: {len(vector)}")
print(f"Sample values: {vector[:5]}")