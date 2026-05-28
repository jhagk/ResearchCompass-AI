from tools.query_builder import build_arxiv_queries

print("Test 1: Deep Learning in healthcare")
queries = build_arxiv_queries("Deep Learning in healthcare")
print(f"Final: {queries}\n")

print("Test 2: AI with Blockchain")
queries = build_arxiv_queries("AI with Blockchain")
print(f"Final: {queries}\n")

print("Test 3: NLP for Indian languages")
queries = build_arxiv_queries("NLP for Indian languages")
print(f"Final: {queries}\n")