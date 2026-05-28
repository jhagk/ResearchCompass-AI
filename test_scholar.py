from tools.scholar_tool import fetch_semantic_scholar_papers
import time

# Test 1 - CS domain
print("Test 1: NLP")
papers = fetch_semantic_scholar_papers("natural language processing")
print(f"Papers: {len(papers)}")
for p in papers[:2]:
    print(f"  - {p['title']}")
    print(f"    Fields: {p['categories']}")
print()

time.sleep(5)  # wait 5 seconds

# Test 2 - Interdisciplinary
print("Test 2: AI with Healthcare")
papers = fetch_semantic_scholar_papers("machine learning healthcare diagnosis")
print(f"Papers: {len(papers)}")
for p in papers[:2]:
    print(f"  - {p['title']}")
    print(f"    Fields: {p['categories']}")
print()

time.sleep(5)  # wait 5 seconds

# Test 3 - Non CS domain
print("Test 3: Blockchain with Law")
papers = fetch_semantic_scholar_papers("blockchain legal contracts")
print(f"Papers: {len(papers)}")
for p in papers[:2]:
    print(f"  - {p['title']}")
    print(f"    Fields: {p['categories']}")