from tools.openalex_tool import fetch_openalex_papers

# Test 1 - CS domain
print("Test 1: NLP")
papers = fetch_openalex_papers("natural language processing")
print(f"Papers: {len(papers)}")
for p in papers[:2]:
    print(f"  - {p['title']}")
    print(f"    Fields: {p['categories']}")
    print(f"    Abstract: {p['abstract'][:100]}...")
print()

# Test 2 - Interdisciplinary
print("Test 2: AI with Healthcare")
papers = fetch_openalex_papers("machine learning healthcare diagnosis")
print(f"Papers: {len(papers)}")
for p in papers[:2]:
    print(f"  - {p['title']}")
    print(f"    Fields: {p['categories']}")
print()

# Test 3 - Non CS domain
print("Test 3: Blockchain with Law")
papers = fetch_openalex_papers("blockchain legal contracts")
print(f"Papers: {len(papers)}")
for p in papers[:2]:
    print(f"  - {p['title']}")
    print(f"    Fields: {p['categories']}")