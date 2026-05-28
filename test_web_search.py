from tools.web_search_tool import web_search

results = web_search.invoke("latest news of today")

print(f"Total results: {len(results)}")
print("------")

for i, r in enumerate(results, 1):
    print(f"{i}. {r['title']}")
    print(f"URL: {r['url']}")
    print(f"Snippet: {r['snippet']}")
    print()