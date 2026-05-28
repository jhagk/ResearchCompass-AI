from agents.topic_suggester import run_topic_suggester

result = run_topic_suggester("natural language processing", level="M.Tech")

print("Domain:", result["domain"])
print("Level:", result["level"])
print("\nSuggested Research Topics:")
print(result["suggested_topics"])
print("\nPapers Used:")
for p in result["papers_used"]:
    print(f"  - {p}")