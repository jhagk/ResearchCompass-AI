from agents.topic_suggester import run_topic_suggester
from agents.citation_scout import run_citation_scout

# Step 1 - get topics first
topics_result = run_topic_suggester("natural language processing", level="M.Tech")

# Step 2 - find key papers
result = run_citation_scout(
    domain="natural language processing",
    topics=topics_result["suggested_topics"]
)

print("Reading List:")
print(result["reading_list"])
print("\nActual Papers from ArXiv:")
for p in result["actual_papers"]:
    print(f"  - {p['title']}")
    print(f"    URL: {p['url']}")
    print()