from agents.trend_scanner import run_trend_scanner

result = run_trend_scanner("natural language processing")

print("Domain:", result["domain"])
print("\nTrending Topics:")
print(result["trends"])
print("\nPapers Used:")
for p in result["papers_used"]:
    print(f"  - {p}")