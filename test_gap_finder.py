from agents.gap_finder import run_gap_finder

result = run_gap_finder("natural language processing")

print("Domain:", result["domain"])
print("\nResearch Gaps Found:")
print(result["gaps"])
print("\nPapers Used:")
for p in result["papers_used"]:
    print(f"  - {p}")