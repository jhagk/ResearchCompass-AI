from tools.arxiv_tool import fetch_arxiv_papers

papers = fetch_arxiv_papers.invoke("blockchain")

print(f"Total papers fetched: {len(papers)}")
print("---")

for i, paper in enumerate(papers[:6]):
    print(f"Paper {i+1}:")
    print(f"  Title    : {paper['title']}")
    print(f"  Authors  : {', '.join(paper['authors'])}")
    print(f"  Published: {paper['published']}")
    print(f"  URL      : {paper['url']}")
    print()