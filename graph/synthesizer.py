import re
from graph.state import ResearchState


def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def synthesize_report(state: ResearchState) -> str:

    domain = state.get("domain", "Unknown")
    level = state.get("level", "M.Tech")
    trends = state.get("trends", "Not available")
    gaps = state.get("gaps", "Not available")
    suggested_topics = state.get("suggested_topics", "Not available")
    source_papers = state.get("source_papers", [])
    novelty_report = state.get("novelty_report", "Not available")
    refined_topics = state.get("refined_topics", "Not available")
    feasibility_report = state.get(
        "feasibility_report", "Not available"
    )
    roadmap = state.get("roadmap", "Not available")
    reading_list = state.get("reading_list", "Not available")
    proposal_guide = state.get("proposal_guide", "Not available")
    actual_papers = state.get("actual_papers", [])

    report = f"""
============================================================
         RESEARCHCOMPASS AI - RESEARCH REPORT
============================================================

Domain  : {domain}
Level   : {level}

------------------------------------------------------------
1. TRENDING TOPICS
------------------------------------------------------------
{trends}

------------------------------------------------------------
2. RESEARCH GAPS
------------------------------------------------------------
{gaps}

------------------------------------------------------------
3. SUGGESTED RESEARCH TOPICS
------------------------------------------------------------
{suggested_topics}

------------------------------------------------------------
3a. EVIDENCE TRAIL
------------------------------------------------------------
"""

    # deduplicate source papers by URL or normalized title
    seen_ids = set()
    unique_source_papers = []
    for paper in source_papers:
        url = paper.get("url", "").strip()
        title = normalize_title(paper.get("title", ""))
        paper_id = url or title
        if paper_id and paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique_source_papers.append(paper)

    if unique_source_papers:
        for i, paper in enumerate(unique_source_papers):
            report += f"""
Paper {i+1}: {paper.get('title', 'Unknown')}
URL      : {paper.get('url', 'Not available')}
"""
    else:
        report += "No source papers available.\n"

    report += f"""
------------------------------------------------------------
4. NOVELTY ANALYSIS
------------------------------------------------------------
{novelty_report}

------------------------------------------------------------
5. REFINED PUBLICATION-STYLE TOPICS
------------------------------------------------------------
{refined_topics}

------------------------------------------------------------
6. FEASIBILITY REPORT
------------------------------------------------------------
{feasibility_report}

------------------------------------------------------------
7. RESEARCH ROADMAP
------------------------------------------------------------
{roadmap}

------------------------------------------------------------
8. STARTER READING LIST
------------------------------------------------------------
{reading_list}

------------------------------------------------------------
9. KEY PAPERS
------------------------------------------------------------
"""

    for i, paper in enumerate(actual_papers):
        source = paper.get("source", "arxiv").upper()
        score = paper.get("relevance_score", 0)
        report += f"""
Paper {i+1}: {paper['title']}
Authors  : {', '.join(paper['authors'])}
Published: {paper['published']}
Source   : {source}
Score    : {score}/100
URL      : {paper['url']}
"""

    report += f"""
------------------------------------------------------------
10. PROPOSAL WRITING GUIDE
------------------------------------------------------------
WARNING: Write every section in YOUR OWN WORDS.
This guide tells you WHAT to write — not the text itself.
Zero AI plagiarism when you follow this guide.

{proposal_guide}
"""

    return report