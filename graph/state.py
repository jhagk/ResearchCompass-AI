from typing import TypedDict, Optional, Annotated
from operator import add


class ResearchState(TypedDict):

    # User input — read only, never written by agents
    domain: str
    level: str

    # Agent outputs — each written by ONE agent only
    trends: Optional[str]
    gaps: Optional[str]
    suggested_topics: Optional[str]
    source_papers: Optional[list]
    novelty_report: Optional[str]
    novelty_results: Optional[list]
    refined_topics: Optional[str]
    feasibility_report: Optional[str]
    roadmap: Optional[str]
    reading_list: Optional[str]

    # Paper metadata
    papers_used: Optional[list]
    actual_papers: Optional[list]

    # Final output
    final_report: Optional[str]
    error: Optional[str]