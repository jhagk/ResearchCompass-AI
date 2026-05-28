import re
from langchain_openai import ChatOpenAI
from tools.arxiv_tool import fetch_arxiv_papers
from tools.arxiv_tool import fetch_arxiv_papers_multi
from tools.query_builder import build_arxiv_queries
from rag.retriever import get_qa_chain
from config import OPENAI_API_KEY, LLM_MODEL


# Domain-specific keyword mappings
DOMAIN_KEYWORDS = {
    "healthcare": ["medical", "clinical", "patient", "hospital",
                   "disease", "diagnosis", "treatment", "health",
                   "imaging", "therapeutic", "biomedical"],
    "deep learning": ["neural network", "cnn", "lstm", "transformer",
                      "convolutional", "deep learning", "classification",
                      "segmentation", "detection"],
    "nlp": ["language", "text", "nlp", "sentiment", "translation",
            "bert", "gpt", "transformer", "corpus"],
    "natural language processing": ["language", "text", "nlp",
                                    "sentiment", "translation",
                                    "bert", "gpt", "transformer"],
    "nlp in health": ["clinical", "medical", "patient",
                      "health", "disease", "diagnosis",
                      "ehr", "clinical notes", "biomedical",
                      "natural language", "text mining"],
    "nlp in healthcare": ["clinical", "medical", "patient",
                          "health", "disease", "diagnosis",
                          "ehr", "clinical notes", "biomedical",
                          "natural language", "text mining"],
    "natural language processing in health": [
        "clinical", "medical", "patient", "health",
        "disease", "diagnosis", "ehr", "biomedical",
        "natural language", "text mining", "clinical notes"
    ],
    "computer vision": ["image", "visual", "detection", "segmentation",
                        "recognition", "object", "video"],
    "blockchain": ["blockchain", "smart contract", "decentralized",
                   "ethereum", "distributed ledger", "consensus"],
    "iot": ["iot", "sensor", "edge computing", "embedded",
            "smart device", "internet of things", "wearable"],
    "internet of things": ["iot", "sensor", "edge computing",
                           "embedded", "smart device", "wearable"],
    "finance": ["financial", "stock", "trading", "market",
                "banking", "fraud", "credit", "risk"],
    "cybersecurity": ["security", "attack", "vulnerability",
                      "malware", "intrusion", "encryption"],
    "robotics": ["robot", "autonomous", "manipulation",
                 "locomotion", "planning", "control"],
    "drug discovery": ["drug", "molecule", "compound", "protein",
                       "binding", "pharmaceutical", "therapeutic"],
}


def get_domain_keywords(domain: str) -> list:
    """
    Returns specific keywords for a domain.
    """
    domain_lower = domain.lower()
    keywords = []

    for key, values in DOMAIN_KEYWORDS.items():
        if key in domain_lower:
            keywords.extend(values)

    keywords = list(set(keywords))

    if not keywords:
        stop_words = {"in", "for", "of", "the", "with",
                      "and", "or", "a", "an", "to", "using"}
        keywords = [
            w for w in domain_lower.split()
            if len(w) > 3 and w not in stop_words
        ]

    return keywords


def is_paper_relevant_to_domain(paper: dict, domain: str) -> bool:
    """
    Fast keyword-based pre-filter.
    Requires at least 2 domain keywords to match.
    """

    keywords = get_domain_keywords(domain)
    if not keywords:
        return True

    title_lower = paper.get("title", "").lower()
    abstract_lower = paper.get("abstract", "").lower()
    combined = title_lower + " " + abstract_lower

    matches = sum(1 for keyword in keywords if keyword in combined)
    return matches >= 2


def llm_relevance_score(paper: dict, domain: str, llm) -> int:
    """
    Uses LLM to score paper relevance.
    Forces natural score variation — not all 95.
    """

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")[:500]

    prompt = f"""
    Domain: {domain}

    Paper:
    Title: {title}
    Abstract: {abstract}

    Rate relevance of this paper to "{domain}" from 0-100.

    Strict scoring rules:
    - 95-100: Paper is EXACTLY about {domain} with direct
              methodology match. Very rare — only 1 in 10 papers.
    - 85-94 : Paper is directly about {domain} — clear match
    - 75-84 : Paper is related to {domain} — good match
    - 60-74 : Paper mentions {domain} but focuses elsewhere
    - 40-59 : Paper loosely related to {domain}
    - 0-39  : Paper is NOT about {domain}

    Important:
    - Do NOT give every paper 95
    - Papers should have DIFFERENT scores
    - Only give 95+ if paper is a perfect match
    - Most papers should score between 75-90
    - Be strict and vary your scores naturally

    Examples of varied scoring:
    - "Deep learning liver cirrhosis CNN" → 94
    - "Machine learning liver disease general" → 82
    - "Medical imaging deep learning survey" → 78
    - "CNN image classification general" → 65
    - "Natural language processing text" → 12

    Respond with ONLY a single integer 0-100.
    No explanation. Just the number.
    """

    try:
        response = llm.invoke(prompt)
        score_text = response.content.strip()
        numbers = re.findall(r'\d+', score_text)
        if numbers:
            score = int(numbers[0])
            score = max(0, min(100, score))
            return score
        return 50
    except Exception:
        return 50


def filter_papers_by_llm(
    papers: list,
    domain: str,
    threshold: int = 80
) -> list:
    """
    Filters papers using LLM relevance scoring.
    Keeps only papers scoring >= threshold.
    Returns papers sorted by score descending.
    """

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

    scored_papers = []

    for paper in papers:
        score = llm_relevance_score(paper, domain, llm)
        print(f"  LLM score {score}/100: {paper['title'][:50]}")

        if score >= threshold:
            paper["relevance_score"] = score
            scored_papers.append(paper)

    # sort by score descending
    scored_papers.sort(
        key=lambda x: x.get("relevance_score", 0),
        reverse=True
    )

    print(
        f"LLM filter: kept {len(scored_papers)}/{len(papers)} papers"
    )
    return scored_papers


def generate_reading_list_from_papers(
    paper_list: list,
    domain: str,
    topics: str,
    llm
) -> str:
    """
    Generates reading list ONLY from actual fetched papers.
    No hallucinated titles — every paper in reading list
    must exist in paper_list.
    """

    if not paper_list:
        return "No papers available for reading list."

    # build context from actual fetched papers only
    papers_context = ""
    for i, paper in enumerate(paper_list):
        papers_context += f"""
Paper {i+1}:
Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Published: {paper['published']}
URL: {paper['url']}
Source: {paper.get('source', 'arxiv').upper()}
"""

    prompt = f"""
    Domain: {domain}

    A student is researching these topics:
    {topics}

    Below are the ONLY real papers available.
    Build a reading list using ONLY these papers.
    Do NOT invent or mention any other paper titles.

    Available papers:
    {papers_context}

    For each paper write EXACTLY:

    ### Paper [number]: [Exact title from above — do not change]
    - **What it covers:** [1 simple sentence about the paper]
    - **Why read it:** [1 simple sentence why relevant to {domain}]
    - **Source:** [source name from above]
    - **URL:** [exact URL from above — do not change]

    Rules:
    - Use ONLY the papers listed above
    - Copy titles EXACTLY as given — do not paraphrase
    - Do NOT add any paper not in the list
    - Do NOT invent paper titles
    - Copy URLs exactly as given
    """

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        # fallback — simple formatted list
        reading_list = ""
        for i, paper in enumerate(paper_list):
            reading_list += f"""
### Paper {i+1}: {paper['title']}
- **Authors:** {', '.join(paper['authors'])}
- **Published:** {paper['published']}
- **Source:** {paper.get('source', 'arxiv').upper()}
- **URL:** {paper['url']}
"""
        return reading_list.strip()


def run_citation_scout(domain: str, topics: str) -> dict:
    """
    Finds 5 key papers to read for given domain and topics.
    Reading list is built DIRECTLY from fetched papers.
    No hallucinated titles — 100% traceable to Key Papers.
    """

    print(f"Citation Scout running for domain: {domain}")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

    # Step 1 - build smart queries
    queries = build_arxiv_queries(domain)

    # Step 2 - fetch papers from ALL 3 sources
    papers = fetch_arxiv_papers_multi(queries)

    if not papers:
        print("Falling back to direct domain search...")
        papers = fetch_arxiv_papers.invoke(domain)

    # Step 3 - fast keyword pre-filter
    candidate_papers = [
        p for p in papers
        if is_paper_relevant_to_domain(p, domain)
        and p.get("url", "")
        and p.get("abstract", "")
    ]

    print(f"After keyword filter: {len(candidate_papers)} candidates")

    # Step 4 - LLM relevance scoring with varied scores
    print("Running LLM relevance scoring...")
    relevant_papers = filter_papers_by_llm(
        candidate_papers,
        domain,
        threshold=80
    )

    # Step 5 - build diverse final list
    # max 3 from semantic scholar, rest from openalex/arxiv
    paper_list = []
    seen_titles = set()

    semantic_papers = [
        p for p in relevant_papers
        if p.get("source") == "semantic_scholar"
    ]
    openalex_papers = [
        p for p in relevant_papers
        if p.get("source") == "openalex"
    ]
    arxiv_papers = [
        p for p in relevant_papers
        if p.get("source", "arxiv") == "arxiv"
    ]

    # add up to 3 from semantic scholar
    for paper in semantic_papers:
        if len(paper_list) >= 3:
            break
        title = paper["title"]
        if title not in seen_titles:
            seen_titles.add(title)
            paper_list.append({
                "title": title,
                "url": paper["url"],
                "published": paper["published"],
                "authors": paper["authors"],
                "source": paper.get("source", "semantic_scholar"),
                "relevance_score": paper.get("relevance_score", 0)
            })

    # fill from openalex
    for paper in openalex_papers:
        if len(paper_list) >= 5:
            break
        title = paper["title"]
        if title not in seen_titles:
            seen_titles.add(title)
            paper_list.append({
                "title": title,
                "url": paper["url"],
                "published": paper["published"],
                "authors": paper["authors"],
                "source": paper.get("source", "openalex"),
                "relevance_score": paper.get("relevance_score", 0)
            })

    # fill from arxiv
    for paper in arxiv_papers:
        if len(paper_list) >= 5:
            break
        title = paper["title"]
        if title not in seen_titles:
            seen_titles.add(title)
            paper_list.append({
                "title": title,
                "url": paper["url"],
                "published": paper["published"],
                "authors": paper["authors"],
                "source": paper.get("source", "arxiv"),
                "relevance_score": paper.get("relevance_score", 0)
            })

    # fallback — more from semantic scholar
    if len(paper_list) < 5:
        print(f"Only {len(paper_list)} — adding more...")
        for paper in semantic_papers:
            if len(paper_list) >= 5:
                break
            title = paper["title"]
            if title not in seen_titles:
                seen_titles.add(title)
                paper_list.append({
                    "title": title,
                    "url": paper["url"],
                    "published": paper["published"],
                    "authors": paper["authors"],
                    "source": paper.get(
                        "source", "semantic_scholar"
                    ),
                    "relevance_score": paper.get(
                        "relevance_score", 0
                    )
                })

    print(f"Final paper list: {len(paper_list)} papers")
    for p in paper_list:
        print(
            f"  - [score:{p.get('relevance_score', 0)}] "
            f"[{p['source'].upper()}] {p['title'][:50]}"
        )

    # Step 6 - generate reading list DIRECTLY from paper_list
    # NO hallucination — only real fetched papers used
    print("Generating reading list from actual papers...")
    reading_list = generate_reading_list_from_papers(
        paper_list, domain, topics, llm
    )

    # Step 7 - get RAG source documents for reference
    qa_chain = get_qa_chain(domain)
    rag_result = qa_chain.invoke({
        "query": f"What are the key research papers in {domain}?"
    })

    # Step 8 - format output
    citations = {
        "domain": domain,
        "reading_list": reading_list,
        "actual_papers": paper_list,
        "papers_used": [
            doc.metadata["title"]
            for doc in rag_result["source_documents"]
        ]
    }

    print(f"Citation Scout done.")
    return citations