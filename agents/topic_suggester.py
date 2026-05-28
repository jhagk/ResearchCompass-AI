from langchain_openai import ChatOpenAI
from rag.retriever import get_qa_chain
from tools.arxiv_tool import fetch_arxiv_papers
from tools.scholar_tool import fetch_semantic_scholar_papers
from config import OPENAI_API_KEY, LLM_MODEL


def discover_emerging_techniques(
    domain: str,
    llm
) -> str:
    """
    Dynamically discovers emerging techniques
    from recent ArXiv papers — not hardcoded list.
    """

    print("Discovering emerging techniques from papers...")

    # fetch recent papers from last 2 years
    recent_papers = fetch_arxiv_papers.invoke(
        f"{domain} new method 2024 2025"
    )

    # extract technique names from paper titles
    titles = "\n".join(
        p.get("title", "") for p in recent_papers[:15]
    )

    prompt = f"""
    Based on these recent paper titles in {domain}:
    {titles}

    Extract the TOP 10 emerging techniques or methods
    that appear NEW (published 2023-2025).

    Rules:
    - Only include techniques that appear in titles above
    - Must be specific method names not generic terms
    - One technique per line
    - No explanation — just technique names

    Example output:
    Vision Mamba
    Federated LoRA
    KAN for medical imaging
    """

    response = llm.invoke(prompt)
    techniques = response.content.strip()
    print(f"Discovered techniques:\n{techniques}")
    return techniques


def discover_india_gaps(
    domain: str,
    papers: list,
    llm
) -> str:
    """
    Dynamically discovers India-specific gaps
    from fetched papers — not hardcoded list.
    """

    print("Discovering India-specific gaps...")

    abstracts = "\n".join(
        f"- {p.get('title', '')}: "
        f"{p.get('abstract', '')[:200]}"
        for p in papers[:10]
    )

    prompt = f"""
    Domain: {domain}

    Recent papers:
    {abstracts}

    What are 5 INDIA-SPECIFIC gaps in {domain}?

    Think about:
    - What works globally but fails in India?
    - What India-specific data/language issues exist?
    - What infrastructure constraints exist in India?
    - What Indian population-specific challenges exist?

    One gap per line. Be specific.
    No explanation — just the gap statement.
    """

    response = llm.invoke(prompt)
    gaps = response.content.strip()
    print(f"India gaps found:\n{gaps}")
    return gaps


def generate_20_candidates(
    domain: str,
    level: str,
    emerging_techniques: str,
    india_gaps: str,
    context: str,
    llm
) -> list:
    """
    Generates 20 candidate topics using
    Emerging Technique + Problem + India formula.
    """

    print("Generating 20 candidate topics...")

    prompt = f"""
    You are a research advisor for {domain}.

    Emerging techniques found from recent papers:
    {emerging_techniques}

    India-specific gaps:
    {india_gaps}

    Research context:
    {context[:1500]}

    Generate EXACTLY 20 candidate research topics.

    FORMULA for each topic:
    [Emerging Technique] + [Specific Problem]
    + [India Context if applicable]

    Rules:
    - Each topic must use one emerging technique above
    - Each topic must address one specific problem
    - At least 10 topics must have India context
    - No generic topics like "CNN for X" or "LSTM for Y"
    - Topics should be specific enough to search on ArXiv

    Return ONLY topic titles — one per line.
    No numbering. No explanation.
    """

    response = llm.invoke(prompt)
    candidates = [
        t.strip() for t in
        response.content.strip().split("\n")
        if t.strip() and len(t.strip()) > 10
    ][:20]

    print(f"Generated {len(candidates)} candidates.")
    return candidates


def validate_topic_novelty(topic: str) -> dict:
    """
    Validates topic novelty using real paper count.
    More reliable than asking LLM.
    """

    papers = fetch_semantic_scholar_papers(
        topic[:60], max_results=10
    )

    paper_count = len(papers)
    avg_citations = 0

    if papers:
        citations = [
            p.get("citation_count", 0) or 0
            for p in papers
        ]
        avg_citations = sum(citations) / len(citations)

    # novelty decision based on real data
    if paper_count == 0:
        novelty = "Unverified"
        verdict = "UNCERTAIN"
    elif paper_count <= 5:
        novelty = "High"
        verdict = "ACCEPT"
    elif paper_count <= 20:
        novelty = "Medium"
        verdict = "ACCEPT"
    elif paper_count <= 50:
        novelty = "Low"
        verdict = "CONSIDER"
    else:
        novelty = "Saturated"
        verdict = "REJECT"

    return {
        "topic": topic,
        "paper_count": paper_count,
        "avg_citations": round(avg_citations, 1),
        "novelty": novelty,
        "verdict": verdict
    }


def validate_all_candidates(
    candidates: list
) -> list:
    """
    Validates all 20 candidates using real paper counts.
    Keeps only ACCEPT and CONSIDER topics.
    """

    print("Validating novelty with real paper counts...")
    validated = []

    for topic in candidates:
        result = validate_topic_novelty(topic)
        print(
            f"  [{result['verdict']}] "
            f"{topic[:50]} "
            f"({result['paper_count']} papers)"
        )

        if result["verdict"] in ["ACCEPT", "CONSIDER"]:
            validated.append({
                "topic": topic,
                "paper_count": result["paper_count"],
                "avg_citations": result["avg_citations"],
                "novelty": result["novelty"]
            })

    print(
        f"Validated: {len(validated)}/{len(candidates)} "
        f"topics accepted."
    )
    return validated


def select_best_5(
    validated_topics: list,
    domain: str,
    level: str,
    context: str,
    llm
) -> str:
    """
    From validated topics selects best 5
    using LLM scoring on pre-validated topics only.
    """

    if not validated_topics:
        return "No valid topics found. Try broader domain."

    # build topics list with paper counts
    topics_text = ""
    for i, t in enumerate(validated_topics):
        topics_text += (
            f"{i+1}. {t['topic']} "
            f"(papers: {t['paper_count']}, "
            f"novelty: {t['novelty']})\n"
        )

    prompt = f"""
    Domain: {domain}
    Level: {level}

    These topics have been validated with real paper counts:
    {topics_text}

    Context from papers:
    {context[:1000]}

    Select the BEST 5 topics and format each as:

    ### Topic [number]: [Topic Title]

    **Novelty Evidence:**
    - Papers found: [number from above]
    - Why novel: [specific reason]

    **Inspired By:**
    - Paper: [paper from context]
    - Finding: [what gap it reveals]

    **Research Gap:**
    [one specific sentence]

    **Generated Topic:**
    [2 sentences — what and why India matters]

    Selection criteria:
    1. Prefer topics with LOW paper count (< 20)
    2. Prefer topics with India context
    3. Prefer topics using newest techniques
    4. Must be feasible for {level} in 1-2 years
    """

    response = llm.invoke(prompt)
    return response.content.strip()


def check_domain_drift(
    topics_text: str,
    domain: str,
    llm
) -> str:
    """
    Checks if selected topics drift from domain.
    """

    prompt = f"""
    Domain: "{domain}"
    Topics: {topics_text}

    For each topic:
    Topic [number]: [VALID or DRIFTED]
    If DRIFTED:
    - Issue: [problem]
    - Fix: [corrected version]

    If all valid: "All topics are valid."
    """

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return "Domain check skipped."


def run_topic_suggester(
    domain: str,
    level: str = "M.Tech"
) -> dict:
    """
    Suggests 5 GENUINELY NOVEL research topics.

    Pipeline:
    1. Discover emerging techniques dynamically
    2. Discover India-specific gaps dynamically
    3. Generate 20 candidate topics
    4. Validate each with real paper counts
    5. Score validated topics with LLM
    6. Return best 5 with evidence
    """

    print(f"Topic Suggester running for domain: {domain}, "
          f"level: {level}")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.5
    )

    # Step 1 - fetch papers
    papers = fetch_arxiv_papers.invoke(domain)

    # Step 2 - get RAG context
    qa_chain = get_qa_chain(domain)
    rag_result = qa_chain.invoke({
        "query": f"""
        What are the main research gaps and unsolved
        problems in {domain}?
        What existing methods fail?
        """
    })
    context = rag_result["result"]

    # Step 3 - discover emerging techniques dynamically
    emerging_techniques = discover_emerging_techniques(
        domain, llm
    )

    # Step 4 - discover India gaps dynamically
    india_gaps = discover_india_gaps(domain, papers, llm)

    # Step 5 - generate 20 candidates
    candidates = generate_20_candidates(
        domain, level,
        emerging_techniques,
        india_gaps,
        context,
        llm
    )

    # Step 6 - validate with real paper counts
    validated = validate_all_candidates(candidates)

    # Step 7 - select best 5 from validated
    print("Selecting best 5 topics...")
    best_topics = select_best_5(
        validated, domain, level, context, llm
    )

    # Step 8 - domain drift check
    print("Checking domain drift...")
    drift_report = check_domain_drift(
        best_topics, domain, llm
    )

    if "DRIFTED" in drift_report:
        print(f"Drift detected: {drift_report}")
    else:
        print("No domain drift detected.")

    # Step 9 - source papers
    source_papers = [
        {
            "title": doc.metadata.get("title", ""),
            "url": doc.metadata.get("url", "")
        }
        for doc in rag_result["source_documents"]
    ]

    print(f"Topic Suggester done.")
    return {
        "domain": domain,
        "level": level,
        "suggested_topics": best_topics,
        "drift_report": drift_report,
        "source_papers": source_papers
    }