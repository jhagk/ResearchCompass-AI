from langchain_openai import ChatOpenAI
from rag.retriever import get_qa_chain
from tools.arxiv_tool import fetch_arxiv_papers
from config import OPENAI_API_KEY, LLM_MODEL


def check_domain_drift(
    topics_text: str,
    domain: str,
    llm
) -> str:
    """
    Checks if any suggested topic drifts away from domain.
    Flags topics using wrong methods for the domain.
    """

    prompt = f"""
    Domain: "{domain}"

    Suggested topics:
    {topics_text}

    Check each topic carefully:
    1. Does it stay within the domain "{domain}"?
    2. Does it use methods appropriate for "{domain}"?

    Common drift examples:
    - Domain is "deep learning" but topic uses
      Random Forest, SVM, Logistic Regression → DRIFT
    - Domain is "NLP" but topic is about image processing → DRIFT
    - Domain is "computer vision" but topic uses
      text processing → DRIFT
    - Domain is "healthcare" but topic is about finance → DRIFT

    For each topic write:
    Topic [number]: [VALID or DRIFTED]
    If DRIFTED:
    - Issue: [what method is wrong]
    - Corrected Title: [better version staying in {domain}]

    If all topics are valid just say "All topics are valid."
    Keep response short and direct.
    """

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return "Domain check skipped."


def run_topic_suggester(domain: str, level: str = "M.Tech") -> dict:
    """
    Suggests 5 novel research topics based on domain and level.
    Each topic is traceable to specific retrieved papers.
    Includes domain drift detection.
    """

    print(f"Topic Suggester running for domain: {domain}, "
          f"level: {level}")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.3
    )

    # Step 1 - fetch papers
    papers = fetch_arxiv_papers.invoke(domain)

    # Step 2 - ask RAG chain with traceability prompt
    qa_chain = get_qa_chain(domain)
    result = qa_chain.invoke({
        "query": f"""
        Based on the research papers provided, suggest 5 novel
        research topics in {domain} suitable for an {level}
        student in India.

        For EACH topic provide EXACTLY this format:

        ### Topic [number]: [Topic Title]

        **Inspired By:**
        - Paper: [exact title of paper from context that inspired this]
        - Finding: [one sentence — what finding in that paper
          suggests this gap]

        **Research Gap:**
        [one sentence — what problem exists that this topic solves]

        **Generated Topic:**
        [2 sentences — what this research will do and why it matters]

        Rules:
        - Every topic MUST cite a real paper from the context
        - Do NOT invent paper titles
        - Only use papers actually provided in the context
        - Topics must be specific to {domain}
        - Must be feasible for {level} in 1-2 years
        - India context preferred where applicable
        - If domain contains "deep learning" only suggest
          deep learning methods — NOT Random Forest or SVM
        - If domain contains "NLP" only suggest NLP methods
        - If domain contains "computer vision" only suggest
          vision methods
        """
    })

    # Step 3 - check for domain drift
    print("Checking for domain drift...")
    drift_report = check_domain_drift(
        result["result"], domain, llm
    )

    if "DRIFTED" in drift_report:
        print(f"Domain drift detected:\n{drift_report}")
    else:
        print("No domain drift detected.")

    # Step 4 - extract source papers used
    source_papers = [
        {
            "title": doc.metadata.get("title", ""),
            "url": doc.metadata.get("url", "")
        }
        for doc in result["source_documents"]
    ]

    print(f"Topic Suggester done.")
    return {
        "domain": domain,
        "level": level,
        "suggested_topics": result["result"],
        "drift_report": drift_report,
        "source_papers": source_papers
    }