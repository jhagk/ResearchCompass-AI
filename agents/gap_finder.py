from tools.arxiv_tool import fetch_arxiv_papers
from rag.retriever import get_qa_chain


def extract_paper_limitations(papers: list, domain: str, llm) -> list:
    """
    Extracts specific limitations from each paper.
    Only extracts limitations directly related to domain.
    """

    limitations = []

    for paper in papers[:10]:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")[:400]

        prompt = f"""
        Domain: {domain}

        Paper Title: {title}
        Abstract: {abstract}

        What specific limitation, unsolved problem, or research gap
        related to {domain} is mentioned or implied in this paper?

        Rules:
        - Be very specific — mention exact method, dataset, or metric
        - The limitation MUST be directly related to {domain}
        - Do NOT mention limitations from other unrelated fields
        - Do NOT say generic things like "need more data"
        - One sentence only
        - If no clear limitation related to {domain} found say "None"
        """

        try:
            response = llm.invoke(prompt)
            limitation = response.content.strip()
            if limitation and limitation != "None":
                limitations.append({
                    "paper": title,
                    "limitation": limitation
                })
        except Exception:
            continue

    return limitations


def run_gap_finder(domain: str) -> dict:
    """
    Finds specific, evidence-based research gaps in given domain.
    All gaps are strictly related to the domain.
    """

    print(f"Gap Finder running for domain: {domain}")

    from langchain_openai import ChatOpenAI
    from config import OPENAI_API_KEY, LLM_MODEL

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.3
    )

    # Step 1 - fetch papers
    papers = fetch_arxiv_papers.invoke(domain)

    # Step 2 - extract specific limitations from each paper
    print("Extracting paper-level limitations...")
    paper_limitations = extract_paper_limitations(
        papers, domain, llm
    )

    # Step 3 - build limitations context
    limitations_text = ""
    for i, item in enumerate(paper_limitations):
        limitations_text += f"""
Paper {i+1}: {item['paper']}
Limitation: {item['limitation']}
"""

    # Step 4 - ask RAG chain with strict domain-specific prompt
    qa_chain = get_qa_chain(domain)
    result = qa_chain.invoke({
        "query": f"""
        You are a senior research advisor analyzing papers in {domain}.

        Based on the retrieved research papers, identify 5 SPECIFIC
        unsolved research problems and gaps in {domain}.

        Paper limitations found:
        {limitations_text}

        For each gap provide EXACTLY this format:

        ### Gap [number]:

        **Current Limitation:**
        [Specific problem — mention exact method, dataset, metric]

        **Why Existing Methods Fail:**
        [One sentence — what specifically breaks down]

        **Evidence from Papers:**
        [Which paper mentions this — be specific]

        **Research Opportunity:**
        [One sentence — what new approach could solve this]

        STRICT Rules:
        - Every gap MUST be directly about {domain}
        - Do NOT mention gaps from other fields like
          general AI, NLP, robotics, finance etc.
          unless they are directly applied to {domain}
        - AVOID generic statements like:
          "need more data", "need better models",
          "need more privacy", "need ethical AI"
        - Each gap must be SPECIFIC to {domain}
        - Must reference actual retrieved papers
        - Must mention specific methods, datasets, or metrics
        - Use simple language

        Examples of GOOD specific gaps for healthcare domain:
        - "BERT-based clinical NLP models trained on English EHRs
          perform 40% worse on Hindi-English code-mixed text"
        - "Current GANs for medical image synthesis fail to preserve
          rare disease morphology in minority classes"
        - "Existing federated learning frameworks require 3x more
          communication rounds for non-IID healthcare data"

        Examples of BAD gaps to AVOID:
        - "LLM post-training methods lack diversity" ← not {domain}
        - "3D editing models lack paired supervision" ← not {domain}
        - "Need more datasets in general" ← too generic
        """
    })

    print(f"Gap Finder done.")
    return {
        "domain": domain,
        "gaps": result["result"],
        "paper_limitations": paper_limitations,
        "papers_used": [
            doc.metadata["title"]
            for doc in result["source_documents"]
        ]
    }