from datetime import datetime
from tools.arxiv_tool import fetch_arxiv_papers
from tools.web_search_tool import web_search
from rag.retriever import get_qa_chain


def run_trend_scanner(domain: str) -> dict:
    """
    Scans ArXiv and web to find emerging and specific
    trending topics in given domain.
    Returns trending areas with brief explanation.
    """

    print(f"Trend Scanner running for domain: {domain}")

    current_year = datetime.now().year

    # Step 1 - fetch papers from arxiv
    papers = fetch_arxiv_papers.invoke(domain)

    # Step 2 - web search for trends
    web_results = web_search.invoke(
        f"latest emerging research trends {domain} {current_year}"
    )

    # Step 3 - ask RAG chain with specific prompt
    qa_chain = get_qa_chain(domain)
    result = qa_chain.invoke({
        "query": f"""
        What are the TOP 5 EMERGING and SPECIFIC research trends
        in {domain} in 2025-2026?

        Rules:
        - Do NOT mention broad areas like
          "Medical Image Analysis" or "Drug Discovery"
        - Mention SPECIFIC emerging techniques and methods
        - Each trend must include a specific method or technology
        - Each trend must be a current development not older than 2 years

        Examples of GOOD specific trends:
        For healthcare:
        "Federated Learning for privacy-preserving medical diagnosis"
        "Vision Transformers replacing CNNs in medical imaging"
        "LLMs being used for clinical note summarization"
        "Self-supervised learning for rare disease detection"

        For NLP:
        "Chain-of-thought prompting improving LLM reasoning"
        "Mixture of Experts replacing dense transformer models"
        "RAG systems for knowledge-grounded text generation"

        For Computer Vision:
        "Segment Anything Model replacing traditional segmentation"
        "Diffusion models for medical image synthesis"
        "CLIP-based zero-shot classification"

        Now list top 5 specific emerging trends for {domain}.
        Use simple language.
        Maximum 5 trends.
        Each trend in 2-3 sentences.
        """
    })

    # Step 4 - format output
    trending_topics = {
        "domain": domain,
        "trends": result["result"],
        "papers_used": [
            doc.metadata["title"]
            for doc in result["source_documents"]
        ],
        "web_results": web_results[:3]
    }

    print(f"Trend Scanner done.")
    return trending_topics