from rag.retriever import get_qa_chain


def run_feasibility_checker(domain: str, topics: str, level: str = "M.Tech") -> dict:
    """
    Checks feasibility of suggested research topics.
    Returns feasibility score and timeline for each topic.
    """

    print(f"Feasibility Checker running for domain: {domain}")

    # Step 1 - ask RAG chain
    qa_chain = get_qa_chain(domain)
    result = qa_chain.invoke({
        "query": f"""
        For a {level} student in India working in {domain},
        evaluate the feasibility of these research topics:
        
        {topics}
        
        For each topic provide:
        - Feasibility score (Easy / Medium / Hard)
        - Estimated time to complete (in months)
        - Data availability (Available / Limited / Difficult)
        - One tip to get started
        
        Use simple language suitable for a student.
        """
    })

    # Step 2 - format output
    feasibility = {
        "domain": domain,
        "level": level,
        "feasibility_report": result["result"],
        "papers_used": [doc.metadata["title"] for doc in result["source_documents"]]
    }

    print(f"Feasibility Checker done.")
    return feasibility