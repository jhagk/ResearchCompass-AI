from itertools import combinations
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL
import numpy as np


def get_query_embedding(text: str, embedder) -> list:
    return embedder.embed_query(text)


def cosine_similarity(vec1: list, vec2: list) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def deduplicate_queries(queries: list, threshold: float = 0.90) -> list:
    if len(queries) <= 1:
        return queries

    embedder = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    print("Computing query embeddings for deduplication...")
    embeddings = []
    for query in queries:
        emb = get_query_embedding(query, embedder)
        embeddings.append(emb)

    keep = [True] * len(queries)

    for i in range(len(queries)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(queries)):
            if not keep[j]:
                continue
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                print(f"Duplicate found (similarity={sim:.2f}):")
                print(f"  Keep   : '{queries[i]}'")
                print(f"  Remove : '{queries[j]}'")
                keep[j] = False

    unique_queries = [q for q, k in zip(queries, keep) if k]
    print(f"Queries after deduplication: {len(unique_queries)} "
          f"(removed {len(queries) - len(unique_queries)})")
    return unique_queries


def extract_domains(user_input: str) -> list:
    """
    Extracts all individual technical domains from user input.
    """

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

    prompt = f"""
    Extract all individual technical research domains from: "{user_input}"
    
    Rules:
    - Return each domain as a short technical keyword (1-3 words)
    - Split combined domains into separate items
    - No explanation, one domain per line
    - Maximum 5 domains
    
    Examples:
    Input: "AI with IoT"
    Output:
    artificial intelligence
    internet of things
    
    Input: "Deep Learning in healthcare"
    Output:
    deep learning
    healthcare
    
    Input: "NLP with Blockchain for finance"
    Output:
    natural language processing
    blockchain
    finance
    
    Now extract for: "{user_input}"
    """

    response = llm.invoke(prompt)
    domains = [d.strip() for d in
               response.content.strip().split("\n")
               if d.strip()]
    print(f"Extracted domains: {domains}")
    return domains


def generate_expanded_queries(user_input: str, domains: list) -> list:
    """
    Generates domain-specific expanded queries using LLM.
    These are research-specific queries that will return
    relevant academic papers.
    """

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

    prompt = f"""
    A researcher is looking for academic papers about: "{user_input}"
    The main domains are: {', '.join(domains)}
    
    Generate 5 SPECIFIC academic search queries that will find
    highly relevant research papers on ArXiv and Semantic Scholar.
    
    Rules:
    - Each query must be 3-6 words
    - Each query must be specific to the research domain
    - Queries must target SPECIFIC research aspects
    - Do NOT generate generic queries like "deep learning" alone
    - Combine domain concepts with specific research aspects
    - One query per line, no numbering or explanation
    
    Examples for "Deep Learning in healthcare":
    medical image analysis deep learning
    clinical diagnosis neural networks
    healthcare AI disease prediction
    electronic health records deep learning
    medical transformers patient outcomes
    
    Examples for "NLP with Blockchain":
    natural language processing smart contracts
    blockchain text analysis
    NLP decentralized systems
    sentiment analysis blockchain transactions
    language models distributed ledger
    
    Examples for "AI with IoT":
    IoT sensor data machine learning
    edge computing deep learning devices
    federated learning IoT networks
    real-time IoT anomaly detection
    smart home AI prediction
    
    Examples for "Computer Vision":
    object detection convolutional neural networks
    image segmentation transformer models
    visual recognition deep learning
    scene understanding neural networks
    video analysis action recognition
    
    Now generate 5 specific queries for: "{user_input}"
    """

    response = llm.invoke(prompt)
    queries = [q.strip() for q in
               response.content.strip().split("\n")
               if q.strip() and len(q.strip()) > 5]

    print(f"Generated expanded queries: {queries[:5]}")
    return queries[:5]


def validate_queries(queries: list, domains: list) -> list:
    """
    Validates each query is relevant to at least one domain.
    """

    keywords = []
    for domain in domains:
        keywords.extend(domain.lower().split())

    validated = []
    for query in queries:
        query_lower = query.lower()
        is_relevant = any(keyword in query_lower for keyword in keywords)
        if is_relevant:
            validated.append(query)
            print(f"Query VALID: '{query}'")
        else:
            print(f"Query INVALID (skipped): '{query}'")

    # if all queries filtered out return original queries
    if not validated:
        print("All queries filtered — using original queries")
        return queries

    return validated


def build_arxiv_queries(user_input: str) -> list:
    """
    Full pipeline:
    1. Extract domains from user input
    2. Generate domain-specific expanded queries using LLM
    3. Validate all queries
    4. Deduplicate semantically similar queries
    5. Return final query list
    """

    # Step 1 - extract domains
    domains = extract_domains(user_input)

    # Step 2 - generate expanded specific queries
    queries = generate_expanded_queries(user_input, domains)

    # Step 3 - validate
    validated = validate_queries(queries, domains)

    # Step 4 - semantic deduplication
    unique_queries = deduplicate_queries(validated, threshold=0.90)

    # Step 5 - limit to 5 queries max
    final_queries = unique_queries[:5]

    print(f"\nFinal queries for '{user_input}':")
    for i, q in enumerate(final_queries):
        print(f"  {i+1}. {q}")

    return final_queries