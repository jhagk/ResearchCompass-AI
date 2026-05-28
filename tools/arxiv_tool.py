import arxiv
import json
import os
import re
import time
import numpy as np
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from config import ARXIV_MAX_RESULTS, CACHE_PATH, OPENAI_API_KEY, EMBEDDING_MODEL


def _get_cache_path(query: str) -> str:
    safe_name = query.replace(" ", "_").replace("/", "_")[:50]
    return os.path.join(CACHE_PATH, f"{safe_name}.json")


def _load_from_cache(query: str):
    path = _get_cache_path(query)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _save_to_cache(query: str, data: list):
    path = _get_cache_path(query)
    with open(path, "w") as f:
        json.dump(data, f)


def normalize_title(title: str) -> str:
    """
    Normalizes paper title for deduplication.
    Handles minor variations like punctuation, case, spaces.
    """
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def deduplicate_papers(papers: list) -> list:
    """
    Removes duplicate papers using best available identifier.
    Priority: URL > normalized title
    Handles duplicates across ArXiv, OpenAlex, Semantic Scholar.
    """
    seen = set()
    unique = []

    for paper in papers:
        # use URL as primary identifier
        # fall back to normalized title
        paper_id = (
            paper.get("url", "").strip()
            or normalize_title(paper.get("title", ""))
        )

        if not paper_id:
            continue

        if paper_id not in seen:
            seen.add(paper_id)
            unique.append(paper)

    print(
        f"Deduplication: {len(papers)} -> {len(unique)} papers "
        f"(removed {len(papers) - len(unique)} duplicates)"
    )
    return unique


def build_boolean_query(query: str) -> str:
    """
    Converts plain query into ArXiv boolean search format.
    Uses simpler format to avoid rate limiting.
    """
    words = query.strip().split()

    if len(words) == 1:
        return f'all:{words[0]}'

    elif len(words) == 2:
        return f'ti:{words[0]} AND all:{words[1]}'

    elif len(words) >= 3:
        all_parts = " OR all:".join(words)
        return f'all:{all_parts}'

    return query


def cosine_similarity(v1: list, v2: list) -> float:
    """
    Computes cosine similarity between two vectors.
    """
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def semantic_rerank(
    papers: list,
    original_query: str,
    top_k: int = 10
) -> list:
    """
    Reranks fetched papers by semantic similarity to original query.
    Ensures most relevant papers come first.
    """

    if not papers:
        return papers

    embedder = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    query_embedding = embedder.embed_query(original_query)

    scored_papers = []
    for paper in papers:
        paper_text = f"{paper['title']} {paper['abstract'][:300]}"
        paper_embedding = embedder.embed_query(paper_text)
        score = cosine_similarity(query_embedding, paper_embedding)
        scored_papers.append((score, paper))

    scored_papers.sort(key=lambda x: x[0], reverse=True)

    reranked = [paper for _, paper in scored_papers[:top_k]]
    print(f"Reranked {len(papers)} papers -> kept top {len(reranked)}")
    return reranked


def filter_by_abstract(papers: list, keywords: list) -> list:
    """
    Filters papers where abstract contains at least one keyword.
    Removes completely irrelevant papers.
    """

    filtered = []
    for paper in papers:
        abstract_lower = paper["abstract"].lower()
        title_lower = paper["title"].lower()
        combined = abstract_lower + " " + title_lower

        is_relevant = any(
            keyword.lower() in combined
            for keyword in keywords
        )

        if is_relevant:
            filtered.append(paper)

    print(f"Abstract filter: {len(papers)} -> {len(filtered)} papers")
    return filtered


@tool
def fetch_arxiv_papers(query: str) -> list:
    """
    Fetch recent research papers from ArXiv.
    Uses boolean search + abstract filtering + semantic reranking.
    """

    cached = _load_from_cache(query)
    if cached:
        print(f"Loaded {len(cached)} papers from cache.")
        return cached

    boolean_query = build_boolean_query(query)
    print(f"Boolean query: {boolean_query}")

    client = arxiv.Client(
        page_size=20,
        delay_seconds=5,
        num_retries=5
    )

    search = arxiv.Search(
        query=boolean_query,
        max_results=ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    papers = []
    try:
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "abstract": result.summary,
                "authors": [a.name for a in result.authors[:3]],
                "url": result.entry_id,
                "published": str(result.published.date()),
                "categories": result.categories
            })
        print(f"Fetched {len(papers)} papers from ArXiv.")

    except Exception as e:
        print(f"Boolean search failed: {e}")
        print("Falling back to plain search...")
        papers = []

    if not papers:
        time.sleep(5)
        try:
            search = arxiv.Search(
                query=query,
                max_results=ARXIV_MAX_RESULTS,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            for result in client.results(search):
                papers.append({
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [a.name for a in result.authors[:3]],
                    "url": result.entry_id,
                    "published": str(result.published.date()),
                    "categories": result.categories
                })
            print(f"Fallback fetched {len(papers)} papers.")

        except Exception as e:
            print(f"Fallback search also failed: {e}")
            return []

    keywords = query.split()
    papers = filter_by_abstract(papers, keywords)
    papers = semantic_rerank(papers, query, top_k=10)

    _save_to_cache(query, papers)
    return papers


def fetch_arxiv_papers_multi(queries: list) -> list:
    """
    Fetch papers from ArXiv + OpenAlex + Semantic Scholar.
    Deduplicates using URL and normalized title.
    Returns clean unique paper list.
    """
    from tools.openalex_tool import fetch_openalex_papers
    from tools.scholar_tool import fetch_semantic_scholar_papers

    all_papers = []
    seen_titles = set()

    for i, query in enumerate(queries):
        print(f"Fetching query {i+1}/{len(queries)}: {query}")

        # fetch from ArXiv
        arxiv_papers = fetch_arxiv_papers.invoke(query)
        for paper in arxiv_papers:
            title_norm = normalize_title(paper["title"])
            if title_norm not in seen_titles:
                seen_titles.add(title_norm)
                paper["source"] = "arxiv"
                all_papers.append(paper)

        # fetch from OpenAlex
        openalex_papers = fetch_openalex_papers(query)
        for paper in openalex_papers:
            title_norm = normalize_title(paper["title"])
            if title_norm not in seen_titles \
               and paper.get("abstract"):
                seen_titles.add(title_norm)
                all_papers.append(paper)

        # fetch from Semantic Scholar
        scholar_papers = fetch_semantic_scholar_papers(query)
        for paper in scholar_papers:
            title_norm = normalize_title(paper["title"])
            if title_norm not in seen_titles \
               and paper["abstract"] != "Abstract not available.":
                seen_titles.add(title_norm)
                all_papers.append(paper)

        if i < len(queries) - 1:
            print("Waiting 3 seconds before next query...")
            time.sleep(3)

    print(f"Total papers before dedup: {len(all_papers)}")
    print(
        f"ArXiv: "
        f"{sum(1 for p in all_papers if p.get('source') == 'arxiv')}"
    )
    print(
        f"OpenAlex: "
        f"{sum(1 for p in all_papers if p.get('source') == 'openalex')}"
    )
    print(
        f"Semantic Scholar: "
        f"{sum(1 for p in all_papers if p.get('source') == 'semantic_scholar')}"
    )

    # final deduplication by URL or normalized title
    all_papers = deduplicate_papers(all_papers)

    print(f"Total unique papers after dedup: {len(all_papers)}")
    return all_papers