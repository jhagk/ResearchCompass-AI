import time
import requests
import json
import os
from config import CACHE_PATH, SEMANTIC_SCHOLAR_API_KEY


def _get_cache_path(query: str) -> str:
    safe_name = query.replace(" ", "_").replace("/", "_")[:50]
    return os.path.join(CACHE_PATH, f"scholar_{safe_name}.json")


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


def _process_papers(raw_papers: list, query: str) -> list:
    """
    Process raw API response into clean paper list.
    Applies relevance filtering.
    """
    papers = []
    query_words = query.lower().split()

    for paper in raw_papers:

        # skip papers with no abstract
        abstract = paper.get("abstract")
        if not abstract:
            continue

        # relevance filter
        title_lower = paper.get("title", "").lower()
        abstract_lower = abstract.lower()
        combined = title_lower + " " + abstract_lower

        matches = sum(
            1 for word in query_words
            if word in combined
        )
        relevance = matches / len(query_words)

        # stricter threshold for longer queries
        if len(query_words) <= 2:
            threshold = 0.4
        elif len(query_words) <= 4:
            threshold = 0.6
        else:
            threshold = 0.7

        if relevance < threshold:
            continue

        # get URL
        url_link = ""
        external_ids = paper.get("externalIds", {})
        if external_ids.get("ArXiv"):
            url_link = f"https://arxiv.org/abs/{external_ids['ArXiv']}"
        elif external_ids.get("DOI"):
            url_link = f"https://doi.org/{external_ids['DOI']}"

        # get authors
        authors = [
            a.get("name", "")
            for a in paper.get("authors", [])[:3]
        ]

        # get fields of study
        fields = paper.get("fieldsOfStudy") or []

        # get citation count
        citation_count = paper.get("citationCount", 0)

        # get year
        year = str(paper.get("year", ""))

        papers.append({
            "title": paper.get("title", ""),
            "abstract": abstract,
            "authors": authors,
            "url": url_link,
            "published": year,
            "categories": fields,
            "citation_count": citation_count,
            "source": "semantic_scholar"
        })

    return papers


def fetch_semantic_scholar_papers(query: str, max_results: int = 10) -> list:
    """
    Fetch papers from Semantic Scholar API.
    Covers 200M+ papers across all domains.
    Uses API key for higher rate limits.
    Retries once on rate limit with 15 second delay.
    """

    cached = _load_from_cache(query)
    if cached:
        print(f"Loaded {len(cached)} Semantic Scholar papers from cache.")
        return cached

    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,authors,year,externalIds,fieldsOfStudy,citationCount"
    }

    # add API key for higher rate limits
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
        print(f"Using Semantic Scholar API key.")
    else:
        print(f"No API key found. Using free tier.")

    papers = []

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15
        )

        # success
        if response.status_code == 200:
            data = response.json()
            raw_papers = data.get("data", [])
            papers = _process_papers(raw_papers, query)
            print(f"Fetched {len(papers)} relevant papers from Semantic Scholar.")

        # rate limited - wait and retry once
        elif response.status_code == 429:
            print("Semantic Scholar rate limited. Waiting 15 seconds...")
            time.sleep(15)

            print("Retrying Semantic Scholar...")
            retry_response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=15
            )

            if retry_response.status_code == 200:
                data = retry_response.json()
                raw_papers = data.get("data", [])
                papers = _process_papers(raw_papers, query)
                print(f"Retry successful. Fetched {len(papers)} papers.")

            elif retry_response.status_code == 429:
                print("Still rate limited after retry. Skipping Semantic Scholar.")
                return []

            else:
                print(f"Retry failed: {retry_response.status_code}")
                return []

        # invalid API key
        elif response.status_code == 403:
            print("Semantic Scholar API key invalid or expired.")
            return []

        else:
            print(f"Semantic Scholar API error: {response.status_code}")
            return []

    except Exception as e:
        print(f"Semantic Scholar fetch failed: {e}")
        return []

    if papers:
        _save_to_cache(query, papers)

    return papers