import time
import requests
import json
import os
from config import CACHE_PATH


def _get_cache_path(query: str) -> str:
    safe_name = query.replace(" ", "_").replace("/", "_")[:50]
    return os.path.join(CACHE_PATH, f"openalex_{safe_name}.json")


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


def reconstruct_abstract(inverted_index: dict) -> str:
    """
    OpenAlex stores abstracts as inverted index.
    This function reconstructs the original abstract text.
    """

    if not inverted_index:
        return ""

    max_pos = 0
    for positions in inverted_index.values():
        for pos in positions:
            if pos > max_pos:
                max_pos = pos

    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word

    return " ".join(words).strip()


def fetch_openalex_papers(query: str, max_results: int = 10) -> list:
    """
    Fetch papers from OpenAlex API.
    200M+ papers across ALL domains.
    No API key needed.
    Uses relevance filtering to ensure quality results.
    """

    cached = _load_from_cache(query)
    if cached:
        print(f"Loaded {len(cached)} OpenAlex papers from cache.")
        return cached

    url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "per-page": max_results * 2,
        "sort": "publication_date:desc",
        "select": "title,abstract_inverted_index,authorships,"
                  "publication_year,doi,concepts,open_access"
    }

    headers = {
        "User-Agent": "ResearchCompassAI/1.0 (mailto:research@example.com)"
    }

    papers = []

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15
        )

        if response.status_code == 200:
            data = response.json()
            raw_papers = data.get("results", [])

            for paper in raw_papers:

                # reconstruct abstract
                abstract = reconstruct_abstract(
                    paper.get("abstract_inverted_index", {})
                )

                if not abstract:
                    continue

                # relevance filter
                query_words = query.lower().split()
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
                    threshold = 0.5
                elif len(query_words) <= 4:
                    threshold = 0.7
                else:
                    threshold = 0.8

                if relevance < threshold:
                    continue

                # get authors
                authors = []
                for authorship in paper.get("authorships", [])[:3]:
                    author = authorship.get("author", {})
                    if author.get("display_name"):
                        authors.append(author["display_name"])

                # get URL
                doi = paper.get("doi", "")
                url_link = doi if doi else ""

                # get concepts
                concepts = [
                    c.get("display_name", "")
                    for c in paper.get("concepts", [])[:5]
                ]

                year = str(paper.get("publication_year", ""))

                papers.append({
                    "title": paper.get("title", ""),
                    "abstract": abstract,
                    "authors": authors,
                    "url": url_link,
                    "published": year,
                    "categories": concepts,
                    "source": "openalex"
                })

                if len(papers) >= max_results:
                    break

            print(f"Fetched {len(papers)} relevant papers from OpenAlex.")

        elif response.status_code == 429:
            print("OpenAlex rate limited. Waiting 5 seconds...")
            time.sleep(5)
            return []

        else:
            print(f"OpenAlex API error: {response.status_code}")
            return []

    except Exception as e:
        print(f"OpenAlex fetch failed: {e}")
        return []

    _save_to_cache(query, papers)
    return papers