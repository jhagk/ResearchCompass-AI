from langchain.tools import tool
from tavily import TavilyClient
from config import TAVILY_API_KEY


@tool
def web_search(query: str) -> list:
    """
    Search the web using Tavily for recent trends,
    blogs, and news related to a research topic.
    """

    results = []

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5
        )

        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")
            })

        print(f"Web search returned {len(results)} results.")

    except Exception as e:
        print(f"Web search failed: {e}")
        return []

    return results