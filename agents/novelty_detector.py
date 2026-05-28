from datetime import datetime
from langchain_openai import ChatOpenAI
from tools.scholar_tool import fetch_semantic_scholar_papers
from config import OPENAI_API_KEY, LLM_MODEL

CURRENT_YEAR = datetime.now().year


# ─────────────────────────────────────────
# Signal scoring functions
# ─────────────────────────────────────────

def paper_count_score(count: int) -> int:
    """
    More papers = less novel.
    Weight: 35%
    """
    if count < 20:
        return 100
    elif count < 50:
        return 80
    elif count < 100:
        return 60
    elif count < 300:
        return 40
    else:
        return 20


def citation_score(avg_citations: float) -> int:
    """
    Highly cited fields are usually more mature.
    Weight: 25%
    """
    if avg_citations < 5:
        return 100
    elif avg_citations < 20:
        return 80
    elif avg_citations < 50:
        return 60
    elif avg_citations < 100:
        return 40
    else:
        return 20


def growth_score(growth_rate: float) -> int:
    """
    Fast-growing topics are often emerging.
    Weight: 25%
    """
    if growth_rate > 0.50:
        return 100
    elif growth_rate > 0.30:
        return 80
    elif growth_rate > 0.15:
        return 60
    elif growth_rate > 0.05:
        return 40
    else:
        return 20


def recency_score(avg_year: float) -> int:
    """
    Recent publications contribute to novelty.
    Weight: 15%
    """
    age = CURRENT_YEAR - avg_year
    if age <= 1:
        return 100
    elif age <= 2:
        return 80
    elif age <= 3:
        return 60
    elif age <= 5:
        return 40
    else:
        return 20


# ─────────────────────────────────────────
# Research Maturity Classification
# ─────────────────────────────────────────

def get_maturity_label(novelty_score: int, stats: dict) -> dict:
    """
    Converts novelty score + stats into Research Maturity
    classification with visual indicator and advice.
    """

    paper_count = stats["paper_count"]
    avg_citations = stats["avg_citations"]

    # handle 0 papers case
    if paper_count == 0:
        return {
            "label": "Insufficient Data",
            "emoji": "⚪",
            "bar": "□□□□",
            "color": "gray",
            "action": "VERIFY TOPIC — Search returned no papers. "
                      "Try a broader or different search query "
                      "before assuming this is emerging.",
            "warning": "Zero papers found — this may indicate "
                       "a retrieval failure or the topic does "
                       "not exist yet. Verify manually."
        }

    if novelty_score >= 80:
        return {
            "label": "Emerging",
            "emoji": "🟢",
            "bar": "■□□□",
            "color": "green",
            "action": "PURSUE — Strong opportunity for new research",
            "warning": None
        }

    elif novelty_score >= 60:
        return {
            "label": "Growing",
            "emoji": "🟡",
            "bar": "■■□□",
            "color": "yellow",
            "action": "PURSUE WITH FOCUS — Add a specific angle "
                      "to differentiate",
            "warning": f"Active field with {paper_count} papers — "
                       f"needs a unique angle"
        }

    elif novelty_score >= 40:
        return {
            "label": "Mature",
            "emoji": "🟠",
            "bar": "■■■□",
            "color": "orange",
            "action": "PROCEED CAREFULLY — Must find very specific "
                      "niche or new application",
            "warning": f"{paper_count} papers with avg "
                       f"{avg_citations} citations — "
                       f"difficult to publish without novel angle"
        }

    else:
        return {
            "label": "Saturated",
            "emoji": "🔴",
            "bar": "■■■■",
            "color": "red",
            "action": "AVOID — Too many existing solutions. "
                      "Choose a more specific sub-problem",
            "warning": f"{paper_count}+ papers with avg "
                       f"{avg_citations} citations — "
                       f"very hard to publish new work here"
        }


# ─────────────────────────────────────────
# Statistics collection
# ─────────────────────────────────────────

def collect_topic_statistics(topic: str) -> dict:
    """
    Collects paper count, citations, growth rate,
    recency and year distribution for a given topic.
    Growth rate capped at 200% to avoid unrealistic values.
    """

    papers = fetch_semantic_scholar_papers(
        topic[:60], max_results=15
    )

    if not papers:
        return {
            "topic": topic,
            "paper_count": 0,
            "avg_citations": 0,
            "growth_rate": 0,
            "avg_year": CURRENT_YEAR,
            "year_distribution": {}
        }

    # citation stats
    citation_counts = [
        p.get("citation_count", 0) or 0
        for p in papers
    ]
    avg_citations = (
        sum(citation_counts) / len(citation_counts)
        if citation_counts else 0
    )

    # year distribution
    year_distribution = {}
    for paper in papers:
        year_str = str(paper.get("published", ""))
        if year_str and year_str.isdigit():
            year = int(year_str)
            if 2000 <= year <= CURRENT_YEAR:
                year_distribution[year] = (
                    year_distribution.get(year, 0) + 1
                )

    # average publication year
    years = list(year_distribution.keys())
    avg_year = (
        sum(y * year_distribution[y] for y in years) /
        sum(year_distribution.values())
        if years else CURRENT_YEAR
    )

    # publication growth rate
    recent = sum(
        year_distribution.get(CURRENT_YEAR - i, 0)
        for i in range(0, 2)
    )
    previous = sum(
        year_distribution.get(CURRENT_YEAR - i, 0)
        for i in range(2, 4)
    )

    if previous > 0:
        growth_rate = (recent - previous) / previous
    else:
        if recent > 0:
            growth_rate = 0.3
        else:
            growth_rate = 0.0

    # cap growth rate at 200% max and -100% min
    growth_rate = min(growth_rate, 2.0)
    growth_rate = max(growth_rate, -1.0)

    return {
        "topic": topic,
        "paper_count": len(papers),
        "avg_citations": round(avg_citations, 1),
        "growth_rate": round(growth_rate, 2),
        "avg_year": round(avg_year, 1),
        "year_distribution": year_distribution
    }


# ─────────────────────────────────────────
# Novelty computation
# ─────────────────────────────────────────

def compute_novelty_score(stats: dict) -> dict:
    """
    Computes weighted novelty score from 4 signals.
    """

    p_score = paper_count_score(stats["paper_count"])
    c_score = citation_score(stats["avg_citations"])
    g_score = growth_score(stats["growth_rate"])
    r_score = recency_score(stats["avg_year"])

    novelty_score = round(
        p_score * 0.35 +
        c_score * 0.25 +
        g_score * 0.25 +
        r_score * 0.15
    )

    maturity = get_maturity_label(novelty_score, stats)

    return {
        "novelty_score": novelty_score,
        "level": maturity["label"],
        "meaning": maturity["label"],
        "maturity": maturity,
        "signal_scores": {
            "paper_count_score": p_score,
            "citation_score": c_score,
            "growth_score": g_score,
            "recency_score": r_score
        }
    }


# ─────────────────────────────────────────
# Batch LLM novelty validation
# ─────────────────────────────────────────

def batch_llm_validate_novelty(
    topics_with_stats: list,
    domain: str,
    llm
) -> list:
    """
    Validates ALL topics in ONE LLM call.
    Much faster than calling LLM 5 times separately.
    Returns list of explanations for each topic.
    """

    topics_context = ""
    for i, item in enumerate(topics_with_stats):
        topic = item["topic"]
        stats = item["stats"]
        novelty = item["novelty"]
        growth_pct = round(stats["growth_rate"] * 100, 1)
        maturity = novelty["maturity"]

        topics_context += f"""
Topic {i+1}: {topic}
- Papers found: {stats['paper_count']}
- Avg citations: {stats['avg_citations']}
- Growth rate: {growth_pct}%
- Avg year: {stats['avg_year']}
- Novelty score: {novelty['novelty_score']}/100
- Maturity: {maturity['label']}
"""

    prompt = f"""
    You are a research advisor for {domain}.

    Analyze these {len(topics_with_stats)} research topics
    and their statistics:

    {topics_context}

    For EACH topic write exactly 2 sentences:
    1. Why this topic has this maturity level
       (mention paper count and citations specifically)
    2. One specific recommendation for the student

    Format:
    Topic 1: [2 sentences]
    Topic 2: [2 sentences]
    Topic 3: [2 sentences]
    Topic 4: [2 sentences]
    Topic 5: [2 sentences]

    Rules:
    - Be direct and honest
    - If Saturated/Mature: tell them clearly
    - If Emerging: encourage strongly
    - Mention specific numbers
    - Simple language for M.Tech students
    - Keep each response to exactly 2 sentences
    """

    try:
        response = llm.invoke(prompt)
        raw_text = response.content.strip()

        # parse responses for each topic
        explanations = []
        lines = raw_text.split("\n")
        current_topic = -1
        current_text = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # detect topic header
            for i in range(len(topics_with_stats)):
                if line.startswith(f"Topic {i+1}:"):
                    if current_topic >= 0 and current_text:
                        explanations.append(current_text.strip())
                    current_topic = i
                    current_text = line.replace(
                        f"Topic {i+1}:", ""
                    ).strip()
                    break
            else:
                if current_topic >= 0:
                    current_text += " " + line

        # add last topic
        if current_text:
            explanations.append(current_text.strip())

        # ensure we have explanation for each topic
        while len(explanations) < len(topics_with_stats):
            explanations.append(
                "Analysis not available."
            )

        return explanations[:len(topics_with_stats)]

    except Exception as e:
        print(f"Batch LLM validation failed: {e}")
        return [
            f"Novelty score: "
            f"{item['novelty']['novelty_score']}/100 — "
            f"{item['novelty']['maturity']['label']}"
            for item in topics_with_stats
        ]


# ─────────────────────────────────────────
# Main agent — optimized with batch processing
# ─────────────────────────────────────────

def run_novelty_detector(domain: str, topics_text: str) -> dict:
    """
    Analyzes each suggested topic for novelty and
    research maturity using 4-signal weighted scoring.

    OPTIMIZED:
    - Batch LLM validation (1 call instead of 5)
    - Faster overall execution
    """

    print(f"Novelty Detector running for domain: {domain}")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

    # Step 1 - extract topics
    extract_prompt = f"""
    Extract only the topic titles from this text.
    Return one topic per line, no numbering, no explanation.
    Text: {topics_text}
    """

    response = llm.invoke(extract_prompt)
    raw_topics = [
        t.strip() for t in
        response.content.strip().split("\n")
        if t.strip() and len(t.strip()) > 10
    ][:5]

    print(f"Extracted topics: {raw_topics}")

    # Step 2 - collect statistics for each topic
    # (Semantic Scholar calls — unavoidable but cached)
    topics_with_stats = []

    for topic in raw_topics:
        print(f"Collecting stats for: {topic[:60]}...")
        stats = collect_topic_statistics(topic)
        novelty = compute_novelty_score(stats)

        maturity = novelty["maturity"]
        growth_pct = round(stats["growth_rate"] * 100, 1)

        print(
            f"  Score: {novelty['novelty_score']}/100 — "
            f"{maturity['emoji']} {maturity['label']}"
        )
        print(
            f"  Papers: {stats['paper_count']} | "
            f"Citations: {stats['avg_citations']} | "
            f"Growth: {growth_pct}%"
        )

        topics_with_stats.append({
            "topic": topic,
            "stats": stats,
            "novelty": novelty
        })

    # Step 3 - batch LLM validation
    # ONE LLM call for all 5 topics instead of 5 separate calls
    print("Running batch LLM validation (1 call for all topics)...")
    explanations = batch_llm_validate_novelty(
        topics_with_stats, domain, llm
    )

    # Step 4 - build final results
    novelty_results = []

    for i, item in enumerate(topics_with_stats):
        topic = item["topic"]
        stats = item["stats"]
        novelty = item["novelty"]
        maturity = novelty["maturity"]
        growth_pct = round(stats["growth_rate"] * 100, 1)
        explanation = explanations[i] if i < len(explanations) \
            else "Analysis not available."

        novelty_results.append({
            "topic": topic,
            "novelty_score": novelty["novelty_score"],
            "level": maturity["label"],
            "maturity": maturity,
            "paper_count": stats["paper_count"],
            "avg_citations": stats["avg_citations"],
            "growth_rate": growth_pct,
            "avg_year": stats["avg_year"],
            "year_distribution": stats["year_distribution"],
            "signal_scores": novelty["signal_scores"],
            "explanation": explanation
        })

    # Step 5 - format report
    novelty_summary = format_novelty_report(novelty_results)

    print(f"\nNovelty Detector done.")
    return {
        "domain": domain,
        "novelty_results": novelty_results,
        "novelty_summary": novelty_summary
    }


def format_novelty_report(results: list) -> str:
    """
    Formats novelty results with Research Maturity display.
    """

    report = ""

    for i, r in enumerate(results):
        maturity = r["maturity"]

        report += f"""
### Topic {i+1}: {r['topic']}

**Novelty Score: {r['novelty_score']}/100**

**Research Maturity:**
{maturity['emoji']} {maturity['bar']} {maturity['label'].upper()}

**Evidence:**
- Papers Found    : {r['paper_count']}
- Avg Citations   : {r['avg_citations']}
- Growth Rate     : +{r['growth_rate']}%
- Avg Pub Year    : {r['avg_year']}

**Signal Breakdown:**
- Paper Count Score : {r['signal_scores']['paper_count_score']}/100
- Citation Score    : {r['signal_scores']['citation_score']}/100
- Growth Score      : {r['signal_scores']['growth_score']}/100
- Recency Score     : {r['signal_scores']['recency_score']}/100

**Action: {maturity['action']}**
"""

        if maturity.get("warning"):
            report += f"""
**Warning:** {maturity['warning']}
"""

        report += f"""
**Analysis:**
{r['explanation']}

---
"""

    return report.strip()