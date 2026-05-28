from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, LLM_MODEL


def run_proposal_guide_generator(
    domain: str,
    level: str,
    refined_topics: str,
    research_gaps: str,
    feasibility_report: str,
    roadmap: str,
    actual_papers: list
) -> dict:
    """
    Generates a section-by-section research proposal
    writing guide for each refined topic.
    Student writes in their own words — zero AI plagiarism.
    """

    print(f"Proposal Guide Generator running for: {domain}")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.3
    )

    # build papers context from actual fetched papers
    papers_context = ""
    for i, paper in enumerate(actual_papers[:5]):
        papers_context += f"""
Paper {i+1}: {paper.get('title', '')}
Authors: {', '.join(paper.get('authors', []))}
URL: {paper.get('url', '')}
"""

    prompt = f"""
    You are a senior research supervisor helping an {level}
    student in India write their research proposal.

    Domain: {domain}

    Refined Topics:
    {refined_topics}

    Research Gaps Found:
    {research_gaps}

    Feasibility Report:
    {feasibility_report}

    Research Roadmap:
    {roadmap}

    Available Papers to Cite:
    {papers_context}

    Generate a PROPOSAL WRITING GUIDE for the TOP 2 topics only.

    For EACH topic generate a complete section-by-section
    writing guide using EXACTLY this format:

    ════════════════════════════════════════
    PROPOSAL GUIDE: [Topic Title]
    ════════════════════════════════════════

    ⚠️ IMPORTANT: This is a WRITING GUIDE only.
    Write every section in YOUR OWN WORDS.
    Do not copy this guide directly.

    ────────────────────────────────────────
    SECTION 1: TITLE
    ────────────────────────────────────────
    Suggested Title: [specific title]
    Alternative Title: [another option]

    ────────────────────────────────────────
    SECTION 2: ABSTRACT (150-200 words)
    ────────────────────────────────────────
    What to write:
    → Sentence 1-2: [specific problem to describe]
    → Sentence 3: [gap to mention]
    → Sentence 4-5: [your proposed solution]
    → Sentence 6: [expected outcome with metric]

    Key terms to include:
    - [term 1]
    - [term 2]
    - [term 3]

    ────────────────────────────────────────
    SECTION 3: INTRODUCTION (400-500 words)
    ────────────────────────────────────────
    Paragraph 1 — Background:
    → What to write: [specific background points]
    → Statistics to mention: [specific numbers]

    Paragraph 2 — Current Methods:
    → What to write: [existing approaches]
    → Methods to mention: [specific methods]

    Paragraph 3 — Why Methods Fail:
    → What to write: [specific failures]

    Paragraph 4 — Your Approach:
    → What to write: [your proposed method]

    Paragraph 5 — Objectives:
    → List 3-4 specific objectives

    Papers to cite in Introduction:
    [list specific papers from available papers]

    ────────────────────────────────────────
    SECTION 4: PROBLEM STATEMENT (200-300 words)
    ────────────────────────────────────────
    What to write:
    → Define exact problem in 1 sentence
    → Why it matters in India specifically
    → What is currently missing
    → What your research addresses

    Key phrases to use:
    - "The primary challenge is..."
    - "Existing approaches fail to..."
    - "This research addresses..."

    ────────────────────────────────────────
    SECTION 5: LITERATURE REVIEW (600-800 words)
    ────────────────────────────────────────
    For each available paper write:

    [Paper Title]:
    → What to discuss: [specific point]
    → Gap to highlight: [specific gap]

    Structure:
    → Para 1: [theme 1]
    → Para 2: [theme 2]
    → Para 3: [theme 3]
    → Para 4: Gap identification

    ────────────────────────────────────────
    SECTION 6: RESEARCH OBJECTIVES (100-150 words)
    ────────────────────────────────────────
    Write exactly 4 objectives:
    Objective 1: To develop [specific thing]
    Objective 2: To implement [specific method]
    Objective 3: To evaluate on [specific dataset]
                 using [specific metric]
    Objective 4: To compare with [specific baseline]

    ────────────────────────────────────────
    SECTION 7: METHODOLOGY (500-700 words)
    ────────────────────────────────────────
    Phase 1: Data Collection
    → Dataset: [specific dataset with source]
    → Preprocessing: [specific steps]
    → Split: [train/test ratio]

    Phase 2: Baseline
    → Model: [specific baseline model]
    → Framework: [specific framework]
    → Metrics: [specific metrics]

    Phase 3: Proposed Model
    → Architecture: [specific architecture]
    → Key innovation: [what is new]
    → Implementation steps: [numbered list]

    Phase 4: Evaluation
    → Comparison: [proposed vs baseline]
    → Ablation study: [what to ablate]

    Diagram to draw:
    → [describe system architecture diagram]

    ────────────────────────────────────────
    SECTION 8: EXPECTED OUTCOMES (100-150 words)
    ────────────────────────────────────────
    What to write:
    → Expected accuracy improvement: [specific target]
    → Expected contribution: [specific claim]
    → Publication venue: [specific journal]

    ────────────────────────────────────────
    SECTION 9: TIMELINE
    ────────────────────────────────────────
    [Use the roadmap provided above]

    ────────────────────────────────────────
    SECTION 10: REFERENCES
    ────────────────────────────────────────
    [List all available papers in IEEE format]

    Rules:
    - Be very specific — no generic advice
    - Every section must reference {domain}
    - Use India context wherever applicable
    - Mention specific datasets, models, metrics
    - Word counts must match university requirements
    - Papers listed must be from available papers only
    """

    response = llm.invoke(prompt)
    proposal_guide = response.content.strip()

    print("Proposal Guide Generator done.")
    return {
        "domain": domain,
        "level": level,
        "proposal_guide": proposal_guide
    }