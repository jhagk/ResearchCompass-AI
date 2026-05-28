from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, LLM_MODEL


def run_roadmap_generator(
    domain: str,
    refined_topics: str,
    feasibility_report: str,
    level: str = "M.Tech"
) -> dict:
    """
    Generates a month-by-month research roadmap
    for each refined topic based on feasibility report.
    """

    print(f"Roadmap Generator running for domain: {domain}")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.3
    )

    prompt = f"""
    You are a senior research advisor helping an {level} student
    in India plan their research journey.

    Domain: {domain}

    Refined Research Topics:
    {refined_topics}

    Feasibility Report:
    {feasibility_report}

    Generate a detailed month-by-month research roadmap
    for EACH topic.

    For each topic use EXACTLY this format:

    ### Topic [number]: [Topic Title]

    **Total Duration:** [X months]
    **Difficulty:** [Easy/Medium/Hard]

    | Month | Phase | Tasks | Deliverable |
    |-------|-------|-------|-------------|
    | 1 | Literature Review | Read 20 key papers, identify gaps, summarize findings | Literature review document |
    | 2 | Dataset Collection | Download datasets, preprocess, split train/test | Clean dataset ready |
    | 3 | Baseline Implementation | Implement baseline model, run experiments | Baseline results |
    | 4 | Model Development | Implement proposed method, tune hyperparameters | Improved model |
    | 5 | Evaluation | Compare with baseline, run ablation study | Evaluation report |
    | 6 | Paper Writing | Write paper, submit to venue | Draft paper |

    Also include:

    **Key Milestones:**
    - Month X: [specific milestone]
    - Month Y: [specific milestone]

    **Tools and Resources Needed:**
    - Framework: [specific framework]
    - Dataset: [specific dataset]
    - Hardware: [specific hardware]
    - Key papers to read first: [2-3 specific papers]

    **Risks and Mitigation:**
    - Risk 1: [specific risk] → Mitigation: [solution]
    - Risk 2: [specific risk] → Mitigation: [solution]

    Rules:
    - Timeline must match feasibility report
    - Tasks must be specific to {domain}
    - Each phase must have clear deliverable
    - Must be realistic for {level} student in India
    - Mention specific tools, datasets, frameworks
    - Use simple language
    """

    response = llm.invoke(prompt)
    roadmap = response.content.strip()

    print("Roadmap Generator done.")
    return {
        "domain": domain,
        "level": level,
        "roadmap": roadmap
    }