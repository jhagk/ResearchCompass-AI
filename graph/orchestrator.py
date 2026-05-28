from langgraph.graph import StateGraph, END
from graph.state import ResearchState
from agents.trend_scanner import run_trend_scanner
from agents.gap_finder import run_gap_finder
from agents.topic_suggester import run_topic_suggester
from agents.novelty_detector import run_novelty_detector
from agents.topic_refiner_agent import run_topic_refiner
from agents.feasibility_checker import run_feasibility_checker
from agents.roadmap_generator import run_roadmap_generator
from agents.citation_scout import run_citation_scout


def trend_scanner_node(state: ResearchState) -> dict:
    print("\n--- Running Trend Scanner ---")
    result = run_trend_scanner(state["domain"])
    # return ONLY changed keys
    return {
        "trends": result["trends"],
        "papers_used": result["papers_used"]
    }


def gap_finder_node(state: ResearchState) -> dict:
    print("\n--- Running Gap Finder ---")
    result = run_gap_finder(state["domain"])
    # return ONLY changed keys
    return {
        "gaps": result["gaps"]
    }


def topic_suggester_node(state: ResearchState) -> dict:
    print("\n--- Running Topic Suggester ---")
    result = run_topic_suggester(
        state["domain"],
        state["level"]
    )
    # return ONLY changed keys
    return {
        "suggested_topics": result["suggested_topics"],
        "source_papers": result.get("source_papers", [])
    }


def novelty_detector_node(state: ResearchState) -> dict:
    print("\n--- Running Novelty Detector ---")
    result = run_novelty_detector(
        domain=state["domain"],
        topics_text=state["suggested_topics"]
    )
    # return ONLY changed keys
    return {
        "novelty_report": result["novelty_summary"],
        "novelty_results": result["novelty_results"]
    }


def topic_refiner_node(state: ResearchState) -> dict:
    print("\n--- Running Topic Refiner ---")
    result = run_topic_refiner(
        domain=state["domain"],
        topics=state["suggested_topics"],
        level=state["level"]
    )
    # return ONLY changed keys
    return {
        "refined_topics": result["refined_topics"]
    }


def feasibility_checker_node(state: ResearchState) -> dict:
    print("\n--- Running Feasibility Checker ---")
    result = run_feasibility_checker(
        domain=state["domain"],
        topics=state["refined_topics"],
        level=state["level"]
    )
    # return ONLY changed keys
    return {
        "feasibility_report": result["feasibility_report"]
    }


def roadmap_generator_node(state: ResearchState) -> dict:
    print("\n--- Running Roadmap Generator ---")
    result = run_roadmap_generator(
        domain=state["domain"],
        refined_topics=state["refined_topics"],
        feasibility_report=state.get(
            "feasibility_report", ""
        ),
        level=state["level"]
    )
    # return ONLY changed keys
    return {
        "roadmap": result["roadmap"]
    }


def citation_scout_node(state: ResearchState) -> dict:
    print("\n--- Running Citation Scout ---")
    result = run_citation_scout(
        domain=state["domain"],
        topics=state["refined_topics"]
    )
    # return ONLY changed keys
    return {
        "reading_list": result["reading_list"],
        "actual_papers": result["actual_papers"]
    }


def build_graph():
    graph = StateGraph(ResearchState)

    # add all nodes
    graph.add_node("trend_scanner", trend_scanner_node)
    graph.add_node("gap_finder", gap_finder_node)
    graph.add_node("topic_suggester", topic_suggester_node)
    graph.add_node("novelty_detector", novelty_detector_node)
    graph.add_node("topic_refiner", topic_refiner_node)
    graph.add_node("feasibility_checker",
                   feasibility_checker_node)
    graph.add_node("roadmap_generator", roadmap_generator_node)
    graph.add_node("citation_scout", citation_scout_node)

    # sequential start
    graph.set_entry_point("trend_scanner")
    graph.add_edge("trend_scanner", "gap_finder")
    graph.add_edge("gap_finder", "topic_suggester")

    # PARALLEL GROUP 1
    # novelty_detector + topic_refiner run together
    graph.add_edge("topic_suggester", "novelty_detector")
    graph.add_edge("topic_suggester", "topic_refiner")

    # both merge at feasibility_checker
    graph.add_edge("novelty_detector", "feasibility_checker")
    graph.add_edge("topic_refiner", "feasibility_checker")

    # PARALLEL GROUP 2
    # roadmap + citation_scout run together
    graph.add_edge("feasibility_checker", "roadmap_generator")
    graph.add_edge("feasibility_checker", "citation_scout")

    # both end
    graph.add_edge("roadmap_generator", END)
    graph.add_edge("citation_scout", END)

    return graph.compile()