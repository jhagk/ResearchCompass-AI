from graph.orchestrator import build_graph
from graph.synthesizer import synthesize_report
from report_generator import generate_pdf_report

graph = build_graph()

print("Starting ResearchCompass AI...")
final_state = graph.invoke({
    "domain": "Deep Learning in healthcare",
    "level": "M.Tech",
    "trends": None,
    "gaps": None,
    "suggested_topics": None,
    "source_papers": None,
    "novelty_report": None,
    "novelty_results": None,
    "refined_topics": None,
    "feasibility_report": None,
    "roadmap": None,
    "proposal_guide": None,
    "reading_list": None,
    "papers_used": None,
    "actual_papers": None,
    "final_report": None,
    "error": None
})

report = synthesize_report(final_state)
print(report)

pdf_path = generate_pdf_report(report, "NLP in Health care")
print(f"PDF saved: {pdf_path}")