from graph.orchestrator import build_graph
from graph.synthesizer import synthesize_report
from report_generator import generate_pdf_report

# Step 1 - run graph
graph = build_graph()
final_state = graph.invoke({
    "domain": "natural language processing",
    "level": "M.Tech",
    "trends": None,
    "gaps": None,
    "suggested_topics": None,
    "feasibility_report": None,
    "reading_list": None,
    "papers_used": None,
    "actual_papers": None,
    "final_report": None,
    "error": None
})

# Step 2 - synthesize
report = synthesize_report(final_state)

# Step 3 - generate PDF
pdf_path = generate_pdf_report(report, "natural language processing")
print(f"PDF saved at: {pdf_path}")