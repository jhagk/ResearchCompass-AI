import re
import streamlit as st
import time
from graph.orchestrator import build_graph
from graph.synthesizer import synthesize_report
from report_generator import generate_pdf_report

import os

# Create required directories
os.makedirs("data/cache", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)
os.makedirs("data/faiss_index", exist_ok=True)
def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


# page config
st.set_page_config(
    page_title="ResearchCompass AI",
    page_icon="🧭",
    layout="wide"
)

# custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a4a6a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1a1a2e;
    }
    .proposal-card {
        background: #f0f7ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1565c0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1a1a2e;
        color: white;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton > button:hover {
        background-color: #16213e;
    }
    .score-badge {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# header
st.markdown(
    '<p class="main-header">🧭 ResearchCompass AI</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Find your perfect research topic using AI '
    '— powered by ArXiv, OpenAlex and Semantic Scholar</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# sidebar
with st.sidebar:
    st.header("Settings")

    level = st.selectbox(
        "Select your academic level",
        ["B.Tech", "M.Tech", "PhD"],
        index=1
    )

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Enter your research domain")
    st.markdown("2. Select your academic level")
    st.markdown("3. Click Generate Report")
    st.markdown("4. Explore results in 10 tabs")
    st.markdown("5. Select topic for proposal guide")
    st.markdown("6. Download PDF report")

    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("- ArXiv")
    st.markdown("- OpenAlex")
    st.markdown("- Semantic Scholar")

    st.markdown("---")
    st.markdown("### Agents")
    st.markdown("- Trend Scanner")
    st.markdown("- Gap Finder")
    st.markdown("- Topic Suggester")
    st.markdown("- Novelty Detector")
    st.markdown("- Topic Refiner")
    st.markdown("- Feasibility Checker")
    st.markdown("- Roadmap Generator")
    st.markdown("- Citation Scout")
    st.markdown("- Proposal Guide (on demand)")

    st.markdown("---")
    st.markdown("### Example Domains")
    st.markdown("- Natural Language Processing")
    st.markdown("- Deep Learning in Healthcare")
    st.markdown("- AI with Blockchain")
    st.markdown("- Computer Vision")
    st.markdown("- NLP for Indian Languages")
    st.markdown("- AI with IoT")

# main input
col1, col2 = st.columns([3, 1])

with col1:
    domain = st.text_input(
        "Enter your research domain",
        placeholder="e.g. Deep Learning in healthcare, "
                    "NLP for Indian languages, AI with IoT"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button(
        "Generate Report",
        type="primary",
        use_container_width=True
    )

# generate report
if generate_btn:
    if not domain:
        st.error("Please enter a research domain first!")
    else:
        st.markdown("---")

        # progress section
        st.markdown("### Generating your research report...")
        progress = st.progress(0)
        status = st.empty()

        try:
            # build graph
            status.info("Initializing ResearchCompass AI...")
            progress.progress(5)
            graph = build_graph()
            time.sleep(0.5)

            status.info(
                "Running Trend Scanner — fetching latest papers..."
            )
            progress.progress(10)

            status.info(
                "Running Gap Finder — analyzing research gaps..."
            )
            progress.progress(18)

            status.info(
                "Running Topic Suggester — generating topics..."
            )
            progress.progress(26)

            status.info(
                "Running Novelty Detector — checking saturation..."
            )
            progress.progress(36)

            status.info(
                "Running Topic Refiner — publication-style titles..."
            )
            progress.progress(46)

            status.info(
                "Running Feasibility Checker..."
            )
            progress.progress(55)

            status.info(
                "Running Roadmap Generator — building timeline..."
            )
            progress.progress(64)

            status.info(
                "Running Citation Scout — finding key papers..."
            )
            progress.progress(75)

            # invoke graph
            final_state = graph.invoke({
                "domain": domain,
                "level": level,
                "trends": None,
                "gaps": None,
                "suggested_topics": None,
                "source_papers": None,
                "novelty_report": None,
                "novelty_results": None,
                "refined_topics": None,
                "feasibility_report": None,
                "roadmap": None,
                "reading_list": None,
                "papers_used": None,
                "actual_papers": None,
                "final_report": None,
                "error": None
            })

            # synthesize report
            status.info("Generating final report...")
            progress.progress(90)
            report = synthesize_report(final_state)

            # generate PDF
            status.info("Saving PDF report...")
            pdf_path = generate_pdf_report(
                report,
                domain,
                novelty_results=final_state.get(
                    "novelty_results", []
                )
            )
            progress.progress(100)
            status.success("Report generated successfully!")

            st.markdown("---")

            # metrics row
            papers = final_state.get("actual_papers", [])
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Domain", domain.title()[:20])
            with col2:
                st.metric("Academic Level", level)
            with col3:
                st.metric("Papers Analyzed", len(papers))
            with col4:
                st.metric("Topics Suggested", "5")

            st.markdown("---")

            # results in 10 tabs
            tab1, tab2, tab3, tab4, tab5, \
                tab6, tab7, tab8, tab9, tab10 = st.tabs([
                    "Trending Topics",
                    "Research Gaps",
                    "Suggested Topics",
                    "Novelty Analysis",
                    "Refined Topics",
                    "Feasibility",
                    "Roadmap",
                    "Reading List",
                    "Key Papers",
                    "Proposal Guide"
                ])

            with tab1:
                st.subheader("Trending Research Topics")
                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )
                st.write(
                    final_state.get("trends", "Not available")
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.subheader("Research Gaps")
                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )
                st.write(
                    final_state.get("gaps", "Not available")
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                st.subheader("Suggested Research Topics")
                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )
                st.write(
                    final_state.get(
                        "suggested_topics", "Not available"
                    )
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # evidence trail — deduplicated
                source_papers = final_state.get(
                    "source_papers", []
                )
                if source_papers:
                    st.markdown("#### Evidence Trail")
                    st.markdown(
                        "Papers that inspired these topics:"
                    )
                    seen_ids = set()
                    count = 0
                    for paper in source_papers:
                        url = paper.get("url", "").strip()
                        title = paper.get("title", "Unknown")
                        paper_id = (
                            url or normalize_title(title)
                        )
                        if paper_id and paper_id not in seen_ids:
                            seen_ids.add(paper_id)
                            count += 1
                            if url:
                                st.markdown(
                                    f"{count}. [{title}]({url})"
                                )
                            else:
                                st.markdown(f"{count}. {title}")

            with tab4:
                st.subheader("Novelty Analysis")
                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )
                st.write(
                    final_state.get(
                        "novelty_report", "Not available"
                    )
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # trend graphs
                novelty_results = final_state.get(
                    "novelty_results", []
                )
                if novelty_results:
                    st.markdown("---")
                    st.subheader("Publication Trend Charts")

                    try:
                        import plotly.graph_objects as go

                        for r in novelty_results:
                            year_dist = r.get(
                                "year_distribution", {}
                            )
                            if not year_dist:
                                continue

                            years = sorted(year_dist.keys())
                            counts = [
                                year_dist[y] for y in years
                            ]

                            if len(years) < 2:
                                continue

                            maturity = r.get("maturity", {})
                            color_map = {
                                "Emerging": "#4caf50",
                                "Growing": "#ff9800",
                                "Mature": "#ff5722",
                                "Saturated": "#f44336",
                                "Insufficient Data": "#9e9e9e"
                            }
                            bar_color = color_map.get(
                                r.get("level", "Growing"),
                                "#2196f3"
                            )

                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=[str(y) for y in years],
                                y=counts,
                                marker_color=bar_color,
                                text=counts,
                                textposition="outside",
                                name="Papers"
                            ))
                            fig.add_trace(go.Scatter(
                                x=[str(y) for y in years],
                                y=counts,
                                mode="lines+markers",
                                line=dict(
                                    color="#1a1a2e",
                                    width=2
                                ),
                                marker=dict(size=6),
                                name="Trend"
                            ))

                            emoji = maturity.get("emoji", "")
                            level_label = r.get("level", "")

                            fig.update_layout(
                                title=dict(
                                    text=f"{emoji} "
                                         f"{r['topic'][:55]}... "
                                         f"— {level_label}",
                                    font=dict(size=13)
                                ),
                                xaxis_title="Year",
                                yaxis_title="Papers Published",
                                height=300,
                                margin=dict(
                                    t=50, b=40, l=40, r=20
                                ),
                                showlegend=False,
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                xaxis=dict(
                                    showgrid=True,
                                    gridcolor="#eeeeee"
                                ),
                                yaxis=dict(
                                    showgrid=True,
                                    gridcolor="#eeeeee"
                                )
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                            score = r.get("novelty_score", 0)
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric(
                                    "Novelty Score",
                                    f"{score}/100"
                                )
                            with col_b:
                                st.metric(
                                    "Papers Found",
                                    r.get("paper_count", 0)
                                )
                            with col_c:
                                st.metric(
                                    "Avg Citations",
                                    r.get("avg_citations", 0)
                                )
                            st.markdown("---")

                    except ImportError:
                        st.info(
                            "Install plotly: pip install plotly"
                        )

            with tab5:
                st.subheader("Refined Publication-Style Topics")
                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )
                st.write(
                    final_state.get(
                        "refined_topics", "Not available"
                    )
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with tab6:
                st.subheader("Feasibility Report")
                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )
                st.write(
                    final_state.get(
                        "feasibility_report", "Not available"
                    )
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with tab7:
                st.subheader("Research Roadmap")
                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )
                st.write(
                    final_state.get("roadmap", "Not available")
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with tab8:
                st.subheader("Starter Reading List")
                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )
                st.write(
                    final_state.get(
                        "reading_list", "Not available"
                    )
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with tab9:
                st.subheader("Key Papers")
                papers = final_state.get("actual_papers", [])
                if papers:
                    for i, paper in enumerate(papers):
                        score = paper.get("relevance_score", 0)
                        title = paper["title"]
                        with st.expander(
                            f"Paper {i+1}: {title}"
                        ):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(
                                    f"**Authors:** "
                                    f"{', '.join(paper['authors'])}"
                                )
                                st.write(
                                    f"**Published:** "
                                    f"{paper['published']}"
                                )
                            with col_b:
                                st.write(
                                    f"**Source:** "
                                    f"{paper.get('source', 'arxiv').upper()}"
                                )
                                if score:
                                    st.markdown(
                                        f"**Relevance:** "
                                        f'<span class="score-badge">'
                                        f"{score}/100</span>",
                                        unsafe_allow_html=True
                                    )
                                if paper.get("url"):
                                    st.markdown(
                                        f"**URL:** "
                                        f"[Open Paper]"
                                        f"({paper['url']})"
                                    )
                else:
                    st.write("No papers available.")

            # ─────────────────────────────────────────
            # Tab 10 — Proposal Guide (On Demand)
            # ─────────────────────────────────────────
            with tab10:
                st.subheader("Research Proposal Writing Guide")
                st.warning(
                    "⚠️ Write every section in YOUR OWN WORDS. "
                    "This guide tells you WHAT to write — "
                    "not the text itself. "
                    "Zero AI plagiarism when you follow this guide."
                )

                st.markdown(
                    "### Select a topic to generate proposal guide:"
                )

                # topic dropdown
                topic_options = [
                    "Topic 1",
                    "Topic 2",
                    "Topic 3",
                    "Topic 4",
                    "Topic 5"
                ]

                selected_topic = st.selectbox(
                    "Which topic do you want the "
                    "proposal guide for?",
                    topic_options,
                    index=0
                )

                generate_proposal_btn = st.button(
                    f"Generate Proposal Guide "
                    f"for {selected_topic}",
                    type="primary"
                )

                if generate_proposal_btn:
                    with st.spinner(
                        f"Generating proposal guide "
                        f"for {selected_topic}... "
                        f"(takes 30-60 seconds)"
                    ):
                        from agents.proposal_guide_generator\
                            import run_proposal_guide_generator
                        from langchain_openai import ChatOpenAI
                        from config import (
                            OPENAI_API_KEY, LLM_MODEL
                        )

                        # extract topic number
                        topic_number = int(
                            selected_topic.replace("Topic ", "")
                        )

                        # extract specific topic text
                        llm = ChatOpenAI(
                            model=LLM_MODEL,
                            openai_api_key=OPENAI_API_KEY,
                            temperature=0
                        )

                        extract_prompt = f"""
                        From the text below extract ONLY
                        Topic {topic_number} completely
                        including all its details.
                        Return nothing else.

                        Text:
                        {final_state.get('refined_topics', '')}
                        """

                        response = llm.invoke(extract_prompt)
                        selected_topic_text = (
                            response.content.strip()
                        )

                        # generate proposal for selected topic
                        result = run_proposal_guide_generator(
                            domain=domain,
                            level=level,
                            refined_topics=selected_topic_text,
                            research_gaps=final_state.get(
                                "gaps", ""
                            ),
                            feasibility_report=final_state.get(
                                "feasibility_report", ""
                            ),
                            roadmap=final_state.get(
                                "roadmap", ""
                            ),
                            actual_papers=final_state.get(
                                "actual_papers", []
                            )
                        )

                        st.success(
                            f"Proposal guide for "
                            f"{selected_topic} generated!"
                        )

                        st.markdown(
                            '<div class="proposal-card">',
                            unsafe_allow_html=True
                        )
                        st.write(result["proposal_guide"])
                        st.markdown(
                            '</div>',
                            unsafe_allow_html=True
                        )

            # download button
            st.markdown("---")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f,
                    file_name=f"ResearchCompass_"
                              f"{domain.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        except Exception as e:
            status.error(f"Error occurred: {str(e)}")
            st.exception(e)