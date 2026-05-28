import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, Image
)
from reportlab.lib.colors import HexColor
from config import REPORTS_PATH


def create_trend_chart(
    topic: str,
    year_distribution: dict,
    level: str
) -> BytesIO:
    """
    Creates a matplotlib bar chart for publication trends.
    Returns as BytesIO for embedding in PDF.
    """

    if not year_distribution:
        return None

    years = sorted(year_distribution.keys())
    counts = [year_distribution[y] for y in years]

    color_map = {
        "Emerging": "#4caf50",
        "Growing": "#ff9800",
        "Mature": "#ff5722",
        "Saturated": "#f44336"
    }
    bar_color = color_map.get(level, "#2196f3")

    fig, ax = plt.subplots(figsize=(7, 2.5))

    bars = ax.bar(
        [str(y) for y in years],
        counts,
        color=bar_color,
        alpha=0.8,
        edgecolor="white"
    )

    # add trend line
    ax.plot(
        [str(y) for y in years],
        counts,
        color="#1a1a2e",
        linewidth=1.5,
        marker="o",
        markersize=4
    )

    # add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333"
        )

    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Papers", fontsize=9)
    ax.set_title(
        f"{topic[:55]}... — {level}",
        fontsize=9,
        pad=8
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(
    report_text: str,
    domain: str,
    novelty_results: list = None
) -> str:
    """
    Converts final report text into PDF.
    Embeds trend charts if novelty_results provided.
    """

    os.makedirs(REPORTS_PATH, exist_ok=True)

    safe_domain = domain.replace(" ", "_").replace("/", "_")
    filename = f"ResearchCompass_{safe_domain}.pdf"
    filepath = os.path.join(REPORTS_PATH, filename)

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "title",
        parent=styles["Normal"],
        fontSize=20,
        textColor=HexColor("#1a1a2e"),
        spaceAfter=6,
        fontName="Helvetica-Bold",
        alignment=1
    )

    subtitle_style = ParagraphStyle(
        "subtitle",
        parent=styles["Normal"],
        fontSize=12,
        textColor=HexColor("#4a4a6a"),
        spaceAfter=20,
        fontName="Helvetica",
        alignment=1
    )

    section_style = ParagraphStyle(
        "section",
        parent=styles["Normal"],
        fontSize=13,
        textColor=HexColor("#16213e"),
        spaceBefore=16,
        spaceAfter=8,
        fontName="Helvetica-Bold"
    )

    body_style = ParagraphStyle(
        "body",
        parent=styles["Normal"],
        fontSize=10,
        textColor=HexColor("#2d2d2d"),
        spaceAfter=6,
        fontName="Helvetica",
        leading=16
    )

    chart_title_style = ParagraphStyle(
        "chart_title",
        parent=styles["Normal"],
        fontSize=11,
        textColor=HexColor("#1a1a2e"),
        spaceBefore=12,
        spaceAfter=4,
        fontName="Helvetica-Bold"
    )

    content = []

    # title
    content.append(Spacer(1, 0.2 * inch))
    content.append(Paragraph("ResearchCompass AI", title_style))
    content.append(
        Paragraph(
            f"Research Report - {domain.title()}",
            subtitle_style
        )
    )
    content.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=HexColor("#1a1a2e")
        )
    )
    content.append(Spacer(1, 0.2 * inch))

    # track if we are in novelty section
    in_novelty_section = False
    novelty_chart_added = set()

    # parse report text
    lines = report_text.split("\n")
    for line in lines:
        line = line.strip()

        if not line:
            content.append(Spacer(1, 0.08 * inch))
            continue

        if line.startswith("===") or line.startswith("---"):
            continue

        # detect novelty section
        if "NOVELTY ANALYSIS" in line.upper():
            in_novelty_section = True
        elif any(
            section in line.upper() for section in [
                "REFINED", "FEASIBILITY", "ROADMAP",
                "READING LIST", "KEY PAPERS"
            ]
        ):
            in_novelty_section = False

        # section headers
        if (line.startswith("1.") or line.startswith("2.") or
                line.startswith("3.") or line.startswith("4.") or
                line.startswith("5.") or line.startswith("6.") or
                line.startswith("7.") or line.startswith("8.") or
                line.startswith("9.")):
            content.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=HexColor("#cccccc")
                )
            )
            content.append(Paragraph(line, section_style))

            # insert trend charts after novelty section header
            if (in_novelty_section and
                    novelty_results and
                    "NOVELTY" in line.upper()):
                content.append(Spacer(1, 0.1 * inch))
                content.append(
                    Paragraph(
                        "Publication Trend Charts:",
                        chart_title_style
                    )
                )
                for r in novelty_results:
                    year_dist = r.get("year_distribution", {})
                    topic = r.get("topic", "")
                    level_label = r.get("level", "Growing")

                    if (topic not in novelty_chart_added
                            and len(year_dist) >= 2):
                        novelty_chart_added.add(topic)
                        buf = create_trend_chart(
                            topic,
                            year_dist,
                            level_label
                        )
                        if buf:
                            img = Image(buf)
                            img.drawWidth = 6.5 * inch
                            img.drawHeight = 2.5 * inch
                            content.append(img)
                            content.append(
                                Spacer(1, 0.1 * inch)
                            )

        elif (line.startswith("Paper") or
              line.startswith("Authors") or
              line.startswith("Published") or
              line.startswith("URL") or
              line.startswith("Source") or
              line.startswith("Score")):
            content.append(Paragraph(line, body_style))

        else:
            line = line.replace("**", "")
            line = line.replace("###", "").strip()
            line = line.replace("##", "").strip()
            if line:
                content.append(Paragraph(line, body_style))

    # footer
    content.append(Spacer(1, 0.3 * inch))
    content.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=HexColor("#1a1a2e")
        )
    )
    content.append(
        Paragraph("Generated by ResearchCompass AI", subtitle_style)
    )

    doc.build(content)
    print(f"PDF report saved to: {filepath}")
    return filepath