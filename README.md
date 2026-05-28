# 🧭 ResearchCompass AI

> **AI-powered research topic discovery for Indian M.Tech and PhD students**
>
> Find your perfect research topic in 5 minutes — not months.

---

## 🎯 Problem It Solves

Every M.Tech and PhD student in India wastes **3-6 months** just choosing a research topic.

- No intelligent tool existed to guide them
- ChatGPT gives generic topics from 2023 training data
- No evidence of which topics are genuinely novel
- No roadmap for how to execute the research

**ResearchCompass AI solves this in 5 minutes with evidence from real papers.**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 Live Paper Fetching | Fetches 60-80 papers from ArXiv + OpenAlex + Semantic Scholar |
| 📊 Novelty Scoring | 4-signal weighted scoring — paper count, citations, growth, recency |
| 📈 Publication Trend Charts | Year-by-year paper growth visualization |
| 🔗 Topic Traceability | Every topic linked to real published papers with DOI |
| 🗺️ Research Roadmap | Month-by-month execution plan with risks and mitigations |
| 📋 Proposal Guide | Section-by-section writing guide (zero AI plagiarism) |
| 📄 PDF Report | Downloadable professional research report |
| 🇮🇳 India-Specific | Datasets, venues, and context specific to India |

---

## 🏗️ Architecture

```
User Input (Domain)
        ↓
Trend Scanner ──→ Gap Finder ──→ Topic Suggester
                                        ↓
                         ┌─────────────┴─────────────┐
                   Novelty Detector            Topic Refiner
                         └─────────────┬─────────────┘
                                        ↓
                                Feasibility Checker
                         ┌─────────────┴─────────────┐
                   Roadmap Generator          Citation Scout
                         └─────────────┬─────────────┘
                                        ↓
                                   PDF Report
```

---

## 🤖 9 Specialized Agents

| Agent | What It Does |
|---|---|
| Trend Scanner | Fetches latest papers and identifies emerging trends |
| Gap Finder | Extracts specific research gaps from papers with evidence |
| Topic Suggester | Generates 5 novel topics traceable to real papers |
| Novelty Detector | Scores each topic using 4-signal weighted system |
| Topic Refiner | Converts topics to publication-style titles with datasets |
| Feasibility Checker | Evaluates difficulty, timeline and data availability |
| Roadmap Generator | Creates month-by-month research plan with risks |
| Citation Scout | Finds 5 key papers with LLM relevance scoring |
| Proposal Guide | Generates section-by-section writing guide |

---

## 📊 Novelty Scoring System

Each topic is scored 0-100 using 4 signals:

```
Paper Count Score    × 35%  (fewer papers = more novel)
Citation Score       × 25%  (lower citations = more novel)
Growth Rate Score    × 25%  (higher growth = more novel)
Recency Score        × 15%  (recent papers = more novel)
```

**Classification:**
- 🟢 80-100 : Emerging — strong opportunity, pursue this topic
- 🟡 60-79  : Growing — active field, add specific angle
- 🟠 40-59  : Mature — proceed carefully, find niche
- 🔴 0-39   : Saturated — avoid, too many existing solutions

---

## 🛠️ Tech Stack

```
Framework     : LangChain + LangGraph
LLM           : OpenAI GPT-4o-mini
Vector Store  : FAISS
Embeddings    : OpenAI text-embedding-3-small
UI            : Streamlit
PDF           : ReportLab + Matplotlib
Charts        : Plotly
Data Sources  : ArXiv API + OpenAlex API + Semantic Scholar API
Web Search    : Tavily API
Language      : Python 3.11+
```

---

## 📁 Project Structure

```
compass ai/
├── agents/
│   ├── trend_scanner.py
│   ├── gap_finder.py
│   ├── topic_suggester.py
│   ├── novelty_detector.py
│   ├── topic_refiner_agent.py
│   ├── feasibility_checker.py
│   ├── roadmap_generator.py
│   ├── citation_scout.py
│   └── proposal_guide_generator.py
├── graph/
│   ├── orchestrator.py
│   ├── state.py
│   └── synthesizer.py
├── rag/
│   ├── document_loader.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── retriever.py
├── tools/
│   ├── arxiv_tool.py
│   ├── openalex_tool.py
│   ├── scholar_tool.py
│   ├── web_search_tool.py
│   └── query_builder.py
├── data/
│   ├── faiss_index_*/
│   ├── reports/
│   └── cache/
├── app.py
├── config.py
├── report_generator.py
├── requirements.txt
└── .env
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ResearchCompass-AI.git
cd ResearchCompass-AI
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
Create a `.env` file in the root folder:
```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
```

**Get free API keys here:**
- OpenAI: https://platform.openai.com
- Tavily: https://app.tavily.com
- Semantic Scholar: https://api.semanticscholar.org

### 5. Run the app
```bash
streamlit run app.py
```

Open browser at: http://localhost:8501

---

## 📋 Requirements

```
langchain==0.3.25
langchain-openai==0.3.16
langchain-community==0.3.24
langgraph==0.3.34
faiss-cpu==1.8.0
openai==1.45.0
streamlit==1.38.0
reportlab==4.2.2
matplotlib
plotly
tavily-python
semanticscholar
arxiv
python-dotenv
requests
```

---

## 🎓 Sample Output

### Report for "Deep Learning in Healthcare"

**Trending Topics:**
- Federated Learning for Privacy-Preserving Diagnosis
- Vision Transformers in Medical Imaging
- Self-Supervised Learning for Rare Disease Detection
- Graph Neural Networks for Patient Data Analysis
- Explainable AI for Clinical Decision Support

**Sample Refined Topic:**
```
Title    : Federated Vision Transformers for Privacy-Preserving
           Diabetic Retinopathy Screening in Rural India
Dataset  : EyePACS (Kaggle)
Baseline : ResNet50
Framework: TensorFlow
Metric   : AUC-ROC
Venue    : IEEE Transactions on Medical Imaging
Timeline : 6-8 months
Difficulty: Medium
```

**Novelty Score:** 89/100 — 🟢 Emerging

---

## 🗺️ Future Roadmap

- [x] Multi-agent LangGraph pipeline
- [x] 3 data source integration
- [x] 4-signal novelty scoring system
- [x] Publication trend charts
- [x] Research roadmap generator
- [x] Proposal guide generator
- [x] PDF report generation
- [x] Parallel agent processing
- [ ] User accounts and history
- [ ] Deploy on Hugging Face Spaces
- [ ] Research Mentor Chat
- [ ] University dashboard
- [ ] Mobile app

---

## 👨‍💻 Author

**Gautam Kumar Jha**
- M.Tech in Artificial Intelligence — Delhi Technological University
- AI/ML Faculty — Galgotias University
- Published researcher in NLP and Sentiment Analysis
- GATE and UGC NET qualified

---

## 📄 License

MIT License — feel free to use and modify.

---

## 🙏 Acknowledgements

- ArXiv for open research paper access
- OpenAlex for open academic metadata
- Semantic Scholar for citation data
- LangChain and LangGraph teams
- OpenAI for GPT-4o-mini

---

⭐ **If this project helped you, please give it a star!**
