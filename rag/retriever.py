from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rag.vector_store import build_and_save, load_vector_store, index_exists
from rag.document_loader import load_and_split_papers
from tools.arxiv_tool import fetch_arxiv_papers_multi as _fetch_multi
from tools.query_builder import build_arxiv_queries
from config import OPENAI_API_KEY, LLM_MODEL, MAX_TOKENS, TEMPERATURE, TOP_K_RESULTS


def get_retriever(domain: str):
    """
    Load or build domain-specific FAISS vector store.
    Uses multi-query fetching for better domain accuracy.
    """

    if not index_exists(domain):
        print(f"Building new index for domain: {domain}")

        # Step 1 - generate smart queries
        queries = build_arxiv_queries(domain)

        # Step 2 - fetch papers directly (not via .invoke)
        papers = _fetch_multi(queries)

        if not papers:
            print("No papers found. Try a different domain.")
            return None

        # Step 3 - chunk and build index
        chunks = load_and_split_papers(papers)
        build_and_save(chunks, domain)

    else:
        print(f"Reusing existing index for: {domain}")  # fast!

    # load existing index
    vector_store = load_vector_store(domain)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )
    return retriever


def get_qa_chain(domain: str):
    """
    Build and return domain-specific RetrievalQA chain.
    """

    prompt_template = """
You are a helpful research assistant for M.Tech and PhD students in India.
Use the following research papers as context to answer the question.
All papers are strictly related to the domain: {domain}

Context:
{{context}}

Question: {{question}}

Instructions:
- Use simple and clear English
- Avoid very technical jargon
- Give answer in easy to understand points
- Each point should be one simple sentence
- Maximum 5 points
- Only refer to topics strictly within the domain
- If you don't know, say "I dont have enough information from the papers."
""".format(domain=domain)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    retriever = get_retriever(domain)

    if retriever is None:
        return None

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain