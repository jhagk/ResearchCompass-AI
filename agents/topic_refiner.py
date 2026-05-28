from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, LLM_MODEL


def refine_topic(generic_topic: str, domain: str, level: str = "M.Tech") -> str:
    """
    Transforms a generic research topic into a specific,
    publication-style research direction.
    """

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7
    )

    prompt = f"""
    You are an expert research advisor helping an {level} student 
    in India find a specific, publishable research topic.

    Generic topic given: "{generic_topic}"
    Research domain: "{domain}"

    Transform this generic topic into a SPECIFIC, publication-style 
    research direction by adding:
    1. A specific METHOD
    2. A specific TARGET PROBLEM
    3. A specific DATASET or DOMAIN CONTEXT
    4. An INNOVATION ANGLE

    Rules:
    - Output ONLY the refined topic title
    - It should sound like a real IEEE/Springer paper title
    - Maximum 20 words
    - No explanation, just the title
    - Must be specific and novel
    - Must be feasible for {level} in 1-2 years

    Examples:
    - "Federated Transformer Networks for Early Sepsis Prediction 
       Using Multimodal ICU Data"
    - "Self-Supervised Vision Transformers for Low-Resource Brain 
       Tumor MRI Segmentation"
    - "Lightweight BERT for Real-Time Hate Speech Detection in 
       Code-Mixed Indian Social Media"

    Now refine: "{generic_topic}"
    """

    response = llm.invoke(prompt)
    refined = response.content.strip()
    print(f"Refined: '{generic_topic}' -> '{refined}'")
    return refined


def refine_all_topics(topics_text: str, domain: str, level: str = "M.Tech") -> str:
    """
    Takes the full topic suggester output and refines
    each topic into a publication-style title with
    full explainability, reasoning and implementation details.
    """

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7
    )

    prompt = f"""
    You are an expert research advisor for {level} students in India.
    
    Domain: "{domain}"
    
    Below are 5 generic research topics. Transform each one into a 
    SPECIFIC, publication-style research direction WITH full reasoning
    AND complete implementation details.

    For each topic provide EXACTLY this format:

    ### Topic [number]: [Refined Publication-Style Title]

    **Why This Topic Matters:**
    [2 sentences explaining real-world importance.
     Mention India context where relevant.]

    **Research Opportunity:**
    [2 sentences explaining what existing systems fail at
     and what gap this research fills.]

    **Method:**
    [Specific AI/ML method to be used]

    **Target Problem:**
    [Exact problem being solved]

    **Innovation Angle:**
    [What makes this novel]

    **Implementation Details:**
    - Recommended Framework: [PyTorch / TensorFlow / HuggingFace]
    - Suggested Dataset: [Specific dataset name and source]
    - Baseline Model: [Specific model to compare against]
    - Hardware Requirement: [GPU specs needed]
    - Evaluation Metric: [Specific metric like F1, AUC-ROC, BLEU]

    **Suggested Publication Venue:**
    [Specific IEEE/Springer/ACM journal or conference name]

    **Difficulty:** [Easy / Medium / Hard]
    **Timeline:** [X-Y months]

    ---

    Generic topics to refine:
    {topics_text}

    Rules:
    - Each refined title must sound like a real research paper
    - Why This Topic Matters must mention real-world impact
    - Research Opportunity must mention what existing work fails at
    - Innovation Angle must be specific and novel
    - Implementation Details must be specific and actionable
    - Dataset names must be real and publicly available
    - Baseline models must be real existing models
    - Mention India context wherever applicable
    - Use simple and clear English
    - Do not use very technical jargon in explanations

    Examples of good Implementation Details:
    
    For Healthcare:
    - Framework: PyTorch
    - Dataset: MIMIC-III (ICU data), NIH Chest X-ray Dataset
    - Baseline: ResNet50, DenseNet121
    - Hardware: NVIDIA Tesla T4 (Google Colab Pro)
    - Metric: AUC-ROC, Sensitivity, Specificity

    For NLP:
    - Framework: HuggingFace Transformers
    - Dataset: IndicGLUE, AI4Bharat
    - Baseline: mBERT, XLM-RoBERTa
    - Hardware: NVIDIA T4 16GB (Google Colab)
    - Metric: F1 Score, BLEU Score

    For Computer Vision:
    - Framework: PyTorch + OpenCV
    - Dataset: ImageNet, COCO, custom dataset
    - Baseline: YOLOv8, ResNet50
    - Hardware: NVIDIA RTX 3060 or Google Colab Pro
    - Metric: mAP, IoU, Accuracy

    For Blockchain:
    - Framework: Ethereum + Solidity + Python
    - Dataset: Ethereum transaction data, custom smart contracts
    - Baseline: Traditional centralized systems
    - Hardware: Standard laptop (no GPU needed)
    - Metric: Transaction time, Gas cost, Security score
    """

    response = llm.invoke(prompt)
    return response.content.strip()