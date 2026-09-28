"""
Builder script for notebooks/07_rag_demo.ipynb.
Constructs and executes the customer feedback RAG demo notebook with full outputs.
"""

import sys
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_DIR = Path("notebooks")
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOK_DIR / "07_rag_demo.ipynb"


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Architecture
    cells.append(nbf.v4.new_markdown_cell("""# FeedbackIQ — Customer Feedback RAG Chatbot Prototype
**Phase 5: Retrieval-Augmented Generation (RAG) Data & Reasoning Layer**

This notebook demonstrates the end-to-end Python RAG pipeline for customer feedback intelligence.

### Pipeline Architecture:
```text
Product ID
    ↓
Retrieve reviews belonging ONLY to this product (Strict Isolation)
    ↓
Prioritize negative reviews when the question is about complaints
    ↓
Generate embeddings using Sentence Transformers (all-MiniLM-L6-v2)
    ↓
Store & Retrieve embeddings using FAISS (IndexFlatIP)
    ↓
Retrieve top relevant reviews
    ↓
Include sentiment + domain aspect information
    ↓
Send grounded context to an LLM (Modular Layer)
    ↓
Generate structured, verifiable answer with citations
```

---
### Key Architectural Guarantees:
1. **Strict Product Isolation**: The FAISS index is partitioned per `parent_asin`, guaranteeing zero cross-product review leakage.
2. **Intent-Based Sentiment Prioritization**: Questions regarding complaints prioritize negative reviews; questions regarding praises prioritize positive reviews.
3. **Aspect Enrichment**: Every retrieved review is enriched with domain-specific taxonomies (*Electronics* vs. *Fashion*).
4. **Zero-Hallucination Grounded Generation**: Answers cite exact review counts, ratings, aspects, and verbatim customer quotes."""))

    # Setup
    cells.append(nbf.v4.new_code_cell("""import os
import sys
from pathlib import Path

# Ensure Windows OpenMP DLL compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except Exception:
    pass

# Add project root to sys.path
root_path = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import pandas as pd
from IPython.display import display, Markdown, HTML

from src.retrieval import ReviewRetriever, ASPECT_DISPLAY_NAMES
from src.rag import FeedbackRAGPipeline, GroundedSynthesizerLLM, get_llm

print("FeedbackIQ RAG Environment initialized successfully.")"""))

    # Section 1: Catalog Overview
    cells.append(nbf.v4.new_markdown_cell("""## 1. Product Catalog Overview
Inspect the products available in the processed feedback database across **Electronics** and **Fashion**."""))

    cells.append(nbf.v4.new_code_cell("""retriever = ReviewRetriever()
products = retriever.get_available_products()

catalog_data = []
for p in products:
    catalog_data.append({
        "Product ID (ASIN)": p["parent_asin"],
        "Domain": p["domain"],
        "Product Title": p["title"][:50] + ("..." if len(p["title"]) > 50 else ""),
        "Reviews": p["total_reviews"],
        "Avg Rating": f"{p['average_rating']} ★",
        "Positive %": f"{p['sentiment_percentages']['positive']}%",
        "Neutral %": f"{p['sentiment_percentages']['neutral']}%",
        "Negative %": f"{p['sentiment_percentages']['negative']}%",
    })

df_catalog = pd.DataFrame(catalog_data)
display(HTML("<h3>Available Products</h3>"))
display(df_catalog)"""))

    # Section 2: Electronics Complaint Query
    cells.append(nbf.v4.new_markdown_cell("""## 2. Complaint Query — Electronics (Fire TV Stick)
* **Product**: `B075X8471B` (Fire TV Stick with Alexa Voice Remote)
* **Question**: `"Why are customers unhappy with this product?"`

The system detects complaint intent, prioritizes negative reviews, performs FAISS semantic search, and generates grounded synthesis."""))

    cells.append(nbf.v4.new_code_cell("""pipeline = FeedbackRAGPipeline(retriever=retriever)

asin_firetv = "B075X8471B"
query_1 = "Why are customers unhappy with this product?"

res_1 = pipeline.answer_question(asin_firetv, query_1, top_k=4)

display(HTML(f"<b>Query Intent Detected:</b> <code>{res_1['retrieval_context']['intent']}</code>"))

# Show retrieved reviews in a table
retrieved_rows = []
for r in res_1["retrieval_context"]["retrieved_reviews"]:
    retrieved_rows.append({
        "Rating": f"{r['rating']} ★",
        "Sentiment": r["sentiment"],
        "Aspects": ", ".join(r["aspect_labels"]),
        "Title": r["title"][:40],
        "Review Snippet": r["text"][:110] + "...",
        "Sim Score": r["similarity_score"],
    })
display(HTML("<h4>Retrieved Isolated Reviews:</h4>"))
display(pd.DataFrame(retrieved_rows))

display(HTML("<h4>Grounded RAG Response:</h4>"))
display(Markdown(res_1["answer"]))"""))

    # Section 3: Electronics Biggest Complaints Query
    cells.append(nbf.v4.new_markdown_cell("""## 3. Complaint Query — Panasonic Earbuds
* **Product**: `B07S764D9V` (Panasonic ErgoFit Wired Earbuds)
* **Question**: `"What are the biggest complaints?"`"""))

    cells.append(nbf.v4.new_code_cell("""asin_earbuds = "B07S764D9V"
query_2 = "What are the biggest complaints?"

res_2 = pipeline.answer_question(asin_earbuds, query_2, top_k=4)

display(HTML(f"<b>Product:</b> {res_2['product_title']} (ASIN: <code>{asin_earbuds}</code>)"))
display(Markdown(res_2["answer"]))"""))

    # Section 4: Positive Feedback Query
    cells.append(nbf.v4.new_markdown_cell("""## 4. Praise Query — Customer Likes & Features
* **Product**: `B075X8471B` (Fire TV Stick)
* **Question**: `"What do customers like about this product?"`

Notice how intent detection automatically shifts to prioritize positive reviews and satisfaction aspects."""))

    cells.append(nbf.v4.new_code_cell("""query_3 = "What do customers like about this product?"
res_3 = pipeline.answer_question(asin_firetv, query_3, top_k=4)

display(HTML(f"<b>Query Intent Detected:</b> <code>{res_3['retrieval_context']['intent']}</code>"))
display(Markdown(res_3["answer"]))"""))

    # Section 5: Fashion Feedback Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 5. Fashion Domain Query — Underwear Sizing & Fabric
* **Product**: `B0956HNH2Y` (Fruit of the Loom Women's Eversoft Cotton Briefs)
* **Question**: `"What is the main issue with this product?"`

Demonstrates domain taxonomy adaptation: detecting **Size & Fit**, **Material & Fabric**, and **Design**."""))

    cells.append(nbf.v4.new_code_cell("""asin_fashion = "B0956HNH2Y"
query_4 = "What is the main issue with this product?"

res_4 = pipeline.answer_question(asin_fashion, query_4, top_k=4)

display(HTML(f"<b>Domain:</b> {res_4['domain']} | <b>Product:</b> {res_4['product_title']}"))
display(Markdown(res_4["answer"]))"""))

    # Section 6: Verification of Product Isolation
    cells.append(nbf.v4.new_markdown_cell("""## 6. Verification of Strict Product Isolation
Verify mathematically that 100% of retrieved reviews belong to the queried product, with zero cross-product contamination."""))

    cells.append(nbf.v4.new_code_cell("""test_asins = ["B075X8471B", "B010BWYDYA", "B07S764D9V", "B0956HNH2Y", "B07TVHSDMQ"]
verification_results = []

for asin in test_asins:
    res = retriever.retrieve_reviews(asin, "overall quality and issues", top_k=5)
    foreign_reviews = [r for r in res["retrieved_reviews"] if r["parent_asin"] != asin]
    verification_results.append({
        "Product ASIN": asin,
        "Domain": res["domain"],
        "Reviews Retrieved": len(res["retrieved_reviews"]),
        "Foreign Reviews": len(foreign_reviews),
        "Isolation Status": "100% Isolated (PASSED)" if len(foreign_reviews) == 0 else "FAILED",
    })

df_verification = pd.DataFrame(verification_results)
display(HTML("<h3>Cross-Product Leakage Test Results</h3>"))
display(df_verification)"""))

    # Section 7: Modular LLM Layer
    cells.append(nbf.v4.new_markdown_cell("""## 7. Modular LLM Layer Demonstration
The pipeline LLM backend is completely decoupled from the retrieval engine via the `BaseLLM` interface.
You can swap between:
- `GroundedSynthesizerLLM`: Zero-hallucination deterministic extractor
- `HuggingFaceLocalLLM`: Local sequence-to-sequence transformer (`google/flan-t5-small`, etc.)
- `OpenAILLM`: OpenAI GPT models (`gpt-4o-mini`, etc.)
- `GeminiLLM`: Google Gemini models (`gemini-1.5-flash`, etc.)"""))

    cells.append(nbf.v4.new_code_cell("""# Demonstrate modular LLM interface
grounded_llm = get_llm("grounded")
print(f"Current default LLM engine: {grounded_llm.__class__.__name__}")

# Example of instantiating pipeline with custom LLM backend
custom_pipeline = FeedbackRAGPipeline(retriever=retriever, llm=grounded_llm)
print("Pipeline configured with modular LLM successfully.")"""))

    # Section 8: Interactive Function
    cells.append(nbf.v4.new_markdown_cell("""## 8. Interactive Query Function
Use this helper function to ask any custom question against any product in the catalog."""))

    cells.append(nbf.v4.new_code_cell("""def ask_feedbackiq(product_id: str, question: str, top_k: int = 5):
    \"\"\"Query FeedbackIQ RAG for a given product and question.\"\"\"
    result = pipeline.answer_question(product_id, question, top_k=top_k)
    display(HTML(f"<h3>{result['product_title']} ({result['product_id']})</h3>"))
    display(HTML(f"<b>Question:</b> <i>{result['question']}</i>"))
    display(Markdown(result["answer"]))

# Example interactive invocation
ask_feedbackiq("B010BWYDYA", "Why are customers unhappy with this tablet?")"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {NOTEBOOK_PATH}. Executing notebook...")

    # Execute notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOK_DIR.resolve())}})

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook {NOTEBOOK_PATH} executed and saved successfully!")


if __name__ == "__main__":
    build_notebook()
