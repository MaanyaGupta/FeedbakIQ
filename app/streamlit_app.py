"""
FeedbackIQ — Streamlit Demonstration Interface.
Customer Feedback Intelligence System with Aspect Extraction & RAG Chatbot.

Run locally:
    streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path

# Ensure Windows OpenMP DLL compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add repository root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except Exception:
    pass

import pandas as pd
import streamlit as st

from src.retrieval import ReviewRetriever, ASPECT_DISPLAY_NAMES
from src.rag import FeedbackRAGPipeline

# Page Configuration
st.set_page_config(
    page_title="FeedbackIQ — Customer Feedback Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for University Project Presentation
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }
    .card-pos { color: #16A34A; }
    .card-neu { color: #D97706; }
    .card-neg { color: #DC2626; }
    .card-neutral { color: #2563EB; }
    .evidence-card {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .aspect-pill {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing FeedbackIQ RAG Engine (Sentence Transformers + FAISS)...")
def get_pipeline() -> FeedbackRAGPipeline:
    """Initialize and cache the RAG pipeline and ReviewRetriever in memory."""
    retriever = ReviewRetriever()
    return FeedbackRAGPipeline(retriever=retriever)


pipeline = get_pipeline()
retriever = pipeline.retriever

# App Header
st.markdown('<div class="main-title">🛍️ FeedbackIQ: Customer Feedback Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Aspect-Based Sentiment Analysis & Grounded Customer Feedback RAG Chatbot</div>',
    unsafe_allow_html=True,
)

# Sidebar: Domain & Product Selection
with st.sidebar:
    st.header("⚙️ Controls & Navigation")
    
    # 1. Domain Selection
    selected_domain = st.selectbox(
        "📁 Select Domain:",
        options=["Electronics", "Fashion"],
        index=0,
        help="Filter products by Amazon product category."
    )

    # Filter catalog by selected domain
    all_products = retriever.get_available_products()
    domain_products = [p for p in all_products if p.get("domain", "").lower() == selected_domain.lower()]

    if not domain_products:
        st.warning(f"No products found for domain '{selected_domain}'.")
        st.stop()

    # 2. Product Selection
    product_options = {
        p["parent_asin"]: f"[{p['parent_asin']}] {p['title'][:45]}... ({p['total_reviews']} reviews)"
        for p in domain_products
    }

    selected_asin = st.selectbox(
        "📦 Select Product:",
        options=list(product_options.keys()),
        format_func=lambda asin: product_options[asin],
        index=0,
        help="Select a product to view sentiment distribution, complaints, and ask questions."
    )

    st.markdown("---")
    st.markdown("**System Architecture:**")
    st.markdown("""
    - **Vector Search**: FAISS `IndexFlatIP`
    - **Embeddings**: `all-MiniLM-L6-v2`
    - **Taxonomy**: Domain-specific aspects
    - **RAG Guarantee**: Strict product review isolation
    """)

# Retrieve selected product details
product_info = retriever.get_product_info(selected_asin)
prod_title = product_info["title"]
total_revs = product_info["total_reviews"]
avg_rating = product_info["average_rating"]
pos_pct = product_info["sentiment_percentages"]["positive"]
neu_pct = product_info["sentiment_percentages"]["neutral"]
neg_pct = product_info["sentiment_percentages"]["negative"]

# Main Layout
st.subheader(f"📌 {prod_title}")
st.caption(f"**Domain:** {selected_domain} | **ASIN:** `{selected_asin}` | **Total Verified Reviews:** {total_revs}")

# 3. Product Sentiment Overview (Metric Cards)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value card-neutral">{avg_rating} ★</div>'
        f'<div class="metric-label">Avg Rating</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value card-neutral">{total_revs}</div>'
        f'<div class="metric-label">Total Reviews</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value card-pos">{pos_pct}%</div>'
        f'<div class="metric-label">Positive</div></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value card-neu">{neu_pct}%</div>'
        f'<div class="metric-label">Neutral</div></div>',
        unsafe_allow_html=True,
    )
with col5:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value card-neg">{neg_pct}%</div>'
        f'<div class="metric-label">Negative</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# 4. Aspect Breakdown (Top Complaints)
st.subheader("🔍 Top Complaint Aspects Breakdown")
st.write("Distribution of customer complaints identified from negative and critical reviews:")

# Retrieve top aspects for product
prod_index_data = retriever._get_or_create_product_index(selected_asin)
prod_df = prod_index_data["df"]
neg_reviews = prod_df[
    (prod_df["sentiment"] == "negative")
    | (prod_df["predicted_sentiment"] == "negative")
    | (prod_df["rating"] <= 2.0)
]
total_negatives = len(neg_reviews)

aspect_counts = {}
for aspects in neg_reviews.get("aspects", []):
    for a in set(aspects):
        aspect_counts[a] = aspect_counts.get(a, 0) + 1

if aspect_counts:
    aspect_cols = st.columns(2)
    sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:6]

    for i, (aspect, count) in enumerate(sorted_aspects):
        col = aspect_cols[i % 2]
        pct = round((count / total_negatives) * 100.0, 1) if total_negatives > 0 else 0.0
        disp_name = ASPECT_DISPLAY_NAMES.get(aspect, aspect.replace("_", " ").title())
        with col:
            st.write(f"**{disp_name}** — `{pct}%` of negative reviews ({count}/{total_negatives})")
            st.progress(min(pct / 100.0, 1.0))
else:
    st.info("No concentrated negative complaint clusters detected for this product.")

st.markdown("---")

# 5. Customer Feedback RAG Chatbot
st.subheader("🤖 Product Feedback AI Assistant (RAG)")
st.write(
    "Ask any customer feedback question. The system strictly isolates reviews for this product, "
    "retrieves semantically relevant customer statements, and produces a zero-hallucination grounded summary."
)

# Suggested quick questions
st.markdown("**Quick Example Questions:**")
example_cols = st.columns(4)
q_suggestions = [
    "Why are customers unhappy with this product?",
    "What are the biggest complaints?",
    "What do customers like about this product?",
    "What is the main issue with this product?",
]

if "current_question" not in st.session_state:
    st.session_state.current_question = q_suggestions[0]

for i, q in enumerate(q_suggestions):
    if example_cols[i].button(q, key=f"btn_q_{i}", use_container_width=True):
        st.session_state.current_question = q

user_question = st.text_input(
    "Ask a question about this product:",
    value=st.session_state.current_question,
    placeholder="e.g. Why are customers unhappy with this product?",
)

if user_question:
    with st.spinner("Retrieving isolated reviews and generating grounded answer..."):
        response = pipeline.answer_question(selected_asin, user_question, top_k=5)

    retrieval_ctx = response["retrieval_context"]
    retrieved_count = retrieval_ctx["retrieved_count"]
    intent = retrieval_ctx["intent"]

    # Evidence Banner
    st.success(
        f"✅ Generated answer using **{retrieved_count}** verified, product-isolated reviews "
        f"(Intent detected: `{intent}`)."
    )

    # AI Answer Box
    st.markdown("### 📋 AI Feedback Synthesis")
    st.markdown(response["answer"])

    # Expandable Inspection Drawer
    with st.expander("🔎 View Grounded Citations & Retrieved Reviews"):
        st.caption(f"Retrieved {retrieved_count} reviews from product `{selected_asin}` via FAISS similarity:")
        for idx, r in enumerate(retrieval_ctx["retrieved_reviews"], 1):
            st.markdown(
                f"""
                <div class="evidence-card">
                    <b>#{idx} — {r['title']}</b> (Rating: <b>{r['rating']} ★</b> | Sentiment: <code>{r['sentiment']}</code>)<br>
                    <span style="color: #475569; font-size: 0.95rem;">"{r['text']}"</span><br>
                    <span style="font-size: 0.8rem; color: #64748B;">Aspects: {', '.join(r['aspect_labels']) if r['aspect_labels'] else 'General'} | Similarity Score: {r['similarity_score']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Footer
st.markdown("---")
st.caption("FeedbackIQ — University Project Prototype • Powered by FAISS, Sentence Transformers & Aspect Extraction")
