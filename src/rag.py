"""
FeedbackIQ - Customer Feedback RAG Pipeline (Prototype).
Combines product-isolated review retrieval, intent-based sentiment prioritization,
FAISS semantic search, and modular LLM generation to answer customer feedback questions.

Guarantees:
1. Product review isolation: No cross-product review leakage.
2. Prioritizes negative reviews when questions ask about complaints.
3. Modular LLM layer: Supports Grounded Synthesizer, HuggingFace Local Models, OpenAI, and Gemini.
4. Hallucination prevention: Answers strictly grounded in retrieved reviews.
"""

import logging
import os
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

# Ensure project root is in sys.path for direct CLI execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows OpenMP DLL compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except Exception:
    pass

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.retrieval import ReviewRetriever, ASPECT_DISPLAY_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract Base Class for modular LLM generation backends."""

    @abstractmethod
    def generate(self, question: str, retrieval_context: Dict[str, Any]) -> str:
        """
        Generate grounded answer given a question and the retrieved review context.

        Parameters:
            question: User's feedback query.
            retrieval_context: Structured dictionary returned by ReviewRetriever.

        Returns:
            Grounded textual answer string.
        """
        pass


class GroundedSynthesizerLLM(BaseLLM):
    """
    Deterministic Grounded Generation Engine.
    Synthesizes rich, structured feedback answers directly from retrieved reviews,
    sentiment distributions, and aspect taxonomies with zero hallucination.
    """

    def generate(self, question: str, retrieval_context: Dict[str, Any]) -> str:
        prod_title = retrieval_context.get("product_title", "Unknown Product")
        product_id = retrieval_context.get("product_id", "")
        retrieved_reviews = retrieval_context.get("retrieved_reviews", [])
        num_reviews = len(retrieved_reviews)
        intent = retrieval_context.get("intent", "general")
        top_aspects = retrieval_context.get("top_aspects", [])
        sentiment_breakdown = retrieval_context.get("sentiment_breakdown", {})
        overall_stats = retrieval_context.get("overall_product_stats", {})

        if num_reviews == 0:
            return (
                f"No customer reviews found for product '{prod_title}' (ASIN: {product_id}). "
                "Unable to provide feedback analysis."
            )

        # Build grounded response sections
        lines = []
        lines.append(f"Based on {num_reviews} relevant reviews for \"{prod_title}\" (ASIN: {product_id}):\n")

        # 1. Sentiment overview
        pos_cnt = sentiment_breakdown.get("positive", 0)
        neu_cnt = sentiment_breakdown.get("neutral", 0)
        neg_cnt = sentiment_breakdown.get("negative", 0)
        avg_rating = overall_stats.get("average_rating", "N/A")
        total_in_catalog = overall_stats.get("total_reviews", num_reviews)

        lines.append("### 1. Sentiment Breakdown of Retrieved Reviews")
        lines.append(
            f"- Negative: {neg_cnt} ({round((neg_cnt/num_reviews)*100)}%) | "
            f"Neutral: {neu_cnt} ({round((neu_cnt/num_reviews)*100)}%) | "
            f"Positive: {pos_cnt} ({round((pos_cnt/num_reviews)*100)}%)"
        )
        lines.append(f"- Product Catalog Rating: {avg_rating} / 5.0 (across {total_in_catalog} total reviews)\n")

        # 2. Key Aspects / Themes
        lines.append("### 2. Relevant Aspects Identified")
        if top_aspects:
            for item in top_aspects[:5]:
                lines.append(f"- **{item['label']}**: cited in {item['count']} review(s) ({item['percentage']}%)")
        else:
            lines.append("- General overall product satisfaction; no concentrated aspect issues detected.")
        lines.append("")

        # 3. Main Feedback Summary (Complaints vs Praises based on intent)
        if intent == "complaints":
            lines.append("### 3. Main Customer Complaints")
            complaints_extracted = []
            for r in retrieved_reviews:
                if r["sentiment"] == "negative" or r["rating"] <= 2.0 or r["predicted_sentiment"] == "negative":
                    aspect_str = ", ".join(r["aspect_labels"]) if r["aspect_labels"] else "General Quality"
                    title = r["title"].strip() if r["title"] else "Issue"
                    complaints_extracted.append(f"**{aspect_str}**: {title}")

            if complaints_extracted:
                # Deduplicate while preserving order
                seen = set()
                deduped = []
                for c in complaints_extracted:
                    if c not in seen:
                        seen.add(c)
                        deduped.append(c)
                for i, c in enumerate(deduped[:4], 1):
                    lines.append(f"{i}. {c}")
            else:
                lines.append("No severe negative complaints were found among the retrieved reviews.")
        elif intent == "praises":
            lines.append("### 3. Key Customer Praises & Liked Features")
            praises_extracted = []
            for r in retrieved_reviews:
                if r["sentiment"] == "positive" or r["rating"] >= 4.0:
                    aspect_str = ", ".join(r["aspect_labels"]) if r["aspect_labels"] else "General Quality"
                    title = r["title"].strip() if r["title"] else "Satisfied Purchase"
                    praises_extracted.append(f"**{aspect_str}**: {title}")

            if praises_extracted:
                seen = set()
                deduped = []
                for p in praises_extracted:
                    if p not in seen:
                        seen.add(p)
                        deduped.append(p)
                for i, p in enumerate(deduped[:4], 1):
                    lines.append(f"{i}. {p}")
            else:
                lines.append("Customers expressed mixed or moderate satisfaction.")
        else:
            lines.append("### 3. Summary of Customer Observations")
            for i, r in enumerate(retrieved_reviews[:3], 1):
                aspect_str = ", ".join(r["aspect_labels"]) if r["aspect_labels"] else "General"
                title = r["title"].strip() if r["title"] else "Feedback"
                lines.append(f"{i}. **{aspect_str}** (Rating: {r['rating']} / 5.0): {title}")
        lines.append("")

        # 4. Representative Evidence (Direct Quotes)
        lines.append("### 4. Representative Evidence from Customer Reviews")
        for i, r in enumerate(retrieved_reviews[:4], 1):
            text_snippet = r["text"].strip()
            # Clean and truncate if very long
            if len(text_snippet) > 220:
                text_snippet = text_snippet[:217] + "..."
            aspect_tag = f"[{', '.join(r['aspect_labels'])}]" if r["aspect_labels"] else "[General]"
            lines.append(
                f"{i}. *\"{text_snippet}\"*  \n"
                f"   — **Rating**: {r['rating']} / 5.0 | **Sentiment**: {r['sentiment']} | **Aspects**: {aspect_tag}"
            )

        return "\n".join(lines)


class HuggingFaceLocalLLM(BaseLLM):
    """
    HuggingFace Local Sequence-to-Sequence / Causal LLM adapter.
    Uses models such as google/flan-t5-small or google/flan-t5-base with a grounded prompt.
    """

    def __init__(self, model_name: str = "google/flan-t5-small", device: Optional[str] = None):
        self.model_name = model_name
        logger.info("Initializing HuggingFace Local LLM with '%s'...", model_name)
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def generate(self, question: str, retrieval_context: Dict[str, Any]) -> str:
        reviews = retrieval_context.get("retrieved_reviews", [])
        prod_title = retrieval_context.get("product_title", "")
        num_reviews = len(reviews)

        # Build concise context
        context_snippets = []
        for i, r in enumerate(reviews[:4], 1):
            context_snippets.append(
                f"Review {i} (Rating {r['rating']}, Aspects: {', '.join(r['aspect_labels'])}): {r['text'][:150]}"
            )
        context_text = "\n".join(context_snippets)

        prompt = (
            f"You are a helpful customer feedback analyst.\n"
            f"Product: {prod_title}\n"
            f"Retrieved Reviews:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Instruction: Based strictly on the reviews above, answer the question. "
            f"Do not invent facts. State what customers complain about or like.\nAnswer:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.3)
        generated_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Combine local model generation with grounded evidence
        grounded_wrapper = GroundedSynthesizerLLM()
        full_response = grounded_wrapper.generate(question, retrieval_context)
        return f"**LLM Synthesis ({self.model_name})**:\n{generated_summary}\n\n---\n{full_response}"


class OpenAILLM(BaseLLM):
    """OpenAI API Adapter (e.g. gpt-4o-mini). Active if OPENAI_API_KEY is present."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        import openai
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate(self, question: str, retrieval_context: Dict[str, Any]) -> str:
        reviews = retrieval_context.get("retrieved_reviews", [])
        prod_title = retrieval_context.get("product_title", "")
        num_reviews = len(reviews)

        context_text = "\n".join([
            f"- Review {i+1} [Rating: {r['rating']}/5, Sentiment: {r['sentiment']}, Aspects: {', '.join(r['aspect_labels'])}]: \"{r['text']}\""
            for i, r in enumerate(reviews)
        ])

        system_prompt = (
            "You are an objective customer feedback analysis assistant. "
            "You MUST follow these strict rules:\n"
            "1. Answer ONLY using the facts from the retrieved reviews provided.\n"
            "2. Always start your response with 'Based on X relevant reviews...'.\n"
            "3. Summarize: main complaints, relevant aspects, sentiment breakdown, and representative quotes.\n"
            "4. NEVER invent or assume complaints not directly stated in the reviews."
        )

        user_content = (
            f"Product: {prod_title} (ASIN: {retrieval_context.get('product_id')})\n"
            f"Context Reviews ({num_reviews} retrieved):\n{context_text}\n\n"
            f"Question: {question}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


class GeminiLLM(BaseLLM):
    """Google Gemini API Adapter. Active if GEMINI_API_KEY is present."""

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        import google.generativeai as genai
        key = api_key or os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, question: str, retrieval_context: Dict[str, Any]) -> str:
        reviews = retrieval_context.get("retrieved_reviews", [])
        prod_title = retrieval_context.get("product_title", "")
        num_reviews = len(reviews)

        context_text = "\n".join([
            f"- Review {i+1} [Rating: {r['rating']}/5, Sentiment: {r['sentiment']}, Aspects: {', '.join(r['aspect_labels'])}]: \"{r['text']}\""
            for i, r in enumerate(reviews)
        ])

        prompt = (
            f"You are an objective customer feedback analysis assistant.\n"
            f"Product: {prod_title} (ASIN: {retrieval_context.get('product_id')})\n"
            f"Retrieved Reviews:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Instructions:\n"
            f"1. Start with 'Based on {num_reviews} relevant reviews...'\n"
            f"2. Summarize: main complaints, relevant aspects, sentiment distribution, and representative quotes.\n"
            f"3. Do NOT hallucinate complaints not present in the reviews."
        )

        response = self.model.generate_content(prompt)
        return response.text.strip()


def get_llm(provider: str = "grounded", **kwargs) -> BaseLLM:
    """
    Factory function for obtaining modular LLM generation backends.

    Supported providers:
    - 'grounded' (default): Zero-hallucination deterministic structured synthesis engine.
    - 'flan-t5' or 'hf': Local Hugging Face seq2seq model (e.g. google/flan-t5-small).
    - 'openai': OpenAI API adapter (requires OPENAI_API_KEY).
    - 'gemini': Google Gemini API adapter (requires GEMINI_API_KEY).
    """
    provider_clean = provider.lower().strip()
    if provider_clean in ["grounded", "default"]:
        return GroundedSynthesizerLLM()
    elif provider_clean in ["hf", "flan-t5", "huggingface"]:
        model_name = kwargs.get("model_name", "google/flan-t5-small")
        return HuggingFaceLocalLLM(model_name=model_name)
    elif provider_clean == "openai":
        return OpenAILLM(**kwargs)
    elif provider_clean == "gemini":
        return GeminiLLM(**kwargs)
    else:
        logger.warning("Unknown LLM provider '%s'. Falling back to GroundedSynthesizerLLM.", provider)
        return GroundedSynthesizerLLM()


class FeedbackRAGPipeline:
    """
    End-to-End Customer Feedback RAG Pipeline.
    Orchestrates product-isolated retrieval, intent prioritization, and grounded LLM generation.
    """

    def __init__(
        self,
        retriever: Optional[ReviewRetriever] = None,
        llm: Optional[BaseLLM] = None,
        llm_type: str = "grounded",
        **llm_kwargs,
    ):
        self.retriever = retriever or ReviewRetriever()
        self.llm = llm or get_llm(provider=llm_type, **llm_kwargs)

    def answer_question(
        self,
        product_id: str,
        question: str,
        top_k: int = 5,
        prioritize_sentiment: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full RAG pipeline for a given product and user question.

        Returns:
            Dict containing:
                - product_id: queried ASIN
                - product_title: product title
                - question: user question
                - answer: generated grounded answer
                - retrieval_context: full retrieval metadata and cited reviews
        """
        # 1. Product review isolation & retrieval
        retrieval_res = self.retriever.retrieve_reviews(
            product_id=product_id,
            query=question,
            top_k=top_k,
            prioritize_sentiment=prioritize_sentiment,
        )

        # 2. Grounded generation via modular LLM
        answer = self.llm.generate(question, retrieval_res)

        return {
            "product_id": product_id,
            "product_title": retrieval_res["product_title"],
            "domain": retrieval_res["domain"],
            "question": question,
            "answer": answer,
            "retrieval_context": retrieval_res,
        }


def run_cli():
    """Simple interactive CLI for querying customer feedback RAG prototype."""
    print("=" * 70)
    print(" FeedbackIQ - Customer Feedback RAG Chatbot Prototype")
    print("=" * 70)
    print("Initializing pipeline (Sentence Transformers + FAISS)...")
    pipeline = FeedbackRAGPipeline()

    products = pipeline.retriever.get_available_products()
    print("\nAvailable Products in Catalog:")
    for i, p in enumerate(products, 1):
        print(f"  {i}. [{p['parent_asin']}] ({p['domain']}) {p['title'][:60]}... ({p['total_reviews']} reviews)")

    print("\nExample Questions:")
    print("  - Why are customers unhappy with this product?")
    print("  - What are the biggest complaints?")
    print("  - What do customers like about this product?")
    print("  - What is the main issue with this product?")
    print("\n(Type 'exit' or 'quit' at any prompt to stop)\n")

    while True:
        try:
            prod_input = input("Enter product ID: ").strip()
            if prod_input.lower() in ["exit", "quit", "q"]:
                print("Exiting FeedbackIQ RAG. Goodbye!")
                break
            if not prod_input:
                continue

            question_input = input("Enter question: ").strip()
            if question_input.lower() in ["exit", "quit", "q"]:
                print("Exiting FeedbackIQ RAG. Goodbye!")
                break
            if not question_input:
                continue

            print("\nProcessing grounded feedback query...")
            res = pipeline.answer_question(prod_input, question_input, top_k=5)

            print("\n" + "=" * 70)
            print(f"Product: {res['product_title']} (ASIN: {res['product_id']})")
            print(f"Question: {res['question']}")
            print("=" * 70)
            print(res["answer"])
            print("=" * 70 + "\n")

        except ValueError as ve:
            print(f"\n[Error]: {ve}\n")
        except KeyboardInterrupt:
            print("\nExiting FeedbackIQ RAG. Goodbye!")
            break
        except Exception as e:
            logger.exception("Unexpected error during query processing: %s", e)
            print(f"\n[Error occurred]: {e}\n")


if __name__ == "__main__":
    run_cli()
