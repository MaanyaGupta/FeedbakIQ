"""
Unit tests for Customer Feedback RAG Pipeline and isolated ReviewRetriever.
Ensures product isolation, sentiment prioritization, aspect tagging, and grounded LLM output.
"""

import pytest
from src.retrieval import ReviewRetriever
from src.rag import FeedbackRAGPipeline, GroundedSynthesizerLLM, get_llm


class TestRAGPipeline:

    @pytest.fixture(scope="class")
    def retriever(self):
        return ReviewRetriever()

    @pytest.fixture(scope="class")
    def pipeline(self, retriever):
        return FeedbackRAGPipeline(retriever=retriever)

    def test_strict_product_isolation(self, retriever):
        """Verify that 100% of retrieved reviews belong strictly to the queried product."""
        target_asin = "B075X8471B"
        res = retriever.retrieve_reviews(target_asin, "battery and connection issues", top_k=5)
        
        assert res["product_id"] == target_asin
        assert len(res["retrieved_reviews"]) > 0
        for r in res["retrieved_reviews"]:
            assert r["parent_asin"] == target_asin, (
                f"Cross-product contamination detected: retrieved {r['parent_asin']} instead of {target_asin}"
            )

    def test_cross_domain_isolation(self, retriever):
        """Verify fashion queries do not retrieve electronics reviews and vice-versa."""
        fashion_asin = "B0956HNH2Y"
        res = retriever.retrieve_reviews(fashion_asin, "sizing and fabric quality", top_k=4)
        
        assert res["domain"] == "Fashion"
        for r in res["retrieved_reviews"]:
            assert r["parent_asin"] == fashion_asin
            assert r["domain"] == "Fashion"

    def test_query_intent_detection(self, retriever):
        """Test intent classification for complaint, praise, and general questions."""
        complaint_q1 = "Why are customers unhappy with this product?"
        complaint_q2 = "What are the biggest complaints and problems?"
        praise_q1 = "What do customers like about this product?"
        praise_q2 = "What are the best features loved by buyers?"
        general_q = "How is the battery life compared to expectation?"

        assert retriever.detect_query_intent(complaint_q1) == "complaints"
        assert retriever.detect_query_intent(complaint_q2) == "complaints"
        assert retriever.detect_query_intent(praise_q1) == "praises"
        assert retriever.detect_query_intent(praise_q2) == "praises"
        assert retriever.detect_query_intent(general_q) == "general"

    def test_complaint_prioritization(self, retriever):
        """Ensure complaint questions prioritize reviews with negative sentiment or low rating."""
        res = retriever.retrieve_reviews(
            "B075X8471B",
            "What are the biggest complaints?",
            top_k=4,
            prioritize_sentiment=True
        )
        assert res["intent"] == "complaints"
        # At least one review must be negative or low rating (<= 2)
        sentiments = [r["sentiment"] for r in res["retrieved_reviews"]]
        ratings = [r["rating"] for r in res["retrieved_reviews"]]
        assert any(s == "negative" or rat <= 2.0 for s, rat in zip(sentiments, ratings))

    def test_aspect_enrichment(self, retriever):
        """Verify that retrieved reviews have detected domain aspects attached."""
        res = retriever.retrieve_reviews("B075X8471B", "Why are customers unhappy?", top_k=3)
        assert "top_aspects" in res
        assert isinstance(res["top_aspects"], list)
        for r in res["retrieved_reviews"]:
            assert "aspects" in r
            assert "aspect_labels" in r
            assert isinstance(r["aspects"], list)

    def test_grounded_answer_structure(self, pipeline):
        """Verify that the generated answer contains required sections and evidence phrasing."""
        target_asin = "B075X8471B"
        query = "Why are customers unhappy with this product?"
        res = pipeline.answer_question(target_asin, query, top_k=4)

        answer = res["answer"]
        assert "Based on 4 relevant reviews" in answer
        assert "Sentiment Breakdown" in answer
        assert "Relevant Aspects" in answer
        assert "Representative Evidence" in answer
        assert res["product_id"] == target_asin

    def test_invalid_product_id(self, pipeline):
        """Verify clear error handling when querying non-existent product ID."""
        with pytest.raises(ValueError, match="not found"):
            pipeline.answer_question("NON_EXISTENT_ASIN_12345", "What is the issue?")

    def test_modular_llm_factory(self):
        """Verify LLM factory returns GroundedSynthesizerLLM by default."""
        llm = get_llm("grounded")
        assert isinstance(llm, GroundedSynthesizerLLM)
        
        llm_default = get_llm("unknown_provider")
        assert isinstance(llm_default, GroundedSynthesizerLLM)
