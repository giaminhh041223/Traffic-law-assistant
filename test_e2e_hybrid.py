"""E2E integration test for the upgraded Hybrid AI RAG pipeline.

Tests the full pipeline with:
  1. QueryRewriter (slang → formal terms)
  2. QueryDeconstructor (compound query splitting)
  3. AIRouter (complexity evaluation)
  4. CoT thought parsing
  5. Citation validation

Requires: Ollama running with qwen2.5:1.5b model loaded.
"""
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.phase4_generation.rag_chain import TrafficLawRAG

def main():
    print("=" * 70)
    print("🚀  HYBRID AI RAG SYSTEM — E2E INTEGRATION TEST")
    print("=" * 70)

    # --- Boot the full pipeline ---
    print("\n⏳ Loading full pipeline (retriever + reranker + LLM + new modules)...")
    try:
        rag = TrafficLawRAG.from_configs()
    except Exception as e:
        print(f"❌ Pipeline boot failed: {e}")
        return

    print("✅ Pipeline loaded successfully!")
    print(f"   ├── Rewriter:      {type(rag.rewriter).__name__}")
    print(f"   ├── Deconstructor: {type(rag.deconstructor).__name__}")
    print(f"   ├── Router:        {type(rag.router).__name__}")
    print(f"   ├── Retriever:     {type(rag.retriever).__name__}")
    print(f"   ├── Reranker:      {type(rag.reranker).__name__}")
    print(f"   └── LLM:          {type(rag.llm).__name__}")

    # --- Test queries ---
    test_queries = [
        # (1) Simple slang query → Rewriter maps "vượt đèn đỏ" + "xe máy"
        "Đi xe máy vượt đèn đỏ bị phạt bao nhiêu tiền?",
        # (2) Compound multi-intent query → Deconstructor splits & parallel search
        "Đi xe máy vượt đèn đỏ và không mang bằng lái bị xử phạt thế nào?",
        # (3) Cloud-complexity query → Router flags as 'cloud'
        "Lái xe ô tô gây tai nạn chết người rồi bỏ trốn bị xử phạt ra sao?",
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"📝 TEST QUERY {i}: {q}")
        print(f"{'─' * 70}")

        result = rag.run(q)

        # Timings
        print(f"\n⏱️  Timings:")
        for stage, t in result.timings.items():
            print(f"   ├── {stage}: {t:.3f}s")

        # Thought (CoT)
        if result.thought:
            print(f"\n💭 CoT Thought (internal):")
            for line in result.thought.split("\n")[:5]:
                print(f"   │ {line.strip()}")
            if result.thought.count("\n") > 5:
                print(f"   │ ... ({result.thought.count(chr(10))+1} lines total)")

        # Answer
        print(f"\n✅ Answer:")
        print(f"   {result.answer[:500]}")
        if len(result.answer) > 500:
            print(f"   ... (truncated, {len(result.answer)} chars)")

        # Sources
        print(f"\n📑 Sources ({len(result.sources)} chunks):")
        for j, s in enumerate(result.sources[:3], 1):
            ce_score = s.get("cross_encoder_score", "N/A")
            chunk_id = s.get("chunk_id", "?")
            text_preview = (s.get("text") or "")[:80]
            print(f"   #{j}  CE={ce_score:.4f}  {chunk_id}")
            print(f"       {text_preview}...")

        # Validation
        if result.validation:
            print(f"\n🔍 Validation: {'✅ PASS' if result.validation.is_valid else '⚠️ FAILED'}")
            if result.validation.supported_citations:
                print(f"   ├── Supported citations: {len(result.validation.supported_citations)}")
            if result.validation.hallucinated_citations:
                print(f"   ├── Hallucinated citations: {result.validation.hallucinated_citations}")
            if result.validation.failures:
                print(f"   └── Failures: {result.validation.failures}")

    print(f"\n{'=' * 70}")
    print("🎉  ALL E2E TESTS COMPLETED")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
