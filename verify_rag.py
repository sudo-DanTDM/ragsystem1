import os
import sys
from src.search import RAGRetriever
from src.generator import RAGGenerator

def run_tests():
    print("Initializing RAG verification tests...")
    
    # 1. Test Retriever initialization
    try:
        retriever = RAGRetriever(use_reranker=False)
        print("[OK] Retriever successfully initialized.")
    except Exception as e:
        print(f"[FAIL] Failed to initialize retriever: {e}")
        sys.exit(1)
        
    # Sample Test Queries
    queries = [
        "What are the main components of a RAG model, and how do they interact?",
        "What are the two sub-layers in each encoder layer of the Transformer model?",
        "Explain how positional encoding is implemented in Transformers and why it is necessary."
    ]
    
    # Run retrieval verification
    for q in queries:
        print(f"\nTesting Query: '{q}'")
        try:
            docs = retriever.search(q, search_type="similarity", k=3)
            print(f"  [OK] Retrieved {len(docs)} document chunks.")
            if len(docs) > 0:
                print(f"  Top Match Source: {docs[0].metadata.get('source')}")
                print(f"  Top Match Section: {docs[0].metadata.get('section')}")
                print(f"  Top Match Page: {docs[0].metadata.get('page')}")
                assert docs[0].metadata.get("source") is not None, "Metadata source is missing"
                assert docs[0].metadata.get("section") is not None, "Metadata section is missing"
                assert docs[0].metadata.get("page") is not None, "Metadata page is missing"
            else:
                print("  [FAIL] No chunks retrieved!")
                sys.exit(1)
        except Exception as e:
            print(f"  [FAIL] Search failed: {e}")
            sys.exit(1)
            
    # 2. Test Generator (Offline / Online check)
    print("\nTesting Generator...")
    try:
        generator = RAGGenerator()
        print("[OK] Generator successfully initialized.")
        # Retrieve context for query
        docs = retriever.search(queries[1], search_type="similarity", k=2)
        response = generator.generate_answer(queries[1], docs)
        print(f"  Mode: {response['mode']}")
        print(f"  Answer Snippet:\n{response['answer'][:300]}...")
        assert "answer" in response, "Response is missing 'answer'"
        assert "sources" in response, "Response is missing 'sources'"
        assert len(response["sources"]) > 0, "Response is missing sources list"
        print("\nAll local RAG component checks PASSED successfully!")
    except Exception as e:
        print(f"[FAIL] Generator test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
