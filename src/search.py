import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import DB_DIR, EMBEDDING_MODEL_NAME

class RAGRetriever:
    def __init__(self, use_reranker=False, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.embedding_func = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=self.embedding_func
        )
        self.use_reranker = use_reranker
        self._reranker = None
        self.reranker_model = reranker_model

    @property
    def reranker(self):
        if self.use_reranker and self._reranker is None:
            print(f"Loading reranking model: {self.reranker_model}...")
            self._reranker = CrossEncoder(self.reranker_model, device="cpu")
        return self._reranker

    def search(self, query, search_type="similarity", k=5, fetch_k=20, lambda_mult=0.5):
        """
        Performs retrieval on the vector store.
        
        Args:
            query (str): The search query.
            search_type (str): 'similarity' or 'mmr'.
            k (int): Number of final documents to return.
            fetch_k (int): Number of documents to fetch for MMR or reranking.
            lambda_mult (float): MMR diversity factor (0.0 to 1.0).
            
        Returns:
            list: List of Document objects.
        """
        # If reranking is enabled, fetch more candidates to rerank
        actual_k = fetch_k if (self.use_reranker and search_type == "similarity") else k
        
        if search_type == "mmr":
            # MMR search
            docs = self.vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
        else:
            # Standard similarity search
            docs = self.vector_store.similarity_search(
                query,
                k=actual_k
            )
            
        # Rerank if enabled
        if self.use_reranker and self.reranker and docs:
            # Pair query with document content
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)
            
            # Sort documents by score
            doc_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            
            # Take top k
            docs = [doc for doc, score in doc_scores[:k]]
            
        return docs

if __name__ == "__main__":
    # Test retrieval
    retriever = RAGRetriever(use_reranker=False)
    query = "What is multi-head attention?"
    docs = retriever.search(query, search_type="similarity", k=3)
    print(f"Retrieved {len(docs)} documents for query: '{query}'")
    for doc in docs:
        print(f"\nSource: {doc.metadata['source']} | Section: {doc.metadata['section']} | Page: {doc.metadata['page']}")
        print(f"Content snippet: {doc.page_content[:150]}...")
