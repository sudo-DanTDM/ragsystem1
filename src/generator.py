import os
import google.generativeai as genai
from src.config import LLM_PROVIDER, GEMINI_MODEL, OPENAI_MODEL

class RAGGenerator:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Configure Gemini
        if self.gemini_api_key:
            # We set the environment variable PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION to python
            # before import to prevent protobuf C++ mismatch crash in google-generativeai
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            genai.configure(api_key=self.gemini_api_key)
            
        # Configure OpenAI
        self.openai_client = None
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            except ImportError:
                print("OpenAI package not installed. Cannot use OpenAI provider.")

    def generate_answer(self, question, retrieved_docs):
        """
        Generates an answer to the question using the retrieved documents as context.
        
        Args:
            question (str): The user's query.
            retrieved_docs (list): List of Document objects retrieved from vector store.
            
        Returns:
            dict: A dictionary containing 'answer', 'sources', and 'mode'.
        """
        # Format sources
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "paper_title": doc.metadata.get("source", "Unknown Paper"),
                "section": doc.metadata.get("section", "Unknown Section"),
                "page": doc.metadata.get("page", "Unknown Page"),
                "text": doc.page_content
            })
            
        # If no documents are retrieved, return early
        if not retrieved_docs:
            return {
                "answer": "No relevant documents were retrieved from the database to answer this question.",
                "sources": [],
                "mode": "No Context"
            }
            
        # Format context for prompt
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            context_str += f"Snippet {i+1} [Source: {doc.metadata.get('source')} | Section: {doc.metadata.get('section')} | Page: {doc.metadata.get('page')}]:\n"
            context_str += f"{doc.page_content}\n\n"
            
        # Define Prompt
        prompt = f"""You are an expert AI research assistant. Your task is to answer the user's question accurately using only the provided context snippets from scientific research papers.

Rules:
1. Ground your answer completely in the provided context snippets.
2. If the context does not contain enough information to answer the question, state: "I'm sorry, but the provided research papers do not contain enough information to answer your question."
3. Do NOT make up, assume, or extrapolate information. Do NOT use outside knowledge not directly stated in the context.
4. Keep the answer clear, precise, professional, and well-structured.

Context Snippets:
{context_str}

Question:
{question}

Answer:"""

        # Check for API Keys
        # 1. Gemini
        if self.provider == "gemini" and self.gemini_api_key:
            try:
                model = genai.GenerativeModel(GEMINI_MODEL)
                response = model.generate_content(prompt)
                return {
                    "answer": response.text.strip(),
                    "sources": sources,
                    "mode": f"Gemini ({GEMINI_MODEL})"
                }
            except Exception as e:
                return {
                    "answer": f"Error generating answer using Gemini API: {str(e)}",
                    "sources": sources,
                    "mode": "Error"
                }
                
        # 2. OpenAI
        elif self.provider == "openai" and self.openai_api_key:
            if not self.openai_client:
                return {
                    "answer": "OpenAI client is not initialized. Please ensure the 'openai' library is installed.",
                    "sources": sources,
                    "mode": "Error"
                }
            try:
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a professional assistant that answers questions based strictly on retrieved research snippets."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                return {
                    "answer": response.choices[0].message.content.strip(),
                    "sources": sources,
                    "mode": f"OpenAI ({OPENAI_MODEL})"
                }
            except Exception as e:
                return {
                    "answer": f"Error generating answer using OpenAI API: {str(e)}",
                    "sources": sources,
                    "mode": "Error"
                }
                
        # 3. Offline/Demo Mode
        else:
            # We don't have keys, but we can return a friendly prompt showing we retrieved the contents correctly
            demo_msg = (
                "⚠️ **API Key Not Configured (Offline Demo Mode)**\n\n"
                "To get natural language answers, please provide a `GEMINI_API_KEY` or `OPENAI_API_KEY` in your `.env` file.\n\n"
                "However, the semantic search has successfully retrieved the most relevant snippets from the research papers. "
                "Here are the top matches that would be fed to the LLM:\n\n"
            )
            for i, doc in enumerate(retrieved_docs[:2]):
                demo_msg += f"**From: {doc.metadata.get('source')} (Section: {doc.metadata.get('section')}, Page {doc.metadata.get('page')})**\n"
                demo_msg += f"> {doc.page_content[:300]}...\n\n"
                
            return {
                "answer": demo_msg,
                "sources": sources,
                "mode": "Offline/Demo"
            }
