import os
import streamlit as st
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# We set protocol buffers implementation before import to avoid C++ mismatch
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from src.config import DATA_DIR, DB_DIR, RESEARCH_PAPERS, LLM_PROVIDER
from src.search import RAGRetriever
from src.generator import RAGGenerator
from src.ingest import extract_chunks_from_pdf, get_paper_display_name

# Set page configuration
st.set_page_config(
    page_title="LitQA - AI Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    code, pre, [class*="code-block"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
    }
    
    /* Title and Header styling */
    .glowing-title {
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 2.8rem;
        margin-top: 10px;
        margin-bottom: 2px;
        filter: drop-shadow(0 2px 10px rgba(139, 92, 246, 0.25));
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Preset Question Buttons */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px rgba(255, 255, 255, 0.08) solid !important;
        color: #e5e7eb !important;
        border-radius: 10px !important;
        padding: 10px 15px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: left !important;
        font-size: 0.95rem !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
        border-color: transparent !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.25);
    }
    div.stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Custom Sidebar Card */
    section[data-testid="stSidebar"] {
        background-color: #0d121f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Source Citations styling */
    .source-container {
        border-left: 3px solid #8b5cf6;
        background: rgba(139, 92, 246, 0.05);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Status indicators */
    .mode-badge {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-top: 5px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "preset_query" not in st.session_state:
    st.session_state.preset_query = None

# App Header
st.markdown("<div class='glowing-title'>📚 LitQA Research Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Interact semantically with AI research papers (Transformer, RAG, GPT-3, and more)</div>", unsafe_allow_html=True)

# SIDEBAR CONFIGURATION
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    
    # LLM Settings
    st.markdown("#### LLM Generation")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["Gemini", "OpenAI"],
        index=0 if LLM_PROVIDER == "gemini" else 1
    ).lower()
    
    # Dynamic API Key inputs
    api_key_input = ""
    if llm_provider == "gemini":
        api_key_input = st.text_input("Gemini API Key", type="password", help="Overrides GEMINI_API_KEY env variable if provided.")
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input
    else:
        api_key_input = st.text_input("OpenAI API Key", type="password", help="Overrides OPENAI_API_KEY env variable if provided.")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input

    # Retrieval Settings
    st.markdown("---")
    st.markdown("#### Retrieval Configuration")
    search_type = st.radio(
        "Search Strategy",
        options=["Semantic Similarity", "MMR (Diverse Chunks)"],
        index=0
    )
    search_strategy = "similarity" if search_type == "Semantic Similarity" else "mmr"
    
    retrieved_k = st.slider("Number of Chunks (k)", min_value=1, max_value=10, value=4, help="Number of context snippets to feed to the LLM.")
    
    use_reranker = st.toggle("Enable Cross-Encoder Reranking", value=False, help="Uses a secondary model to re-score and rank retrieved results.")
    
    # Dynamic Document Ingestion / Uploads
    st.markdown("---")
    st.markdown("#### 📤 Upload Research Paper")
    uploaded_file = st.file_uploader("Upload custom PDF", type=["pdf"])
    
    # Create DB dir if needed
    if not DB_DIR.exists() or not any(DB_DIR.iterdir()):
        st.warning("⚠️ Vector store is empty. Please run ingestion to load default papers.")
        if st.button("🚀 Ingest Default Papers"):
            with st.spinner("Ingesting papers..."):
                from src.ingest import ingest_all_papers
                ingest_all_papers(force=True)
            st.success("Default papers successfully ingested!")
            st.rerun()

# Initialize Retriever and Generator
@st.cache_resource(show_spinner=False)
def get_rag_components(rerank):
    # Re-instantiate if reranker toggle changes
    retriever = RAGRetriever(use_reranker=rerank)
    generator = RAGGenerator()
    return retriever, generator

retriever, generator = get_rag_components(use_reranker)

# Handle Uploaded File
if uploaded_file is not None:
    dest_path = DATA_DIR / uploaded_file.name
    if not dest_path.exists():
        with st.spinner(f"Ingesting uploaded paper: {uploaded_file.name}..."):
            # Write to disk
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Extract chunks
            try:
                docs = extract_chunks_from_pdf(dest_path)
                if docs:
                    # Add to vector DB
                    retriever.vector_store.add_documents(docs)
                    st.toast(f"✅ Successfully ingested {len(docs)} chunks from {uploaded_file.name}!", icon="🎉")
                else:
                    st.error("No text could be extracted from this PDF.")
            except Exception as e:
                st.error(f"Error parsing PDF: {e}")
                if dest_path.exists():
                    dest_path.unlink()

# Display current documents indexed
with st.sidebar:
    st.markdown("---")
    st.markdown("#### 📄 Indexed Documents")
    # List PDFs in DATA_DIR
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if pdf_files:
        for f in pdf_files:
            display_name = get_paper_display_name(f.name)
            st.markdown(f"- 📄 **{display_name}** (`{f.name}`)")
    else:
        st.markdown("*None indexed yet.*")

# PRESET QUESTION BUTTONS
st.markdown("#### 🎯 Quick Test Presets")
presets = [
    "What are the main components of a RAG model, and how do they interact?",
    "What are the two sub-layers in each encoder layer of the Transformer model?",
    "Explain how positional encoding is implemented in Transformers and why it is necessary.",
    "Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?",
    "What is few-shot learning, and how does GPT-3 implement it during inference?"
]

col1, col2 = st.columns(2)
with col1:
    for q in presets[:3]:
        if st.button(q, key=q):
            st.session_state.preset_query = q
with col2:
    for q in presets[3:]:
        if st.button(q, key=q):
            st.session_state.preset_query = q

# Chat Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            # Display source citations
            st.markdown("<span class='mode-badge'>Generation: " + message.get("mode", "Unknown") + "</span>", unsafe_allow_html=True)
            with st.expander("🔍 Citations & Evidence Chunks"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1}: {src['paper_title']}** | *{src['section']}* (Page {src['page']})")
                    st.markdown(f"<div class='source-container'>{src['text']}</div>", unsafe_allow_html=True)

# Question handling flow
query = st.chat_input("Ask a question about the research papers...")

# Trigger query if preset is clicked
if st.session_state.preset_query:
    query = st.session_state.preset_query
    st.session_state.preset_query = None

if query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    # Process RAG
    with st.chat_message("assistant"):
        with st.spinner("Searching vector database..."):
            # Update provider in generator in case user changed it dynamically
            generator.provider = llm_provider
            generator.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            generator.openai_api_key = os.getenv("OPENAI_API_KEY")
            if generator.gemini_api_key:
                genai.configure(api_key=generator.gemini_api_key)
            if generator.openai_api_key and not generator.openai_client:
                try:
                    from openai import OpenAI
                    generator.openai_client = OpenAI(api_key=generator.openai_api_key)
                except ImportError:
                    pass

            # Search ChromaDB
            retrieved_docs = retriever.search(
                query=query,
                search_type=search_strategy,
                k=retrieved_k,
                fetch_k=retrieved_k * 4
            )
            
        with st.spinner("Formulating grounded response..."):
            # Generate answer
            response_dict = generator.generate_answer(query, retrieved_docs)
            
        # Display answer
        st.markdown(response_dict["answer"])
        
        # Display badge
        st.markdown("<span class='mode-badge'>Generation: " + response_dict["mode"] + "</span>", unsafe_allow_html=True)
        
        # Display collapsible citations
        if response_dict["sources"]:
            with st.expander("🔍 Citations & Evidence Chunks"):
                for idx, src in enumerate(response_dict["sources"]):
                    st.markdown(f"**Source {idx+1}: {src['paper_title']}** | *{src['section']}* (Page {src['page']})")
                    st.markdown(f"<div class='source-container'>{src['text']}</div>", unsafe_allow_html=True)
                    
        # Append assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_dict["answer"],
            "sources": response_dict["sources"],
            "mode": response_dict["mode"]
        })
