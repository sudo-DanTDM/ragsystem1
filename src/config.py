import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# Chunking & Embeddings Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# LLM Configurations
# Defaults to gemini, can be overridden by environment
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower() 
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Target research papers to preload
RESEARCH_PAPERS = {
    "Attention Is All You Need": {
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
        "filename": "attention_is_all_you_need.pdf",
        "display_name": "Attention Is All You Need (Transformer)"
    },
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": {
        "url": "https://arxiv.org/pdf/2005.11401.pdf",
        "filename": "rag_paper.pdf",
        "display_name": "Retrieval-Augmented Generation (RAG)"
    },
    "Language Models are Few-Shot Learners": {
        "url": "https://arxiv.org/pdf/2005.14165.pdf",
        "filename": "gpt3_paper.pdf",
        "display_name": "Language Models are Few-Shot Learners (GPT-3)"
    }
}
