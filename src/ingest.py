import os
import re
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import DATA_DIR, DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME, RESEARCH_PAPERS

def get_paper_display_name(filename):
    for title, info in RESEARCH_PAPERS.items():
        if info["filename"] == filename:
            return title
    return filename

def extract_chunks_from_pdf(pdf_path):
    """
    Extracts text from PDF page-by-page, tracks section headers,
    splits text into chunks, and returns a list of Document objects with metadata.
    """
    reader = PdfReader(pdf_path)
    paper_title = get_paper_display_name(pdf_path.name)
    
    # Pattern to match numbered headings (e.g., '3.2.2 Multi-Head Attention')
    section_pattern = re.compile(r'^\s*(\d+(?:\.\d+)*)\s+([A-Z][a-zA-Z\s,\-\:\(\)\/]+)$')
    title_less_sections = ["Abstract", "Introduction", "Conclusion", "References", "Discussion"]
    
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    current_section = "Abstract"
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text()
        if not text:
            continue
            
        lines = text.split("\n")
        page_blocks = []
        current_block_lines = []
        
        for line in lines:
            line_str = line.strip()
            
            # Check for standard numbered section header
            match = section_pattern.match(line_str)
            if match:
                # Save previous block accumulated on this page
                if current_block_lines:
                    block_text = "\n".join(current_block_lines).strip()
                    if block_text:
                        page_blocks.append((current_section, block_text))
                    current_block_lines = []
                
                # Update the active section
                current_section = f"{match.group(1)} {match.group(2)}".strip()
                current_block_lines.append(line)
            # Check for title-less section headers
            elif line_str in title_less_sections:
                if current_block_lines:
                    block_text = "\n".join(current_block_lines).strip()
                    if block_text:
                        page_blocks.append((current_section, block_text))
                    current_block_lines = []
                
                current_section = line_str
                current_block_lines.append(line)
            else:
                current_block_lines.append(line)
                
        # Append the final block remaining on the page
        if current_block_lines:
            block_text = "\n".join(current_block_lines).strip()
            if block_text:
                page_blocks.append((current_section, block_text))
                
        # Split each block into chunks and create Document objects
        for sec, block_text in page_blocks:
            chunks = text_splitter.split_text(block_text)
            for chunk in chunks:
                if not chunk.strip():
                    continue
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": paper_title,
                        "filename": pdf_path.name,
                        "section": sec,
                        "page": page_num
                    }
                )
                documents.append(doc)
                
    return documents

def ingest_all_papers(force=False):
    """
    Ingests all PDFs from the data directory into ChromaDB.
    """
    print("Starting document ingestion...")
    
    # Initialize the local embedding function
    embedding_func = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    # If DB already exists, load it directly (unless force is True)
    if DB_DIR.exists() and any(DB_DIR.iterdir()) and not force:
        print(f"Vector database already exists at {DB_DIR}. Skipping ingestion.")
        try:
            vector_store = Chroma(persist_directory=str(DB_DIR), embedding_function=embedding_func)
            doc_count = len(vector_store.get()["ids"])
            print(f"Loaded existing vector store containing {doc_count} chunks.")
            return vector_store
        except Exception as e:
            print(f"Error loading existing vector database: {e}. Re-ingesting...")
            
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found to ingest. Please run downloader first.")
        return None
        
    all_docs = []
    for pdf_path in pdf_files:
        print(f"Parsing {pdf_path.name}...")
        try:
            docs = extract_chunks_from_pdf(pdf_path)
            all_docs.extend(docs)
            print(f"Extracted {len(docs)} chunks from {pdf_path.name}")
        except Exception as e:
            print(f"Failed to parse {pdf_path.name}: {e}")
            
    if not all_docs:
        print("No chunks extracted from documents.")
        return None
        
    print(f"Creating vector database with {len(all_docs)} total chunks...")
    try:
        vector_store = Chroma.from_documents(
            documents=all_docs,
            embedding=embedding_func,
            persist_directory=str(DB_DIR)
        )
        print("Vector database successfully initialized and saved!")
        return vector_store
    except Exception as e:
        print(f"Error creating vector database: {e}")
        return None

if __name__ == "__main__":
    ingest_all_papers()
