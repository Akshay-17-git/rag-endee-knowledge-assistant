"""
utils/document_processor.py
Handles document text extraction and chunking into overlapping segments.
"""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text string
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def chunk_text_by_words(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """
    Split text into overlapping chunks by word count.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum number of words per chunk
        overlap: Number of overlapping words between consecutive chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    
    return chunks


def process_document(document: str, source_name: str = "document", chunk_size: int = 150) -> List[Dict[str, Any]]:
    """
    Process a document into chunks with metadata.
    
    Args:
        document: Raw document text
        source_name: Name/identifier for the source document
        chunk_size: Maximum words per chunk
        
    Returns:
        List of chunk dictionaries with 'id', 'text', 'source', and 'chunk_id'
    """
    # Clean the text
    cleaned_text = clean_text(document)
    
    # Chunk the text
    text_chunks = chunk_text_by_words(cleaned_text, chunk_size=chunk_size, overlap=30)
    
    # Create chunk dictionaries with metadata
    chunks = []
    for idx, text in enumerate(text_chunks):
        chunk_id = f"{source_name}_chunk_{idx}"
        chunks.append({
            "id": chunk_id,
            "text": text,
            "source": source_name,
            "chunk_id": idx
        })
    
    return chunks


def extract_text_from_pdf(file) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file: Streamlit uploaded file object
        
    Returns:
        Extracted text string
    """
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except ImportError:
        # Fallback to PyPDF2
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"


def extract_text_from_txt(file) -> str:
    """
    Extract text from a text file.
    
    Args:
        file: Streamlit uploaded file object
        
    Returns:
        Extracted text string
    """
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error extracting text: {str(e)}"
