import os
from pypdf import PdfReader
from docx import Document

def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extracts text from various file formats based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in ['.txt', '.md', '.json', '.html']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
            
    elif ext == '.pdf':
        text_content = []
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        except Exception as e:
            raise ValueError(f"Failed to read PDF file: {e}")
        return "\n".join(text_content)
        
    elif ext == '.docx':
        text_content = []
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text_content.append(paragraph.text)
        except Exception as e:
            raise ValueError(f"Failed to read DOCX file: {e}")
        return "\n".join(text_content)
        
    else:
        # Fallback to reading as text file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Unsupported file format: {ext}. Error during raw read: {e}")

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """Splits a single string of text into list of chunks with overlapping regions."""
    if not text.strip():
        return []
        
    chunks = []
    # Standard sliding window approach based on characters
    # If chunk_size <= chunk_overlap, set standard overlap
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 10
        
    step = chunk_size - chunk_overlap
    
    for i in range(0, len(text), step):
        chunk_str = text[i:i + chunk_size].strip()
        if chunk_str:
            chunks.append({
                "content": chunk_str,
                "start_idx": i,
                "end_idx": i + len(chunk_str)
            })
            
    return chunks
