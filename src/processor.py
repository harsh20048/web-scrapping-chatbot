"""
Text processor module for cleaning and chunking text data.
This module prepares scraped content for embedding and retrieval.
"""

import re
from typing import List, Dict, Any


class TextProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """
        Initialize the text processor.
        
        Args:
            chunk_size (int): Maximum size of each text chunk
            chunk_overlap (int): Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing extra whitespace and unwanted characters.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove multiple spaces, newlines, and tabs
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags that might have been missed
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # Remove special characters that add little value
        text = re.sub(r'[^\w\s.,!?:;()\-\']', ' ', text)
        
        # Normalize whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into coherent chunks: first by paragraphs, then by sentences, then by fixed size if needed.
        Args:
            text (str): Text to split into chunks
        Returns:
            List[str]: List of text chunks
        """
        import re
        # If text is shorter than chunk_size, return it as a single chunk
        if len(text) < self.chunk_size:
            return [text]

        # Guard against excessively large pages
        MAX_TEXT_LENGTH = 100_000
        MAX_CHUNKS = 200
        if len(text) > MAX_TEXT_LENGTH:
            print(f"[WARNING] Skipping page: cleaned text too large (length={len(text)})")
            return []

        # Step 1: Split by paragraph (two or more newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Step 2: If paragraph is too long, split by sentence boundaries
            if len(para) > self.chunk_size:
                # Split by sentence (using punctuation)
                sentences = re.split(r'(?<=[.!?]) +', para)
                current_chunk = ''
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                        current_chunk += (' ' if current_chunk else '') + sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(para)

        # Step 3: If any chunk is still too long, split by fixed size
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Split by fixed size with overlap
                start = 0
                while start < len(chunk):
                    end = min(start + self.chunk_size, len(chunk))
                    final_chunks.append(chunk[start:end])
                    start = end - self.chunk_overlap if (end - self.chunk_overlap > start) else end

        # Limit to MAX_CHUNKS
        if len(final_chunks) > MAX_CHUNKS:
            print(f"[WARNING] Reached max chunk limit ({MAX_CHUNKS}) for this page. Truncating.")
            final_chunks = final_chunks[:MAX_CHUNKS]
        return [c.strip() for c in final_chunks if c.strip()]
    
    def process_page(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a page by cleaning and chunking its content.
        
        Args:
            page_data (Dict): Page data including content, title, and URL
            
        Returns:
            List[Dict]: List of document chunks with metadata
        """
        cleaned_text = self.clean_text(page_data['content'])
        chunks = self.chunk_text(cleaned_text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                'content': chunk,
                'metadata': {
                    'source': page_data['url'],
                    'title': page_data['title'],
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            }
            documents.append(doc)
        
        return documents
    
    def process_all_pages(self, pages_content: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all scraped pages.
        
        Args:
            pages_content (Dict): Dictionary of scraped pages
            
        Returns:
            List[Dict]: List of all document chunks with metadata
        """
        all_documents = []
        
        for url, page_data in pages_content.items():
            page_documents = self.process_page(page_data)
            all_documents.extend(page_documents)
        
        return all_documents


if __name__ == "__main__":
    # Example usage
    test_text = """
    This is a test paragraph with multiple sentences. It should be chunked properly. 
    This is another paragraph that should be in a separate chunk if possible. 
    Here's a third paragraph with more content to test the chunking functionality.
    """
    
    processor = TextProcessor(chunk_size=100, chunk_overlap=20)
    cleaned = processor.clean_text(test_text)
    chunks = processor.chunk_text(cleaned)
    
    print(f"Cleaned text: {cleaned[:100]}...")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:50]}...")
