"""
Script to scrape a website and add its content to the RAG system.
"""

import time
import sys
from src.rag import RAG

def scrape_website(url):
    """
    Scrape a website and add it to the RAG database.
    
    Args:
        url (str): The URL to scrape
    """
    print(f"Scraping website: {url}")
    
    # Initialize RAG system
    rag = RAG()
    
    # Get current stats
    before_stats = rag.vectordb.get_collection_stats()
    before_count = before_stats.get('document_count', 0)
    print(f"Database contains {before_count} documents before scraping")
    
    # Start scraping
    start_time = time.time()
    print("Starting scraping process...")
    
    # Scrape and process the website
    docs_added = rag.scrape_website(url)
    
    # Calculate time taken
    elapsed_time = time.time() - start_time
    
    # Get updated stats
    after_stats = rag.vectordb.get_collection_stats()
    after_count = after_stats.get('document_count', 0)
    
    # Print results
    print(f"\nScraping completed in {elapsed_time:.2f} seconds")
    print(f"Documents added: {docs_added}")
    print(f"Database now contains {after_count} documents (added {after_count - before_count})")
    
    # Test a query
    print("\nTesting a query with the new data:")
    result = rag.query("What is this website about?", mode="deep")
    print(f"Answer: {result['answer']}")
    
    return docs_added

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scrape_website.py <url>")
        print("Example: python scrape_website.py https://example.com")
        sys.exit(1)
        
    url = sys.argv[1]
    scrape_website(url) 