"""
Dedicated scraper for the Contisoft Technologies website.
"""

import os
import sys
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Import our modules
from src.embeddings import EmbeddingModel
from src.vectordb import VectorDB
from src.rag import RAG

# Configuration
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
HEADERS = {
    'User-Agent': USER_AGENT,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0'
}

# Define the base URL
BASE_URL = 'https://www.contisofttechno.com'

# List of known product pages to ensure we scrape them
PRODUCT_PAGES = [
    '/index.html',
    '/e-Stamp-software.html',
    '/bank-guarantte-india.html',
    '/legal-contract-management-system.html',
    '/Document-Management-Software.html',
    '/procurement-india.html',
    '/About-Us.html',
    '/contact-us.html',
    '/supplier-performance-evaluation-software.html',
    '/supplier-material-tracking-followup-software.html',
    '/conti-auction.html',
    '/vendor-compliance-monitoring-software.html'
]

def is_valid_url(url):
    """Check if the URL is valid and belongs to the target domain."""
    try:
        parsed = urlparse(url)
        return (
            parsed.netloc == urlparse(BASE_URL).netloc or 
            not parsed.netloc and urlparse(urljoin(BASE_URL, url)).netloc == urlparse(BASE_URL).netloc
        )
    except Exception:
        return False

def normalize_url(url):
    """Normalize a URL by resolving relative URLs and removing fragments."""
    full_url = urljoin(BASE_URL, url)
    parsed = urlparse(full_url)
    # Remove fragments
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

def get_page_content(url):
    """
    Get the content of a web page.
    
    Args:
        url (str): URL to get content from
        
    Returns:
        tuple: (title, content, links)
    """
    try:
        # Normalize the URL
        full_url = normalize_url(url)
        
        # Additional retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add a small delay to avoid overwhelming the server
                time.sleep(1)
                
                response = requests.get(
                    full_url, 
                    headers=HEADERS, 
                    timeout=15
                )
                response.raise_for_status()
                
                # Parse the content with BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Get the title
                title = soup.title.string if soup.title else url.split('/')[-1]
                
                # Remove script, style and other non-content elements
                for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
                    element.decompose()
                
                # Extract the text content
                text = soup.get_text(separator=' ')
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                content = ' '.join(lines)
                
                # Get all links
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href and not href.startswith('#') and not href.startswith('javascript:'):
                        if is_valid_url(href):
                            links.append(normalize_url(href))
                
                print(f"Successfully scraped {full_url}")
                print(f"  Title: {title}")
                print(f"  Content length: {len(content)} characters")
                print(f"  Links found: {len(links)}")
                
                return title, content, links
                
            except (requests.RequestException, Exception) as e:
                print(f"Attempt {attempt+1}/{max_retries} failed for {full_url}: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"Failed to get {full_url} after {max_retries} attempts")
                    return None, None, []
                time.sleep(2)  # Wait before retrying
    
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None, None, []

def scrape_website():
    """
    Scrape the Contisoft Technologies website.
    
    Returns:
        list: List of scraped pages
    """
    # Initialize variables
    visited_urls = set()
    to_visit = set()
    scraped_data = []
    
    # Start with the product pages
    for page in PRODUCT_PAGES:
        to_visit.add(normalize_url(page))
    
    # Also add the base URL
    to_visit.add(BASE_URL)
    
    # Set a limit to avoid infinite loops
    max_pages = 50
    
    # Scrape the website
    print(f"Starting to scrape {BASE_URL}...")
    print(f"Will scrape up to {max_pages} pages")
    
    while to_visit and len(scraped_data) < max_pages:
        # Get the next URL to visit
        url = to_visit.pop()
        
        # Skip if we've already visited this URL
        if url in visited_urls:
            continue
        
        # Mark as visited
        visited_urls.add(url)
        
        # Get the page content
        title, content, links = get_page_content(url)
        
        # Skip if we couldn't get the content
        if not content:
            continue
        
        # Add to scraped data
        scraped_data.append({
            'url': url,
            'title': title,
            'content': content
        })
        
        print(f"Scraped {len(scraped_data)}/{max_pages} pages")
        
        # Add new links to visit
        for link in links:
            if link not in visited_urls and len(visited_urls) < max_pages:
                to_visit.add(link)
    
    print(f"Scraping completed. Scraped {len(scraped_data)} pages")
    
    # Save the scraped data
    with open('scraped_contisoft.json', 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved scraped data to scraped_contisoft.json")
    
    return scraped_data

def process_for_rag(scraped_data):
    """
    Process the scraped data for the RAG system.
    
    Args:
        scraped_data (list): List of scraped pages
        
    Returns:
        list: List of processed documents
    """
    # Initialize the embedding model
    embedding_model = EmbeddingModel()
    
    # Process the scraped data
    documents = []
    
    print("Processing scraped data...")
    
    for page in scraped_data:
        # Create a document
        document = {
            'content': page['content'],
            'metadata': {
                'source': page['url'],
                'title': page['title']
            }
        }
        documents.append(document)
    
    print(f"Generated {len(documents)} documents")
    
    # Generate embeddings
    print("Generating embeddings...")
    embedded_documents = embedding_model.embed_documents(documents)
    print(f"Generated embeddings for {len(embedded_documents)} documents")
    
    return embedded_documents

def add_to_database(embedded_documents):
    """
    Add the embedded documents to the vector database.
    
    Args:
        embedded_documents (list): List of embedded documents
        
    Returns:
        int: Number of documents added
    """
    # Initialize the vector database
    vectordb = VectorDB()
    
    # Add the documents
    print("Adding documents to vector database...")
    vectordb.add_documents(embedded_documents)
    
    # Get collection stats
    stats = vectordb.get_collection_stats()
    doc_count = stats.get('document_count', 0)
    
    print(f"Added {len(embedded_documents)} documents to vector database")
    print(f"Vector database now contains {doc_count} documents")
    
    return len(embedded_documents)

def test_rag():
    """
    Test the RAG system.
    
    Returns:
        dict: Test results
    """
    # Initialize the RAG system
    rag = RAG()
    
    # Test queries
    test_queries = [
        "What is e-Stamp software?",
        "What services does Contisoft Technologies offer?",
        "What is the contract management system?",
        "Tell me about vendor compliance monitoring",
        "How to contact Contisoft Technologies?"
    ]
    
    # Test results
    results = {}
    
    print("Testing the RAG system...")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Get the answer
        result = rag.query(query, mode="deep")
        
        # Print the answer
        print(f"Answer: {result['answer']}")
        
        # Store the result
        results[query] = result
    
    return results

def main():
    """Main function."""
    # Check if Ollama is running
    try:
        # First check if Ollama is installed and running
        from fix_ollama import fix_ollama_issues
        fix_ollama_issues()
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        print("Continuing anyway...")
    
    # Scrape the website
    scraped_data = scrape_website()
    
    if not scraped_data:
        print("No data was scraped. Exiting.")
        return
    
    # Process the data for RAG
    embedded_documents = process_for_rag(scraped_data)
    
    if not embedded_documents:
        print("No embeddings were generated. Exiting.")
        return
    
    # Add to the database
    docs_added = add_to_database(embedded_documents)
    
    # Test the RAG system
    test_results = test_rag()
    
    print("\nScraping and processing completed successfully!")
    print(f"Added {docs_added} documents to the RAG system")
    print("\nYou can now ask questions about Contisoft Technologies using the RAG system")

if __name__ == "__main__":
    main() 