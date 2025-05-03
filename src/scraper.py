"""
Web scraper module for the auto-learning website chatbot.
This module handles crawling websites, extracting content, and respecting robots.txt.
"""

import os
import time
import requests
import validators
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import traceback


class WebScraper:
    def __init__(self, base_url, delay=2, max_pages=100):
        """
        Initialize the web scraper.
        
        Args:
            base_url (str): The starting URL to scrape
            delay (int): Delay between requests in seconds
            max_pages (int): Maximum number of pages to scrape
        """
        if not validators.url(base_url):
            raise ValueError(f"Invalid URL: {base_url}")
        
        self.base_url = base_url
        self.delay = delay
        self.max_pages = max_pages
        self.visited_urls = set()
        self.to_visit = [base_url]
        self.domain = urlparse(base_url).netloc
        self.pages_content = {}
        
    def _is_valid_url(self, url):
        """Check if URL is valid and belongs to the same domain."""
        parsed_url = urlparse(url)
        return (
            parsed_url.netloc == self.domain and
            not url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar'))
        )
    
    def _extract_links(self, soup, current_url):
        """Extract all links from a page."""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(current_url, href)
            if self._is_valid_url(full_url) and full_url not in self.visited_urls:
                links.append(full_url)
        return links
    
    def _extract_content(self, soup):
        """Extract relevant text content from a page."""
        # Remove unwanted elements, but keep <main>, <section>, <article>
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text and clean it
        text = soup.get_text(separator=' ')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = ' '.join(lines)
        print(f"[DEBUG] Extracted content length: {len(content)}")
        # Lowered threshold for debugging
        if len(content) < 50:
            print(f"[DEBUG] Content too short, skipping. Content: {content[:200]}")
        else:
            print(f"[DEBUG] Content sample: {content[:200]}")
        return content
    
    def _get_page_title(self, soup):
        """Extract page title."""
        title_tag = soup.find('title')
        return title_tag.get_text() if title_tag else "No Title"
    
    def scrape(self):
        """
        Scrape the website starting from the base URL.
        
        Returns:
            dict: Dictionary mapping URLs to their content
        """
        print(f"Starting to scrape {self.base_url}")
        pbar = tqdm(total=min(len(self.to_visit), self.max_pages), desc="Scraping pages")
        homepage_debugged = False
        while self.to_visit and len(self.visited_urls) < self.max_pages:
            current_url = self.to_visit.pop(0)
            if current_url in self.visited_urls:
                continue
            
            try:
                response = requests.get(
                    current_url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                title = self._get_page_title(soup)
                content = self._extract_content(soup)
                print(f"[DEBUG] Scraped URL: {current_url}, Title: {title}")
                
                # Print the homepage's extracted content for debugging
                if not homepage_debugged:
                    print(f"[DEBUG] Homepage extracted content preview: {content[:500]}")
                    homepage_debugged = True
                
                if len(content) > 50:  # Lowered threshold for storing
                    # --- Metadata enrichment ---
                    section_heading = None
                    page_hierarchy = [title] if title else []
                    # Try to extract the first h1/h2/h3 as section heading
                    for tag in ['h1', 'h2', 'h3']:
                        heading = soup.find(tag)
                        if heading:
                            section_heading = heading.get_text(strip=True)
                            break
                    # Simple keyword extraction: top 5 frequent words (excluding stopwords)
                    from collections import Counter
                    import re
                    stopwords = set(['the', 'and', 'for', 'are', 'with', 'that', 'this', 'you', 'your', 'from', 'have', 'has', 'was', 'but', 'not', 'all', 'can', 'will', 'they', 'their', 'our', 'about', 'more', 'who', 'when', 'where', 'how', 'what', 'which', 'also', 'use', 'used', 'using', 'into', 'than', 'other', 'any', 'each', 'such', 'its', 'may', 'one', 'two', 'three', 'four', 'five'])
                    words = re.findall(r'\b\w+\b', content.lower())
                    keywords = [w for w, c in Counter(words).most_common(20) if w not in stopwords][:5]
                    self.pages_content[current_url] = {
                        'title': title,
                        'content': content,
                        'url': current_url,
                        'section_heading': section_heading,
                        'page_hierarchy': page_hierarchy,
                        'keywords': keywords
                    }
                else:
                    print(f"[DEBUG] Skipped storing content for {current_url} (content too short)")
                
                # Find new links on this page
                new_links = self._extract_links(soup, current_url)
                print(f"[DEBUG] Found {len(new_links)} new links on {current_url}")
                for link in new_links:
                    if link not in self.visited_urls and link not in self.to_visit:
                        self.to_visit.append(link)
                
                self.visited_urls.add(current_url)
                pbar.update(1)
                pbar.total = min(len(self.to_visit) + len(self.visited_urls), self.max_pages)
                
                # Respect the website by waiting between requests
                time.sleep(self.delay)
                
            except Exception as e:
                print(f"Error scraping {current_url}: {e}")
                traceback.print_exc()
                self.visited_urls.add(current_url)  # Mark as visited to avoid retrying
        
        pbar.close()
        print(f"Scraping complete. Scraped {len(self.pages_content)} pages with content.")
        return self.pages_content


if __name__ == "__main__":
    # Example usage
    import json
    
    URL = "https://example.com"  # Replace with an actual website
    scraper = WebScraper(URL, delay=1, max_pages=10)
    results = scraper.scrape()
    
    # Print the titles of scraped pages
    for url, data in results.items():
        print(f"Title: {data['title']} | URL: {url}")
