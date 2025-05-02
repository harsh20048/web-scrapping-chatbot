"""
Web scraper module using Selenium for JavaScript-rendered websites.
"""

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import validators

class SeleniumWebScraper:
    def __init__(self, base_url, delay=2, max_pages=10):
        if not validators.url(base_url):
            raise ValueError(f"Invalid URL: {base_url}")
        self.base_url = base_url
        self.delay = delay
        self.max_pages = max_pages
        self.visited_urls = set()
        self.to_visit = [base_url]
        self.domain = urlparse(base_url).netloc
        self.pages_content = {}

        # Set up headless Chrome
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--window-size=1280,800')
        self.driver = webdriver.Chrome(options=chrome_options)

    def _is_valid_url(self, url):
        parsed_url = urlparse(url)
        return (
            parsed_url.netloc == self.domain and
            not url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar'))
        )

    def _extract_links(self, soup, current_url):
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(current_url, href)
            if self._is_valid_url(full_url) and full_url not in self.visited_urls:
                links.append(full_url)
        return links

    def _extract_content(self, soup):
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        text = soup.get_text(separator=' ')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = ' '.join(lines)
        return content

    def _get_page_title(self, soup):
        title_tag = soup.find('title')
        return title_tag.get_text() if title_tag else "No Title"

    def scrape(self):
        print(f"[Selenium] Starting to scrape {self.base_url}")
        homepage_debugged = False
        while self.to_visit and len(self.visited_urls) < self.max_pages:
            current_url = self.to_visit.pop(0)
            if current_url in self.visited_urls:
                continue
            try:
                self.driver.get(current_url)
                time.sleep(self.delay + 1)  # Wait for JS to load
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                title = self._get_page_title(soup)
                content = self._extract_content(soup)
                if not homepage_debugged:
                    print(f"[Selenium DEBUG] Homepage extracted content preview: {content[:500]}")
                    homepage_debugged = True
                if len(content) > 50:
                    self.pages_content[current_url] = {
                        'title': title,
                        'content': content,
                        'url': current_url
                    }
                new_links = self._extract_links(soup, current_url)
                for link in new_links:
                    if link not in self.visited_urls and link not in self.to_visit:
                        self.to_visit.append(link)
                self.visited_urls.add(current_url)
            except WebDriverException as e:
                print(f"[Selenium ERROR] Error scraping {current_url}: {e}")
                self.visited_urls.add(current_url)
        print(f"[Selenium] Scraping complete. Scraped {len(self.pages_content)} pages with content.")
        self.driver.quit()
        return self.pages_content

if __name__ == "__main__":
    URL = "https://example.com"  # Replace with your target site
    scraper = SeleniumWebScraper(URL, delay=2, max_pages=5)
    results = scraper.scrape()
    for url, data in results.items():
        print(f"Title: {data['title']} | URL: {url}")
