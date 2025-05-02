# Auto-Learning Website Chatbot

A self-learning chatbot that can be deployed on any website. Once embedded, it automatically scrapes all the pages of the website, extracts relevant text data, and builds a vector-based knowledge base without any manual input.

## Features

- **Automatic Web Scraping**: Crawls through all pages of a website to gather content
- **Vector-Based Knowledge Base**: Stores and retrieves information efficiently
- **Local LLM Integration**: Uses Mistral 7B via Ollama for generating responses
- **Easy Deployment**: Simple embedding script for any website

## Tech Stack

- **LLM**: Mistral 7B via Ollama (open-source)
- **Web Scraper**: BeautifulSoup with requests
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB (local, fast)
- **Chat Interface**: HTML + JS / Flask API
- **Retrieval**: RAG (Retrieval-Augmented Generation) pipeline

## Setup and Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install Ollama and download Mistral 7B model:
   ```
   # Follow instructions at https://ollama.ai to install Ollama
   ollama pull mistral:7b
   ```
4. Start the application:
   ```
   python app.py
   ```
5. Access the demo at http://localhost:5000

## Embedding on Your Website

Add the following script tag to your website:

```html
<script src="https://your-server-address/embed/chatbot.js"></script>
```

## Usage

1. Start the application and navigate to the website you want to scrape
2. The chatbot will automatically crawl and index the website
3. Users can then ask questions through the chat interface
4. The system retrieves relevant context and generates responses using Mistral 7B

## Configuration

Configuration options can be set in a `.env` file:

```
WEBSITE_URL=https://example.com
SCRAPE_DELAY=2
MAX_PAGES=100
OLLAMA_HOST=http://localhost:11434
MODEL_NAME=mistral:7b
```
