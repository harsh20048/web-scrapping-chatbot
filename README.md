<div align="center">

# Auto-Learning Website Chatbot

**Embed once. It learns your entire website automatically — no training, no manual input, no maintenance.**

<br/>

![Python](https://img.shields.io/badge/Python-Backend-185FA5?style=for-the-badge&logo=python&logoColor=white)
![Mistral 7B](https://img.shields.io/badge/Mistral%207B-Local%20LLM-7F77DD?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-1D9E75?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-EF9F27?style=for-the-badge)

<br/>

[![Live Demo](https://img.shields.io/badge/View%20Demo-%E2%86%92-1D9E75?style=flat-square)](https://your-demo-link.com)
[![Documentation](https://img.shields.io/badge/Read%20Docs-%E2%86%92-378ADD?style=flat-square)](#setup-and-installation)

</div>


---

## The Problem

Every website needs a chatbot. But building one means writing FAQs, uploading documents, curating knowledge bases, and updating everything manually whenever content changes.

**Auto-Learning Website Chatbot eliminates all of that.**

---

## Solution

Drop in a single `<script>` tag. The chatbot automatically crawls every page of the website, extracts relevant content, and builds a vector-based knowledge base — entirely on its own. Users can then ask questions in natural language and get accurate, context-aware answers powered by a locally-running LLM.

No data leaves your server. No API costs. No manual setup.

---

## Key Features

| Feature | Description |
|---|---|
| **Automatic Web Scraping** | Crawls all pages of a target website and extracts relevant text content automatically |
| **RAG Pipeline** | Retrieval-Augmented Generation ensures answers are grounded in real website content |
| **Local LLM** | Runs Mistral 7B via Ollama — fully on-premise, no external API calls, no usage costs |
| **Vector Knowledge Base** | ChromaDB stores and retrieves embeddings locally with fast semantic search |
| **One-Line Embed** | A single `<script>` tag deploys the chatbot on any website |
| **Privacy First** | All data stays on your own infrastructure — no third-party data exposure |

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│            EMBED (any website)                   │
│                                                  │
│  <script src="your-server/embed/chatbot.js">     │
│                                                  │
│  ↕ HTTP / Flask API                              │
│                                                  │
│            BACKEND (Flask + Python)              │
│                                                  │
│  Scraper (BeautifulSoup + requests)              │
│       ↓                                          │
│  Embeddings (sentence-transformers)              │
│       ↓                                          │
│  Vector Store (ChromaDB)                         │
│       ↓                                          │
│  RAG Retrieval → Mistral 7B (Ollama)             │
│       ↓                                          │
│  Response → Chat UI                              │
└──────────────────────────────────────────────────┘
```

### Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-D85A30?style=flat-square)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-7F77DD?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1D9E75?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square)
![Mistral 7B](https://img.shields.io/badge/Mistral%207B-Open%20Source-EF9F27?style=flat-square)

---

## Setup and Installation

### Prerequisites

![Python](https://img.shields.io/badge/Python-3.10%2B-185FA5?style=flat-square&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Required-000000?style=flat-square)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/auto-learning-chatbot
cd auto-learning-chatbot

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Ollama and pull Mistral 7B
# Follow instructions at https://ollama.ai to install Ollama
ollama pull mistral:7b

# 4. Start the application
python app.py
```

Access the demo at **http://localhost:5000**

---

## Embed on Your Website

Add a single line to any webpage:

```html
<script src="https://your-server-address/embed/chatbot.js"></script>
```

That's it. The chatbot widget will appear on your site and begin learning automatically.

---

## Configuration

Create a `.env` file in the project root:

```bash
WEBSITE_URL=https://example.com    # The website to scrape and learn from
SCRAPE_DELAY=2                     # Delay between page requests (seconds)
MAX_PAGES=100                      # Maximum number of pages to crawl
OLLAMA_HOST=http://localhost:11434 # Ollama server address
MODEL_NAME=mistral:7b              # LLM model to use
```

---

## How It Works

```
1. Embed the script tag on your website
      ↓
2. Scraper crawls all pages (BeautifulSoup + requests)
      ↓
3. Text is chunked and embedded (all-MiniLM-L6-v2)
      ↓
4. Embeddings stored in ChromaDB vector store
      ↓
5. User asks a question via chat widget
      ↓
6. RAG retrieves the most relevant content chunks
      ↓
7. Mistral 7B generates a grounded, accurate response
      ↓
8. Answer displayed in the chat interface
```

---

## Why Now

![Privacy](https://img.shields.io/badge/Trend-On--Premise%20AI%20Demand%20Rising-1D9E75?style=flat-square)
![Cost](https://img.shields.io/badge/Advantage-Zero%20API%20Costs-7F77DD?style=flat-square)
![Market](https://img.shields.io/badge/TAM-$6B%2B%20Chatbot%20Market-D85A30?style=flat-square)

Enterprises are increasingly wary of sending data to third-party AI APIs. Local LLM deployment is becoming the standard for privacy-conscious companies. Auto-Learning Website Chatbot is built for exactly this moment — powerful AI, fully on-premise, with zero per-query costs.

---

## Roadmap

| Phase | Focus | Timeline | Status |
|---|---|---|---|
| ![p1](https://img.shields.io/badge/Phase%201-1D9E75?style=flat-square) | Core scraping, RAG pipeline, chat UI | Shipped | ✅ Done |
| ![p2](https://img.shields.io/badge/Phase%202-378ADD?style=flat-square) | Auto re-crawl on content change detection | Q3 2025 | 🔄 In Progress |
| ![p3](https://img.shields.io/badge/Phase%203-7F77DD?style=flat-square) | Multi-website support & admin dashboard | Q4 2025 | 📅 Planned |
| ![p4](https://img.shields.io/badge/Phase%204-EF9F27?style=flat-square) | Analytics, handoff to human agents, CRM sync | Q1 2026 | 📅 Planned |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push and open a Pull Request

---

## License

[MIT](./LICENSE)

---

<div align="center">

Built with ❤️ by the Auto-Learning Chatbot team

</div>
