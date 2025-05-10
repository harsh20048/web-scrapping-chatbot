"""
Main Flask application for the auto-learning website chatbot.
Provides API endpoints and serves the demo interface.
"""

import os
import json as _json
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
from src.rag import RAG
from src.ai_extractor import extract_with_diffbot, extract_with_openai
import threading
import requests
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
config = {
    'scrape_delay': int(os.getenv('SCRAPE_DELAY', '1')),
    'max_pages': int(os.getenv('MAX_PAGES', '50')),
    'ollama_host': os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
    'model_name': os.getenv('DEEP_MODEL_NAME', 'phi3.5:3.8b-mini-instruct-q3_K_S'),
    'quick_model_name': os.getenv('QUICK_MODEL_NAME', 'llama3.2:3b-instruct-q3_K_M'),
    'speed_model_name': os.getenv('SPEED_MODEL_NAME', 'llama3.2:1b-instruct-q2_K'),
    'deep_model_name': os.getenv('DEEP_MODEL_NAME', 'phi3.5:3.8b-mini-instruct-q3_K_S'),
    'deepseek_model_name': os.getenv('DEEPSEEK_MODEL_NAME', 'codegemma:2b-code'),
    'chunk_size': int(os.getenv('CHUNK_SIZE', '300')),
    'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '30')),
    'persist_directory': os.getenv('PERSIST_DIRECTORY', './data/chroma'),
    'collection_name': os.getenv('COLLECTION_NAME', 'website_content')
}

CHAT_HISTORY_FILE = 'chat_history.json'

def clean_llm_response(response_text):
    """Clean LLM response to remove meta-commentary and reasoning."""
    
    # Patterns to identify final answers after reasoning
    patterns = [
        r'(?:.*\n)+(.+)$',  # Get the last line that often contains the final answer
        r'.*My (?:final )?(?:answer|response) (?:is|would be):?\s*(.+)',  # Look for "My answer is..." pattern
        r'.*(?:In summary|To summarize|In conclusion|Therefore)[,:]?\s*(.+)',  # Summary statements
        r'.*(?:putting|wrap|let me|I\'ll|I will).*(?:together|it up|summarize|conclude).*:\s*(.+)'  # Wrapping up pattern
    ]
    
    # Check if response appears to have meta-commentary (multiple lines with more than 3 sentences)
    lines = response_text.strip().split('\n')
    if len(lines) > 3 or len(re.findall(r'[.!?]\s+', response_text)) > 3:
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match and match.group(1).strip():
                return match.group(1).strip()
    
    # Remove any "thinking out loud" prefixes
    response_text = re.sub(r'^(?:Let me|First|I\'ll|I will|Deep:).*?(?=[A-Z])', '', response_text.strip(), flags=re.IGNORECASE)
    
    # Return the first non-meta sentence if no clear final answer
    sentences = re.split(r'(?<=[.!?])\s+', response_text.strip())
    for sentence in sentences:
        # Skip sentences that look like meta-commentary
        if not re.search(r'(?:I\'ll|I will|need to|let me|first|should|looking at|reviewing)', sentence, re.IGNORECASE):
            return sentence
    
    # If all else fails, return the original but limit to first 100 words
    words = response_text.split()
    if len(words) > 100:
        return ' '.join(words[:100]) + "..."
    return response_text

def load_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return _json.load(f)
    except Exception:
        return []

def save_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            _json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save chat history: {e}")

# Initialize RAG system
rag = RAG()

# Store scraping status
scraping_status = {
    'is_scraping': False,
    'current_url': None,
    'documents_processed': 0,
    'error': None,
    'elapsed_time': None
}

chat_history = load_chat_history()

def store_chat_message(role, message):
    chat_history.append({'role': role, 'message': message})
    if len(chat_history) > 200:
        chat_history.pop(0)
    save_chat_history()

@app.route('/')
def index():
    """Render the demo page."""
    return render_template('index.html')

@app.route('/embed/chatbot.js')
def chatbot_js():
    """Serve the embeddable chatbot script."""
    return send_from_directory('embed', 'chatbot.js')

@app.route('/api/scrape', methods=['POST'])
def scrape_website():
    """API endpoint to trigger website scraping."""
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    if scraping_status['is_scraping']:
        return jsonify({
            'error': 'Already scraping a website',
            'current_url': scraping_status['current_url']
        }), 409
    
    # Reset status
    scraping_status['is_scraping'] = True
    scraping_status['current_url'] = url
    scraping_status['documents_processed'] = 0
    scraping_status['error'] = None
    scraping_status['elapsed_time'] = None
    
    # Start scraping in a background thread
    def scrape_task():
        import time
        start_time = time.time()
        try:
            num_docs = rag.scrape_website(url)
            scraping_status['documents_processed'] = num_docs
        except Exception as e:
            scraping_status['error'] = str(e)
        finally:
            end_time = time.time()
            scraping_status['elapsed_time'] = round(end_time - start_time, 2)
            scraping_status['is_scraping'] = False
    
    thread = threading.Thread(target=scrape_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'Scraping started', 'url': url})

@app.route('/api/scrape/status', methods=['GET'])
def get_scrape_status():
    """API endpoint to check scraping status."""
    return jsonify(scraping_status)

@app.route('/api/query', methods=['POST'])
def query():
    """API endpoint to query the chatbot."""
    data = request.json
    user_query = data.get('query')
    mode = data.get('mode', 'speed')  # Default to speed mode for faster responses
    if not user_query:
        return jsonify({'error': 'Query is required'}), 400
    # Get optional parameters with defaults
    n_results = data.get('n_results', 1)  # Default to 1 result for speed
    temperature = data.get('temperature', 0.3)  # Lower temperature for more focused answers
    max_tokens = data.get('max_tokens', 128)  # Limit token count for faster responses
    
    store_chat_message('user', user_query)
    result = rag.query(user_query, n_results=n_results, temperature=temperature, max_tokens=max_tokens, mode=mode)
    
    # Clean the response to remove any meta-commentary 
    if 'answer' in result:
        result['answer'] = clean_llm_response(result['answer'])
    
    # Store cleaned response in chat history
    store_chat_message('bot', result.get('answer', ''))
    
    return jsonify(result)

@app.route('/api/reset', methods=['POST'])
def reset_database():
    """API endpoint to reset the vector database."""
    result = rag.reset_database()
    return jsonify(result)

@app.route('/api/status', methods=['GET'])
def get_status():
    """API endpoint to get system status."""
    try:
        # Get database stats
        db_stats = rag.vectordb.get_collection_stats()
        
        # Check Ollama connectivity
        ollama_status = 'disconnected'
        ollama_models = []
        try:
            # Only wait 2 seconds max to check if Ollama is responsive
            resp = requests.get(f"{config['ollama_host']}/api/tags", timeout=2)
            if resp.status_code == 200:
                ollama_status = 'connected'
                # List available models
                models_data = resp.json()
                if 'models' in models_data:
                    ollama_models = [m.get('name') for m in models_data.get('models', [])]                
        except Exception:
            # Don't report the specific error, just note that it's disconnected
            pass
            
        return jsonify({
            'status': 'ok',
            'database': db_stats,
            'ollama_host': config['ollama_host'],
            'model_name': config['model_name'],
            'ollama_status': ollama_status,
            'available_models': ollama_models
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/ai_extract', methods=['POST'])
def ai_extract():
    """API endpoint for AI-powered structured/semantic extraction using Diffbot or OpenAI."""
    data = request.json
    url = data.get('url')
    provider = data.get('provider', 'diffbot')
    token = os.getenv('DIFFBOT_TOKEN') or data.get('token')
    openai_key = os.getenv('OPENAI_API_KEY') or data.get('openai_key')
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    if provider == 'openai':
        # Use OpenAI to extract structured data from scraped content
        # For demo, fetch content with requests and BeautifulSoup
        import requests
        from bs4 import BeautifulSoup
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(resp.content, 'html.parser')
            text = soup.get_text(separator=' ')
        except Exception as e:
            return jsonify({'error': f'Failed to fetch or parse page: {e}'}), 500
        if not openai_key:
            return jsonify({'error': 'OpenAI API key is required'}), 400
        store_chat_message('user', f'[AI Extract][OpenAI] {url}')
        result = extract_with_openai(text, openai_key)
        store_chat_message('bot', result or '[OpenAI extraction failed]')
        if result is None:
            return jsonify({'error': 'OpenAI extraction failed'}), 500
        return jsonify({'provider': 'openai', 'result': result})
    # Default: Diffbot
    if not token:
        return jsonify({'error': 'Diffbot API token is required'}), 400
    store_chat_message('user', f'[AI Extract][Diffbot] {url}')
    result = extract_with_diffbot(url, token)
    store_chat_message('bot', result or '[Diffbot extraction failed]')
    if result is None:
        return jsonify({'error': 'Diffbot extraction failed'}), 500
    return jsonify({'provider': 'diffbot', 'result': result})

@app.route('/api/ollama_deepseek', methods=['POST'])
def ollama_deepseek():
    data = request.get_json()
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400
    try:
        # Retrieve relevant context from scraped content (RAG)
        rag_result = rag.query(prompt, n_results=3, temperature=0.3, max_tokens=256, mode="deepseek")
        context_docs = rag_result.get('contexts', [])
        context_str = ''
        for i, doc in enumerate(context_docs):
            title = doc.get('title', '')
            source = doc.get('source', '')
            text = doc.get('text', '')
            context_str += f"[Doc {i+1}] Title: {title}\nSource: {source}\n{text}\n\n"
        
        # Updated prompt with stronger instructions against meta-commentary
        full_prompt = "Context information from website:\n" + \
            f"{context_str}\n\n" + \
            f"User question: {prompt}\n\n" + \
            "Please answer based ONLY on the context information provided above.\n" + \
            "NEVER respond with phrases like \"As an AI language model\" or other meta-commentary.\n" + \
            "If the answer is not present in the context, say you don't know.\n" + \
            "DO NOT explain your reasoning or thinking - provide ONLY the direct answer to the question.\n" + \
            "Be brief and concise."
        
        # Call local Ollama DeepSeek model with improved error handling
        ollama_url = config['ollama_host'] + '/api/generate'
        payload = {
            'model': config['deepseek_model_name'],
            'prompt': full_prompt,
            'stream': False
        }
        
        try:
            # Use shorter timeout to fail faster
            resp = requests.post(ollama_url, json=payload, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                
                # Clean the response to remove meta-commentary
                response_text = result.get('response', '').strip()
                cleaned_response = clean_llm_response(response_text)
                
                return jsonify({'result': cleaned_response})
            else:
                error_msg = resp.text if resp.text else 'Unknown error'
                
                # More user-friendly error messages
                if resp.status_code == 404:
                    return jsonify({'error': f'Model "{config["deepseek_model_name"]}" not found. Please run: ollama pull {config["deepseek_model_name"]}'})  
                elif resp.status_code == 500:
                    return jsonify({'error': 'Ollama server error. This might be due to limited resources or the model being too large.'})
                else:
                    return jsonify({'error': f'Ollama error ({resp.status_code}): {error_msg}'})
                
        except requests.exceptions.ConnectionError:
            return jsonify({'error': 'Cannot connect to Ollama. Please make sure Ollama is running on your machine.'})
        except requests.exceptions.Timeout:
            return jsonify({'error': 'Connection to Ollama timed out. Please check if Ollama is running properly.'})
    except Exception as e:
        # Generic fallback error handler with more helpful message
        error_message = str(e)
        if 'Connection refused' in error_message:
            return jsonify({'error': 'Cannot connect to Ollama server. Please ensure Ollama is running on your machine.'})
        elif 'model not found' in error_message.lower():
            return jsonify({'error': f'The requested model "{config["deepseek_model_name"]}" is not installed. Try running: ollama pull {config["deepseek_model_name"]}'})    
        else:
            return jsonify({'error': f'Error processing your request: {error_message}. Please try again later.'})

@app.route('/api/chat_history', methods=['GET'])
def get_chat_history():
    """API endpoint to get chat history."""
    return jsonify({'history': chat_history})

@app.route('/api/download_chat_history', methods=['GET'])
def download_chat_history():
    """API endpoint to download chat history as a JSON file."""
    response = jsonify({'history': chat_history})
    response.headers.set('Content-Disposition', 'attachment; filename=chat_history.json')
    response.headers.set('Content-Type', 'application/json')
    return response

@app.route('/view_chunks')
def view_chunks():
    """View all stored text chunks from the vector database."""
    try:
        # Get all stored documents (chunks)
        # We use rag.vectordb.collection to access the ChromaDB collection directly
        # ChromaDB's API: collection.get() returns all documents if no filters are given
        results = rag.vectordb.collection.get()
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        # Prepare a simple HTML response
        html = '<h2>Stored Web-Scraped Chunks</h2>'
        html += f'<p>Total chunks: {len(documents)}</p>'
        html += '<ul>'
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            title = meta.get('title', 'No Title') if isinstance(meta, dict) else meta
            source = meta.get('source', 'Unknown') if isinstance(meta, dict) else meta
            html += f'<li><b>Chunk {i+1}</b><br><b>Title:</b> {title}<br><b>Source:</b> {source}<br><b>Content:</b> {doc[:500]}...<br></li>'
        html += '</ul>'
        return html
    except Exception as e:
        return f'<p>Error fetching chunks: {e}</p>', 500

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('./data/chroma', exist_ok=True)
    
    # Create embed directory if it doesn't exist
    os.makedirs('./embed', exist_ok=True)
    
    # For development, run with debug=True
    app.run(debug=True, host='0.0.0.0', port=5000)
