"""
Main Flask application for the auto-learning website chatbot.
Provides API endpoints and serves the demo interface.
"""

import os
import json
import time
import threading
import requests
import re
import logging
import traceback

from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
from src.rag import RAG

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize chat history
chat_history = []
CHAT_HISTORY_FILE = 'chat_history.json'

try:
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            chat_history = json.load(f)
except Exception as e:
    logger.error(f"Error loading chat history: {e}")
    chat_history = []

def save_chat_history():
    """Save chat history to file."""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")

def store_chat_message(role, message):
    """Add a message to chat history and save."""
    global chat_history
    chat_history.append({'role': role, 'message': message, 'timestamp': time.time()})
    if len(chat_history) > 200:  # Limit history to 200 messages
        chat_history = chat_history[-200:]
    save_chat_history()

# Initialize Flask app
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
            return json.load(f)
    except Exception:
        return []

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
        logger.error("No URL provided for scraping")
        return jsonify({'error': 'URL is required'}), 400
    
    if scraping_status['is_scraping']:
        logger.warning("Already scraping a website")
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
            error_details = traceback.format_exc()
            logger.error(f"Error in scraping: {str(e)}\n{error_details}")
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
    """
    Handle different query modes (flash, deep, deepseek) and return the result.
    """
    if not request.json or 'query' not in request.json:
        logger.error("No query provided in request")
        return jsonify({'error': 'No query provided'}), 400
    
    user_query = request.json['query']
    mode = request.json.get('mode', 'flash')  # Default to flash mode
    
    logger.info(f"Processing query in {mode} mode: {user_query[:50]}...")
    
    try:
        if not rag.has_documents():
            logger.warning("No documents in database for query")
            return jsonify({'error': 'No documents in database. Please scrape a website first.'}), 400
        
        if mode == 'deepseek':
            # Use deepseek code model
            temperature = request.json.get('temperature', 0.1)
            max_tokens = request.json.get('max_tokens', 1024)
            model_name = config['deepseek_model_name']
            
            logger.debug(f"Using deepseek mode with model: {model_name}")
            
            result = rag.query(
                user_query=user_query,
                temperature=temperature,
                max_tokens=max_tokens,
                mode="deepseek"
            )
            return jsonify({'answer': result['answer'], 'contexts': result.get('contexts', [])})
        
        elif mode == 'deep':
            # Use phi model for deeper understanding
            model_name = config['deep_model_name']
            logger.debug(f"Using deep mode with model: {model_name}")
            
            result = rag.query(
                user_query=user_query,
                temperature=0.2,
                max_tokens=1024,
                mode="deep"
            )
            return jsonify({'answer': result['answer'], 'contexts': result.get('contexts', [])})
            
        elif mode == 'flash':
            # Use the fastest model for quick responses
            model_name = config['speed_model_name']  # Using the speed model for flash mode
            logger.debug(f"Using flash mode with model: {model_name}")
            
            result = rag.query(
                user_query=user_query,
                temperature=0.1,
                max_tokens=512,
                mode="speed"  # Use speed model for flash mode
            )
            # Clean the response to be more direct in flash mode
            result['answer'] = clean_llm_response(result['answer'])
            return jsonify({'answer': result['answer'], 'contexts': result.get('contexts', [])})
            
        else:
            # Default RAG query with standard model
            logger.debug(f"Using default mode: {mode}")
            # If unknown mode, default to flash mode
            result = rag.query(
                user_query=user_query,
                temperature=0.1,
                max_tokens=512,
                mode="speed"  # Use speed model for default
            )
            result['answer'] = clean_llm_response(result['answer'])
            return jsonify({'answer': result['answer'], 'contexts': result.get('contexts', [])})
            
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in query: {str(e)}\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500

@app.route('/api/ollama_deepseek', methods=['POST'])
def ollama_deepseek():
    """Deep analysis mode using direct Ollama completion"""
    if not request.json or 'prompt' not in request.json:
        logger.error("No prompt provided for deep analysis")
        return jsonify({'error': 'No prompt provided'}), 400
    
    prompt = request.json['prompt']
    logger.info(f"Processing deep analysis: {prompt[:50]}...")
    
    try:
        # Get documents to analyze using vectordb query instead of retriever
        query_embedding = rag.embedding_model.embed_query(prompt)
        retrieved_docs = rag.vectordb.query(query_embedding, n_results=5)
        
        if not retrieved_docs:
            logger.warning("No relevant documents found for deep analysis")
            return jsonify({'error': 'No relevant documents found. Please try another query or scrape more content.'}), 400
        
        # Prepare context for the model
        context_text = "\n\n".join([doc['content'] for doc in retrieved_docs])
        
        # Create a detailed analysis prompt
        system_prompt = """You are an expert analyst that performs deep analysis of text. 
        Given the following content from a website, provide a detailed analysis in response to the user's query.
        Focus on extracting the most relevant information and providing a comprehensive answer.
        If you don't know or the information is not in the provided content, say so honestly.
        Format your response in markdown for better readability."""
        
        full_prompt = f"{system_prompt}\n\nCONTENT TO ANALYZE:\n{context_text}\n\nQUERY: {prompt}\n\nANALYSIS:"
        
        # Call Ollama API directly for more control
        response = requests.post(
            f"{config['ollama_host']}/api/generate",
            json={
                "model": config['deep_model_name'],
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 1024
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            # Extract contexts for reference
            contexts = []
            for doc in retrieved_docs[:3]:
                contexts.append({
                    "text": doc['content'][:200] + "...",
                    "source": doc['metadata'].get('source', '') if isinstance(doc['metadata'], dict) else '',
                    "title": doc['metadata'].get('title', '') if isinstance(doc['metadata'], dict) else ''
                })
            
            # Store in chat history
            store_chat_message('user', prompt)
            store_chat_message('bot', result['response'])
            
            return jsonify({
                'result': result['response'],
                'contexts': contexts
            })
        else:
            error_details = traceback.format_exc()
            logger.error(f"Error in deep analysis: {response.status_code}\n{error_details}")
            return jsonify({'error': f"Ollama API error: {response.status_code}", 'details': error_details}), response.status_code
            
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in deep analysis: {str(e)}\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500

@app.route('/api/reset', methods=['POST'])
def reset_database():
    """API endpoint to reset the vector database."""
    try:
        result = rag.reset_database()
        return jsonify(result)
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in resetting database: {str(e)}\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """API endpoint to get system status."""
    try:
        db_stats = rag.vectordb.get_collection_stats()
        
        return jsonify({
            'status': 'ok',
            'database': db_stats,
            'ollama_host': config['ollama_host'],
            'model_name': config['model_name']
        })
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in getting status: {str(e)}\n{error_details}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'details': error_details
        }), 500

@app.route('/api/chat_history', methods=['GET', 'POST'])
def chat_history_endpoint():
    """API endpoint to get or update chat history."""
    global chat_history
    
    if request.method == 'GET':
        return jsonify({'history': chat_history})
    
    elif request.method == 'POST':
        try:
            data = request.json
            if not data:
                logger.error("No data provided for chat history")
                return jsonify({'error': 'No data provided'}), 400
            
            user_message = data.get('user_message', '')
            bot_response = data.get('bot_response', '')
            
            if not user_message or not bot_response:
                logger.error("Missing message or response in chat history update")
                return jsonify({'error': 'Missing message or response'}), 400
            
            # Add to chat history
            store_chat_message('user', user_message)
            store_chat_message('bot', bot_response)
            
            return jsonify({'status': 'success'})
        
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error updating chat history: {str(e)}\n{error_details}")
            return jsonify({'error': str(e), 'details': error_details}), 500

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
        error_details = traceback.format_exc()
        logger.error(f"Error in viewing chunks: {str(e)}\n{error_details}")
        return f'<p>Error fetching chunks: {e}</p>', 500

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('./data/chroma', exist_ok=True)
    
    # Create embed directory if it doesn't exist
    os.makedirs('./embed', exist_ok=True)
    
    # For development, run with debug=True
    app.run(debug=True, host='0.0.0.0', port=5000)
