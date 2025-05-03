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
import argparse
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
from src.rag import RAG
from src.model_trainer import ModelTrainer


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
    'model_name': os.getenv('MODEL_NAME', 'phi3.5:3.8b-mini-instruct-q3_K_S'),
    'ollama_host': os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
    'quick_model_name': os.getenv('QUICK_MODEL_NAME', 'llama3.2:3b-instruct-q3_K_M'),
    'speed_model_name': os.getenv('SPEED_MODEL_NAME', 'phi3:3.8b-mini-4k-instruct-q4_K_M'),  # Upgraded to phi3 for better accuracy
    'deep_model_name': os.getenv('DEEP_MODEL_NAME', 'phi3.5:3.8b-mini-instruct-q3_K_S'),
    'deepseek_model_name': os.getenv('DEEPSEEK_MODEL_NAME', 'deepseek-r1:1.5b'),
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

# Initialize ModelTrainer
model_trainer = ModelTrainer(rag)

# Store scraping status
scraping_status = {
    'is_scraping': False,
    'current_url': None,
    'documents_processed': 0,
    'error': None,
    'elapsed_time': None
}

# Store training status
training_status = {
    'is_training': False,
    'is_evaluating': False,
    'generated_examples': 0,
    'accuracy': None,
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

@app.route('/api/scraped_data')
def get_scraped_data():
    """API endpoint to get all scraped data for display in the UI."""
    try:
        # Get view type from query parameters (summary or detailed)
        view_type = request.args.get('view_type', 'summary')
        
        # Get all stored documents (chunks)
        results = rag.vectordb.collection.get()
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        
        # Format the response
        formatted_docs = []
        for i, (content, metadata) in enumerate(zip(documents, metadatas)):
            doc = {
                'content': content,
                'metadata': metadata
            }
            formatted_docs.append(doc)
        
        return jsonify({
            'status': 'success',
            'document_count': len(formatted_docs),
            'documents': formatted_docs
        })
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in getting scraped data: {str(e)}\n{error_details}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'details': error_details
        }), 500

@app.route('/api/train_flash_model', methods=['POST'])
def train_flash_model():
    """API endpoint to train the Flash model with synthetic QA dataset."""
    global training_status
    
    try:
        # Check if already training
        if training_status['is_training']:
            return jsonify({
                'status': 'in_progress',
                'message': 'Training is already in progress',
                'training_status': training_status
            })
        
        # Get training parameters from request
        data = request.json or {}
        num_examples = data.get('num_examples', 50)  # Default to 50 examples
        
        # Start training in a separate thread
        def training_task():
            global training_status
            start_time = time.time()
            training_status['is_training'] = True
            training_status['error'] = None
            
            try:
                # Generate synthetic dataset
                training_data = model_trainer.create_synthetic_dataset(num_examples=num_examples)
                training_status['generated_examples'] = len(training_data)
                training_status['elapsed_time'] = time.time() - start_time
                training_status['is_training'] = False
                logger.info(f"Training completed: {len(training_data)} examples generated")
            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Error in training: {str(e)}\n{error_details}")
                training_status['error'] = str(e)
                training_status['is_training'] = False
        
        # Start training thread
        threading.Thread(target=training_task).start()
        
        return jsonify({
            'status': 'started',
            'message': f'Started training with {num_examples} synthetic examples',
            'training_status': training_status
        })
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error starting training: {str(e)}\n{error_details}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'details': error_details
        }), 500

@app.route('/api/evaluate_flash_model', methods=['POST'])
def evaluate_flash_model():
    """API endpoint to evaluate the Flash model with the synthetic QA dataset."""
    global training_status
    
    try:
        # Check if already evaluating
        if training_status['is_evaluating']:
            return jsonify({
                'status': 'in_progress',
                'message': 'Evaluation is already in progress',
                'training_status': training_status
            })
        
        # Get evaluation parameters from request
        data = request.json or {}
        num_examples = data.get('num_examples', 20)  # Default to 20 examples for evaluation
        
        # Start evaluation in a separate thread
        def evaluation_task():
            global training_status
            start_time = time.time()
            training_status['is_evaluating'] = True
            training_status['error'] = None
            
            try:
                # Evaluate the model
                results = model_trainer.evaluate_flash_mode(num_examples=num_examples)
                training_status['accuracy'] = results['metrics']['accuracy']
                training_status['elapsed_time'] = time.time() - start_time
                training_status['is_evaluating'] = False
                logger.info(f"Evaluation completed: Accuracy = {results['metrics']['accuracy']}")
            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Error in evaluation: {str(e)}\n{error_details}")
                training_status['error'] = str(e)
                training_status['is_evaluating'] = False
        
        # Start evaluation thread
        threading.Thread(target=evaluation_task).start()
        
        return jsonify({
            'status': 'started',
            'message': f'Started evaluation with {num_examples} examples',
            'training_status': training_status
        })
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error starting evaluation: {str(e)}\n{error_details}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'details': error_details
        }), 500

@app.route('/api/train_status')
def get_train_status():
    """API endpoint to get the current training/evaluation status."""
    global training_status
    
    return jsonify({
        'status': 'success',
        'training_status': training_status
    })

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('./data/chroma', exist_ok=True)
    
    # Create embed directory if it doesn't exist
    os.makedirs('./embed', exist_ok=True)
    
    # Create data directory for training if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # For development, run with debug=True
    app.run(debug=True, host='0.0.0.0', port=5000)
