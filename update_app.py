"""
Script to create a properly encoded .env file
"""

def create_env_file():
    env_content = """# Environment configuration for web scraping chatbot

# LLM Model Settings
OLLAMA_HOST=http://localhost:11434

# Model names - Primary models
QUICK_MODEL_NAME=llama3.2:3b-instruct-q3_K_M
SPEED_MODEL_NAME=llama3.2:1b-instruct-q2_K
DEEP_MODEL_NAME=phi3.5:3.8b-mini-instruct-q3_K_S
DEEPSEEK_MODEL_NAME=codegemma:2b-code

# Backup models
QUICK_BACKUP_MODEL=tinyllama:1.1b-chat
SPEED_BACKUP_MODEL=phi:latest
DEEP_BACKUP_MODEL=gemma:2b-instruct-q4_0
DEEPSEEK_BACKUP_MODEL=deepseek-coder:1.3b

# Scraper settings
SCRAPE_DELAY=1
MAX_PAGES=50

# Processing settings
CHUNK_SIZE=300
CHUNK_OVERLAP=30

# Storage settings
PERSIST_DIRECTORY=./data/chroma
COLLECTION_NAME=website_content
"""
    
    # Write the .env file with utf-8 encoding
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print(".env file created successfully with utf-8 encoding")

if __name__ == "__main__":
    create_env_file()
