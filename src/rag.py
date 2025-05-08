"""
RAG (Retrieval-Augmented Generation) module.
Combines web scraping, embeddings, vector database, and LLM for question answering.
"""

from typing import Dict, Any, List
import os
import re
from .scraper import WebScraper
from .processor import TextProcessor
from .embeddings import EmbeddingModel
from .vectordb import VectorDB
from .ollama_client import OllamaClient
from .gemini_client import GeminiClient


class RAG:
    def __init__(self, base_url=None, config=None):
        """
        Initialize the RAG system.
        
        Args:
            base_url (str, optional): Base URL for web scraping
            config (Dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.base_url = base_url
        
        # Default configuration
        self.scraper_config = {
            'delay': self.config.get('scrape_delay', 1),  # Reduced delay
            'max_pages': self.config.get('max_pages', 50)  # Reduced max pages
        }
        
        self.processor_config = {
            'chunk_size': self.config.get('chunk_size', 300),  # Smaller chunks
            'chunk_overlap': self.config.get('chunk_overlap', 30)  # Reduced overlap
        }
        
        self.embedding_model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        
        self.vectordb_config = {
            'persist_directory': self.config.get('persist_directory', './data/chroma'),
            'collection_name': self.config.get('collection_name', 'website_content')
        }
        
        # Model names from Ollama list
        self.ollama_config = {
            'host': self.config.get('ollama_host', 'http://localhost:11434'),
            'model': self.config.get('model_name', 'phi:latest')
        }
        
        # --- Primary LLMs for modes ---
        self.quick_model_name = self.config.get('quick_model_name', os.getenv('QUICK_MODEL_NAME', 'llama3.2:3b-instruct-q3_K_M'))
        self.speed_model_name = self.config.get('speed_model_name', os.getenv('SPEED_MODEL_NAME', 'llama3.2:1b-instruct-q2_K'))
        self.deep_model_name = self.config.get('deep_model_name', os.getenv('DEEP_MODEL_NAME', 'phi3.5:3.8b-mini-instruct-q3_K_S'))
        self.deepseek_model_name = self.config.get('deepseek_model_name', os.getenv('DEEPSEEK_MODEL_NAME', 'codegemma:2b-code'))
        
        # --- Backup LLMs for each mode ---
        self.quick_backup_model = self.config.get('quick_backup_model', os.getenv('QUICK_BACKUP_MODEL', 'tinyllama:1.1b-chat'))
        self.speed_backup_model = self.config.get('speed_backup_model', os.getenv('SPEED_BACKUP_MODEL', 'phi:latest'))
        self.deep_backup_model = self.config.get('deep_backup_model', os.getenv('DEEP_BACKUP_MODEL', 'gemma:2b-instruct-q4_0'))
        self.deepseek_backup_model = self.config.get('deepseek_backup_model', os.getenv('DEEPSEEK_BACKUP_MODEL', 'deepseek-coder:1.3b'))
        
        # Initialize components as needed
        self.processor = TextProcessor(
            chunk_size=self.processor_config['chunk_size'],
            chunk_overlap=self.processor_config['chunk_overlap']
        )
        
        self.embedding_model = EmbeddingModel(model_name=self.embedding_model_name)
        
        self.vectordb = VectorDB(
            persist_directory=self.vectordb_config['persist_directory'],
            collection_name=self.vectordb_config['collection_name']
        )
        
        # Initialize primary models
        self.llm = OllamaClient(
            host=self.ollama_config['host'],
            model=self.deep_model_name
        )
        
        self.quick_llm = OllamaClient(
            host=self.ollama_config['host'],
            model=self.quick_model_name
        )
        
        self.speed_llm = OllamaClient(
            host=self.ollama_config['host'],
            model=self.speed_model_name
        )
        
        self.deepseek_llm = OllamaClient(
            host=self.ollama_config['host'],
            model=self.deepseek_model_name
        )
        
        # Initialize backup models
        self.quick_backup_llm = OllamaClient(
            host=self.ollama_config['host'],
            model=self.quick_backup_model
        )
        
        self.speed_backup_llm = OllamaClient(
            host=self.ollama_config['host'],
            model=self.speed_backup_model
        )
        
        self.deep_backup_llm = OllamaClient(
            host=self.ollama_config['host'],
            model=self.deep_backup_model
        )
        
        self.deepseek_backup_llm = OllamaClient(
            host=self.ollama_config['host'],
            model=self.deepseek_backup_model
        )
        
        # Initialize Gemini client for quick mode
        self.gemini_api_key = self.config.get('gemini_api_key', os.getenv('GEMINI_API_KEY'))
        self.gemini_client = None
        if self.gemini_api_key:
            self.gemini_client = GeminiClient(api_key=self.gemini_api_key)
        
        # Patterns for greeting detection
        self.greeting_patterns = [
            r'^hi$', r'^hello$', r'^hey$', r'^greetings$', r'^howdy$', 
            r'^hi\s+there$', r'^hello\s+there$', r'^hey\s+there$',
            r'^good\s+morning$', r'^good\s+afternoon$', r'^good\s+evening$'
        ]
        
        # Patterns for website summary/briefing requests
        self.summary_patterns = [
            r'(tell|give)\s+me\s+about\s+the\s+website',
            r'(brief|briefing|summarize|summary|overview)\s+(me\s+)?(about\s+)?(of\s+)?the\s+website',
            r'what\'?s?\s+(?:this|the)\s+website\s+about',
            r'describe\s+(?:this|the)\s+website'
        ]
    
    def is_greeting(self, query):
        """Check if the query is a greeting."""
        query_lower = query.lower().strip()
        for pattern in self.greeting_patterns:
            if re.match(pattern, query_lower):
                return True
        return False
    
    def is_website_summary_request(self, query):
        """Check if the query is requesting a website summary."""
        query_lower = query.lower().strip()
        for pattern in self.summary_patterns:
            if re.search(pattern, query_lower):
                return True
        return False
    
    def handle_greeting(self, mode="quick"):
        """Generate a greeting response."""
        greeting_response = "Hello! I'm your web scraping chatbot assistant. I can answer questions about the website content you've scraped. How can I help you today?"
        
        # Add mode-specific information
        if mode == "quick":
            greeting_response += " I'm currently in Quick mode, which provides fast responses."
        elif mode == "speed":
            greeting_response += " I'm currently in Speed mode, balancing speed and quality."
        elif mode == "deep":
            greeting_response += " I'm currently in Deep mode, providing more thorough analysis."
        elif mode == "deepseek":
            greeting_response += " I'm currently in DeepSeek mode, specialized for code-related questions."
        
        return {
            "answer": greeting_response,
            "contexts": []
        }
    
    def get_website_summary(self, mode="deep"):
        """Generate a summary of the scraped website content."""
        import time
        
        print("[DEBUG] Generating website summary")
        
        # Check if collection has documents
        stats = self.vectordb.get_collection_stats()
        if stats.get('document_count', 0) == 0:
            return {
                "answer": "I don't have any website content in my database yet. Please scrape a website first using the 'Start Scraping' button.",
                "contexts": []
            }
        
        # Get all documents to analyze website content
        # We'll limit to 20 to prevent overloading
        results = self.vectordb.collection.get(limit=20)
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        
        if not documents:
            return {
                "answer": "I have website content in the database, but can't access it right now. Please try again later.",
                "contexts": []
            }
        
        # Extract metadata to find website title, main topics, etc.
        sources = set()
        titles = set()
        
        for meta in metadatas:
            if isinstance(meta, dict):
                source = meta.get('source', '')
                title = meta.get('title', '')
                if source and 'http' in source:
                    sources.add(source)
                if title:
                    titles.add(title)
        
        # Concatenate some of the document content for analysis
        content_sample = "\n\n".join(documents[:10])
        
        # Use the deep model for website summary (more comprehensive)
        primary_model = self.llm
        backup_model = self.deep_backup_llm
        
        # Create a prompt that guides the model to generate a summary
        summary_prompt = """Based on the following website content, provide a comprehensive yet concise summary of what this website is about. 
Focus on identifying:
1. The main purpose or subject of the website
2. Key topics or themes covered
3. The target audience
4. Any services, products, or information it provides

If the website appears to be about a specific organization, product, or service, clearly state what it is.

WEBSITE CONTENT SAMPLES:
"""
        
        # Add source and title information
        if sources:
            summary_prompt += "\nWebsite URLs:\n" + "\n".join(sorted(list(sources)[:5])) + "\n\n"
        
        if titles:
            summary_prompt += "Page Titles:\n" + "\n".join(sorted(list(titles)[:5])) + "\n\n"
        
        # Add content samples
        summary_prompt += "Content Excerpts:\n" + content_sample
        
        summary_prompt += "\n\nBased on the above information, provide a clear, accurate summary of what this website is about in 2-3 paragraphs."
        
        try:
            # Try with primary model first using enhanced prompt
            t_start = time.time()
            answer = primary_model.answer_with_rag(summary_prompt, [], temperature=0.3, max_tokens=512)
            print(f"[DEBUG] Website summary generated in {time.time() - t_start:.2f}s")
        except Exception as e:
            # If primary model fails, fall back to backup model
            print(f"[DEBUG] Primary LLM failed with error: {str(e)}. Falling back to backup model.")
            answer = backup_model.answer_with_rag(summary_prompt, [], temperature=0.3, max_tokens=512)
        
        # Prepare contexts for reference
        contexts = []
        for i, (doc, meta) in enumerate(zip(documents[:3], metadatas[:3])):
            contexts.append({
                "text": doc,
                "source": meta.get('source', '') if isinstance(meta, dict) else '',
                "title": meta.get('title', '') if isinstance(meta, dict) else ''
            })
        
        return {
            "answer": answer,
            "contexts": contexts
        }
    
    def scrape_website(self, url=None):
        """
        Scrape a website and process its content.
        
        Args:
            url (str, optional): URL to scrape, uses base_url if not provided
            
        Returns:
            int: Number of documents added to the vector database
        """
        import traceback
        target_url = url or self.base_url
        if not target_url:
            raise ValueError("No URL provided for scraping")
        
        print(f"Starting website scraping process for {target_url}")
        
        try:
            # Initialize the scraper
            scraper = WebScraper(
                base_url=target_url,
                delay=self.scraper_config['delay'],
                max_pages=self.scraper_config['max_pages']
            )
            
            # Scrape the website
            pages_content = scraper.scrape()
            print(f"[DEBUG] Scraper returned {len(pages_content)} pages with content")
            if len(pages_content) == 0:
                print("[DEBUG] No pages with content were scraped. Aborting pipeline.")
                return 0
            
            # Process the scraped content
            print("Processing scraped content...")
            documents = self.processor.process_all_pages(pages_content)
            print(f"[DEBUG] Processor returned {len(documents)} chunks")
            if len(documents) == 0:
                print("[DEBUG] No chunks generated from scraped content. Aborting pipeline.")
                return 0
            
            # Generate embeddings
            print("Generating embeddings...")
            embedded_documents = self.embedding_model.embed_documents(documents)
            print(f"[DEBUG] Embedded {len(embedded_documents)} chunks")
            if len(embedded_documents) == 0:
                print("[DEBUG] No embeddings generated. Aborting pipeline.")
                return 0
            
            # Store in vector database
            print("Storing in vector database...")
            self.vectordb.add_documents(embedded_documents)
            print(f"Scraping and processing complete. Added {len(embedded_documents)} documents to the vector database.")
            return len(embedded_documents)
        except Exception as e:
            print(f"[ERROR] scrape_website failed: {e}")
            traceback.print_exc()
            return 0
    
    def query(self, user_query, n_results=1, temperature=0.3, max_tokens=128, mode="speed"):  # Default to speed mode
        """
        Answer a user query using the RAG system, supporting multiple modes with fallback options.
        Args:
            user_query (str): User's question
            n_results (int): Number of results to retrieve
            temperature (float): LLM temperature parameter
            max_tokens (int): Maximum tokens for LLM response
            mode (str): "quick", "speed", "deep", or "deepseek"
        Returns:
            Dict: Contains answer and retrieved contexts
        """
        import time
        print(f"Processing query: {user_query} [mode={mode}]")
        
        # Check if this is a greeting
        if self.is_greeting(user_query):
            print(f"[DEBUG] Detected greeting, providing standard greeting response")
            return self.handle_greeting(mode)
        
        # Check if this is a website summary request
        if self.is_website_summary_request(user_query):
            print(f"[DEBUG] Detected website summary request, generating comprehensive summary")
            return self.get_website_summary(mode)
            
        try:
            t0 = time.time()
            # Set mode-specific parameters
            if mode == "quick":
                n_results = 2
                temperature = 0.3
                max_tokens = 256
                primary_model = self.quick_llm
                backup_model = self.quick_backup_llm
            elif mode == "speed":
                n_results = 1
                temperature = 0.2
                max_tokens = 128
                primary_model = self.speed_llm
                backup_model = self.speed_backup_llm
            elif mode == "deepseek":
                n_results = 3
                temperature = 0.4
                max_tokens = 256
                primary_model = self.deepseek_llm
                backup_model = self.deepseek_backup_llm
            else:  # deep mode (default)
                n_results = 4  # Reduced from 8 to 4 for better performance
                temperature = 0.5
                max_tokens = 512  # Reduced from 1024 to 512
                primary_model = self.llm
                backup_model = self.deep_backup_llm
            
            # Check if collection has documents
            stats = self.vectordb.get_collection_stats()
            if stats.get('document_count', 0) == 0:
                return {
                    "answer": "I don't have any website content in my database yet. Please scrape a website first using the 'Start Scraping' button.",
                    "contexts": []
                }
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(user_query)
            print(f"[DEBUG] Query embedding shape: {query_embedding.shape if hasattr(query_embedding, 'shape') else type(query_embedding)}")
            
            # Retrieve relevant documents
            retrieved_docs = self.vectordb.query(query_embedding, n_results=n_results)
            print(f"[DEBUG] Retrieved {len(retrieved_docs)} docs from vector DB")
            if not retrieved_docs:
                print("[DEBUG] No relevant documents found for query.")
                return {
                    "answer": "I couldn't find any relevant information about that in the website content. Could you try asking a different question related to the website you scraped?",
            # Create an enhanced prompt with few-shot examples for better accuracy
            if mode == "flash":
                # Specialized prompt for Flash mode with few-shot examples
                enhanced_prompt = f"""You are an expert assistant specializing in Contisoft Technologies and their products including RenewalHelp, supplier management solutions, and IT services.

Context information about Contisoft Technologies is below:
---
"""
                for i, doc in enumerate(retrieved_docs):
                    enhanced_prompt += f"{doc['content']}\n\n"
                
                enhanced_prompt += """---

Follow these steps carefully to answer the user's question about Contisoft Technologies:
1. Read ALL context information completely and identify ALL relevant facts
2. Extract the EXACT information from the context needed to answer correctly and comprehensively
3. Formulate a direct, factual answer using ONLY information present in the context
4. Include specific product names, features, or services mentioned when relevant
5. Use exact terminology, product names, and descriptions from the context

If the specific information isn't explicitly stated in the context, say "I don't have enough information about that aspect in the website content."

Here are some examples of good responses:

Example 1:
Question: What is RenewalHelp?
Answer: RenewalHelp is a flagship product that automates the software renewal process. It tracks license expirations, sends timely reminders, and provides a centralized dashboard for managing all software renewals in one place.

Example 2:
Question: Does the company offer consulting services?
Answer: Yes, the company offers IT consulting services focused on digital transformation, cloud migration, and software implementation. Their consultants work with businesses to optimize IT infrastructure and implement custom software solutions.

Example 3:
Question: What programming languages do they use?
Answer: I don't have enough information about the specific programming languages used in the website content.

Question: {user_query}
Answer: """
            else:
                # Standard prompt for other modes
                enhanced_prompt = f"""Based on the following context from a website, please answer the user's question.
Be friendly, helpful, and concise. Provide a direct and factual answer to the question based only on the information in the context.

User question: {user_query}

Website context:
"""
                for i, doc in enumerate(retrieved_docs):
                    enhanced_prompt += f"[Document {i+1}]: {doc['content']}\n\n"
                
                enhanced_prompt += """
Instructions:
1. Stay strictly within the facts presented in the context
2. If the context doesn't provide enough information, clearly state that
3. Be direct and to the point in your answer
4. Do not mention the documents or refer to them explicitly in your answer

Your answer:"""
            try:
                # Try with primary model first using enhanced prompt
                answer = primary_model.answer_with_rag(enhanced_prompt, [], temperature=temperature, max_tokens=max_tokens)
                print(f"[DEBUG] Primary LLM answer generated successfully")
            except Exception as e:
                # If primary model fails, fall back to backup model
                print(f"[DEBUG] Primary LLM failed with error: {str(e)}. Falling back to backup model.")
                answer = backup_model.answer_with_rag(enhanced_prompt, [], temperature=temperature, max_tokens=max_tokens)
                print(f"[DEBUG] Backup LLM answer generated successfully")
        
        t_llm_end = time.time()
        print(f"[DEBUG] LLM answer: {answer[:200]}...")
        print(f"[DEBUG] LLM response time: {t_llm_end - t0:.2f}s")
        
        # Format contexts for response
        contexts = []
        for doc in retrieved_docs:
            contexts.append({
                "text": doc['content'],
                "source": doc['metadata'].get('source', ''),
                "title": doc['metadata'].get('title', '')
            })
        print(f"[DEBUG] Total query time: {time.time() - t0:.2f}s")
        return {
            "answer": answer,
            "contexts": contexts
        }
    except Exception as e:
        import traceback
        print(f"[ERROR] query failed: {e}")
        traceback.print_exc()
        return {
            "answer": "I'm having trouble processing your question. Please try again or try a different question about the website content.",
            "contexts": []
        }

def reset_database(self):
    """Reset the vector database."""
    self.vectordb.reset_collection()
    return {"status": "success", "message": "Vector database reset successfully"}
            
            t_llm_end = time.time()
            print(f"[DEBUG] LLM answer: {answer[:200]}...")
            print(f"[DEBUG] LLM response time: {t_llm_end - t_llm_start:.2f}s")
            
            # Format contexts for response
            contexts = []
            for doc in retrieved_docs:
                contexts.append({
                    "text": doc['content'],
                    "source": doc['metadata'].get('source', ''),
                    "title": doc['metadata'].get('title', '')
                })
            print(f"[DEBUG] Total query time: {time.time() - t0:.2f}s")
            return {
                "answer": answer,
                "contexts": contexts
            }
        except Exception as e:
            import traceback
            print(f"[ERROR] query failed: {e}")
            traceback.print_exc()
            return {
                "answer": "I'm having trouble processing your question. Please try again or try a different question about the website content.",
                "contexts": []
            }
    
    def reset_database(self):
        """Reset the vector database."""
        self.vectordb.reset_collection()
        return {"status": "success", "message": "Vector database reset successfully"}


if __name__ == "__main__":
    # Example usage
    rag = RAG(base_url="https://example.com")
    
    # Scrape website (uncomment to run)
    # rag.scrape_website()
    
    # Example query
    result = rag.query("What services do you offer?")
    print(f"Answer: {result['answer']}")
    print(f"Based on {len(result['contexts'])} context documents")
