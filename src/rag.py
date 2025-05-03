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

# Optional cross-encoder for reranking
try:
    from sentence_transformers import CrossEncoder
except ImportError:  # Fallback when package is not installed
    CrossEncoder = None


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

        # --- Prompt engineering: few-shot QA examples for Contisoft Techno ---
        self.few_shot_examples = [
            {
                'q': 'What services does Contisoft Techno offer?',
                'a': 'Contisoft Techno offers web development, mobile app development, and digital marketing services.'
            },
            {
                'q': 'Where is Contisoft Techno located?',
                'a': 'Contisoft Techno is located in Chennai, Tamil Nadu, India.'
            },
            {
                'q': 'What is RenewalHelp?',
                'a': 'RenewalHelp is a software product by Contisoft Techno for managing renewals and subscriptions.'
            }
        ]
        
        # --- Primary LLMs for modes ---
        self.quick_model_name = self.config.get('quick_model_name', os.getenv('QUICK_MODEL_NAME', 'llama3.2:3b-instruct-q3_K_M'))
        self.speed_model_name = self.config.get('speed_model_name', os.getenv('SPEED_MODEL_NAME', 'llama3.2:1b-instruct-q2_K'))
        self.deep_model_name = self.config.get('deep_model_name', os.getenv('DEEP_MODEL_NAME', 'phi3.5:3.8b-mini-instruct-q3_K_S'))
        self.deepseek_model_name = self.config.get('deepseek_model_name', os.getenv('DEEPSEEK_MODEL_NAME', 'deepseek-r1:1.5b'))
        
        # --- Backup LLMs for each mode ---
        self.quick_backup_model = self.config.get('quick_backup_model', os.getenv('QUICK_BACKUP_MODEL', 'tinyllama:1.1b-chat'))
        self.speed_backup_model = self.config.get('speed_backup_model', os.getenv('SPEED_BACKUP_MODEL', 'phi:latest'))
        self.deep_backup_model = self.config.get('deep_backup_model', os.getenv('DEEP_BACKUP_MODEL', 'gemma:2b-instruct-q4_0'))
        self.deepseek_backup_model = self.config.get('deepseek_backup_model', os.getenv('DEEPSEEK_BACKUP_MODEL', 'deepseek-coder:1.3b'))  # Update the DeepSeek backup model configuration
        
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
            r'describe\s+(?:this|the)\s+website',
            # Add more forgiving patterns that can handle typos
            r'(tell|give|brief|briefing|summarize|summary|overview|describe).*website',
            r'(tell|give|brief|briefing|summarize|summary|overview|describe).*site',
            r'what.*website.*about',
            r'what.*site.*about',
            r'.*about.*website',
            r'.*about.*site'
        ]
        
        # Lightweight cross-encoder re-ranker (improves Flash accuracy)
        self.re_ranker = None
        if CrossEncoder is not None:
            try:
                # MiniLM model is ~120 MB and fast enough for Flash mode
                self.re_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
                print('[INFO] Cross-encoder re-ranker initialised for Flash mode.')
            except Exception as _e:
                print(f'[WARNING] Could not load cross-encoder model: {_e}')
                self.re_ranker = None
    
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
    
    def check_for_typos(self, query):
        """
        Check for common typos in queries and suggest corrections.
        Returns (corrected_query, has_correction)
        """
        query_lower = query.lower().strip()
        
        # Common typo corrections
        typo_corrections = {
            'websiye': 'website',
            'websit': 'website',
            'webste': 'website',
            'wesite': 'website',
            'webite': 'website',
            'websie': 'website',
            'wbsite': 'website',
            'ebsite': 'website',
            'bref': 'brief',
            'brif': 'brief',
            'breif': 'brief',
            'ef': 'brief',
            'sumary': 'summary',
            'summry': 'summary',
            'sumarize': 'summarize',
            'summerize': 'summarize',
            'sumrize': 'summarize',
            'desribe': 'describe',
            'descibe': 'describe',
            'discribe': 'describe'
        }
        
        # Check if any typos are in the query
        corrected_query = query_lower
        has_correction = False
        
        for typo, correction in typo_corrections.items():
            if typo in query_lower:
                corrected_query = corrected_query.replace(typo, correction)
                has_correction = True
        
        return corrected_query, has_correction
    
    def has_documents(self):
        """Check if the vector database has any documents."""
        stats = self.vectordb.get_collection_stats()
        return stats.get('document_count', 0) > 0
    
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
        if not self.has_documents():
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
    
    def query(self, user_query, n_results=1, temperature=0.1, max_tokens=512, mode="default"):  
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
        
        # Start timing
        t0 = time.time()
        
        # Check for typos and correct if needed
        corrected_query, has_correction = self.check_for_typos(user_query)
        
        # If we corrected typos, use the corrected query for processing
        # but keep the original for reference
        original_query = user_query
        if has_correction:
            print(f"[DEBUG] Corrected typos: '{original_query}' -> '{corrected_query}'")
            user_query = corrected_query
        
        # Check if this is a greeting
        if self.is_greeting(user_query):
            print(f"[DEBUG] Detected greeting, providing standard greeting response")
            return self.handle_greeting(mode)
        
        # Check if this is a website summary request
        if self.is_website_summary_request(user_query):
            print(f"[DEBUG] Detected website summary request, generating comprehensive summary")
            return self.get_website_summary(mode)
            
        try:
            # Check if we have documents
            if not self.has_documents():
                return {
                    "answer": "No documents in the database. Please scrape a website first.",
                    "contexts": []
                }
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(user_query)
            print(f"[DEBUG] Query embedding shape: {query_embedding.shape if hasattr(query_embedding, 'shape') else type(query_embedding)}")
            
            # --- Hybrid Retrieval for Flash mode ---
            if mode == "flash":
                n_semantic = 10  # semantic results
                n_results = 15  # final number of docs to feed LLM

                # 1. Semantic search (vector DB)
                semantic_docs = self.vectordb.query(query_embedding, n_results=n_semantic)

                # 2. Keyword / phrase search over *all* documents
                query_words   = user_query.lower().split()
                query_phrases = [" ".join(query_words[i:i+2]) for i in range(len(query_words)-1)]
                query_word_set = set(query_words)

                # Start the reranking list with the semantic results.  Give them a high
                # base-score so they are preferred when relevance is comparable.
                reranked_docs: List[Tuple[float, Dict[str, Any]]] = [(100.0, doc) for doc in semantic_docs]

                all_docs = self.vectordb.get_all_documents() if hasattr(self.vectordb, "get_all_documents") else []
                for doc in all_docs:
                    # Skip docs already included from the semantic search
                    if doc in semantic_docs:
                        continue

                    content_lower = doc["content"].lower()

                    # --- Scoring heuristics ----------------------------------------------------
                    keyword_score = len(query_word_set.intersection(set(content_lower.split())))               # 1 pt / keyword
                    phrase_score  = sum(2 for phrase in query_phrases if phrase in content_lower)              # 2 pt / 2-gram match
                    title_score   = 0
                    if isinstance(doc.get("metadata"), dict) and doc["metadata"].get("title"):
                        title_lower  = doc["metadata"]["title"].lower()
                        title_score = len(query_word_set.intersection(set(title_lower.split()))) * 2          # 2 pt / keyword in title
                    length_score  = min(len(content_lower.split()) // 150, 3)                                  # favour richer docs
                    total_score   = keyword_score + phrase_score + title_score + length_score
                    if total_score > 0:
                        reranked_docs.append((total_score, doc))

                # 3. Sort by score (desc) and de-duplicate w.r.t. source+chunk_id
                reranked_docs.sort(key=lambda x: x[0], reverse=True)
                retrieved_docs = []
                seen_ids = set()
                for score, doc in reranked_docs:
                    meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                    unique_id = f"{meta.get('source','')}#{meta.get('chunk_id', -1)}"
                    if unique_id in seen_ids:
                        continue
                    retrieved_docs.append(doc)
                    seen_ids.add(unique_id)
                    if len(retrieved_docs) >= n_results:
                        break

                print(f"[DEBUG] Enhanced hybrid retrieval: {len(retrieved_docs)} docs (semantic+keyword+phrase, advanced reranking)")

                # ---------------- Cross-encoder re-ranking -------------------
                if self.re_ranker and retrieved_docs:
                    try:
                        pairs   = [(user_query, doc['content'][:512]) for doc in retrieved_docs]
                        ce_scores = self.re_ranker.predict(pairs)
                        # combine with previous score by simple addition (ce is 0-5 scale)
                        for i, score in enumerate(ce_scores):
                            # add small weight so semantic still matters
                            retrieved_docs[i]['_ce_score'] = float(score)
                        retrieved_docs.sort(key=lambda d: d.get('_ce_score', 0), reverse=True)
                        retrieved_docs = retrieved_docs[:n_results]
                        print(f"[DEBUG] Cross-encoder reranked top {len(retrieved_docs)} docs")
                    except Exception as _e:
                        print(f"[WARNING] Cross-encoder reranking failed: {_e}")

                if not retrieved_docs:
                    print("[DEBUG] No relevant documents found for query.")
                    return {
                        "answer": "I couldn't find any relevant information about that in the website content. Could you try asking a different question related to the website you scraped?",
                        "contexts": []
                    }

                # Prepare the prompt with context and query
                context_text = "\n\n".join([doc['content'] for doc in retrieved_docs])
            else:
                # Retrieve relevant documents (original logic)
                retrieved_docs = self.vectordb.query(query_embedding, n_results=n_results)
                print(f"[DEBUG] Retrieved {len(retrieved_docs)} docs from vector DB")
                if not retrieved_docs:
                    print("[DEBUG] No relevant documents found for query.")
                    return {
                        "answer": "I couldn't find any relevant information about that in the website content. Could you try asking a different question related to the website you scraped?",
                        "contexts": []
                    }
                # Prepare the prompt with context and query
                context_text = "\n\n".join([doc['content'] for doc in retrieved_docs])
            
            # Use the appropriate prompt template based on mode
            if mode == "speed" or mode == "flash":
                prompt_template = """You are an expert assistant specializing in Contisoft Technologies and their products including RenewalHelp, supplier management solutions, and IT services.

Context information about Contisoft Technologies is below:
---
{context}
---

Follow these steps carefully to answer the user's question about Contisoft Technologies:
1. Read ALL context information completely and identify ALL relevant facts about Contisoft Technologies
2. Extract the EXACT information from the context needed to answer correctly and comprehensively
3. Formulate a direct, factual answer using ONLY information present in the context
4. Include specific product names, features, or services that Contisoft offers when relevant
5. Use exact terminology, product names, and descriptions from the context about Contisoft Technologies

If the specific information isn't explicitly stated in the context, say "I don't have enough information about that aspect of Contisoft Technologies in the website content."

Do NOT add ANY information beyond what's in the context, even if it seems obvious or helpful.
Respond directly to the question with relevant details about Contisoft Technologies and their offerings.

Question: {question}
Thinking: I'll carefully analyze ALL the context provided about Contisoft Technologies to find the exact information...
Answer: """
            elif mode == "deepseek":
                prompt_template = """You are a code and technical expert that provides precise answers based on the provided context.

Context information is below:
---
{context}
---

Given the context information and no prior knowledge, provide a technically accurate answer to the question.
If the question involves code, provide working code examples and explanations.
If the answer cannot be determined from the context, say "I don't have enough information to answer this question."

Question: {question}
Answer: """
            else:  # deep mode (default)
                prompt_template = """You are a helpful, accurate assistant that provides detailed answers based on the provided context.

Context information is below:
---
{context}
---

Given the context information and no prior knowledge, provide a comprehensive answer to the question.
If the answer cannot be determined from the context, say "I don't have enough information to answer this question."
Include relevant details from the context and organize your answer in a clear, readable format.

Question: {question}
Answer: """
            
            # Set mode-specific parameters
            if mode == "quick":
                n_results = 2
                temperature = 0.3
                max_tokens = 256
                primary_model = self.quick_llm
                backup_model = self.quick_backup_llm
            elif mode == "speed" or mode == "flash":
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
            else:  # deep mode
                n_results = 4  # Reduced from 8 to 4 for better performance
                temperature = 0.5
                max_tokens = 512  # Reduced from 1024 to 512
                primary_model = self.llm
                backup_model = self.deep_backup_llm
            
            # Prepare the prompt with context and query
            prompt = prompt_template.format(context=context_text, question=corrected_query)
            
            # Try primary model first
            try:
                response = primary_model.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                # Make sure response is a string before calling strip()
                if isinstance(response, dict) and 'response' in response:
                    answer = response['response'].strip()
                elif isinstance(response, str):
                    answer = response.strip()
                else:
                    answer = str(response)
            except Exception as e:
                # Log the error and try backup model
                print(f"Error with primary model: {str(e)}")
                try:
                    response = backup_model.generate(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    # Make sure response is a string before calling strip()
                    if isinstance(response, dict) and 'response' in response:
                        answer = response['response'].strip()
                    elif isinstance(response, str):
                        answer = response.strip()
                    else:
                        answer = str(response)
                except Exception as e2:
                    # If both models fail, return error
                    print(f"Error with backup model: {str(e2)}")
                    return {
                        "answer": f"Error generating response: {str(e2)}",
                        "contexts": []
                    }
            
            # If we corrected typos, add a note at the beginning of the answer
            if has_correction and not answer.startswith("I noticed"):
                answer = f"I noticed you may have meant '{corrected_query}' instead of '{original_query}'. Here's my answer: \n\n{answer}"
            
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
            print(f"[ERROR] Query failed: {str(e)}")
            print(traceback.format_exc())
            return {
                "answer": f"I encountered an error while processing your query. Please try again or rephrase your question. Error details: {str(e)}",
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
