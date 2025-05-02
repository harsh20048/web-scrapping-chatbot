import os
import requests
from typing import List, Dict, Any

class GeminiClient:
    def __init__(self, api_key=None, model="gemini-pro"):
        """
        Simple Gemini API client for Google Gemini Free API.
        Args:
            api_key (str): Google API key (if None, will use env var GEMINI_API_KEY)
            model (str): Gemini model name
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google Gemini API key must be provided or set in GEMINI_API_KEY env var.")
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    def answer_with_rag(self, query: str, context_docs: List[Dict[str, Any]], temperature: float = 0.3, max_tokens: int = 256) -> str:
        """
        Generate an answer using Gemini API.
        Args:
            query (str): User query
            context_docs (List[Dict]): Retrieved context documents
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate
        Returns:
            str: Generated answer
        """
        context_parts = []
        
        for i, doc in enumerate(context_docs):
            content = doc.get('text', doc.get('content', ''))
            source = doc.get('source', 'Unknown')
            title = doc.get('title', 'Unknown')
            
            # Format with source and title for better traceability
            context_part = f"[Doc {i+1}] Source: {source}\nTitle: {title}\nContent: {content}"
            context_parts.append(context_part)
        
        context_string = "\n\n".join(context_parts)
        
        prompt = f"""CRITICAL INSTRUCTIONS:
You are a chatbot that ONLY answers questions based on the provided context from scraped website content.

Context information from website:
{context_string}

User question: {query}

CRITICAL RULES:
1. NEVER invent or fabricate information not present in the context
2. NEVER create fictional scenarios, puzzles, or content not in the context
3. NEVER respond with phrases like "As an AI language model" or "I'm sorry, but as an AI language model..."
4. If you don't know the answer based on the context, say ONLY: "I don't have enough information from the website content to answer that question."
5. ALWAYS maintain a direct, concise tone as the website's representative
6. If asked to describe or brief about the website, ONLY include information specifically found in the context
7. DO NOT MAKE ANYTHING UP - stick STRICTLY to what's in the context

Answer based ONLY on the above context. Do not make up information that isn't there. If the answer isn't in the context, say you don't have enough information."""
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            # Gemini returns answer in data['candidates'][0]['content']['parts'][0]['text']
            return data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "Sorry, I couldn't generate a response.")
        except Exception as e:
            print(f"[GeminiClient] Error: {e}")
            return f"Error: {str(e)}"
