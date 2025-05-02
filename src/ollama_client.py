"""
Ollama client module for interacting with the Mistral 7B LLM.
Handles prompting and response generation.
"""

import requests
from typing import Dict, Any, List, Optional


class OllamaClient:
    def __init__(self, host="http://localhost:11434", model="phi:latest"):
        """
        Initialize the Ollama client.
        
        Args:
            host (str): Host address of the Ollama server
            model (str): Model to use for generation
        """
        self.host = host
        self.model = model
        self.api_url = f"{host}/api/generate"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """
        Test the connection to the Ollama server.
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code != 200:
                print(f"Warning: Ollama server returned status code {response.status_code}")
                print(f"Make sure Ollama is running at {self.host} and the model {self.model} is downloaded")
                print(f"You can download the model with: ollama pull {self.model}")
            else:
                models = response.json().get("models", [])
                # Extract just the model names without version/tags for comparison
                model_names = [model.get("name").split(':')[0] for model in models]
                # Get base model name without tag for comparison
                base_model = self.model.split(':')[0] if ':' in self.model else self.model
                
                if base_model not in model_names:
                    print(f"Warning: Model {self.model} not found in available models: {[m.get('name') for m in models]}")
                    print(f"You can download the model with: ollama pull {self.model}")
                else:
                    print(f"Successfully connected to Ollama server at {self.host}")
                    print(f"Using model: {self.model}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to Ollama server at {self.host}")
            print(f"Error: {e}")
            print("Make sure Ollama is installed and running")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt (str): User prompt
            system_prompt (str, optional): System prompt for context
            temperature (float): Sampling temperature (0.0 to 1.0)
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            Dict: Response from the LLM
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error generating response: {e}")
            return {"response": f"Error: {str(e)}"}
    
    def create_rag_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Create a RAG (Retrieval-Augmented Generation) prompt with context.
        
        Args:
            query (str): User query
            context_docs (List[Dict]): Retrieved context documents
            
        Returns:
            Dict: Contains system_prompt and user_prompt
        """
        # Create context string from retrieved documents - shorter format
        context_parts = []
        
        for i, doc in enumerate(context_docs):
            content = doc['content']
            metadata = doc['metadata']
            source = metadata.get('source', 'Unknown')
            title = metadata.get('title', 'Unknown')
            
            # Format with source and title for better traceability
            context_part = f"[Doc {i+1}] Source: {source}\nTitle: {title}\nContent: {content}"
            context_parts.append(context_part)
        
        context_string = "\n\n".join(context_parts)
        
        # Updated system prompt with strict guidelines against fabrication
        system_prompt = """You are a chatbot that ONLY answers questions based on the provided context from scraped website content.

CRITICAL RULES:
1. NEVER invent or fabricate information not present in the context
2. NEVER create fictional scenarios, puzzles, or content not in the context
3. NEVER respond with phrases like "As an AI language model" or "I'm sorry, but as an AI language model..."
4. If you don't know the answer based on the context, say ONLY: "I don't have enough information from the website content to answer that question."
5. ALWAYS maintain a direct, concise tone as the website's representative
6. If asked to describe or brief about the website, ONLY include information specifically found in the context
7. DO NOT MAKE ANYTHING UP - stick STRICTLY to what's in the context"""
        
        # Explicit user prompt to reinforce proper behavior
        user_prompt = f"""Context information from website:\n{context_string}\n\nUser question: {query}\n\nAnswer based ONLY on the above context. Do not make up information that isn't there. If the answer isn't in the context, say you don't have enough information."""
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }
    
    def answer_with_rag(self, query: str, context_docs: List[Dict[str, Any]], 
                         temperature: float = 0.3, max_tokens: int = 128) -> str:
        """
        Generate an answer using RAG approach.
        
        Args:
            query (str): User query
            context_docs (List[Dict]): Retrieved context documents
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: Generated answer
        """
        prompts = self.create_rag_prompt(query, context_docs)
        
        response = self.generate(
            prompt=prompts["user_prompt"],
            system_prompt=prompts["system_prompt"],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.get("response", "Sorry, I couldn't generate a response.")


if __name__ == "__main__":
    # Example usage
    client = OllamaClient()
    
    # Test simple generation
    response = client.generate(
        prompt="What is the capital of France?",
        system_prompt="You are a helpful assistant."
    )
    
    print("Example response:")
    print(response.get("response", "No response"))
