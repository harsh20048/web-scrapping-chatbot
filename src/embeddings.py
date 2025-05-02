"""
Embeddings module for converting text to vector representations.
Uses sentence-transformers for generating embeddings.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): Name of the model to use for embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded. Dimension: {self.embedding_dim}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of document chunks.
        
        Args:
            documents (List[Dict]): List of document chunks
            
        Returns:
            List[Dict]: Documents with embeddings added
        """
        # Extract text content from documents
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings for all texts
        embeddings = self.get_embeddings(texts)
        
        # Add embeddings to documents
        for i, doc in enumerate(documents):
            # Convert numpy array to list before storing
            if hasattr(embeddings[i], 'tolist'):
                doc['embedding'] = embeddings[i].tolist()
            else:
                doc['embedding'] = embeddings[i]
        
        return documents
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.
        
        Args:
            query (str): Query string
            
        Returns:
            np.ndarray: Query embedding
        """
        embedding = self.model.encode(query)
        # Keep as numpy array since some callers may expect it
        # The vectordb will handle conversion to list if needed
        return embedding


if __name__ == "__main__":
    # Example usage
    embedding_model = EmbeddingModel()
    
    # Test with some example texts
    example_texts = [
        "This is a test sentence.",
        "This is another test sentence, but longer.",
        "This is a completely different sentence about another topic."
    ]
    
    embeddings = embedding_model.get_embeddings(example_texts)
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings[0].shape}")
    
    # Compute similarity between first two sentences
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])
    print(f"Similarity between first two sentences: {sim[0][0]:.4f}")
    
    # Compare to similarity with the third sentence
    sim = cosine_similarity([embeddings[0]], [embeddings[2]])
    print(f"Similarity between first and third sentences: {sim[0][0]:.4f}")
