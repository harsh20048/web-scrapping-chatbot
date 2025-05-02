"""
Vector database module for storing and retrieving document embeddings.
Uses ChromaDB as the underlying vector store.
"""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np


class VectorDB:
    def __init__(self, persist_directory="./data/chroma", collection_name="website_content"):
        """
        Initialize the vector database.
        
        Args:
            persist_directory (str): Directory to persist the database
            collection_name (str): Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure the persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Initialize collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except Exception as e:
            print(f"Creating new collection: {collection_name}")
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents (List[Dict]): List of documents to add
        """
        # Extract documents in the format expected by ChromaDB
        ids = [f"doc_{i}" for i in range(len(documents))]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        embeddings = [doc['embedding'] for doc in documents if 'embedding' in doc]
        
        # Make sure the collection exists
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
        except Exception:
            print(f"Collection not found, creating new collection: {self.collection_name}")
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Add documents to the collection
        if embeddings and len(embeddings) == len(documents):
            # If embeddings are provided
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )
        else:
            # If embeddings are not provided, ChromaDB will generate them
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
    
    def query(self, embedding, n_results=5):
        """
        Query the vector database.
        
        Args:
            embedding: Query embedding
            n_results (int): Number of results to return
            
        Returns:
            List[Dict]: List of retrieved documents
        """
        # Make sure the collection exists
        try:
            # Convert numpy array to list if needed
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
                
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
            
            # Format results
            documents = []
            for i in range(len(results['documents'][0])):
                document = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                documents.append(document)
            
            return documents
        except Exception as e:
            print(f"Error querying vector database: {e}")
            return []
    
    def get_collection_stats(self):
        """
        Get statistics about the collection.
        
        Returns:
            Dict: Collection statistics
        """
        try:
            # Get collection count
            count = self.collection.count()
            
            # Return statistics
            return {
                'document_count': count
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {
                'document_count': 0,
                'error': str(e)
            }
    
    def reset_collection(self):
        """Reset (delete and recreate) the collection."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
        
        # Recreate collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection: {self.collection_name}")


if __name__ == "__main__":
    # Example usage
    db = VectorDB()
    stats = db.get_collection_stats()
    print(f"Collection has {stats['document_count']} documents")
