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
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size=150) -> None:
        """
        Add documents to the vector database in batches to avoid exceeding ChromaDB's limit.
        
        Args:
            documents (List[Dict]): List of documents to add
            batch_size (int): Maximum number of documents to add in a single batch
                               (ChromaDB has a limit of 166, so we stay below that)
        """
        # Make sure the collection exists
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
        except Exception:
            print(f"Collection not found, creating new collection: {self.collection_name}")
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Process documents in batches
        total_docs = len(documents)
        print(f"Adding {total_docs} documents to vector database in batches of {batch_size}...")
        
        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            batch = documents[batch_start:batch_end]
            
            # Generate IDs for this batch
            # Use a global counter to ensure unique IDs across batches
            ids = [f"doc_{batch_start + i}" for i in range(len(batch))]
            
            # Extract data for this batch
            contents = [doc['content'] for doc in batch]
            metadatas = [doc['metadata'] for doc in batch]
            
            # Check if embeddings are provided
            if 'embedding' in batch[0]:
                embeddings = [doc['embedding'] for doc in batch]
                # Add documents with embeddings
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
            
            print(f"Added batch {batch_start//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}: {len(batch)} documents")
            
        print(f"Successfully added all {total_docs} documents to the vector database.")
    
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
    
    def get_all_documents(self):
        """
        Get all documents from the vector database.
        
        Returns:
            List[Dict]: List of documents with their content and metadata
        """
        try:
            # Make sure the collection exists
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            
            # Get all documents
            results = self.collection.get()
            
            # Format the results into a list of dictionaries
            documents = []
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                documents.append({
                    'content': doc,
                    'metadata': metadata
                })
                
            return documents
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
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
