"""
Script to view the contents of the vector database.
"""

from src.rag import RAG

def view_database(limit=10):
    """
    View the contents of the vector database.
    
    Args:
        limit (int): Maximum number of documents to show
    """
    print(f"Viewing vector database contents (limit: {limit})")
    
    # Initialize RAG system
    rag = RAG()
    
    # Get stats
    stats = rag.vectordb.get_collection_stats()
    doc_count = stats.get('document_count', 0)
    print(f"Database contains {doc_count} total documents")
    
    # Get documents
    results = rag.vectordb.collection.get()
    documents = results.get('documents', [])
    metadatas = results.get('metadatas', [])
    
    # Show documents
    print(f"\nShowing {min(limit, len(documents))} of {len(documents)} documents:")
    print("=" * 80)
    
    for i, (doc, meta) in enumerate(zip(documents[:limit], metadatas[:limit])):
        print(f"Document {i+1}:")
        title = meta.get('title', 'No Title') if isinstance(meta, dict) else 'No Title'
        source = meta.get('source', 'Unknown') if isinstance(meta, dict) else 'Unknown'
        print(f"  Title: {title}")
        print(f"  Source: {source}")
        
        # Show truncated content
        content = doc[:300] + "..." if len(doc) > 300 else doc
        print(f"  Content: {content}")
        print("-" * 80)
    
    return {
        'total_documents': doc_count,
        'sample_documents': [{"title": m.get('title', 'No Title'), "source": m.get('source', 'Unknown')} 
                             for m in metadatas[:limit] if isinstance(m, dict)]
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='View vector database contents')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of documents to show')
    parser.add_argument('--source', type=str, help='Filter by source URL')
    
    args = parser.parse_args()
    
    view_database(args.limit) 