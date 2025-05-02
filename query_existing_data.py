"""
Script to test queries on the existing data in the RAG system.
"""

from src.rag import RAG

def test_query(query, mode="speed"):
    """
    Test a query on the RAG system.
    
    Args:
        query (str): The query to test
        mode (str): The query mode (speed, quick, or deep)
    """
    print(f"Testing query: '{query}' [mode={mode}]")
    
    # Initialize RAG system
    rag = RAG()
    
    # Get stats
    stats = rag.vectordb.get_collection_stats()
    doc_count = stats.get('document_count', 0)
    print(f"Database contains {doc_count} documents")
    
    # Process query
    result = rag.query(query, mode=mode)
    
    # Print results
    print("\nAnswer:")
    print("-" * 80)
    print(result['answer'])
    print("-" * 80)
    
    # Show sources
    print(f"\nBased on {len(result['contexts'])} documents:")
    for i, ctx in enumerate(result['contexts']):
        source = ctx.get('source', 'Unknown source')
        print(f"  Document {i+1}: {source}")
    
    return result

if __name__ == "__main__":
    print("RAG Query Tester")
    print("=" * 50)
    
    while True:
        query = input("\nEnter your query (or 'quit' to exit): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        mode = input("Enter mode (speed, quick, deep) [default: speed]: ").lower()
        if not mode or mode not in ['speed', 'quick', 'deep']:
            mode = 'speed'
            
        test_query(query, mode)
        
    print("\nThank you for testing the RAG system!") 