"""
Test RAG responses to verify it's using the web-scraped data correctly.
"""

from src.rag import RAG

def test_rag_responses():
    """Test if the RAG system is giving responses based on scraped data."""
    # Initialize RAG system
    rag = RAG()
    
    # Get database stats
    stats = rag.vectordb.get_collection_stats()
    print(f"Vector database contains {stats.get('document_count')} documents")
    
    # Try different query modes
    test_queries = [
        "What services does this company offer?",
        "Tell me about the main products",
        "Who are the key team members?",
        "What technologies do you use?"
    ]
    
    # Test with different modes
    modes = ["speed", "quick", "deep"]
    
    for query in test_queries:
        print("\n" + "="*50)
        print(f"QUERY: {query}")
        print("="*50)
        
        for mode in modes:
            print(f"\n--- Mode: {mode} ---")
            result = rag.query(user_query=query, mode=mode)
            
            print(f"Answer: {result['answer']}")
            print(f"Based on {len(result['contexts'])} documents:")
            
            # Show snippets of context documents
            for i, ctx in enumerate(result['contexts']):
                source = ctx.get('source', 'Unknown')
                snippet = ctx.get('text', '')[:150] + "..." if len(ctx.get('text', '')) > 150 else ctx.get('text', '')
                print(f"  Doc {i+1} from {source}: {snippet}")
            
            print(f"Response time: {result.get('time', 'N/A')}s")
    
    # Clean up
    print("\nTest completed.")

if __name__ == "__main__":
    test_rag_responses() 