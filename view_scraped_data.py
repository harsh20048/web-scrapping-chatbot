#!/usr/bin/env python
"""
Utility to view scraped data stored in the Chroma vector database.
This script displays all documents stored in the database with their metadata.
"""

import os
import sys
from colorama import init, Fore, Style
from src.vectordb import VectorDB

def main():
    # Initialize colorama for colored output
    init()
    
    try:
        # Load the vector database
        persist_directory = './data/chroma'
        collection_name = 'website_content'
        
        print(f"{Fore.CYAN}Loading vector database from {persist_directory}...{Style.RESET_ALL}")
        db = VectorDB(persist_directory=persist_directory, collection_name=collection_name)
        
        # Get collection statistics
        stats = db.get_collection_stats()
        doc_count = stats.get('document_count', 0)
        
        if doc_count == 0:
            print(f"{Fore.YELLOW}No documents found in the database. Have you scraped any websites yet?{Style.RESET_ALL}")
            return
            
        print(f"{Fore.GREEN}Found {doc_count} documents in the database.{Style.RESET_ALL}")
        
        # Get all documents
        all_docs = db.get_all_documents() if hasattr(db, 'get_all_documents') else []
        
        if not all_docs:
            # Try alternative method if get_all_documents is not available
            print(f"{Fore.YELLOW}Using alternative method to retrieve documents...{Style.RESET_ALL}")
            collection = db.collection
            all_docs = []
            
            # Get documents in batches to avoid memory issues
            batch_size = 50
            for i in range(0, doc_count, batch_size):
                result = collection.get(limit=batch_size, offset=i)
                if result and 'documents' in result:
                    for j, doc in enumerate(result['documents']):
                        metadata = result['metadatas'][j] if 'metadatas' in result else {}
                        all_docs.append({
                            'content': doc,
                            'metadata': metadata
                        })
        
        # Display the documents
        print(f"{Fore.CYAN}==== Web Scraped Content ===={Style.RESET_ALL}")
        print(f"Total documents: {len(all_docs)}")
        
        # Group by source URL
        sources = {}
        for doc in all_docs:
            source = doc.get('metadata', {}).get('source', 'Unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(doc)
        
        print(f"{Fore.CYAN}Found content from {len(sources)} unique URLs{Style.RESET_ALL}")
        
        # Automatically display source summaries
        print("\nDisplaying source summaries:")
        choice = "1"  # Auto-select summary view
        
        if choice == '1':
            # Show summary of each source
            for i, (source, docs) in enumerate(sources.items(), 1):
                title = docs[0].get('metadata', {}).get('title', 'Unknown Title')
                print(f"\n{Fore.GREEN}[{i}] Source: {source}{Style.RESET_ALL}")
                print(f"Title: {title}")
                print(f"Chunks: {len(docs)}")
                # Show snippet of first chunk
                if docs:
                    content = docs[0].get('content', '')
                    print(f"Sample: {content[:150]}..." if len(content) > 150 else f"Sample: {content}")
        
        elif choice == '2':
            # Show all document chunks
            for i, doc in enumerate(all_docs, 1):
                source = doc.get('metadata', {}).get('source', 'Unknown')
                title = doc.get('metadata', {}).get('title', 'Unknown Title')
                chunk_id = doc.get('metadata', {}).get('chunk_id', 'Unknown')
                content = doc.get('content', '')
                
                print(f"\n{Fore.GREEN}[{i}/{len(all_docs)}] {title} (Source: {source}, Chunk: {chunk_id}){Style.RESET_ALL}")
                print(f"{content[:500]}..." if len(content) > 500 else content)
                
                # Pause after every 5 documents
                if i % 5 == 0 and i < len(all_docs):
                    input(f"{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")
        
        elif choice == '3':
            # Search documents by keyword
            keyword = input("Enter keyword to search for: ")
            keyword = keyword.lower()
            
            matching_docs = []
            for doc in all_docs:
                content = doc.get('content', '').lower()
                if keyword in content:
                    matching_docs.append(doc)
            
            print(f"{Fore.GREEN}Found {len(matching_docs)} documents containing '{keyword}'{Style.RESET_ALL}")
            
            for i, doc in enumerate(matching_docs, 1):
                source = doc.get('metadata', {}).get('source', 'Unknown')
                title = doc.get('metadata', {}).get('title', 'Unknown Title')
                content = doc.get('content', '')
                
                print(f"\n{Fore.GREEN}[{i}/{len(matching_docs)}] {title} (Source: {source}){Style.RESET_ALL}")
                
                # Highlight the keyword in the content
                highlight_content = content
                for line in content.split('\n'):
                    if keyword in line.lower():
                        start = max(0, line.lower().find(keyword) - 40)
                        excerpt = line[start:start+100]
                        highlight_content = f"...{excerpt}..."
                        break
                        
                print(highlight_content)
                
                # Pause after every 5 documents
                if i % 5 == 0 and i < len(matching_docs):
                    input(f"{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
