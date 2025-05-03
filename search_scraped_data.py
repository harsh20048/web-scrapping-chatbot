#!/usr/bin/env python
"""
Utility to search scraped data in the Chroma vector database.
This script allows searching for specific keywords to find related documents.
"""

import os
import sys
from colorama import init, Fore, Style
from src.vectordb import VectorDB
import argparse

def main():
    # Initialize colorama for colored output
    init()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Search scraped data for keywords')
    parser.add_argument('keyword', nargs='?', default=None, help='Keyword to search for')
    parser.add_argument('--count', '-c', type=int, default=5, help='Number of results to display')
    args = parser.parse_args()
    
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
            # Try alternative method
            print(f"{Fore.YELLOW}Using alternative method to retrieve documents...{Style.RESET_ALL}")
            collection = db.collection
            all_docs = []
            
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
        
        # Get keyword from command line or prompt
        keyword = args.keyword
        if not keyword:
            keyword = input("Enter keyword to search for: ")
        
        keyword = keyword.lower()
        
        # Search for the keyword
        matching_docs = []
        for doc in all_docs:
            content = doc.get('content', '').lower()
            if keyword in content:
                matching_docs.append(doc)
        
        print(f"{Fore.GREEN}Found {len(matching_docs)} documents containing '{keyword}'{Style.RESET_ALL}")
        
        # Display the matching documents
        for i, doc in enumerate(matching_docs[:args.count], 1):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            title = doc.get('metadata', {}).get('title', 'Unknown Title')
            content = doc.get('content', '')
            
            print(f"\n{Fore.GREEN}[{i}/{len(matching_docs)}] {title}{Style.RESET_ALL}")
            print(f"Source: {source}")
            
            # Find and highlight the context around the keyword
            lines = content.split('\n')
            matched_lines = []
            for line in lines:
                if keyword in line.lower():
                    matched_lines.append(line)
            
            if matched_lines:
                print(f"{Fore.YELLOW}--- Matching context ---{Style.RESET_ALL}")
                for line in matched_lines[:3]:  # Show up to 3 matching lines
                    # Highlight the keyword
                    index = line.lower().find(keyword)
                    if index >= 0:
                        before = line[:index]
                        match = line[index:index+len(keyword)]
                        after = line[index+len(keyword):]
                        print(f"{before}{Fore.RED}{match}{Style.RESET_ALL}{after}")
                    else:
                        print(line)
            else:
                # Show the first part of the content
                print(f"{content[:250]}..." if len(content) > 250 else content)
        
        # Show notice if there are more results
        if len(matching_docs) > args.count:
            print(f"\n{Fore.YELLOW}Showing {args.count} of {len(matching_docs)} matching documents. Run with --count option to see more.{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
