import json
import argparse
from typing import List, Dict, Any

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the JSON data from the file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"Successfully loaded data with {len(data)} entries.")
            return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def search_by_keyword(data: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
    """Search for entries containing the keyword in title or content."""
    keyword = keyword.lower()
    results = []
    
    for entry in data:
        if keyword in entry.get('title', '').lower() or keyword in entry.get('content', '').lower():
            results.append(entry)
    
    return results

def display_entry_summary(entry: Dict[str, Any], show_content: bool = False) -> None:
    """Display a summary of an entry."""
    print(f"\nTitle: {entry.get('title', 'No title')}")
    print(f"URL: {entry.get('url', 'No URL')}")
    
    if show_content:
        content = entry.get('content', 'No content')
        # Limit content to first 200 characters for summary
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"Content preview: {content}")

def get_unique_pages(data: List[Dict[str, Any]]) -> List[str]:
    """Get a list of unique page titles."""
    return sorted(list(set(entry.get('title', '') for entry in data)))

def main():
    parser = argparse.ArgumentParser(description='Search Contisoft scraped data')
    parser.add_argument('--file', default='scraped_contisoft.json', help='Path to the JSON file')
    parser.add_argument('--search', help='Keyword to search for')
    parser.add_argument('--list-pages', action='store_true', help='List all unique page titles')
    parser.add_argument('--show-content', action='store_true', help='Show content preview in search results')
    
    args = parser.parse_args()
    
    data = load_data(args.file)
    if not data:
        return
    
    if args.list_pages:
        print("\nUnique page titles:")
        for title in get_unique_pages(data):
            print(f"- {title}")
    
    if args.search:
        results = search_by_keyword(data, args.search)
        print(f"\nFound {len(results)} results for '{args.search}':")
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            display_entry_summary(result, args.show_content)
    
    if not args.list_pages and not args.search:
        print("\nUse --search to search for keywords or --list-pages to list all unique pages.")
        print("Example: python search_contisoft.py --search 'contract management' --show-content")

if __name__ == "__main__":
    main() 