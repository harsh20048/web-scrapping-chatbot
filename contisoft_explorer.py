import streamlit as st
import json
import pandas as pd
from typing import Dict, List, Any

def load_data(file_path: str = 'scraped_contisoft.json') -> List[Dict[str, Any]]:
    """Load the JSON data from the file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []

def search_data(data: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
    """Search for entries containing the search term in title or content."""
    if not search_term:
        return data
    
    search_term = search_term.lower()
    results = []
    
    for entry in data:
        title = entry.get('title', '').lower()
        content = entry.get('content', '').lower()
        
        if search_term in title or search_term in content:
            results.append(entry)
    
    return results

def main():
    st.set_page_config(
        page_title="Contisoft Data Explorer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Contisoft Technologies - Web Data Explorer")
    st.write("Explore and search through the scraped data from Contisoft Technologies website.")
    
    # Load data
    data = load_data()
    
    if not data:
        st.warning("No data loaded. Please make sure 'scraped_contisoft.json' exists in the current directory.")
        return
    
    # Display data info
    st.sidebar.header("Data Overview")
    st.sidebar.info(f"Total entries: {len(data)}")
    
    # Search functionality
    st.sidebar.header("Search")
    search_term = st.sidebar.text_input("Enter search term:")
    
    # Filter data
    filtered_data = search_data(data, search_term)
    
    st.sidebar.success(f"Found {len(filtered_data)} results")
    
    # Display options
    st.sidebar.header("Display Options")
    show_urls = st.sidebar.checkbox("Show URLs", value=True)
    show_content = st.sidebar.checkbox("Show Content", value=False)
    
    # Display data in tabs
    tab1, tab2 = st.tabs(["List View", "Detail View"])
    
    with tab1:
        # Create a DataFrame for easier display
        df_data = []
        for i, entry in enumerate(filtered_data):
            df_data.append({
                "ID": i+1,
                "Title": entry.get('title', 'No title'),
                "URL": entry.get('url', 'No URL') if show_urls else "Hidden"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        # Select an entry to view details
        if filtered_data:
            selected_index = st.selectbox(
                "Select an entry to view details:",
                range(len(filtered_data)),
                format_func=lambda i: filtered_data[i].get('title', f'Entry {i+1}')
            )
            
            selected_entry = filtered_data[selected_index]
            
            st.subheader(selected_entry.get('title', 'No Title'))
            
            if show_urls:
                st.write(f"**URL:** {selected_entry.get('url', 'No URL')}")
            
            if show_content:
                st.write("### Content:")
                st.write(selected_entry.get('content', 'No content'))
            else:
                st.info("Enable 'Show Content' in the sidebar to view the content.")
        else:
            st.info("No entries found. Try a different search term.")

if __name__ == "__main__":
    main() 