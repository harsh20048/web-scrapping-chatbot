"""
Script to completely reset the ChromaDB database by deleting its files.
"""

import os
import shutil
import sys

def reset_database():
    """Completely reset the ChromaDB database by deleting its files."""
    # Path to the ChromaDB directory
    db_path = './data/chroma'
    
    print(f"Checking if ChromaDB directory exists: {db_path}")
    if os.path.exists(db_path):
        print(f"ChromaDB directory exists, removing: {db_path}")
        try:
            shutil.rmtree(db_path)
            print(f"Successfully removed ChromaDB directory: {db_path}")
        except Exception as e:
            print(f"Error removing ChromaDB directory: {e}")
            print("Please make sure no processes are using the database.")
            print("You may need to stop your Flask application first.")
            return False
    else:
        print(f"ChromaDB directory does not exist: {db_path}")
    
    # Create a new directory
    try:
        print(f"Creating new ChromaDB directory: {db_path}")
        os.makedirs(db_path, exist_ok=True)
        print(f"Successfully created ChromaDB directory: {db_path}")
        return True
    except Exception as e:
        print(f"Error creating ChromaDB directory: {e}")
        return False

if __name__ == "__main__":
    print("=== ChromaDB Database Reset Tool ===")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        print("Force reset mode enabled.")
    else:
        confirm = input("This will delete ALL your existing vector database data. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Reset cancelled.")
            sys.exit(0)
    
    success = reset_database()
    
    if success:
        print("\nDatabase has been completely reset!")
        print("You can now run your application to create a new database.")
    else:
        print("\nDatabase reset failed.")
        print("You may need to manually delete the ChromaDB directory:")
        print("  ./data/chroma") 