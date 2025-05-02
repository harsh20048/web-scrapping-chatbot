"""
Script to check and fix Ollama model issues.
"""

import subprocess
import requests
import time
import sys
import os

def run_command(command):
    """Run a command and return the output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{command}': {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_ollama_running():
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Ollama is not running: {e}")
        return False

def start_ollama():
    """Start Ollama."""
    if sys.platform == "win32":
        print("Starting Ollama on Windows...")
        # On Windows, just provide instructions
        print("Please start Ollama by double-clicking on the Ollama icon or running it from the Start menu.")
        return False
    else:
        print("Starting Ollama...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Wait for Ollama to start
            for _ in range(10):
                time.sleep(1)
                if check_ollama_running():
                    print("Ollama started successfully.")
                    return True
            print("Failed to start Ollama after waiting.")
            return False
        except Exception as e:
            print(f"Error starting Ollama: {e}")
            return False

def check_model_installed(model_name):
    """Check if the specified model is installed."""
    if not check_ollama_running():
        if not start_ollama():
            print("Ollama is not running and could not be started.")
            return False
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            installed_models = [model["name"] for model in models]
            return model_name in installed_models
        return False
    except Exception as e:
        print(f"Error checking if model is installed: {e}")
        return False

def pull_model(model_name):
    """Pull a model."""
    print(f"Downloading model: {model_name}")
    try:
        result = run_command(f"ollama pull {model_name}")
        if result:
            print(f"Successfully downloaded model: {model_name}")
            return True
        return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def fix_ollama_issues():
    """Check and fix Ollama issues."""
    print("Checking Ollama installation...")
    
    # Check if Ollama is installed
    ollama_version = run_command("ollama --version")
    if not ollama_version:
        print("Ollama is not installed or not in the PATH.")
        print("Please install Ollama from: https://ollama.ai/download")
        return False
    
    print(f"Ollama version: {ollama_version}")
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("Ollama is not running.")
        if not start_ollama():
            print("Please start Ollama manually before continuing.")
            return False
    
    # Models to check
    models = [
        "phi:latest",
        "tinyllama:latest", 
        "mistral:7b"
    ]
    
    missing_models = []
    
    # Check if models are installed
    for model in models:
        print(f"Checking if model {model} is installed...")
        if check_model_installed(model):
            print(f"Model {model} is already installed.")
        else:
            print(f"Model {model} is not installed.")
            missing_models.append(model)
    
    # Install missing models
    if missing_models:
        print(f"Need to download {len(missing_models)} models: {', '.join(missing_models)}")
        for model in missing_models:
            if pull_model(model):
                print(f"Model {model} downloaded successfully.")
            else:
                print(f"Failed to download model {model}.")
    else:
        print("All required models are installed.")
    
    # Final check
    if check_ollama_running():
        print("Ollama is running properly.")
        return True
    else:
        print("Ollama is still not running properly.")
        return False

if __name__ == "__main__":
    print("=== Ollama Model Checker and Fixer ===")
    success = fix_ollama_issues()
    if success:
        print("\nAll Ollama issues have been fixed!")
    else:
        print("\nSome Ollama issues could not be fixed automatically.")
        print("Please check the errors above and fix them manually.") 