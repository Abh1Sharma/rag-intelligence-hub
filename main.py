#!/usr/bin/env python3
"""
Main entry point for the RAG Dashboard System.
Provides options to run the FastAPI backend, Streamlit frontend, or both.
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def run_api_server():
    """Run the FastAPI backend server."""
    print("🚀 Starting FastAPI backend server...")
    from src.api import run_development_server
    run_development_server()

def run_streamlit_app():
    """Run the Streamlit frontend application."""
    print("🎨 Starting Streamlit frontend application...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

def run_both():
    """Run both FastAPI backend and Streamlit frontend."""
    import threading
    import time
    
    print("🚀 Starting both FastAPI backend and Streamlit frontend...")
    
    # Start FastAPI in a separate thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Wait a moment for API to start
    time.sleep(3)
    
    # Start Streamlit in main thread
    run_streamlit_app()

def setup_environment():
    """Set up the development environment."""
    print("🔧 Setting up RAG Dashboard System environment...")
    
    # Create necessary directories
    directories = [
        "uploads",
        "temp", 
        "logs",
        "chroma_db",
        "demo"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✓ Created directory: {directory}")
    
    # Create demo document if it doesn't exist
    demo_doc_path = Path("demo/sample_document.txt")
    if not demo_doc_path.exists():
        demo_content = """# Sample Document for RAG Dashboard System

## Introduction
This is a sample document for demonstrating the RAG (Retrieval-Augmented Generation) Dashboard System. The system allows users to upload documents and ask questions about their content using AI-powered natural language processing.

## Key Features
- **Document Upload**: Support for PDF and TXT files
- **AI-Powered Q&A**: Ask questions about uploaded documents
- **Vector Search**: Efficient similarity search using embeddings
- **Modern Interface**: Clean, intuitive web interface
- **Real-time Processing**: Fast document processing and query responses

## How It Works
1. Upload your documents (PDF or TXT format)
2. The system processes and indexes the content
3. Ask questions in natural language
4. Get AI-generated answers based on your documents

## Technology Stack
- **Backend**: FastAPI with Python
- **Frontend**: Streamlit
- **AI Models**: Google Gemini for text generation, OpenAI for embeddings
- **Vector Database**: ChromaDB for similarity search
- **Document Processing**: PyPDF2 for PDF extraction

## Use Cases
- Research and document analysis
- Knowledge base querying
- Content summarization
- Information extraction from large documents

This system demonstrates the power of combining modern AI technologies with user-friendly interfaces to create practical document analysis tools.
"""
        demo_doc_path.write_text(demo_content)
        print(f"   ✓ Created demo document: {demo_doc_path}")
    
    # Check environment file
    env_file = Path(".env")
    if not env_file.exists():
        print("   ⚠ .env file not found. Please create one with your API keys.")
        print("   📝 Example .env content:")
        print("      OPENAI_API_KEY=your_openai_key_here")
        print("      GEMINI_API_KEY=your_gemini_key_here")
    else:
        print("   ✓ .env file found")
    
    print("✅ Environment setup complete!")

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="RAG Dashboard System")
    parser.add_argument(
        "mode",
        choices=["api", "streamlit", "both", "setup"],
        help="Run mode: 'api' for FastAPI backend, 'streamlit' for frontend, 'both' for both, 'setup' for environment setup"
    )
    
    args = parser.parse_args()
    
    if args.mode == "setup":
        setup_environment()
    elif args.mode == "api":
        run_api_server()
    elif args.mode == "streamlit":
        run_streamlit_app()
    elif args.mode == "both":
        run_both()

if __name__ == "__main__":
    main()