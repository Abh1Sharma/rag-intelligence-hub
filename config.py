"""
Configuration management for the RAG Dashboard System.
Centralizes all configuration settings with environment variable support.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Central configuration class for the RAG system."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
    
    # Chroma Database Configuration
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
    
    # Document Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB in bytes
    SUPPORTED_FILE_TYPES: list = [".pdf", ".txt"]
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("PORT", "8000"))
    
    # Streamlit Configuration
    STREAMLIT_HOST: str = os.getenv("STREAMLIT_HOST", "localhost")
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # File Storage Configuration
    UPLOAD_DIRECTORY: str = os.getenv("UPLOAD_DIRECTORY", "./uploads")
    TEMP_DIRECTORY: str = os.getenv("TEMP_DIRECTORY", "./temp")
    
    # Retrieval Configuration
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Demo Mode Configuration
    DEMO_DOCUMENT_PATH: str = os.getenv("DEMO_DOCUMENT_PATH", "./demo/sample_document.txt")
    DEMO_QUESTIONS: list = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the most important findings mentioned?"
    ]
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/rag_system.log")
    
    # UI Configuration
    APP_TITLE: str = "RAG Dashboard System"
    APP_DESCRIPTION: str = "Upload documents and ask questions with AI-powered answers"
    THEME_PRIMARY_COLOR: str = "#1f77b4"
    THEME_BACKGROUND_COLOR: str = "#ffffff"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Create necessary directories
        os.makedirs(cls.UPLOAD_DIRECTORY, exist_ok=True)
        os.makedirs(cls.TEMP_DIRECTORY, exist_ok=True)
        os.makedirs(cls.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)
        
        return True

# Global configuration instance
config = Config()

# Validate configuration on import
config.validate_config()