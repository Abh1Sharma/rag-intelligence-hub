"""
Logging configuration for the RAG Dashboard System.
Provides structured logging with performance tracking capabilities.
"""

import logging
import time
from functools import wraps
from typing import Any, Callable
from config import config

def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Set level
        logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    
    return logger

def log_performance(operation_name: str):
    """Decorator to log performance metrics for operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = setup_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{operation_name} completed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{operation_name} failed after {execution_time:.2f} seconds: {str(e)}")
                raise
        
        return wrapper
    return decorator

# Global logger instance
logger = setup_logger(__name__)