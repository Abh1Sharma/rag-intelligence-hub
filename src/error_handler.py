"""
Comprehensive error handling system for the RAG Dashboard System.
Provides centralized error management, logging, and user-friendly error messages.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ErrorType(Enum):
    """Enumeration of error types in the RAG system."""
    
    # API-related errors
    API_KEY_MISSING = "api_key_missing"
    API_KEY_INVALID = "api_key_invalid"
    API_RATE_LIMIT = "api_rate_limit"
    API_QUOTA_EXCEEDED = "api_quota_exceeded"
    API_SERVICE_UNAVAILABLE = "api_service_unavailable"
    API_TIMEOUT = "api_timeout"
    API_INVALID_RESPONSE = "api_invalid_response"
    
    # Document processing errors
    DOCUMENT_NOT_FOUND = "document_not_found"
    DOCUMENT_INVALID_FORMAT = "document_invalid_format"
    DOCUMENT_TOO_LARGE = "document_too_large"
    DOCUMENT_CORRUPTED = "document_corrupted"
    DOCUMENT_EMPTY = "document_empty"
    
    # Embedding errors
    EMBEDDING_GENERATION_FAILED = "embedding_generation_failed"
    EMBEDDING_INVALID_DIMENSIONS = "embedding_invalid_dimensions"
    EMBEDDING_BATCH_FAILED = "embedding_batch_failed"
    
    # Vector database errors
    DATABASE_CONNECTION_FAILED = "database_connection_failed"
    DATABASE_STORAGE_FAILED = "database_storage_failed"
    DATABASE_QUERY_FAILED = "database_query_failed"
    DATABASE_NOT_INITIALIZED = "database_not_initialized"
    
    # Query processing errors
    QUERY_EMPTY = "query_empty"
    QUERY_TOO_LONG = "query_too_long"
    QUERY_INVALID_FORMAT = "query_invalid_format"
    QUERY_NO_CONTEXT = "query_no_context"
    
    # System errors
    SYSTEM_OUT_OF_MEMORY = "system_out_of_memory"
    SYSTEM_DISK_FULL = "system_disk_full"
    SYSTEM_NETWORK_ERROR = "system_network_error"
    SYSTEM_CONFIGURATION_ERROR = "system_configuration_error"
    
    # User input errors
    USER_INPUT_INVALID = "user_input_invalid"
    USER_INPUT_MISSING = "user_input_missing"
    USER_PERMISSION_DENIED = "user_permission_denied"
    
    # Unknown errors
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for errors."""
    component: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorDetails:
    """Detailed error information."""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    user_message: str
    context: ErrorContext
    original_exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    suggested_actions: List[str] = field(default_factory=list)
    retry_possible: bool = False
    retry_delay: Optional[float] = None


class RAGException(Exception):
    """Base exception class for all RAG system errors."""
    
    def __init__(
        self,
        error_type: ErrorType,
        message: str,
        user_message: str = None,
        context: ErrorContext = None,
        original_exception: Exception = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        suggested_actions: List[str] = None,
        retry_possible: bool = False,
        retry_delay: float = None
    ):
        self.error_type = error_type
        self.message = message
        self.user_message = user_message or self._generate_user_message(error_type)
        self.context = context
        self.original_exception = original_exception
        self.severity = severity
        self.suggested_actions = suggested_actions or []
        self.retry_possible = retry_possible
        self.retry_delay = retry_delay
        self.timestamp = datetime.now()
        
        super().__init__(self.message)
    
    def _generate_user_message(self, error_type: ErrorType) -> str:
        """Generate user-friendly error messages."""
        user_messages = {
            ErrorType.API_KEY_MISSING: "API key is not configured. Please check your configuration.",
            ErrorType.API_KEY_INVALID: "API key is invalid. Please verify your credentials.",
            ErrorType.API_RATE_LIMIT: "Service is temporarily busy. Please try again in a moment.",
            ErrorType.API_QUOTA_EXCEEDED: "API quota exceeded. Please check your usage limits.",
            ErrorType.API_SERVICE_UNAVAILABLE: "Service is temporarily unavailable. Please try again later.",
            ErrorType.API_TIMEOUT: "Request timed out. Please try again.",
            ErrorType.DOCUMENT_NOT_FOUND: "Document not found. Please check the file path.",
            ErrorType.DOCUMENT_INVALID_FORMAT: "Invalid document format. Please upload a PDF or TXT file.",
            ErrorType.DOCUMENT_TOO_LARGE: "Document is too large. Please upload a smaller file.",
            ErrorType.DOCUMENT_CORRUPTED: "Document appears to be corrupted. Please try a different file.",
            ErrorType.DOCUMENT_EMPTY: "Document is empty. Please upload a file with content.",
            ErrorType.DATABASE_CONNECTION_FAILED: "Database connection failed. Please try again.",
            ErrorType.DATABASE_STORAGE_FAILED: "Failed to save data. Please try again.",
            ErrorType.QUERY_EMPTY: "Please enter a question.",
            ErrorType.QUERY_TOO_LONG: "Question is too long. Please shorten your query.",
            ErrorType.QUERY_NO_CONTEXT: "No relevant information found. Please upload relevant documents.",
            ErrorType.SYSTEM_NETWORK_ERROR: "Network error occurred. Please check your connection.",
            ErrorType.USER_INPUT_INVALID: "Invalid input. Please check your data and try again.",
            ErrorType.UNKNOWN_ERROR: "An unexpected error occurred. Please try again."
        }
        return user_messages.get(error_type, "An error occurred. Please try again.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'error_type': self.error_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'user_message': self.user_message,
            'timestamp': self.timestamp.isoformat(),
            'context': {
                'component': self.context.component if self.context else None,
                'operation': self.context.operation if self.context else None,
                'additional_data': self.context.additional_data if self.context else {}
            },
            'suggested_actions': self.suggested_actions,
            'retry_possible': self.retry_possible,
            'retry_delay': self.retry_delay
        }


class ErrorHandler:
    """Centralized error handler for the RAG system."""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_history_size = 1000
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> ErrorDetails:
        """
        Handle and process errors with comprehensive logging and analysis.
        
        Args:
            error: The original exception
            context: Error context information
            severity: Error severity level
            
        Returns:
            ErrorDetails object with processed error information
        """
        # Determine error type
        error_type = self._classify_error(error)
        
        # Generate error details
        error_details = ErrorDetails(
            error_type=error_type,
            severity=severity,
            message=str(error),
            user_message=self._generate_user_message(error_type, error),
            context=context,
            original_exception=error,
            stack_trace=traceback.format_exc(),
            suggested_actions=self._get_suggested_actions(error_type),
            retry_possible=self._is_retry_possible(error_type),
            retry_delay=self._get_retry_delay(error_type)
        )
        
        # Log the error
        self._log_error(error_details)
        
        # Update error statistics
        self._update_error_stats(error_type)
        
        # Add to error history
        self._add_to_history(error_details)
        
        return error_details
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error based on exception type and message."""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # API-related errors
        if "api key" in error_str:
            if "not configured" in error_str or "missing" in error_str:
                return ErrorType.API_KEY_MISSING
            elif "invalid" in error_str or "unauthorized" in error_str:
                return ErrorType.API_KEY_INVALID
        
        if "rate limit" in error_str or "429" in error_str:
            return ErrorType.API_RATE_LIMIT
        
        if "quota" in error_str or "exceeded" in error_str:
            return ErrorType.API_QUOTA_EXCEEDED
        
        if "timeout" in error_str or "timed out" in error_str:
            return ErrorType.API_TIMEOUT
        
        if "service unavailable" in error_str or "503" in error_str:
            return ErrorType.API_SERVICE_UNAVAILABLE
        
        # Document processing errors
        if "file not found" in error_str or "no such file" in error_str:
            return ErrorType.DOCUMENT_NOT_FOUND
        
        if "invalid format" in error_str or "unsupported" in error_str:
            return ErrorType.DOCUMENT_INVALID_FORMAT
        
        if "too large" in error_str or "file size" in error_str:
            return ErrorType.DOCUMENT_TOO_LARGE
        
        if "corrupted" in error_str or "damaged" in error_str:
            return ErrorType.DOCUMENT_CORRUPTED
        
        if "empty" in error_str and ("document" in error_str or "file" in error_str):
            return ErrorType.DOCUMENT_EMPTY
        
        # Database errors
        if "database" in error_str or "chroma" in error_str:
            if "connection" in error_str:
                return ErrorType.DATABASE_CONNECTION_FAILED
            elif "storage" in error_str or "store" in error_str:
                return ErrorType.DATABASE_STORAGE_FAILED
            elif "query" in error_str:
                return ErrorType.DATABASE_QUERY_FAILED
            else:
                return ErrorType.DATABASE_NOT_INITIALIZED
        
        # Query processing errors
        if "query" in error_str:
            if "empty" in error_str or "cannot be empty" in error_str:
                return ErrorType.QUERY_EMPTY
            elif "too long" in error_str:
                return ErrorType.QUERY_TOO_LONG
            elif "no context" in error_str or "no relevant" in error_str:
                return ErrorType.QUERY_NO_CONTEXT
        
        # Embedding errors
        if "embedding" in error_str:
            if "generation" in error_str or "failed" in error_str:
                return ErrorType.EMBEDDING_GENERATION_FAILED
            elif "dimensions" in error_str:
                return ErrorType.EMBEDDING_INVALID_DIMENSIONS
            elif "batch" in error_str:
                return ErrorType.EMBEDDING_BATCH_FAILED
        
        # Network errors
        if "network" in error_str or "connection" in error_str:
            return ErrorType.SYSTEM_NETWORK_ERROR
        
        # Memory errors
        if "memory" in error_str or "memoryerror" in error_type_name:
            return ErrorType.SYSTEM_OUT_OF_MEMORY
        
        # Default to unknown error
        return ErrorType.UNKNOWN_ERROR
    
    def _generate_user_message(self, error_type: ErrorType, original_error: Exception) -> str:
        """Generate user-friendly error message."""
        base_messages = {
            ErrorType.API_KEY_MISSING: "API configuration issue. Please contact support.",
            ErrorType.API_RATE_LIMIT: "Service is busy. Please wait a moment and try again.",
            ErrorType.DOCUMENT_INVALID_FORMAT: "Please upload a valid PDF or TXT file.",
            ErrorType.DOCUMENT_TOO_LARGE: "File is too large. Please upload a smaller file (max 10MB).",
            ErrorType.QUERY_EMPTY: "Please enter a question about your documents.",
            ErrorType.QUERY_NO_CONTEXT: "No relevant information found. Try uploading more documents or rephrasing your question.",
            ErrorType.DATABASE_CONNECTION_FAILED: "System temporarily unavailable. Please try again.",
            ErrorType.SYSTEM_NETWORK_ERROR: "Network connection issue. Please check your internet connection.",
            ErrorType.UNKNOWN_ERROR: "An unexpected error occurred. Please try again or contact support."
        }
        
        return base_messages.get(error_type, "An error occurred. Please try again.")
    
    def _get_suggested_actions(self, error_type: ErrorType) -> List[str]:
        """Get suggested actions for error recovery."""
        suggestions = {
            ErrorType.API_KEY_MISSING: [
                "Check your API key configuration",
                "Verify environment variables are set",
                "Contact administrator for API access"
            ],
            ErrorType.API_RATE_LIMIT: [
                "Wait a few moments before retrying",
                "Reduce the frequency of requests",
                "Consider upgrading your API plan"
            ],
            ErrorType.DOCUMENT_INVALID_FORMAT: [
                "Upload a PDF or TXT file",
                "Check that the file is not corrupted",
                "Try converting the document to a supported format"
            ],
            ErrorType.DOCUMENT_TOO_LARGE: [
                "Split the document into smaller parts",
                "Compress the file if possible",
                "Remove unnecessary content"
            ],
            ErrorType.QUERY_EMPTY: [
                "Enter a question about your documents",
                "Be specific about what you want to know",
                "Try example questions provided"
            ],
            ErrorType.QUERY_NO_CONTEXT: [
                "Upload relevant documents first",
                "Try rephrasing your question",
                "Use more specific keywords"
            ],
            ErrorType.DATABASE_CONNECTION_FAILED: [
                "Try again in a few moments",
                "Check your internet connection",
                "Contact support if the issue persists"
            ]
        }
        
        return suggestions.get(error_type, ["Try again", "Contact support if the issue persists"])
    
    def _is_retry_possible(self, error_type: ErrorType) -> bool:
        """Determine if the error is retryable."""
        retryable_errors = {
            ErrorType.API_RATE_LIMIT,
            ErrorType.API_TIMEOUT,
            ErrorType.API_SERVICE_UNAVAILABLE,
            ErrorType.DATABASE_CONNECTION_FAILED,
            ErrorType.SYSTEM_NETWORK_ERROR,
            ErrorType.EMBEDDING_GENERATION_FAILED
        }
        return error_type in retryable_errors
    
    def _get_retry_delay(self, error_type: ErrorType) -> Optional[float]:
        """Get recommended retry delay for retryable errors."""
        retry_delays = {
            ErrorType.API_RATE_LIMIT: 5.0,
            ErrorType.API_TIMEOUT: 2.0,
            ErrorType.API_SERVICE_UNAVAILABLE: 10.0,
            ErrorType.DATABASE_CONNECTION_FAILED: 3.0,
            ErrorType.SYSTEM_NETWORK_ERROR: 5.0,
            ErrorType.EMBEDDING_GENERATION_FAILED: 2.0
        }
        return retry_delays.get(error_type)
    
    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level based on severity."""
        log_message = f"[{error_details.error_type.value}] {error_details.message}"
        
        if error_details.context:
            log_message += f" | Component: {error_details.context.component}"
            log_message += f" | Operation: {error_details.context.operation}"
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, exc_info=error_details.original_exception)
        elif error_details.severity == ErrorSeverity.HIGH:
            logger.error(log_message, exc_info=error_details.original_exception)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _update_error_stats(self, error_type: ErrorType):
        """Update error statistics."""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
    
    def _add_to_history(self, error_details: ErrorDetails):
        """Add error to history with size limit."""
        self.error_history.append(error_details)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and analytics."""
        total_errors = sum(self.error_counts.values())
        
        return {
            'total_errors': total_errors,
            'error_counts': {error_type.value: count for error_type, count in self.error_counts.items()},
            'most_common_errors': sorted(
                [(error_type.value, count) for error_type, count in self.error_counts.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'recent_errors': len([e for e in self.error_history if (datetime.now() - e.context.timestamp).seconds < 3600]),
            'history_size': len(self.error_history)
        }
    
    def clear_history(self):
        """Clear error history and statistics."""
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("Error history and statistics cleared")


# Global error handler instance
error_handler = ErrorHandler()


def handle_rag_error(
    error: Exception,
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    **context_data
) -> RAGException:
    """
    Convenience function to handle RAG system errors.
    
    Args:
        error: Original exception
        component: Component where error occurred
        operation: Operation being performed
        severity: Error severity level
        **context_data: Additional context data
        
    Returns:
        RAGException with processed error information
    """
    context = ErrorContext(
        component=component,
        operation=operation,
        additional_data=context_data
    )
    
    error_details = error_handler.handle_error(error, context, severity)
    
    return RAGException(
        error_type=error_details.error_type,
        message=error_details.message,
        user_message=error_details.user_message,
        context=context,
        original_exception=error,
        severity=severity,
        suggested_actions=error_details.suggested_actions,
        retry_possible=error_details.retry_possible,
        retry_delay=error_details.retry_delay
    )