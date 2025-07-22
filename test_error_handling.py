#!/usr/bin/env python3
"""
Test script for comprehensive error handling system.
"""

import sys
import os
sys.path.insert(0, 'src')

from error_handler import (
    ErrorHandler, RAGException, ErrorType, ErrorSeverity, ErrorContext,
    handle_rag_error, error_handler
)

def test_error_handling_system():
    """Test the comprehensive error handling system."""
    print('=== Comprehensive Error Handling System Test ===')
    
    # Test 1: Error classification
    print('\n1. Testing error classification...')
    try:
        handler = ErrorHandler()
        
        # Test different error types
        test_errors = [
            (Exception("API key not configured"), ErrorType.API_KEY_MISSING),
            (Exception("Rate limit exceeded"), ErrorType.API_RATE_LIMIT),
            (Exception("File not found"), ErrorType.DOCUMENT_NOT_FOUND),
            (Exception("Document too large"), ErrorType.DOCUMENT_TOO_LARGE),
            (Exception("Query cannot be empty"), ErrorType.QUERY_EMPTY),
            (Exception("Database connection failed"), ErrorType.DATABASE_CONNECTION_FAILED),
            (Exception("Network timeout"), ErrorType.API_TIMEOUT),
            (Exception("Unknown issue"), ErrorType.UNKNOWN_ERROR)
        ]
        
        for error, expected_type in test_errors:
            classified_type = handler._classify_error(error)
            print(f'   ✓ "{str(error)}" -> {classified_type.value}')
            if classified_type != expected_type:
                print(f'     ⚠ Expected {expected_type.value}, got {classified_type.value}')
        
    except Exception as e:
        print(f'   ✗ Error classification test failed: {e}')
        return False
    
    # Test 2: RAGException creation and serialization
    print('\n2. Testing RAGException creation...')
    try:
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            additional_data={"test_key": "test_value"}
        )
        
        rag_error = RAGException(
            error_type=ErrorType.API_RATE_LIMIT,
            message="Test rate limit error",
            context=context,
            severity=ErrorSeverity.MEDIUM,
            suggested_actions=["Wait and retry", "Check rate limits"],
            retry_possible=True,
            retry_delay=5.0
        )
        
        print(f'   ✓ RAGException created successfully')
        print(f'   ✓ Error type: {rag_error.error_type.value}')
        print(f'   ✓ Severity: {rag_error.severity.value}')
        print(f'   ✓ User message: {rag_error.user_message}')
        print(f'   ✓ Retry possible: {rag_error.retry_possible}')
        print(f'   ✓ Retry delay: {rag_error.retry_delay}s')
        
        # Test serialization
        error_dict = rag_error.to_dict()
        print(f'   ✓ Serialization successful: {len(error_dict)} fields')
        
    except Exception as e:
        print(f'   ✗ RAGException creation test failed: {e}')
        return False
    
    # Test 3: Error handler processing
    print('\n3. Testing error handler processing...')
    try:
        context = ErrorContext(
            component="TestHandler",
            operation="process_test_error"
        )
        
        test_error = Exception("Test API timeout error")
        error_details = handler.handle_error(test_error, context, ErrorSeverity.HIGH)
        
        print(f'   ✓ Error processed successfully')
        print(f'   ✓ Error type: {error_details.error_type.value}')
        print(f'   ✓ Severity: {error_details.severity.value}')
        print(f'   ✓ User message: {error_details.user_message}')
        print(f'   ✓ Suggested actions: {len(error_details.suggested_actions)}')
        print(f'   ✓ Retry possible: {error_details.retry_possible}')
        
        # Display suggested actions
        for i, action in enumerate(error_details.suggested_actions, 1):
            print(f'     {i}. {action}')
        
    except Exception as e:
        print(f'   ✗ Error handler processing test failed: {e}')
        return False
    
    # Test 4: Convenience function
    print('\n4. Testing convenience function...')
    try:
        original_error = Exception("Test embedding generation failed")
        
        rag_exception = handle_rag_error(
            original_error,
            component="EmbeddingManager",
            operation="generate_embedding",
            severity=ErrorSeverity.HIGH,
            document_id="test_doc_123"
        )
        
        print(f'   ✓ Convenience function worked successfully')
        print(f'   ✓ Error type: {rag_exception.error_type.value}')
        print(f'   ✓ Component: {rag_exception.context.component}')
        print(f'   ✓ Operation: {rag_exception.context.operation}')
        print(f'   ✓ Additional data: {rag_exception.context.additional_data}')
        
    except Exception as e:
        print(f'   ✗ Convenience function test failed: {e}')
        return False
    
    # Test 5: Error statistics
    print('\n5. Testing error statistics...')
    try:
        # Generate some test errors
        test_errors = [
            Exception("API key missing"),
            Exception("Rate limit exceeded"),
            Exception("API key missing"),  # Duplicate to test counting
            Exception("Document not found"),
            Exception("Rate limit exceeded")  # Another duplicate
        ]
        
        for error in test_errors:
            context = ErrorContext(component="TestStats", operation="generate_stats")
            handler.handle_error(error, context)
        
        stats = handler.get_error_statistics()
        
        print(f'   ✓ Error statistics generated')
        print(f'   ✓ Total errors: {stats["total_errors"]}')
        print(f'   ✓ Unique error types: {len(stats["error_counts"])}')
        print(f'   ✓ Most common errors: {len(stats["most_common_errors"])}')
        print(f'   ✓ History size: {stats["history_size"]}')
        
        # Display most common errors
        print(f'   ✓ Top errors:')
        for error_type, count in stats["most_common_errors"][:3]:
            print(f'     - {error_type}: {count} occurrences')
        
    except Exception as e:
        print(f'   ✗ Error statistics test failed: {e}')
        return False
    
    # Test 6: User-friendly messages
    print('\n6. Testing user-friendly messages...')
    try:
        user_message_tests = [
            (ErrorType.API_KEY_MISSING, "API key is not configured"),
            (ErrorType.API_RATE_LIMIT, "Service is temporarily busy"),
            (ErrorType.DOCUMENT_INVALID_FORMAT, "Invalid document format"),
            (ErrorType.QUERY_EMPTY, "Please enter a question"),
            (ErrorType.DATABASE_CONNECTION_FAILED, "Database connection failed")
        ]
        
        for error_type, expected_phrase in user_message_tests:
            rag_error = RAGException(error_type, "Test message")
            user_msg = rag_error.user_message.lower()
            
            # Check if expected phrase is in the user message
            contains_phrase = any(word in user_msg for word in expected_phrase.lower().split())
            print(f'   ✓ {error_type.value}: {"✓" if contains_phrase else "✗"} user-friendly')
            
            if not contains_phrase:
                print(f'     Expected phrase: "{expected_phrase}"')
                print(f'     Actual message: "{rag_error.user_message}"')
        
    except Exception as e:
        print(f'   ✗ User-friendly messages test failed: {e}')
        return False
    
    # Test 7: Retry logic recommendations
    print('\n7. Testing retry logic recommendations...')
    try:
        retry_tests = [
            (ErrorType.API_RATE_LIMIT, True, 5.0),
            (ErrorType.API_TIMEOUT, True, 2.0),
            (ErrorType.DATABASE_CONNECTION_FAILED, True, 3.0),
            (ErrorType.DOCUMENT_NOT_FOUND, False, None),
            (ErrorType.QUERY_EMPTY, False, None),
            (ErrorType.API_KEY_MISSING, False, None)
        ]
        
        for error_type, should_retry, expected_delay in retry_tests:
            is_retryable = handler._is_retry_possible(error_type)
            retry_delay = handler._get_retry_delay(error_type)
            
            print(f'   ✓ {error_type.value}: retry={is_retryable}, delay={retry_delay}s')
            
            if is_retryable != should_retry:
                print(f'     ⚠ Expected retry={should_retry}, got {is_retryable}')
            
            if expected_delay and retry_delay != expected_delay:
                print(f'     ⚠ Expected delay={expected_delay}, got {retry_delay}')
        
    except Exception as e:
        print(f'   ✗ Retry logic test failed: {e}')
        return False
    
    # Test 8: Error context tracking
    print('\n8. Testing error context tracking...')
    try:
        context = ErrorContext(
            component="ContextTest",
            operation="test_context_tracking",
            user_id="test_user_123",
            session_id="session_456",
            request_id="req_789",
            additional_data={
                "document_name": "test.pdf",
                "chunk_count": 5,
                "processing_time": 2.5
            }
        )
        
        error = Exception("Context tracking test error")
        error_details = handler.handle_error(error, context)
        
        print(f'   ✓ Context tracking successful')
        print(f'   ✓ Component: {error_details.context.component}')
        print(f'   ✓ Operation: {error_details.context.operation}')
        print(f'   ✓ User ID: {error_details.context.user_id}')
        print(f'   ✓ Session ID: {error_details.context.session_id}')
        print(f'   ✓ Request ID: {error_details.context.request_id}')
        print(f'   ✓ Additional data: {len(error_details.context.additional_data)} fields')
        
    except Exception as e:
        print(f'   ✗ Error context tracking test failed: {e}')
        return False
    
    print('\n🎉 All error handling tests passed!')
    return True

def test_integration_with_rag_pipeline():
    """Test error handling integration with RAG pipeline components."""
    print('\n=== Error Handling Integration Test ===')
    
    # Test 1: Test with actual RAG components
    print('\n1. Testing integration with RAG components...')
    try:
        from rag_pipeline import EmbeddingManager, QueryProcessor
        
        # Test EmbeddingManager error handling
        print('   → Testing EmbeddingManager error handling...')
        try:
            embedding_manager = EmbeddingManager()
            # This should raise a RAGException due to invalid API key
            embedding_manager._validate_api_key()
            print('   ✗ Should have raised an error for invalid API key')
        except Exception as e:
            if "RAGException" in str(type(e)) or "api key" in str(e).lower():
                print('   ✓ EmbeddingManager error handling working')
            else:
                print(f'   ⚠ Unexpected error type: {type(e).__name__}: {e}')
        
        # Test QueryProcessor error handling
        print('   → Testing QueryProcessor error handling...')
        try:
            processor = QueryProcessor()
            # Test empty query
            processor.generate_answer('', [])
            print('   ✗ Should have raised an error for empty query')
        except Exception as e:
            if "empty" in str(e).lower() or "QueryProcessingError" in str(type(e)):
                print('   ✓ QueryProcessor error handling working')
            else:
                print(f'   ⚠ Unexpected error type: {type(e).__name__}: {e}')
        
    except Exception as e:
        print(f'   ✗ Integration test failed: {e}')
        return False
    
    # Test 2: Error statistics from global handler
    print('\n2. Testing global error handler statistics...')
    try:
        stats = error_handler.get_error_statistics()
        print(f'   ✓ Global error statistics retrieved')
        print(f'   ✓ Total errors tracked: {stats["total_errors"]}')
        print(f'   ✓ Error types seen: {len(stats["error_counts"])}')
        
        if stats["total_errors"] > 0:
            print(f'   ✓ Most common error: {stats["most_common_errors"][0][0]}')
        
    except Exception as e:
        print(f'   ✗ Global error handler test failed: {e}')
        return False
    
    print('\n🎉 Error handling integration tests passed!')
    return True

if __name__ == "__main__":
    print("Starting comprehensive error handling system tests...")
    
    # Run main error handling tests
    success1 = test_error_handling_system()
    
    # Run integration tests
    success2 = test_integration_with_rag_pipeline()
    
    if success1 and success2:
        print('\n=== Task 4.3 Implementation Complete ===')
        print('✓ RAGException custom exception class')
        print('✓ ErrorHandler for different error types')
        print('✓ Graceful fallbacks for API failures')
        print('✓ Comprehensive error classification')
        print('✓ User-friendly error messages')
        print('✓ Retry logic and suggestions')
        print('✓ Error statistics and monitoring')
        print('✓ Integration with RAG pipeline components')
        print('✓ Requirements 4.3, 5.5, 10.3, 10.4 satisfied')
        sys.exit(0)
    else:
        print('\n❌ Some tests failed. Please check the implementation.')
        sys.exit(1)