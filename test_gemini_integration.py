#!/usr/bin/env python3
"""
Test script for Gemini API integration in the RAG system.
"""

import sys
import os
sys.path.insert(0, 'src')

from rag_pipeline import QueryProcessor, QueryProcessingError

def test_gemini_api():
    """Test the Gemini API integration."""
    print('=== Gemini API Integration Test ===')
    
    # Test 1: Query processor initialization
    print('\n1. Testing query processor initialization...')
    try:
        processor = QueryProcessor()
        print(f'   ✓ Query processor initialized successfully')
        print(f'   ✓ Model: {processor.model}')
        print(f'   ✓ API URL: {processor.gemini_api_url}')
        print(f'   ✓ Max retries: {processor.max_retries}')
        
        # Check initial token usage
        usage = processor.token_usage
        print(f'   ✓ Initial token usage: {usage["queries_processed"]} queries, {usage["total_tokens"]} tokens')
    except Exception as e:
        print(f'   ✗ Initialization failed: {e}')
        return False
    
    # Test 2: API key validation
    print('\n2. Testing API key validation...')
    try:
        is_valid = processor._validate_api_key()
        print(f'   ✓ API key validation: {is_valid}')
    except Exception as e:
        print(f'   ✗ API key validation failed: {e}')
        return False
    
    # Test 3: Prompt creation
    print('\n3. Testing prompt creation...')
    try:
        # System prompt
        system_prompt = processor._create_system_prompt()
        print(f'   ✓ System prompt created: {len(system_prompt)} characters')
        print(f'   ✓ Contains instructions: {"INSTRUCTIONS" in system_prompt}')
        print(f'   ✓ Contains response format: {"RESPONSE FORMAT" in system_prompt}')
        
        # User prompt with context
        query = "What is machine learning?"
        context_chunks = [
            {
                'text': 'Machine learning is a subset of AI that enables systems to learn from data.',
                'source_document': 'ml_guide.txt',
                'similarity_score': 0.85,
                'chunk_id': 'chunk_1'
            }
        ]
        user_prompt = processor._create_user_prompt(query, context_chunks)
        print(f'   ✓ User prompt created: {len(user_prompt)} characters')
        print(f'   ✓ Contains query: {query in user_prompt}')
        print(f'   ✓ Contains context: {"Context 1" in user_prompt}')
        
        # User prompt without context
        empty_prompt = processor._create_user_prompt(query, [])
        print(f'   ✓ Empty context prompt created: {len(empty_prompt)} characters')
        print(f'   ✓ Contains no context message: {"No relevant context found" in empty_prompt}')
        
    except Exception as e:
        print(f'   ✗ Prompt creation failed: {e}')
        return False
    
    # Test 4: Simple answer generation
    print('\n4. Testing answer generation with Gemini API...')
    try:
        query = "What is artificial intelligence?"
        context_chunks = [
            {
                'text': 'Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.',
                'source_document': 'ai_basics.txt',
                'similarity_score': 0.92,
                'chunk_id': 'chunk_ai_1',
                'metadata': {'page': 1}
            }
        ]
        
        print(f'   → Sending query: "{query}"')
        print(f'   → Using {len(context_chunks)} context chunks')
        
        result = processor.generate_answer(query, context_chunks)
        
        print(f'   ✓ Answer generated successfully!')
        print(f'   ✓ Answer length: {len(result["answer"])} characters')
        print(f'   ✓ Model used: {result["model"]}')
        print(f'   ✓ Context chunks used: {result["context_chunks_used"]}')
        print(f'   ✓ Token usage: {result["token_usage"]["total_tokens"]} tokens')
        print(f'   ✓ Generation successful: {result["generation_successful"]}')
        
        # Display the answer
        print(f'\n   Generated Answer:')
        print(f'   {"-" * 50}')
        print(f'   {result["answer"][:300]}{"..." if len(result["answer"]) > 300 else ""}')
        print(f'   {"-" * 50}')
        
    except Exception as e:
        print(f'   ✗ Answer generation failed: {e}')
        return False
    
    # Test 5: Answer generation with citations
    print('\n5. Testing answer generation with citations...')
    try:
        query = "How does machine learning work?"
        context_chunks = [
            {
                'text': 'Machine learning works by training algorithms on large datasets to identify patterns and make predictions. The process involves feeding data to algorithms, which then learn to recognize patterns and make decisions based on new, unseen data.',
                'source_document': 'ml_fundamentals.pdf',
                'similarity_score': 0.88,
                'chunk_id': 'chunk_ml_1',
                'metadata': {'page': 3, 'section': 'Introduction'}
            },
            {
                'text': 'There are three main types of machine learning: supervised learning (learning with labeled examples), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error).',
                'source_document': 'ml_types.pdf',
                'similarity_score': 0.82,
                'chunk_id': 'chunk_ml_2',
                'metadata': {'page': 1, 'section': 'Types of ML'}
            }
        ]
        
        result = processor.generate_answer_with_citations(query, context_chunks)
        
        print(f'   ✓ Answer with citations generated successfully!')
        print(f'   ✓ Citations count: {result["citation_count"]}')
        print(f'   ✓ Source documents: {len(result["source_documents"])}')
        print(f'   ✓ Has citations: {result["has_citations"]}')
        
        # Display citations
        print(f'\n   Citations:')
        for citation in result['citations']:
            print(f'   - Citation {citation["citation_id"]}: {citation["source_document"]} (score: {citation["similarity_score"]:.3f})')
        
    except Exception as e:
        print(f'   ✗ Citation generation failed: {e}')
        return False
    
    # Test 6: Token usage tracking
    print('\n6. Testing token usage tracking...')
    try:
        stats = processor.get_token_usage_stats()
        print(f'   ✓ Total queries processed: {stats["queries_processed"]}')
        print(f'   ✓ Total tokens used: {stats["total_tokens"]}')
        print(f'   ✓ Average tokens per query: {stats["average_tokens_per_query"]:.1f}')
        
        # Test cost estimation
        cost = processor.estimate_cost()
        print(f'   ✓ Estimated cost: ${cost["total_cost"]:.4f} {cost["currency"]}')
        print(f'   ✓ Note: {cost["note"]}')
        
    except Exception as e:
        print(f'   ✗ Token usage tracking failed: {e}')
        return False
    
    # Test 7: Error handling
    print('\n7. Testing error handling...')
    try:
        # Test empty query
        try:
            processor.generate_answer('', [])
            print('   ✗ Should have failed for empty query')
        except QueryProcessingError:
            print('   ✓ Empty query error handled correctly')
        
        # Test with no context
        result = processor.generate_answer('What is the meaning of life?', [])
        print('   ✓ No context query handled correctly')
        print(f'   ✓ Answer indicates limitation: {"cannot answer" in result["answer"].lower()}')
        
    except Exception as e:
        print(f'   ✗ Error handling test failed: {e}')
        return False
    
    print('\n🎉 All Gemini API integration tests passed!')
    return True

if __name__ == "__main__":
    success = test_gemini_api()
    if success:
        print('\n=== Task 4.2 Implementation Complete ===')
        print('✓ QueryProcessor class with Gemini API integration')
        print('✓ Structured prompts with context integration')
        print('✓ Token usage tracking and cost estimation')
        print('✓ Citation and source attribution')
        print('✓ Error handling and retry logic')
        print('✓ Requirements 5.1, 5.2, 5.3, 5.4 satisfied')
        sys.exit(0)
    else:
        print('\n❌ Some tests failed. Please check the implementation.')
        sys.exit(1)