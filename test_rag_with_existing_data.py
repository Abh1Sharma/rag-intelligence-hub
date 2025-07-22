#!/usr/bin/env python3
"""
Test script for RAG pipeline using existing embedded data.
"""

import sys
import os
sys.path.insert(0, 'src')

from rag_pipeline import QueryProcessor, ChromaVectorDatabase

def test_with_existing_data():
    """Test the RAG pipeline with existing embedded data."""
    print('=== RAG Test with Existing Data ===')
    
    # Test 1: Check existing data in vector database
    print('\n1. Checking existing data in vector database...')
    try:
        db = ChromaVectorDatabase()
        stats = db.get_collection_stats()
        
        print(f'   ✓ Vector database initialized')
        print(f'   ✓ Total chunks in database: {stats["total_chunks"]}')
        print(f'   ✓ Collection name: {stats["collection_name"]}')
        
        if stats["total_chunks"] > 0:
            print(f'   ✓ Documents represented: {stats.get("documents_represented", "N/A")}')
            if "document_distribution" in stats:
                print(f'   ✓ Document distribution:')
                for doc, count in stats["document_distribution"].items():
                    print(f'     - {doc}: {count} chunks')
        else:
            print('   ⚠ No existing data found in vector database')
            return False
            
    except Exception as e:
        print(f'   ✗ Database check failed: {e}')
        return False
    
    # Test 2: Initialize query processor
    print('\n2. Testing query processor with Gemini...')
    try:
        processor = QueryProcessor()
        print(f'   ✓ Query processor initialized')
        print(f'   ✓ Model: {processor.model}')
        print(f'   ✓ API key configured: {bool(processor.gemini_api_key)}')
        
    except Exception as e:
        print(f'   ✗ Query processor initialization failed: {e}')
        return False
    
    # Test 3: Manual similarity search (bypassing embedding generation)
    print('\n3. Testing manual similarity search with existing data...')
    try:
        # Get a sample chunk to use its embedding for similarity search
        sample_results = db.collection.get(limit=1, include=['embeddings', 'documents', 'metadatas'])
        
        if sample_results['ids'] and sample_results['embeddings']:
            sample_embedding = sample_results['embeddings'][0]
            sample_text = sample_results['documents'][0]
            sample_metadata = sample_results['metadatas'][0]
            
            print(f'   ✓ Found sample chunk: "{sample_text[:100]}..."')
            print(f'   ✓ Sample embedding dimensions: {len(sample_embedding)}')
            print(f'   ✓ Sample metadata: {sample_metadata.get("source_document", "Unknown")}')
            
            # Perform similarity search using the sample embedding
            similar_chunks = db.similarity_search(sample_embedding, top_k=3)
            
            print(f'   ✓ Similarity search completed')
            print(f'   ✓ Found {len(similar_chunks)} similar chunks')
            
            for i, chunk in enumerate(similar_chunks[:2]):
                print(f'     - Chunk {i+1}: {chunk["similarity_score"]:.3f} similarity')
                print(f'       Source: {chunk["metadata"].get("source_document", "Unknown")}')
                print(f'       Text: "{chunk["text"][:80]}..."')
            
        else:
            print('   ⚠ No embeddings found in existing data')
            return False
            
    except Exception as e:
        print(f'   ✗ Manual similarity search failed: {e}')
        return False
    
    # Test 4: Generate answer using retrieved context
    print('\n4. Testing answer generation with retrieved context...')
    try:
        # Use the similar chunks as context for answer generation
        query = "What is the main topic discussed in the document?"
        
        # Format chunks for the query processor
        context_chunks = []
        for i, chunk in enumerate(similar_chunks[:2]):
            context_chunk = {
                'text': chunk['text'],
                'source_document': chunk['metadata'].get('source_document', 'Unknown Document'),
                'similarity_score': chunk['similarity_score'],
                'chunk_id': chunk['chunk_id'],
                'metadata': chunk['metadata']
            }
            context_chunks.append(context_chunk)
        
        print(f'   → Query: "{query}"')
        print(f'   → Using {len(context_chunks)} context chunks')
        
        # Generate answer
        result = processor.generate_answer_with_citations(query, context_chunks)
        
        print(f'   ✓ Answer generated successfully!')
        print(f'   ✓ Answer length: {len(result["answer"])} characters')
        print(f'   ✓ Model used: {result["model"]}')
        print(f'   ✓ Context chunks used: {result["context_chunks_used"]}')
        print(f'   ✓ Citations: {result["citation_count"]}')
        print(f'   ✓ Token usage: {result["token_usage"]["total_tokens"]} tokens')
        
        # Display the answer
        print(f'\n   Generated Answer:')
        print(f'   {"-" * 60}')
        print(f'   {result["answer"]}')
        print(f'   {"-" * 60}')
        
        # Display citations
        print(f'\n   Citations:')
        for citation in result['citations']:
            print(f'   - Citation {citation["citation_id"]}: {citation["source_document"]}')
            print(f'     Relevance: {citation["similarity_score"]:.3f}')
            print(f'     Preview: "{citation["text_preview"][:100]}..."')
        
    except Exception as e:
        print(f'   ✗ Answer generation failed: {e}')
        return False
    
    # Test 5: Test different types of queries
    print('\n5. Testing different query types...')
    try:
        test_queries = [
            "Can you summarize the key points?",
            "What are the main findings mentioned?",
            "What specific details are provided?"
        ]
        
        for i, test_query in enumerate(test_queries, 1):
            print(f'   → Test query {i}: "{test_query}"')
            
            result = processor.generate_answer(test_query, context_chunks[:1])  # Use 1 chunk
            
            print(f'     ✓ Answer generated: {len(result["answer"])} chars')
            print(f'     ✓ Tokens used: {result["token_usage"]["total_tokens"]}')
            
            # Show brief answer preview
            answer_preview = result["answer"][:150].replace('\n', ' ')
            print(f'     ✓ Preview: "{answer_preview}..."')
            print()
        
    except Exception as e:
        print(f'   ✗ Multiple query test failed: {e}')
        return False
    
    # Test 6: Token usage statistics
    print('\n6. Testing token usage statistics...')
    try:
        stats = processor.get_token_usage_stats()
        cost = processor.estimate_cost()
        
        print(f'   ✓ Total queries processed: {stats["queries_processed"]}')
        print(f'   ✓ Total tokens used: {stats["total_tokens"]}')
        print(f'   ✓ Average tokens per query: {stats["average_tokens_per_query"]:.1f}')
        print(f'   ✓ Estimated cost: ${cost["total_cost"]:.4f} {cost["currency"]}')
        print(f'   ✓ Note: {cost["note"]}')
        
    except Exception as e:
        print(f'   ✗ Token usage statistics failed: {e}')
        return False
    
    print('\n🎉 RAG test with existing data completed successfully!')
    return True

if __name__ == "__main__":
    success = test_with_existing_data()
    if success:
        print('\n=== RAG Pipeline with Gemini Successfully Tested ===')
        print('✓ Vector database operations working')
        print('✓ Similarity search with existing embeddings')
        print('✓ Answer generation using Gemini API')
        print('✓ Citation and source attribution')
        print('✓ Token usage tracking and cost estimation')
        print('✓ Multiple query types supported')
        print('\n✅ Task 4.2 Implementation Complete and Tested!')
        sys.exit(0)
    else:
        print('\n❌ Test failed. Check the implementation or data availability.')
        sys.exit(1)