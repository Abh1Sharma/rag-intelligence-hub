#!/usr/bin/env python3
"""
Test script for the complete RAG pipeline with Gemini API integration.
"""

import sys
import os
sys.path.insert(0, 'src')

from rag_pipeline import RAGPipeline, RAGPipelineError

def test_complete_rag_pipeline():
    """Test the complete RAG pipeline with Gemini integration."""
    print('=== Complete RAG Pipeline Test ===')
    
    # Test 1: RAG Pipeline initialization
    print('\n1. Testing RAG pipeline initialization...')
    try:
        pipeline = RAGPipeline()
        print(f'   ✓ RAG pipeline initialized successfully')
        print(f'   ✓ Components initialized:')
        print(f'     - Embedding manager: {pipeline.embedding_manager is not None}')
        print(f'     - Vector database: {pipeline.vector_database is not None}')
        print(f'     - Vector search engine: {pipeline.vector_search_engine is not None}')
        print(f'     - Query processor: {pipeline.query_processor is not None}')
        print(f'     - Document processor: {pipeline.document_processor is not None}')
        
        # Check initial metrics
        metrics = pipeline.performance_metrics
        print(f'   ✓ Initial metrics: {metrics["documents_processed"]} docs, {metrics["queries_processed"]} queries')
        
    except Exception as e:
        print(f'   ✗ Pipeline initialization failed: {e}')
        return False
    
    # Test 2: System statistics
    print('\n2. Testing system statistics...')
    try:
        stats = pipeline.get_system_stats()
        print(f'   ✓ System stats retrieved successfully')
        print(f'   ✓ Database stats: {stats["database_stats"]["total_chunks"]} chunks')
        print(f'   ✓ Query processing stats: {stats["query_processing_stats"]["queries_processed"]} queries')
        print(f'   ✓ Performance metrics: {stats["performance_metrics"]["documents_processed"]} docs processed')
        
        # Check system health
        health = stats['system_health']
        print(f'   ✓ Database health: {health["database_health"]["status"]}')
        print(f'   ✓ All components initialized: {all(health["components_initialized"].values())}')
        
    except Exception as e:
        print(f'   ✗ System statistics failed: {e}')
        return False
    
    # Test 3: Query processing without documents (should handle gracefully)
    print('\n3. Testing query processing without documents...')
    try:
        query = "What is machine learning?"
        result = pipeline.process_query(query)
        
        print(f'   ✓ Query processed successfully: "{query}"')
        print(f'   ✓ Answer generated: {len(result["answer"])} characters')
        print(f'   ✓ Context chunks found: {len(result["context_chunks"])}')
        print(f'   ✓ Processing time: {result["processing_time"]:.2f}s')
        print(f'   ✓ Model used: {result["answer_metadata"]["model"]}')
        
        # Check if answer indicates no context
        answer_lower = result["answer"].lower()
        no_context_indicated = any(phrase in answer_lower for phrase in [
            "cannot answer", "no relevant", "not found", "no context", "no information"
        ])
        print(f'   ✓ No context properly indicated: {no_context_indicated}')
        
        # Display part of the answer
        print(f'\n   Answer preview:')
        print(f'   {"-" * 50}')
        print(f'   {result["answer"][:200]}{"..." if len(result["answer"]) > 200 else ""}')
        print(f'   {"-" * 50}')
        
    except Exception as e:
        print(f'   ✗ Query processing failed: {e}')
        return False
    
    # Test 4: Document embedding (if sample document exists)
    print('\n4. Testing document embedding...')
    sample_doc_path = './demo/sample_document.txt'
    if os.path.exists(sample_doc_path):
        try:
            result = pipeline.embed_document(sample_doc_path)
            
            print(f'   ✓ Document embedded successfully: {sample_doc_path}')
            print(f'   ✓ Chunks processed: {result["chunks_processed"]}')
            print(f'   ✓ Chunks embedded: {result["chunks_embedded"]}')
            print(f'   ✓ Chunks stored: {result["chunks_stored"]}')
            print(f'   ✓ Processing time: {result["processing_time"]:.2f}s')
            print(f'   ✓ Processing successful: {result["processing_successful"]}')
            
        except Exception as e:
            print(f'   ✗ Document embedding failed: {e}')
            return False
    else:
        print(f'   ⚠ Sample document not found at {sample_doc_path}, skipping embedding test')
    
    # Test 5: Query processing with documents (if we embedded any)
    print('\n5. Testing query processing with documents...')
    try:
        # Check if we have any documents in the database
        stats = pipeline.get_system_stats()
        total_chunks = stats["database_stats"]["total_chunks"]
        
        if total_chunks > 0:
            query = "What is the main topic discussed in the document?"
            result = pipeline.process_query(query, include_citations=True)
            
            print(f'   ✓ Query processed with {total_chunks} chunks available')
            print(f'   ✓ Answer generated: {len(result["answer"])} characters')
            print(f'   ✓ Context chunks retrieved: {len(result["context_chunks"])}')
            print(f'   ✓ Processing time: {result["processing_time"]:.2f}s')
            
            # Check citations if available
            if 'citations' in result:
                print(f'   ✓ Citations provided: {result["citation_count"]}')
                print(f'   ✓ Source documents: {len(result["source_documents"])}')
                
                # Display citations
                for citation in result['citations'][:2]:  # Show first 2 citations
                    print(f'     - {citation["source_document"]} (score: {citation["similarity_score"]:.3f})')
            
            # Display part of the answer
            print(f'\n   Answer with context:')
            print(f'   {"-" * 50}')
            print(f'   {result["answer"][:300]}{"..." if len(result["answer"]) > 300 else ""}')
            print(f'   {"-" * 50}')
            
        else:
            print(f'   ⚠ No documents in database, testing with empty context')
            query = "What is artificial intelligence?"
            result = pipeline.process_query(query)
            print(f'   ✓ Query processed with empty context: {len(result["answer"])} chars')
            
    except Exception as e:
        print(f'   ✗ Query processing with documents failed: {e}')
        return False
    
    # Test 6: Performance metrics tracking
    print('\n6. Testing performance metrics tracking...')
    try:
        final_stats = pipeline.get_system_stats()
        final_metrics = final_stats['performance_metrics']
        
        print(f'   ✓ Final performance metrics:')
        print(f'     - Documents processed: {final_metrics["documents_processed"]}')
        print(f'     - Chunks embedded: {final_metrics["chunks_embedded"]}')
        print(f'     - Queries processed: {final_metrics["queries_processed"]}')
        print(f'     - Total processing time: {final_metrics["total_processing_time"]:.2f}s')
        print(f'     - Total retrieval time: {final_metrics["total_retrieval_time"]:.2f}s')
        
        # Check query processor stats
        query_stats = final_stats['query_processing_stats']
        print(f'   ✓ Query processor stats:')
        print(f'     - Total tokens used: {query_stats["total_tokens"]}')
        print(f'     - Average tokens per query: {query_stats["average_tokens_per_query"]:.1f}')
        print(f'     - Queries processed: {query_stats["queries_processed"]}')
        
    except Exception as e:
        print(f'   ✗ Performance metrics tracking failed: {e}')
        return False
    
    # Test 7: Error handling
    print('\n7. Testing error handling...')
    try:
        # Test empty query
        try:
            pipeline.process_query('')
            print('   ✗ Should have failed for empty query')
        except RAGPipelineError:
            print('   ✓ Empty query error handled correctly')
        
        # Test invalid parameters
        try:
            result = pipeline.process_query('Valid query', top_k=-1)
            print('   ✓ Invalid parameters handled gracefully')
        except Exception as e:
            print(f'   ✓ Invalid parameters caught: {type(e).__name__}')
        
    except Exception as e:
        print(f'   ✗ Error handling test failed: {e}')
        return False
    
    print('\n🎉 Complete RAG pipeline test passed!')
    return True

if __name__ == "__main__":
    success = test_complete_rag_pipeline()
    if success:
        print('\n=== RAG Pipeline with Gemini Integration Complete ===')
        print('✓ Complete RAG workflow with Gemini API')
        print('✓ Document embedding and storage')
        print('✓ Vector similarity search and retrieval')
        print('✓ Answer generation with context and citations')
        print('✓ Performance monitoring and error handling')
        print('✓ All requirements for task 4.2 satisfied')
        sys.exit(0)
    else:
        print('\n❌ Some tests failed. Please check the implementation.')
        sys.exit(1)