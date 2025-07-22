"""
Unit tests for the core RAG pipeline functionality.
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock

from src.rag_pipeline import RAGPipeline, RAGPipelineError


class TestRAGPipeline:
    """Test cases for RAGPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config to use temp directory
        with patch('src.rag_pipeline.config.CHROMA_PERSIST_DIRECTORY', self.temp_dir):
            with patch('src.rag_pipeline.config.CHROMA_COLLECTION_NAME', 'pipeline_test'):
                self.pipeline = RAGPipeline()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test RAG pipeline initialization."""
        assert self.pipeline.embedding_manager is not None
        assert self.pipeline.vector_database is not None
        assert self.pipeline.vector_search_engine is not None
        assert self.pipeline.document_processor is not None
        
        # Check performance metrics initialization
        assert self.pipeline.performance_metrics['documents_processed'] == 0
        assert self.pipeline.performance_metrics['chunks_embedded'] == 0
        assert self.pipeline.performance_metrics['queries_processed'] == 0
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_process_and_embed_document_success(self):
        """Test successful document processing and embedding."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write("This is a test document for RAG pipeline testing.")
            tmp_path = tmp.name
        
        try:
            # Mock the embedding generation
            mock_response = Mock()
            mock_response.data = [Mock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
            mock_response.usage.total_tokens = 10
            
            with patch.object(self.pipeline.embedding_manager.client.embeddings, 'create', return_value=mock_response):
                result = self.pipeline.process_and_embed_document(tmp_path)
                
                assert result['pipeline_successful'] is True
                assert result['storage_successful'] is True
                assert result['chunks_processed'] > 0
                assert result['chunks_embedded'] > 0
                assert result['chunks_stored'] > 0
                assert result['processing_time'] > 0
                assert 'document_info' in result
                
                # Check that performance metrics were updated
                assert self.pipeline.performance_metrics['documents_processed'] == 1
                assert self.pipeline.performance_metrics['chunks_embedded'] > 0
                
        finally:
            os.unlink(tmp_path)
    
    def test_process_and_embed_document_with_progress(self):
        """Test document processing with progress callback."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write("Test content for progress tracking.")
            tmp_path = tmp.name
        
        progress_updates = []
        
        def progress_callback(message, progress):
            progress_updates.append((message, progress))
        
        try:
            # Mock the embedding generation
            mock_response = Mock()
            mock_response.data = [Mock()]
            mock_response.data[0].embedding = [0.1] * 1536
            mock_response.usage.total_tokens = 5
            
            with patch.object(self.pipeline.embedding_manager.client.embeddings, 'create', return_value=mock_response):
                with patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key'):
                    result = self.pipeline.process_and_embed_document(tmp_path, progress_callback)
                    
                    assert result['pipeline_successful'] is True
                    assert len(progress_updates) > 0
                    
                    # Check that progress was tracked
                    messages = [update[0] for update in progress_updates]
                    assert any("Processing document" in msg for msg in messages)
                    assert any("Generating embeddings" in msg for msg in messages)
                    assert any("Storing in vector database" in msg for msg in messages)
                    
        finally:
            os.unlink(tmp_path)
    
    def test_process_and_embed_document_file_not_found(self):
        """Test processing with non-existent file."""
        with pytest.raises(RAGPipelineError) as exc_info:
            self.pipeline.process_and_embed_document("/nonexistent/file.txt")
        assert "RAG pipeline failed" in str(exc_info.value)
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_process_and_embed_documents_batch(self):
        """Test batch processing of multiple documents."""
        # Create multiple temporary test files
        file_paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                tmp.write(f"Test document {i+1} content for batch processing.")
                file_paths.append(tmp.name)
        
        try:
            # Mock the embedding generation
            mock_response = Mock()
            mock_response.data = [Mock()]
            mock_response.data[0].embedding = [0.1] * 1536
            mock_response.usage.total_tokens = 8
            
            with patch.object(self.pipeline.embedding_manager.client.embeddings, 'create', return_value=mock_response):
                results = self.pipeline.process_and_embed_documents(file_paths)
                
                assert len(results) == 3
                
                # Check that all documents were processed successfully
                successful_count = sum(1 for r in results if r['pipeline_successful'])
                assert successful_count == 3
                
                # Check performance metrics
                assert self.pipeline.performance_metrics['documents_processed'] == 3
                
        finally:
            for file_path in file_paths:
                os.unlink(file_path)
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_query_documents_success(self):
        """Test successful document querying."""
        # First, store some test chunks
        test_chunks = [
            {
                'chunk_id': 'query_test_0',
                'text': 'Machine learning is a powerful AI technique.',
                'embedding': [0.1, 0.2, 0.3] * 512,
                'source_document': 'ml_guide.txt',
                'token_count': 8
            }
        ]
        
        self.pipeline.vector_database.store_chunks(test_chunks)
        
        # Mock embedding generation for query
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.15, 0.25, 0.35] * 512  # Similar to stored chunk
        mock_response.usage.total_tokens = 5
        
        with patch.object(self.pipeline.embedding_manager.client.embeddings, 'create', return_value=mock_response):
            result = self.pipeline.query_documents("machine learning", top_k=5)
            
            assert 'results' in result
            assert 'metadata' in result
            assert 'pipeline_metadata' in result
            
            # Check pipeline metadata
            pipeline_meta = result['pipeline_metadata']
            assert 'retrieval_time' in pipeline_meta
            assert 'pipeline_version' in pipeline_meta
            assert 'components_used' in pipeline_meta
            
            # Check that performance metrics were updated
            assert self.pipeline.performance_metrics['queries_processed'] == 1
            assert self.pipeline.performance_metrics['total_retrieval_time'] > 0
    
    def test_query_documents_empty_query(self):
        """Test querying with empty query."""
        with pytest.raises(RAGPipelineError) as exc_info:
            self.pipeline.query_documents("")
        assert "Query cannot be empty" in str(exc_info.value)
    
    def test_get_pipeline_status_healthy(self):
        """Test getting pipeline status when healthy."""
        status = self.pipeline.get_pipeline_status()
        
        assert 'pipeline_healthy' in status
        assert 'components' in status
        assert 'database_stats' in status
        assert 'performance_metrics' in status
        assert 'performance_averages' in status
        assert 'last_updated' in status
        
        # Check component status
        components = status['components']
        assert 'document_processor' in components
        assert 'embedding_manager' in components
        assert 'vector_database' in components
        assert 'vector_search_engine' in components
        
        # Check performance averages
        averages = status['performance_averages']
        assert 'avg_processing_time' in averages
        assert 'avg_retrieval_time' in averages
        assert 'chunks_per_document' in averages
    
    def test_reset_pipeline_metrics(self):
        """Test resetting pipeline performance metrics."""
        # Set some metrics first
        self.pipeline.performance_metrics['documents_processed'] = 5
        self.pipeline.performance_metrics['queries_processed'] = 10
        
        # Reset metrics
        self.pipeline.reset_pipeline_metrics()
        
        # Verify reset
        assert self.pipeline.performance_metrics['documents_processed'] == 0
        assert self.pipeline.performance_metrics['queries_processed'] == 0
        assert self.pipeline.performance_metrics['total_processing_time'] == 0.0
        assert self.pipeline.performance_metrics['total_retrieval_time'] == 0.0
    
    def test_validate_pipeline_health_healthy(self):
        """Test pipeline health validation when healthy."""
        health = self.pipeline.validate_pipeline_health()
        
        assert 'overall_healthy' in health
        assert 'component_health' in health
        assert 'issues' in health
        assert 'recommendations' in health
        
        # Should be healthy by default
        assert health['overall_healthy'] is True
        assert len(health['issues']) == 0
        
        # Check component health
        component_health = health['component_health']
        assert component_health['document_processor'] is True
        assert component_health['embedding_manager'] is True
        assert component_health['vector_database'] is True
        assert component_health['vector_search_engine'] is True
    
    def test_validate_pipeline_health_with_performance_warning(self):
        """Test health validation with performance warnings."""
        # Set high processing time to trigger performance warning
        self.pipeline.performance_metrics['documents_processed'] = 1
        self.pipeline.performance_metrics['total_processing_time'] = 35.0  # > 30 seconds
        
        health = self.pipeline.validate_pipeline_health()
        
        # Should still be healthy but have performance recommendations
        assert health['overall_healthy'] is True
        recommendations = ' '.join(health['recommendations'])
        assert 'optimizing document processing' in recommendations


class TestRAGPipelineIntegration:
    """Integration tests for the complete RAG pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('src.rag_pipeline.config.CHROMA_PERSIST_DIRECTORY', self.temp_dir):
            with patch('src.rag_pipeline.config.CHROMA_COLLECTION_NAME', 'integration_pipeline_test'):
                self.pipeline = RAGPipeline()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_end_to_end_pipeline_workflow(self):
        """Test complete end-to-end RAG pipeline workflow."""
        # Step 1: Create test document
        test_content = """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.
        
        Key concepts include supervised learning, unsupervised learning, and reinforcement learning.
        Each approach has different applications and use cases in real-world scenarios.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name
        
        try:
            # Step 2: Mock embedding generation
            mock_response = Mock()
            mock_response.data = [Mock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
            mock_response.usage.total_tokens = 15
            
            with patch.object(self.pipeline.embedding_manager.client.embeddings, 'create', return_value=mock_response):
                # Step 3: Process and embed document
                process_result = self.pipeline.process_and_embed_document(tmp_path)
                
                assert process_result['pipeline_successful'] is True
                assert process_result['chunks_processed'] > 0
                
                # Step 4: Query the processed document
                query_result = self.pipeline.query_documents("machine learning concepts", top_k=3)
                
                assert 'results' in query_result
                assert 'metadata' in query_result
                assert 'pipeline_metadata' in query_result
                
                # Step 5: Check pipeline status
                status = self.pipeline.get_pipeline_status()
                
                assert status['pipeline_healthy'] is True
                assert status['performance_metrics']['documents_processed'] == 1
                assert status['performance_metrics']['queries_processed'] == 1
                
                # Step 6: Validate health
                health = self.pipeline.validate_pipeline_health()
                
                assert health['overall_healthy'] is True
                assert len(health['issues']) == 0
                
        finally:
            os.unlink(tmp_path)
    
    def test_pipeline_component_integration(self):
        """Test that all pipeline components are properly integrated."""
        # Verify component initialization
        assert self.pipeline.embedding_manager is not None
        assert self.pipeline.vector_database is not None
        assert self.pipeline.vector_search_engine is not None
        assert self.pipeline.document_processor is not None
        
        # Verify components are properly connected
        assert self.pipeline.vector_search_engine.embedding_manager is not None
        assert self.pipeline.vector_search_engine.vector_database is not None
        
        # Test database health
        db_health = self.pipeline.vector_database.health_check()
        assert db_health['status'] == 'healthy'
        
        # Test pipeline status
        status = self.pipeline.get_pipeline_status()
        assert 'pipeline_healthy' in status
        assert 'components' in status
    
    def test_pipeline_error_handling(self):
        """Test error handling throughout the pipeline."""
        # Test with invalid file
        with pytest.raises(RAGPipelineError):
            self.pipeline.process_and_embed_document("/invalid/path/file.txt")
        
        # Test with empty query
        with pytest.raises(RAGPipelineError):
            self.pipeline.query_documents("")
        
        # Pipeline should still be functional after errors
        status = self.pipeline.get_pipeline_status()
        assert 'pipeline_healthy' in status
    
    def test_performance_metrics_tracking(self):
        """Test that performance metrics are properly tracked."""
        initial_metrics = self.pipeline.performance_metrics.copy()
        
        # All metrics should start at 0
        assert initial_metrics['documents_processed'] == 0
        assert initial_metrics['chunks_embedded'] == 0
        assert initial_metrics['queries_processed'] == 0
        assert initial_metrics['total_processing_time'] == 0.0
        assert initial_metrics['total_retrieval_time'] == 0.0
        
        # Test reset functionality
        self.pipeline.performance_metrics['documents_processed'] = 5
        self.pipeline.reset_pipeline_metrics()
        
        assert self.pipeline.performance_metrics['documents_processed'] == 0
    
    def test_pipeline_status_comprehensive(self):
        """Test comprehensive pipeline status reporting."""
        status = self.pipeline.get_pipeline_status()
        
        # Check all required fields
        required_fields = [
            'pipeline_healthy', 'components', 'database_stats',
            'performance_metrics', 'performance_averages', 'last_updated'
        ]
        
        for field in required_fields:
            assert field in status
        
        # Check component status
        components = status['components']
        required_components = [
            'document_processor', 'embedding_manager', 
            'vector_database', 'vector_search_engine'
        ]
        
        for component in required_components:
            assert component in components
        
        # Check performance averages
        averages = status['performance_averages']
        required_averages = [
            'avg_processing_time', 'avg_retrieval_time', 'chunks_per_document'
        ]
        
        for avg in required_averages:
            assert avg in averages