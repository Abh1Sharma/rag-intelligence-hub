"""
Unit tests for OpenAI embeddings integration.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.rag_pipeline import EmbeddingManager, EmbeddingResult, EmbeddingError


class TestEmbeddingManager:
    """Test cases for EmbeddingManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = EmbeddingManager()
        
        # Mock embedding response
        self.mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 300  # 1536 dimensions (typical)
        self.mock_response = Mock()
        self.mock_response.data = [Mock()]
        self.mock_response.data[0].embedding = self.mock_embedding
        self.mock_response.usage.total_tokens = 10
    
    def test_validate_api_key_missing(self):
        """Test API key validation with missing key."""
        with patch('src.rag_pipeline.config.OPENAI_API_KEY', ''):
            with pytest.raises(EmbeddingError) as exc_info:
                self.manager._validate_api_key()
            assert "not configured" in str(exc_info.value)
    
    def test_validate_api_key_test_key(self):
        """Test API key validation with test key."""
        with patch('src.rag_pipeline.config.OPENAI_API_KEY', 'test_key_for_setup_validation'):
            with pytest.raises(EmbeddingError) as exc_info:
                self.manager._validate_api_key()
            assert "not configured" in str(exc_info.value)
    
    def test_validate_api_key_valid(self):
        """Test API key validation with valid key."""
        with patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-valid-key'):
            result = self.manager._validate_api_key()
            assert result is True
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_generate_embedding_success(self):
        """Test successful single embedding generation."""
        with patch.object(self.manager.client.embeddings, 'create', return_value=self.mock_response):
            result = self.manager.generate_embedding("Test text")
            
            assert isinstance(result, EmbeddingResult)
            assert result.text == "Test text"
            assert result.embedding == self.mock_embedding
            assert result.token_count == 10
            assert result.model == self.manager.model
            assert isinstance(result.timestamp, float)
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_generate_embedding_empty_text(self):
        """Test embedding generation with empty text."""
        with pytest.raises(EmbeddingError) as exc_info:
            self.manager.generate_embedding("")
        assert "empty text" in str(exc_info.value)
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_generate_embedding_with_retry(self):
        """Test embedding generation with retry on failure."""
        # Mock first call to fail, second to succeed
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = [
            Exception("Temporary error"),
            self.mock_response
        ]
        
        with patch.object(self.manager, 'client', mock_client):
            with patch('time.sleep'):  # Speed up test
                result = self.manager.generate_embedding("Test text")
                
                assert isinstance(result, EmbeddingResult)
                assert mock_client.embeddings.create.call_count == 2
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_generate_embedding_max_retries_exceeded(self):
        """Test embedding generation when max retries exceeded."""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Persistent error")
        
        with patch.object(self.manager, 'client', mock_client):
            with patch('time.sleep'):  # Speed up test
                with pytest.raises(EmbeddingError) as exc_info:
                    self.manager.generate_embedding("Test text")
                
                assert "Failed to generate embedding" in str(exc_info.value)
                assert mock_client.embeddings.create.call_count == 3  # max_retries
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_generate_embeddings_batch_success(self):
        """Test successful batch embedding generation."""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # Mock batch response
        mock_batch_response = Mock()
        mock_batch_response.data = []
        for i in range(len(texts)):
            mock_data = Mock()
            mock_data.embedding = [0.1 * (i + 1)] * 1536
            mock_batch_response.data.append(mock_data)
        mock_batch_response.usage.total_tokens = 30
        
        with patch.object(self.manager.client.embeddings, 'create', return_value=mock_batch_response):
            with patch('time.sleep'):  # Speed up test
                results = self.manager.generate_embeddings_batch(texts)
                
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert isinstance(result, EmbeddingResult)
                    assert result.text == texts[i]
                    assert len(result.embedding) == 1536
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_generate_embeddings_batch_empty_list(self):
        """Test batch embedding with empty list."""
        results = self.manager.generate_embeddings_batch([])
        assert results == []
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_generate_embeddings_batch_filters_empty_texts(self):
        """Test batch embedding filters out empty texts."""
        texts = ["Valid text", "", "  ", "Another valid text"]
        
        # Mock response for 2 valid texts
        mock_batch_response = Mock()
        mock_batch_response.data = []
        for i in range(2):  # Only 2 valid texts
            mock_data = Mock()
            mock_data.embedding = [0.1] * 1536
            mock_batch_response.data.append(mock_data)
        mock_batch_response.usage.total_tokens = 20
        
        with patch.object(self.manager.client.embeddings, 'create', return_value=mock_batch_response):
            with patch('time.sleep'):
                results = self.manager.generate_embeddings_batch(texts)
                
                assert len(results) == 2
                assert results[0].text == "Valid text"
                assert results[1].text == "Another valid text"
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_generate_embeddings_for_chunks_success(self):
        """Test embedding generation for document chunks."""
        chunks = [
            {'text': 'Chunk 1 text', 'chunk_id': 'chunk_0'},
            {'text': 'Chunk 2 text', 'chunk_id': 'chunk_1'}
        ]
        
        # Mock batch response
        mock_batch_response = Mock()
        mock_batch_response.data = []
        for i in range(2):
            mock_data = Mock()
            mock_data.embedding = [0.1 * (i + 1)] * 1536
            mock_batch_response.data.append(mock_data)
        mock_batch_response.usage.total_tokens = 20
        
        with patch.object(self.manager.client.embeddings, 'create', return_value=mock_batch_response):
            with patch('time.sleep'):
                enhanced_chunks = self.manager.generate_embeddings_for_chunks(chunks)
                
                assert len(enhanced_chunks) == 2
                for i, chunk in enumerate(enhanced_chunks):
                    assert 'embedding' in chunk
                    assert 'embedding_model' in chunk
                    assert 'embedding_timestamp' in chunk
                    assert 'embedding_dimensions' in chunk
                    assert chunk['chunk_id'] == f'chunk_{i}'
                    assert len(chunk['embedding']) == 1536
    
    def test_generate_embeddings_for_chunks_empty_list(self):
        """Test chunk embedding with empty list."""
        result = self.manager.generate_embeddings_for_chunks([])
        assert result == []
    
    def test_calculate_similarity_identical_embeddings(self):
        """Test similarity calculation with identical embeddings."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        similarity = self.manager.calculate_similarity(embedding, embedding)
        assert similarity == 1.0
    
    def test_calculate_similarity_orthogonal_embeddings(self):
        """Test similarity calculation with orthogonal embeddings."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        similarity = self.manager.calculate_similarity(embedding1, embedding2)
        assert similarity == 0.5  # Normalized to 0-1 range
    
    def test_calculate_similarity_different_dimensions(self):
        """Test similarity calculation with different dimensions."""
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.1, 0.2]
        
        with pytest.raises(EmbeddingError) as exc_info:
            self.manager.calculate_similarity(embedding1, embedding2)
        assert "same dimensions" in str(exc_info.value)
    
    def test_calculate_similarity_zero_vectors(self):
        """Test similarity calculation with zero vectors."""
        embedding1 = [0.0, 0.0, 0.0]
        embedding2 = [0.1, 0.2, 0.3]
        similarity = self.manager.calculate_similarity(embedding1, embedding2)
        assert similarity == 0.0
    
    def test_get_embedding_stats_empty_list(self):
        """Test statistics generation with empty list."""
        stats = self.manager.get_embedding_stats([])
        
        assert stats['total_embeddings'] == 0
        assert stats['total_tokens'] == 0
        assert stats['average_tokens_per_embedding'] == 0
        assert stats['embedding_dimensions'] == 0
        assert stats['models_used'] == []
    
    def test_get_embedding_stats_with_results(self):
        """Test statistics generation with embedding results."""
        results = [
            EmbeddingResult(
                text="Text 1",
                embedding=[0.1] * 1536,
                token_count=10,
                model="text-embedding-ada-002",
                timestamp=1000.0
            ),
            EmbeddingResult(
                text="Text 2",
                embedding=[0.2] * 1536,
                token_count=15,
                model="text-embedding-ada-002",
                timestamp=1001.0
            )
        ]
        
        stats = self.manager.get_embedding_stats(results)
        
        assert stats['total_embeddings'] == 2
        assert stats['total_tokens'] == 25
        assert stats['average_tokens_per_embedding'] == 12.5
        assert stats['embedding_dimensions'] == 1536
        assert stats['models_used'] == ["text-embedding-ada-002"]
        assert stats['generation_timespan']['start'] == 1000.0
        assert stats['generation_timespan']['end'] == 1001.0


class TestEmbeddingIntegration:
    """Integration tests for embedding functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = EmbeddingManager()
    
    def test_embedding_manager_configuration(self):
        """Test that embedding manager is properly configured."""
        assert self.manager.model == "text-embedding-ada-002"  # From config
        assert self.manager.max_retries == 3
        assert self.manager.retry_delay == 1.0
        assert self.manager.rate_limit_delay == 0.1
    
    def test_embedding_result_dataclass(self):
        """Test EmbeddingResult dataclass functionality."""
        result = EmbeddingResult(
            text="Test text",
            embedding=[0.1, 0.2, 0.3],
            token_count=5,
            model="test-model",
            timestamp=1000.0
        )
        
        assert result.text == "Test text"
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.token_count == 5
        assert result.model == "test-model"
        assert result.timestamp == 1000.0
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_integration_with_document_chunks(self):
        """Test integration with document processing chunks."""
        # Sample chunks from document processing
        chunks = [
            {
                'chunk_id': 'doc1_0',
                'text': 'This is the first chunk of text from a document.',
                'token_count': 12,
                'source_document': 'test.txt'
            },
            {
                'chunk_id': 'doc1_1', 
                'text': 'This is the second chunk with different content.',
                'token_count': 10,
                'source_document': 'test.txt'
            }
        ]
        
        # Mock the OpenAI response
        mock_batch_response = Mock()
        mock_batch_response.data = []
        for i in range(2):
            mock_data = Mock()
            mock_data.embedding = [0.1 * (i + 1)] * 1536
            mock_batch_response.data.append(mock_data)
        mock_batch_response.usage.total_tokens = 22
        
        with patch.object(self.manager.client.embeddings, 'create', return_value=mock_batch_response):
            with patch('time.sleep'):
                enhanced_chunks = self.manager.generate_embeddings_for_chunks(chunks)
                
                # Verify original chunk data is preserved
                for i, chunk in enumerate(enhanced_chunks):
                    assert chunk['chunk_id'] == f'doc1_{i}'
                    assert chunk['source_document'] == 'test.txt'
                    assert 'token_count' in chunk  # Original field preserved
                    
                    # Verify embedding data is added
                    assert 'embedding' in chunk
                    assert 'embedding_model' in chunk
                    assert 'embedding_timestamp' in chunk
                    assert 'embedding_dimensions' in chunk
                    assert len(chunk['embedding']) == 1536
    
    def test_similarity_calculation_accuracy(self):
        """Test accuracy of similarity calculations."""
        # Test with known vectors
        vec1 = [1.0, 0.0, 0.0]  # Unit vector along x-axis
        vec2 = [0.0, 1.0, 0.0]  # Unit vector along y-axis
        vec3 = [1.0, 0.0, 0.0]  # Same as vec1
        
        # Orthogonal vectors should have similarity of 0.5 (normalized)
        sim_orthogonal = self.manager.calculate_similarity(vec1, vec2)
        assert abs(sim_orthogonal - 0.5) < 0.001
        
        # Identical vectors should have similarity of 1.0
        sim_identical = self.manager.calculate_similarity(vec1, vec3)
        assert abs(sim_identical - 1.0) < 0.001
        
        # Test with opposite vectors
        vec4 = [-1.0, 0.0, 0.0]  # Opposite to vec1
        sim_opposite = self.manager.calculate_similarity(vec1, vec4)
        assert abs(sim_opposite - 0.0) < 0.001
    
    def test_error_handling_chain(self):
        """Test that errors propagate correctly through the system."""
        # Test API key validation error
        with patch('src.rag_pipeline.config.OPENAI_API_KEY', ''):
            with pytest.raises(EmbeddingError):
                self.manager.generate_embedding("test")
        
        # Test empty text error
        with patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test'):
            with pytest.raises(EmbeddingError):
                self.manager.generate_embedding("")
        
        # Test dimension mismatch error
        with pytest.raises(EmbeddingError):
            self.manager.calculate_similarity([1, 2], [1, 2, 3])