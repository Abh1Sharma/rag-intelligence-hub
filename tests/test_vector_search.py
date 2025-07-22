"""
Unit tests for vector similarity search functionality.
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock

from src.rag_pipeline import VectorSearchEngine, EmbeddingError, VectorDatabaseError


class TestVectorSearchEngine:
    """Test cases for VectorSearchEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config to use temp directory
        with patch('src.rag_pipeline.config.CHROMA_PERSIST_DIRECTORY', self.temp_dir):
            with patch('src.rag_pipeline.config.CHROMA_COLLECTION_NAME', 'search_test'):
                self.search_engine = VectorSearchEngine()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_generate_query_embedding_success(self):
        """Test successful query embedding generation."""
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        mock_response.usage.total_tokens = 5
        
        with patch.object(self.search_engine.embedding_manager.client.embeddings, 'create', return_value=mock_response):
            embedding = self.search_engine.generate_query_embedding("test query")
            
            assert len(embedding) == 1536
            assert isinstance(embedding, list)
            assert all(isinstance(x, float) for x in embedding)
    
    def test_generate_query_embedding_empty_query(self):
        """Test query embedding generation with empty query."""
        with pytest.raises(EmbeddingError) as exc_info:
            self.search_engine.generate_query_embedding("")
        assert "cannot be empty" in str(exc_info.value)
    
    def test_highlight_context_basic(self):
        """Test basic context highlighting."""
        text = "Machine learning is a subset of artificial intelligence."
        query = "machine learning"
        
        highlighted = self.search_engine._highlight_context(text, query)
        
        assert "**MACHINE**" in highlighted
        assert "**LEARNING**" in highlighted
    
    def test_highlight_context_case_insensitive(self):
        """Test case-insensitive highlighting."""
        text = "Deep Learning uses neural networks."
        query = "deep learning"
        
        highlighted = self.search_engine._highlight_context(text, query)
        
        assert "**DEEP**" in highlighted
        assert "**LEARNING**" in highlighted
    
    def test_highlight_context_short_terms(self):
        """Test that short terms are not highlighted."""
        text = "AI is a broad field of computer science."
        query = "AI is"
        
        highlighted = self.search_engine._highlight_context(text, query)
        
        # "AI" and "is" should not be highlighted (too short)
        assert "**AI**" not in highlighted
        assert "**IS**" not in highlighted
    
    def test_generate_relevance_explanation_high_score(self):
        """Test relevance explanation for high similarity score."""
        explanation = self.search_engine._generate_relevance_explanation(0.95, "test query")
        
        assert "Highly relevant" in explanation
        assert "test query" in explanation
        assert "0.950" in explanation
    
    def test_generate_relevance_explanation_good_score(self):
        """Test relevance explanation for good similarity score."""
        explanation = self.search_engine._generate_relevance_explanation(0.75, "test query")
        
        assert "Good match" in explanation
        assert "test query" in explanation
    
    def test_generate_relevance_explanation_moderate_score(self):
        """Test relevance explanation for moderate similarity score."""
        explanation = self.search_engine._generate_relevance_explanation(0.6, "test query")
        
        assert "Moderate match" in explanation
        assert "test query" in explanation
    
    def test_generate_relevance_explanation_weak_score(self):
        """Test relevance explanation for weak similarity score."""
        explanation = self.search_engine._generate_relevance_explanation(0.3, "test query")
        
        assert "Weak match" in explanation
        assert "test query" in explanation
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_search_similar_chunks_success(self):
        """Test successful similarity search with context."""
        # First store some test chunks
        test_chunks = [
            {
                'chunk_id': 'search_chunk_0',
                'text': 'Machine learning algorithms learn from data.',
                'embedding': [0.1, 0.2, 0.3] * 512,
                'source_document': 'ml_guide.txt',
                'chunk_index': 0,
                'token_count': 8
            }
        ]
        
        self.search_engine.vector_database.store_chunks(test_chunks)
        
        # Mock embedding generation for query
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.15, 0.25, 0.35] * 512  # Similar to stored chunk
        mock_response.usage.total_tokens = 5
        
        with patch.object(self.search_engine.embedding_manager.client.embeddings, 'create', return_value=mock_response):
            results = self.search_engine.search_similar_chunks("machine learning", top_k=5, include_context=True)
            
            assert 'results' in results
            assert 'metadata' in results
            assert 'query_embedding' in results
            
            # Check metadata
            metadata = results['metadata']
            assert metadata['query'] == "machine learning"
            assert metadata['total_results'] >= 0
            assert 'search_timestamp' in metadata
            
            # Check results structure if any found
            if results['results']:
                result = results['results'][0]
                assert 'rank' in result
                assert 'chunk_id' in result
                assert 'text' in result
                assert 'similarity_score' in result
                assert 'highlighted_text' in result
                assert 'relevance_explanation' in result
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_search_similar_chunks_no_context(self):
        """Test similarity search without context highlighting."""
        # Store test chunk
        test_chunks = [
            {
                'chunk_id': 'no_context_test',
                'text': 'Test content for search.',
                'embedding': [0.5] * 1536,
                'source_document': 'test.txt'
            }
        ]
        
        self.search_engine.vector_database.store_chunks(test_chunks)
        
        # Mock embedding generation
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.5] * 1536
        mock_response.usage.total_tokens = 3
        
        with patch.object(self.search_engine.embedding_manager.client.embeddings, 'create', return_value=mock_response):
            results = self.search_engine.search_similar_chunks("test", include_context=False)
            
            # Results should not have context highlighting
            if results['results']:
                result = results['results'][0]
                assert 'highlighted_text' not in result
                assert 'relevance_explanation' not in result
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_search_by_document_success(self):
        """Test searching within a specific document."""
        # Store chunks from different documents
        test_chunks = [
            {
                'chunk_id': 'doc1_chunk0',
                'text': 'Content from document 1',
                'embedding': [0.1] * 1536,
                'source_document': 'doc1.txt'
            },
            {
                'chunk_id': 'doc2_chunk0',
                'text': 'Content from document 2',
                'embedding': [0.2] * 1536,
                'source_document': 'doc2.txt'
            }
        ]
        
        self.search_engine.vector_database.store_chunks(test_chunks)
        
        # Mock embedding generation
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.15] * 1536
        mock_response.usage.total_tokens = 3
        
        with patch.object(self.search_engine.embedding_manager.client.embeddings, 'create', return_value=mock_response):
            results = self.search_engine.search_by_document("content", "doc1.txt", top_k=5)
            
            assert 'results' in results
            assert 'metadata' in results
            
            # All results should be from doc1.txt
            for result in results['results']:
                assert result['source_document'] == 'doc1.txt'
            
            # Metadata should indicate filtering
            assert results['metadata']['filtered_by_document'] == 'doc1.txt'
    
    def test_get_search_suggestions_with_documents(self):
        """Test search suggestions based on existing documents."""
        # Store some chunks to create document distribution
        test_chunks = [
            {
                'chunk_id': 'suggest_test_0',
                'text': 'Machine learning content',
                'embedding': [0.1] * 1536,
                'source_document': 'machine_learning_guide.txt'
            },
            {
                'chunk_id': 'suggest_test_1',
                'text': 'Deep learning content',
                'embedding': [0.2] * 1536,
                'source_document': 'deep_learning_basics.txt'
            }
        ]
        
        self.search_engine.vector_database.store_chunks(test_chunks)
        
        suggestions = self.search_engine.get_search_suggestions("machine", max_suggestions=5)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
        
        # Should include document-based suggestion
        document_suggestions = [s for s in suggestions if 'machine_learning_guide.txt' in s]
        assert len(document_suggestions) > 0
    
    def test_get_search_suggestions_generic(self):
        """Test generic search suggestions."""
        suggestions = self.search_engine.get_search_suggestions("python", max_suggestions=3)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
        
        # Should include generic suggestions
        generic_suggestions = [s for s in suggestions if 'what is python' in s or 'how does python' in s]
        assert len(generic_suggestions) > 0
    
    def test_get_search_suggestions_short_query(self):
        """Test search suggestions with short query."""
        suggestions = self.search_engine.get_search_suggestions("ai", max_suggestions=5)
        
        # Should still return some suggestions even for short queries
        assert isinstance(suggestions, list)
    
    def test_analyze_search_performance_no_results(self):
        """Test performance analysis with no results."""
        search_results = {
            'results': [],
            'metadata': {
                'query': 'test query',
                'total_results': 0,
                'top_k_requested': 5,
                'average_similarity': 0,
                'max_similarity': 0,
                'min_similarity': 0
            }
        }
        
        analysis = self.search_engine.analyze_search_performance(search_results)
        
        assert analysis['quality_score'] == 0.0
        assert analysis['coverage_score'] == 0.0
        assert analysis['diversity_score'] == 0.0
        assert analysis['overall_score'] == 0.0
        assert 'No results found' in analysis['recommendations'][0]
    
    def test_analyze_search_performance_with_results(self):
        """Test performance analysis with good results."""
        search_results = {
            'results': [
                {
                    'rank': 1,
                    'similarity_score': 0.9,
                    'source_document': 'doc1.txt'
                },
                {
                    'rank': 2,
                    'similarity_score': 0.8,
                    'source_document': 'doc2.txt'
                },
                {
                    'rank': 3,
                    'similarity_score': 0.7,
                    'source_document': 'doc1.txt'
                }
            ],
            'metadata': {
                'query': 'test query',
                'total_results': 3,
                'top_k_requested': 5,
                'average_similarity': 0.8,
                'max_similarity': 0.9,
                'min_similarity': 0.7
            }
        }
        
        analysis = self.search_engine.analyze_search_performance(search_results)
        
        assert analysis['quality_score'] > 0.5
        assert analysis['coverage_score'] > 0.5
        assert analysis['diversity_score'] > 0.5
        assert analysis['overall_score'] > 0.5
        
        assert 'metrics' in analysis
        assert analysis['metrics']['unique_documents'] == 2
        assert analysis['metrics']['total_results'] == 3
    
    def test_analyze_search_performance_low_quality(self):
        """Test performance analysis with low quality results."""
        search_results = {
            'results': [
                {
                    'rank': 1,
                    'similarity_score': 0.3,
                    'source_document': 'doc1.txt'
                }
            ],
            'metadata': {
                'query': 'test query',
                'total_results': 1,
                'top_k_requested': 5,
                'average_similarity': 0.3,
                'max_similarity': 0.3,
                'min_similarity': 0.3
            }
        }
        
        analysis = self.search_engine.analyze_search_performance(search_results)
        
        assert analysis['quality_score'] < 0.5
        assert 'more specific search terms' in ' '.join(analysis['recommendations'])


class TestVectorSearchEngineIntegration:
    """Integration tests for vector search engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('src.rag_pipeline.config.CHROMA_PERSIST_DIRECTORY', self.temp_dir):
            with patch('src.rag_pipeline.config.CHROMA_COLLECTION_NAME', 'integration_search_test'):
                self.search_engine = VectorSearchEngine()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_end_to_end_search_workflow(self):
        """Test complete search workflow from storage to retrieval."""
        # Step 1: Store diverse chunks
        test_chunks = [
            {
                'chunk_id': 'ml_chunk_0',
                'text': 'Machine learning is a method of data analysis that automates analytical model building.',
                'embedding': [0.1, 0.2, 0.3] * 512,
                'source_document': 'ml_intro.txt',
                'chunk_index': 0,
                'token_count': 15,
                'character_count': 95,
                'word_count': 15
            },
            {
                'chunk_id': 'dl_chunk_0',
                'text': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers.',
                'embedding': [0.2, 0.3, 0.4] * 512,
                'source_document': 'dl_guide.txt',
                'chunk_index': 0,
                'token_count': 16,
                'character_count': 92,
                'word_count': 16
            },
            {
                'chunk_id': 'ai_chunk_0',
                'text': 'Artificial intelligence encompasses machine learning and other computational approaches.',
                'embedding': [0.3, 0.4, 0.5] * 512,
                'source_document': 'ai_overview.txt',
                'chunk_index': 0,
                'token_count': 12,
                'character_count': 85,
                'word_count': 12
            }
        ]
        
        # Store chunks
        result = self.search_engine.vector_database.store_chunks(test_chunks)
        assert result is True
        
        # Step 2: Mock embedding generation for search
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.15, 0.25, 0.35] * 512  # Similar to ML chunk
        mock_response.usage.total_tokens = 8
        
        with patch.object(self.search_engine.embedding_manager.client.embeddings, 'create', return_value=mock_response):
            # Step 3: Perform search
            search_results = self.search_engine.search_similar_chunks(
                "machine learning algorithms", 
                top_k=3, 
                similarity_threshold=0.0,
                include_context=True
            )
            
            # Step 4: Verify search results
            assert 'results' in search_results
            assert 'metadata' in search_results
            assert len(search_results['results']) <= 3
            
            # Check result structure
            if search_results['results']:
                result = search_results['results'][0]
                assert 'rank' in result
                assert 'chunk_id' in result
                assert 'text' in result
                assert 'similarity_score' in result
                assert 'highlighted_text' in result
                assert 'relevance_explanation' in result
                assert 'source_document' in result
            
            # Step 5: Analyze performance
            performance = self.search_engine.analyze_search_performance(search_results)
            assert 'quality_score' in performance
            assert 'coverage_score' in performance
            assert 'diversity_score' in performance
            assert 'overall_score' in performance
            assert 'recommendations' in performance
    
    def test_search_engine_components_integration(self):
        """Test that search engine components work together correctly."""
        # Verify that search engine has access to both embedding manager and vector database
        assert self.search_engine.embedding_manager is not None
        assert self.search_engine.vector_database is not None
        
        # Test database health
        health = self.search_engine.vector_database.health_check()
        assert health['status'] == 'healthy'
        
        # Test that components are properly initialized
        assert self.search_engine.vector_database.collection is not None
        assert self.search_engine.embedding_manager.client is not None
    
    @patch('src.rag_pipeline.config.OPENAI_API_KEY', 'sk-test-key')
    def test_search_with_various_thresholds(self):
        """Test search behavior with different similarity thresholds."""
        # Store test chunk
        test_chunks = [
            {
                'chunk_id': 'threshold_test',
                'text': 'Test content for threshold experiments.',
                'embedding': [0.5] * 1536,
                'source_document': 'threshold_test.txt'
            }
        ]
        
        self.search_engine.vector_database.store_chunks(test_chunks)
        
        # Mock embedding generation
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.6] * 1536  # Somewhat similar
        mock_response.usage.total_tokens = 3
        
        with patch.object(self.search_engine.embedding_manager.client.embeddings, 'create', return_value=mock_response):
            # Test with low threshold
            results_low = self.search_engine.search_similar_chunks("test", similarity_threshold=0.1)
            
            # Test with high threshold
            results_high = self.search_engine.search_similar_chunks("test", similarity_threshold=0.9)
            
            # Low threshold should return more results than high threshold
            assert len(results_low['results']) >= len(results_high['results'])
    
    def test_error_propagation_in_search(self):
        """Test that errors are properly propagated through the search pipeline."""
        # Test with invalid query
        with pytest.raises(EmbeddingError):
            self.search_engine.search_similar_chunks("")
        
        # Test search suggestions with database issues
        suggestions = self.search_engine.get_search_suggestions("test")
        assert isinstance(suggestions, list)  # Should handle gracefully