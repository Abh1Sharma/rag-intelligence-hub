"""
Unit tests for Chroma vector database integration.
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock

from src.rag_pipeline import ChromaVectorDatabase, VectorDatabaseError


class TestChromaVectorDatabase:
    """Test cases for ChromaVectorDatabase class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config to use temp directory
        with patch('src.rag_pipeline.config.CHROMA_PERSIST_DIRECTORY', self.temp_dir):
            with patch('src.rag_pipeline.config.CHROMA_COLLECTION_NAME', 'test_collection'):
                self.db = ChromaVectorDatabase()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_success(self):
        """Test successful database initialization."""
        assert self.db.client is not None
        assert self.db.collection is not None
        assert self.db.collection_name == 'test_collection'
        assert self.db.persist_directory == self.temp_dir
    
    def test_validate_connection_success(self):
        """Test connection validation with valid connection."""
        # Should not raise exception
        self.db._validate_connection()
    
    def test_validate_connection_no_client(self):
        """Test connection validation with no client."""
        self.db.client = None
        
        with pytest.raises(VectorDatabaseError) as exc_info:
            self.db._validate_connection()
        assert "not properly initialized" in str(exc_info.value)
    
    def test_store_chunks_success(self):
        """Test successful chunk storage."""
        chunks = [
            {
                'chunk_id': 'test_chunk_0',
                'text': 'This is test chunk 0',
                'embedding': [0.1, 0.2, 0.3] * 512,  # 1536 dimensions
                'source_document': 'test.txt',
                'source_path': '/path/to/test.txt',
                'chunk_index': 0,
                'token_count': 5,
                'character_count': 20,
                'word_count': 4,
                'chunk_type': 'sentence_based',
                'embedding_model': 'text-embedding-ada-002',
                'embedding_timestamp': 1000.0,
                'embedding_dimensions': 1536
            },
            {
                'chunk_id': 'test_chunk_1',
                'text': 'This is test chunk 1',
                'embedding': [0.2, 0.3, 0.4] * 512,  # 1536 dimensions
                'source_document': 'test.txt',
                'source_path': '/path/to/test.txt',
                'chunk_index': 1,
                'token_count': 5,
                'character_count': 20,
                'word_count': 4,
                'chunk_type': 'sentence_based',
                'embedding_model': 'text-embedding-ada-002',
                'embedding_timestamp': 1001.0,
                'embedding_dimensions': 1536
            }
        ]
        
        result = self.db.store_chunks(chunks)
        assert result is True
        
        # Verify chunks were stored
        stats = self.db.get_collection_stats()
        assert stats['total_chunks'] == 2
    
    def test_store_chunks_empty_list(self):
        """Test storing empty chunk list."""
        result = self.db.store_chunks([])
        assert result is True
    
    def test_store_chunks_no_embeddings(self):
        """Test storing chunks without embeddings."""
        chunks = [
            {
                'chunk_id': 'test_chunk_0',
                'text': 'This is test chunk without embedding',
                'source_document': 'test.txt'
            }
        ]
        
        with pytest.raises(VectorDatabaseError) as exc_info:
            self.db.store_chunks(chunks)
        assert "No chunks with embeddings found" in str(exc_info.value)
    
    def test_similarity_search_success(self):
        """Test successful similarity search."""
        # First store some chunks
        chunks = [
            {
                'chunk_id': 'search_test_0',
                'text': 'Machine learning is fascinating',
                'embedding': [0.1, 0.2, 0.3] * 512,
                'source_document': 'ml.txt',
                'token_count': 4
            },
            {
                'chunk_id': 'search_test_1',
                'text': 'Deep learning uses neural networks',
                'embedding': [0.2, 0.3, 0.4] * 512,
                'source_document': 'dl.txt',
                'token_count': 5
            }
        ]
        
        self.db.store_chunks(chunks)
        
        # Perform similarity search
        query_embedding = [0.15, 0.25, 0.35] * 512  # Similar to first chunk
        results = self.db.similarity_search(query_embedding, top_k=2, similarity_threshold=0.0)
        
        assert len(results) <= 2
        assert all('chunk_id' in result for result in results)
        assert all('similarity_score' in result for result in results)
        assert all('text' in result for result in results)
    
    def test_similarity_search_empty_embedding(self):
        """Test similarity search with empty embedding."""
        with pytest.raises(VectorDatabaseError) as exc_info:
            self.db.similarity_search([])
        assert "cannot be empty" in str(exc_info.value)
    
    def test_similarity_search_with_threshold(self):
        """Test similarity search with high threshold."""
        # Store a chunk
        chunks = [
            {
                'chunk_id': 'threshold_test',
                'text': 'Test content',
                'embedding': [1.0, 0.0, 0.0] * 512,
                'source_document': 'test.txt'
            }
        ]
        
        self.db.store_chunks(chunks)
        
        # Search with very different embedding and high threshold
        query_embedding = [0.0, 1.0, 0.0] * 512
        results = self.db.similarity_search(query_embedding, top_k=5, similarity_threshold=0.9)
        
        # Should return no results due to high threshold
        assert len(results) == 0
    
    def test_get_chunk_by_id_success(self):
        """Test retrieving chunk by ID."""
        # Store a chunk
        chunks = [
            {
                'chunk_id': 'retrieve_test',
                'text': 'Content to retrieve',
                'embedding': [0.5, 0.5, 0.5] * 512,
                'source_document': 'retrieve.txt'
            }
        ]
        
        self.db.store_chunks(chunks)
        
        # Retrieve the chunk
        result = self.db.get_chunk_by_id('retrieve_test')
        
        assert result is not None
        assert result['chunk_id'] == 'retrieve_test'
        assert result['text'] == 'Content to retrieve'
        assert 'embedding' in result
        assert 'metadata' in result
    
    def test_get_chunk_by_id_not_found(self):
        """Test retrieving non-existent chunk."""
        result = self.db.get_chunk_by_id('nonexistent_chunk')
        assert result is None
    
    def test_delete_chunks_by_document_success(self):
        """Test deleting chunks by document name."""
        # Store chunks from different documents
        chunks = [
            {
                'chunk_id': 'doc1_chunk0',
                'text': 'Content from document 1',
                'embedding': [0.1] * 1536,
                'source_document': 'document1.txt'
            },
            {
                'chunk_id': 'doc1_chunk1',
                'text': 'More content from document 1',
                'embedding': [0.2] * 1536,
                'source_document': 'document1.txt'
            },
            {
                'chunk_id': 'doc2_chunk0',
                'text': 'Content from document 2',
                'embedding': [0.3] * 1536,
                'source_document': 'document2.txt'
            }
        ]
        
        self.db.store_chunks(chunks)
        
        # Delete chunks from document1
        deleted_count = self.db.delete_chunks_by_document('document1.txt')
        
        assert deleted_count == 2
        
        # Verify only document2 chunks remain
        stats = self.db.get_collection_stats()
        assert stats['total_chunks'] == 1
    
    def test_delete_chunks_by_document_not_found(self):
        """Test deleting chunks from non-existent document."""
        deleted_count = self.db.delete_chunks_by_document('nonexistent.txt')
        assert deleted_count == 0
    
    def test_get_collection_stats_empty(self):
        """Test getting stats from empty collection."""
        stats = self.db.get_collection_stats()
        
        assert stats['total_chunks'] == 0
        assert stats['collection_name'] == 'test_collection'
        assert stats['persist_directory'] == self.temp_dir
    
    def test_get_collection_stats_with_data(self):
        """Test getting stats from collection with data."""
        # Store some test chunks
        chunks = [
            {
                'chunk_id': 'stats_test_0',
                'text': 'Stats test content 1',
                'embedding': [0.1] * 1536,
                'source_document': 'stats1.txt',
                'token_count': 10,
                'embedding_model': 'text-embedding-ada-002'
            },
            {
                'chunk_id': 'stats_test_1',
                'text': 'Stats test content 2',
                'embedding': [0.2] * 1536,
                'source_document': 'stats2.txt',
                'token_count': 15,
                'embedding_model': 'text-embedding-ada-002'
            }
        ]
        
        self.db.store_chunks(chunks)
        
        stats = self.db.get_collection_stats()
        
        assert stats['total_chunks'] == 2
        assert stats['documents_represented'] == 2
        assert 'stats1.txt' in stats['document_distribution']
        assert 'stats2.txt' in stats['document_distribution']
        assert 'text-embedding-ada-002' in stats['embedding_models_used']
        assert stats['average_tokens_per_chunk'] == 12.5
    
    def test_reset_collection_success(self):
        """Test resetting collection."""
        # Store some chunks
        chunks = [
            {
                'chunk_id': 'reset_test',
                'text': 'Content to be reset',
                'embedding': [0.1] * 1536,
                'source_document': 'reset.txt'
            }
        ]
        
        self.db.store_chunks(chunks)
        
        # Verify chunks exist
        assert self.db.get_collection_stats()['total_chunks'] == 1
        
        # Reset collection
        result = self.db.reset_collection()
        assert result is True
        
        # Verify collection is empty
        assert self.db.get_collection_stats()['total_chunks'] == 0
    
    def test_health_check_healthy(self):
        """Test health check with healthy database."""
        health = self.db.health_check()
        
        assert health['status'] == 'healthy'
        assert health['client_initialized'] is True
        assert health['collection_initialized'] is True
        assert health['collection_name'] == 'test_collection'
        assert health['total_documents'] == 0
        assert health['persist_directory'] == self.temp_dir
    
    def test_health_check_unhealthy(self):
        """Test health check with unhealthy database."""
        # Break the connection
        self.db.client = None
        
        health = self.db.health_check()
        
        assert health['status'] == 'unhealthy'
        assert 'error' in health
        assert health['client_initialized'] is False


class TestChromaVectorDatabaseIntegration:
    """Integration tests for Chroma vector database."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('src.rag_pipeline.config.CHROMA_PERSIST_DIRECTORY', self.temp_dir):
            with patch('src.rag_pipeline.config.CHROMA_COLLECTION_NAME', 'integration_test'):
                self.db = ChromaVectorDatabase()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_workflow_integration(self):
        """Test complete workflow from storage to retrieval."""
        # Sample chunks with realistic embeddings
        chunks = [
            {
                'chunk_id': 'workflow_0',
                'text': 'Machine learning algorithms can learn patterns from data.',
                'embedding': [0.1, 0.2, 0.3, 0.4] * 384,  # 1536 dimensions
                'source_document': 'ml_guide.txt',
                'source_path': '/docs/ml_guide.txt',
                'chunk_index': 0,
                'token_count': 10,
                'character_count': 55,
                'word_count': 9,
                'chunk_type': 'sentence_based',
                'embedding_model': 'text-embedding-ada-002',
                'embedding_timestamp': 1000.0,
                'embedding_dimensions': 1536
            },
            {
                'chunk_id': 'workflow_1',
                'text': 'Deep learning uses neural networks with multiple layers.',
                'embedding': [0.2, 0.3, 0.4, 0.5] * 384,  # 1536 dimensions
                'source_document': 'dl_basics.txt',
                'source_path': '/docs/dl_basics.txt',
                'chunk_index': 0,
                'token_count': 12,
                'character_count': 54,
                'word_count': 9,
                'chunk_type': 'sentence_based',
                'embedding_model': 'text-embedding-ada-002',
                'embedding_timestamp': 1001.0,
                'embedding_dimensions': 1536
            }
        ]
        
        # Step 1: Store chunks
        result = self.db.store_chunks(chunks)
        assert result is True
        
        # Step 2: Verify storage
        stats = self.db.get_collection_stats()
        assert stats['total_chunks'] == 2
        assert stats['documents_represented'] == 2
        
        # Step 3: Perform similarity search
        query_embedding = [0.15, 0.25, 0.35, 0.45] * 384  # Similar to first chunk
        search_results = self.db.similarity_search(query_embedding, top_k=2)
        
        assert len(search_results) <= 2
        assert all(result['similarity_score'] > 0 for result in search_results)
        
        # Step 4: Retrieve specific chunk
        specific_chunk = self.db.get_chunk_by_id('workflow_0')
        assert specific_chunk is not None
        assert specific_chunk['text'] == chunks[0]['text']
        
        # Step 5: Delete chunks from one document
        deleted_count = self.db.delete_chunks_by_document('ml_guide.txt')
        assert deleted_count == 1
        
        # Step 6: Verify deletion
        final_stats = self.db.get_collection_stats()
        assert final_stats['total_chunks'] == 1
    
    def test_persistence_across_instances(self):
        """Test that data persists across database instances."""
        # Store data in first instance
        chunks = [
            {
                'chunk_id': 'persist_test',
                'text': 'This should persist across instances',
                'embedding': [0.5] * 1536,
                'source_document': 'persist.txt'
            }
        ]
        
        self.db.store_chunks(chunks)
        
        # Create new instance with same directory
        with patch('src.rag_pipeline.config.CHROMA_PERSIST_DIRECTORY', self.temp_dir):
            with patch('src.rag_pipeline.config.CHROMA_COLLECTION_NAME', 'integration_test'):
                new_db = ChromaVectorDatabase()
        
        # Verify data persists
        stats = new_db.get_collection_stats()
        assert stats['total_chunks'] == 1
        
        retrieved_chunk = new_db.get_chunk_by_id('persist_test')
        assert retrieved_chunk is not None
        assert retrieved_chunk['text'] == 'This should persist across instances'
    
    def test_error_handling_integration(self):
        """Test error handling in integrated scenarios."""
        # Test storing invalid data
        invalid_chunks = [{'chunk_id': 'invalid', 'text': 'no embedding'}]
        
        with pytest.raises(VectorDatabaseError):
            self.db.store_chunks(invalid_chunks)
        
        # Test searching with invalid embedding
        with pytest.raises(VectorDatabaseError):
            self.db.similarity_search([])
        
        # Verify database is still functional after errors
        health = self.db.health_check()
        assert health['status'] == 'healthy'
    
    def test_large_batch_operations(self):
        """Test operations with larger batches of data."""
        # Create larger batch of chunks
        large_batch = []
        for i in range(50):
            chunk = {
                'chunk_id': f'batch_chunk_{i}',
                'text': f'This is batch chunk number {i} with some content.',
                'embedding': [0.1 * (i % 10)] * 1536,
                'source_document': f'batch_doc_{i % 5}.txt',  # 5 different documents
                'token_count': 10 + (i % 5)
            }
            large_batch.append(chunk)
        
        # Store large batch
        result = self.db.store_chunks(large_batch)
        assert result is True
        
        # Verify storage
        stats = self.db.get_collection_stats()
        assert stats['total_chunks'] == 50
        assert stats['documents_represented'] == 5
        
        # Test search with larger result set
        query_embedding = [0.05] * 1536
        results = self.db.similarity_search(query_embedding, top_k=20)
        assert len(results) <= 20
        
        # Test bulk deletion
        deleted_count = self.db.delete_chunks_by_document('batch_doc_0.txt')
        assert deleted_count == 10  # Every 5th chunk from 0, 5, 10, ..., 45
        
        final_stats = self.db.get_collection_stats()
        assert final_stats['total_chunks'] == 40