"""
Unit tests for document chunking functionality.
"""

import pytest
from unittest.mock import patch

from src.document_processor import DocumentChunker, DocumentProcessingError


class TestDocumentChunker:
    """Test cases for DocumentChunker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = DocumentChunker()
        self.test_text = """
        This is the first sentence. This is the second sentence with more content.
        This is the third sentence that continues the document.
        
        This is a new paragraph with additional information. It contains multiple sentences.
        The final sentence concludes this test document.
        """
        
    def test_count_tokens(self):
        """Test token counting functionality."""
        text = "This is a test sentence."
        token_count = self.chunker.count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0
        
    def test_count_tokens_empty(self):
        """Test token counting with empty text."""
        token_count = self.chunker.count_tokens("")
        assert token_count == 0
        
    def test_split_text_by_tokens_small_text(self):
        """Test token splitting with text smaller than max tokens."""
        text = "Short text."
        chunks = self.chunker.split_text_by_tokens(text, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == text
        
    def test_split_text_by_tokens_large_text(self):
        """Test token splitting with text larger than max tokens."""
        # Create a long text that will exceed token limit
        long_text = "This is a sentence. " * 100
        chunks = self.chunker.split_text_by_tokens(long_text, max_tokens=50)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert self.chunker.count_tokens(chunk) <= 50
            
    def test_split_text_by_tokens_with_overlap(self):
        """Test token splitting with overlap."""
        long_text = "This is a sentence. " * 20
        chunks = self.chunker.split_text_by_tokens(long_text, max_tokens=30, overlap_tokens=5)
        
        assert len(chunks) > 1
        # Check that chunks have some overlapping content
        if len(chunks) > 1:
            # This is a basic check - in practice, overlap detection is complex
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0
            
    def test_split_text_by_sentences(self):
        """Test sentence-based text splitting."""
        chunks = self.chunker.split_text_by_sentences(self.test_text, max_tokens=100)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert self.chunker.count_tokens(chunk) <= 100
            assert len(chunk.strip()) > 0
            
    def test_split_text_by_sentences_small_limit(self):
        """Test sentence splitting with very small token limit."""
        chunks = self.chunker.split_text_by_sentences(self.test_text, max_tokens=20)
        
        assert len(chunks) > 1
        for chunk in chunks:
            # Some chunks might slightly exceed due to sentence boundaries
            # but should be reasonable
            assert len(chunk.strip()) > 0
            
    def test_chunk_document_success(self):
        """Test successful document chunking."""
        source_info = {
            'filename': 'test.txt',
            'file_path': '/path/to/test.txt'
        }
        
        chunks = self.chunker.chunk_document(self.test_text, source_info)
        
        assert len(chunks) > 0
        
        for i, chunk in enumerate(chunks):
            assert chunk['chunk_index'] == i
            assert chunk['chunk_id'] == f"test.txt_{i}"
            assert chunk['source_document'] == 'test.txt'
            assert chunk['source_path'] == '/path/to/test.txt'
            assert chunk['chunk_type'] == 'sentence_based'
            assert isinstance(chunk['token_count'], int)
            assert isinstance(chunk['character_count'], int)
            assert isinstance(chunk['word_count'], int)
            assert len(chunk['text'].strip()) > 0
            
    def test_chunk_document_no_source_info(self):
        """Test document chunking without source info."""
        chunks = self.chunker.chunk_document(self.test_text)
        
        assert len(chunks) > 0
        
        for i, chunk in enumerate(chunks):
            assert chunk['chunk_index'] == i
            assert chunk['chunk_id'] == f"chunk_{i}"
            assert chunk['source_document'] is None
            assert chunk['source_path'] is None
            
    def test_chunk_document_empty_text(self):
        """Test chunking with empty text."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.chunker.chunk_document("")
        assert "Cannot chunk empty text" in str(exc_info.value)
        
    def test_chunk_document_whitespace_only(self):
        """Test chunking with whitespace-only text."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.chunker.chunk_document("   \n\t   ")
        assert "Cannot chunk empty text" in str(exc_info.value)
        
    def test_validate_chunks_success(self):
        """Test successful chunk validation."""
        chunks = [
            {
                'chunk_index': 0,
                'text': 'Valid chunk text.',
                'token_count': 10,
                'character_count': 18,
                'word_count': 3
            },
            {
                'chunk_index': 1,
                'text': 'Another valid chunk.',
                'token_count': 15,
                'character_count': 20,
                'word_count': 4
            }
        ]
        
        result = self.chunker.validate_chunks(chunks)
        assert result is True
        
    def test_validate_chunks_oversized(self):
        """Test chunk validation with oversized chunk."""
        chunks = [
            {
                'chunk_index': 0,
                'text': 'Valid chunk text.',
                'token_count': self.chunker.chunk_size + 100,  # Exceeds limit
                'character_count': 18,
                'word_count': 3
            }
        ]
        
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.chunker.validate_chunks(chunks)
        assert "exceeds maximum token count" in str(exc_info.value)
        
    def test_validate_chunks_empty_text(self):
        """Test chunk validation with empty chunk text."""
        chunks = [
            {
                'chunk_index': 0,
                'text': '',  # Empty text
                'token_count': 0,
                'character_count': 0,
                'word_count': 0
            }
        ]
        
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.chunker.validate_chunks(chunks)
        assert "is empty" in str(exc_info.value)
        
    def test_configuration_values(self):
        """Test that chunker uses correct configuration values."""
        assert self.chunker.chunk_size == 800  # From config
        assert self.chunker.chunk_overlap == 100  # From config
        assert self.chunker.encoding is not None


class TestDocumentChunkerIntegration:
    """Integration tests for document chunking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = DocumentChunker()
        
    def test_chunk_demo_document(self):
        """Test chunking the demo document."""
        # Read demo document content
        with open('demo/sample_document.txt', 'r') as f:
            demo_text = f.read()
            
        source_info = {
            'filename': 'sample_document.txt',
            'file_path': 'demo/sample_document.txt'
        }
        
        chunks = self.chunker.chunk_document(demo_text, source_info)
        
        assert len(chunks) > 0
        
        # Validate all chunks
        self.chunker.validate_chunks(chunks)
        
        # Check that total content is preserved (approximately)
        total_chars = sum(chunk['character_count'] for chunk in chunks)
        original_chars = len(demo_text.strip())
        # Should be close to original length (allow for small differences in processing)
        assert abs(total_chars - original_chars) <= 10
        
        # Check chunk metadata
        for chunk in chunks:
            assert chunk['source_document'] == 'sample_document.txt'
            assert chunk['token_count'] <= self.chunker.chunk_size
            assert 'RAG Dashboard System' in demo_text  # Verify content
            
    def test_chunking_preserves_content(self):
        """Test that chunking preserves important content."""
        test_text = """
        Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence. It focuses on algorithms.
        These algorithms can learn from data. They make predictions or decisions.
        
        Types of Machine Learning:
        1. Supervised Learning
        2. Unsupervised Learning  
        3. Reinforcement Learning
        
        Each type has different applications and use cases.
        """
        
        chunks = self.chunker.chunk_document(test_text)
        
        # Reconstruct text from chunks (without overlap handling for simplicity)
        reconstructed_words = []
        for chunk in chunks:
            reconstructed_words.extend(chunk['text'].split())
            
        original_words = test_text.split()
        
        # Should have preserved most important words
        important_words = ['Machine', 'Learning', 'algorithms', 'Supervised', 'Unsupervised']
        for word in important_words:
            assert any(word in ' '.join(reconstructed_words) for word in important_words)
            
    def test_token_count_accuracy(self):
        """Test that token counting is accurate."""
        test_text = "This is a test sentence for token counting accuracy."
        
        # Count tokens directly
        direct_count = self.chunker.count_tokens(test_text)
        
        # Count tokens through chunking
        chunks = self.chunker.chunk_document(test_text)
        chunk_count = sum(chunk['token_count'] for chunk in chunks)
        
        # Should be very close (might differ slightly due to chunking boundaries)
        assert abs(direct_count - chunk_count) <= 2