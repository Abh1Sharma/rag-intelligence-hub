"""
Integration tests for the complete document processing pipeline.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock

from src.document_processor import DocumentProcessor, DocumentProcessingError


class TestDocumentProcessor:
    """Test cases for DocumentProcessor orchestrator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
        self.test_txt_content = """
        This is a comprehensive test document for the RAG Dashboard System.
        It contains multiple sentences and paragraphs to test the complete processing pipeline.
        
        The document processor should extract this text, chunk it appropriately,
        and provide detailed metadata about the processing results.
        
        This ensures that the entire workflow from text extraction to chunking works correctly.
        """
        
    def test_process_document_success(self):
        """Test successful complete document processing."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(self.test_txt_content)
            tmp_path = tmp.name
        
        try:
            result = self.processor.process_document(tmp_path)
            
            # Verify document info
            assert result['document_info']['filename'] == Path(tmp_path).name
            assert result['document_info']['extraction_successful'] is True
            assert result['document_info']['character_count'] > 0
            assert result['document_info']['word_count'] > 0
            
            # Verify chunks
            assert len(result['chunks']) > 0
            for chunk in result['chunks']:
                assert 'chunk_id' in chunk
                assert 'text' in chunk
                assert 'token_count' in chunk
                assert chunk['source_document'] == Path(tmp_path).name
                
            # Verify processing summary
            assert result['processing_summary']['processing_successful'] is True
            assert result['processing_summary']['total_chunks'] == len(result['chunks'])
            assert result['processing_summary']['total_tokens'] > 0
            
        finally:
            os.unlink(tmp_path)
    
    def test_process_document_with_progress_callback(self):
        """Test document processing with progress tracking."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(self.test_txt_content)
            tmp_path = tmp.name
        
        # Mock progress callback
        progress_callback = Mock()
        
        try:
            result = self.processor.process_document(tmp_path, progress_callback)
            
            # Verify processing was successful
            assert result['processing_summary']['processing_successful'] is True
            
            # Verify progress callback was called
            assert progress_callback.call_count >= 4  # At least 4 progress updates
            
            # Check some of the progress calls
            progress_calls = [call[0] for call in progress_callback.call_args_list]
            assert any("Extracting text" in call[0] for call in progress_calls)
            assert any("Chunking document" in call[0] for call in progress_calls)
            assert any("Validating chunks" in call[0] for call in progress_calls)
            assert any("Finalizing" in call[0] for call in progress_calls)
            
        finally:
            os.unlink(tmp_path)
    
    def test_process_document_file_not_found(self):
        """Test processing with non-existent file."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.processor.process_document("/nonexistent/file.txt")
        assert "File not found" in str(exc_info.value)
    
    def test_process_document_from_bytes_success(self):
        """Test processing document from bytes."""
        content_bytes = self.test_txt_content.encode('utf-8')
        filename = "test_document.txt"
        
        result = self.processor.process_document_from_bytes(content_bytes, filename)
        
        # Verify document info
        assert result['document_info']['filename'] == filename
        assert result['document_info']['file_path'] is None  # No file path for bytes
        assert result['document_info']['extraction_successful'] is True
        assert result['document_info']['file_size'] == len(content_bytes)
        
        # Verify chunks
        assert len(result['chunks']) > 0
        for chunk in result['chunks']:
            assert chunk['source_document'] == filename
            
        # Verify processing summary
        assert result['processing_summary']['processing_successful'] is True
    
    def test_process_document_from_bytes_with_progress(self):
        """Test bytes processing with progress callback."""
        content_bytes = self.test_txt_content.encode('utf-8')
        filename = "test_document.txt"
        progress_callback = Mock()
        
        result = self.processor.process_document_from_bytes(content_bytes, filename, progress_callback)
        
        assert result['processing_summary']['processing_successful'] is True
        assert progress_callback.call_count >= 4
    
    def test_process_documents_batch_success(self):
        """Test batch processing of multiple documents."""
        # Create multiple temporary files
        file_paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                tmp.write(f"Test document {i+1}. {self.test_txt_content}")
                file_paths.append(tmp.name)
        
        try:
            results = self.processor.process_documents(file_paths)
            
            assert len(results) == 3
            
            for i, result in enumerate(results):
                assert result['processing_summary']['processing_successful'] is True
                assert result['document_info']['filename'] == Path(file_paths[i]).name
                assert len(result['chunks']) > 0
                
        finally:
            for file_path in file_paths:
                os.unlink(file_path)
    
    def test_process_documents_batch_with_errors(self):
        """Test batch processing with some failed files."""
        # Create mix of valid and invalid files
        valid_file_paths = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                tmp.write(f"Valid document {i+1}. {self.test_txt_content}")
                valid_file_paths.append(tmp.name)
        
        # Add non-existent file
        file_paths = valid_file_paths + ["/nonexistent/file.txt"]
        
        try:
            results = self.processor.process_documents(file_paths)
            
            assert len(results) == 3
            
            # First two should be successful
            for i in range(2):
                assert results[i]['processing_summary']['processing_successful'] is True
                
            # Last one should have failed
            assert results[2]['processing_summary']['processing_successful'] is False
            assert 'error_message' in results[2]['processing_summary']
            
        finally:
            for file_path in valid_file_paths:
                os.unlink(file_path)
    
    def test_process_documents_batch_with_progress(self):
        """Test batch processing with progress callback."""
        # Create temporary files
        file_paths = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                tmp.write(f"Test document {i+1}. Content here.")
                file_paths.append(tmp.name)
        
        progress_callback = Mock()
        
        try:
            results = self.processor.process_documents(file_paths, progress_callback)
            
            assert len(results) == 2
            assert all(r['processing_summary']['processing_successful'] for r in results)
            
            # Should have progress updates for each file plus completion
            assert progress_callback.call_count >= 3
            
        finally:
            for file_path in file_paths:
                os.unlink(file_path)
    
    def test_get_processing_stats_success(self):
        """Test processing statistics generation."""
        # Create sample results
        results = [
            {
                'document_info': {'file_type': '.txt'},
                'chunks': [
                    {'token_count': 150},
                    {'token_count': 300}
                ],
                'processing_summary': {
                    'processing_successful': True,
                    'total_chunks': 2,
                    'total_tokens': 450
                }
            },
            {
                'document_info': {'file_type': '.pdf'},
                'chunks': [
                    {'token_count': 600}
                ],
                'processing_summary': {
                    'processing_successful': True,
                    'total_chunks': 1,
                    'total_tokens': 600
                }
            },
            {
                'document_info': {'file_type': '.txt'},
                'chunks': [],
                'processing_summary': {
                    'processing_successful': False,
                    'total_chunks': 0,
                    'total_tokens': 0
                }
            }
        ]
        
        stats = self.processor.get_processing_stats(results)
        
        assert stats['total_documents'] == 3
        assert stats['successful_documents'] == 2
        assert stats['failed_documents'] == 1
        assert stats['total_chunks'] == 3
        assert stats['total_tokens'] == 1050
        assert stats['average_chunks_per_document'] == 1.5
        assert stats['average_tokens_per_document'] == 525
        
        # Check document types
        assert stats['document_types']['.txt'] == 1  # Only successful ones counted
        assert stats['document_types']['.pdf'] == 1
        
        # Check chunk size distribution
        assert stats['chunk_size_distribution']['small_chunks'] == 1  # 150 tokens
        assert stats['chunk_size_distribution']['medium_chunks'] == 1  # 300 tokens
        assert stats['chunk_size_distribution']['large_chunks'] == 1   # 600 tokens
    
    def test_get_processing_stats_no_successful_results(self):
        """Test statistics with no successful results."""
        results = [
            {
                'processing_summary': {
                    'processing_successful': False,
                    'total_chunks': 0,
                    'total_tokens': 0
                }
            }
        ]
        
        stats = self.processor.get_processing_stats(results)
        
        assert stats['total_documents'] == 1
        assert stats['successful_documents'] == 0
        assert stats['failed_documents'] == 1
        assert stats['total_chunks'] == 0
        assert stats['total_tokens'] == 0
        assert stats['average_chunks_per_document'] == 0
        assert stats['average_tokens_per_document'] == 0
    
    def test_validate_file_before_processing_success(self):
        """Test file validation before processing."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            result = self.processor.validate_file_before_processing(tmp_path)
            assert result is True
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_before_processing_invalid(self):
        """Test file validation with invalid file."""
        with pytest.raises(DocumentProcessingError):
            self.processor.validate_file_before_processing("test.docx")


class TestDocumentProcessorIntegration:
    """Integration tests for complete document processing workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
    
    def test_process_demo_document_complete_workflow(self):
        """Test complete processing workflow with demo document."""
        demo_path = "demo/sample_document.txt"
        
        if os.path.exists(demo_path):
            result = self.processor.process_document(demo_path)
            
            # Verify complete workflow
            assert result['processing_summary']['processing_successful'] is True
            assert result['document_info']['extraction_successful'] is True
            assert len(result['chunks']) > 0
            
            # Verify chunk quality
            for chunk in result['chunks']:
                assert len(chunk['text'].strip()) > 0
                assert chunk['token_count'] > 0
                assert chunk['source_document'] == 'sample_document.txt'
                assert 'RAG Dashboard System' in result['document_info']['text_content']
            
            # Test statistics generation
            stats = self.processor.get_processing_stats([result])
            assert stats['successful_documents'] == 1
            assert stats['total_chunks'] > 0
    
    def test_error_handling_throughout_pipeline(self):
        """Test error handling at different stages of the pipeline."""
        # Test with corrupted/invalid content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write("")  # Empty file
            tmp_path = tmp.name
        
        try:
            with pytest.raises(DocumentProcessingError) as exc_info:
                self.processor.process_document(tmp_path)
            assert "appears to be empty" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)
    
    def test_processing_preserves_document_structure(self):
        """Test that processing preserves important document structure."""
        structured_content = """
        Title: Test Document
        
        Section 1: Introduction
        This is the introduction section with important information.
        
        Section 2: Main Content
        This section contains the main content of the document.
        It has multiple paragraphs and detailed information.
        
        Section 3: Conclusion
        This is the conclusion section that summarizes everything.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(structured_content)
            tmp_path = tmp.name
        
        try:
            result = self.processor.process_document(tmp_path)
            
            # Verify structure is preserved in chunks
            all_chunk_text = " ".join(chunk['text'] for chunk in result['chunks'])
            
            assert "Title: Test Document" in all_chunk_text
            assert "Section 1: Introduction" in all_chunk_text
            assert "Section 2: Main Content" in all_chunk_text
            assert "Section 3: Conclusion" in all_chunk_text
            
        finally:
            os.unlink(tmp_path)
    
    def test_component_integration(self):
        """Test that all components work together correctly."""
        # Verify that processor uses the same instances as global ones
        from src.document_processor import text_extractor, document_chunker, document_processor
        
        assert isinstance(self.processor.text_extractor, type(text_extractor))
        assert isinstance(self.processor.document_chunker, type(document_chunker))
        
        # Verify configuration is consistent
        assert self.processor.document_chunker.chunk_size == document_chunker.chunk_size
        assert self.processor.text_extractor.max_file_size == text_extractor.max_file_size