"""
Unit tests for text extraction functionality.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from src.document_processor import TextExtractor, DocumentProcessingError


class TestTextExtractor:
    """Test cases for TextExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = TextExtractor()
        self.test_txt_content = "This is a test document.\nIt has multiple lines.\nFor testing purposes."
        
    def test_validate_file_valid_txt(self):
        """Test file validation for valid TXT file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            result = self.extractor.validate_file(tmp_path)
            assert result is True
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_invalid_format(self):
        """Test file validation for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.raises(DocumentProcessingError) as exc_info:
                self.extractor.validate_file(tmp_path)
            assert "Unsupported file format" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_too_large(self):
        """Test file validation for oversized file."""
        # Create a file larger than max size
        large_content = b"x" * (self.extractor.max_file_size + 1)
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(large_content)
            tmp_path = tmp.name
        
        try:
            with pytest.raises(DocumentProcessingError) as exc_info:
                self.extractor.validate_file(tmp_path)
            assert "File too large" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)
    
    def test_extract_txt_text_success(self):
        """Test successful TXT text extraction."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(self.test_txt_content)
            tmp_path = tmp.name
        
        try:
            result = self.extractor.extract_txt_text(tmp_path)
            assert result == self.test_txt_content
        finally:
            os.unlink(tmp_path)
    
    def test_extract_txt_text_empty_file(self):
        """Test TXT extraction with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write("")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(DocumentProcessingError) as exc_info:
                self.extractor.extract_txt_text(tmp_path)
            assert "appears to be empty" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)
    
    def test_extract_txt_text_file_not_found(self):
        """Test TXT extraction with non-existent file."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.extractor.extract_txt_text("/nonexistent/file.txt")
        assert "File not found" in str(exc_info.value)
    
    def test_extract_txt_from_bytes_success(self):
        """Test TXT extraction from bytes."""
        content_bytes = self.test_txt_content.encode('utf-8')
        result = self.extractor._extract_txt_from_bytes(content_bytes)
        assert result == self.test_txt_content
    
    def test_extract_txt_from_bytes_empty(self):
        """Test TXT extraction from empty bytes."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.extractor._extract_txt_from_bytes(b"")
        assert "appears to be empty" in str(exc_info.value)
    
    def test_extract_text_from_bytes_txt(self):
        """Test text extraction from TXT bytes with filename."""
        content_bytes = self.test_txt_content.encode('utf-8')
        result = self.extractor.extract_text_from_bytes(content_bytes, "test.txt")
        assert result == self.test_txt_content
    
    def test_extract_text_from_bytes_unsupported(self):
        """Test text extraction from unsupported format bytes."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.extractor.extract_text_from_bytes(b"content", "test.docx")
        assert "Unsupported file format" in str(exc_info.value)
    
    def test_extract_text_success(self):
        """Test complete text extraction with metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(self.test_txt_content)
            tmp_path = tmp.name
        
        try:
            result = self.extractor.extract_text(tmp_path)
            
            assert result['filename'] == Path(tmp_path).name
            assert result['file_path'] == tmp_path
            assert result['file_type'] == '.txt'
            assert result['text_content'] == self.test_txt_content
            assert result['character_count'] == len(self.test_txt_content)
            assert result['word_count'] == len(self.test_txt_content.split())
            assert result['extraction_successful'] is True
            assert result['file_size'] > 0
        finally:
            os.unlink(tmp_path)
    
    @patch('PyPDF2.PdfReader')
    def test_extract_pdf_text_success(self, mock_pdf_reader):
        """Test successful PDF text extraction."""
        # Mock PDF reader
        mock_page = mock_pdf_reader.return_value.pages[0]
        mock_page.extract_text.return_value = "Test PDF content"
        mock_pdf_reader.return_value.is_encrypted = False
        mock_pdf_reader.return_value.pages = [mock_page]
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"fake pdf content")
            tmp_path = tmp.name
        
        try:
            result = self.extractor.extract_pdf_text(tmp_path)
            assert result == "Test PDF content"
        finally:
            os.unlink(tmp_path)
    
    @patch('PyPDF2.PdfReader')
    def test_extract_pdf_text_encrypted(self, mock_pdf_reader):
        """Test PDF extraction with encrypted file."""
        mock_pdf_reader.return_value.is_encrypted = True
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"fake pdf content")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(DocumentProcessingError) as exc_info:
                self.extractor.extract_pdf_text(tmp_path)
            assert "encrypted PDF" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)
    
    @patch('PyPDF2.PdfReader')
    def test_extract_pdf_text_no_content(self, mock_pdf_reader):
        """Test PDF extraction with no text content."""
        mock_page = mock_pdf_reader.return_value.pages[0]
        mock_page.extract_text.return_value = ""
        mock_pdf_reader.return_value.is_encrypted = False
        mock_pdf_reader.return_value.pages = [mock_page]
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"fake pdf content")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(DocumentProcessingError) as exc_info:
                self.extractor.extract_pdf_text(tmp_path)
            assert "No text content found" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)


class TestTextExtractorIntegration:
    """Integration tests for text extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = TextExtractor()
    
    def test_extract_demo_document(self):
        """Test extraction of the demo document."""
        demo_path = "demo/sample_document.txt"
        if os.path.exists(demo_path):
            result = self.extractor.extract_text(demo_path)
            
            assert result['extraction_successful'] is True
            assert result['character_count'] > 0
            assert result['word_count'] > 0
            assert "RAG Dashboard System" in result['text_content']
    
    def test_supported_formats_configuration(self):
        """Test that supported formats are properly configured."""
        assert '.pdf' in self.extractor.supported_formats
        assert '.txt' in self.extractor.supported_formats
        assert self.extractor.max_file_size > 0