"""
Document processing module for the RAG Dashboard System.
Handles file upload, text extraction, and document chunking.
"""

import os
import io
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# Document processing imports
import PyPDF2
import tiktoken

from config import config
from utils.logger import setup_logger, log_performance

logger = setup_logger(__name__)


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass


class TextExtractor:
    """Handles text extraction from various file formats."""
    
    def __init__(self):
        self.supported_formats = config.SUPPORTED_FILE_TYPES
        self.max_file_size = config.MAX_FILE_SIZE
        
    def validate_file(self, file_path: str, file_size: Optional[int] = None) -> bool:
        """
        Validate file format and size.
        
        Args:
            file_path: Path to the file
            file_size: Size of the file in bytes (optional)
            
        Returns:
            bool: True if file is valid
            
        Raises:
            DocumentProcessingError: If file is invalid
        """
        # Check file extension
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise DocumentProcessingError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        # Check file size
        if file_size is None and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
        
        if file_size and file_size > self.max_file_size:
            max_mb = self.max_file_size / (1024 * 1024)
            current_mb = file_size / (1024 * 1024)
            raise DocumentProcessingError(
                f"File too large: {current_mb:.1f}MB. "
                f"Maximum allowed size: {max_mb:.1f}MB"
            )
        
        return True
    
    @log_performance("PDF text extraction")
    def extract_pdf_text(self, file_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
            
        Raises:
            DocumentProcessingError: If extraction fails
        """
        try:
            text_content = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    raise DocumentProcessingError("Cannot process encrypted PDF files")
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(page_text)
                        logger.debug(f"Extracted text from page {page_num + 1}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                        continue
                
                if not text_content:
                    raise DocumentProcessingError("No text content found in PDF")
                
                full_text = "\n\n".join(text_content)
                logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
                return full_text
                
        except PyPDF2.errors.PdfReadError as e:
            raise DocumentProcessingError(f"Invalid or corrupted PDF file: {str(e)}")
        except FileNotFoundError:
            raise DocumentProcessingError(f"File not found: {file_path}")
        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract PDF text: {str(e)}")
    
    @log_performance("TXT text extraction")
    def extract_txt_text(self, file_path: str) -> str:
        """
        Extract text from TXT file with encoding detection.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            str: File content as string
            
        Raises:
            DocumentProcessingError: If extraction fails
        """
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                
                # Check for empty content after successful read
                if not content.strip():
                    raise DocumentProcessingError("File appears to be empty")
                
                logger.info(f"Successfully read TXT file with {encoding} encoding")
                logger.info(f"Extracted {len(content)} characters from TXT file")
                return content
                
            except UnicodeDecodeError:
                logger.debug(f"Failed to decode with {encoding}, trying next encoding")
                continue
            except FileNotFoundError:
                raise DocumentProcessingError(f"File not found: {file_path}")
            except DocumentProcessingError:
                # Re-raise our custom exceptions (like empty file)
                raise
            except Exception as e:
                logger.debug(f"Error with {encoding}: {str(e)}")
                continue
        
        raise DocumentProcessingError(
            f"Could not decode text file with any of the attempted encodings: "
            f"{', '.join(encodings_to_try)}"
        )
    
    def extract_text_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        """
        Extract text from file bytes (useful for uploaded files).
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename for format detection
            
        Returns:
            str: Extracted text content
            
        Raises:
            DocumentProcessingError: If extraction fails
        """
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == '.pdf':
            return self._extract_pdf_from_bytes(file_bytes)
        elif file_ext == '.txt':
            return self._extract_txt_from_bytes(file_bytes)
        else:
            raise DocumentProcessingError(f"Unsupported file format: {file_ext}")
    
    def _extract_pdf_from_bytes(self, file_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            pdf_file = io.BytesIO(file_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            if pdf_reader.is_encrypted:
                raise DocumentProcessingError("Cannot process encrypted PDF files")
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                    continue
            
            if not text_content:
                raise DocumentProcessingError("No text content found in PDF")
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract PDF text from bytes: {str(e)}")
    
    def _extract_txt_from_bytes(self, file_bytes: bytes) -> str:
        """Extract text from TXT bytes."""
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                content = file_bytes.decode(encoding)
                if not content.strip():
                    raise DocumentProcessingError("File appears to be empty")
                return content
            except UnicodeDecodeError:
                continue
        
        raise DocumentProcessingError(
            f"Could not decode text file with any supported encoding"
        )
    
    @log_performance("Text extraction")
    def extract_text(self, file_path: str) -> Dict[str, any]:
        """
        Extract text from a file with metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing extracted text and metadata
            
        Raises:
            DocumentProcessingError: If extraction fails
        """
        # Validate file first
        self.validate_file(file_path)
        
        file_ext = Path(file_path).suffix.lower()
        filename = Path(file_path).name
        
        try:
            if file_ext == '.pdf':
                text_content = self.extract_pdf_text(file_path)
            elif file_ext == '.txt':
                text_content = self.extract_txt_text(file_path)
            else:
                raise DocumentProcessingError(f"Unsupported file format: {file_ext}")
            
            # Get file metadata
            file_size = os.path.getsize(file_path)
            
            result = {
                'filename': filename,
                'file_path': file_path,
                'file_type': file_ext,
                'file_size': file_size,
                'text_content': text_content,
                'character_count': len(text_content),
                'word_count': len(text_content.split()),
                'extraction_successful': True
            }
            
            logger.info(f"Successfully extracted text from {filename}: "
                       f"{result['character_count']} chars, {result['word_count']} words")
            
            return result
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"Unexpected error during text extraction: {str(e)}")


class DocumentChunker:
    """Handles document chunking with configurable size and overlap."""
    
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def split_text_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
        """
        Split text into chunks by token count.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        # Encode the entire text
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Calculate end position
            end = start + max_tokens
            
            # Get chunk tokens
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            if end >= len(tokens):
                break
            start = end - overlap_tokens
        
        return chunks
    
    def split_text_by_sentences(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
        """
        Split text into chunks by sentences, respecting token limits.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        # Simple sentence splitting (can be enhanced with NLTK if needed)
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in '.!?' and len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add remaining text as last sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Group sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed token limit
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self.count_tokens(potential_chunk) <= max_tokens:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk with current sentence
                if self.count_tokens(sentence) <= max_tokens:
                    current_chunk = sentence
                else:
                    # Sentence is too long, split it by tokens
                    sentence_chunks = self.split_text_by_tokens(sentence, max_tokens, 0)
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlap if requested
        if overlap_tokens > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks, overlap_tokens)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str], overlap_tokens: int) -> List[str]:
        """Add overlap between chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Get last tokens from previous chunk
            prev_tokens = self.encoding.encode(prev_chunk)
            if len(prev_tokens) > overlap_tokens:
                overlap_text = self.encoding.decode(prev_tokens[-overlap_tokens:])
                overlapped_chunk = overlap_text + " " + current_chunk
            else:
                overlapped_chunk = prev_chunk + " " + current_chunk
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    @log_performance("Document chunking")
    def chunk_document(self, text: str, source_info: Dict[str, any] = None) -> List[Dict[str, any]]:
        """
        Chunk a document into smaller pieces with metadata.
        
        Args:
            text: Text content to chunk
            source_info: Optional source document information
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text.strip():
            raise DocumentProcessingError("Cannot chunk empty text")
        
        # Use sentence-based chunking for better semantic coherence
        text_chunks = self.split_text_by_sentences(text, self.chunk_size, self.chunk_overlap)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_info = {
                'chunk_id': f"{source_info.get('filename', 'unknown')}_{i}" if source_info else f"chunk_{i}",
                'chunk_index': i,
                'text': chunk_text,
                'token_count': self.count_tokens(chunk_text),
                'character_count': len(chunk_text),
                'word_count': len(chunk_text.split()),
                'source_document': source_info.get('filename') if source_info else None,
                'source_path': source_info.get('file_path') if source_info else None,
                'chunk_type': 'sentence_based'
            }
            chunks.append(chunk_info)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        logger.info(f"Average tokens per chunk: {sum(c['token_count'] for c in chunks) / len(chunks):.1f}")
        
        return chunks
    
    def validate_chunks(self, chunks: List[Dict[str, any]]) -> bool:
        """
        Validate that chunks meet requirements.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            bool: True if all chunks are valid
            
        Raises:
            DocumentProcessingError: If validation fails
        """
        for i, chunk in enumerate(chunks):
            if chunk['token_count'] > self.chunk_size:
                raise DocumentProcessingError(
                    f"Chunk {i} exceeds maximum token count: {chunk['token_count']} > {self.chunk_size}"
                )
            
            if not chunk['text'].strip():
                raise DocumentProcessingError(f"Chunk {i} is empty")
        
        return True


class DocumentProcessor:
    """
    Orchestrates the complete document processing pipeline.
    Coordinates text extraction, chunking, and progress tracking.
    """
    
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.document_chunker = DocumentChunker()
        
    @log_performance("Complete document processing")
    def process_document(self, file_path: str, progress_callback=None) -> Dict[str, any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dict containing processed document information and chunks
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        try:
            # Step 1: Extract text
            if progress_callback:
                progress_callback("Extracting text from document...", 0.2)
            
            extraction_result = self.text_extractor.extract_text(file_path)
            
            # Step 2: Chunk document
            if progress_callback:
                progress_callback("Chunking document into segments...", 0.6)
            
            chunks = self.document_chunker.chunk_document(
                extraction_result['text_content'],
                source_info=extraction_result
            )
            
            # Step 3: Validate chunks
            if progress_callback:
                progress_callback("Validating chunks...", 0.8)
            
            self.document_chunker.validate_chunks(chunks)
            
            # Step 4: Compile results
            if progress_callback:
                progress_callback("Finalizing processing...", 1.0)
            
            result = {
                'document_info': extraction_result,
                'chunks': chunks,
                'processing_summary': {
                    'total_chunks': len(chunks),
                    'total_tokens': sum(chunk['token_count'] for chunk in chunks),
                    'average_tokens_per_chunk': sum(chunk['token_count'] for chunk in chunks) / len(chunks),
                    'total_characters': sum(chunk['character_count'] for chunk in chunks),
                    'total_words': sum(chunk['word_count'] for chunk in chunks),
                    'processing_successful': True
                }
            }
            
            logger.info(f"Successfully processed document {extraction_result['filename']}: "
                       f"{result['processing_summary']['total_chunks']} chunks, "
                       f"{result['processing_summary']['total_tokens']} tokens")
            
            return result
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"Unexpected error during document processing: {str(e)}")
    
    @log_performance("Batch document processing")
    def process_documents(self, file_paths: List[str], progress_callback=None) -> List[Dict[str, any]]:
        """
        Process multiple documents through the pipeline.
        
        Args:
            file_paths: List of paths to document files
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of processed document results
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        results = []
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            try:
                if progress_callback:
                    progress_callback(f"Processing file {i+1} of {total_files}: {Path(file_path).name}", 
                                    i / total_files)
                
                result = self.process_document(file_path)
                results.append(result)
                
            except DocumentProcessingError as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                # Add error result instead of failing completely
                error_result = {
                    'document_info': {
                        'filename': Path(file_path).name,
                        'file_path': file_path,
                        'extraction_successful': False,
                        'error_message': str(e)
                    },
                    'chunks': [],
                    'processing_summary': {
                        'total_chunks': 0,
                        'total_tokens': 0,
                        'processing_successful': False,
                        'error_message': str(e)
                    }
                }
                results.append(error_result)
        
        if progress_callback:
            progress_callback("Batch processing complete", 1.0)
        
        successful_count = sum(1 for r in results if r['processing_summary']['processing_successful'])
        logger.info(f"Batch processing complete: {successful_count}/{total_files} files processed successfully")
        
        return results
    
    def process_document_from_bytes(self, file_bytes: bytes, filename: str, progress_callback=None) -> Dict[str, any]:
        """
        Process a document from bytes (useful for uploaded files).
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dict containing processed document information and chunks
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        try:
            # Step 1: Extract text from bytes
            if progress_callback:
                progress_callback("Extracting text from uploaded file...", 0.2)
            
            text_content = self.text_extractor.extract_text_from_bytes(file_bytes, filename)
            
            # Create extraction result structure
            extraction_result = {
                'filename': filename,
                'file_path': None,  # No file path for bytes
                'file_type': Path(filename).suffix.lower(),
                'file_size': len(file_bytes),
                'text_content': text_content,
                'character_count': len(text_content),
                'word_count': len(text_content.split()),
                'extraction_successful': True
            }
            
            # Step 2: Chunk document
            if progress_callback:
                progress_callback("Chunking document into segments...", 0.6)
            
            chunks = self.document_chunker.chunk_document(
                text_content,
                source_info=extraction_result
            )
            
            # Step 3: Validate chunks
            if progress_callback:
                progress_callback("Validating chunks...", 0.8)
            
            self.document_chunker.validate_chunks(chunks)
            
            # Step 4: Compile results
            if progress_callback:
                progress_callback("Finalizing processing...", 1.0)
            
            result = {
                'document_info': extraction_result,
                'chunks': chunks,
                'processing_summary': {
                    'total_chunks': len(chunks),
                    'total_tokens': sum(chunk['token_count'] for chunk in chunks),
                    'average_tokens_per_chunk': sum(chunk['token_count'] for chunk in chunks) / len(chunks),
                    'total_characters': sum(chunk['character_count'] for chunk in chunks),
                    'total_words': sum(chunk['word_count'] for chunk in chunks),
                    'processing_successful': True
                }
            }
            
            logger.info(f"Successfully processed uploaded document {filename}: "
                       f"{result['processing_summary']['total_chunks']} chunks, "
                       f"{result['processing_summary']['total_tokens']} tokens")
            
            return result
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"Unexpected error during document processing from bytes: {str(e)}")
    
    def get_processing_stats(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Generate processing statistics from a list of results.
        
        Args:
            results: List of processing results
            
        Returns:
            Dict containing processing statistics
        """
        successful_results = [r for r in results if r['processing_summary']['processing_successful']]
        failed_results = [r for r in results if not r['processing_summary']['processing_successful']]
        
        if not successful_results:
            return {
                'total_documents': len(results),
                'successful_documents': 0,
                'failed_documents': len(failed_results),
                'total_chunks': 0,
                'total_tokens': 0,
                'average_chunks_per_document': 0,
                'average_tokens_per_document': 0
            }
        
        total_chunks = sum(r['processing_summary']['total_chunks'] for r in successful_results)
        total_tokens = sum(r['processing_summary']['total_tokens'] for r in successful_results)
        
        stats = {
            'total_documents': len(results),
            'successful_documents': len(successful_results),
            'failed_documents': len(failed_results),
            'total_chunks': total_chunks,
            'total_tokens': total_tokens,
            'average_chunks_per_document': total_chunks / len(successful_results),
            'average_tokens_per_document': total_tokens / len(successful_results),
            'document_types': {},
            'chunk_size_distribution': {
                'small_chunks': 0,  # < 200 tokens
                'medium_chunks': 0,  # 200-500 tokens
                'large_chunks': 0   # > 500 tokens
            }
        }
        
        # Analyze document types
        for result in successful_results:
            file_type = result['document_info']['file_type']
            stats['document_types'][file_type] = stats['document_types'].get(file_type, 0) + 1
        
        # Analyze chunk size distribution
        for result in successful_results:
            for chunk in result['chunks']:
                token_count = chunk['token_count']
                if token_count < 200:
                    stats['chunk_size_distribution']['small_chunks'] += 1
                elif token_count <= 500:
                    stats['chunk_size_distribution']['medium_chunks'] += 1
                else:
                    stats['chunk_size_distribution']['large_chunks'] += 1
        
        return stats
    
    def validate_file_before_processing(self, file_path: str) -> bool:
        """
        Validate a file before processing.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if file is valid for processing
            
        Raises:
            DocumentProcessingError: If file is invalid
        """
        return self.text_extractor.validate_file(file_path)


# Global instances
text_extractor = TextExtractor()
document_chunker = DocumentChunker()
document_processor = DocumentProcessor()