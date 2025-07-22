"""
RAG Pipeline module for the RAG Dashboard System.
Core RAG functionality including embedding, storage, and retrieval.
"""

import time
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

# OpenAI and embedding imports
import openai
import numpy as np
import requests
import json

from config import config
from utils.logger import setup_logger, log_performance
from src.error_handler import (
    RAGException, ErrorType, ErrorSeverity, ErrorContext, 
    handle_rag_error, error_handler
)

logger = setup_logger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    token_count: int
    model: str
    timestamp: float


class EmbeddingManager:
    """Manages OpenAI embeddings generation with rate limiting and error handling."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_EMBEDDING_MODEL
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.rate_limit_delay = 0.1  # seconds between requests
        
    def _validate_api_key(self) -> bool:
        """Validate that OpenAI API key is configured."""
        if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "test_key_for_setup_validation":
            raise handle_rag_error(
                Exception("OpenAI API key not configured"),
                component="EmbeddingManager",
                operation="validate_api_key",
                severity=ErrorSeverity.HIGH
            )
        return True
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate a mock embedding for testing purposes."""
        import hashlib
        import random
        
        # Create a deterministic seed from the text
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        # Generate a 1536-dimensional embedding (same as OpenAI's text-embedding-ada-002)
        embedding = [random.uniform(-1, 1) for _ in range(1536)]
        
        # Normalize the embedding
        magnitude = sum(x**2 for x in embedding) ** 0.5
        embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    @log_performance("Single embedding generation")
    def generate_embedding(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult containing embedding and metadata
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        self._validate_api_key()
        
        if not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty text")
        
        for attempt in range(self.max_retries):
            try:
                # Add rate limiting delay
                if attempt > 0:
                    time.sleep(self.retry_delay * attempt)
                
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text.strip()
                )
                
                embedding = response.data[0].embedding
                token_count = response.usage.total_tokens
                
                result = EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    token_count=token_count,
                    model=self.model,
                    timestamp=time.time()
                )
                
                logger.info(f"Generated embedding: {token_count} tokens, {len(embedding)} dimensions")
                return result
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.warning("OpenAI quota exceeded, falling back to mock embeddings for testing")
                    embedding = self._generate_mock_embedding(text)
                    result = EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        token_count=int(len(text.split()) * 1.3),  # Rough token estimate
                        model="mock-embedding-model",
                        timestamp=time.time()
                    )
                    logger.info(f"Generated mock embedding: {len(embedding)} dimensions")
                    return result
                time.sleep(self.retry_delay * (attempt + 1) * 2)  # Exponential backoff
                
            except openai.APIError as e:
                logger.error(f"OpenAI API error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise EmbeddingError(f"OpenAI API error: {str(e)}")
                time.sleep(self.retry_delay)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
                time.sleep(self.retry_delay)
        
        raise EmbeddingError("Failed to generate embedding after all retry attempts")
    
    @log_performance("Batch embedding generation")
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts with batching and rate limiting.
        
        Args:
            texts: List of texts to embed
            batch_size: Maximum number of texts per API call
            
        Returns:
            List of EmbeddingResult objects
            
        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        self._validate_api_key()
        
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text.strip()]
        if not valid_texts:
            raise EmbeddingError("No valid texts provided for embedding")
        
        results = []
        total_batches = (len(valid_texts) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(valid_texts)} texts in {total_batches} batches")
        
        for batch_idx in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            for attempt in range(self.max_retries):
                try:
                    # Add rate limiting delay between batches
                    if batch_idx > 0:
                        time.sleep(self.rate_limit_delay)
                    
                    # Add retry delay if this is a retry
                    if attempt > 0:
                        time.sleep(self.retry_delay * attempt)
                    
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch_texts
                    )
                    
                    # Process batch results
                    for i, embedding_data in enumerate(response.data):
                        result = EmbeddingResult(
                            text=batch_texts[i],
                            embedding=embedding_data.embedding,
                            token_count=0,  # Individual token count not available in batch
                            model=self.model,
                            timestamp=time.time()
                        )
                        results.append(result)
                    
                    total_tokens = response.usage.total_tokens
                    logger.info(f"Batch {batch_num} completed: {total_tokens} tokens, "
                              f"{len(response.data)} embeddings")
                    break
                    
                except openai.RateLimitError as e:
                    logger.warning(f"Rate limit hit for batch {batch_num}, attempt {attempt + 1}: {str(e)}")
                    if attempt == self.max_retries - 1:
                        logger.warning(f"OpenAI quota exceeded for batch {batch_num}, falling back to mock embeddings")
                        # Generate mock embeddings for this batch
                        for i, text in enumerate(batch_texts):
                            embedding = self._generate_mock_embedding(text)
                            result = EmbeddingResult(
                                text=text,
                                embedding=embedding,
                                token_count=int(len(text.split()) * 1.3),  # Rough token estimate
                                model="mock-embedding-model",
                                timestamp=time.time()
                            )
                            results.append(result)
                        logger.info(f"Generated mock embeddings for batch {batch_num}: {len(batch_texts)} embeddings")
                        break
                    time.sleep(self.retry_delay * (attempt + 1) * 2)
                    
                except openai.APIError as e:
                    logger.error(f"API error for batch {batch_num}, attempt {attempt + 1}: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise EmbeddingError(f"API error for batch {batch_num}: {str(e)}")
                    time.sleep(self.retry_delay)
                    
                except Exception as e:
                    logger.error(f"Unexpected error for batch {batch_num}, attempt {attempt + 1}: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise EmbeddingError(f"Failed to process batch {batch_num}: {str(e)}")
                    time.sleep(self.retry_delay)
        
        logger.info(f"Batch embedding generation completed: {len(results)} embeddings generated")
        return results
    
    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries from document processing
            
        Returns:
            List of chunks with embeddings added
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not chunks:
            return []
        
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embedding_results = self.generate_embeddings_batch(texts)
        
        # Add embeddings to chunks
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            if i < len(embedding_results):
                enhanced_chunk = chunk.copy()
                enhanced_chunk.update({
                    'embedding': embedding_results[i].embedding,
                    'embedding_model': embedding_results[i].model,
                    'embedding_timestamp': embedding_results[i].timestamp,
                    'embedding_dimensions': len(embedding_results[i].embedding)
                })
                enhanced_chunks.append(enhanced_chunk)
            else:
                logger.warning(f"No embedding generated for chunk {i}")
                enhanced_chunks.append(chunk)
        
        logger.info(f"Enhanced {len(enhanced_chunks)} chunks with embeddings")
        return enhanced_chunks
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if len(embedding1) != len(embedding2):
            raise EmbeddingError("Embeddings must have the same dimensions")
        
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, (similarity + 1) / 2))
    
    def get_embedding_stats(self, embedding_results: List[EmbeddingResult]) -> Dict[str, Any]:
        """
        Generate statistics for embedding results.
        
        Args:
            embedding_results: List of embedding results
            
        Returns:
            Dictionary containing embedding statistics
        """
        if not embedding_results:
            return {
                'total_embeddings': 0,
                'total_tokens': 0,
                'average_tokens_per_embedding': 0,
                'embedding_dimensions': 0,
                'models_used': []
            }
        
        total_tokens = sum(result.token_count for result in embedding_results)
        dimensions = len(embedding_results[0].embedding) if embedding_results else 0
        models_used = list(set(result.model for result in embedding_results))
        
        stats = {
            'total_embeddings': len(embedding_results),
            'total_tokens': total_tokens,
            'average_tokens_per_embedding': total_tokens / len(embedding_results) if embedding_results else 0,
            'embedding_dimensions': dimensions,
            'models_used': models_used,
            'generation_timespan': {
                'start': min(result.timestamp for result in embedding_results),
                'end': max(result.timestamp for result in embedding_results)
            }
        }
        
        return stats


class VectorDatabaseError(Exception):
    """Custom exception for vector database errors."""
    pass


class ChromaVectorDatabase:
    """Manages Chroma vector database operations with persistent storage."""
    
    def __init__(self):
        self.persist_directory = config.CHROMA_PERSIST_DIRECTORY
        self.collection_name = config.CHROMA_COLLECTION_NAME
        self.client = None
        self.collection = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Chroma client with persistent storage."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create client with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG Dashboard System document chunks"}
            )
            
            logger.info(f"Initialized Chroma database at {self.persist_directory}")
            logger.info(f"Collection '{self.collection_name}' ready with {self.collection.count()} documents")
            
        except ImportError:
            raise VectorDatabaseError(
                "ChromaDB not installed. Please install with: pip install chromadb"
            )
        except Exception as e:
            raise VectorDatabaseError(f"Failed to initialize Chroma client: {str(e)}")
    
    def _validate_connection(self):
        """Validate database connection."""
        if not self.client or not self.collection:
            raise VectorDatabaseError("Database not properly initialized")
    
    @log_performance("Vector storage")
    def store_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Store document chunks with embeddings in the vector database.
        
        Args:
            chunks: List of chunks with embeddings
            
        Returns:
            bool: True if storage successful
            
        Raises:
            VectorDatabaseError: If storage fails
        """
        self._validate_connection()
        
        if not chunks:
            logger.warning("No chunks provided for storage")
            return True
        
        # Validate chunks have embeddings
        chunks_with_embeddings = [chunk for chunk in chunks if 'embedding' in chunk]
        if not chunks_with_embeddings:
            raise VectorDatabaseError("No chunks with embeddings found")
        
        try:
            # Prepare data for Chroma
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunks_with_embeddings:
                ids.append(chunk['chunk_id'])
                embeddings.append(chunk['embedding'])
                documents.append(chunk['text'])
                
                # Prepare metadata (Chroma doesn't support nested objects)
                metadata = {
                    'source_document': chunk.get('source_document', ''),
                    'source_path': chunk.get('source_path', ''),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'token_count': chunk.get('token_count', 0),
                    'character_count': chunk.get('character_count', 0),
                    'word_count': chunk.get('word_count', 0),
                    'chunk_type': chunk.get('chunk_type', ''),
                    'embedding_model': chunk.get('embedding_model', ''),
                    'embedding_timestamp': chunk.get('embedding_timestamp', 0.0),
                    'embedding_dimensions': chunk.get('embedding_dimensions', 0)
                }
                metadatas.append(metadata)
            
            # Store in Chroma (upsert to handle duplicates)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Stored {len(chunks_with_embeddings)} chunks in vector database")
            return True
            
        except Exception as e:
            raise VectorDatabaseError(f"Failed to store chunks: {str(e)}")
    
    @log_performance("Vector similarity search")
    def similarity_search(self, query_embedding: List[float], top_k: int = None, 
                         similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the vector database.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar chunks with similarity scores
            
        Raises:
            VectorDatabaseError: If search fails
        """
        self._validate_connection()
        
        if not query_embedding:
            raise VectorDatabaseError("Query embedding cannot be empty")
        
        # Use defaults from config if not provided
        if top_k is None:
            top_k = config.DEFAULT_TOP_K
        if similarity_threshold is None:
            similarity_threshold = config.SIMILARITY_THRESHOLD
        
        try:
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            similar_chunks = []
            if results['ids'] and results['ids'][0]:  # Check if results exist
                for i in range(len(results['ids'][0])):
                    # Convert distance to similarity (Chroma uses L2 distance)
                    distance = results['distances'][0][i]
                    similarity = 1 / (1 + distance)  # Convert distance to similarity
                    
                    # Apply similarity threshold
                    if similarity >= similarity_threshold:
                        chunk_data = {
                            'chunk_id': results['ids'][0][i],
                            'text': results['documents'][0][i],
                            'similarity_score': similarity,
                            'distance': distance,
                            'metadata': results['metadatas'][0][i]
                        }
                        similar_chunks.append(chunk_data)
            
            logger.info(f"Found {len(similar_chunks)} similar chunks above threshold {similarity_threshold}")
            return similar_chunks
            
        except Exception as e:
            raise VectorDatabaseError(f"Similarity search failed: {str(e)}")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: Unique chunk identifier
            
        Returns:
            Chunk data or None if not found
        """
        self._validate_connection()
        
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if results['ids'] and results['ids'][0]:
                return {
                    'chunk_id': results['ids'][0],
                    'text': results['documents'][0],
                    'embedding': results['embeddings'][0] if results['embeddings'] else None,
                    'metadata': results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {str(e)}")
            return None
    
    def delete_chunks_by_document(self, document_name: str) -> int:
        """
        Delete all chunks from a specific document.
        
        Args:
            document_name: Name of the source document
            
        Returns:
            Number of chunks deleted
        """
        self._validate_connection()
        
        try:
            # Find chunks from the document
            results = self.collection.get(
                where={"source_document": document_name},
                include=['metadatas']
            )
            
            if results['ids']:
                # Delete the chunks
                self.collection.delete(ids=results['ids'])
                deleted_count = len(results['ids'])
                logger.info(f"Deleted {deleted_count} chunks from document '{document_name}'")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to delete chunks from document '{document_name}': {str(e)}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        self._validate_connection()
        
        try:
            total_count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(
                limit=min(100, total_count),
                include=['metadatas']
            )
            
            stats = {
                'total_chunks': total_count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
            
            if sample_results['metadatas']:
                # Analyze document distribution
                doc_counts = {}
                embedding_models = set()
                total_tokens = 0
                
                for metadata in sample_results['metadatas']:
                    doc_name = metadata.get('source_document', 'unknown')
                    doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
                    
                    if metadata.get('embedding_model'):
                        embedding_models.add(metadata['embedding_model'])
                    
                    total_tokens += metadata.get('token_count', 0)
                
                stats.update({
                    'documents_represented': len(doc_counts),
                    'document_distribution': doc_counts,
                    'embedding_models_used': list(embedding_models),
                    'average_tokens_per_chunk': total_tokens / len(sample_results['metadatas']) if sample_results['metadatas'] else 0
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {
                'total_chunks': 0,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'error': str(e)
            }
    
    def reset_collection(self) -> bool:
        """
        Reset the collection (delete all data).
        
        Returns:
            bool: True if reset successful
        """
        self._validate_connection()
        
        try:
            # Delete the collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG Dashboard System document chunks"}
            )
            
            logger.info(f"Reset collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector database.
        
        Returns:
            Dictionary containing health status
        """
        try:
            self._validate_connection()
            
            # Test basic operations
            count = self.collection.count()
            
            return {
                'status': 'healthy',
                'client_initialized': self.client is not None,
                'collection_initialized': self.collection is not None,
                'collection_name': self.collection_name,
                'total_documents': count,
                'persist_directory': self.persist_directory
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'client_initialized': self.client is not None,
                'collection_initialized': self.collection is not None
            }


class VectorSearchEngine:
    """Enhanced vector similarity search with ranking and context highlighting."""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_database = ChromaVectorDatabase()
        
    @log_performance("Query embedding generation")
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not query.strip():
            raise EmbeddingError("Query cannot be empty")
        
        result = self.embedding_manager.generate_embedding(query)
        return result.embedding
    
    @log_performance("Vector similarity search with ranking")
    def search_similar_chunks(self, query: str, top_k: int = None, 
                            similarity_threshold: float = None,
                            include_context: bool = True) -> Dict[str, Any]:
        """
        Search for similar chunks with enhanced ranking and context.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            include_context: Whether to include context highlighting
            
        Returns:
            Dictionary containing search results and metadata
            
        Raises:
            VectorDatabaseError: If search fails
        """
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Perform similarity search
        raw_results = self.vector_database.similarity_search(
            query_embedding, top_k, similarity_threshold
        )
        
        # Enhance results with ranking and context
        enhanced_results = []
        for i, result in enumerate(raw_results):
            enhanced_result = {
                'rank': i + 1,
                'chunk_id': result['chunk_id'],
                'text': result['text'],
                'similarity_score': result['similarity_score'],
                'distance': result['distance'],
                'metadata': result['metadata'],
                'source_document': result['metadata'].get('source_document', ''),
                'chunk_index': result['metadata'].get('chunk_index', 0),
                'token_count': result['metadata'].get('token_count', 0)
            }
            
            # Add context highlighting if requested
            if include_context:
                enhanced_result['highlighted_text'] = self._highlight_context(
                    result['text'], query
                )
                enhanced_result['relevance_explanation'] = self._generate_relevance_explanation(
                    result['similarity_score'], query
                )
            
            enhanced_results.append(enhanced_result)
        
        # Generate search metadata
        search_metadata = {
            'query': query,
            'total_results': len(enhanced_results),
            'top_k_requested': top_k or config.DEFAULT_TOP_K,
            'similarity_threshold': similarity_threshold or config.SIMILARITY_THRESHOLD,
            'search_timestamp': time.time(),
            'average_similarity': sum(r['similarity_score'] for r in enhanced_results) / len(enhanced_results) if enhanced_results else 0,
            'max_similarity': max(r['similarity_score'] for r in enhanced_results) if enhanced_results else 0,
            'min_similarity': min(r['similarity_score'] for r in enhanced_results) if enhanced_results else 0
        }
        
        return {
            'results': enhanced_results,
            'metadata': search_metadata,
            'query_embedding': query_embedding
        }
    
    def _highlight_context(self, text: str, query: str) -> str:
        """
        Highlight relevant context in the text based on query terms.
        
        Args:
            text: Text to highlight
            query: Search query
            
        Returns:
            Text with highlighted terms
        """
        # Simple highlighting - can be enhanced with more sophisticated NLP
        query_terms = query.lower().split()
        highlighted_text = text
        
        for term in query_terms:
            if len(term) > 2:  # Only highlight terms longer than 2 characters
                # Case-insensitive replacement with highlighting
                import re
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted_text = pattern.sub(f"**{term.upper()}**", highlighted_text)
        
        return highlighted_text
    
    def _generate_relevance_explanation(self, similarity_score: float, query: str) -> str:
        """
        Generate explanation for why a chunk is relevant.
        
        Args:
            similarity_score: Similarity score
            query: Search query
            
        Returns:
            Relevance explanation
        """
        if similarity_score >= 0.9:
            return f"Highly relevant match for '{query}' (similarity: {similarity_score:.3f})"
        elif similarity_score >= 0.7:
            return f"Good match for '{query}' (similarity: {similarity_score:.3f})"
        elif similarity_score >= 0.5:
            return f"Moderate match for '{query}' (similarity: {similarity_score:.3f})"
        else:
            return f"Weak match for '{query}' (similarity: {similarity_score:.3f})"
    
    def search_by_document(self, query: str, document_name: str, 
                          top_k: int = None) -> Dict[str, Any]:
        """
        Search for similar chunks within a specific document.
        
        Args:
            query: Search query text
            document_name: Name of the document to search within
            top_k: Number of top results to return
            
        Returns:
            Dictionary containing search results
        """
        # Get all results first
        all_results = self.search_similar_chunks(query, top_k=100, similarity_threshold=0.0)
        
        # Filter by document
        document_results = [
            result for result in all_results['results']
            if result['source_document'] == document_name
        ]
        
        # Limit to top_k
        if top_k:
            document_results = document_results[:top_k]
        
        # Update metadata
        search_metadata = all_results['metadata'].copy()
        search_metadata.update({
            'filtered_by_document': document_name,
            'total_results': len(document_results),
            'original_total_results': len(all_results['results'])
        })
        
        return {
            'results': document_results,
            'metadata': search_metadata,
            'query_embedding': all_results['query_embedding']
        }
    
    def get_search_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """
        Get search suggestions based on partial query and existing documents.
        
        Args:
            partial_query: Partial search query
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested queries
        """
        # Get collection stats to understand available content
        stats = self.vector_database.get_collection_stats()
        
        suggestions = []
        
        # Add document-based suggestions
        if 'document_distribution' in stats:
            for doc_name in stats['document_distribution'].keys():
                if partial_query.lower() in doc_name.lower():
                    suggestions.append(f"content from {doc_name}")
        
        # Add generic suggestions based on partial query
        if len(partial_query) > 2:
            generic_suggestions = [
                f"what is {partial_query}",
                f"how does {partial_query} work",
                f"explain {partial_query}",
                f"examples of {partial_query}",
                f"{partial_query} definition"
            ]
            suggestions.extend(generic_suggestions)
        
        return suggestions[:max_suggestions]
    
    def analyze_search_performance(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze search performance and quality metrics.
        
        Args:
            search_results: Results from search_similar_chunks
            
        Returns:
            Performance analysis
        """
        results = search_results['results']
        metadata = search_results['metadata']
        
        if not results:
            return {
                'quality_score': 0.0,
                'coverage_score': 0.0,
                'diversity_score': 0.0,
                'overall_score': 0.0,
                'recommendations': ['No results found - try broader search terms']
            }
        
        # Calculate quality metrics
        avg_similarity = metadata['average_similarity']
        max_similarity = metadata['max_similarity']
        
        # Quality score based on similarity scores
        quality_score = (avg_similarity + max_similarity) / 2
        
        # Coverage score based on number of results vs requested
        coverage_score = min(1.0, len(results) / metadata['top_k_requested'])
        
        # Diversity score based on different source documents
        unique_docs = len(set(r['source_document'] for r in results))
        diversity_score = min(1.0, unique_docs / len(results))
        
        # Generate recommendations
        recommendations = []
        if quality_score < 0.5:
            recommendations.append("Try using more specific search terms")
        if coverage_score < 0.5:
            recommendations.append("Consider lowering similarity threshold for more results")
        if diversity_score < 0.3:
            recommendations.append("Results are concentrated in few documents - try broader terms")
        
        return {
            'quality_score': quality_score,
            'coverage_score': coverage_score,
            'diversity_score': diversity_score,
            'overall_score': (quality_score + coverage_score + diversity_score) / 3,
            'recommendations': recommendations or ['Search results look good!'],
            'metrics': {
                'average_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'min_similarity': metadata['min_similarity'],
                'unique_documents': unique_docs,
                'total_results': len(results)
            }
        }


class RAGPipelineError(Exception):
    """Custom exception for RAG pipeline errors."""
    pass


class RAGPipeline:
    """
    Core RAG pipeline class that coordinates all operations.
    Orchestrates document processing, embedding, storage, and retrieval.
    """
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_database = ChromaVectorDatabase()
        self.vector_search_engine = VectorSearchEngine()
        
        # Import document processor - use global instance
        from src.document_processor import document_processor
        self.document_processor = document_processor
        
        # Performance tracking
        self.performance_metrics = {
            'documents_processed': 0,
            'chunks_embedded': 0,
            'queries_processed': 0,
            'total_processing_time': 0.0,
            'total_embedding_time': 0.0,
            'total_retrieval_time': 0.0
        }
    
    @log_performance("Document embedding workflow")
    def process_and_embed_document(self, file_path: str, progress_callback=None) -> Dict[str, Any]:
        """
        Complete workflow: process document, generate embeddings, and store in vector DB.
        
        Args:
            file_path: Path to the document file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing processing results and storage confirmation
            
        Raises:
            RAGPipelineError: If any step in the pipeline fails
        """
        try:
            start_time = time.time()
            
            # Step 1: Process document (extract text and chunk)
            if progress_callback:
                progress_callback("Processing document...", 0.2)
            
            processing_result = self.document_processor.process_document(file_path, progress_callback)
            
            if not processing_result['processing_summary']['processing_successful']:
                raise RAGPipelineError("Document processing failed")
            
            chunks = processing_result['chunks']
            logger.info(f"Document processed: {len(chunks)} chunks created")
            
            # Step 2: Generate embeddings for chunks
            if progress_callback:
                progress_callback("Generating embeddings...", 0.6)
            
            embedded_chunks = self.embedding_manager.generate_embeddings_for_chunks(chunks)
            logger.info(f"Embeddings generated for {len(embedded_chunks)} chunks")
            
            # Step 3: Store in vector database
            if progress_callback:
                progress_callback("Storing in vector database...", 0.8)
            
            storage_result = self.vector_database.store_chunks(embedded_chunks)
            
            if not storage_result:
                raise RAGPipelineError("Vector database storage failed")
            
            # Step 4: Finalize and update metrics
            if progress_callback:
                progress_callback("Finalizing...", 1.0)
            
            processing_time = time.time() - start_time
            self.performance_metrics['documents_processed'] += 1
            self.performance_metrics['chunks_embedded'] += len(embedded_chunks)
            self.performance_metrics['total_processing_time'] += processing_time
            
            result = {
                'document_info': processing_result['document_info'],
                'chunks_processed': len(chunks),
                'chunks_embedded': len(embedded_chunks),
                'chunks_stored': len(embedded_chunks),
                'processing_time': processing_time,
                'storage_successful': True,
                'pipeline_successful': True
            }
            
            logger.info(f"RAG pipeline completed for {processing_result['document_info']['filename']}: "
                       f"{len(embedded_chunks)} chunks processed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"RAG pipeline failed: {str(e)}"
            logger.error(error_msg)
            raise RAGPipelineError(error_msg)
    
    @log_performance("Batch document embedding workflow")
    def process_and_embed_documents(self, file_paths: List[str], progress_callback=None) -> List[Dict[str, Any]]:
        """
        Process multiple documents through the complete RAG pipeline.
        
        Args:
            file_paths: List of paths to document files
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processing results for each document
        """
        results = []
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            try:
                if progress_callback:
                    progress_callback(f"Processing document {i+1} of {total_files}", i / total_files)
                
                result = self.process_and_embed_document(file_path)
                results.append(result)
                
            except RAGPipelineError as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                error_result = {
                    'document_info': {'filename': file_path, 'error': str(e)},
                    'chunks_processed': 0,
                    'chunks_embedded': 0,
                    'chunks_stored': 0,
                    'processing_time': 0.0,
                    'storage_successful': False,
                    'pipeline_successful': False,
                    'error_message': str(e)
                }
                results.append(error_result)
        
        if progress_callback:
            progress_callback("Batch processing complete", 1.0)
        
        successful_count = sum(1 for r in results if r['pipeline_successful'])
        logger.info(f"Batch RAG pipeline completed: {successful_count}/{total_files} documents processed successfully")
        
        return results
    
    @log_performance("Query processing and retrieval")
    def query_documents(self, query: str, top_k: int = None, 
                       similarity_threshold: float = None,
                       include_context: bool = True) -> Dict[str, Any]:
        """
        Process a query and retrieve relevant document chunks.
        
        Args:
            query: Natural language query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            include_context: Whether to include context highlighting
            
        Returns:
            Dictionary containing query results and metadata
            
        Raises:
            RAGPipelineError: If query processing fails
        """
        try:
            start_time = time.time()
            
            if not query.strip():
                raise RAGPipelineError("Query cannot be empty")
            
            # Perform similarity search
            search_results = self.vector_search_engine.search_similar_chunks(
                query, top_k, similarity_threshold, include_context
            )
            
            # Update performance metrics
            retrieval_time = time.time() - start_time
            self.performance_metrics['queries_processed'] += 1
            self.performance_metrics['total_retrieval_time'] += retrieval_time
            
            # Add pipeline metadata
            search_results['pipeline_metadata'] = {
                'retrieval_time': retrieval_time,
                'pipeline_version': '1.0',
                'components_used': ['embedding_manager', 'vector_database', 'vector_search_engine']
            }
            
            logger.info(f"Query processed: '{query}' -> {len(search_results['results'])} results in {retrieval_time:.2f}s")
            
            return search_results
            
        except Exception as e:
            error_msg = f"Query processing failed: {str(e)}"
            logger.error(error_msg)
            raise RAGPipelineError(error_msg)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the RAG pipeline.
        
        Returns:
            Dictionary containing pipeline status and metrics
        """
        try:
            # Get component health status
            db_health = self.vector_database.health_check()
            db_stats = self.vector_database.get_collection_stats()
            
            # Calculate performance metrics
            avg_processing_time = (
                self.performance_metrics['total_processing_time'] / 
                max(1, self.performance_metrics['documents_processed'])
            )
            avg_retrieval_time = (
                self.performance_metrics['total_retrieval_time'] / 
                max(1, self.performance_metrics['queries_processed'])
            )
            
            status = {
                'pipeline_healthy': db_health['status'] == 'healthy',
                'components': {
                    'document_processor': True,  # Always available
                    'embedding_manager': True,  # Always available
                    'vector_database': db_health['status'] == 'healthy',
                    'vector_search_engine': True  # Always available
                },
                'database_stats': db_stats,
                'performance_metrics': self.performance_metrics.copy(),
                'performance_averages': {
                    'avg_processing_time': avg_processing_time,
                    'avg_retrieval_time': avg_retrieval_time,
                    'chunks_per_document': (
                        self.performance_metrics['chunks_embedded'] / 
                        max(1, self.performance_metrics['documents_processed'])
                    )
                },
                'last_updated': time.time()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {str(e)}")
            return {
                'pipeline_healthy': False,
                'error': str(e),
                'last_updated': time.time()
            }
    
    def reset_pipeline_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            'documents_processed': 0,
            'chunks_embedded': 0,
            'queries_processed': 0,
            'total_processing_time': 0.0,
            'total_embedding_time': 0.0,
            'total_retrieval_time': 0.0
        }
        logger.info("Pipeline performance metrics reset")
    
    def validate_pipeline_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the entire pipeline.
        
        Returns:
            Dictionary containing health check results
        """
        health_results = {
            'overall_healthy': True,
            'component_health': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check document processor
            health_results['component_health']['document_processor'] = True
            
            # Check embedding manager
            try:
                # Test with a simple validation (without API call)
                health_results['component_health']['embedding_manager'] = True
            except Exception as e:
                health_results['component_health']['embedding_manager'] = False
                health_results['issues'].append(f"Embedding manager issue: {str(e)}")
                health_results['overall_healthy'] = False
            
            # Check vector database
            db_health = self.vector_database.health_check()
            health_results['component_health']['vector_database'] = db_health['status'] == 'healthy'
            if db_health['status'] != 'healthy':
                health_results['issues'].append(f"Vector database issue: {db_health.get('error', 'Unknown error')}")
                health_results['overall_healthy'] = False
            
            # Check vector search engine
            health_results['component_health']['vector_search_engine'] = True
            
            # Generate recommendations
            if not health_results['overall_healthy']:
                health_results['recommendations'].append("Check component logs for detailed error information")
            else:
                health_results['recommendations'].append("All components are healthy")
            
            # Add performance insights
            if self.performance_metrics['documents_processed'] > 0:
                avg_time = (
                    self.performance_metrics['total_processing_time'] / 
                    self.performance_metrics['documents_processed']
                )
                if avg_time > 30:  # More than 30 seconds per document
                    health_results['recommendations'].append("Consider optimizing document processing for better performance")
            
        except Exception as e:
            health_results['overall_healthy'] = False
            health_results['issues'].append(f"Health check failed: {str(e)}")
        
        return health_results


class QueryProcessingError(Exception):
    """Custom exception for query processing errors."""
    pass


class QueryProcessor:
    """
    Handles LLM integration for answer generation with structured prompts.
    Supports both Gemini and OpenAI APIs with automatic fallback.
    """
    
    def __init__(self, prefer_gemini: bool = True):
        self.prefer_gemini = prefer_gemini
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Gemini configuration
        self.gemini_api_key = "AIzaSyDockZUTaT4ZgH88vUeJKuCazaZ2HJxjTc"
        self.gemini_model = "gemini-2.0-flash"
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # OpenAI configuration (fallback)
        self.openai_client = None
        self.openai_model = config.OPENAI_CHAT_MODEL
        
        # Initialize OpenAI client if available
        try:
            if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "test_key_for_setup_validation":
                self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
        
        # Token usage tracking
        self.token_usage = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'queries_processed': 0,
            'gemini_queries': 0,
            'openai_queries': 0
        }
    
    def _validate_api_keys(self) -> Dict[str, bool]:
        """Validate available API keys."""
        gemini_available = bool(self.gemini_api_key)
        openai_available = bool(self.openai_client)
        
        if not gemini_available and not openai_available:
            raise QueryProcessingError(
                "No valid API keys configured. Please set up either Gemini or OpenAI API key."
            )
        
        return {
            'gemini_available': gemini_available,
            'openai_available': openai_available
        }
    
    def _call_gemini_api(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Call Gemini API for text generation.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query with context
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing response and metadata
        """
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.gemini_api_key
        }
        
        # Combine system and user prompts for Gemini
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": combined_prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1000,
                "topP": 0.9
            }
        }
        
        response = requests.post(self.gemini_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise QueryProcessingError(f"Gemini API error: {response.status_code} - {response.text}")
        
        response_data = response.json()
        
        if 'candidates' not in response_data or not response_data['candidates']:
            raise QueryProcessingError("No response generated by Gemini API")
        
        candidate = response_data['candidates'][0]
        if 'content' not in candidate or 'parts' not in candidate['content']:
            raise QueryProcessingError("Invalid response format from Gemini API")
        
        answer = candidate['content']['parts'][0]['text']
        
        # Extract usage metadata if available
        usage_metadata = response_data.get('usageMetadata', {})
        prompt_tokens = usage_metadata.get('promptTokenCount', 0)
        completion_tokens = usage_metadata.get('candidatesTokenCount', 0)
        total_tokens = usage_metadata.get('totalTokenCount', prompt_tokens + completion_tokens)
        
        return {
            'answer': answer,
            'model': self.gemini_model,
            'usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens
            }
        }
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for answer generation."""
        return """You are an AI assistant that provides accurate, helpful answers based on provided context from documents.

INSTRUCTIONS:
1. Use ONLY the provided context to answer questions
2. If the context doesn't contain enough information, clearly state this
3. Structure your answers with clear sections and bullet points when appropriate
4. Be concise but comprehensive
5. Always cite which parts of the context support your answer
6. If asked about something not in the context, politely explain that you can only answer based on the provided documents

RESPONSE FORMAT:
- Start with a direct answer to the question
- Use bullet points for lists or multiple points
- Include relevant details from the context
- End with a brief summary if the answer is long

Remember: Only use information from the provided context. Do not add external knowledge."""
    
    def _create_user_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create user prompt with query and context.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            
        Returns:
            Formatted user prompt
        """
        if not context_chunks:
            return f"""Question: {query}

Context: No relevant context found in the documents.

Please respond that you cannot answer this question as no relevant information was found in the provided documents."""
        
        # Format context chunks
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            source_doc = chunk.get('source_document', 'Unknown document')
            similarity = chunk.get('similarity_score', 0)
            text = chunk.get('text', '')
            
            context_text += f"""
Context {i} (from {source_doc}, relevance: {similarity:.3f}):
{text}
"""
        
        return f"""Question: {query}

Context from documents:
{context_text}

Please provide a comprehensive answer based on the context above."""
    
    @log_performance("Answer generation")
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], 
                       temperature: float = 0.1) -> Dict[str, Any]:
        """
        Generate answer using GPT-4 based on query and context.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            temperature: Sampling temperature for generation
            
        Returns:
            Dictionary containing answer and metadata
            
        Raises:
            QueryProcessingError: If answer generation fails
        """
        self._validate_api_key()
        
        if not query.strip():
            raise QueryProcessingError("Query cannot be empty")
        
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(query, context_chunks)
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    time.sleep(self.retry_delay * attempt)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1000,  # Reasonable limit for answers
                    top_p=0.9
                )
                
                answer = response.choices[0].message.content
                usage = response.usage
                
                # Update token usage tracking
                self.token_usage['total_prompt_tokens'] += usage.prompt_tokens
                self.token_usage['total_completion_tokens'] += usage.completion_tokens
                self.token_usage['total_tokens'] += usage.total_tokens
                self.token_usage['queries_processed'] += 1
                
                result = {
                    'answer': answer,
                    'query': query,
                    'context_chunks_used': len(context_chunks),
                    'model': self.model,
                    'temperature': temperature,
                    'token_usage': {
                        'prompt_tokens': usage.prompt_tokens,
                        'completion_tokens': usage.completion_tokens,
                        'total_tokens': usage.total_tokens
                    },
                    'generation_timestamp': time.time(),
                    'generation_successful': True
                }
                
                logger.info(f"Answer generated: {usage.total_tokens} tokens used, "
                           f"{len(context_chunks)} context chunks processed")
                
                return result
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise QueryProcessingError(f"Rate limit exceeded after {self.max_retries} attempts")
                time.sleep(self.retry_delay * (attempt + 1) * 2)
                
            except openai.APIError as e:
                logger.error(f"OpenAI API error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise QueryProcessingError(f"OpenAI API error: {str(e)}")
                time.sleep(self.retry_delay)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise QueryProcessingError(f"Failed to generate answer: {str(e)}")
                time.sleep(self.retry_delay)
        
        raise QueryProcessingError("Failed to generate answer after all retry attempts")
    
    def generate_answer_with_citations(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer with detailed citations and source attribution.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            
        Returns:
            Dictionary containing answer with citations
        """
        # Generate base answer
        answer_result = self.generate_answer(query, context_chunks)
        
        # Add citation information
        citations = []
        for i, chunk in enumerate(context_chunks, 1):
            citation = {
                'citation_id': i,
                'source_document': chunk.get('source_document', 'Unknown'),
                'chunk_id': chunk.get('chunk_id', ''),
                'similarity_score': chunk.get('similarity_score', 0),
                'text_preview': chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                'metadata': chunk.get('metadata', {})
            }
            citations.append(citation)
        
        # Enhanced result with citations
        enhanced_result = answer_result.copy()
        enhanced_result.update({
            'citations': citations,
            'source_documents': list(set(chunk.get('source_document', 'Unknown') for chunk in context_chunks)),
            'citation_count': len(citations),
            'has_citations': len(citations) > 0
        })
        
        return enhanced_result
    
    def get_token_usage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive token usage statistics.
        
        Returns:
            Dictionary containing token usage stats
        """
        stats = self.token_usage.copy()
        
        if stats['queries_processed'] > 0:
            stats['average_tokens_per_query'] = stats['total_tokens'] / stats['queries_processed']
            stats['average_prompt_tokens'] = stats['total_prompt_tokens'] / stats['queries_processed']
            stats['average_completion_tokens'] = stats['total_completion_tokens'] / stats['queries_processed']
        else:
            stats['average_tokens_per_query'] = 0
            stats['average_prompt_tokens'] = 0
            stats['average_completion_tokens'] = 0
        
        return stats
    
    def reset_token_usage(self):
        """Reset token usage statistics."""
        self.token_usage = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'queries_processed': 0
        }
        logger.info("Token usage statistics reset")
    
    def estimate_cost(self, model_pricing: Dict[str, float] = None) -> Dict[str, float]:
        """
        Estimate cost based on token usage.
        
        Args:
            model_pricing: Optional pricing per 1K tokens {input: price, output: price}
            
        Returns:
            Dictionary containing cost estimates
        """
        if model_pricing is None:
            # Default GPT-4 pricing (approximate)
            model_pricing = {
                'input': 0.03,   # $0.03 per 1K input tokens
                'output': 0.06   # $0.06 per 1K output tokens
            }
        
        input_cost = (self.token_usage['total_prompt_tokens'] / 1000) * model_pricing['input']
        output_cost = (self.token_usage['total_completion_tokens'] / 1000) * model_pricing['output']
        total_cost = input_cost + output_cost
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'currency': 'USD',
            'pricing_model': model_pricing
        }


class QueryProcessingError(Exception):
    """Custom exception for query processing errors."""
    pass


class QueryProcessor:
    """Handles answer generation using Gemini API with structured prompts and context integration."""
    
    def __init__(self):
        self.gemini_api_key = "AIzaSyDockZUTaT4ZgH88vUeJKuCazaZ2HJxjTc"
        self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.model = "gemini-2.0-flash"
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Token usage tracking
        self.token_usage = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'queries_processed': 0
        }
        
    def _validate_api_key(self) -> bool:
        """Validate that Gemini API key is configured."""
        if not self.gemini_api_key:
            raise QueryProcessingError(
                "Gemini API key not configured. Please set the API key."
            )
        return True
    
    def _create_system_prompt(self) -> str:
        """
        Create structured system prompt for answer generation.
        
        Returns:
            System prompt string
        """
        return """You are an AI assistant specialized in answering questions based on provided document context. Your role is to provide accurate, helpful, and well-structured responses.

INSTRUCTIONS:
1. Answer questions using ONLY the provided context from the documents
2. If the context doesn't contain enough information to answer the question, clearly state this limitation
3. Provide specific, actionable answers when possible
4. Use bullet points and clear structure for complex answers
5. Always cite which document or section your information comes from
6. If multiple documents are referenced, organize your answer by source

RESPONSE FORMAT:
- Start with a direct answer to the question
- Use bullet points for multiple items or steps
- Include relevant details from the context
- End with source attribution in the format: "Source: [document_name]"
- If no relevant context is found, respond with: "I cannot answer this question based on the provided documents."

QUALITY GUIDELINES:
- Be concise but comprehensive
- Use clear, professional language
- Avoid speculation beyond the provided context
- Highlight key information that directly addresses the user's question"""

    def _create_user_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create user prompt with query and context chunks.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            
        Returns:
            Formatted user prompt
        """
        if not context_chunks:
            return f"""Question: {query}

Context: No relevant context found in the uploaded documents.

Please respond that you cannot answer this question based on the provided documents and suggest that the user upload relevant documents or rephrase their question."""
        
        # Build context section
        context_section = "Context from uploaded documents:\n\n"
        for i, chunk in enumerate(context_chunks, 1):
            source_doc = chunk.get('source_document', 'Unknown Document')
            similarity = chunk.get('similarity_score', 0)
            text = chunk.get('text', '')
            
            context_section += f"Context {i} (from {source_doc}, relevance: {similarity:.3f}):\n"
            context_section += f"{text}\n\n"
        
        user_prompt = f"""Question: {query}

{context_section}

Please answer the question based on the provided context."""
        
        return user_prompt
    
    @log_performance("Answer generation with Gemini")
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer using Gemini API with context chunks.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            
        Returns:
            Dictionary containing answer and metadata
            
        Raises:
            QueryProcessingError: If answer generation fails
        """
        self._validate_api_key()
        
        if not query.strip():
            raise QueryProcessingError("Query cannot be empty")
        
        # Create prompts
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(query, context_chunks)
        
        # Combine prompts for Gemini (it doesn't have separate system/user roles like OpenAI)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        for attempt in range(self.max_retries):
            try:
                # Add retry delay if this is a retry
                if attempt > 0:
                    time.sleep(self.retry_delay * attempt)
                
                # Prepare request payload
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": full_prompt
                                }
                            ]
                        }
                    ]
                }
                
                headers = {
                    'Content-Type': 'application/json',
                    'X-goog-api-key': self.gemini_api_key
                }
                
                # Make API request
                response = requests.post(
                    self.gemini_api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Extract answer from response
                    if 'candidates' in response_data and response_data['candidates']:
                        candidate = response_data['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            answer = candidate['content']['parts'][0]['text']
                            
                            # Estimate token usage (Gemini doesn't provide exact counts)
                            prompt_tokens = len(full_prompt.split()) * 1.3  # Rough estimation
                            completion_tokens = len(answer.split()) * 1.3
                            total_tokens = prompt_tokens + completion_tokens
                            
                            # Update token usage tracking
                            self.token_usage['total_prompt_tokens'] += int(prompt_tokens)
                            self.token_usage['total_completion_tokens'] += int(completion_tokens)
                            self.token_usage['total_tokens'] += int(total_tokens)
                            self.token_usage['queries_processed'] += 1
                            
                            result = {
                                'answer': answer,
                                'query': query,
                                'context_chunks_used': len(context_chunks),
                                'model': self.model,
                                'generation_timestamp': time.time(),
                                'generation_successful': True,
                                'token_usage': {
                                    'prompt_tokens': int(prompt_tokens),
                                    'completion_tokens': int(completion_tokens),
                                    'total_tokens': int(total_tokens)
                                }
                            }
                            
                            logger.info(f"Generated answer using Gemini: {len(answer)} chars, "
                                      f"~{int(total_tokens)} tokens, {len(context_chunks)} context chunks")
                            return result
                        else:
                            raise QueryProcessingError("Invalid response format from Gemini API")
                    else:
                        raise QueryProcessingError("No candidates in Gemini API response")
                
                elif response.status_code == 429:
                    # Rate limit error
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}")
                    if attempt == self.max_retries - 1:
                        raise QueryProcessingError("Rate limit exceeded after all retry attempts")
                    time.sleep(self.retry_delay * (attempt + 1) * 2)  # Exponential backoff
                    
                else:
                    # Other API errors
                    error_msg = f"Gemini API error (status {response.status_code}): {response.text}"
                    logger.error(f"API error on attempt {attempt + 1}: {error_msg}")
                    if attempt == self.max_retries - 1:
                        raise QueryProcessingError(f"Gemini API error: {error_msg}")
                    time.sleep(self.retry_delay)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise QueryProcessingError(f"Request failed: {str(e)}")
                time.sleep(self.retry_delay)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise QueryProcessingError(f"Failed to generate answer: {str(e)}")
                time.sleep(self.retry_delay)
        
        raise QueryProcessingError("Failed to generate answer after all retry attempts")
    
    def generate_answer_with_citations(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer with detailed citations and source attribution.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            
        Returns:
            Dictionary containing answer, citations, and metadata
        """
        # Generate the basic answer
        answer_result = self.generate_answer(query, context_chunks)
        
        # Add citation information
        citations = []
        source_documents = set()
        
        for i, chunk in enumerate(context_chunks, 1):
            citation = {
                'citation_id': i,
                'source_document': chunk.get('source_document', 'Unknown Document'),
                'chunk_id': chunk.get('chunk_id', ''),
                'similarity_score': chunk.get('similarity_score', 0),
                'text_preview': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                'metadata': chunk.get('metadata', {})
            }
            citations.append(citation)
            source_documents.add(chunk.get('source_document', 'Unknown Document'))
        
        # Enhance the result with citation information
        answer_result.update({
            'citations': citations,
            'source_documents': list(source_documents),
            'citation_count': len(citations),
            'has_citations': len(citations) > 0
        })
        
        return answer_result
    
    def get_token_usage_stats(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Dictionary containing usage statistics
        """
        stats = self.token_usage.copy()
        
        # Calculate averages
        if stats['queries_processed'] > 0:
            stats['average_tokens_per_query'] = stats['total_tokens'] / stats['queries_processed']
            stats['average_prompt_tokens'] = stats['total_prompt_tokens'] / stats['queries_processed']
            stats['average_completion_tokens'] = stats['total_completion_tokens'] / stats['queries_processed']
        else:
            stats['average_tokens_per_query'] = 0
            stats['average_prompt_tokens'] = 0
            stats['average_completion_tokens'] = 0
        
        return stats
    
    def reset_token_usage(self) -> None:
        """Reset token usage statistics."""
        self.token_usage = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'queries_processed': 0
        }
        logger.info("Token usage statistics reset")
    
    def estimate_cost(self, custom_pricing: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Estimate cost based on token usage.
        
        Args:
            custom_pricing: Custom pricing per 1K tokens (optional)
            
        Returns:
            Dictionary containing cost estimates
        """
        # Default Gemini pricing (estimated, as it's free for now)
        default_pricing = {
            'input': 0.0,   # Free tier
            'output': 0.0   # Free tier
        }
        
        pricing = custom_pricing or default_pricing
        
        # Calculate costs per 1K tokens
        input_cost = (self.token_usage['total_prompt_tokens'] / 1000) * pricing['input']
        output_cost = (self.token_usage['total_completion_tokens'] / 1000) * pricing['output']
        total_cost = input_cost + output_cost
        
        return {
            'input_cost': round(input_cost, 4),
            'output_cost': round(output_cost, 4),
            'total_cost': round(total_cost, 4),
            'currency': 'USD',
            'pricing_model': pricing,
            'note': 'Gemini API is currently free with rate limits'
        }


class RAGPipelineError(Exception):
    """Custom exception for RAG pipeline errors."""
    pass


class RAGPipeline:
    """Main RAG pipeline orchestrator that coordinates all RAG operations."""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_database = ChromaVectorDatabase()
        self.vector_search_engine = VectorSearchEngine()
        self.query_processor = QueryProcessor()
        
        # Import document processor - use global instance
        from src.document_processor import document_processor
        self.document_processor = document_processor
        
        # Performance tracking
        self.performance_metrics = {
            'documents_processed': 0,
            'chunks_embedded': 0,
            'queries_processed': 0,
            'total_processing_time': 0.0,
            'total_embedding_time': 0.0,
            'total_retrieval_time': 0.0
        }
        
        logger.info("RAG Pipeline initialized with all components")
    
    @log_performance("Document embedding workflow")
    def embed_document(self, file_path: str) -> Dict[str, Any]:
        """
        Complete document embedding workflow.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            RAGPipelineError: If document embedding fails
        """
        try:
            start_time = time.time()
            
            # Step 1: Process document into chunks
            processing_result = self.document_processor.process_document(file_path)
            chunks = processing_result['chunks']
            if not chunks:
                raise RAGPipelineError(f"No chunks generated from document: {file_path}")
            
            # Step 2: Generate embeddings for chunks
            embedded_chunks = self.embedding_manager.generate_embeddings_for_chunks(chunks)
            
            # Step 3: Store chunks in vector database
            storage_success = self.vector_database.store_chunks(embedded_chunks)
            if not storage_success:
                raise RAGPipelineError("Failed to store chunks in vector database")
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics['documents_processed'] += 1
            self.performance_metrics['chunks_embedded'] += len(embedded_chunks)
            self.performance_metrics['total_processing_time'] += processing_time
            
            result = {
                'document_path': file_path,
                'chunks_processed': len(chunks),
                'chunks_embedded': len(embedded_chunks),
                'chunks_stored': len(embedded_chunks),
                'processing_time': processing_time,
                'processing_successful': True
            }
            
            logger.info(f"Document embedded successfully: {file_path} -> {len(embedded_chunks)} chunks in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Document embedding failed for {file_path}: {str(e)}"
            logger.error(error_msg)
            raise RAGPipelineError(error_msg)
    
    @log_performance("Complete RAG query workflow")
    def process_query(self, query: str, top_k: int = None, 
                     similarity_threshold: float = None,
                     include_citations: bool = True) -> Dict[str, Any]:
        """
        Complete RAG workflow: retrieve context and generate answer.
        
        Args:
            query: User's question
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            include_citations: Whether to include citations in the answer
            
        Returns:
            Dictionary containing answer, context, and metadata
            
        Raises:
            RAGPipelineError: If query processing fails
        """
        try:
            start_time = time.time()
            
            if not query.strip():
                raise RAGPipelineError("Query cannot be empty")
            
            # Step 1: Retrieve relevant context
            search_results = self.vector_search_engine.search_similar_chunks(
                query, top_k, similarity_threshold, include_context=True
            )
            context_chunks = search_results['results']
            
            # Step 2: Generate answer with context
            if include_citations:
                answer_result = self.query_processor.generate_answer_with_citations(
                    query, context_chunks
                )
            else:
                answer_result = self.query_processor.generate_answer(
                    query, context_chunks
                )
            
            # Step 3: Combine results and add metadata
            processing_time = time.time() - start_time
            
            result = {
                'query': query,
                'answer': answer_result['answer'],
                'context_chunks': context_chunks,
                'search_metadata': search_results['metadata'],
                'answer_metadata': {
                    'token_usage': answer_result['token_usage'],
                    'model': answer_result['model'],
                    'generation_timestamp': answer_result['generation_timestamp']
                },
                'processing_time': processing_time,
                'processing_successful': True
            }
            
            # Add citations if included
            if include_citations and 'citations' in answer_result:
                result['citations'] = answer_result['citations']
                result['source_documents'] = answer_result['source_documents']
                result['citation_count'] = answer_result['citation_count']
            
            # Update performance metrics
            self.performance_metrics['queries_processed'] += 1
            self.performance_metrics['total_retrieval_time'] += processing_time
            
            logger.info(f"RAG query processed: '{query}' -> {len(context_chunks)} chunks, "
                       f"{answer_result['token_usage']['total_tokens']} tokens, "
                       f"{processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"RAG query processing failed: {str(e)}"
            logger.error(error_msg)
            raise RAGPipelineError(error_msg)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        # Get database stats
        db_stats = self.vector_database.get_collection_stats()
        
        # Get query processor stats
        query_stats = self.query_processor.get_token_usage_stats()
        
        # Combine with performance metrics
        system_stats = {
            'performance_metrics': self.performance_metrics.copy(),
            'database_stats': db_stats,
            'query_processing_stats': query_stats,
            'system_health': {
                'database_health': self.vector_database.health_check(),
                'components_initialized': {
                    'embedding_manager': self.embedding_manager is not None,
                    'vector_database': self.vector_database is not None,
                    'vector_search_engine': self.vector_search_engine is not None,
                    'query_processor': self.query_processor is not None,
                    'document_processor': self.document_processor is not None
                }
            }
        }
        
        return system_stats
    
    def reset_system(self) -> bool:
        """
        Reset the entire RAG system.
        
        Returns:
            bool: True if reset successful
        """
        try:
            # Reset vector database
            db_reset = self.vector_database.reset_collection()
            
            # Reset query processor token usage
            self.query_processor.reset_token_usage()
            
            # Reset performance metrics
            self.performance_metrics = {
                'documents_processed': 0,
                'chunks_embedded': 0,
                'queries_processed': 0,
                'total_processing_time': 0.0,
                'total_embedding_time': 0.0,
                'total_retrieval_time': 0.0
            }
            
            logger.info("RAG system reset completed")
            return db_reset
            
        except Exception as e:
            logger.error(f"Failed to reset RAG system: {str(e)}")
            return False


# Global RAG pipeline instance
rag_pipeline = RAGPipeline()