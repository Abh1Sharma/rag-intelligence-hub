"""
FastAPI backend for the RAG Dashboard System.
Provides RESTful API endpoints for document processing and query handling.
"""

import time
import os
import uuid
import shutil
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from config import config
from src.rag_pipeline import rag_pipeline
from src.error_handler import RAGException, ErrorType, ErrorSeverity, error_handler
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.APP_TITLE,
    description=config.APP_DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models for Request/Response Validation

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component health status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: bool = Field(True, description="Error flag")
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    user_message: str = Field(..., description="User-friendly error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested recovery actions")
    retry_possible: bool = Field(False, description="Whether retry is recommended")
    retry_delay: Optional[float] = Field(None, description="Recommended retry delay in seconds")


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Response message")
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File type/extension")
    chunks_processed: Optional[int] = Field(None, description="Number of chunks created")
    chunks_embedded: Optional[int] = Field(None, description="Number of chunks embedded")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")


class DocumentInfo(BaseModel):
    """Document information model."""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File type/extension")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    chunk_count: int = Field(..., description="Number of chunks")
    processing_status: str = Field(..., description="Processing status")


class DocumentListResponse(BaseModel):
    """Document list response model."""
    documents: List[DocumentInfo] = Field(..., description="List of uploaded documents")
    total_count: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks across all documents")


class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of top results to retrieve")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    include_context: bool = Field(True, description="Whether to include context in response")
    include_citations: bool = Field(True, description="Whether to include citations")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()


class ContextChunk(BaseModel):
    """Context chunk model for query responses."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    source_document: str = Field(..., description="Source document name")
    similarity_score: float = Field(..., description="Similarity score")
    chunk_index: int = Field(..., description="Chunk index in document")
    highlighted_text: Optional[str] = Field(None, description="Text with query terms highlighted")


class Citation(BaseModel):
    """Citation model for query responses."""
    citation_id: int = Field(..., description="Citation number")
    source_document: str = Field(..., description="Source document name")
    chunk_id: str = Field(..., description="Source chunk ID")
    similarity_score: float = Field(..., description="Relevance score")
    text_preview: str = Field(..., description="Preview of cited text")


class QueryResponse(BaseModel):
    """Query response model."""
    success: bool = Field(..., description="Query success status")
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original query")
    context_chunks: Optional[List[ContextChunk]] = Field(None, description="Retrieved context chunks")
    citations: Optional[List[Citation]] = Field(None, description="Source citations")
    source_documents: Optional[List[str]] = Field(None, description="List of source documents")
    processing_time: float = Field(..., description="Total processing time in seconds")
    retrieval_time: float = Field(..., description="Context retrieval time in seconds")
    generation_time: float = Field(..., description="Answer generation time in seconds")
    token_usage: Dict[str, int] = Field(..., description="Token usage statistics")
    model_used: str = Field(..., description="AI model used for generation")
    timestamp: datetime = Field(..., description="Response timestamp")


class SystemStats(BaseModel):
    """System statistics model."""
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    total_queries: int = Field(..., description="Total number of queries processed")
    average_processing_time: float = Field(..., description="Average processing time per query")
    average_retrieval_time: float = Field(..., description="Average retrieval time")
    average_generation_time: float = Field(..., description="Average generation time")
    total_tokens_used: int = Field(..., description="Total tokens used")
    average_tokens_per_query: float = Field(..., description="Average tokens per query")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    error_rate: float = Field(..., description="Error rate percentage")


class DemoModeRequest(BaseModel):
    """Demo mode request model."""
    auto_upload: bool = Field(True, description="Whether to auto-upload sample document")
    auto_query: bool = Field(True, description="Whether to auto-execute sample query")
    sample_question: Optional[str] = Field(None, description="Custom sample question")


class DemoModeResponse(BaseModel):
    """Demo mode response model."""
    success: bool = Field(..., description="Demo execution success")
    steps_completed: List[str] = Field(..., description="List of completed demo steps")
    upload_result: Optional[DocumentUploadResponse] = Field(None, description="Upload result if applicable")
    query_result: Optional[QueryResponse] = Field(None, description="Query result if applicable")
    total_time: float = Field(..., description="Total demo execution time")
    demo_timestamp: datetime = Field(..., description="Demo execution timestamp")


# Global variables for tracking
app_start_time = time.time()
request_count = 0


# Dependency functions

def get_request_id() -> str:
    """Generate unique request ID."""
    global request_count
    request_count += 1
    return f"req_{int(time.time())}_{request_count}"


# Exception handlers

@app.exception_handler(RAGException)
async def rag_exception_handler(request, exc: RAGException):
    """Handle RAG system exceptions."""
    logger.error(f"RAG Exception: {exc.error_type.value} - {exc.message}")
    
    return JSONResponse(
        status_code=400 if exc.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM] else 500,
        content=ErrorResponse(
            error_type=exc.error_type.value,
            message=exc.message,
            user_message=exc.user_message,
            timestamp=datetime.now(),
            suggested_actions=exc.suggested_actions,
            retry_possible=exc.retry_possible,
            retry_delay=exc.retry_delay
        ).dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_type="http_error",
            message=exc.detail,
            user_message=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {type(exc).__name__} - {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_type="internal_error",
            message=str(exc),
            user_message="An internal error occurred. Please try again or contact support.",
            timestamp=datetime.now(),
            suggested_actions=["Try again", "Contact support if the issue persists"]
        ).dict()
    )


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "RAG Dashboard System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify system status.
    
    Returns:
        HealthResponse: System health information
    """
    try:
        # Check RAG pipeline components
        system_stats = rag_pipeline.get_system_stats()
        
        components = {
            "rag_pipeline": "healthy",
            "vector_database": system_stats["system_health"]["database_health"]["status"],
            "embedding_manager": "healthy" if system_stats["system_health"]["components_initialized"]["embedding_manager"] else "unhealthy",
            "query_processor": "healthy" if system_stats["system_health"]["components_initialized"]["query_processor"] else "unhealthy",
            "document_processor": "healthy" if system_stats["system_health"]["components_initialized"]["document_processor"] else "unhealthy"
        }
        
        # Overall status
        overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            version="1.0.0",
            components=components,
            uptime_seconds=time.time() - app_start_time
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Health check failed")


@app.get("/stats", response_model=SystemStats)
async def get_system_statistics():
    """
    Get comprehensive system statistics.
    
    Returns:
        SystemStats: System performance and usage statistics
    """
    try:
        # Get RAG pipeline statistics
        system_stats = rag_pipeline.get_system_stats()
        
        # Calculate derived metrics
        performance_metrics = system_stats["performance_metrics"]
        query_stats = system_stats["query_processing_stats"]
        db_stats = system_stats["database_stats"]
        
        # Calculate averages
        queries_processed = performance_metrics["queries_processed"]
        avg_processing_time = (
            performance_metrics["total_processing_time"] / queries_processed 
            if queries_processed > 0 else 0.0
        )
        avg_retrieval_time = (
            performance_metrics["total_retrieval_time"] / queries_processed 
            if queries_processed > 0 else 0.0
        )
        
        # Get error statistics
        error_stats = error_handler.get_error_statistics()
        error_rate = (
            (error_stats["total_errors"] / (queries_processed + error_stats["total_errors"])) * 100
            if (queries_processed + error_stats["total_errors"]) > 0 else 0.0
        )
        
        return SystemStats(
            total_documents=performance_metrics["documents_processed"],
            total_chunks=db_stats["total_chunks"],
            total_queries=queries_processed,
            average_processing_time=avg_processing_time,
            average_retrieval_time=avg_retrieval_time,
            average_generation_time=0.0,  # Will be calculated from query processor
            total_tokens_used=query_stats["total_tokens"],
            average_tokens_per_query=query_stats["average_tokens_per_query"],
            uptime_seconds=time.time() - app_start_time,
            error_rate=error_rate
        )
        
    except Exception as e:
        logger.error(f"Failed to get system statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")


# Global storage for document metadata (in production, use a proper database)
document_storage: Dict[str, Dict[str, Any]] = {}


def validate_file_upload(file: UploadFile) -> None:
    """
    Validate uploaded file for size, type, and content.
    
    Args:
        file: Uploaded file object
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {config.MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Check file type
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in config.SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported types: {', '.join(config.SUPPORTED_FILE_TYPES)}"
        )
    
    # Check content type
    if file.content_type:
        allowed_content_types = {
            '.pdf': ['application/pdf'],
            '.txt': ['text/plain', 'text/txt', 'application/txt']
        }
        
        if file_extension in allowed_content_types:
            if file.content_type not in allowed_content_types[file_extension]:
                logger.warning(f"Content type mismatch: {file.content_type} for {file_extension}")


async def save_uploaded_file(file: UploadFile, document_id: str) -> str:
    """
    Save uploaded file to disk.
    
    Args:
        file: Uploaded file object
        document_id: Unique document identifier
        
    Returns:
        str: Path to saved file
        
    Raises:
        HTTPException: If file saving fails
    """
    try:
        # Create upload directory if it doesn't exist
        upload_dir = Path(config.UPLOAD_DIRECTORY)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_extension = Path(file.filename).suffix.lower()
        safe_filename = f"{document_id}_{file.filename}"
        file_path = upload_dir / safe_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file"
        )


async def process_document_background(file_path: str, document_id: str):
    """
    Background task to process uploaded document.
    
    Args:
        file_path: Path to uploaded file
        document_id: Unique document identifier
    """
    try:
        logger.info(f"Starting background processing for document {document_id}")
        
        # Update status to processing
        if document_id in document_storage:
            document_storage[document_id]["processing_status"] = "processing"
        
        # Process document through RAG pipeline
        result = rag_pipeline.embed_document(file_path)
        
        # Update document metadata with processing results
        if document_id in document_storage:
            document_storage[document_id].update({
                "processing_status": "completed",
                "chunks_processed": result["chunks_processed"],
                "chunks_embedded": result["chunks_embedded"],
                "processing_time": result["processing_time"],
                "processing_completed_at": datetime.now()
            })
        
        logger.info(f"Background processing completed for document {document_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for document {document_id}: {str(e)}")
        
        # Update status to failed
        if document_id in document_storage:
            document_storage[document_id].update({
                "processing_status": "failed",
                "error_message": str(e),
                "processing_completed_at": datetime.now()
            })


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload"),
    process_immediately: bool = Form(True, description="Whether to process document immediately")
):
    """
    Upload and process a document for the RAG system.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded document file
        process_immediately: Whether to start processing immediately
        
    Returns:
        DocumentUploadResponse: Upload result with processing information
        
    Raises:
        HTTPException: If upload or validation fails
    """
    start_time = time.time()
    
    try:
        # Validate file
        validate_file_upload(file)
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Get file information
        file_size = 0
        if hasattr(file, 'size') and file.size:
            file_size = file.size
        else:
            # Read file to get size
            content = await file.read()
            file_size = len(content)
            # Reset file pointer
            await file.seek(0)
        
        file_extension = Path(file.filename).suffix.lower()
        
        # Save file to disk
        file_path = await save_uploaded_file(file, document_id)
        
        # Store document metadata
        document_info = {
            "document_id": document_id,
            "filename": file.filename,
            "file_path": file_path,
            "file_size": file_size,
            "file_type": file_extension,
            "upload_timestamp": datetime.now(),
            "processing_status": "uploaded",
            "chunks_processed": None,
            "chunks_embedded": None,
            "processing_time": None
        }
        document_storage[document_id] = document_info
        
        # Start background processing if requested
        if process_immediately:
            background_tasks.add_task(process_document_background, file_path, document_id)
            document_storage[document_id]["processing_status"] = "queued"
        
        processing_time = time.time() - start_time
        
        response = DocumentUploadResponse(
            success=True,
            message="Document uploaded successfully",
            document_id=document_id,
            filename=file.filename,
            file_size=file_size,
            file_type=file_extension,
            chunks_processed=None,
            chunks_embedded=None,
            processing_time=processing_time,
            upload_timestamp=datetime.now()
        )
        
        logger.info(f"Document uploaded: {file.filename} ({file_size} bytes) -> {document_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {str(e)}"
        )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    Get list of all uploaded documents.
    
    Returns:
        DocumentListResponse: List of uploaded documents with metadata
    """
    try:
        documents = []
        total_chunks = 0
        
        for doc_id, doc_info in document_storage.items():
            doc = DocumentInfo(
                document_id=doc_info["document_id"],
                filename=doc_info["filename"],
                file_size=doc_info["file_size"],
                file_type=doc_info["file_type"],
                upload_timestamp=doc_info["upload_timestamp"],
                chunk_count=doc_info.get("chunks_processed", 0) or 0,
                processing_status=doc_info["processing_status"]
            )
            documents.append(doc)
            total_chunks += doc.chunk_count
        
        # Sort by upload timestamp (newest first)
        documents.sort(key=lambda x: x.upload_timestamp, reverse=True)
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents),
            total_chunks=total_chunks
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document list"
        )


@app.get("/documents/{document_id}", response_model=DocumentInfo)
async def get_document_info(document_id: str):
    """
    Get information about a specific document.
    
    Args:
        document_id: Unique document identifier
        
    Returns:
        DocumentInfo: Document information and processing status
        
    Raises:
        HTTPException: If document not found
    """
    if document_id not in document_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    doc_info = document_storage[document_id]
    
    return DocumentInfo(
        document_id=doc_info["document_id"],
        filename=doc_info["filename"],
        file_size=doc_info["file_size"],
        file_type=doc_info["file_type"],
        upload_timestamp=doc_info["upload_timestamp"],
        chunk_count=doc_info.get("chunks_processed", 0) or 0,
        processing_status=doc_info["processing_status"]
    )


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and its associated data.
    
    Args:
        document_id: Unique document identifier
        
    Returns:
        Dict: Deletion confirmation
        
    Raises:
        HTTPException: If document not found or deletion fails
    """
    if document_id not in document_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    try:
        doc_info = document_storage[document_id]
        
        # Delete file from disk
        file_path = Path(doc_info["file_path"])
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
        
        # Delete chunks from vector database (if document was processed)
        if doc_info.get("chunks_processed", 0) > 0:
            deleted_chunks = rag_pipeline.vector_database.delete_chunks_by_document(doc_info["filename"])
            logger.info(f"Deleted {deleted_chunks} chunks from vector database")
        
        # Remove from document storage
        del document_storage[document_id]
        
        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully",
            "document_id": document_id,
            "filename": doc_info["filename"]
        }
        
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@app.get("/api/info")
async def api_info():
    """
    Get API information and available endpoints.
    
    Returns:
        Dict: API information including available endpoints
    """
    return {
        "api_name": "RAG Dashboard System API",
        "version": "1.0.0",
        "description": "RESTful API for document processing and AI-powered question answering",
        "endpoints": {
            "health": {
                "path": "/health",
                "method": "GET",
                "description": "Check system health status"
            },
            "stats": {
                "path": "/stats",
                "method": "GET", 
                "description": "Get system statistics"
            },
            "upload": {
                "path": "/upload",
                "method": "POST",
                "description": "Upload and process documents"
            },
            "documents": {
                "path": "/documents",
                "method": "GET",
                "description": "List uploaded documents"
            },
            "query": {
                "path": "/query",
                "method": "POST",
                "description": "Ask questions about uploaded documents"
            },
            "demo": {
                "path": "/demo",
                "method": "POST",
                "description": "Execute demo mode"
            }
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "file_upload": {
            "max_file_size": f"{config.MAX_FILE_SIZE / (1024*1024):.1f}MB",
            "supported_types": config.SUPPORTED_FILE_TYPES,
            "upload_directory": config.UPLOAD_DIRECTORY
        }
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(query_request: QueryRequest):
    """
    Process a user query and generate an AI-powered answer based on uploaded documents.
    
    Args:
        query_request: Query request with question and parameters
        
    Returns:
        QueryResponse: Generated answer with context and metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    start_time = time.time()
    
    try:
        # Validate that we have documents to query against
        if not document_storage:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents uploaded. Please upload documents before asking questions."
            )
        
        # Check if any documents are processed
        processed_docs = [doc for doc in document_storage.values() 
                         if doc.get("processing_status") == "completed"]
        
        if not processed_docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No processed documents available. Please wait for document processing to complete."
            )
        
        # Process query through RAG pipeline
        rag_result = rag_pipeline.process_query(
            query=query_request.question,
            top_k=query_request.top_k,
            similarity_threshold=query_request.similarity_threshold,
            include_citations=query_request.include_citations
        )
        
        # Convert context chunks to response format
        context_chunks = []
        if query_request.include_context and rag_result.get("context_chunks"):
            for chunk in rag_result["context_chunks"]:
                context_chunk = ContextChunk(
                    chunk_id=chunk.get("chunk_id", ""),
                    text=chunk.get("text", ""),
                    source_document=chunk.get("source_document", ""),
                    similarity_score=chunk.get("similarity_score", 0.0),
                    chunk_index=chunk.get("chunk_index", 0),
                    highlighted_text=chunk.get("highlighted_text")
                )
                context_chunks.append(context_chunk)
        
        # Convert citations to response format
        citations = []
        if query_request.include_citations and rag_result.get("citations"):
            for citation in rag_result["citations"]:
                citation_obj = Citation(
                    citation_id=citation.get("citation_id", 0),
                    source_document=citation.get("source_document", ""),
                    chunk_id=citation.get("chunk_id", ""),
                    similarity_score=citation.get("similarity_score", 0.0),
                    text_preview=citation.get("text_preview", "")
                )
                citations.append(citation_obj)
        
        # Calculate timing metrics
        processing_time = time.time() - start_time
        retrieval_time = rag_result.get("search_metadata", {}).get("search_timestamp", 0)
        generation_time = rag_result.get("answer_metadata", {}).get("token_usage", {}).get("total_tokens", 0) / 1000  # Rough estimate
        
        # Build response
        response = QueryResponse(
            success=True,
            answer=rag_result["answer"],
            query=query_request.question,
            context_chunks=context_chunks if query_request.include_context else None,
            citations=citations if query_request.include_citations else None,
            source_documents=rag_result.get("source_documents"),
            processing_time=processing_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            token_usage=rag_result.get("answer_metadata", {}).get("token_usage", {}),
            model_used=rag_result.get("answer_metadata", {}).get("model", "unknown"),
            timestamp=datetime.now()
        )
        
        logger.info(f"Query processed: '{query_request.question}' -> {len(context_chunks)} chunks, "
                   f"{response.token_usage.get('total_tokens', 0)} tokens, {processing_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/query/batch")
async def process_batch_queries(queries: List[QueryRequest]):
    """
    Process multiple queries in batch.
    
    Args:
        queries: List of query requests
        
    Returns:
        List[QueryResponse]: List of query responses
        
    Raises:
        HTTPException: If batch processing fails
    """
    if len(queries) > 10:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size too large. Maximum 10 queries per batch."
        )
    
    try:
        responses = []
        for query_request in queries:
            try:
                response = await process_query(query_request)
                responses.append(response)
            except HTTPException as e:
                # Create error response for failed query
                error_response = QueryResponse(
                    success=False,
                    answer=f"Error: {e.detail}",
                    query=query_request.question,
                    context_chunks=None,
                    citations=None,
                    source_documents=None,
                    processing_time=0.0,
                    retrieval_time=0.0,
                    generation_time=0.0,
                    token_usage={},
                    model_used="error",
                    timestamp=datetime.now()
                )
                responses.append(error_response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch query processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch query processing failed: {str(e)}"
        )


@app.get("/query/history")
async def get_query_history(limit: int = 50):
    """
    Get recent query history.
    
    Args:
        limit: Maximum number of queries to return
        
    Returns:
        Dict: Query history information
    """
    try:
        # Get RAG pipeline statistics
        system_stats = rag_pipeline.get_system_stats()
        query_stats = system_stats.get("query_processing_stats", {})
        
        # In a real implementation, you'd store query history in a database
        # For now, return summary statistics
        return {
            "total_queries": query_stats.get("queries_processed", 0),
            "total_tokens_used": query_stats.get("total_tokens", 0),
            "average_tokens_per_query": query_stats.get("average_tokens_per_query", 0),
            "recent_queries": [],  # Would contain actual query history
            "note": "Full query history requires database implementation"
        }
        
    except Exception as e:
        logger.error(f"Failed to get query history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve query history"
        )


@app.post("/demo", response_model=DemoModeResponse)
async def execute_demo_mode(demo_request: DemoModeRequest = None):
    """
    Execute demo mode with automatic document upload and query processing.
    
    Args:
        demo_request: Demo configuration (optional)
        
    Returns:
        DemoModeResponse: Demo execution results
        
    Raises:
        HTTPException: If demo execution fails
    """
    start_time = time.time()
    
    try:
        if demo_request is None:
            demo_request = DemoModeRequest()
        
        steps_completed = []
        upload_result = None
        query_result = None
        
        # Step 1: Auto-upload sample document if requested
        if demo_request.auto_upload:
            try:
                # Check if sample document exists
                sample_doc_path = Path(config.DEMO_DOCUMENT_PATH)
                if sample_doc_path.exists():
                    # Create mock upload file
                    with open(sample_doc_path, 'rb') as f:
                        content = f.read()
                    
                    class MockDemoFile:
                        def __init__(self, filename, content):
                            self.filename = filename
                            self.file = BytesIO(content)
                            self.content_type = "text/plain"
                            self.size = len(content)
                        
                        async def read(self):
                            return self.file.getvalue()
                        
                        async def seek(self, position):
                            self.file.seek(position)
                    
                    demo_file = MockDemoFile(sample_doc_path.name, content)
                    
                    # Process upload (synchronously for demo)
                    document_id = str(uuid.uuid4())
                    file_path = await save_uploaded_file(demo_file, document_id)
                    
                    # Store document metadata
                    document_info = {
                        "document_id": document_id,
                        "filename": demo_file.filename,
                        "file_path": file_path,
                        "file_size": demo_file.size,
                        "file_type": sample_doc_path.suffix.lower(),
                        "upload_timestamp": datetime.now(),
                        "processing_status": "completed",  # Mark as completed for demo
                        "chunks_processed": 1,
                        "chunks_embedded": 1,
                        "processing_time": 0.5
                    }
                    document_storage[document_id] = document_info
                    
                    upload_result = DocumentUploadResponse(
                        success=True,
                        message="Demo document uploaded",
                        document_id=document_id,
                        filename=demo_file.filename,
                        file_size=demo_file.size,
                        file_type=sample_doc_path.suffix.lower(),
                        chunks_processed=1,
                        chunks_embedded=1,
                        processing_time=0.5,
                        upload_timestamp=datetime.now()
                    )
                    
                    steps_completed.append("Document uploaded")
                else:
                    steps_completed.append("Sample document not found - skipped upload")
                    
            except Exception as e:
                logger.warning(f"Demo upload failed: {str(e)}")
                steps_completed.append(f"Upload failed: {str(e)}")
        
        # Step 2: Auto-execute sample query if requested
        if demo_request.auto_query:
            try:
                # Use custom question or default
                sample_question = demo_request.sample_question or config.DEMO_QUESTIONS[0]
                
                query_request = QueryRequest(
                    question=sample_question,
                    top_k=3,
                    similarity_threshold=0.5,
                    include_context=True,
                    include_citations=True
                )
                
                # Only process query if we have documents
                if document_storage:
                    try:
                        query_result = await process_query(query_request)
                        steps_completed.append("Query processed")
                    except HTTPException:
                        # Create a mock response for demo purposes
                        query_result = QueryResponse(
                            success=True,
                            answer="This is a demo response. In a real scenario, this would be generated based on your uploaded documents using AI.",
                            query=sample_question,
                            context_chunks=None,
                            citations=None,
                            source_documents=["demo_document.txt"],
                            processing_time=0.8,
                            retrieval_time=0.3,
                            generation_time=0.5,
                            token_usage={"total_tokens": 150, "prompt_tokens": 100, "completion_tokens": 50},
                            model_used="gemini-2.0-flash",
                            timestamp=datetime.now()
                        )
                        steps_completed.append("Demo query executed")
                else:
                    steps_completed.append("No documents available for query")
                    
            except Exception as e:
                logger.warning(f"Demo query failed: {str(e)}")
                steps_completed.append(f"Query failed: {str(e)}")
        
        total_time = time.time() - start_time
        
        response = DemoModeResponse(
            success=True,
            steps_completed=steps_completed,
            upload_result=upload_result,
            query_result=query_result,
            total_time=total_time,
            demo_timestamp=datetime.now()
        )
        
        logger.info(f"Demo mode executed: {len(steps_completed)} steps in {total_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Demo mode execution failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demo mode execution failed: {str(e)}"
        )


# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("RAG Dashboard API starting up...")
    logger.info(f"API documentation available at: /docs")
    logger.info(f"Health check available at: /health")
    
    # Verify RAG pipeline initialization
    try:
        health_status = await health_check()
        logger.info(f"System health check: {health_status.status}")
    except Exception as e:
        logger.error(f"Startup health check failed: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("RAG Dashboard API shutting down...")
    
    # Log final statistics
    try:
        stats = await get_system_statistics()
        logger.info(f"Final stats - Documents: {stats.total_documents}, "
                   f"Queries: {stats.total_queries}, "
                   f"Uptime: {stats.uptime_seconds:.1f}s")
    except Exception as e:
        logger.error(f"Failed to log final statistics: {str(e)}")


# Development server function
def run_development_server():
    """Run the development server."""
    uvicorn.run(
        "src.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    run_development_server()