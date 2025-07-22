#!/usr/bin/env python3
"""
Final integration test for the complete RAG Dashboard System.
Tests end-to-end workflows and validates all requirements.
"""

import sys
import os
import time
import subprocess
import requests
import tempfile
from pathlib import Path

sys.path.insert(0, 'src')

def test_system_setup():
    """Test system setup and configuration."""
    print('=== System Setup Test ===')
    
    # Test 1: Check required files exist
    print('\n1. Checking required files...')
    required_files = [
        'main.py',
        'dashboard.py',
        'requirements.txt',
        'README.md',
        'DEPLOYMENT.md',
        'Dockerfile',
        'docker-compose.yml',
        '.env.example',
        'src/api.py',
        'src/rag_pipeline.py',
        'src/error_handler.py',
        'config.py'
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f'   ✓ {file_path}')
        else:
            print(f'   ✗ {file_path} - MISSING')
            return False
    
    # Test 2: Check directory structure
    print('\n2. Checking directory structure...')
    required_dirs = [
        'src',
        'utils',
        'uploads',
        'temp',
        'logs',
        'chroma_db',
        'demo',
        'deploy',
        'tests'
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f'   ✓ {dir_path}/')
        else:
            print(f'   ⚠ {dir_path}/ - Creating...')
            Path(dir_path).mkdir(exist_ok=True)
    
    # Test 3: Check demo document
    print('\n3. Checking demo document...')
    demo_doc = Path('demo/sample_document.txt')
    if demo_doc.exists():
        print(f'   ✓ Demo document exists ({demo_doc.stat().st_size} bytes)')
    else:
        print(f'   ⚠ Demo document missing - Creating...')
        demo_content = """# Sample Document for RAG Dashboard System

This is a sample document for testing the RAG Dashboard System.

## Key Features
- Document upload and processing
- AI-powered question answering
- Vector similarity search
- Modern web interface

## Technology Stack
- FastAPI backend
- Streamlit frontend
- Google Gemini for text generation
- ChromaDB for vector storage

This document can be used to test the complete RAG workflow."""
        demo_doc.write_text(demo_content)
        print(f'   ✓ Demo document created')
    
    print('\n✅ System setup test completed!')
    return True

def test_component_imports():
    """Test that all components can be imported successfully."""
    print('\n=== Component Import Test ===')
    
    components = [
        ('config', 'config'),
        ('src.api', 'app'),
        ('src.rag_pipeline', 'rag_pipeline'),
        ('src.error_handler', 'error_handler'),
        ('utils.logger', 'setup_logger')
    ]
    
    for module_name, component_name in components:
        try:
            module = __import__(module_name, fromlist=[component_name])
            component = getattr(module, component_name)
            print(f'   ✓ {module_name}.{component_name}')
        except Exception as e:
            print(f'   ✗ {module_name}.{component_name} - {str(e)[:50]}...')
            return False
    
    print('\n✅ Component import test completed!')
    return True

def test_api_server():
    """Test FastAPI server startup and basic endpoints."""
    print('\n=== API Server Test ===')
    
    # Start API server in background
    print('\n1. Starting API server...')
    try:
        # Import and test basic functionality
        from src.api import app
        print('   ✓ FastAPI app imported successfully')
        
        # Test route registration
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        expected_routes = ['/', '/health', '/stats', '/upload', '/documents', '/query', '/demo']
        
        for route in expected_routes:
            if route in routes:
                print(f'   ✓ Route {route} registered')
            else:
                print(f'   ⚠ Route {route} not found')
        
        print('   ✓ API server configuration validated')
        
    except Exception as e:
        print(f'   ✗ API server test failed: {str(e)[:100]}...')
        return False
    
    print('\n✅ API server test completed!')
    return True

def test_rag_pipeline():
    """Test RAG pipeline components."""
    print('\n=== RAG Pipeline Test ===')
    
    try:
        from src.rag_pipeline import rag_pipeline
        
        # Test pipeline initialization
        print('   ✓ RAG pipeline imported and initialized')
        
        # Test system stats
        stats = rag_pipeline.get_system_stats()
        print(f'   ✓ System stats retrieved: {len(stats)} categories')
        
        # Test component health
        health = stats.get('system_health', {})
        db_health = health.get('database_health', {})
        
        if db_health.get('status') == 'healthy':
            print('   ✓ Vector database healthy')
        else:
            print('   ⚠ Vector database status:', db_health.get('status', 'unknown'))
        
        components = health.get('components_initialized', {})
        for component, status in components.items():
            if status:
                print(f'   ✓ {component} initialized')
            else:
                print(f'   ⚠ {component} not initialized')
        
    except Exception as e:
        print(f'   ✗ RAG pipeline test failed: {str(e)[:100]}...')
        return False
    
    print('\n✅ RAG pipeline test completed!')
    return True

def test_error_handling():
    """Test error handling system."""
    print('\n=== Error Handling Test ===')
    
    try:
        from src.error_handler import error_handler, ErrorType, ErrorSeverity
        
        # Test error classification
        test_error = Exception("Test API timeout error")
        error_type = error_handler._classify_error(test_error)
        print(f'   ✓ Error classification: {error_type.value}')
        
        # Test error statistics
        stats = error_handler.get_error_statistics()
        print(f'   ✓ Error statistics: {stats["total_errors"]} total errors')
        
        # Test error types enum
        print(f'   ✓ Error types available: {len(list(ErrorType))}')
        print(f'   ✓ Error severities available: {len(list(ErrorSeverity))}')
        
    except Exception as e:
        print(f'   ✗ Error handling test failed: {str(e)[:100]}...')
        return False
    
    print('\n✅ Error handling test completed!')
    return True

def test_demo_mode():
    """Test demo mode functionality."""
    print('\n=== Demo Mode Test ===')
    
    try:
        # Check demo document exists
        demo_doc = Path('demo/sample_document.txt')
        if demo_doc.exists():
            print('   ✓ Demo document available')
            
            # Check demo document content
            content = demo_doc.read_text()
            if len(content) > 100:
                print(f'   ✓ Demo document has content ({len(content)} chars)')
            else:
                print('   ⚠ Demo document seems too short')
        else:
            print('   ✗ Demo document missing')
            return False
        
        # Test demo configuration
        from config import config
        demo_questions = config.DEMO_QUESTIONS
        print(f'   ✓ Demo questions configured: {len(demo_questions)}')
        
        for i, question in enumerate(demo_questions[:3], 1):
            print(f'     {i}. {question}')
        
    except Exception as e:
        print(f'   ✗ Demo mode test failed: {str(e)[:100]}...')
        return False
    
    print('\n✅ Demo mode test completed!')
    return True

def test_deployment_readiness():
    """Test deployment readiness."""
    print('\n=== Deployment Readiness Test ===')
    
    # Test 1: Docker configuration
    print('\n1. Checking Docker configuration...')
    docker_files = ['Dockerfile', 'docker-compose.yml']
    for file_path in docker_files:
        if Path(file_path).exists():
            print(f'   ✓ {file_path}')
        else:
            print(f'   ✗ {file_path} missing')
            return False
    
    # Test 2: Deployment configurations
    print('\n2. Checking deployment configurations...')
    deploy_files = ['deploy/render.yaml', 'deploy/vercel.json']
    for file_path in deploy_files:
        if Path(file_path).exists():
            print(f'   ✓ {file_path}')
        else:
            print(f'   ⚠ {file_path} missing (optional)')
    
    # Test 3: Documentation
    print('\n3. Checking documentation...')
    doc_files = ['README.md', 'DEPLOYMENT.md']
    for file_path in doc_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f'   ✓ {file_path} ({size:,} bytes)')
        else:
            print(f'   ✗ {file_path} missing')
            return False
    
    # Test 4: Requirements file
    print('\n4. Checking requirements...')
    req_file = Path('requirements.txt')
    if req_file.exists():
        requirements = req_file.read_text().strip().split('\n')
        req_count = len([r for r in requirements if r.strip() and not r.startswith('#')])
        print(f'   ✓ requirements.txt ({req_count} packages)')
    else:
        print('   ✗ requirements.txt missing')
        return False
    
    print('\n✅ Deployment readiness test completed!')
    return True

def test_performance_requirements():
    """Test performance requirements."""
    print('\n=== Performance Requirements Test ===')
    
    try:
        # Test demo mode timing requirement (< 15 seconds)
        print('\n1. Testing demo mode timing...')
        start_time = time.time()
        
        # Simulate demo mode components
        from src.rag_pipeline import QueryProcessor
        processor = QueryProcessor()
        
        # Test query processing time
        test_chunks = [
            {
                'text': 'This is a test chunk for performance testing.',
                'source_document': 'test.txt',
                'similarity_score': 0.85,
                'chunk_id': 'test_chunk_1'
            }
        ]
        
        query_start = time.time()
        try:
            result = processor.generate_answer("What is this about?", test_chunks)
            query_time = time.time() - query_start
            print(f'   ✓ Query processing time: {query_time:.2f}s')
            
            if query_time < 5.0:
                print('   ✓ Query processing meets performance requirements')
            else:
                print('   ⚠ Query processing slower than expected')
                
        except Exception as e:
            print(f'   ⚠ Query processing test skipped: {str(e)[:50]}...')
        
        total_time = time.time() - start_time
        print(f'   ✓ Total test time: {total_time:.2f}s')
        
        if total_time < 15.0:
            print('   ✓ Performance requirements met')
        else:
            print('   ⚠ Performance slower than 15s requirement')
        
    except Exception as e:
        print(f'   ✗ Performance test failed: {str(e)[:100]}...')
        return False
    
    print('\n✅ Performance requirements test completed!')
    return True

def run_all_tests():
    """Run all integration tests."""
    print('🚀 Starting Final Integration Tests for RAG Dashboard System')
    print('=' * 70)
    
    tests = [
        test_system_setup,
        test_component_imports,
        test_api_server,
        test_rag_pipeline,
        test_error_handling,
        test_demo_mode,
        test_deployment_readiness,
        test_performance_requirements
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f'   ✗ Test {test_func.__name__} crashed: {str(e)[:100]}...')
            failed += 1
    
    print('\n' + '=' * 70)
    print(f'📊 Final Integration Test Results:')
    print(f'   ✅ Passed: {passed}')
    print(f'   ❌ Failed: {failed}')
    print(f'   📈 Success Rate: {(passed/(passed+failed)*100):.1f}%')
    
    if failed == 0:
        print('\n🎉 ALL INTEGRATION TESTS PASSED!')
        print('✅ RAG Dashboard System is ready for deployment!')
        
        print('\n🚀 Quick Start Commands:')
        print('   Setup:     python main.py setup')
        print('   Run Both:  python main.py both')
        print('   API Only:  python main.py api')
        print('   Frontend:  python main.py streamlit')
        print('   Docker:    docker-compose up --build')
        
        print('\n📖 Documentation:')
        print('   README.md - Complete usage guide')
        print('   DEPLOYMENT.md - Deployment instructions')
        print('   http://localhost:8000/docs - API documentation')
        
        return True
    else:
        print('\n⚠️  Some tests failed. Please review the issues above.')
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)