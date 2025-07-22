"""
Streamlit Dashboard for the RAG Dashboard System.
Provides a modern, user-friendly interface for document upload and AI-powered querying.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page configuration with modern settings
st.set_page_config(
    page_title="RAG Intelligence Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/rag-dashboard',
        'Report a bug': 'https://github.com/your-repo/rag-dashboard/issues',
        'About': """
        # RAG Intelligence Hub
        
        A modern, AI-powered document analysis system that combines:
        - **Document Processing**: Advanced text extraction and chunking
        - **Vector Search**: Semantic similarity matching
        - **AI Generation**: Intelligent answer synthesis
        - **Modern UI**: Beautiful, responsive interface
        
        Built with ❤️ using Streamlit, FastAPI, OpenAI, and Chroma.
        """
    }
)

# Modern CSS styling with glassmorphism and advanced animations
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container with glassmorphism */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 0 20px 20px 0;
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Main header with animated gradient */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        animation: gradientShift 6s ease infinite;
        position: relative;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: translateX(-50%) scaleX(1); }
        50% { opacity: 0.7; transform: translateX(-50%) scaleX(1.2); }
    }
    
    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Success message with modern styling */
    .success-message {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
        color: #065f46;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(16, 185, 129, 0.2);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .success-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10b981, #059669);
        animation: shimmer 2s ease-in-out infinite;
    }
    
    /* Error message styling */
    .error-message {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.1));
        color: #7f1d1d;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(239, 68, 68, 0.2);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .error-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ef4444, #dc2626);
    }
    
    /* Info message styling */
    .info-message {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(37, 99, 235, 0.1));
        color: #1e3a8a;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .info-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #2563eb);
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Modern buttons with hover effects */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar metrics with modern cards */
    .sidebar-metric {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .sidebar-metric:hover {
        transform: translateX(4px);
        border-left-color: #764ba2;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.1));
    }
    
    /* Demo button with special styling */
    .demo-button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 1rem 2rem;
        font-weight: 700;
        font-size: 1.1rem;
        width: 100%;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(240, 147, 251, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .demo-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.5);
    }
    
    .demo-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .demo-button:hover::before {
        left: 100%;
    }
    
    /* Context chunks with modern styling */
    .context-chunk {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
        transition: all 0.3s ease;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .context-chunk:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border-left-color: #138496;
    }
    
    /* Citations with elegant styling */
    .citation {
        background: linear-gradient(135deg, rgba(255, 243, 205, 0.3), rgba(255, 234, 167, 0.3));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 234, 167, 0.4);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    
    .citation:hover {
        transform: translateX(4px);
        border-color: rgba(255, 234, 167, 0.6);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: rgba(255, 255, 255, 0.5);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.1));
    }
    
    /* Text input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
    }
    
    /* Loading animation */
    @keyframes loading {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-spinner {
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 3px solid #667eea;
        width: 30px;
        height: 30px;
        animation: loading 1s linear infinite;
        margin: 0 auto;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            padding: 1rem;
        }
        
        .glass-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# Configuration - Dynamic API URL for deployment
import os
API_BASE_URL = os.getenv("API_BASE_URL", "https://web-production-9e501.up.railway.app")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'system_stats' not in st.session_state:
    st.session_state.system_stats = {}
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def check_api_connection():
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_stats():
    """Get system statistics from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def get_documents():
    """Get list of uploaded documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"documents": [], "total_count": 0, "total_chunks": 0}
    except:
        return {"documents": [], "total_count": 0, "total_chunks": 0}

def upload_document(file, process_immediately=True):
    """Upload a document to the API."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"process_immediately": process_immediately}
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            data=data,
            timeout=30
        )
        
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def process_query(question, top_k=5, similarity_threshold=0.5, include_context=True, include_citations=True):
    """Process a query through the API."""
    try:
        payload = {
            "question": question,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "include_context": include_context,
            "include_citations": include_citations
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=60
        )
        
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def execute_demo_mode():
    """Execute demo mode through the API."""
    try:
        payload = {
            "auto_upload": True,
            "auto_query": True,
            "sample_question": "What is the main topic of this document?"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/demo",
            json=payload,
            timeout=30
        )
        
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def render_sidebar():
    """Render the sidebar with system statistics and controls."""
    with st.sidebar:
        st.markdown("## 📊 System Status")
        
        # API Connection Status with modern styling
        api_connected = check_api_connection()
        if api_connected:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.2)); 
                        color: #065f46; padding: 1rem; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);
                        text-align: center; margin-bottom: 1rem;">
                <span style="font-size: 1.2rem;">🟢</span> <strong>API Connected</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.2)); 
                        color: #7f1d1d; padding: 1rem; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.3);
                        text-align: center; margin-bottom: 1rem;">
                <span style="font-size: 1.2rem;">🔴</span> <strong>API Disconnected</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**Please start the FastAPI backend:**")
            st.code("python main.py api", language="bash")
            return
        
        # Get and display system statistics
        stats = get_system_stats()
        st.session_state.system_stats = stats
        
        if stats:
            st.markdown("### 📈 Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get("total_documents", 0))
                st.metric("Queries", stats.get("total_queries", 0))
            
            with col2:
                st.metric("Chunks", stats.get("total_chunks", 0))
                st.metric("Tokens Used", stats.get("total_tokens_used", 0))
            
            # Performance metrics
            st.markdown("### ⚡ Performance")
            avg_time = stats.get("average_processing_time", 0)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
            error_rate = stats.get("error_rate", 0)
            st.metric("Error Rate", f"{error_rate:.1f}%")
            
            uptime = stats.get("uptime_seconds", 0)
            uptime_hours = uptime / 3600
            st.metric("Uptime", f"{uptime_hours:.1f}h")
        
        # Dark mode toggle
        st.markdown("### 🎨 Appearance")
        dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        # Demo mode button
        st.markdown("### 🚀 Quick Demo")
        if st.button("Run Demo Mode", key="demo_button"):
            with st.spinner("Running demo..."):
                success, result = execute_demo_mode()
                if success:
                    st.success("Demo completed successfully!")
                    st.json(result)
                else:
                    st.error(f"Demo failed: {result.get('detail', 'Unknown error')}")

def render_main_header():
    """Render the main header with modern styling."""
    st.markdown("""
    <div class="main-header">
        🚀 RAG Intelligence Hub
        <div style="font-size: 1.2rem; font-weight: 400; margin-top: 0.5rem; color: rgba(255,255,255,0.8);">
            Upload documents and unlock AI-powered insights
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_upload_section():
    """Render the document upload section."""
    st.markdown("## 📁 Document Upload")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a document to upload",
        type=['pdf', 'txt'],
        help="Supported formats: PDF, TXT (Max size: 10MB)"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        process_immediately = st.checkbox("Process immediately", value=True)
    
    with col2:
        upload_button = st.button("Upload Document", disabled=uploaded_file is None)
    
    if upload_button and uploaded_file:
        # Modern loading animation
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <div class="loading-spinner"></div>
            <div style="margin-top: 1rem; color: rgba(255,255,255,0.8); font-weight: 500;">
                🚀 Uploading and processing document...
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        success, result = upload_document(uploaded_file, process_immediately)
        
        if success:
            st.markdown(f"""
            <div class="success-message">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">✅</span>
                    <strong style="font-size: 1.1rem;">Document uploaded successfully!</strong>
                </div>
                <div style="font-size: 0.9rem; opacity: 0.9;">
                    <strong>File:</strong> {result['filename']}<br>
                    <strong>Size:</strong> {result['file_size']:,} bytes<br>
                    <strong>ID:</strong> <code style="background: rgba(0,0,0,0.1); padding: 0.2rem 0.4rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace;">{result['document_id']}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Refresh documents list
            time.sleep(1)
            st.rerun()
        else:
            st.markdown(f"""
            <div class="error-message">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">❌</span>
                    <strong style="font-size: 1.1rem;">Upload failed</strong>
                </div>
                <div style="font-size: 0.9rem; opacity: 0.9;">
                    {result.get('detail', 'Unknown error')}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_documents_list():
    """Render the list of uploaded documents."""
    st.markdown("## 📚 Uploaded Documents")
    
    docs_data = get_documents()
    documents = docs_data.get("documents", [])
    
    if not documents:
        st.markdown("""
        <div class="info-message">
            📝 No documents uploaded yet. Upload a document above to get started!
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Documents summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", docs_data.get("total_count", 0))
    with col2:
        st.metric("Total Chunks", docs_data.get("total_chunks", 0))
    with col3:
        processed_count = sum(1 for doc in documents if doc.get("processing_status") == "completed")
        st.metric("Processed", processed_count)
    
    # Documents table
    for doc in documents:
        with st.expander(f"📄 {doc['filename']} ({doc['processing_status']})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Size:** {doc['file_size']:,} bytes")
                st.write(f"**Type:** {doc['file_type']}")
                st.write(f"**Chunks:** {doc['chunk_count']}")
            
            with col2:
                st.write(f"**Status:** {doc['processing_status']}")
                st.write(f"**Uploaded:** {doc['upload_timestamp'][:19]}")
                
                if st.button(f"Delete", key=f"delete_{doc['document_id']}"):
                    # TODO: Implement delete functionality
                    st.warning("Delete functionality coming soon!")

def render_query_section():
    """Render the query interface."""
    st.markdown("## 💬 Ask Questions")
    
    # Check if we have processed documents
    docs_data = get_documents()
    documents = docs_data.get("documents", [])
    processed_docs = [doc for doc in documents if doc.get("processing_status") == "completed"]
    
    if not processed_docs:
        st.markdown("""
        <div class="info-message">
            ⏳ Please upload and process documents before asking questions.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Query input
    question = st.text_area(
        "Enter your question:",
        placeholder="What is the main topic discussed in the documents?",
        height=100
    )
    
    # Query parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    
    with col2:
        similarity_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    with col3:
        include_context = st.checkbox("Include context", value=True)
        include_citations = st.checkbox("Include citations", value=True)
    
    # Example questions
    st.markdown("**Example questions:**")
    example_questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the most important findings mentioned?",
        "What specific details are provided about [topic]?"
    ]
    
    cols = st.columns(len(example_questions))
    for i, example in enumerate(example_questions):
        with cols[i]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.example_question = example
                st.rerun()
    
    # Use example question if selected
    if hasattr(st.session_state, 'example_question'):
        question = st.session_state.example_question
        delattr(st.session_state, 'example_question')
    
    # Submit query
    if st.button("Ask Question", disabled=not question.strip()):
        with st.spinner("Processing your question..."):
            success, result = process_query(
                question=question,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                include_context=include_context,
                include_citations=include_citations
            )
            
            if success:
                render_query_results(result)
                
                # Add to query history
                st.session_state.query_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "timestamp": datetime.now(),
                    "processing_time": result["processing_time"]
                })
            else:
                st.markdown(f"""
                <div class="error-message">
                    ❌ Query failed: {result.get('detail', 'Unknown error')}
                </div>
                """, unsafe_allow_html=True)

def render_query_results(result):
    """Render query results with modern styling."""
    st.markdown("""
    <div style="display: flex; align-items: center; margin: 1.5rem 0 1rem 0;">
        <span style="font-size: 1.8rem; margin-right: 0.5rem;">🎯</span>
        <h3 style="margin: 0; color: white; font-weight: 600;">AI Answer</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Answer with modern glass card styling
    answer = result["answer"]
    st.markdown(f"""
    <div class="glass-card" style="position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2);"></div>
        <div style="font-size: 1.05rem; line-height: 1.7; color: rgba(255,255,255,0.95); margin-top: 0.5rem;">
            {answer.replace('\\n', '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern copy button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📋 Copy Answer", key="copy_answer"):
            st.markdown("""
            <div style="color: #10b981; font-size: 0.9rem; text-align: center; margin-top: 0.5rem;">
                ✓ Copied to clipboard!
            </div>
            """, unsafe_allow_html=True)
    
    # Performance metrics with modern cards
    st.markdown("""
    <div style="display: flex; align-items: center; margin: 2rem 0 1rem 0;">
        <span style="font-size: 1.5rem; margin-right: 0.5rem;">⚡</span>
        <h4 style="margin: 0; color: white; font-weight: 500;">Performance Metrics</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Processing Time", f"{result['processing_time']:.2f}s", delta=None)
    with col2:
        st.metric("Retrieval Time", f"{result.get('retrieval_time', 0):.2f}s", delta=None)
    with col3:
        st.metric("Generation Time", f"{result.get('generation_time', 0):.2f}s", delta=None)
    with col4:
        st.metric("Tokens Used", result['token_usage'].get('total_tokens', 0), delta=None)
    
    # Context chunks with modern styling
    if result.get("context_chunks"):
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 2rem 0 1rem 0;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">📖</span>
            <h4 style="margin: 0; color: white; font-weight: 500;">Retrieved Context</h4>
        </div>
        """, unsafe_allow_html=True)
        
        show_context = st.checkbox("Show retrieved context", value=False, key="show_context_toggle")
        
        if show_context:
            for i, chunk in enumerate(result["context_chunks"], 1):
                st.markdown(f"""
                <div class="context-chunk">
                    <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 1rem;">
                        <div>
                            <strong style="color: #17a2b8; font-size: 1.1rem;">Context {i}</strong>
                            <span style="margin-left: 1rem; font-size: 0.9rem; opacity: 0.8;">
                                Similarity: <strong>{chunk['similarity_score']:.3f}</strong>
                            </span>
                        </div>
                    </div>
                    <div style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 0.5rem;">
                        📄 Source: {chunk['source_document']}
                    </div>
                    <div style="line-height: 1.6; font-size: 0.95rem;">
                        {chunk['text']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Citations with modern styling
    if result.get("citations"):
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 2rem 0 1rem 0;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">📚</span>
            <h4 style="margin: 0; color: white; font-weight: 500;">Source Citations</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for citation in result["citations"]:
            st.markdown(f"""
            <div class="citation">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="background: rgba(255,234,167,0.8); color: #92400e; padding: 0.2rem 0.5rem; border-radius: 6px; font-weight: 600; font-size: 0.8rem; margin-right: 0.5rem;">
                        [{citation['citation_id']}]
                    </span>
                    <strong style="color: #92400e;">{citation['source_document']}</strong>
                    <span style="margin-left: auto; font-size: 0.8rem; opacity: 0.8;">
                        Relevance: {citation['similarity_score']:.3f}
                    </span>
                </div>
                <div style="font-style: italic; font-size: 0.9rem; line-height: 1.5; opacity: 0.9;">
                    {citation['text_preview']}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_analytics():
    """Render analytics and visualizations."""
    st.markdown("## 📊 Analytics")
    
    docs_data = get_documents()
    documents = docs_data.get("documents", [])
    
    if not documents:
        st.info("Upload documents to see analytics.")
        return
    
    # Document type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Document Types")
        type_counts = {}
        for doc in documents:
            file_type = doc.get("file_type", "unknown")
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        if type_counts:
            fig = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="Document Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Processing Status")
        status_counts = {}
        for doc in documents:
            status = doc.get("processing_status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            fig = px.bar(
                x=list(status_counts.keys()),
                y=list(status_counts.values()),
                title="Processing Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Chunks per document
    st.markdown("### Chunks per Document")
    chunk_data = []
    for doc in documents:
        chunk_data.append({
            "Document": doc["filename"][:20] + "..." if len(doc["filename"]) > 20 else doc["filename"],
            "Chunks": doc.get("chunk_count", 0)
        })
    
    if chunk_data:
        df = pd.DataFrame(chunk_data)
        fig = px.bar(df, x="Document", y="Chunks", title="Document Chunk Distribution")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function."""
    # Apply dark mode if enabled
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
            .stApp {
                background-color: #1e1e1e;
                color: #ffffff;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    render_main_header()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "💬 Query", "📊 Analytics", "📚 Documents"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_query_section()
    
    with tab3:
        render_analytics()
    
    with tab4:
        render_documents_list()
    
    # Modern footer
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; text-align: center; 
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
                backdrop-filter: blur(20px); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.2);">
        <div style="color: rgba(255,255,255,0.9); font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
            🚀 RAG Intelligence Hub
        </div>
        <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            Powered by <strong>Streamlit</strong> • <strong>FastAPI</strong> • <strong>OpenAI</strong> • <strong>Chroma</strong>
        </div>
        <div style="margin-top: 1rem;">
            <a href="http://localhost:8000/docs" target="_blank" 
               style="color: #667eea; text-decoration: none; font-weight: 500; 
                      padding: 0.5rem 1rem; border: 1px solid rgba(102, 126, 234, 0.3); 
                      border-radius: 8px; transition: all 0.3s ease;">
                📚 API Documentation
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()