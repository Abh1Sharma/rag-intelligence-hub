"""
Simplified Streamlit Dashboard for RAG Intelligence Hub
This version works with external API backend
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="RAG Intelligence Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://web-production-9e501.up.railway.app")

# Modern CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        animation: gradientShift 6s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .success-message {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
        color: #065f46;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(16, 185, 129, 0.2);
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.1));
        color: #7f1d1d;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(239, 68, 68, 0.2);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_connection():
    """Check if the API is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_document(file):
    """Upload document to API."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"process_immediately": True}
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            data=data,
            timeout=30
        )
        
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def process_query(question):
    """Process query through API."""
    try:
        payload = {
            "question": question,
            "top_k": 5,
            "similarity_threshold": 0.5,
            "include_context": True,
            "include_citations": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=60
        )
        
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def main():
    """Main application."""
    # Header
    st.markdown("""
    <div class="main-header">
        🚀 RAG Intelligence Hub
        <div style="font-size: 1.2rem; font-weight: 400; margin-top: 0.5rem; color: rgba(255,255,255,0.8);">
            Upload documents and unlock AI-powered insights
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API connection
    if not check_api_connection():
        st.error("❌ Cannot connect to API backend. Please check if the backend is running.")
        st.info(f"API URL: {API_BASE_URL}")
        return
    
    st.success("✅ Connected to API backend")
    
    # Tabs
    tab1, tab2 = st.tabs(["📁 Upload", "💬 Query"])
    
    with tab1:
        st.markdown("## 📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a document to upload",
            type=['pdf', 'txt'],
            help="Supported formats: PDF, TXT"
        )
        
        if st.button("Upload Document", disabled=uploaded_file is None):
            if uploaded_file:
                with st.spinner("Uploading and processing document..."):
                    success, result = upload_document(uploaded_file)
                    
                    if success:
                        st.markdown(f"""
                        <div class="success-message">
                            ✅ <strong>Document uploaded successfully!</strong><br>
                            <strong>File:</strong> {result['filename']}<br>
                            <strong>Size:</strong> {result['file_size']:,} bytes
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-message">
                            ❌ <strong>Upload failed:</strong> {result.get('detail', 'Unknown error')}
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## 💬 Ask Questions")
        
        question = st.text_area(
            "Enter your question:",
            placeholder="What is the main topic discussed in the documents?",
            height=100
        )
        
        if st.button("Ask Question", disabled=not question.strip()):
            with st.spinner("Processing your question..."):
                success, result = process_query(question)
                
                if success:
                    st.markdown("### 🎯 Answer")
                    st.markdown(f"""
                    <div class="glass-card">
                        {result['answer'].replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Performance metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                    with col2:
                        st.metric("Tokens Used", result['token_usage'].get('total_tokens', 0))
                    with col3:
                        st.metric("Model", result.get('model_used', 'Unknown'))
                else:
                    st.markdown(f"""
                    <div class="error-message">
                        ❌ <strong>Query failed:</strong> {result.get('detail', 'Unknown error')}
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()