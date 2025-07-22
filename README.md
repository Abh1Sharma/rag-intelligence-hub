# 🚀 RAG Intelligence Hub

A modern, AI-powered document analysis system that combines advanced document processing, vector search, and intelligent answer generation with a beautiful, responsive interface.

![RAG Intelligence Hub](https://img.shields.io/badge/AI-Powered-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)

## ✨ Features

- **🔄 Document Processing**: Advanced text extraction from PDF and TXT files
- **🧠 AI-Powered Search**: Semantic similarity matching using OpenAI embeddings
- **💬 Intelligent Q&A**: GPT-4 powered answer generation with context
- **🎨 Modern UI**: Beautiful glassmorphism design with smooth animations
- **📊 Analytics**: Real-time performance metrics and visualizations
- **🚀 Production Ready**: Containerized and deployment-ready

## 🎯 Demo

![Demo GIF](https://via.placeholder.com/800x400/667eea/ffffff?text=RAG+Intelligence+Hub+Demo)

## 🛠️ Tech Stack

- **Frontend**: Streamlit with custom CSS/animations
- **Backend**: FastAPI with async support
- **AI/ML**: OpenAI GPT-4 & Embeddings
- **Vector DB**: Chroma for semantic search
- **Processing**: PyPDF2, tiktoken for document handling
- **Deployment**: Docker, Railway, Render ready

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-intelligence-hub.git
   cd rag-intelligence-hub
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the application**
   ```bash
   # Terminal 1: Start FastAPI backend
   python main.py api

   # Terminal 2: Start Streamlit frontend
   streamlit run dashboard.py
   ```

6. **Open your browser**
   - Frontend: http://localhost:8501
   - API Docs: http://localhost:8000/docs

## 📖 Usage

### Upload Documents
1. Go to the "Upload" tab
2. Select PDF or TXT files
3. Click "Upload Document"
4. Wait for processing to complete

### Ask Questions
1. Navigate to the "Query" tab
2. Type your question about the uploaded documents
3. Adjust similarity threshold if needed
4. Get AI-powered answers with source citations

### View Analytics
- Check the "Analytics" tab for document insights
- Monitor system performance in the sidebar
- View processing statistics and metrics

## 🌐 Deployment

### Railway (Recommended)
```bash
# Push to GitHub first
git push origin main

# Deploy to Railway
# 1. Go to railway.app
# 2. Connect GitHub repo
# 3. Add OPENAI_API_KEY environment variable
# 4. Deploy!
```

### Docker
```bash
# Build and run
docker build -t rag-hub .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key rag-hub
```

### Streamlit Community Cloud
1. Push to GitHub (public repo)
2. Go to share.streamlit.io
3. Connect repository and deploy

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## 🔧 Configuration

Key environment variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_CHAT_MODEL=gpt-4
CHROMA_PERSIST_DIRECTORY=./chroma_db
API_HOST=0.0.0.0
PORT=8000
```

## 📊 API Documentation

The FastAPI backend provides a comprehensive REST API:

- **POST /upload**: Upload and process documents
- **POST /query**: Ask questions about documents
- **GET /documents**: List uploaded documents
- **GET /stats**: System performance metrics
- **POST /demo**: Run demo mode

Visit http://localhost:8000/docs for interactive API documentation.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │   OpenAI API    │
│   Frontend      │◄──►│    Backend      │◄──►│   (GPT-4 +      │
│                 │    │                 │    │   Embeddings)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │  Document       │              │
         │              │  Processor      │              │
         │              └─────────────────┘              │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Storage  │    │   Chroma DB     │    │   Logging &     │
│   (uploads/)    │    │   (Vector Store)│    │   Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing powerful AI models
- Streamlit team for the amazing framework
- FastAPI for the high-performance backend framework
- Chroma for the vector database solution

## 📞 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/rag-intelligence-hub/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/rag-intelligence-hub/discussions)

---

<div align="center">
  <strong>Built with ❤️ using AI and modern web technologies</strong>
</div>