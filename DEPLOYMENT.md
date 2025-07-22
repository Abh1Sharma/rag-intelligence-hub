# RAG Intelligence Hub - Deployment Guide

## 🚀 Quick Deploy to Railway

### Prerequisites
- GitHub account
- Railway account (free)
- OpenAI API key

### Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy to Railway**
   - Go to [railway.app](https://railway.app)
   - Click "Deploy from GitHub"
   - Select your repository
   - Add environment variables:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `PORT`: 8501 (for Streamlit)
     - `API_BASE_URL`: https://your-app-name.railway.app

3. **Configure Custom Domain (Optional)**
   - In Railway dashboard, go to Settings
   - Add custom domain
   - Update DNS records

## 🌐 Alternative Deployment Options

### Streamlit Community Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Add secrets in dashboard

### Render
1. Create account at [render.com](https://render.com)
2. Create new Web Service
3. Connect GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `./start.sh`

### Docker Deployment
```bash
# Build image
docker build -t rag-hub .

# Run container
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key rag-hub
```

## 🔧 Environment Variables

Required for all deployments:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PORT`: Port for the application (usually 8501)
- `API_BASE_URL`: Base URL for API calls

Optional:
- `CHROMA_PERSIST_DIRECTORY`: Database storage path
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)

## 🛡️ Security Considerations

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive data
3. **Enable HTTPS** in production
4. **Set up monitoring** and logging
5. **Regular backups** of vector database

## 📊 Monitoring

- Check Railway logs for errors
- Monitor OpenAI API usage
- Set up uptime monitoring
- Track user analytics (optional)

## 🔄 Updates

To update your deployed app:
1. Push changes to GitHub
2. Railway auto-deploys from main branch
3. Check deployment logs for issues

## 💰 Cost Estimation

**Railway (Recommended)**
- Free tier: $0/month (500 hours)
- Pro: $5/month + usage

**Render**
- Free tier: $0/month (limited)
- Starter: $7/month

**OpenAI API**
- Embeddings: ~$0.0001 per 1K tokens
- GPT-4: ~$0.03 per 1K tokens
- Estimated: $10-50/month for moderate usage

## 🆘 Troubleshooting

**Common Issues:**
1. **API Key not working**: Check environment variables
2. **Port binding errors**: Ensure PORT is set correctly
3. **File upload issues**: Check upload directory permissions
4. **Database errors**: Verify Chroma persistence path

**Getting Help:**
- Check Railway logs
- Review GitHub issues
- Contact support if needed