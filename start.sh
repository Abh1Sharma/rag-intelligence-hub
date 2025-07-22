#!/bin/bash

# Start FastAPI backend in background
python main.py api &

# Wait for API to start
sleep 5

# Start Streamlit frontend
streamlit run dashboard.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true