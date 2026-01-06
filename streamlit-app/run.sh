#!/bin/bash
# Run Streamlit WellLog Analyzer

cd "$(dirname "$0")"

echo "🛢️ WellLog Analyzer Pro - Streamlit Version"
echo "============================================="

# Install dependencies if needed
pip install streamlit lasio pandas numpy matplotlib dlisio -q

# Run Streamlit
streamlit run app.py --server.port 8501
