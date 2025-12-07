#!/bin/bash
# Run Flask backend server

cd "$(dirname "$0")"

# Install dependencies if needed
pip install -r requirements.txt -q

# Run Flask
python app.py
