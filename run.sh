#!/bin/bash
# Activate conda env and run streamlit
source $(conda info --base)/etc/profile.d/conda.sh
conda activate logai_env
streamlit run app.py
