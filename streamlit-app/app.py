"""
WellLog Analyzer Pro - Main Entry Point
Professional LAS file analysis and visualization suite.
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Petrophysical Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1d24 0%, #2d3748 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #3d4852;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00D4AA;
        margin: 0;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #8892a0;
        margin-top: 0.5rem;
    }
    
    .feature-card {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 25px;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 15px;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1d24;
        margin-bottom: 10px;
    }
    
    .feature-desc {
        font-size: 0.95rem;
        color: #495057;
        line-height: 1.5;
    }
    
    .badge-new {
        background: #00D4AA;
        color: #1a1d24;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <div class="main-title">Petrophysical Analysis</div>
    <div class="main-subtitle">Professional Well Log Analysis & Petrophysical Workflows</div>
</div>
""", unsafe_allow_html=True)

# Welcome message
st.markdown("""
### Welcome to Petrophysical Analysis

This application provides comprehensive petrophysical analysis workflows for well log data processing and analysis.
""")

st.markdown("---")

# Feature card
st.markdown("### Available Tools")

st.markdown("""
<div class="feature-card">
    <div class="feature-icon"></div>
    <div class="feature-title">Petrophysical Analysis</div>
    <div class="feature-desc">
        Comprehensive petrophysical analysis workflow with advanced processing capabilities.
        <br><br>
        <strong>Features:</strong>
        <ul>
            <li>Automated Outlier Detection & Despiking (Isolation Forest, ABOD)</li>
            <li>Tool Startup Noise Removal (Rolling variance + slope check)</li>
            <li>Log Splicing & Concatenation (Cross-correlation, DTW)</li>
            <li>Depth Alignment (Correlation-based, Siamese Neural Networks)</li>
            <li>Vertical Log Visualizations</li>
            <li>Multi-format support (LAS, ASCII, DLIS)</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Quick start guide
st.markdown("""
### Quick Start

1. **Navigate to Petrophysical Analysis** in the sidebar
2. **Upload LAS/ASCII/DLIS files** using the file upload widgets
3. **Select your workflow** (Outlier Detection, Noise Removal, Splicing, or Depth Alignment)
4. **Configure parameters** and visualize results
5. **Export processed data** in LAS format

### Supported File Formats

- **Input:** LAS 2.0/3.0 files (.las, .LAS), ASCII files (.txt, .dat, .csv), DLIS files (.dlis)
- **Output:** LAS, PNG (visualizations)

### Tips

- Use the automated curve identification to quickly map your log curves
- Enable vertical log visualizations to see before/after comparisons
- Combine multiple workflows for comprehensive data processing
- Export processed logs for use in other software
""")

# Sidebar info
with st.sidebar:
    st.markdown("### Petrophysical Analysis")
    st.markdown("---")
    st.markdown("""
    **Navigation**

    Access the Petrophysical Analysis tool from the pages menu above.

    ---

    **About**

    Professional petrophysical analysis workflows for well log data processing.

    Version: 2.0.0
    """)
