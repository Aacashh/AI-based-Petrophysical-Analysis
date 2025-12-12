"""
WellLog Analyzer Pro - AI Splicing Page
Automated merging of overlapping well log runs using Global Cross-Correlation
and Constrained Dynamic Time Warping (DTW).

This is an educational "glass box" interface that explains each step of the algorithm.
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
import os
import io

# Add parent directories to path for shared module access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.las_parser import load_las, extract_header_info, detect_depth_units, get_available_curves
from shared.curve_mapping import get_curve_mapping
from shared.data_processing import process_data, export_to_las
from shared.splicing import (
    splice_logs, find_common_curves, get_recommended_correlation_curve,
    DEFAULT_GRID_STEP, DEFAULT_SEARCH_WINDOW, DEFAULT_DTW_WINDOW
)
from plotting import export_plot_to_bytes

# Page configuration
st.set_page_config(
    page_title="AI Splicing | WellLog Analyzer Pro",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1d24 0%, #2d3748 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #3d4852;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #00D4AA;
        margin: 0;
    }
    
    .main-subtitle {
        font-size: 0.9rem;
        color: #8892a0;
        margin-top: 0.3rem;
    }
    
    .step-header {
        background: linear-gradient(90deg, #00D4AA 0%, #00A080 100%);
        color: #1a1d24;
        padding: 10px 15px;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    .algorithm-box {
        background: #f1f5f9;
        border-left: 4px solid #00D4AA;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
        font-family: 'Inter', sans-serif;
    }
    
    .algorithm-box code {
        font-family: 'JetBrains Mono', monospace;
        background: #e2e8f0;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.85rem;
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, #1a1d24 0%, #2d3748 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #3d4852;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00D4AA;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8892a0;
        margin-top: 5px;
    }
    
    .file-info {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    
    .file-name {
        font-weight: 600;
        color: #1a1d24;
        font-size: 1.1rem;
    }
    
    .file-meta {
        color: #6c757d;
        font-size: 0.85rem;
        margin-top: 5px;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🔗 AI Splicing</div>
    <div class="main-subtitle">Automated Log Run Merging • Global Correlation + Constrained DTW</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_qc_plot(shallow_depth, shallow_signal, deep_depth, deep_signal,
                   corrected_deep_depth, corrected_deep_signal,
                   correction_depth, correction_delta,
                   overlap_start, overlap_end, splice_point,
                   curve_name="Signal"):
    """
    Create professional 3-track QC plot for splicing results.
    
    Track 1 (Before): Raw Shallow (Green) vs Raw Deep (Red)
    Track 2 (After): Raw Shallow (Green) vs Corrected Deep (Blue)
    Track 3 (Corrections): Shift curve showing depth delta vs depth
    """
    # Calculate figure height based on depth range
    all_depths = np.concatenate([shallow_depth, deep_depth, corrected_deep_depth])
    depth_min = np.nanmin(all_depths)
    depth_max = np.nanmax(all_depths)
    depth_range = depth_max - depth_min
    
    # Scale figure height (cap at reasonable size)
    height_in = min(16, max(8, depth_range / 100))
    
    fig, axes = plt.subplots(1, 3, figsize=(14, height_in), sharey=True)
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.08, top=0.93, bottom=0.05, left=0.08, right=0.97)
    
    # Common styling
    grid_color = '#CCCCCC'
    track_bg = '#FAFAFA'
    border_color = '#333333'
    
    # Colors
    shallow_color = '#00AA00'  # Green
    deep_raw_color = '#CC0000'  # Red
    deep_corr_color = '#0066CC'  # Blue
    overlap_color = '#FFE4B5'  # Light orange for overlap region
    splice_color = '#FF6600'  # Orange for splice line
    
    for ax in axes:
        ax.set_facecolor(track_bg)
        ax.tick_params(axis='both', which='major', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(1)
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=grid_color, alpha=0.7)
    
    # =========================================================================
    # TRACK 1: BEFORE (Raw Shallow vs Raw Deep)
    # =========================================================================
    ax1 = axes[0]
    ax1.set_title("BEFORE\n(Raw Logs)", fontsize=10, fontweight='bold', pad=10)
    
    # Shade overlap region
    ax1.axhspan(overlap_start, overlap_end, color=overlap_color, alpha=0.3, zorder=0)
    
    # Plot curves
    ax1.plot(shallow_signal, shallow_depth, color=shallow_color, linewidth=1.2, 
             label='Shallow (Reference)')
    ax1.plot(deep_signal, deep_depth, color=deep_raw_color, linewidth=1.2, 
             label='Deep (Raw)')
    
    # Auto-scale x-axis
    all_signals = np.concatenate([shallow_signal, deep_signal])
    valid_signals = all_signals[~np.isnan(all_signals)]
    if len(valid_signals) > 0:
        sig_min, sig_max = np.percentile(valid_signals, [2, 98])
        sig_range = sig_max - sig_min
        ax1.set_xlim(sig_min - 0.1*sig_range, sig_max + 0.1*sig_range)
    
    ax1.set_xlabel(curve_name, fontsize=9, fontweight='bold')
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()
    
    ax1.set_ylim(depth_max, depth_min)  # Inverted
    ax1.set_ylabel("Depth (m)", fontsize=10, fontweight='bold')
    
    # Legend
    ax1.legend(loc='lower right', fontsize=7, framealpha=0.9)
    
    # =========================================================================
    # TRACK 2: AFTER (Raw Shallow vs Corrected Deep)
    # =========================================================================
    ax2 = axes[1]
    ax2.set_title("AFTER\n(Spliced Result)", fontsize=10, fontweight='bold', pad=10)
    
    # Shade overlap region
    ax2.axhspan(overlap_start, overlap_end, color=overlap_color, alpha=0.3, zorder=0)
    
    # Splice point line
    ax2.axhline(splice_point, color=splice_color, linewidth=2, linestyle='--', 
                label=f'Splice Point ({splice_point:.1f}m)')
    
    # Plot curves
    ax2.plot(shallow_signal, shallow_depth, color=shallow_color, linewidth=1.2,
             label='Shallow (Reference)')
    ax2.plot(corrected_deep_signal, corrected_deep_depth, color=deep_corr_color, 
             linewidth=1.2, label='Deep (Corrected)')
    
    # Same x-axis limits as Track 1
    if len(valid_signals) > 0:
        ax2.set_xlim(sig_min - 0.1*sig_range, sig_max + 0.1*sig_range)
    
    ax2.set_xlabel(curve_name, fontsize=9, fontweight='bold')
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    
    ax2.legend(loc='lower right', fontsize=7, framealpha=0.9)
    
    # =========================================================================
    # TRACK 3: CORRECTIONS (Shift Curve)
    # =========================================================================
    ax3 = axes[2]
    ax3.set_title("DEPTH CORRECTIONS\n(Original - Corrected)", fontsize=10, fontweight='bold', pad=10)
    
    # Shade overlap region
    ax3.axhspan(overlap_start, overlap_end, color=overlap_color, alpha=0.3, zorder=0)
    
    # Zero line
    ax3.axvline(0, color='#888888', linewidth=1, linestyle='-', alpha=0.5)
    
    # Plot correction curve
    ax3.plot(correction_delta, correction_depth, color='#9933FF', linewidth=1.5,
             label='Depth Correction')
    ax3.fill_betweenx(correction_depth, 0, correction_delta, 
                      where=~np.isnan(correction_delta),
                      color='#9933FF', alpha=0.2)
    
    # X-axis limits based on correction range
    valid_delta = correction_delta[~np.isnan(correction_delta)]
    if len(valid_delta) > 0 and np.any(valid_delta != 0):
        delta_abs_max = max(abs(np.min(valid_delta)), abs(np.max(valid_delta)), 0.5)
        ax3.set_xlim(-delta_abs_max * 1.2, delta_abs_max * 1.2)
    else:
        ax3.set_xlim(-1, 1)
    
    ax3.set_xlabel("Δ Depth (m)", fontsize=9, fontweight='bold')
    ax3.xaxis.set_label_position('top')
    ax3.xaxis.tick_top()
    
    ax3.legend(loc='lower right', fontsize=7, framealpha=0.9)
    
    # Add text annotation
    ax3.text(0.95, 0.02, 
             "Positive = stretched down\nNegative = compressed up",
             transform=ax3.transAxes, fontsize=7, color='#666666',
             ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    
    return fig


def display_file_info(las, label, depth_unit):
    """Display formatted file information."""
    header = extract_header_info(las)
    curves = get_available_curves(las)
    
    st.markdown(f"""
    <div class="file-info">
        <div class="file-name">📄 {label}: {header['WELL']}</div>
        <div class="file-meta">
            Depth: {header['STRT']} - {header['STOP']} {depth_unit} &nbsp;|&nbsp; 
            Step: {header['STEP']} {depth_unit} &nbsp;|&nbsp;
            Curves: {len(curves)}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return header, curves


# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Splicing Parameters")
    
    st.markdown("#### 🔍 Global Alignment")
    max_search = st.slider(
        "Max Search Window (m)",
        min_value=5.0,
        max_value=50.0,
        value=DEFAULT_SEARCH_WINDOW,
        step=1.0,
        help="Maximum depth shift to search for correlation. "
             "Increase if runs have large depth discrepancy."
    )
    
    st.markdown("#### 🔧 Elastic Correction")
    max_elastic = st.slider(
        "Max Elastic Stretch (m)",
        min_value=1.0,
        max_value=20.0,
        value=DEFAULT_DTW_WINDOW,
        step=0.5,
        help="Maximum local stretch/squeeze allowed by DTW. "
             "Keep small (< 5m) for geological realism."
    )
    
    st.markdown("#### 📏 Resampling")
    grid_step = st.selectbox(
        "Grid Resolution",
        options=[0.0762, 0.1524, 0.3048],
        index=1,
        format_func=lambda x: f"{x:.4f}m ({x/0.3048:.2f}ft)",
        help="Common depth grid step for signal alignment."
    )
    
    st.markdown("---")
    
    st.markdown("""
    ### 📖 Algorithm Overview
    
    **Stage 1: Global Shift**
    - Cross-correlation finds bulk depth offset
    - Corrects for depth encoder errors
    
    **Stage 2: Elastic Warp**
    - Constrained DTW handles cable stretch
    - Sakoe-Chiba band prevents over-warping
    
    **Stage 3: Splice**
    - Merge at overlap midpoint
    - Preserve data integrity
    """)


# =============================================================================
# SECTION 1: DATA INTAKE
# =============================================================================

st.markdown("## 📥 Section 1: Data Intake")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Reference Log (Shallow Run)")
    shallow_file = st.file_uploader(
        "Upload shallow run LAS file",
        type=['las', 'LAS'],
        key='shallow_file',
        help="The shallow log run will be used as the reference (fixed)."
    )

with col2:
    st.markdown("### Target Log (Deep Run)")
    deep_file = st.file_uploader(
        "Upload deep run LAS file", 
        type=['las', 'LAS'],
        key='deep_file',
        help="The deep log run will be shifted and warped to match the reference."
    )

# Process uploaded files
if shallow_file and deep_file:
    try:
        # Load LAS files
        shallow_las = load_las(shallow_file)
        deep_las = load_las(deep_file)
        
        shallow_unit = detect_depth_units(shallow_las)
        deep_unit = detect_depth_units(deep_las)
        
        # Display file info
        col1, col2 = st.columns(2)
        with col1:
            shallow_header, shallow_curves = display_file_info(
                shallow_las, "Reference (Shallow)", shallow_unit
            )
        with col2:
            deep_header, deep_curves = display_file_info(
                deep_las, "Target (Deep)", deep_unit
            )
        
        # Find common curves
        shallow_curve_names = [c['mnemonic'] for c in shallow_curves]
        deep_curve_names = [c['mnemonic'] for c in deep_curves]
        common_curves = find_common_curves(shallow_curve_names, deep_curve_names)
        
        # Remove depth curves from selection
        depth_names = {'DEPT', 'DEPTH', 'MD', 'TVD'}
        selectable_curves = [c for c in common_curves if c.upper() not in depth_names]
        
        if not selectable_curves:
            st.error("❌ No common curves found between the two files. "
                    "Please ensure both files have at least one matching curve.")
        else:
            # Curve selection
            recommended = get_recommended_correlation_curve(selectable_curves)
            default_idx = selectable_curves.index(recommended) if recommended in selectable_curves else 0
            
            st.markdown("### 📊 Correlation Curve Selection")
            
            correlation_curve = st.selectbox(
                "Select curve for correlation",
                options=selectable_curves,
                index=default_idx,
                help="Choose a curve present in both files. GR (Gamma Ray) is typically best."
            )
            
            st.info(f"ℹ️ **{len(selectable_curves)}** common curves found. "
                   f"Recommended: **{recommended or selectable_curves[0]}**")
            
            # =================================================================
            # SECTION 2: GLASS BOX EXECUTION
            # =================================================================
            
            st.markdown("---")
            st.markdown("## 🔬 Section 2: Algorithm Execution")
            st.markdown("*The 'Glass Box' - Watch the algorithm work step by step*")
            
            if st.button("🚀 Run AI Splicing", type="primary", use_container_width=True):
                
                # Get data from LAS files
                shallow_mapping = get_curve_mapping(shallow_las)
                deep_mapping = get_curve_mapping(deep_las)
                
                shallow_df = process_data(shallow_las, shallow_mapping)
                deep_df = process_data(deep_las, deep_mapping)
                
                shallow_depth = shallow_df['DEPTH'].values
                deep_depth = deep_df['DEPTH'].values
                
                # Get correlation curves
                if correlation_curve in shallow_df.columns:
                    shallow_signal = shallow_df[correlation_curve].values
                else:
                    st.error(f"Curve {correlation_curve} not found in shallow file")
                    st.stop()
                    
                if correlation_curve in deep_df.columns:
                    deep_signal = deep_df[correlation_curve].values
                else:
                    st.error(f"Curve {correlation_curve} not found in deep file")
                    st.stop()
                
                # Progress tracking
                progress_messages = []
                
                def progress_callback(step, message):
                    progress_messages.append((step, message))
                
                # Execute splicing with progress display
                with st.status("🔄 Running AI Splicing Algorithm...", expanded=True) as status:
                    
                    # ---------------------------------------------------------
                    # STEP 1: Preprocessing
                    # ---------------------------------------------------------
                    st.markdown('<div class="step-header">Step 1: Preprocessing & Grid Alignment</div>', 
                               unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="algorithm-box">
                    <strong>What's happening:</strong><br>
                    1. Creating a common depth grid with <code>{:.4f}m</code> step covering both files<br>
                    2. Resampling both signals via linear interpolation<br>
                    3. Applying <strong>Z-Score Normalization</strong>: <code>(x - μ) / σ</code><br>
                    &nbsp;&nbsp;&nbsp;→ This removes tool calibration bias between runs<br>
                    4. Filling NaN values with 0 (normalized mean) for correlation math only
                    </div>
                    """.format(grid_step), unsafe_allow_html=True)
                    
                    st.write("⏳ Resampling logs to common grid...")
                    
                    # Run the splicing algorithm
                    try:
                        result = splice_logs(
                            shallow_depth=shallow_depth,
                            shallow_signal=shallow_signal,
                            deep_depth=deep_depth,
                            deep_signal=deep_signal,
                            grid_step=grid_step,
                            max_search_meters=max_search,
                            max_elastic_meters=max_elastic,
                            progress_callback=progress_callback
                        )
                        
                        st.write("✅ Preprocessing complete!")
                        
                        # -------------------------------------------------
                        # STEP 2: Global Shift Detection
                        # -------------------------------------------------
                        st.markdown('<div class="step-header">Step 2: Global Shift Detection</div>', 
                                   unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="algorithm-box">
                        <strong>What's happening:</strong><br>
                        1. Computing cross-correlation between normalized signals<br>
                        2. Searching for correlation peak within ±<code>{max_search}m</code> window<br>
                        3. Peak location indicates optimal global depth shift<br><br>
                        <strong>Why this works:</strong> Cross-correlation measures similarity at different 
                        alignments. The peak corresponds to where the two signals best match.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display bulk shift metric
                        shift_dir = "shallower" if result.bulk_shift_meters > 0 else "deeper"
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown(f"""
                            <div class="metric-highlight">
                                <div class="metric-value">{abs(result.bulk_shift_meters):.3f} m</div>
                                <div class="metric-label">Bulk Shift Detected ({shift_dir})</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        📍 **Interpretation:** The Deep Run was recorded **{abs(result.bulk_shift_meters):.3f} meters 
                        {shift_dir}** than actual. This is corrected by shifting the entire Deep log.
                        """)
                        
                        # -------------------------------------------------
                        # STEP 3: Elastic Warping
                        # -------------------------------------------------
                        st.markdown('<div class="step-header">Step 3: Elastic Correction (Constrained DTW)</div>', 
                                   unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="algorithm-box">
                        <strong>What's happening:</strong><br>
                        1. Extracting overlap region: <code>{result.overlap_start:.1f}m</code> to 
                           <code>{result.overlap_end:.1f}m</code><br>
                        2. Running <strong>Dynamic Time Warping</strong> with Sakoe-Chiba constraint<br>
                        &nbsp;&nbsp;&nbsp;→ Constraint band: ±<code>{max_elastic}m</code> (prevents over-warping)<br>
                        3. Finding optimal elastic path that minimizes squared distance<br>
                        4. Applying sub-meter depth corrections for cable stretch effects<br><br>
                        <strong>Why constrained DTW:</strong> Standard DTW can warp signals unrealistically. 
                        The Sakoe-Chiba band forces the warp to stay near the diagonal, 
                        preserving geological features.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # DTW metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                label="DTW Cost",
                                value=f"{result.dtw_cost:.2f}",
                                help="Total squared distance along optimal path"
                            )
                        
                        with col2:
                            overlap_length = result.overlap_end - result.overlap_start
                            st.metric(
                                label="Overlap Length",
                                value=f"{overlap_length:.1f} m",
                                help="Length of overlapping section"
                            )
                        
                        with col3:
                            # Calculate max correction
                            valid_delta = result.correction_delta[~np.isnan(result.correction_delta)]
                            max_correction = np.max(np.abs(valid_delta)) if len(valid_delta) > 0 else 0
                            st.metric(
                                label="Max Elastic Correction",
                                value=f"{max_correction:.3f} m",
                                help="Maximum local depth adjustment"
                            )
                        
                        # -------------------------------------------------
                        # STEP 4: Final Merge
                        # -------------------------------------------------
                        st.markdown('<div class="step-header">Step 4: Final Splice & Merge</div>', 
                                   unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="algorithm-box">
                        <strong>What's happening:</strong><br>
                        1. Splice point determined at overlap midpoint: <code>{result.splice_point:.1f}m</code><br>
                        2. Taking Shallow log data above splice point<br>
                        3. Taking corrected Deep log data at and below splice point<br>
                        4. Concatenating into seamless merged log<br><br>
                        <strong>Result:</strong> A single continuous log with depth-corrected data from both runs.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success(f"✅ **Splicing Complete!** Merged log has "
                                  f"{len(result.merged_depth)} samples.")
                        
                        status.update(label="✅ AI Splicing Complete!", state="complete")
                        
                        # Store results in session state
                        st.session_state['splicing_result'] = result
                        st.session_state['shallow_depth'] = shallow_depth
                        st.session_state['shallow_signal'] = shallow_signal
                        st.session_state['deep_depth'] = deep_depth
                        st.session_state['deep_signal'] = deep_signal
                        st.session_state['correlation_curve'] = correlation_curve
                        st.session_state['shallow_header'] = shallow_header
                        st.session_state['deep_header'] = deep_header
                        st.session_state['depth_unit'] = shallow_unit
                        
                    except Exception as e:
                        status.update(label="❌ Error during splicing", state="error")
                        st.error(f"Error during splicing: {str(e)}")
                        st.exception(e)
            
            # =================================================================
            # SECTION 3: QC PLOTTING (if results exist)
            # =================================================================
            
            if 'splicing_result' in st.session_state:
                st.markdown("---")
                st.markdown("## 📊 Section 3: Quality Control Visualization")
                
                result = st.session_state['splicing_result']
                shallow_depth = st.session_state['shallow_depth']
                shallow_signal = st.session_state['shallow_signal']
                deep_depth = st.session_state['deep_depth']
                deep_signal = st.session_state['deep_signal']
                correlation_curve = st.session_state['correlation_curve']
                
                st.markdown("""
                This 3-track QC plot allows you to verify the splicing quality:
                - **Track 1 (Before):** Shows original mismatch between runs
                - **Track 2 (After):** Shows alignment after correction
                - **Track 3 (Corrections):** Shows the applied depth adjustments
                """)
                
                # Create QC plot
                fig = create_qc_plot(
                    shallow_depth=shallow_depth,
                    shallow_signal=shallow_signal,
                    deep_depth=deep_depth,
                    deep_signal=deep_signal,
                    corrected_deep_depth=result.corrected_deep_depth,
                    corrected_deep_signal=result.corrected_deep_signal,
                    correction_depth=result.correction_depth,
                    correction_delta=result.correction_delta,
                    overlap_start=result.overlap_start,
                    overlap_end=result.overlap_end,
                    splice_point=result.splice_point,
                    curve_name=correlation_curve
                )
                
                st.pyplot(fig)
                
                # Summary metrics
                st.markdown("### 📈 Splicing Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Bulk Shift", f"{result.bulk_shift_meters:.3f} m")
                with col2:
                    st.metric("Overlap Region", 
                             f"{result.overlap_start:.1f} - {result.overlap_end:.1f} m")
                with col3:
                    st.metric("Splice Point", f"{result.splice_point:.1f} m")
                with col4:
                    st.metric("Merged Samples", f"{len(result.merged_depth)}")
                
                # Export options
                st.markdown("### 💾 Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    png_data = export_plot_to_bytes(fig, format='png', dpi=150)
                    st.download_button(
                        label="📷 Download QC Plot (PNG)",
                        data=png_data,
                        file_name="splicing_qc_plot.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col2:
                    pdf_data = export_plot_to_bytes(fig, format='pdf', dpi=150)
                    st.download_button(
                        label="📄 Download QC Plot (PDF)",
                        data=pdf_data,
                        file_name="splicing_qc_plot.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                with col3:
                    # Create merged LAS export
                    import pandas as pd
                    merged_df = pd.DataFrame({
                        'DEPTH': result.merged_depth,
                        correlation_curve: result.merged_signal
                    })
                    
                    merged_header = {
                        'WELL': f"{st.session_state['shallow_header']['WELL']}_SPLICED",
                        'STRT': result.merged_depth[0],
                        'STOP': result.merged_depth[-1],
                        'STEP': np.median(np.diff(result.merged_depth))
                    }
                    
                    las_data = export_to_las(merged_df, merged_header, 
                                            st.session_state['depth_unit'])
                    st.download_button(
                        label="📋 Export Merged LAS",
                        data=las_data,
                        file_name=f"{merged_header['WELL']}.las",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                plt.close(fig)
                
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        st.exception(e)

else:
    # Welcome/instructions when no files uploaded
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; background: #f8f9fa; border-radius: 10px; margin-top: 20px;">
        <div style="font-size: 4rem; margin-bottom: 20px;">🔗</div>
        <h2 style="color: #00D4AA; margin-bottom: 15px;">AI-Powered Log Splicing</h2>
        <p style="color: #6c757d; font-size: 1.1rem; max-width: 700px; margin: 0 auto; line-height: 1.6;">
            Upload two LAS files (Shallow Run and Deep Run) to automatically merge them 
            using industry-standard depth correction algorithms.
            <br><br>
            <strong>The algorithm will:</strong>
        </p>
        <div style="text-align: left; max-width: 500px; margin: 20px auto; color: #495057;">
            <p>✓ Find global depth shift using cross-correlation</p>
            <p>✓ Apply elastic correction for cable stretch using DTW</p>
            <p>✓ Splice logs at the overlap midpoint</p>
            <p>✓ Generate professional QC visualization</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Algorithm explanation
    st.markdown("---")
    st.markdown("### 📚 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Stage 1: Global Cross-Correlation
        
        Cross-correlation is used to find the optimal **bulk shift** between two signals:
        
        ```
        correlation(τ) = Σ reference(t) × target(t + τ)
        ```
        
        The lag τ with maximum correlation indicates how much the deep log 
        needs to be shifted to align with the shallow log.
        
        **Why it's needed:** Depth encoders can drift, or wireline stretch 
        can cause systematic depth errors between runs.
        """)
    
    with col2:
        st.markdown("""
        #### Stage 2: Constrained DTW
        
        Dynamic Time Warping finds an **elastic alignment** between signals:
        
        ```
        DTW(i,j) = dist(i,j) + min(DTW(i-1,j), 
                                   DTW(i,j-1), 
                                   DTW(i-1,j-1))
        ```
        
        The **Sakoe-Chiba band** constraint prevents unrealistic warping by 
        limiting how far the path can deviate from the diagonal.
        
        **Why it's needed:** Cable stretch isn't constant—it varies with 
        depth, tool weight, and winch speed.
        """)

