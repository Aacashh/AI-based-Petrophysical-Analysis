import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatter

# Industry-standard colors
COLORS = {
    'GR': '#00AA00',           # Green for Gamma Ray
    'GR_FILL': '#90EE90',      # Light green for sand shading
    'RES_DEEP': '#0066CC',     # Blue for Deep Resistivity
    'RES_MED': '#CC0000',      # Red for Medium Resistivity  
    'RES_SHAL': '#FF8C00',     # Orange for Shallow Resistivity
    'DENS': '#CC0000',         # Red for Density
    'NEUT': '#0066CC',         # Blue for Neutron
    'CROSS_GAS': '#FFFF00',    # Yellow for gas crossover
    'CROSS_SHALE': '#808080',  # Grey for shale crossover
    'GRID': '#CCCCCC',         # Light grey grid
    'TRACK_BG': '#FAFAFA',     # Near-white track background
    'BORDER': '#333333',       # Dark border
}

def create_log_plot(df, mapping, settings, show_gr_fill=False, show_dn_fill=False):
    """
    Creates a professional Matplotlib figure matching Schlumberger Techlog style.
    
    Args:
        df: DataFrame with log data
        mapping: Curve mapping dictionary
        settings: Plot settings dictionary
        show_gr_fill: Enable GR sand indication shading
        show_dn_fill: Enable Density-Neutron crossover fill
    """
    depth_min = df['DEPTH'].min()
    depth_max = df['DEPTH'].max()
    depth_range = depth_max - depth_min
    
    # Calculate figure height based on scale
    scale_ratio = settings.get('scale_ratio', 500)
    height_cm = (depth_range * 100) / scale_ratio
    height_in = max(5, height_cm / 2.54)
    
    # Fixed track width: ~3.5 inches each (≈350px at 100dpi)
    # 3 tracks + spacing = ~12 inches total width
    figsize = (12, height_in)
    
    # Create figure with professional styling
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)
    fig.patch.set_facecolor('white')
    
    # Adjust spacing between tracks
    plt.subplots_adjust(wspace=0.05, top=0.97, bottom=0.02, left=0.08, right=0.98)
    
    # Track references
    ax_gr = axes[0]
    ax_res = axes[1]
    ax_dn = axes[2]
    
    # Common styling for all tracks
    for ax in axes:
        ax.set_facecolor(COLORS['TRACK_BG'])
        ax.tick_params(axis='both', which='major', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(COLORS['BORDER'])
            spine.set_linewidth(1)
    
    depth = df['DEPTH'].values
    
    # =============================================
    # TRACK 1: GAMMA RAY
    # =============================================
    _plot_gamma_ray(ax_gr, df, mapping, settings, depth, show_gr_fill)
    
    # Set depth axis on first track
    ax_gr.set_ylim(depth_max, depth_min)  # Inverted
    ax_gr.set_ylabel("Depth (m)", fontsize=10, fontweight='bold')
    ax_gr.yaxis.set_tick_params(labelsize=8)
    
    # =============================================
    # TRACK 2: RESISTIVITY
    # =============================================
    _plot_resistivity(ax_res, df, mapping, settings, depth)
    
    # =============================================
    # TRACK 3: DENSITY-NEUTRON
    # =============================================
    _plot_density_neutron(ax_dn, df, mapping, settings, depth, show_dn_fill)
    
    # Apply tight layout for professional appearance
    fig.tight_layout()
    
    return fig


def _plot_gamma_ray(ax, df, mapping, settings, depth, show_fill=False):
    """Plot Gamma Ray track with optional sand shading."""
    
    # Track header
    ax.set_title("GAMMA RAY", fontsize=10, fontweight='bold', pad=10)
    
    # Grid styling
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color=COLORS['GRID'], alpha=0.4)
    
    gr_min = settings.get('gr_min', 0)
    gr_max = settings.get('gr_max', 150)
    ax.set_xlim(gr_min, gr_max)
    
    # X-axis styling
    ax.set_xlabel("API", fontsize=9, color=COLORS['GR'], fontweight='bold')
    ax.tick_params(axis='x', colors=COLORS['GR'], labelsize=8)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.spines['top'].set_color(COLORS['GR'])
    ax.spines['top'].set_linewidth(2)
    
    if mapping['GR'] and mapping['GR'] in df.columns:
        gr_data = df[mapping['GR']].values
        
        # Plot GR curve
        ax.plot(gr_data, depth, color=COLORS['GR'], linewidth=1.5, label='GR')
        
        # Optional: Sand shading (fill from curve to right edge)
        if show_fill:
            # Create sand indication: shade right of GR curve
            # Low GR (left) = sand, High GR (right) = shale
            sand_threshold = (gr_min + gr_max) / 2  # Middle point
            ax.fill_betweenx(depth, gr_data, gr_min, 
                            where=~np.isnan(gr_data),
                            color=COLORS['GR_FILL'], alpha=0.3,
                            label='Sand indication')
    else:
        ax.text(0.5, 0.5, "No GR Curve", transform=ax.transAxes, 
                ha='center', va='center', fontsize=12, color='gray')


def _plot_resistivity(ax, df, mapping, settings, depth):
    """Plot Resistivity track with logarithmic scale."""
    
    # Track header
    ax.set_title("RESISTIVITY", fontsize=10, fontweight='bold', pad=10)
    
    # Logarithmic scale
    ax.set_xscale('log')
    
    res_min = settings.get('res_min', 0.2)
    res_max = settings.get('res_max', 2000)
    ax.set_xlim(res_min, res_max)
    
    # Professional log grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color=COLORS['GRID'], alpha=0.4)
    
    # X-axis styling  
    ax.set_xlabel("ohm.m", fontsize=9, fontweight='bold')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=7)
    
    # Set log locator for better tick marks
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    
    plotted = []
    
    # Deep Resistivity
    if mapping['RES_DEEP'] and mapping['RES_DEEP'] in df.columns:
        ax.plot(df[mapping['RES_DEEP']], depth, 
                color=COLORS['RES_DEEP'], linewidth=1.5, label='Deep (RT)')
        plotted.append(mpatches.Patch(color=COLORS['RES_DEEP'], label='Deep'))
    
    # Medium Resistivity
    if mapping['RES_MED'] and mapping['RES_MED'] in df.columns:
        ax.plot(df[mapping['RES_MED']], depth,
                color=COLORS['RES_MED'], linewidth=1.5, label='Med (RM)')
        plotted.append(mpatches.Patch(color=COLORS['RES_MED'], label='Med'))
    
    # Shallow Resistivity
    if mapping['RES_SHAL'] and mapping['RES_SHAL'] in df.columns:
        ax.plot(df[mapping['RES_SHAL']], depth,
                color=COLORS['RES_SHAL'], linewidth=1.5, label='Shallow (RS)')
        plotted.append(mpatches.Patch(color=COLORS['RES_SHAL'], label='Shallow'))
    
    if plotted:
        ax.legend(handles=plotted, loc='upper right', fontsize=7, 
                 framealpha=0.9, edgecolor=COLORS['BORDER'])
    else:
        ax.text(0.5, 0.5, "No Resistivity Curves", transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')


def _plot_density_neutron(ax, df, mapping, settings, depth, show_fill=False):
    """Plot Density-Neutron overlay with dual axes."""
    
    # Track header
    ax.set_title("DENSITY - NEUTRON", fontsize=10, fontweight='bold', pad=10)
    
    # Grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    
    # Primary axis: Density (Red)
    ax_dens = ax
    dens_min = settings.get('dens_min', 1.95)
    dens_max = settings.get('dens_max', 2.95)
    ax_dens.set_xlim(dens_min, dens_max)
    ax_dens.set_xlabel("RHOB (g/cc)", fontsize=9, color=COLORS['DENS'], fontweight='bold')
    ax_dens.tick_params(axis='x', colors=COLORS['DENS'], labelsize=8)
    ax_dens.xaxis.set_label_position('top')
    ax_dens.xaxis.tick_top()
    ax_dens.spines['top'].set_color(COLORS['DENS'])
    ax_dens.spines['top'].set_linewidth(2)
    
    # Secondary axis: Neutron (Blue) - reversed scale
    ax_neut = ax_dens.twiny()
    neut_min = settings.get('neut_min', -0.15)
    neut_max = settings.get('neut_max', 0.45)
    ax_neut.set_xlim(neut_max, neut_min)  # Reversed!
    ax_neut.set_xlabel("NPHI (v/v)", fontsize=9, color=COLORS['NEUT'], fontweight='bold')
    ax_neut.tick_params(axis='x', colors=COLORS['NEUT'], labelsize=8)
    ax_neut.spines['top'].set_position(('outward', 35))
    ax_neut.spines['top'].set_color(COLORS['NEUT'])
    ax_neut.spines['top'].set_linewidth(2)
    
    # Get data
    dens_data = None
    neut_data = None
    
    # Plot Density curve
    if mapping['DENS'] and mapping['DENS'] in df.columns:
        dens_data = df[mapping['DENS']].values
        ax_dens.plot(dens_data, depth, color=COLORS['DENS'], linewidth=1.5, label='RHOB')
    
    # Plot Neutron curve (dashed for distinction)
    if mapping['NEUT'] and mapping['NEUT'] in df.columns:
        neut_data = df[mapping['NEUT']].values
        # Transform neutron to density axis for overlay
        neut_transformed = _transform_neutron_to_density(neut_data, neut_min, neut_max, dens_min, dens_max)
        ax_dens.plot(neut_transformed, depth, color=COLORS['NEUT'], 
                    linewidth=1.5, linestyle='--', label='NPHI')
        
        # Crossover fill if enabled
        if show_fill and dens_data is not None:
            _add_crossover_fill(ax_dens, dens_data, neut_transformed, depth)
    
    # Legend
    if dens_data is not None or neut_data is not None:
        handles = []
        if mapping['DENS'] and mapping['DENS'] in df.columns:
            handles.append(mpatches.Patch(color=COLORS['DENS'], label='RHOB'))
        if mapping['NEUT'] and mapping['NEUT'] in df.columns:
            handles.append(mpatches.Patch(color=COLORS['NEUT'], label='NPHI'))
        ax_dens.legend(handles=handles, loc='upper right', fontsize=7,
                      framealpha=0.9, edgecolor=COLORS['BORDER'])
    else:
        ax.text(0.5, 0.5, "No Density/Neutron Curves", transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')


def _transform_neutron_to_density(neut_data, neut_min, neut_max, dens_min, dens_max):
    """
    Transform neutron values to density axis for overlay.
    Neutron scale is reversed relative to density.
    """
    # Normalize to 0-1, then reverse
    normalized = (neut_data - neut_min) / (neut_max - neut_min)
    reversed_norm = 1 - normalized  # Reverse for overlay
    
    # Scale to density axis
    return dens_min + reversed_norm * (dens_max - dens_min)


def _add_crossover_fill(ax, dens_data, neut_transformed, depth):
    """
    Add crossover fill between density and neutron.
    - Yellow/light when neutron > density (gas effect)
    - Grey when density > neutron (shale)
    """
    # Create masks for valid data points
    valid = ~np.isnan(dens_data) & ~np.isnan(neut_transformed)
    
    # Gas effect: Neutron crosses to the right of density (gas indication)
    ax.fill_betweenx(depth, dens_data, neut_transformed,
                     where=valid & (neut_transformed > dens_data),
                     color=COLORS['CROSS_GAS'], alpha=0.4,
                     label='Gas effect')
    
    # Shale: Density crosses to the right of neutron
    ax.fill_betweenx(depth, dens_data, neut_transformed,
                     where=valid & (dens_data > neut_transformed),
                     color=COLORS['CROSS_SHALE'], alpha=0.3,
                     label='Shale')


def export_plot_to_bytes(fig, format='png', dpi=150):
    """
    Export matplotlib figure to bytes for download.
    
    Args:
        fig: Matplotlib figure
        format: 'png', 'jpg', or 'pdf'
        dpi: Resolution for output
    
    Returns:
        Bytes of the exported image/document
    """
    import io
    buf = io.BytesIO()
    
    if format.lower() == 'pdf':
        fig.savefig(buf, format='pdf', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    elif format.lower() in ['jpg', 'jpeg']:
        fig.savefig(buf, format='jpeg', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none', quality=95)
    else:  # PNG default
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    
    buf.seek(0)
    return buf.getvalue()
