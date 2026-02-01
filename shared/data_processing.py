"""
Data Processing Module
Handles data transformation, null handling, smoothing, and export.
"""

import pandas as pd
import numpy as np
import lasio
import io

# Standard null values in LAS files
NULL_VALUES = [-999.25, -999, -9999, -9999.25, -999.2500, -999.00]


def handle_null_values(df, null_values=None):
    """
    Replace null values with NaN to create gaps in plots.
    Optimized using vectorized pandas operations.
    
    Args:
        df: pandas DataFrame
        null_values: List of null value markers
        
    Returns:
        DataFrame with nulls replaced by NaN
    """
    if null_values is None:
        null_values = NULL_VALUES
    
    df_clean = df.copy()
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            # Vectorized replacement - much faster than loop
            df_clean[col] = df_clean[col].replace(null_values, np.nan)
    
    return df_clean


def apply_smoothing(df, window=5, columns=None, exclude=['DEPTH']):
    """
    Apply moving average smoothing to curves.
    
    Args:
        df: pandas DataFrame
        window: Smoothing window size
        columns: List of columns to smooth (None = all numeric)
        exclude: Columns to exclude from smoothing
        
    Returns:
        DataFrame with smoothed values
    """
    if window <= 0:
        return df
    
    df_smooth = df.copy()
    
    if columns is None:
        columns = [col for col in df.columns if col not in exclude 
                   and pd.api.types.is_numeric_dtype(df[col])]
    
    for col in columns:
        if col in df_smooth.columns:
            df_smooth[col] = df_smooth[col].rolling(
                window=window, center=True, min_periods=1
            ).mean()
    
    return df_smooth


def normalize_porosity_units(df, las, mapping):
    """
    Normalize neutron porosity values from percentage or PU units to v/v (decimal).

    ISSUE 3 FIX: Unit Normalization Rules (from client feedback):
    - PU or % → divide by 100 → V/V
    - ft3/ft3 → V/V (no change, already volumetric fraction)
    - m3/m3 → V/V (no change, already volumetric fraction)
    - v/v → no change (already in correct format)

    This normalization is applied BEFORE curve selection/scoring to ensure
    consistent neutron responses across vendors and tools.

    Args:
        df: pandas DataFrame with curve data
        las: lasio.LASFile object to check units
        mapping: Curve mapping dictionary

    Returns:
        DataFrame with normalized NPHI values
    """
    neut_col = mapping.get('NEUT')
    if not neut_col or neut_col not in df.columns:
        return df

    # Get the unit of the neutron curve from the LAS file
    neut_unit = None
    try:
        for curve in las.curves:
            if curve.mnemonic == neut_col:
                neut_unit = curve.unit.upper().strip() if curve.unit else None
                break
    except:
        pass

    # Units that require division by 100 (percentage-based)
    percentage_units = [
        '%', 'PU', 'PERCENT', 'PCT', 'P.U.', 'P.U',
        'PERC', 'POROSITY UNITS', 'POR UNITS',
        'V/V%', 'VOL%', 'PORCENT'
    ]

    # Units that are already in v/v (no conversion needed)
    volumetric_units = [
        'V/V', 'FRAC', 'DEC', 'DECIMAL', 'FRACTION',
        'FT3/FT3', 'M3/M3', 'CC/CC'
    ]

    if not neut_unit:
        # Try to detect from data range if unit not specified
        data = df[neut_col].dropna()
        if len(data) > 10:
            # Use percentiles for robust range detection
            data_p05 = data.quantile(0.05)
            data_p95 = data.quantile(0.95)

            # If data range suggests percentage (values typically 0-60 for neutron)
            # v/v values are typically -0.15 to 0.60
            if data_p95 > 1.0 and data_p05 > -1.0:
                # Likely percentage units (values > 1.0 suggest not decimal)
                neut_unit = 'PU'
            elif -0.2 < data_p05 and data_p95 < 1.0:
                # Already in v/v, no conversion needed
                neut_unit = 'V/V'

    # Normalize unit string for comparison
    neut_unit_clean = neut_unit.upper().replace(' ', '').replace('.', '') if neut_unit else ''

    # Convert if unit is % or PU (percentage units)
    if neut_unit_clean in [u.upper().replace(' ', '').replace('.', '') for u in percentage_units]:
        df = df.copy()
        # Convert from percentage to decimal v/v (divide by 100)
        # Standard scaling: PU / 100 = v/v
        df[neut_col] = df[neut_col] / 100.0

    # ft3/ft3, m3/m3 are already volumetric fractions - no conversion needed
    # v/v, frac, dec are already in correct format - no conversion needed

    return df


def process_data(las, mapping, smooth_window=0):
    """
    Convert LAS data to clean DataFrame with proper null handling.

    Args:
        las: lasio.LASFile object
        mapping: Curve mapping dictionary
        smooth_window: Smoothing window (0 = no smoothing)

    Returns:
        pandas DataFrame
    """
    df = las.df()
    index_name = df.index.name or 'DEPT'

    # Identify depth-related column names that would conflict after standardization
    depth_aliases = {'DEPT', 'DEPTH', 'MD', 'DEPTM', 'DEPTHM'}
    if mapping.get('DEPTH'):
        depth_aliases.add(mapping['DEPTH'].upper())

    # Find data columns that are depth duplicates (will conflict with index after reset)
    cols_to_drop = []
    for col in df.columns:
        col_upper = col.upper()
        # Drop if it matches the index name or is a depth alias
        if col == index_name or col_upper in depth_aliases:
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = df.reset_index()

    # Handle any remaining duplicate column names by adding suffix
    if df.columns.duplicated().any():
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            dup_indices = cols[cols == dup].index.tolist()
            for i, idx in enumerate(dup_indices[1:], 1):
                cols.iloc[idx] = f"{dup}_{i}"
        df.columns = cols

    # Standardize depth column name to 'DEPTH'
    first_col = df.columns[0]
    if first_col != 'DEPTH':
        df.rename(columns={first_col: 'DEPTH'}, inplace=True)

    # Final safety check: ensure no duplicate DEPTH columns
    depth_cols = [c for c in df.columns if c == 'DEPTH']
    if len(depth_cols) > 1:
        # Keep only the first DEPTH column, rename others
        seen_depth = False
        new_cols = []
        for col in df.columns:
            if col == 'DEPTH':
                if not seen_depth:
                    new_cols.append('DEPTH')
                    seen_depth = True
                else:
                    new_cols.append('DEPTH_DUP')
            else:
                new_cols.append(col)
        df.columns = new_cols
    
    # Handle null values
    df = handle_null_values(df)
    
    # Normalize neutron porosity units (% or PU to v/v)
    df = normalize_porosity_units(df, las, mapping)
    
    # Apply smoothing if requested
    if smooth_window > 0:
        df = apply_smoothing(df, window=smooth_window)
    
    return df


def get_auto_scale(df, column, default_min, default_max, margin=0.1):
    """
    Calculate auto-scaling when data exceeds standard ranges.
    
    Args:
        df: pandas DataFrame
        column: Column name to scale
        default_min: Default minimum value
        default_max: Default maximum value
        margin: Margin as fraction of range
        
    Returns:
        Tuple (min, max)
    """
    if column not in df.columns:
        return default_min, default_max
    
    data = df[column].dropna()
    if len(data) == 0:
        return default_min, default_max
    
    data_min = data.min()
    data_max = data.max()
    
    # Check if data is within default range
    if data_min >= default_min and data_max <= default_max:
        return default_min, default_max
    
    # Expand range with margin
    range_val = data_max - data_min
    if range_val == 0:
        range_val = 0.1
    
    return data_min - range_val * margin, data_max + range_val * margin


def get_depth_range(df, depth_col='DEPTH'):
    """
    Get the depth range of the data.
    
    Args:
        df: pandas DataFrame
        depth_col: Depth column name
        
    Returns:
        Tuple (min_depth, max_depth)
    """
    if depth_col not in df.columns:
        return 0, 0
    
    return float(df[depth_col].min()), float(df[depth_col].max())


def filter_by_depth(df, start_depth, end_depth, depth_col='DEPTH'):
    """
    Filter DataFrame to a depth range.
    
    Args:
        df: pandas DataFrame
        start_depth: Start depth
        end_depth: End depth
        depth_col: Depth column name
        
    Returns:
        Filtered DataFrame
    """
    if depth_col not in df.columns:
        return df
    
    return df[(df[depth_col] >= start_depth) & (df[depth_col] <= end_depth)]


def export_to_las(df, header_info, depth_unit='m'):
    """
    Export DataFrame to LAS file format.
    
    Args:
        df: pandas DataFrame
        header_info: Dictionary with well metadata
        depth_unit: Depth unit string
        
    Returns:
        LAS file content as string
    """
    las_out = lasio.LASFile()
    
    # Set header info
    las_out.well.WELL = header_info.get('WELL', 'UNKNOWN')
    las_out.well.STRT = df['DEPTH'].min() if 'DEPTH' in df.columns else 0
    las_out.well.STOP = df['DEPTH'].max() if 'DEPTH' in df.columns else 0
    las_out.well.STEP = df['DEPTH'].diff().median() if 'DEPTH' in df.columns and len(df) > 1 else 0
    las_out.well.NULL = -999.25
    
    # Add curves
    for col in df.columns:
        if col == 'DEPTH':
            las_out.append_curve('DEPT', df[col].values, unit=depth_unit, descr='Depth')
        else:
            # Replace NaN with null value for export
            values = df[col].fillna(-999.25).values
            las_out.append_curve(col, values, unit='', descr=col)
    
    # Write to string
    output = io.StringIO()
    las_out.write(output)
    return output.getvalue()


def df_to_json(df, depth_col='DEPTH'):
    """
    Convert DataFrame to JSON format for API response.
    Optimized for performance using vectorized operations.
    
    Args:
        df: pandas DataFrame
        depth_col: Depth column name
        
    Returns:
        Dictionary with curve data
    """
    result = {
        'depth': df[depth_col].tolist() if depth_col in df.columns else [],
        'curves': {}
    }
    
    for col in df.columns:
        if col != depth_col:
            # Use numpy's where for fast NaN replacement - vectorized operation
            series = df[col]
            # Convert NaN to None in a fast vectorized way
            mask = pd.isna(series)
            if mask.any():
                # Only process if there are NaN values
                values = series.values.tolist()
                # Fast in-place replacement using numpy
                null_indices = np.where(mask)[0]
                for idx in null_indices:
                    values[idx] = None
                result['curves'][col] = values
            else:
                # No NaN, just convert directly
                result['curves'][col] = series.tolist()
    
    return result
