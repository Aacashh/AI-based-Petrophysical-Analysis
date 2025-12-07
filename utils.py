import lasio
import pandas as pd
import numpy as np
import io

# Standard null values in LAS files
NULL_VALUES = [-999.25, -999, -9999, -9999.25, -999.2500]

def load_las(file_obj):
    """
    Loads a LAS file from a file-like object (uploaded file).
    Returns a lasio.LASFile object.
    """
    try:
        bytes_data = file_obj.read()
        str_data = bytes_data.decode("utf-8", errors="ignore")
        return lasio.read(io.StringIO(str_data))
    except Exception as e:
        raise ValueError(f"Error loading LAS file: {e}")

def get_curve_mapping(las):
    """
    Maps LAS curves to standard tracks based on mnemonics.
    Returns a dictionary: { 'GR': 'CURVE_NAME', 'RES_DEEP': ..., ... }
    """
    keys = las.keys()
    mapping = {
        'GR': None,
        'RES_DEEP': None,
        'RES_MED': None,
        'RES_SHAL': None,
        'DENS': None,
        'NEUT': None,
        'DEPTH': None
    }

    # Helper to find first match
    def find_match(mnemonics):
        for m in mnemonics:
            for k in keys:
                if k.upper() == m.upper():
                    return k
        return None

    # 1. Depth
    mapping['DEPTH'] = find_match(['DEPT', 'DEPTH', 'DPTH'])

    # 2. Gamma Ray
    mapping['GR'] = find_match(['GR', 'GRC', 'GR_EDTC', 'SGR'])

    # 3. Resistivity
    # Deep
    mapping['RES_DEEP'] = find_match(['RT', 'RDEP', 'RLLD', 'RLL3', 'ILD'])
    # Medium
    mapping['RES_MED'] = find_match(['RM', 'ILM', 'RILS']) 
    # Shallow
    mapping['RES_SHAL'] = find_match(['RS', 'MSFL', 'RXOZ', 'RLL1', 'SFL'])

    # 4. Density
    mapping['DENS'] = find_match(['RHOB', 'RHOZ', 'DEN'])

    # 5. Neutron
    mapping['NEUT'] = find_match(['NPHI', 'TNPH', 'NPHZ', 'CNPOR'])

    return mapping

def handle_null_values(df, null_values=None):
    """
    Replace null values with NaN to create gaps in plots.
    This ensures lines don't connect across missing data.
    """
    if null_values is None:
        null_values = NULL_VALUES
    
    df_clean = df.copy()
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            for null_val in null_values:
                df_clean[col] = df_clean[col].replace(null_val, np.nan)
    
    return df_clean

def process_data(las, mapping, smooth_window=0):
    """
    Converts LAS data to a clean DataFrame with mapped columns.
    Handles null values properly to create gaps in plots.
    """
    df = las.df()
    df = df.reset_index()  # Make Depth a column if it's the index
    
    # Ensure Depth is available
    if mapping['DEPTH'] and mapping['DEPTH'] in df.columns:
        df.rename(columns={mapping['DEPTH']: 'DEPTH'}, inplace=True)
    elif df.index.name and df.index.name.upper() in ['DEPT', 'DEPTH']:
        df['DEPTH'] = df.index

    # Handle null values - replace with NaN to create gaps
    df = handle_null_values(df)

    # Apply smoothing if requested
    if smooth_window > 0:
        for col in df.columns:
            if col != 'DEPTH' and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].rolling(window=smooth_window, center=True, min_periods=1).mean()

    return df

def detect_depth_units(las):
    """
    Detect depth units from LAS header.
    Returns 'm' for meters, 'ft' for feet.
    """
    # Check STRT unit first
    try:
        if 'STRT' in las.well:
            unit = las.well.STRT.unit.upper()
            if 'F' in unit:
                return 'ft'
            elif 'M' in unit:
                return 'm'
    except:
        pass
    
    # Check curve definition for DEPT
    try:
        for curve in las.curves:
            if curve.mnemonic.upper() in ['DEPT', 'DEPTH']:
                unit = curve.unit.upper()
                if 'F' in unit:
                    return 'ft'
                elif 'M' in unit:
                    return 'm'
    except:
        pass
    
    return 'm'  # Default to meters

def get_auto_scale(df, column, default_min, default_max, margin=0.1):
    """
    Calculate auto-scaling when data exceeds standard ranges.
    Returns (min, max) tuple.
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
    
    # Otherwise, expand the range with margin
    range_val = data_max - data_min
    if range_val == 0:
        range_val = 0.1
    
    return data_min - range_val * margin, data_max + range_val * margin

def extract_header_info(las):
    """
    Extracts metadata for the header panel.
    """
    def safe_get(attr, default=''):
        try:
            if attr in las.well:
                val = getattr(las.well, attr).value
                return val if val else default
        except:
            pass
        return default
    
    info = {
        'WELL': safe_get('WELL', 'UNKNOWN'),
        'FIELD': safe_get('FLD', '') or safe_get('FIELD', ''),
        'LOC': safe_get('LOC', '') or safe_get('LOCATION', ''),
        'COMP': safe_get('COMP', '') or safe_get('OPER', ''),
        'STRT': safe_get('STRT', 0),
        'STOP': safe_get('STOP', 0),
        'STEP': safe_get('STEP', 0),
        'CTRY': safe_get('CTRY', ''),
        'SRVC': safe_get('SRVC', ''),
        'DATE': safe_get('DATE', ''),
    }
    return info

def export_to_las(df, header_info, filename, depth_unit='m'):
    """
    Export processed DataFrame to a new LAS file.
    Returns the LAS file content as a string.
    """
    las_out = lasio.LASFile()
    
    # Set header info
    las_out.well.WELL = header_info.get('WELL', 'UNKNOWN')
    las_out.well.STRT = df['DEPTH'].min()
    las_out.well.STOP = df['DEPTH'].max()
    las_out.well.STEP = df['DEPTH'].diff().median() if len(df) > 1 else 0
    las_out.well.NULL = -999.25
    
    # Add curves
    for col in df.columns:
        if col == 'DEPTH':
            las_out.append_curve('DEPT', df[col].values, unit=depth_unit, descr='Depth')
        else:
            las_out.append_curve(col, df[col].values, unit='', descr=col)
    
    # Write to string
    output = io.StringIO()
    las_out.write(output)
    return output.getvalue()
